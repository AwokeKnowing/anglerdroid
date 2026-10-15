"""Ship-thread raw bag: RS z16 + IMU ring + wheel encoders + 80×60 labels.

Capture only copies into a slot (queue size 1). Disk + 80×60 scatter live
on the ship thread so the 30 Hz grab path never waits on encode or NVMe.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Optional

import numpy as np

from perception.fast_ego80 import (
    EGO80_H, EGO80_W, downsample_labels_4, label_from_z16,
)

RS_H, RS_W = 480, 848
IMU_MAX = 16  # samples drained per depth frame


class RawBag:
    def __init__(self, dest: str, seconds: float = 300.0, fps: float = 30.0):
        self.dest = os.path.abspath(dest)
        os.makedirs(self.dest, exist_ok=True)
        self.seconds = float(seconds)
        self.n_max = int(round(seconds * fps)) + 8
        self.n = 0
        self.drops = 0
        self.t0 = time.monotonic()
        self._closed = False
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._free: queue.Queue = queue.Queue(maxsize=2)
        for _ in range(2):
            self._free.put(self._new_slot())

        # Atlas / GPU viz never touch this bag. Labels are rebuildable from z16
        # (from_bag.py --rebuild) — skip the two n_max×80×60 memmaps on collect.
        self.slim = os.environ.get("KEVIN_RAW_SLIM", "1").strip() != "0"
        # Do NOT punch 14 GB holes at t=0 — that stalls capture on Orin NVMe.
        self._f1 = open(os.path.join(self.dest, "rs1_z16.bin"), "wb", buffering=4 * 1024 * 1024)
        self._f2 = open(os.path.join(self.dest, "rs2_z16.bin"), "wb", buffering=4 * 1024 * 1024)
        self.lab80 = None
        self.lab_live = None
        if not self.slim:
            self.lab80 = np.lib.format.open_memmap(
                os.path.join(self.dest, "labels80.npy"), mode="w+",
                dtype=np.uint8, shape=(self.n_max, EGO80_H, EGO80_W))
            self.lab_live = np.lib.format.open_memmap(
                os.path.join(self.dest, "labels80_live.npy"), mode="w+",
                dtype=np.uint8, shape=(self.n_max, EGO80_H, EGO80_W))
        self.stamps = np.lib.format.open_memmap(
            os.path.join(self.dest, "stamps.npy"), mode="w+",
            dtype=np.float64, shape=(self.n_max,))
        self.wheels = np.lib.format.open_memmap(
            os.path.join(self.dest, "wheels.npy"), mode="w+",
            dtype=np.float32, shape=(self.n_max, 8))
        # IMU packed as [n_imu, gx,gy,gz, ax,ay,az, t0..t15 padded]
        self.imu = np.lib.format.open_memmap(
            os.path.join(self.dest, "imu.npy"), mode="w+",
            dtype=np.float32, shape=(self.n_max, 1 + IMU_MAX * 7))
        self.meta = {
            "seconds": self.seconds,
            "fps": fps,
            "n_max": self.n_max,
            "rs": [RS_H, RS_W],
            "ego80": [EGO80_H, EGO80_W],
            "t0_wall": time.time(),
            "intrinsics": {},
            "slim": self.slim,
            "motion_only": True,
        }
        self._meta_path = os.path.join(self.dest, "meta.json")
        self._write_meta()
        self._thr = threading.Thread(target=self._ship, name="rawbag", daemon=True)
        self._thr.start()
        print("raw_bag: %s n_max=%d (%.0fs @ %.0f Hz) slim=%s motion_only" % (
            self.dest, self.n_max, seconds, fps, self.slim), flush=True)

    def _new_slot(self):
        return {
            "t": 0.0,
            "rs1": np.zeros((RS_H, RS_W), dtype=np.uint16),
            "rs2": np.zeros((RS_H, RS_W), dtype=np.uint16),
            "live": None,
            "wheels": np.zeros(8, dtype=np.float32),
            "imu": np.zeros(1 + IMU_MAX * 7, dtype=np.float32),
            "has1": False,
            "has2": False,
        }

    def set_intrinsics(self, name: str, intr: dict):
        self.meta["intrinsics"][name] = {k: float(v) if isinstance(v, (int, float)) else v
                                         for k, v in intr.items()}
        self._write_meta()

    def offer(
        self,
        *,
        t: float,
        rs1_depth: Optional[np.ndarray],
        rs2_depth: Optional[np.ndarray],
        live_labels=None,
        wheels=None,
        imu_pack=None,
    ) -> None:
        # Drops are skipped frames, not slots — do not burn n_max on them.
        if self._closed or self.n >= self.n_max:
            return
        if not getattr(self, "_offered", False):
            self._offered = True
            print("raw_bag: first offer", flush=True)
            try:
                open(os.path.join(self.dest, "first_offer"), "w").write("1\n")
            except Exception:
                pass
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
            self.drops += 1
            return
        slot["t"] = float(t)
        slot["has1"] = False
        slot["has2"] = False
        slot["live"] = None
        if rs1_depth is not None and getattr(rs1_depth, "ndim", 0) == 2:
            if slot["rs1"].shape != rs1_depth.shape:
                slot["rs1"] = np.zeros_like(rs1_depth)
            np.copyto(slot["rs1"], rs1_depth)
            slot["has1"] = True
            self.meta["rs"] = [int(rs1_depth.shape[0]), int(rs1_depth.shape[1])]
        if rs2_depth is not None and getattr(rs2_depth, "ndim", 0) == 2:
            if slot["rs2"].shape != rs2_depth.shape:
                slot["rs2"] = np.zeros_like(rs2_depth)
            np.copyto(slot["rs2"], rs2_depth)
            slot["has2"] = True
        if live_labels is not None:
            slot["live"] = np.array(live_labels, dtype=np.uint8, copy=True)
        if wheels is not None:
            slot["wheels"].fill(0.0)
            n = min(8, len(wheels))
            slot["wheels"][:n] = np.asarray(wheels[:n], dtype=np.float32)
        if imu_pack is not None:
            slot["imu"].fill(0.0)
            n = min(slot["imu"].size, len(imu_pack))
            slot["imu"][:n] = np.asarray(imu_pack[:n], dtype=np.float32)
        try:
            self._q.put_nowait(slot)
        except queue.Full:
            try:
                old = self._q.get_nowait()
                self._free.put_nowait(old)
                self.drops += 1
                self._q.put_nowait(slot)
            except queue.Empty:
                self._free.put_nowait(slot)
                self.drops += 1

    def _ship(self):
        while True:
            slot = self._q.get()
            if slot is None:
                break
            try:
                i = self.n
                if i >= self.n_max:
                    continue
                if slot["has1"]:
                    self._f1.write(np.ascontiguousarray(slot["rs1"]).tobytes())
                if slot["has2"]:
                    self._f2.write(np.ascontiguousarray(slot["rs2"]).tobytes())
                self.stamps[i] = slot["t"]
                np.copyto(self.wheels[i], slot["wheels"])
                np.copyto(self.imu[i], slot["imu"])
                if self.lab80 is not None:
                    try:
                        label_from_z16(
                            slot["rs1"] if slot["has1"] else None,
                            slot["rs2"] if slot["has2"] else None,
                            intr1=self.meta.get("intrinsics", {}).get("rs1"),
                            intr2=self.meta.get("intrinsics", {}).get("rs2"),
                            labels_out=self.lab80[i],
                        )
                    except Exception as e:
                        print("raw_bag: label skip %s" % e, flush=True)
                    live = slot["live"]
                    if live is not None and live.shape == (240, 320):
                        downsample_labels_4(live, out=self.lab_live[i])
                    elif live is not None and live.shape == (EGO80_H, EGO80_W):
                        np.copyto(self.lab_live[i], live)
                self.n = i + 1
                if self.n % 150 == 0:
                    print("raw_bag: n=%d drops=%d dt=%.1fs" % (
                        self.n, self.drops, slot["t"] - self.t0), flush=True)
            except Exception as e:
                print("raw_bag: ship %s" % e, flush=True)
            finally:
                try:
                    self._free.put_nowait(slot)
                except queue.Full:
                    pass

    def done(self) -> bool:
        # Wall clock is the collect budget. Frame count is motion-gated and
        # much smaller than seconds*30, so do not stop on n.
        return (time.monotonic() - self.t0) >= self.seconds

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._thr.join(timeout=8.0)
        n = int(self.n)
        self.meta["n"] = n
        self.meta["drops"] = int(self.drops)
        self.meta["elapsed_s"] = time.monotonic() - self.t0
        self._write_meta()
        # Flush memmaps and shrink-note (keep full files; n is in meta).
        for fh in (getattr(self, "_f1", None), getattr(self, "_f2", None)):
            try:
                if fh:
                    fh.flush()
                    fh.close()
            except Exception:
                pass
        for arr in (self.lab80, self.lab_live, self.stamps, self.wheels, self.imu):
            if arr is None:
                continue
            try:
                arr.flush()
            except Exception:
                pass
        # Contact sheet: first / mid / last 80×60 for sim match.
        try:
            self._contact(n)
        except Exception as e:
            print("raw_bag: contact skip %s" % e, flush=True)
        print("raw_bag: closed n=%d drops=%d dest=%s" % (n, self.drops, self.dest),
              flush=True)

    def _contact(self, n: int):
        if n < 1 or self.lab80 is None:
            return
        import cv2
        lut = np.array([(28, 28, 32), (70, 90, 70), (40, 160, 70), (40, 70, 210)],
                       dtype=np.uint8)
        idxs = [0, n // 2, n - 1]
        tiles = []
        for i in idxs:
            a = lut[np.clip(self.lab80[i], 0, 3)]
            b = lut[np.clip(self.lab_live[i], 0, 3)]
            a = cv2.resize(a, (240, 180), interpolation=cv2.INTER_NEAREST)
            b = cv2.resize(b, (240, 180), interpolation=cv2.INTER_NEAREST)
            tiles.append(np.concatenate([a, b], axis=1))
        sheet = np.concatenate(tiles, axis=0)
        cv2.putText(sheet, "mine | live-ds  n=%d" % n, (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        path = os.path.join(self.dest, "contact80.png")
        cv2.imwrite(path, sheet)

    def _write_meta(self):
        tmp = self._meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=2)
        os.replace(tmp, self._meta_path)


def pack_imu(gyro_list, accel_list, t_list) -> np.ndarray:
    out = np.zeros(1 + IMU_MAX * 7, dtype=np.float32)
    n = min(IMU_MAX, len(gyro_list))
    out[0] = float(n)
    for i in range(n):
        base = 1 + i * 7
        g = gyro_list[i]
        a = accel_list[i] if i < len(accel_list) else (0, 0, 0)
        out[base:base + 3] = g
        out[base + 3:base + 6] = a
        out[base + 6] = t_list[i] if i < len(t_list) else 0.0
    return out


def pack_wheels(snap: dict) -> np.ndarray:
    return np.array([
        float(snap.get("vl", 0.0)),
        float(snap.get("vr", 0.0)),
        float(snap.get("tps_l", 0.0)),
        float(snap.get("tps_r", 0.0)),
        float(snap.get("cmd_l", 0.0) or 0.0),
        float(snap.get("cmd_r", 0.0) or 0.0),
        float(snap.get("ok", 0.0)),
        float(snap.get("age_s", 0.0)),
    ], dtype=np.float32)
