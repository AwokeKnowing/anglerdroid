"""Shared-memory transport between the emulator world (emu/world.py) and the HAL shims (hal/sim/*).

One block per stream, seqlock protocol: writer bumps seq to odd, writes payload, bumps to even.
Readers copy the payload and retry if seq changed. All stamps are CLOCK_MONOTONIC ns (same clock the
robot code reads with time.monotonic()), so the code under test sees real-time sensor stamps.
"""
from __future__ import annotations
import json, os, time, struct
import numpy as np
from multiprocessing import shared_memory, resource_tracker

HDR = 64                      # seq u64, stamp_ns u64, frame_no u64, reserved
INFO_PATH = os.environ.get("KEVIN_EMU_INFO", "/dev/shm/kevin_emu.json")


def _untrack(shm):
    # readers must not unlink on exit (python <3.13 tracker quirk)
    try:
        resource_tracker.unregister(shm._name if hasattr(shm, "_name") else "/" + shm.name, "shared_memory")
    except Exception:
        pass


class Block:
    """A named shm block holding a fixed set of numpy arrays (layout = list of (name, shape, dtype))."""

    def __init__(self, name: str, layout, create: bool = False):
        self.layout = [(n, tuple(s), np.dtype(d)) for n, s, d in layout]
        size = HDR + sum(int(np.prod(s)) * d.itemsize for _, s, d in self.layout)
        if create:
            try:
                old = shared_memory.SharedMemory(name=name); old.close(); old.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            _untrack(self.shm)
        self.name = name
        buf = self.shm.buf
        self.hdr = np.ndarray((8,), np.uint64, buf, 0)
        self.arr = {}
        off = HDR
        for n, s, d in self.layout:
            self.arr[n] = np.ndarray(s, d, buf, off)
            off += int(np.prod(s)) * d.itemsize
        if create:
            self.hdr[:] = 0

    # ---- writer ----
    def write(self, stamp_ns: int, **arrays):
        self.hdr[0] += 1                      # odd: writing
        for k, v in arrays.items():
            np.copyto(self.arr[k], v, casting="unsafe")
        self.hdr[1] = stamp_ns
        self.hdr[2] += 1
        self.hdr[0] += 1                      # even: stable

    # ---- reader ----
    @property
    def seq(self) -> int:
        return int(self.hdr[0])

    def read(self, out: dict | None = None):
        """Consistent snapshot -> (seq, stamp_ns, dict of copies)."""
        for _ in range(100):
            s0 = int(self.hdr[0])
            if s0 & 1:
                time.sleep(0.0002); continue
            res = out if out is not None else {k: np.empty_like(a) for k, a in self.arr.items()}
            for k, a in self.arr.items():
                np.copyto(res[k], a)
            st = int(self.hdr[1])
            if int(self.hdr[0]) == s0:
                return s0, st, res
        raise TimeoutError("shm %s: writer stuck" % self.name)

    def wait_new(self, last_seq: int, timeout_s: float) -> bool:
        t_end = time.monotonic() + timeout_s
        while True:
            s = int(self.hdr[0])
            if s != last_seq and not (s & 1):
                return True
            if time.monotonic() > t_end:
                return False
            time.sleep(0.0005)

    def close(self, unlink=False):
        self.shm.close()
        if unlink:
            try: self.shm.unlink()
            except FileNotFoundError: pass


class ImuRing:
    """Ring of IMU samples: t_ns(f8) gyro(3 f4) accel(3 f4). hdr[0] = total written."""
    N = 2048
    DT = np.dtype([("t", "<i8"), ("g", "<f4", 3), ("a", "<f4", 3)])

    def __init__(self, name: str, create=False):
        size = HDR + self.N * self.DT.itemsize
        if create:
            try:
                old = shared_memory.SharedMemory(name=name); old.close(); old.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        else:
            self.shm = shared_memory.SharedMemory(name=name); _untrack(self.shm)
        self.hdr = np.ndarray((8,), np.uint64, self.shm.buf, 0)
        self.ring = np.ndarray((self.N,), self.DT, self.shm.buf, HDR)
        if create:
            self.hdr[:] = 0

    def push(self, t_ns, g, a):
        i = int(self.hdr[0]) % self.N
        self.ring[i] = (t_ns, g, a)
        self.hdr[0] += 1

    def since(self, n_seen: int):
        n = int(self.hdr[0])
        if n - n_seen > self.N:
            n_seen = n - self.N
        idx = np.arange(n_seen, n) % self.N
        return n, self.ring[idx].copy()


def write_info(info: dict):
    tmp = INFO_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(info, f)
    os.replace(tmp, INFO_PATH)


def read_info(wait_s: float = 10.0) -> dict:
    t_end = time.monotonic() + wait_s
    while True:
        try:
            with open(INFO_PATH) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() > t_end:
                raise RuntimeError("emulator world not running (no %s)" % INFO_PATH)
            time.sleep(0.1)
