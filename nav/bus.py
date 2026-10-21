"""Shared-memory bus between the nav (brain) process and the safety core process (nav/safety_core.py).

Seqlock blocks (writer: seq odd while writing, even when stable; readers copy and retry on change).  All times are
time.monotonic() seconds (CLOCK_MONOTONIC is system-wide, so both processes read the same clock).
  sc_rs1    camera pump -> core : RS1 verts (n x 3 f32), n, stamp (camera), t_recv (host monotonic)
  sc_rs2    camera pump -> core : the last RS2_RING RS2 depth frames u16 + stamps, t_recv, n = frames written (Rs2Ring)
  sc_rs2pose brain -> core      : RS2 extrinsics (T 3, R 9) when nav.autocal updates them
  sc_cmd    brain -> core       : desired v, w, t_sent (the core is the only writer to the wheels)
  sc_wheel  core -> brain       : vl, vr, t, i_l, i_r, i_age (nan = none), battery
  sc_state  core -> brain       : last decision (see safety_core.STATE_FIELDS)
  sc_ego    core -> brain       : the core's RS1 ego labels per frame (nav.ego_shared.EgoShared reads them; KEVIN_EGO_SHARED)
  sc_viz    core -> brain/viewer: 5 Hz snapshot of the core's 2-D obstacle points (robot frame, now) + band rects
  sc_rs2img / sc_rs2info / sc_imu  RS2 camera process -> brain: full RS2 frame (depth, colour, IR pair), intrinsics, IMU ring
"""
from __future__ import annotations
import os, time
import numpy as np
from multiprocessing import shared_memory, resource_tracker

HDR = 64
PREFIX = os.environ.get("KEVIN_SC_PREFIX", "kevin_sc")
MAX_VERTS = 480 * 848
IMU_RING = 1024
EGO_H, EGO_W = 240, 320   # robot_config FRAME_H, FRAME_W (nav.perceive.Ego label grid; checked in nav.ego_shared)
RS2_COLOR_SHAPE = (240, 320, 3)   # = robot_config FRAME_H, FRAME_W: cameras.RSCamera.color (resized colour); tests/test_rs2_color_bus.py
RS2_RING = 2       # the core pairs each RS1 frame with the RS2 frame captured nearest to it, so it keeps the previous one too


def _untrack(shm):
    try: resource_tracker.unregister(shm._name if hasattr(shm, "_name") else "/" + shm.name, "shared_memory")
    except Exception: pass


class Block:
    def __init__(self, name, layout, create=False):
        self.layout = [(n, tuple(s), np.dtype(d)) for n, s, d in layout]
        size = HDR + sum(int(np.prod(s)) * d.itemsize for _, s, d in self.layout)
        name = "%s_%s" % (PREFIX, name)
        if create:
            try:
                old = shared_memory.SharedMemory(name=name); old.close(); old.unlink()
            except FileNotFoundError: pass
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        else:
            self.shm = shared_memory.SharedMemory(name=name); _untrack(self.shm)
        self.name = name; buf = self.shm.buf
        self.hdr = np.ndarray((8,), np.uint64, buf, 0); self.arr = {}; off = HDR
        for n, s, d in self.layout:
            self.arr[n] = np.ndarray(s, d, buf, off); off += int(np.prod(s)) * d.itemsize
        if create: self.hdr[:] = 0

    def write(self, **arrays):
        self.hdr[0] += 1
        for k, v in arrays.items():
            a = self.arr[k]
            if np.ndim(v) and np.shape(v) != a.shape: a.reshape(-1)[:np.size(v)] = np.asarray(v).reshape(-1)
            else: np.copyto(a, v, casting="unsafe")
        self.hdr[0] += 1

    @property
    def seq(self): return int(self.hdr[0])

    def read(self, keys=None, out=None):
        keys = keys or [k for k in self.arr]
        for _ in range(200):
            s0 = int(self.hdr[0])
            if s0 & 1: time.sleep(0.0001); continue
            res = out if out is not None else {}
            for k in keys:
                if k in res and res[k].shape == self.arr[k].shape: np.copyto(res[k], self.arr[k])
                else: res[k] = self.arr[k].copy()
            if int(self.hdr[0]) == s0: return s0, res
        raise TimeoutError("bus %s: writer stuck" % self.name)

    def try_read(self, keys=None, out=None, budget_s=0.005):
        """read() for the safety core: never raises.  Returns (seq, res), or (None, None) when no consistent copy was
        possible within budget_s (writer paused mid-write = stale data; the caller treats it as no new frame).
        `out` may hold a torn copy afterwards, so on (None, None) the caller must drop it."""
        keys = keys or [k for k in self.arr]; t_end = time.monotonic() + budget_s
        while True:
            s0 = int(self.hdr[0])
            if not (s0 & 1):
                res = out if out is not None else {}
                for k in keys:
                    if k in res and res[k].shape == self.arr[k].shape: np.copyto(res[k], self.arr[k])
                    else: res[k] = self.arr[k].copy()
                if int(self.hdr[0]) == s0: return s0, res
            if time.monotonic() >= t_end: return None, None
            time.sleep(0.0001)

    def close(self, unlink=False):
        self.shm.close()
        if unlink:
            try: self.shm.unlink()
            except FileNotFoundError: pass


def layouts(depth_shape):
    H, W = depth_shape
    return dict(
        rs1=[("verts", (MAX_VERTS, 3), "f4"), ("n", (1,), "i8"), ("stamp", (1,), "f8"), ("t_recv", (1,), "f8")],
        rs2=[("depth", (RS2_RING, H, W), "u2"), ("stamp", (RS2_RING,), "f8"), ("t_recv", (RS2_RING,), "f8"), ("n", (1,), "i8")],
        rs2pose=[("T", (3,), "f8"), ("R", (9,), "f8"), ("set", (1,), "i8")],
        cmd=[("v", (1,), "f8"), ("w", (1,), "f8"), ("t", (1,), "f8")],
        wheel=[("vl", (1,), "f8"), ("vr", (1,), "f8"), ("t", (1,), "f8"), ("il", (1,), "f8"), ("ir", (1,), "f8"), ("iage", (1,), "f8"), ("batt", (1,), "f8")],
        state=[("f", (25,), "f8")],      # = len(safety_core.STATE_FIELDS)
        rs1info=[("f", (6,), "f8"), ("set", (1,), "i8")],
        viz=[("pts", (2000, 2), "f4"), ("n", (1,), "i8"), ("bands", (8,), "f8"), ("t", (1,), "f8")],
        # RS2 camera process (nav/camproc.py) -> brain: the full RS2 frame for SLAM/VLN, its intrinsics, and the IMU ring
        rs2img=[("depth", (H, W), "u2"), ("color", RS2_COLOR_SHAPE, "u1"), ("ir1", (H, W), "u1"), ("ir2", (H, W), "u1"), ("has_ir", (1,), "i8"),
                ("stamp", (1,), "f8"), ("t_recv", (1,), "f8"), ("n", (1,), "i8")],
        rs2info=[("f", (6,), "f8"), ("set", (1,), "i8")],
        imu=[("t", (IMU_RING,), "f8"), ("g", (IMU_RING, 3), "f4"), ("a", (IMU_RING, 3), "f4"), ("w", (1,), "i8"), ("ok", (1,), "i8")],
        # core -> brain: the RS1 ego labels the core computed for its own decision (one Ego, not two): labels, height,
        # map labels (synthetic too-close/cliff obstacles excluded; has_map=0 -> same as lab), safety-grade obstacle mask,
        # occlusion flag; camera stamp, capture time (core FrameClock, host monotonic), core frame counter, sc_rs1 seq
        ego=[("lab", (EGO_H, EGO_W), "u1"), ("hgt", (EGO_H, EGO_W), "u1"), ("lab_map", (EGO_H, EGO_W), "u1"), ("has_map", (1,), "i8"),
             ("obs_mask", (EGO_H, EGO_W), "u1"), ("occluded", (1,), "i8"), ("stamp", (1,), "f8"), ("t_cap", (1,), "f8"),
             ("t_pub", (1,), "f8"), ("frame", (1,), "i8"), ("rs1_seq", (1,), "i8")],
    )


def open_all(depth_shape, create=False):
    return {k: Block(k, L, create=create) for k, L in layouts(depth_shape).items()}


class Rs2Ring:
    """writer side of sc_rs2: frame k (1-based) lives in slot (k-1) % RS2_RING."""
    def __init__(self, block):
        self.b = block; self.d = np.zeros(block.arr["depth"].shape, np.uint16)
        self.s = np.zeros(RS2_RING); self.t = np.zeros(RS2_RING); self.n = 0
    def push(self, depth, stamp, t_recv):
        j = self.n % RS2_RING; self.d[j] = depth; self.s[j] = stamp; self.t[j] = t_recv; self.n += 1
        self.b.write(depth=self.d, stamp=self.s, t_recv=self.t, n=self.n)
