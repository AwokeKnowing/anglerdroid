"""Shared-memory bus between the nav (brain) process and the safety core process (nav/safety_core.py).

Seqlock blocks (writer: seq odd while writing, even when stable; readers copy and retry on change).  All times are
time.monotonic() seconds (CLOCK_MONOTONIC is system-wide, so both processes read the same clock).
  sc_rs1    camera pump -> core : RS1 verts (n x 3 f32), n, stamp (camera), t_recv (host monotonic)
  sc_rs2    camera pump -> core : RS2 depth u16, stamp, t_recv
  sc_rs2pose brain -> core      : RS2 extrinsics (T 3, R 9) when nav.autocal updates them
  sc_cmd    brain -> core       : desired v, w, t_sent (the core is the only writer to the wheels)
  sc_wheel  core -> brain       : vl, vr, t, i_l, i_r, i_age (nan = none), battery
  sc_state  core -> brain       : last decision (see safety_core.STATE_FIELDS)
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

    def close(self, unlink=False):
        self.shm.close()
        if unlink:
            try: self.shm.unlink()
            except FileNotFoundError: pass


def layouts(depth_shape):
    H, W = depth_shape
    return dict(
        rs1=[("verts", (MAX_VERTS, 3), "f4"), ("n", (1,), "i8"), ("stamp", (1,), "f8"), ("t_recv", (1,), "f8")],
        rs2=[("depth", (H, W), "u2"), ("stamp", (1,), "f8"), ("t_recv", (1,), "f8")],
        rs2pose=[("T", (3,), "f8"), ("R", (9,), "f8"), ("set", (1,), "i8")],
        cmd=[("v", (1,), "f8"), ("w", (1,), "f8"), ("t", (1,), "f8")],
        wheel=[("vl", (1,), "f8"), ("vr", (1,), "f8"), ("t", (1,), "f8"), ("il", (1,), "f8"), ("ir", (1,), "f8"), ("iage", (1,), "f8"), ("batt", (1,), "f8")],
        state=[("f", (24,), "f8")],
        rs1info=[("f", (6,), "f8"), ("set", (1,), "i8")],
        viz=[("pts", (2000, 2), "f4"), ("n", (1,), "i8"), ("bands", (8,), "f8"), ("t", (1,), "f8")],
        # RS2 camera process (nav/camproc.py) -> brain: the full RS2 frame for SLAM/VLN, its intrinsics, and the IMU ring
        rs2img=[("depth", (H, W), "u2"), ("color", (480, 640, 3), "u1"), ("ir1", (H, W), "u1"), ("ir2", (H, W), "u1"), ("has_ir", (1,), "i8"),
                ("stamp", (1,), "f8"), ("t_recv", (1,), "f8"), ("n", (1,), "i8")],
        rs2info=[("f", (6,), "f8"), ("set", (1,), "i8")],
        imu=[("t", (IMU_RING,), "f8"), ("g", (IMU_RING, 3), "f4"), ("a", (IMU_RING, 3), "f4"), ("w", (1,), "i8"), ("ok", (1,), "i8")],
    )


def open_all(depth_shape, create=False):
    return {k: Block(k, L, create=create) for k, L in layouts(depth_shape).items()}
