#!/usr/bin/env python3
"""Time 80×60 label_frame_80 on this box. Wheels never touched.

  python3 src/probe_ego80_ms.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from cameras import RSCamera
    from perception.fast_ego80 import (
        label_frame_80, labels_to_ego_float, stamp_throttle_float,
    )

    import pyrealsense2 as rs
    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]
    serials.sort()
    if len(serials) < 1:
        print("probe_ego80: no RealSense")
        return 2
    rs1 = RSCamera(serials[0], compute_pointcloud=True)
    rs2 = RSCamera(serials[1], compute_pointcloud=True) if len(serials) > 1 else None
    lab = np.zeros((60, 80), dtype=np.uint8)
    flo = np.zeros((60, 80), dtype=np.float32)
    # warmup + cache fill
    for _ in range(8):
        rs1.grab_latest()
        if rs2:
            rs2.grab_latest()
        label_frame_80(rs1.verts, rs2.verts if rs2 else None, labels_out=lab)
        labels_to_ego_float(lab, out=flo)
        stamp_throttle_float(flo, 0.0, 0.0, 0.0, 0.0)
    n = int(os.environ.get("KEVIN_EGO80_N", "120"))
    ms = []
    n1 = n2 = 0
    for _ in range(n):
        rs1.grab_latest()
        if rs2:
            rs2.grab_latest()
        v1, v2 = rs1.verts, (rs2.verts if rs2 else None)
        n1 = 0 if v1 is None else len(v1)
        n2 = 0 if v2 is None else len(v2)
        t0 = time.perf_counter()
        label_frame_80(v1, v2, labels_out=lab)
        labels_to_ego_float(lab, out=flo)
        stamp_throttle_float(flo, 0.1, 0.0, 0.0, 0.0)
        ms.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ms, dtype=np.float64)
    print(
        "ego80  verts rs1=%d rs2=%d  n=%d  p50=%.2fms  p95=%.2fms  "
        "max=%.2fms  mean=%.2fms  30Hz_ok=%s  60Hz_ok=%s"
        % (n1, n2, a.size, np.median(a), np.percentile(a, 95), a.max(), a.mean(),
           "yes" if np.percentile(a, 95) < 20.0 else "NO",
           "yes" if np.percentile(a, 95) < 12.0 else "NO"),
        flush=True,
    )
    rs1.stop()
    if rs2:
        rs2.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
