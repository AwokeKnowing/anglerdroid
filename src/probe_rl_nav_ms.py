#!/usr/bin/env python3
"""Time PPO/ONNX on this box. Wheels never touched.

  python3 src/probe_rl_nav_ms.py
  python3 src/probe_rl_nav_ms.py --ckpt ~/.kevin/rl_nav.onnx --n 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.kevin/rl_nav.onnx"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()
    from rl_nav_live import RlNavLive, EGO_H, EGO_W

    path = os.path.expanduser(args.ckpt)
    if not os.path.isfile(path):
        print("probe_rl_nav: missing %s" % path)
        return 2
    net = RlNavLive(path)
    net._last_t = 0.0
    ego = np.full((EGO_H, EGO_W), 0.5, dtype=np.float32)
    ego[20:28, 38:48] = 1.0
    for _ in range(int(args.warmup)):
        net._last_t = 0.0
        net.act(ego, plus_x=1.2)
    ms = []
    for _ in range(int(args.n)):
        net._last_t = 0.0
        t0 = time.perf_counter()
        net.act(ego, plus_x=1.2)
        ms.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ms, dtype=np.float64)
    print(
        "rl_nav infer  backend=%s  n=%d  p50=%.2fms  p95=%.2fms  "
        "max=%.2fms  mean=%.2fms  5Hz_ok=%s"
        % (net._backend, a.size, np.median(a), np.percentile(a, 95),
           a.max(), a.mean(), "yes" if np.percentile(a, 95) < 180.0 else "NO"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
