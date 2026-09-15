#!/usr/bin/env python3
"""SPS bench for the GPU insect vec-env. Does not touch sim/explore/."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_SIM = Path(__file__).resolve().parent.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

from gpu_insect.env import GpuInsectVec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    env = GpuInsectVec(n=args.n, seed=7, device=args.device)
    act = torch.zeros(env.n, 4, device=env.torch_dev)
    act[:, 0] = 0.4
    for _ in range(args.warmup):
        env.step(act)
        act = torch.randn(env.n, 4, device=env.torch_dev).clamp(-1, 1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(args.steps):
        if i % 12 == 0:
            act = torch.randn(env.n, 4, device=env.torch_dev).clamp(-1, 1)
        env.step(act)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = args.n * args.steps / max(dt, 1e-6)
    used = torch.cuda.memory_allocated() / (1024 ** 2)
    st = env.stats()
    snap = env.env0_cpu()
    print(
        "gpu_insect N=%d steps=%d wall=%.3fs sps=%.0f gpu=%.0fMiB nbox0=%d "
        "win=%.3f path=%.2f plus_x=%.2f xy=%.2f,%.2f"
        % (args.n, args.steps, dt, sps, used, snap["nbox"], st["win_frac"],
           st["mean_path"], snap["plus_x"], snap["x"], snap["y"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
