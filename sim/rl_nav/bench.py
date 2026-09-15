#!/usr/bin/env python3
"""SPS bench for rl_nav. Does not touch sim/explore/."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_SIM = Path(__file__).resolve().parent.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

from rl_nav.env import RlNavVec  # noqa: E402
from rl_nav import kernels as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    env = RlNavVec(n=args.n, seed=7, device=args.device)
    prim = torch.zeros(env.n, device=env.torch_dev, dtype=torch.int32)
    write = torch.zeros(env.n, K.MEM_N, device=env.torch_dev)
    for _ in range(args.warmup):
        env.step(prim, write)
        prim = torch.randint(0, K.N_PRIM, (env.n,), device=env.torch_dev, dtype=torch.int32)
        write = torch.randn(env.n, K.MEM_N, device=env.torch_dev).clamp(-1, 1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(args.steps):
        if i % 8 == 0:
            prim = torch.randint(0, K.N_PRIM, (env.n,), device=env.torch_dev, dtype=torch.int32)
            write = torch.randn(env.n, K.MEM_N, device=env.torch_dev).clamp(-1, 1)
        env.step(prim, write)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = args.n * args.steps / max(dt, 1e-6)
    used = torch.cuda.memory_allocated() / (1024 ** 2)
    st = env.stats()
    snap = env.env0_cpu()
    print(
        "rl_nav N=%d steps=%d wall=%.3fs sps=%.0f gpu=%.0fMiB nbox0=%d "
        "hits=%.2f crash=%.2f plus_x=%.2f xy=%.2f,%.2f ego_nz=%.3f"
        % (args.n, args.steps, dt, sps, used, snap["nbox"], st["mean_hits"],
           st["crash_frac"], snap["plus_x"], snap["x"], snap["y"],
           float((env.ego > 0).float().mean().item())),
        flush=True,
    )


if __name__ == "__main__":
    main()
