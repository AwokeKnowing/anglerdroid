#!/usr/bin/env python3
"""Write a short env0 clip with current look (self stamp + throttle marks)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cv2  # noqa: E402

from dash import _ego_occupancy, _stamp_self, _stamp_throttle, snaps_to_clip  # noqa: E402
from env import RlNavVec  # noqa: E402
from kernels import LOOK_KNOBS, N_V, N_W  # noqa: E402

OUT = _HERE / "out"


def _prims(n, device):
    iv = torch.randint(4, N_V, (n,), device=device)
    iw = torch.randint(1, N_W - 1, (n,), device=device)
    return (iv * N_W + iw).to(dtype=torch.int32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 16
    env = RlNavVec(n=n, seed=11, device="cuda:0")
    env.set_look(LOOK_KNOBS)
    write = torch.zeros(n, 64, device=env.torch_dev)
    snaps = []
    for t in range(240):
        env.step(_prims(n, env.torch_dev), write)
        if t % 2 == 0:
            snaps.append(env.env0_cpu())
    stem = OUT / "clips" / "clip_look_throt"
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = snaps_to_clip(snaps, stem, st=env.stats(), note="look+throttle seed", sps=0.0)
    snap = snaps[min(40, len(snaps) - 1)]
    ego = cv2.resize(_ego_occupancy(snap["ego"]), (320, 240), interpolation=cv2.INTER_NEAREST)
    _stamp_self(ego)
    _stamp_throttle(ego, snap)
    ego_path = OUT / "bag_probe" / "look_self_blob.png"
    ego_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ego_path), ego)
    for src, dest in (
        (out["png"], OUT / "last.png"),
        (out["mp4"], OUT / "last.mp4"),
        (out.get("gif"), OUT / "last.gif"),
        (out["last"], OUT / "last_frame.png"),
    ):
        if src is None:
            continue
        dest = Path(dest)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(src.resolve())
    print("seed clip", out["png"], "frames", out["n"], flush=True)


if __name__ == "__main__":
    main()
