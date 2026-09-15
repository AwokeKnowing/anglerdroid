#!/usr/bin/env python3
"""Rebuild / inspect a Kevin raw bag as 80×60 unknown/clear/obstacle.

Uses sim/rl_nav/ego80.py (copied live labeler). Depth size comes from
meta.intrinsics, not a hardcoded 848.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bag_io import load_meta, mmap_z16  # noqa: E402
from ego80 import (  # noqa: E402
    CLEAR, EGO80_H, EGO80_W, OBSTACLE, SELF, UNKNOWN,
    label_from_z16, labels_to_ego_float,
)


def _stats(lab):
    n = lab.size
    return {
        "unknown": float(np.mean(lab == UNKNOWN)),
        "self": float(np.mean(lab == SELF)),
        "clear": float(np.mean(lab == CLEAR)),
        "obstacle": float(np.mean(lab == OBSTACLE)),
        "n": int(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-run z16→80x60 into labels80_rebuild.npy")
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()
    bag = args.bag
    meta = load_meta(bag)
    n = int(meta.get("n") or 0)
    lab_path = bag / "labels80.npy"
    if lab_path.is_file():
        lab = np.load(lab_path, mmap_mode="r")
        if n <= 0:
            n = int(lab.shape[0])
        s = _stats(np.asarray(lab[:n]))
        print("labels80 n=%d unk=%.3f self=%.3f clear=%.3f obs=%.3f" % (
            n, s["unknown"], s["self"], s["clear"], s["obstacle"]))
    live_p = bag / "labels80_live.npy"
    if live_p.is_file():
        live = np.load(live_p, mmap_mode="r")
        nn = n if n > 0 else int(live.shape[0])
        s = _stats(np.asarray(live[:nn]))
        print("labels80_live n=%d unk=%.3f self=%.3f clear=%.3f obs=%.3f" % (
            nn, s["unknown"], s["self"], s["clear"], s["obstacle"]))
        if lab_path.is_file():
            a = np.asarray(lab[:nn])
            b = np.asarray(live[:nn])
            agree = float(np.mean(a == b))
            print("mine vs live-ds agree=%.3f" % agree)
    if args.rebuild:
        z1 = mmap_z16(bag / "rs1_z16.bin", meta, "rs1") if (bag / "rs1_z16.bin").is_file() else None
        z2 = mmap_z16(bag / "rs2_z16.bin", meta, "rs2") if (bag / "rs2_z16.bin").is_file() else None
        if z1 is None:
            print("no rs1_z16.bin"); return
        nfr = int(z1.shape[0])
        print("z16 rs1=%s rs2=%s" % (z1.shape, None if z2 is None else z2.shape), flush=True)
        out = np.lib.format.open_memmap(
            str(bag / "labels80_rebuild.npy"), mode="w+",
            dtype=np.uint8, shape=(nfr, EGO80_H, EGO80_W))
        intr = meta.get("intrinsics", {})
        t0 = time.monotonic()
        every = max(1, int(args.every))
        k = 0
        for i in range(0, nfr, every):
            z2i = z2[i] if z2 is not None and i < z2.shape[0] else None
            label_from_z16(
                z1[i], z2i,
                intr1=intr.get("rs1"), intr2=intr.get("rs2"),
                labels_out=out[i],
            )
            k += 1
            if k % 150 == 0:
                dt = time.monotonic() - t0
                hz = k / max(dt, 1e-6)
                print("rebuild %d/%d  %.0f Hz" % (i, nfr, hz), flush=True)
        out.flush()
        dt = time.monotonic() - t0
        print("rebuild done %d frames in %.1fs (%.0f Hz)" % (k, dt, k / max(dt, 1e-6)))
        s = _stats(np.asarray(out[:nfr]))
        print("rebuild unk=%.3f self=%.3f clear=%.3f obs=%.3f" % (
            s["unknown"], s["self"], s["clear"], s["obstacle"]))
    ego_p = bag / "ego80_float.npy"
    src = bag / "labels80_rebuild.npy"
    if not src.is_file():
        src = lab_path
    if src.is_file() and not ego_p.is_file():
        lab = np.load(src, mmap_mode="r")
        nn = int(lab.shape[0])
        ego = np.lib.format.open_memmap(
            str(ego_p), mode="w+", dtype=np.float32, shape=(nn, EGO80_H, EGO80_W))
        for i in range(nn):
            labels_to_ego_float(lab[i], out=ego[i])
        ego.flush()
        print("wrote", ego_p)


if __name__ == "__main__":
    main()
