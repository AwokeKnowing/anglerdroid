"""Kevin 80×60 FOV stencil for the house sim.

Bag rebuilds show a stable RS1 house-footprint ∪ RS2 cone — not a filled
AABB. Bake that silhouette once from z16, then k_obs only paints inside it.
No deproject on the 5 Hz path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
STENCIL_PATH = _HERE / "fov80_rs1.npy"
FOV_THRESH = 0.22


def geometric_fallback(h: int = 60, w: int = 80) -> np.ndarray:
    """Coarse RS1 blob + 80°/2.5 m cone if no bag stencil is on disk."""
    from kernels import (
        EGO_H, EGO_PX, EGO_W, EGO_X0, EGO_X1, EGO_Y0, EGO_Y1,
        RS1_X0, RS1_X1, RS1_Y0, RS1_Y1, RS2_HALF, RS2_RANGE,
    )
    assert h == EGO_H and w == EGO_W
    ys = EGO_Y1 - (np.arange(h, dtype=np.float32) + 0.5) * EGO_PX
    xs = EGO_X0 + (np.arange(w, dtype=np.float32) + 0.5) * ((EGO_X1 - EGO_X0) / w)
    xx, yy = np.meshgrid(xs, ys)
    rs1 = (xx >= RS1_X0) & (xx <= RS1_X1) & (yy >= RS1_Y0) & (yy <= RS1_Y1)
    dist = np.sqrt(xx * xx + yy * yy)
    ang = np.arctan2(yy, xx)
    rs2 = (dist > 0.05) & (dist <= RS2_RANGE) & (np.abs(ang) <= RS2_HALF)
    return (rs1 | rs2).astype(np.uint8)


def load_stencil() -> np.ndarray:
    if STENCIL_PATH.is_file():
        m = np.load(STENCIL_PATH)
        if m.shape == (60, 80):
            return m.astype(np.uint8)
    return geometric_fallback()


def bake_from_bags(bags, out_path: Path | None = None, n_frames: int = 400) -> np.ndarray:
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from bag_io import load_meta, mmap_z16
    from ego80 import CLEAR, EGO80_H, EGO80_W, OBSTACLE, SELF, UNKNOWN, label_from_z16

    acc = np.zeros((EGO80_H, EGO80_W), dtype=np.float64)
    n = 0
    buf = np.zeros((EGO80_H, EGO80_W), dtype=np.uint8)
    for bag in bags:
        bag = Path(bag)
        meta = load_meta(bag)
        z1p = bag / "rs1_z16.bin"
        if not z1p.is_file() or z1p.stat().st_size < 10_000:
            continue
        z1 = mmap_z16(z1p, meta, "rs1")
        z2p = bag / "rs2_z16.bin"
        z2 = mmap_z16(z2p, meta, "rs2") if z2p.is_file() and z2p.stat().st_size > 0 else None
        nfr = int(z1.shape[0])
        step = max(1, nfr // max(1, n_frames // max(1, len(list(bags)))))
        for i in range(0, nfr, step):
            z2i = z2[i] if z2 is not None and i < z2.shape[0] else None
            label_from_z16(
                z1[i], z2i,
                intr1=(meta.get("intrinsics") or {}).get("rs1"),
                intr2=(meta.get("intrinsics") or {}).get("rs2"),
                labels_out=buf,
            )
            acc += (buf != UNKNOWN).astype(np.float64)
            n += 1
    if n < 1:
        return geometric_fallback()
    p = acc / float(n)
    stencil = (p >= FOV_THRESH).astype(np.uint8)
    dest = Path(out_path) if out_path is not None else STENCIL_PATH
    np.save(dest, stencil)
    meta_path = dest.with_suffix(".json")
    import json
    meta_path.write_text(json.dumps({
        "n_frames": n,
        "thresh": FOV_THRESH,
        "known_frac": float(stencil.mean()),
        "mean_p_known": float(p.mean()),
        "p_known_inside": float(p[stencil == 1].mean()) if np.any(stencil) else 0.0,
        "bags": [str(b) for b in bags],
    }, indent=2))
    png = dest.with_suffix(".png")
    rgb = np.zeros((EGO80_H, EGO80_W, 3), dtype=np.uint8)
    rgb[..., 1] = (np.clip(p, 0, 1) * 255).astype(np.uint8)
    rgb[stencil == 0] = 18
    try:
        import cv2
        vis = np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)
        cv2.imwrite(str(png), vis[:, :, ::-1])
    except Exception:
        pass
    print("fov stencil n=%d known_frac=%.3f p_in=%.3f -> %s" % (
        n, stencil.mean(), float(p[stencil == 1].mean()) if np.any(stencil) else 0.0, dest),
        flush=True)
    return stencil


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="*", type=Path)
    ap.add_argument("--n-frames", type=int, default=400)
    args = ap.parse_args()
    bags = args.bags or [
        _HERE / "bags" / "wander_5min_pilot",
        _HERE / "bags" / "wander_pilot_d",
    ]
    bake_from_bags(bags, n_frames=args.n_frames)
