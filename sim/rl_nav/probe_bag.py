"""Sample bag frames → 80×60 labels, contact tiles, vs sim k_obs rates.

Transfer target is the rebuild tensor (same ego80.py sim will match), not a
pretty camera. Live labels80_live is a check when present. Sim paints the
bag-baked FOV silhouette — no RealSense projection in the 5 Hz loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bag_io import load_meta, mmap_z16  # noqa: E402
from ego80 import (  # noqa: E402
    CLEAR, EGO80_H, EGO80_W, OBSTACLE, SELF, UNKNOWN,
    label_from_z16, paint_self_80,
)

PAL = np.array([
    [18, 18, 22],       # UNKNOWN
    [40, 90, 220],      # SELF
    [40, 170, 90],      # CLEAR
    [210, 50, 40],      # OBSTACLE
], dtype=np.uint8)


def _colorize(lab: np.ndarray) -> np.ndarray:
    return PAL[np.clip(lab.astype(np.int32), 0, 3)]


def _ego_float_to_lab(ego: np.ndarray) -> np.ndarray:
    """Policy 0/0.5/1 raster → bag labels, with SELF footprint overlaid."""
    x = np.asarray(ego, dtype=np.float32)
    if x.ndim == 3:
        x = x[0]
    lab = np.full(x.shape, UNKNOWN, dtype=np.uint8)
    lab[(x >= 0.25) & (x < 0.75)] = CLEAR
    lab[x >= 0.75] = OBSTACLE
    paint_self_80(lab)
    return lab


def _depth_preview(z16: np.ndarray) -> np.ndarray:
    z = z16.astype(np.float32)
    m = z > 0
    out = np.zeros((*z.shape, 3), dtype=np.uint8)
    if not np.any(m):
        return out
    lo, hi = np.percentile(z[m], [5, 95])
    hi = max(hi, lo + 1.0)
    g = np.clip((z - lo) / (hi - lo), 0, 1)
    g = (255.0 * (1.0 - g)).astype(np.uint8)
    out[..., 0] = g
    out[..., 1] = g
    out[..., 2] = np.where(m, g, 0)
    return out


def _scale(img: np.ndarray, s: int = 4) -> np.ndarray:
    return np.repeat(np.repeat(img, s, axis=0), s, axis=1)


def _stats(lab: np.ndarray) -> dict:
    return {
        "unk": float(np.mean(lab == UNKNOWN)),
        "self": float(np.mean(lab == SELF)),
        "clear": float(np.mean(lab == CLEAR)),
        "obs": float(np.mean(lab == OBSTACLE)),
    }


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path = Path(path)
    try:
        import cv2
        cv2.imwrite(str(path), rgb[:, :, ::-1])
        return
    except Exception:
        pass
    h, w = rgb.shape[:2]
    path.with_suffix(".ppm").write_bytes(
        ("P6\n%d %d\n255\n" % (w, h)).encode("ascii") + rgb.tobytes())


def _hstack(parts: list) -> np.ndarray:
    gap = np.full((parts[0].shape[0], 4, 3), 8, dtype=np.uint8)
    row = parts[0]
    for p in parts[1:]:
        row = np.concatenate([row, gap, p], axis=1)
    return row


def _vstack(parts: list) -> np.ndarray:
    gap = np.full((4, parts[0].shape[1], 3), 8, dtype=np.uint8)
    sheet = parts[0]
    for t in parts[1:]:
        sheet = np.concatenate([sheet, gap, t], axis=0)
    return sheet


def _sim_bundle(n: int = 8) -> dict | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from env import RlNavVec
        env = RlNavVec(n=max(n, 64), seed=3, device="cuda:0")
        ego = env.ego.detach().cpu().numpy()
        unk = float(np.mean(ego < 0.25))
        clear = float(np.mean((ego >= 0.25) & (ego < 0.75)))
        obs = float(np.mean(ego >= 0.75))
        tiles = [_ego_float_to_lab(ego[i]) for i in range(n)]
        return {
            "unk": unk, "self": 0.0, "clear": clear, "obs": obs,
            "n": int(ego.shape[0]), "tiles": tiles,
        }
    except Exception as e:
        print("sim k_obs skip: %s" % e, flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path, nargs="?",
                    default=_HERE / "bags" / "wander_5min_pilot")
    ap.add_argument("--n-tiles", type=int, default=8)
    ap.add_argument("--out", type=Path, default=_HERE / "out" / "bag_probe")
    args = ap.parse_args()
    bag = args.bag
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(bag)
    z1 = mmap_z16(bag / "rs1_z16.bin", meta, "rs1")
    z2p = bag / "rs2_z16.bin"
    z2 = mmap_z16(z2p, meta, "rs2") if z2p.is_file() and z2p.stat().st_size > 0 else None
    nfr = int(z1.shape[0])
    print("bag %s  z1=%s z2=%s meta.n=%s" % (
        bag.name, z1.shape, None if z2 is None else z2.shape, meta.get("n")))
    live_p = bag / "labels80_live.npy"
    live = np.load(live_p, mmap_mode="r") if live_p.is_file() else None

    idxs = np.linspace(0, nfr - 1, num=max(1, args.n_tiles), dtype=np.int32)
    rebuild = np.zeros((len(idxs), EGO80_H, EGO80_W), dtype=np.uint8)
    rows = []
    for k, i in enumerate(idxs):
        z2i = z2[i] if z2 is not None and i < z2.shape[0] else None
        label_from_z16(
            z1[i], z2i,
            intr1=(meta.get("intrinsics") or {}).get("rs1"),
            intr2=(meta.get("intrinsics") or {}).get("rs2"),
            labels_out=rebuild[k],
        )
        st = _stats(rebuild[k])
        rec = {"i": int(i), **st}
        if live is not None and i < live.shape[0]:
            rec["live_agree"] = float(np.mean(rebuild[k] == live[i]))
            rec["live"] = _stats(np.asarray(live[i]))
        rows.append(rec)
        print("frame %d  unk=%.3f clr=%.3f obs=%.3f self=%.3f%s" % (
            i, st["unk"], st["clear"], st["obs"], st["self"],
            ("  live_agree=%.3f" % rec["live_agree"]) if "live_agree" in rec else ""))

    step = max(1, nfr // 400)
    acc = np.zeros(4, dtype=np.int64)
    n_s = 0
    buf = np.zeros((EGO80_H, EGO80_W), dtype=np.uint8)
    for i in range(0, nfr, step):
        z2i = z2[i] if z2 is not None and i < z2.shape[0] else None
        label_from_z16(
            z1[i], z2i,
            intr1=(meta.get("intrinsics") or {}).get("rs1"),
            intr2=(meta.get("intrinsics") or {}).get("rs2"),
            labels_out=buf,
        )
        acc[UNKNOWN] += int(np.count_nonzero(buf == UNKNOWN))
        acc[SELF] += int(np.count_nonzero(buf == SELF))
        acc[CLEAR] += int(np.count_nonzero(buf == CLEAR))
        acc[OBSTACLE] += int(np.count_nonzero(buf == OBSTACLE))
        n_s += 1
    tot = max(1, int(acc.sum()))
    bag_rates = {
        "unk": acc[UNKNOWN] / tot,
        "self": acc[SELF] / tot,
        "clear": acc[CLEAR] / tot,
        "obs": acc[OBSTACLE] / tot,
        "n_frames": n_s,
        "step": step,
    }
    print("bag subsample n=%d  unk=%.3f clr=%.3f obs=%.3f self=%.3f" % (
        n_s, bag_rates["unk"], bag_rates["clear"], bag_rates["obs"], bag_rates["self"]))

    sim = _sim_bundle(n=args.n_tiles)
    sim_rates = None
    if sim:
        sim_rates = {k: sim[k] for k in ("unk", "self", "clear", "obs", "n")}
        print("sim k_obs  n=%d  unk=%.3f clr=%.3f obs=%.3f" % (
            sim_rates["n"], sim_rates["unk"], sim_rates["clear"], sim_rates["obs"]))

    import cv2
    tiles = []
    look = []
    for k, i in enumerate(idxs):
        lab = _scale(_colorize(rebuild[k]), 4)
        d = cv2.resize(
            _depth_preview(z1[i]), (lab.shape[1], lab.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        parts = [d, lab]
        if live is not None and i < live.shape[0]:
            parts.append(_scale(_colorize(np.asarray(live[i])), 4))
        tiles.append(_hstack(parts))
        if sim and k < len(sim["tiles"]):
            look.append(_hstack([
                lab,
                _scale(_colorize(sim["tiles"][k]), 4),
            ]))
    _write_png(out_dir / ("contact_%s.png" % bag.name), _vstack(tiles))
    if look:
        _write_png(out_dir / ("look_%s.png" % bag.name), _vstack(look))
        print("wrote", out_dir / ("look_%s.png" % bag.name))

    summary = {
        "bag": str(bag),
        "z1": list(z1.shape),
        "z2": None if z2 is None else list(z2.shape),
        "tiles": rows,
        "bag_rates": bag_rates,
        "sim_rates": sim_rates,
        "strategy": (
            "Match sim k_obs to bag rebuild LOOK: baked FOV stencil, 3-tone "
            "UNKNOWN/CLEAR/OBS. No z16 deproject in the sim loop."
        ),
    }
    (out_dir / ("stats_%s.json" % bag.name)).write_text(json.dumps(summary, indent=2))
    print("wrote", out_dir / ("contact_%s.png" % bag.name))


if __name__ == "__main__":
    main()
