#!/usr/bin/env python3
"""Real RS z16 → 80×60 next to sim k_obs, same HUD LUT / 320×240 ego panel."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent / "src"
for p in (_HERE, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bag_io import load_meta, mmap_z16  # noqa: E402
from perception.fast_ego80 import (  # noqa: E402
    CLEAR, OBSTACLE, SELF, UNKNOWN,
    label_from_z16, labels_to_ego_float, render_policy_bgr,
)

OUT = _HERE / "out" / "transfer"
_EGO = (8, 428, 240, 320)  # y, x, h, w in dash.py compose_hud


def _caption(bgr: np.ndarray, name: str) -> np.ndarray:
    bar = np.full((26, bgr.shape[1], 3), 12, dtype=np.uint8)
    cv2.putText(bar, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([bar, bgr], axis=0)


def _gap(h: int, w: int = 8) -> np.ndarray:
    return np.full((h, w, 3), 8, dtype=np.uint8)


def _row(tiles: list, gap: int = 8) -> np.ndarray:
    out = tiles[0]
    for t in tiles[1:]:
        out = np.concatenate([out, _gap(out.shape[0], gap), t], axis=1)
    return out


def _stack(rows: list, gap: int = 10) -> np.ndarray:
    w = max(r.shape[1] for r in rows)

    def pad(im):
        if im.shape[1] >= w:
            return im
        return np.concatenate([im, np.full((im.shape[0], w - im.shape[1], 3), 8, dtype=np.uint8)], 1)

    out = pad(rows[0])
    for r in rows[1:]:
        out = np.concatenate([out, np.full((gap, w, 3), 8, dtype=np.uint8), pad(r)], 0)
    return out


def _sim_ego() -> np.ndarray | None:
    p = _HERE / "out" / "last_frame.png"
    if not p.is_file():
        return None
    img = cv2.imread(str(p))
    if img is None:
        return None
    y, x, h, w = _EGO
    if img.shape[0] < y + h or img.shape[1] < x + w:
        return None
    return img[y:y + h, x:x + w]


def _rates(lab: np.ndarray) -> dict:
    n = float(lab.size)
    return {
        "unk": float(np.count_nonzero(lab == UNKNOWN)) / n,
        "self": float(np.count_nonzero(lab == SELF)) / n,
        "clear": float(np.count_nonzero(lab == CLEAR)) / n,
        "obs": float(np.count_nonzero(lab == OBSTACLE)) / n,
    }


def main():
    bag = _HERE / "bags" / "wander_5min_pilot"
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta(bag)
    z1 = mmap_z16(bag / "rs1_z16.bin", meta, "rs1")
    z2p = bag / "rs2_z16.bin"
    z2 = mmap_z16(z2p, meta, "rs2") if z2p.is_file() and z2p.stat().st_size > 0 else None
    nfr = int(z1.shape[0])
    idxs = np.linspace(0, nfr - 1, num=6, dtype=np.int32)
    intr1 = (meta.get("intrinsics") or {}).get("rs1")
    intr2 = (meta.get("intrinsics") or {}).get("rs2")

    labs = []
    tiles = []
    for i in idxs:
        buf = np.zeros((60, 80), dtype=np.uint8)
        z2i = z2[i] if z2 is not None and i < z2.shape[0] else None
        label_from_z16(z1[i], z2i, intr1=intr1, intr2=intr2, labels_out=buf)
        labs.append(buf.copy())
        tiles.append(_caption(
            render_policy_bgr(labels_to_ego_float(buf), scale=4),
            "real RS  #%d" % int(i),
        ))

    sim = _sim_ego()
    sim_tile = _caption(sim, "sim k_obs  (watch)") if sim is not None else None
    hero = _row([
        sim_tile if sim_tile is not None else tiles[0],
        tiles[0],
    ], gap=16)
    strip = _stack([_row(tiles[:3]), _row(tiles[3:])])
    page = _stack([hero, strip], gap=16)
    cv2.imwrite(str(OUT / "rs_like_sim.png"), page)

    kr = np.mean([list(_rates(k).values()) for k in labs], axis=0)
    keys = ["unk", "self", "clear", "obs"]
    stats = {
        "bag": bag.name,
        "n_tiles": len(idxs),
        "kevin_z16_rates": dict(zip(keys, [float(x) for x in kr])),
        "wrote": str(OUT / "rs_like_sim.png"),
    }
    (OUT / "transfer_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    print("wrote", OUT / "rs_like_sim.png", flush=True)


if __name__ == "__main__":
    main()
