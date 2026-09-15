#!/usr/bin/env python3
"""Bag vs sim ego look. Does not start train — wait for a human OK."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bag_io import load_meta, mmap_z16  # noqa: E402
from dash import compose_hud, save_contact, save_gif  # noqa: E402
from ego80 import (  # noqa: E402
    CLEAR, EGO80_H, EGO80_W, OBSTACLE, SELF, UNKNOWN,
    label_from_z16, paint_self_80, self_mask_80,
)
from env import RlNavVec  # noqa: E402
from kernels import LOOK_CLEAN, LOOK_KNOBS, N_V, N_W  # noqa: E402

PAL = np.array([
    [18, 18, 22],
    [40, 90, 220],
    [40, 170, 90],
    [210, 50, 40],
], dtype=np.uint8)
OUT = _HERE / "out" / "bag_probe"


def _colorize(lab: np.ndarray) -> np.ndarray:
    return PAL[np.clip(lab.astype(np.int32), 0, 3)]


def _scale(img: np.ndarray, s: int = 4) -> np.ndarray:
    return np.repeat(np.repeat(img, s, axis=0), s, axis=1)


def _ego_float_to_lab(ego: np.ndarray) -> np.ndarray:
    x = np.asarray(ego, dtype=np.float32)
    lab = np.full(x.shape, UNKNOWN, dtype=np.uint8)
    lab[(x >= 0.25) & (x < 0.75)] = CLEAR
    lab[x >= 0.75] = OBSTACLE
    paint_self_80(lab)
    return lab


def _bag_tiles(bag: Path, n: int = 8) -> list:
    meta = load_meta(bag)
    z1 = mmap_z16(bag / "rs1_z16.bin", meta, "rs1")
    z2p = bag / "rs2_z16.bin"
    z2 = mmap_z16(z2p, meta, "rs2") if z2p.is_file() and z2p.stat().st_size > 0 else None
    nfr = int(z1.shape[0])
    idxs = np.linspace(0, nfr - 1, num=n, dtype=np.int32)
    out = []
    buf = np.zeros((EGO80_H, EGO80_W), dtype=np.uint8)
    for i in idxs:
        z2i = z2[i] if z2 is not None and i < z2.shape[0] else None
        label_from_z16(
            z1[i], z2i,
            intr1=(meta.get("intrinsics") or {}).get("rs1"),
            intr2=(meta.get("intrinsics") or {}).get("rs2"),
            labels_out=buf,
        )
        out.append(buf.copy())
    return out


def _rates(ego: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(ego < 0.25)),
        float(np.mean((ego >= 0.25) & (ego < 0.75))),
        float(np.mean(ego >= 0.75)),
    )


def _lab_rates(lab: np.ndarray) -> tuple[float, float, float, float]:
    n = float(lab.size)
    return (
        float(np.count_nonzero(lab == UNKNOWN)) / n,
        float(np.count_nonzero(lab == CLEAR)) / n,
        float(np.count_nonzero(lab == OBSTACLE)) / n,
        float(np.count_nonzero(lab == SELF)) / n,
    )


def _caption_row(imgs: list, labels: list) -> np.ndarray:
    tiles = []
    for img, name in zip(imgs, labels):
        t = _scale(_colorize(img), 4)
        bar = np.full((22, t.shape[1], 3), 12, dtype=np.uint8)
        cv2.putText(bar, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([bar, t], axis=0))
    gap = np.full((tiles[0].shape[0], 6, 3), 8, dtype=np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        out = np.concatenate([out, gap, t], axis=1)
    return out


def _write_stack(rows: list, path: Path) -> None:
    gap = np.full((8, rows[0].shape[1], 3), 8, dtype=np.uint8)
    stacked = rows[0]
    for r in rows[1:]:
        if r.shape[1] != stacked.shape[1]:
            w = max(r.shape[1], stacked.shape[1])
            def _pad(im):
                if im.shape[1] >= w:
                    return im
                p = np.full((im.shape[0], w - im.shape[1], 3), 8, dtype=np.uint8)
                return np.concatenate([im, p], axis=1)
            stacked = _pad(stacked)
            r = _pad(r)
        stacked = np.concatenate([stacked, gap, r], axis=0)
    cv2.imwrite(str(path), stacked[:, :, ::-1])


def _fwd_prims(n: int, device) -> torch.Tensor:
    iv = torch.randint(4, N_V, (n,), device=device)
    iw = torch.randint(2, N_W - 2, (n,), device=device)
    return (iv * N_W + iw).to(dtype=torch.int32)


def _pick_envs(ego: np.ndarray, n: int = 8) -> list:
    obs = np.mean(ego >= 0.75, axis=(1, 2))
    # spread: some occupied, some open, not all the same dirt
    order = np.argsort(obs)
    idxs = np.linspace(0, len(order) - 1, num=n, dtype=np.int32)
    return [_ego_float_to_lab(ego[int(order[i])]) for i in idxs]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bag = _HERE / "bags" / "wander_5min_pilot"
    n = 32
    env = RlNavVec(n=n, seed=7, device="cuda:0")
    print("compiled — look knobs are runtime", flush=True)
    env.set_curriculum(3)
    env.reset_all()
    write = torch.zeros(n, 64, device=env.torch_dev)
    snaps = []
    for t in range(80):
        prim = _fwd_prims(n, env.torch_dev)
        env.step(prim, write)
        if t % 4 == 0:
            snaps.append(env.env0_cpu())

    hud_frames = [compose_hud(s, st=env.stats(), note="look roll") for s in snaps]
    save_contact(hud_frames, OUT / "sim_hud_contact.png")
    cv2.imwrite(str(OUT / "sim_hud_last.png"), hud_frames[-1])
    save_gif(hud_frames, OUT / "sim_hud.gif", duration_ms=120)

    bag_labs = _bag_tiles(bag, n=8)
    br = np.mean([_lab_rates(b) for b in bag_labs], axis=0)
    print("bag rebuild  unk=%.3f clr=%.3f obs=%.3f self=%.3f" % tuple(br), flush=True)
    sm = self_mask_80()
    wheel_obs = [int(np.count_nonzero((b == OBSTACLE) & sm)) for b in bag_labs]
    print("bag obs-in-self cells", wheel_obs, flush=True)

    env.set_look(LOOK_CLEAN)
    ego_clean = env.ego.detach().cpu().numpy()
    print("sim clean   unk=%.3f clr=%.3f obs=%.3f" % _rates(ego_clean), flush=True)

    env.set_look(LOOK_KNOBS)
    ego_blob = env.ego.detach().cpu().numpy()
    print("sim blobs   unk=%.3f clr=%.3f obs=%.3f" % _rates(ego_blob), flush=True)

    clean_labs = _pick_envs(ego_clean)
    blob_labs = _pick_envs(ego_blob)

    rows = [
        _caption_row(bag_labs, ["bag %d" % i for i in range(len(bag_labs))]),
        _caption_row(clean_labs, ["clean %d" % i for i in range(len(clean_labs))]),
        _caption_row(blob_labs, ["blob %d" % i for i in range(len(blob_labs))]),
    ]
    _write_stack(rows, OUT / "look_bag_clean_dirt.png")

    pairs = []
    for i, (b, c, d) in enumerate(zip(bag_labs, clean_labs, blob_labs)):
        pairs.append(_caption_row([b, c, d], ["bag", "sim clean", "sim blob"]))
    _write_stack(pairs, OUT / "look_triples.png")

    time_labs = []
    time_names = []
    for k in range(8):
        prim = _fwd_prims(n, env.torch_dev)
        env.step(prim, write)
        time_labs.append(_ego_float_to_lab(env.ego[0].detach().cpu().numpy()))
        time_names.append("t+%d" % k)
    _write_stack(
        [_caption_row(time_labs, time_names)],
        OUT / "look_time_strip.png",
    )

    cy, cx = 29, 20
    def _crop(lab):
        return lab[max(0, cy - 10):cy + 10, max(0, cx - 10):cx + 10]
    self_row = _caption_row(
        [_crop(bag_labs[0]), _crop(clean_labs[0]), _crop(blob_labs[0])],
        ["bag self", "sim clean self", "sim blob self"],
    )
    cv2.imwrite(str(OUT / "look_self_crop.png"), self_row[:, :, ::-1])

    items = [("bag", lab) for lab in bag_labs] + [("sim", lab) for lab in blob_labs]
    rng = np.random.default_rng(11)
    order = rng.permutation(len(items))
    shuffled = [items[int(i)] for i in order]
    tiles = [_scale(_colorize(lab), 4) for _, lab in shuffled]
    grid = np.concatenate(
        [np.concatenate(tiles[r * 4:(r + 1) * 4], axis=1) for r in range(4)],
        axis=0,
    )
    cv2.imwrite(str(OUT / "blind_grid.png"), grid[:, :, ::-1])
    key = [{"i": int(k), "src": shuffled[k][0]} for k in range(len(shuffled))]
    (OUT / "blind_grid_key.json").write_text(json.dumps(key, indent=2))
    print("wrote", OUT / "look_bag_clean_dirt.png", flush=True)
    print("wrote", OUT / "look_triples.png", flush=True)
    print("wrote", OUT / "look_time_strip.png", flush=True)
    print("wrote", OUT / "look_self_crop.png", flush=True)


if __name__ == "__main__":
    main()
