#!/usr/bin/env python3
"""Fit RS2 down-pitch so bag floor is horizontal in side view.

The red cone wedge is floor classified as OBS: phys_h slopes with range
when the mount angle is wrong. Same plot as vision._render_side_view
(fwd vs height). Picks the down-pitch that zeros floor slope.
"""
from __future__ import annotations

import math
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
    _scaled_intr, decimate_z16, deproject_z16, label_from_z16, labels_to_ego_float,
    render_policy_bgr, FW_CAM_H, FW_COS_P, FW_SIN_P,
)

OUT = _HERE / "out" / "transfer"
CAM_H0 = float(FW_CAM_H)
NOMINAL_DOWN = abs(math.radians(25.6 - 90.0))  # 64.4°


def phys_fwd(y, z, down_rad, cam_h=CAM_H0):
    s = math.sin(down_rad)
    c = math.cos(down_rad)
    phys_h = cam_h - y * c - z * s
    fwd = -s * y + c * z
    return phys_h, fwd


def _side_view(y, z, down_rad, cam_h, title):
    W, H = 400, 240
    sv = np.full((H, W, 3), 20, dtype=np.uint8)
    max_d, h_lo, h_hi = 2.5, -0.20, 0.55

    def to_px(fwd, height):
        px = int(fwd / max_d * (W - 1))
        py = int((1.0 - (height - h_lo) / (h_hi - h_lo)) * (H - 1))
        return px, py

    _, y0 = to_px(0, 0.0)
    if 0 <= y0 < H:
        sv[y0, :] = (60, 60, 60)
    _, yt = to_px(0, 0.05)
    if 0 <= yt < H:
        sv[yt, :] = (0, 50, 80)

    phys_h, fwd = phys_fwd(y, z, down_rad, cam_h)
    m = (fwd > 0.15) & (fwd < 2.4) & (np.abs(z) > 0.05)
    n = int(m.sum())
    if n > 8000:
        rng = np.random.default_rng(0)
        idx = np.where(m)[0]
        pick = rng.choice(idx, 8000, replace=False)
        sel = np.zeros_like(m)
        sel[pick] = True
        m = sel
    ph, fd = phys_h[m], fwd[m]
    for i in range(len(ph)):
        px, py = to_px(float(fd[i]), float(ph[i]))
        if 0 <= px < W and 0 <= py < H:
            sv[py, px] = (0, 200, 100) if ph[i] < 0.05 else (60, 60, 200)
    sv[0, :] = sv[-1, :] = sv[:, 0] = sv[:, -1] = (80, 80, 80)
    bar = np.full((28, W, 3), 12, dtype=np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([bar, sv], axis=0)


def floor_slope(y, z, down_rad, cam_h=CAM_H0):
    phys_h, fwd = phys_fwd(y, z, down_rad, cam_h)
    m = (fwd > 0.25) & (fwd < 1.80) & (np.abs(z) > 0.1)
    if int(m.sum()) < 400:
        return 99.0, 99.0, 0
    fd, ph = fwd[m], phys_h[m]
    bins = np.linspace(0.25, 1.80, 9)
    xs, hs = [], []
    for i in range(len(bins) - 1):
        inb = (fd >= bins[i]) & (fd < bins[i + 1])
        if int(inb.sum()) < 30:
            continue
        xs.append(0.5 * (bins[i] + bins[i + 1]))
        hs.append(float(np.percentile(ph[inb], 20)))
    if len(xs) < 3:
        return 99.0, 99.0, 0
    slope, intercept = np.polyfit(np.array(xs), np.array(hs), 1)
    return float(slope), float(intercept), len(xs)


def collect_yz(bag: Path, n_frames: int = 24):
    meta = load_meta(bag)
    z2p = bag / "rs2_z16.bin"
    z2 = mmap_z16(z2p, meta, "rs2")
    intr2 = (meta.get("intrinsics") or {}).get("rs2")
    idxs = np.linspace(0, z2.shape[0] - 1, num=n_frames, dtype=np.int32)
    ys, zs, xs = [], [], []
    for i in idxs:
        fx, fy, ppx, ppy = _scaled_intr(intr2, 4)
        v = deproject_z16(decimate_z16(z2[i], 4), fx, fy, ppx, ppy)
        if len(v) < 50:
            continue
        keep = (np.abs(v[:, 0]) < 0.35) & (v[:, 2] > 0.25) & (v[:, 2] < 2.2)
        v = v[keep]
        if len(v) == 0:
            continue
        xs.append(v[:, 0])
        ys.append(v[:, 1])
        zs.append(v[:, 2])
    if not ys:
        raise SystemExit("no rs2 verts")
    return np.concatenate(ys), np.concatenate(zs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bag = _HERE / "bags" / "wander_5min_pilot"
    y, z = collect_yz(bag)
    print("n pts", len(y), "nominal down-deg", math.degrees(NOMINAL_DOWN),
          "sin/cos", FW_SIN_P, FW_COS_P, flush=True)

    best = None
    rows = []
    for deg in np.linspace(25.0, 80.0, 111):
        down = math.radians(float(deg))
        slope, icept, nb = floor_slope(y, z, down)
        score = abs(slope) + 0.15 * abs(icept)
        rows.append((score, deg, slope, icept, nb))
        if best is None or score < best[0]:
            best = (score, deg, slope, icept, nb)
    rows.sort()
    print("best down-pitch %.2f°  floor slope %.3f (m/m = %.1f°)  h0=%.1fcm" % (
        best[1], best[2], math.degrees(math.atan(best[2])), best[3] * 100), flush=True)
    print("top5", [(round(d, 2), round(s, 4), round(h * 100, 1)) for _, d, s, h, _ in rows[:5]], flush=True)

    nom_s, nom_h, _ = floor_slope(y, z, NOMINAL_DOWN)
    print("nominal 64.4° slope %.3f (%.1f°) h0=%.1fcm" % (
        nom_s, math.degrees(math.atan(nom_s)), nom_h * 100), flush=True)

    down_best = math.radians(best[1])
    # Shift cam_h so 20th-percentile floor sits at 0.
    cam_h = CAM_H0 - best[3]
    sv0 = _side_view(y, z, NOMINAL_DOWN, CAM_H0, "side  NOW  64.4 deg down  (red=phys_h>=5cm)")
    sv1 = _side_view(y, z, down_best, cam_h, "side  FIT  %.1f deg down  cam_h=%.2f" % (best[1], cam_h))
    cv2.imwrite(str(OUT / "side_now.png"), sv0)
    cv2.imwrite(str(OUT / "side_fit.png"), sv1)
    cv2.imwrite(str(OUT / "side_before_after.png"), np.concatenate([sv0, sv1], axis=1))

    # Ego tile at fitted pitch (monkeypatch module constants for one rebuild).
    import perception.fast_ego80 as fe
    fe.FW_SIN_P = math.sin(down_best)
    fe.FW_COS_P = math.cos(down_best)
    fe.FW_CAM_H = cam_h
    meta = load_meta(bag)
    z1 = mmap_z16(bag / "rs1_z16.bin", meta, "rs1")
    z2 = mmap_z16(bag / "rs2_z16.bin", meta, "rs2")
    lab = np.zeros((60, 80), np.uint8)
    fe.label_from_z16(
        z1[0], z2[0],
        intr1=(meta.get("intrinsics") or {}).get("rs1"),
        intr2=(meta.get("intrinsics") or {}).get("rs2"),
        labels_out=lab,
    )
    bgr = render_policy_bgr(labels_to_ego_float(lab), scale=4)
    bar = np.full((24, bgr.shape[1], 3), 12, np.uint8)
    cv2.putText(bar, "ego after pitch fit %.1f deg" % best[1], (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "ego_after_pitch.png"), np.concatenate([bar, bgr], 0))
    (OUT / "pitch_fit.json").write_text(
        __import__("json").dumps({
            "nominal_down_deg": math.degrees(NOMINAL_DOWN),
            "fit_down_deg": best[1],
            "fit_cam_h": cam_h,
            "nominal_slope": nom_s,
            "fit_slope": best[2],
            "fit_h0_before_cam_h_shift": best[3],
        }, indent=2)
    )
    print("wrote", OUT / "side_before_after.png", OUT / "ego_after_pitch.png", flush=True)


if __name__ == "__main__":
    main()
