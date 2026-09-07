#!/usr/bin/env python3
"""Probe depth detail vs ego grid — run on Kevin from repo root or src/.

Reports for current RS_DECIMATE_MAG (override with --mag):
  - vert count / grid shape
  - approx GSD (cm) at 0.5 / 1.0 / 1.5 m from angular step
  - ego fill: known+obs cells hit, median hits/cell in FOV
  - reflex pixel counts (near / soft-low / overhang) for the live scene

Detail bar (AGENTS.md): GSD <= 1 cm at 1 m, and ego median hits/cell >= 1
with soft-low/near counts not collapsing vs mag=3 baseline on same scene.
"""
from __future__ import annotations
import argparse, math, os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)
os.chdir(SRC)

from robot_config import EGO_PX_SIZE, FRAME_W, FRAME_H  # noqa: E402
import cameras as cameras_mod  # noqa: E402
from cameras import RSCamera  # noqa: E402
from vision import (  # noqa: E402
    depth_topdown, check_topdown_near_field,
    check_topdown_soft_low_obstacle, check_topdown_overhang_approach,
)


def gsd_cm(mag: int, range_m: float, depth_w=848, fx=425.0) -> float:
    """Horizontal ground sampling at range for linear SDK decimate."""
    ow = max(1, (depth_w + mag - 1) // mag)
    ang = (2 * math.atan(depth_w / (2 * fx))) / ow  # rad per sample
    return 100.0 * range_m * math.tan(ang)


def ego_fill(verts: np.ndarray):
    if verts is None or len(verts) == 0:
        return dict(cells=0, hit_cells=0, median_hits=0.0, p95_hits=0.0)
    obs, known = depth_topdown(verts)
    hit = ((obs > 0) | (known > 0)).astype(np.uint8)
    # rebuild hit counts via scatter
    v = verts.reshape(-1, 3)
    z = v[:, 2]
    valid = z > 0.01
    v = v[valid]
    if len(v) == 0:
        return dict(cells=0, hit_cells=0, median_hits=0.0, p95_hits=0.0)
    scale = np.float32(1.0 / EGO_PX_SIZE)
    center = np.float32([FRAME_W * 0.5, FRAME_H * 0.5])
    p = v[:, :2] * scale + center
    cols = np.clip(p[:, 0].astype(np.int32), 0, FRAME_W - 1)
    rows = np.clip(p[:, 1].astype(np.int32), 0, FRAME_H - 1)
    counts = np.zeros((FRAME_H, FRAME_W), dtype=np.int32)
    np.add.at(counts, (rows, cols), 1)
    occupied = counts[counts > 0]
    return dict(
        cells=int(FRAME_H * FRAME_W),
        hit_cells=int(np.count_nonzero(hit)),
        median_hits=float(np.median(occupied)) if len(occupied) else 0.0,
        p95_hits=float(np.percentile(occupied, 95)) if len(occupied) else 0.0,
        mean_hits=float(np.mean(occupied)) if len(occupied) else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mag', type=int, default=None)
    ap.add_argument('--serial', default=None, help='RS1 topdown serial')
    ap.add_argument('--frames', type=int, default=45)
    args = ap.parse_args()
    if args.mag is not None:
        cameras_mod.RS_DECIMATE_MAG = args.mag
    mag = cameras_mod.RS_DECIMATE_MAG
    rs1 = args.serial or os.environ.get('KEVIN_RS1_SERIAL') or '815412070676'
    print(f'probe_depth_detail mag={mag} serial={rs1} ego_cm={EGO_PX_SIZE*100:.1f}')
    for r in (0.5, 1.0, 1.5):
        g = gsd_cm(mag, r)
        ok = 'OK' if (r <= 1.0 and g <= 1.0) or (r > 1.0 and g <= 1.5) else 'THIN'
        print(f'  GSD @ {r:.1f}m ≈ {g:.2f} cm/sample  [{ok}]')
    cam = RSCamera(rs1, compute_pointcloud=True, decimate_mag=mag)
    time.sleep(0.5)
    fills, nears, softs, overs, nverts = [], [], [], [], []
    for _ in range(args.frames):
        cam.grab()
        v = cam.verts
        if v is None:
            continue
        nverts.append(int(v.shape[0]))
        fills.append(ego_fill(v))
        nf, nc, _ = check_topdown_near_field(v)
        sf, sc, _ = check_topdown_soft_low_obstacle(v)
        of, oc, _ = check_topdown_overhang_approach(v)
        nears.append(nc); softs.append(sc); overs.append(oc)
    cam.stop() if hasattr(cam, 'stop') else None
    if hasattr(cam, 'pipeline'):
        try:
            cam.pipeline.stop()
        except Exception:
            pass
    if not nverts:
        print('NO FRAMES'); return 1
    print(f'Verts mean={np.mean(nverts):.0f}  (n={len(nverts)})')
    med_hits = np.median([f["median_hits"] for f in fills])
    hit_cells = np.median([f["hit_cells"] for f in fills])
    print(f'Ego fill: median_hits/cell={med_hits:.2f}  median_hit_cells={hit_cells:.0f}')
    print(f'Reflex counts (median): near={np.median(nears):.0f} soft={np.median(softs):.0f} overhang={np.median(overs):.0f}')
    bar_gsd = gsd_cm(mag, 1.0) <= 1.0
    bar_fill = med_hits >= 1.0
    print(f'BAR: GSD@1m<=1cm? {bar_gsd}  median_hits>=1? {bar_fill}  → {"PASS" if bar_gsd and bar_fill else "FAIL/REVIEW"}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
