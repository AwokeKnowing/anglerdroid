#!/usr/bin/env python3
"""KEVIN_EGO_DEPTHCULL equivalence: the depth-space cull Ego must give bit-identical outputs to the transform Ego
(lab, hgt, lab_map, cliff/too-close pixel counts, obstacle_xy) on synthetic RS1 scenes: tilted floor (+-2 deg),
small posts just above / below the cutoff the code applies (c2 - clip), a drop (cliff path) and a lens-covering blob
(too-close path, which also triggers the unlevelled map pass).
Usage: KEVIN_HAL=sim python tests/test_ego_depthcull.py   (or pytest)"""
import os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "src")):
    if p not in sys.path: sys.path.insert(0, p)
import nav.perceive as NP
import perception.ego_rs1 as E

I = dict(fx=419.86, fy=419.86, ppx=419.42, ppy=232.67, width=848.0, height=480.0)
H, W = 160, 284


def _rays():
    f = I["width"] / W
    u, v = np.meshgrid((np.arange(W) + 0.5) * f, (np.arange(H) + 0.5) * f)
    return ((u - I["ppx"]) / I["fx"]).astype(np.float32), ((v - I["ppy"]) / I["fy"]).astype(np.float32)


def scene(pitch, roll, floor=0.9545, posts=(), hole=None, cover=None, seed=0):
    """Z grid of a plane tilted by (pitch, roll) deg + flat-topped posts (cx, cy, w, h above floor) + a drop + a covered patch."""
    rx, ry = _rays(); c0, c1 = np.tan(np.radians(roll)), np.tan(np.radians(pitch))
    Z = floor / (1 - c0 * rx - c1 * ry)                                 # plane z = c0 x + c1 y + floor
    Z = Z + np.random.default_rng(seed).normal(0, 0.002, Z.shape)       # stereo-like noise
    for (cx, cy, w, h) in posts:
        zt = (floor - h) / (1 - c0 * rx - c1 * ry); m = (np.abs(rx * zt - cx) <= w / 2) & (np.abs(ry * zt - cy) <= w / 2)
        Z[m] = zt[m]
    if hole is not None:
        r0, r1, q0, q1 = hole; Z[r0:r1, q0:q1] += 0.12                   # 12 cm drop
    if cover is not None:
        r0, r1, q0, q1 = cover; Z[r0:r1, q0:q1] = 0.0                    # too close / invalid
    return np.stack([rx * Z, ry * Z, Z], -1).reshape(-1, 3).astype(np.float32)


def _pair():
    a, b = NP.Ego(), NP.Ego()
    for e in (a, b): e.intr = I
    a.DEPTHCULL, b.DEPTHCULL = False, True
    return a, b


def _same(a, b):
    assert np.array_equal(a.lab, b.lab), int((a.lab != b.lab).sum())
    assert np.array_equal(a.hgt, b.hgt), int((a.hgt != b.hgt).sum())
    assert (a.lab_map is None) == (b.lab_map is None)
    if a.lab_map is not None: assert np.array_equal(a.lab_map, b.lab_map)
    assert a.cliff_px == b.cliff_px and a.too_close_px == b.too_close_px
    assert np.array_equal(a.obstacle_xy(max_n=1500), b.obstacle_xy(max_n=1500))


def test_small_posts_around_cutoff():
    hc = 0.9545 - float(E.TD_FLOOR_CLIP)                               # cutoff the code applies for this floor
    posts = [(cx, cy, w, hc + d) for (cx, cy) in ((0.3, 0.1), (-0.5, -0.3), (0.7, 0.35)) for w in (0.01, 0.02, 0.03)
             for d in (-0.001, 0.0005)]
    posts = [(cx + 0.04 * i, cy, w, h) for i, (cx, cy, w, h) in enumerate(posts)]
    for k, (p, r) in enumerate(((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-1.3, 1.7))):
        a, b = _pair()
        v = scene(p, r, posts=posts, seed=k); a.update(v); b.update(v); _same(a, b)


def test_cliff_and_too_close_paths():
    a, b = _pair(); seen = set()
    for k in range(6):                                                  # persistence: CLIFF_FRAMES / TC_FRAMES
        v = scene(1.5, -1.0, posts=[(0.2, 0.0, 0.02, 0.05)], hole=(100, 140, 180, 240), cover=(10, 40, 40, 80), seed=k)
        a.update(v); b.update(v); _same(a, b)
        if a.cliff_px: seen.add("cliff")
        if a.too_close_px: seen.add("tc")
        if a.lab_map is not None: seen.add("map")
    assert seen == {"cliff", "tc", "map"}, seen


if __name__ == "__main__":
    test_small_posts_around_cutoff(); test_cliff_and_too_close_paths(); print("test_ego_depthcull: OK")
