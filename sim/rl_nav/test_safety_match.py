#!/usr/bin/env python3
"""Numpy stand-ins for the live-vs-sim throttle / hull bugs.

Live SafetyGuard scans FOOT-width strips and uses one ang_scale =
f(min(left, right)). Sim used to fire four centre rays and scale +w
from the left gap only / −w from the right — so a rectangle next to a
wall could spin the far corner in. Contact was eight sparse probes and
was not counted (crashed stayed 0).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("kernels.py").read_text()


def _f(name: str) -> float:
    m = re.search(rf"^{name}\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+))", _SRC, re.M)
    if m is None:
        raise AssertionError("missing " + name)
    return float(m.group(1))


class K:
    LAT_HARD_M = _f("LAT_HARD_M")
    LAT_SAFE_M = _f("LAT_SAFE_M")
    STOP_M = _f("STOP_M")
    THROT_M = _f("THROT_M")
    SCAN_FWD_LX = _f("SCAN_FWD_LX")
    SCAN_BWD_LX = _f("SCAN_BWD_LX")
    SCAN_SIDE_LY = _f("SCAN_SIDE_LY")
    SCAN_FACE_HALF = _f("SCAN_FACE_HALF")
    N_STRIP = int(_f("N_STRIP"))
    HULL_X0 = _f("HULL_X0")
    HULL_X1 = _f("HULL_X1")
    HULL_Y0 = _f("HULL_Y0")
    HULL_Y1 = _f("HULL_Y1")
    TRACK_M = _f("TRACK_M")


def _lat(d):
    if d <= K.LAT_HARD_M:
        return 0.0
    if d >= K.LAT_SAFE_M:
        return 1.0
    return (d - K.LAT_HARD_M) / (K.LAT_SAFE_M - K.LAT_HARD_M)


def _gap(d, stop=K.STOP_M, throt=K.THROT_M):
    if d <= stop:
        return 0.0
    if d >= throt:
        return 1.0
    t = (d - stop) / (throt - stop)
    return t * t * (3.0 - 2.0 * t)


def test_live_one_ang_scale_kills_both_yaw_signs():
    # Wall 3 cm off the right side. Left is open.
    sl, sr = _lat(1.0), _lat(0.03)
    old_left_turn = sl          # sim used to allow this
    old_right_turn = sr
    live = min(sl, sr)
    assert old_left_turn > 0.95
    assert old_right_turn < 0.20
    assert live < 0.20
    # Live (and new sim) apply live to +w and −w.


def test_strip_sees_front_corner_ray_misses():
    # Obstacle sitting on the front-left corner of the FOOT, 4 cm ahead of
    # the bumper, 20 cm left of centre. Centre ray goes +X from the nose
    # and never hits it. Strip min across y ∈ ±SCAN_FACE_HALF does.
    obs_x, obs_y = 0.22, 0.20
    ray_y = 0.0
    strip_ys = np.linspace(-K.SCAN_FACE_HALF, K.SCAN_FACE_HALF, K.N_STRIP)
    ray_hit = abs(obs_y - ray_y) < 0.02
    strip_hit = np.any(np.abs(strip_ys - obs_y) < 0.04)
    assert not ray_hit
    assert strip_hit


def test_old_foot_probes_miss_rear_corner():
    # Old 8-point skeleton had no rear corners. A wall at the rear-left
    # corner of the physical hull was invisible to contact.
    old = {
        (0.15, 0.0), (0.15, 0.16), (0.15, -0.16),
        (0.0, 0.16), (0.0, -0.16), (0.0, 0.0),
        (-0.18, 0.0), (-0.26, 0.0),
    }
    corner = (K.HULL_X0, K.HULL_Y1)  # rear-left steel
    nearest = min(math.hypot(px - corner[0], py - corner[1]) for px, py in old)
    assert nearest > 0.08
    assert K.HULL_X0 <= corner[0] <= K.HULL_X1
    assert K.HULL_Y0 <= corner[1] <= K.HULL_Y1


def test_yaw_sweeps_corner_into_side_wall():
    # Robot parallel to a wall 1 cm outside the right hull. In-place +30°
    # yaw swings the front-right corner into the wall. Live has to reverse
    # first; old sim allowed the yaw (no rear/front-right contact sample
    # at 19 cm, and +w ignored the right gap).
    wall_y = K.HULL_Y0 - 0.01  # just outside right side at yaw=0
    yaw = math.radians(30.0)
    c, s = math.cos(yaw), math.sin(yaw)
    # +w (CCW) drives the REAR-right corner into a right-hand wall.
    fx, fy = K.HULL_X0, K.HULL_Y0
    wy = s * fx + c * fy
    assert wy < wall_y  # crossed the wall


def test_crashed_flag_is_dead_contact_is_the_metric():
    # Document the contract: episode cut==2 is unused; n_contact counts
    # overlapping policy steps.
    src = _SRC
    assert "crashed = int(0)" in src
    assert "n_contact[e] = n_contact[e] + 1" in src
    assert "cut[e] = 2" in src  # still there, never reached
    assert "CONTACT_PEN" in src


def test_scan_faces_match_live_foot():
    assert abs(K.SCAN_FWD_LX - 0.18) < 1e-9
    assert abs(K.SCAN_BWD_LX + 0.20) < 1e-9
    assert abs(K.SCAN_SIDE_LY - 0.25) < 1e-9
    assert abs(K.SCAN_FACE_HALF - 0.25) < 1e-9


def test_hull_matches_tape():
    assert abs(K.HULL_X1 - 0.18) < 1e-9
    assert abs(K.HULL_X0 + 0.20) < 1e-9
    assert abs(K.HULL_Y1 - 0.227) < 1e-9
    assert abs(K.TRACK_M - 0.354) < 1e-9
    src = Path(__file__).with_name("kernels.py").read_text()
    assert "WANDER_CLEAR_R = 0.25" in src
    assert "BODY_R = 0.20" in src


if __name__ == "__main__":
    for fn in (
        test_live_one_ang_scale_kills_both_yaw_signs,
        test_strip_sees_front_corner_ray_misses,
        test_old_foot_probes_miss_rear_corner,
        test_yaw_sweeps_corner_into_side_wall,
        test_crashed_flag_is_dead_contact_is_the_metric,
        test_scan_faces_match_live_foot,
        test_hull_matches_tape,
    ):
        fn()
        print("ok", fn.__name__)
    print("test_safety_match ok")
