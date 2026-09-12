"""test_insect_planner.py — 5 Hz reachable (v,w) on a fake ego heightmap.

Run: python3 test_insect_planner.py
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from insect_planner import InsectPlanner  # noqa: E402
from robot_config import RCX, FRAME_H, FRAME_W  # noqa: E402


def _blank():
    h = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    return {"ego_persistent": h.copy(), "ego_height": h}


def _force(p):
    p._last_replan = 0.0
    p._prev = None


def test_open_floor_goes_forward():
    p = InsectPlanner()
    p.set_wander_mode(True)
    _force(p)
    cmd = p.tick(_blank(), (0.0, 0.0, 0.0), 0.033)
    assert cmd is not None
    assert cmd["fwd_mps"] > 0.05, cmd
    assert abs(cmd["ang_rads"]) < 0.35, "open floor should not spin, got %s" % cmd
    print("open floor v=%.3f w=%.3f" % (cmd["fwd_mps"], cmd["ang_rads"]))


def test_wall_ahead_does_not_charge():
    p = InsectPlanner()
    p.set_wander_mode(True)
    _force(p)
    obs = _blank()
    col = RCX + 32
    obs["ego_height"][:, col:col + 6] = 40
    obs["ego_persistent"][:, col:col + 6] = 200
    cmd = p.tick(obs, (0.0, 0.0, 0.0), 0.033)
    assert cmd is not None
    if cmd["fwd_mps"] > 0.20:
        assert abs(cmd["ang_rads"]) > 0.15, cmd
    print("wall ahead v=%.3f w=%.3f throt=%.2f" % (
        cmd["fwd_mps"], cmd["ang_rads"], p._debug.get("throt", -1)))


def test_hot_doughnut_prefers_peel():
    p = InsectPlanner()
    p.set_wander_mode(True)
    p.visit[:] = 0
    r_m = 0.80
    for i in range(32):
        for j in range(32):
            dx = (j - 16) * 0.10
            dy = (16 - i) * 0.10
            rr = np.hypot(dx, dy - 0.80)
            if abs(rr - r_m) < 0.18:
                p.visit[i, j] = 18.0
    p.visit[16, 16] = 20.0
    p._cmd_v, p._cmd_w = 0.28, 0.35
    _force(p)
    p._prev = (0.28, 0.35)
    cmd = p.tick(_blank(), (0.0, 0.0, 0.0), 0.033)
    assert cmd is not None
    print("doughnut peel v=%.3f w=%.3f hot=%.2f" % (
        cmd["fwd_mps"], cmd["ang_rads"], p._debug.get("hot", -1)))
    assert cmd["ang_rads"] < 0.20, "should peel off the CCW doughnut, got %s" % cmd


def test_hold_is_cheap():
    p = InsectPlanner()
    p.set_wander_mode(True)
    _force(p)
    p.tick(_blank(), (0.0, 0.0, 0.0), 0.033)
    t0 = time.perf_counter()
    cmd = p.tick(_blank(), (0.02, 0.0, 0.0), 0.033)
    ms = (time.perf_counter() - t0) * 1e3
    assert cmd is not None
    assert ms < 5.0, "hold tick should skip scoring, took %.1f ms" % ms
    print("hold tick %.2f ms" % ms)


def test_inactive_none():
    p = InsectPlanner()
    assert p.tick(_blank(), (0.0, 0.0, 0.0), 0.033) is None


if __name__ == "__main__":
    test_inactive_none()
    test_open_floor_goes_forward()
    test_wall_ahead_does_not_charge()
    test_hot_doughnut_prefers_peel()
    test_hold_is_cheap()
    print("ok")
