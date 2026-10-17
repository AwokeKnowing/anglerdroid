"""test_insect_planner.py — 5 Hz reachable (v,w) on a fake ego heightmap.

Run: python3 test_insect_planner.py
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from insect_planner import InsectPlanner, FOOT_R_PX  # noqa: E402
from robot_config import RCX, RCY, FRAME_H, FRAME_W  # noqa: E402


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
    assert cmd["fwd_mps"] > 0.015, cmd
    assert abs(cmd["ang_rads"]) < 0.35, "open floor should not spin, got %s" % cmd
    assert p._goal is not None, "open floor should pick a ~2 m goal"
    gx, gy = p._goal
    assert abs(gx - 2.0) < 0.35 and abs(gy) < 0.35, p._goal
    print("open floor v=%.3f w=%.3f goal=(%.2f,%.2f)" % (
        cmd["fwd_mps"], cmd["ang_rads"], gx, gy))


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


def test_close_wall_yaws_without_unstuck():
    p = InsectPlanner()
    assert float(p.hp.get("unstuck", 0.0)) < 0.5
    p.set_wander_mode(True)
    _force(p)
    obs = _blank()
    col = RCX + 16
    obs["ego_height"][:, col:col + 6] = 40
    obs["ego_persistent"][:, col:col + 6] = 200
    cmd = p.tick(obs, (0.0, 0.0, 0.0), 0.033)
    assert cmd is not None
    assert not p._in_unstuck
    assert cmd["fwd_mps"] <= 0.05, cmd
    assert abs(cmd["ang_rads"]) > 0.15, cmd
    print("heuristic pin v=%.3f w=%.3f plus_x=%.2f" % (
        cmd["fwd_mps"], cmd["ang_rads"], p._plus_x))


def test_close_wall_enters_unstuck():
    p = InsectPlanner(hp={"unstuck": 1.0})
    p.set_wander_mode(True)
    _force(p)
    obs = _blank()
    col = RCX + 16
    obs["ego_height"][:, col:col + 6] = 40
    obs["ego_persistent"][:, col:col + 6] = 200
    cmd = p.tick(obs, (0.0, 0.0, 0.0), 0.033)
    assert cmd is not None
    assert p._in_unstuck, "plus-x pin should hand off to unstuck"
    assert cmd["fwd_mps"] <= 0.05, cmd
    assert abs(cmd["ang_rads"]) >= 0.20, cmd
    print("unstuck pin v=%.3f w=%.3f src=%s" % (
        cmd["fwd_mps"], cmd["ang_rads"], p._debug.get("unstuck_src")))


def test_unstuck_hysteresis_then_heuristic():
    p = InsectPlanner(hp={"unstuck": 1.0})
    p.set_wander_mode(True)
    wall = _blank()
    col = RCX + 16
    wall["ego_height"][:, col:col + 6] = 40
    wall["ego_persistent"][:, col:col + 6] = 200
    p.tick(wall, (0.0, 0.0, 0.0), 0.033)
    assert p._in_unstuck
    for _ in range(20):
        p.tick(_blank(), (0.0, 0.0, 0.0), 0.033)
    assert not p._in_unstuck, "clear plus-x should return to 5 Hz heuristic"
    p.visit[:] = 0
    p._cmd_v = p._cmd_w = 0.0
    _force(p)
    cmd = p.tick(_blank(), (0.0, 0.0, 0.0), 0.033)
    assert cmd["fwd_mps"] > 0.015, cmd
    print("after clear v=%.3f unstuck=%s" % (cmd["fwd_mps"], p._in_unstuck))


def test_goal_pulls_left_on_open_floor():
    p = InsectPlanner()
    p.set_wander_mode(True)
    p.set_goal(0.0, 3.0)
    assert p._wander, "goal must keep wander doughnut"
    _force(p)
    cmd = p.tick(_blank(), (0.0, 0.0, 0.0), 0.033)
    assert cmd is not None
    assert cmd["ang_rads"] > 0.08, "goal at +y should yaw left, got %s" % cmd
    print("goal left v=%.3f w=%.3f" % (cmd["fwd_mps"], cmd["ang_rads"]))


def test_goal_arrival_clears():
    p = InsectPlanner()
    p.set_goal(0.10, 0.05)
    cmd = p.tick(_blank(), (0.0, 0.0, 0.0), 0.033)
    assert p._goal is None
    assert cmd is None or cmd["fwd_mps"] == 0.0
    assert not p.is_active()
    print("arrived, active=%s" % p.is_active())


def test_inactive_none():
    p = InsectPlanner()
    assert p.tick(_blank(), (0.0, 0.0, 0.0), 0.033) is None


def test_side_blob_does_not_thread():
    """A yellow blob off the centerline but inside chassis width must not be a free charge."""
    p = InsectPlanner()
    p.set_wander_mode(True)
    _force(p)
    obs = _blank()
    obs["ego_persistent"][RCY - 16:RCY - 8, RCX + 36:RCX + 56] = 200
    obs["ego_height"][RCY - 16:RCY - 8, RCX + 36:RCX + 56] = 40
    cmd = p.tick(obs, (0.0, 0.0, 0.0), 0.033)
    assert cmd is not None
    charging = cmd["fwd_mps"] > 0.22 and abs(cmd["ang_rads"]) < 0.10
    assert not charging, "threaded a side blob, got %s plus_x=%.2f" % (cmd, p._plus_x)
    print("side blob v=%.3f w=%.3f plus_x=%.2f" % (
        cmd["fwd_mps"], cmd["ang_rads"], p._plus_x))


def test_blocked_forward_seeks_no_goal():
    p = InsectPlanner()
    p.set_wander_mode(True)
    _force(p)
    obs = _blank()
    col = RCX + 20
    obs["ego_persistent"][:, col:col + 8] = 200
    cmd = p.tick(obs, (0.0, 0.0, 0.0), 0.033)
    assert p._goal is None, p._goal
    assert p._seeking
    assert cmd["fwd_mps"] <= 0.02, cmd
    assert abs(cmd["ang_rads"]) >= 0.20, cmd
    print("seek v=%.3f w=%.3f plus_x=%.2f" % (
        cmd["fwd_mps"], cmd["ang_rads"], p._plus_x))


def test_pin_does_not_command_forward():
    p = InsectPlanner()
    p.set_wander_mode(True)
    _force(p)
    p._cmd_v, p._cmd_w = 0.28, 0.0
    obs = _blank()
    col = RCX + 16
    obs["ego_persistent"][:, col:col + 6] = 200
    cmd = p.tick(obs, (0.0, 0.0, 0.0), 0.033)
    assert cmd["fwd_mps"] <= 0.0, "must not push +v into plus-x pin, got %s" % cmd
    print("pin clamp v=%.3f plus_x=%.2f" % (cmd["fwd_mps"], p._plus_x))


def test_no_trail_reverses_straight():
    """Start-stuck / slow crawl: no useful IMU path → body reverse, w≈0."""
    p = InsectPlanner()
    p.set_wander_mode(True)
    _force(p)
    cmd = p.tick(_blank(), (0.0, 0.0, 0.0), 0.033, safety_fwd=0.20)
    assert p._recover_mode == "reverse", p._recover_mode
    assert p._reverse_src == "straight", p._reverse_src
    assert p._reverse_xy is None
    assert cmd["fwd_mps"] < 0.0, cmd
    assert abs(cmd["ang_rads"]) < 0.08, cmd
    print("straight reverse v=%.3f w=%.3f" % (cmd["fwd_mps"], cmd["ang_rads"]))


def test_last_clear_used_when_trail_is_junk():
    p = InsectPlanner()
    p.set_wander_mode(True)
    p._last_clear_xy = (0.0, 0.0)
    p._clear_trail = [(0.0, 0.0)]
    p._trail_world = [(0.88, 0.0), (0.89, 0.0), (0.90, 0.0)]  # crawl, < 18 cm
    _force(p)
    cmd = p.tick(_blank(), (0.90, 0.0, 0.0), 0.033, safety_fwd=0.20)
    assert p._reverse_src == "clear", p._reverse_src
    assert cmd["fwd_mps"] < 0.0, cmd
    assert p._reverse_xy is not None
    print("clear-history reverse v=%.3f tgt=%s" % (cmd["fwd_mps"], p._reverse_xy))
    p = InsectPlanner()
    p.set_wander_mode(True)
    p._trail_world = [(i * 0.05, 0.0) for i in range(20)]
    _force(p)
    cmd = p.tick(_blank(), (0.90, 0.0, 0.0), 0.033, safety_fwd=0.20)
    assert p._recover_mode == "reverse", p._recover_mode
    assert p._reverse_src == "trail", p._reverse_src
    assert cmd["fwd_mps"] < 0.0, cmd
    print("trail reverse v=%.3f tgt=%s" % (cmd["fwd_mps"], p._reverse_xy))


def test_pin_recovers_reverse_then_yaw():
    p = InsectPlanner()
    p.set_wander_mode(True)
    p._trail_world = [(i * 0.05, 0.0) for i in range(20)]
    wall = _blank()
    col = RCX + 16
    wall["ego_persistent"][:, col:col + 6] = 200
    pose = (0.90, 0.0, 0.0)
    cmd = None
    for _ in range(5):
        cmd = p.tick(wall, pose, 0.08)
    assert p._recover_mode == "reverse", p._recover_mode
    assert cmd["fwd_mps"] < 0.0, cmd
    # Throttle lifts and map clears → keep reversing a bit, then yaw, then done.
    for _ in range(6):
        cmd = p.tick(_blank(), pose, 0.08, safety_fwd=1.0)
    assert p._recover_mode in ("reverse", "yaw", ""), p._recover_mode
    for _ in range(16):
        cmd = p.tick(_blank(), pose, 0.08, safety_fwd=1.0)
    assert not p._recover, "should resume after reverse-past + yaw-clear"
    assert cmd["fwd_mps"] >= 0.0, cmd
    print("after recover v=%.3f mode=%r" % (cmd["fwd_mps"], p._recover_mode))


def test_keep_clear_fills_sub_turn_gaps():
    """Wander inflate is a 25 cm disk: hallways under ~50 cm are not a path."""
    assert FOOT_R_PX == 25
    p = InsectPlanner()
    occ = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    occ[:RCY - 20, :] = 255
    occ[RCY + 20:, :] = 255
    inf = p._inflate_occ(occ)
    assert inf[RCY, RCX + 40] > 0, "40 cm hallway should be impassable"
    occ2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    occ2[:RCY - 30, :] = 255
    occ2[RCY + 30:, :] = 255
    inf2 = p._inflate_occ(occ2)
    assert inf2[RCY, RCX + 40] == 0, "60 cm hallway should stay open"
    print("keep-clear 40cm hallway blocked / 60cm open")


if __name__ == "__main__":
    test_inactive_none()
    test_open_floor_goes_forward()
    test_wall_ahead_does_not_charge()
    test_hot_doughnut_prefers_peel()
    test_hold_is_cheap()
    test_close_wall_yaws_without_unstuck()
    test_close_wall_enters_unstuck()
    test_unstuck_hysteresis_then_heuristic()
    test_goal_pulls_left_on_open_floor()
    test_goal_arrival_clears()
    test_side_blob_does_not_thread()
    test_blocked_forward_seeks_no_goal()
    test_pin_does_not_command_forward()
    test_no_trail_reverses_straight()
    test_last_clear_used_when_trail_is_junk()
    test_pin_recovers_reverse_then_yaw()
    test_keep_clear_fills_sub_turn_gaps()
    print("ok")
