"""local_executive.py – Mid-layer continuous local motion (no ROS).

Async goal mailbox for high-level agents (Kevins Doctor / tools):
  - set_goal_xy(x, y)  world meters (map frame, same as PoseEstimator)
  - set_wander()       keep rolling ~1 m free-space goals
  - clear()

Each main-loop tick (~30 Hz): drive via planner backend:
  - 'vfh'  (default): rolling subgoal → VFH on atlas quadrant
  - 'mppi': MppiCostmapPlanner on ego-space vis._persistent_obs

Never blocks. Capture/map thread stays untouched.
"""

from __future__ import annotations

import math
import threading
import time

import navigator

LOOKAHEAD_M = 1.0
GOAL_REACHED_M = 0.35
WANDER_REFRESH_S = 0.4

_lock = threading.Lock()
_mode = "idle"  # idle | xy | wander
_goal_xy = None  # (x, y) world m
_active = False
_last_wander_t = 0.0
_dbg = {}
_planner = "vfh"  # 'vfh' | 'mppi'
_mppi = None  # lazy MppiCostmapPlanner


def _ensure_mppi():
    global _mppi
    if _mppi is None:
        from mppi_costmap import MppiCostmapPlanner
        _mppi = MppiCostmapPlanner()
    return _mppi


def set_planner(name: str) -> None:
    """Select backend: 'vfh' (default) or 'mppi'."""
    global _planner
    name = (name or "vfh").strip().lower()
    if name not in ("vfh", "mppi"):
        raise ValueError("planner must be 'vfh' or 'mppi', got %r" % name)
    with _lock:
        _planner = name
    if name == "mppi":
        _ensure_mppi()
    print("local_executive: planner backend = %s" % name)


def get_planner() -> str:
    with _lock:
        return _planner


def set_goal_xy(x, y):
    """High-level: drive toward world-frame point (meters). Non-blocking."""
    global _mode, _goal_xy, _active
    with _lock:
        _goal_xy = (float(x), float(y))
        _mode = "xy"
        _active = True
        planner = _planner
    if planner == "mppi":
        _ensure_mppi().set_goal(float(x), float(y))


def set_wander():
    """High-level: continuously pick ~1 m free headings. Non-blocking."""
    global _mode, _goal_xy, _active, _last_wander_t
    with _lock:
        _mode = "wander"
        _goal_xy = None
        _active = True
        _last_wander_t = 0.0
        planner = _planner
    if planner == "mppi":
        _ensure_mppi().set_wander_mode(True)


def clear():
    global _mode, _goal_xy, _active
    with _lock:
        _mode = "idle"
        _goal_xy = None
        _active = False
        planner = _planner
    navigator.clear_goal()
    if planner == "mppi" and _mppi is not None:
        _mppi.cancel()


def is_active():
    with _lock:
        return _active


def status():
    with _lock:
        out = {
            "active": _active,
            "mode": _mode,
            "goal_xy": _goal_xy,
            "planner": _planner,
            "dbg": dict(_dbg),
        }
        planner = _planner
    if planner == "mppi" and _mppi is not None:
        out["mppi"] = _mppi.get_debug_state()
    return out


def _world_to_robot_heading_deg(gx, gy, rx, ry, rtheta):
    """Heading in navigator frame: 0=forward, +CCW (left positive)."""
    dx = gx - rx
    dy = gy - ry
    desired = math.atan2(dy, dx)
    err = (desired - rtheta + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(err)


def _rolling_point_toward(gx, gy, rx, ry, lookahead_m):
    dx, dy = gx - rx, gy - ry
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return gx, gy, dist
    if dist <= lookahead_m:
        return gx, gy, dist
    s = lookahead_m / dist
    return rx + dx * s, ry + dy * s, dist


def _tick_mppi(obs_map, pose_x, pose_y, pose_theta):
    """MPPI backend: uses ego-space obs_map (vis._persistent_obs)."""
    global _dbg, _active, _mode

    mppi = _ensure_mppi()
    if pose_x is None or pose_y is None:
        return None
    if obs_map is None:
        # Empty free map keeps planner ticking (open-space geometric bias)
        import numpy as np
        from robot_config import FRAME_H, FRAME_W
        obs_map = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    pose = (float(pose_x), float(pose_y), float(pose_theta or 0.0))
    cmd = mppi.tick(obs_map, pose, 0.033)
    with _lock:
        if not mppi.is_active():
            _active = False
            _mode = "idle"
            _dbg = {"event": "mppi_inactive"}
        else:
            _dbg = {
                "mode": _mode,
                "planner": "mppi",
                "cmd": cmd,
                "mppi_ms": mppi.get_debug_state().get("tick_ms"),
            }
    if cmd is None:
        return None
    return (float(cmd["fwd_mps"]), float(cmd["ang_rads"]))


def _tick_vfh(atlas, pose_x, pose_y, pose_theta, mode, goal):
    """VFH backend: rolling subgoal → navigator on atlas quadrant."""
    global _last_wander_t, _dbg, _active, _mode

    if mode == "xy":
        if pose_x is None or goal is None:
            return None
        sx, sy, dist = _rolling_point_toward(
            goal[0], goal[1], pose_x, pose_y, LOOKAHEAD_M)
        if dist < GOAL_REACHED_M:
            clear()
            with _lock:
                _dbg = {"event": "reached", "dist": dist, "planner": "vfh"}
            return (0.0, 0.0)
        hdg = _world_to_robot_heading_deg(
            sx, sy, pose_x, pose_y, pose_theta or 0.0)
        navigator.set_goal(hdg)
        with _lock:
            _dbg = {"mode": "xy", "dist": dist, "hdg": hdg, "sub": (sx, sy),
                    "planner": "vfh"}
        return navigator.compute_twist(atlas) if atlas is not None else None

    if mode == "wander":
        now = time.monotonic()
        if now - _last_wander_t >= WANDER_REFRESH_S:
            navigator.set_goal(0.0)
            _last_wander_t = now
        with _lock:
            _dbg = {"mode": "wander", "hdg": 0.0, "planner": "vfh"}
        if atlas is None:
            return None
        return navigator.compute_twist(atlas)

    return None


def tick(atlas, pose_x=None, pose_y=None, pose_theta=None, obs_map=None):
    """Call from 30 Hz main loop. Returns (fwd, ang) or None if inactive.

    atlas: full 640x480 vision atlas (VFH path).
    obs_map: ego-space vis._persistent_obs (240x320) for MPPI path.
    """
    with _lock:
        if not _active:
            return None
        mode = _mode
        goal = _goal_xy
        planner = _planner

    if mode == "idle":
        return None

    if planner == "mppi":
        return _tick_mppi(obs_map, pose_x, pose_y, pose_theta)

    return _tick_vfh(atlas, pose_x, pose_y, pose_theta, mode, goal)
