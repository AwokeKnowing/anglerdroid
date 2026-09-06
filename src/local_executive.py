"""local_executive.py – Mid-layer continuous local motion (no ROS).

Async goal mailbox for high-level agents (Kevins Doctor / tools):
  - set_goal_xy(x, y)  world meters (map frame, same as PoseEstimator)
  - set_wander()       keep rolling ~1 m free-space goals
  - clear()

Each main-loop tick (~30 Hz): pick a rolling subgoal within LOOKAHEAD_M,
convert to a robot-relative heading, drive existing VFH navigator toward it.
Never blocks. Capture/map thread stays untouched.

Phase-1 path: VFH executor now; CUDA MPPI plugs in later behind the same API.
"""

from __future__ import annotations

import math
import threading
import time

import navigator

LOOKAHEAD_M = 1.0
GOAL_REACHED_M = 0.35
WANDER_REFRESH_S = 0.4
# Atlas obstacle quad uses ~pixels; MAX_RANGE=130px. Ego topdown scale in vision
# is typically ~FRAME related; VFH heading 0 = forward (atlas robot faces RIGHT).


_lock = threading.Lock()
_mode = "idle"  # idle | xy | wander
_goal_xy = None  # (x, y) world m
_active = False
_last_wander_t = 0.0
_dbg = {}


def set_goal_xy(x, y):
    """High-level: drive toward world-frame point (meters). Non-blocking."""
    global _mode, _goal_xy, _active
    with _lock:
        _goal_xy = (float(x), float(y))
        _mode = "xy"
        _active = True


def set_wander():
    """High-level: continuously pick ~1 m free headings. Non-blocking."""
    global _mode, _goal_xy, _active, _last_wander_t
    with _lock:
        _mode = "wander"
        _goal_xy = None
        _active = True
        _last_wander_t = 0.0


def clear():
    global _mode, _goal_xy, _active
    with _lock:
        _mode = "idle"
        _goal_xy = None
        _active = False
    navigator.clear_goal()


def is_active():
    with _lock:
        return _active


def status():
    with _lock:
        return {
            "active": _active,
            "mode": _mode,
            "goal_xy": _goal_xy,
            "dbg": dict(_dbg),
        }


def _world_to_robot_heading_deg(gx, gy, rx, ry, rtheta):
    """Heading in navigator frame: 0=forward, +CCW (left positive)."""
    dx = gx - rx
    dy = gy - ry
    # world yaw: 0 = +x; robot theta same convention in PoseEstimator
    desired = math.atan2(dy, dx)
    err = (desired - rtheta + math.pi) % (2 * math.pi) - math.pi
    # navigator: 0=forward, +left. World err already CCW from robot heading.
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


def tick(atlas, pose_x=None, pose_y=None, pose_theta=None):
    """Call from 30 Hz main loop. Returns (fwd, ang) or None if inactive.

    If pose_* omitted, only wander/heading modes that don't need world pose
    (wander uses VFH forward bias via navigator goal heading 0 refresh).
    """
    global _last_wander_t, _dbg, _active, _mode

    with _lock:
        if not _active:
            return None
        mode = _mode
        goal = _goal_xy

    if mode == "idle":
        return None

    if mode == "xy":
        if pose_x is None or goal is None:
            return None
        sx, sy, dist = _rolling_point_toward(goal[0], goal[1], pose_x, pose_y, LOOKAHEAD_M)
        if dist < GOAL_REACHED_M:
            clear()
            with _lock:
                _dbg = {"event": "reached", "dist": dist}
            return (0.0, 0.0)
        hdg = _world_to_robot_heading_deg(sx, sy, pose_x, pose_y, pose_theta or 0.0)
        navigator.set_goal(hdg)
        with _lock:
            _dbg = {"mode": "xy", "dist": dist, "hdg": hdg, "sub": (sx, sy)}
        return navigator.compute_twist(atlas) if atlas is not None else None

    if mode == "wander":
        now = time.monotonic()
        # Refresh rolling forward-ish goal heading periodically; VFH finds clear bin nearby.
        if now - _last_wander_t >= WANDER_REFRESH_S:
            # Prefer slight left/right alternation is not needed — VFH picks clear near 0°.
            navigator.set_goal(0.0)  # keep seeking forward clear space ~1 m (VFH MAX_RANGE)
            _last_wander_t = now
        with _lock:
            _dbg = {"mode": "wander", "hdg": 0.0}
        if atlas is None:
            return None
        return navigator.compute_twist(atlas)

    return None
