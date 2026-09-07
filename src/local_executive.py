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
import numpy as np
import keepouts
import threading
import time

import navigator
import tools

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
    """MPPI backend: uses policy observation from Vision.get_policy_observation().
    
    Falls back to legacy obs_map (vis._persistent_obs) if policy obs unavailable.
    """
    global _dbg, _active, _mode

    mppi = _ensure_mppi()
    if pose_x is None or pose_y is None:
        return None
    
    # Try to get policy observation (labeled heightmap layers)
    policy_obs = None
    slam_locked = False
    try:
        vis = tools.get_vision()
        if vis:
            policy_obs = vis.get_policy_observation()
            slam_locked = vis.slam_locked
    except Exception as e:
        print("local_executive: get_policy_observation failed: %s" % e)
    
    # Choose input: policy obs dict (preferred) or legacy array (fallback)
    if policy_obs is not None and policy_obs.get('metadata', {}).get('topdown_ok'):
        # Use policy observation (contains ego_height for graduated costing)
        mppi_input = policy_obs
        using_policy_obs = True
    else:
        # Fallback: legacy _persistent_obs array
        if obs_map is None:
            import numpy as np
            from robot_config import FRAME_H, FRAME_W
            obs_map = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        mppi_input = obs_map
        using_policy_obs = False

    pose = (float(pose_x), float(pose_y), float(pose_theta or 0.0))
    
    # Paint keepouts onto ego map (only if SLAM locked)
    try:
        if isinstance(mppi_input, dict):
            # Paint keepouts onto a copy — policy buffers are shared/prealloc
            ego_pers = mppi_input.get('ego_persistent')
            if ego_pers is not None:
                painted = keepouts.paint_ego(
                    np.array(ego_pers, copy=True), pose, slam_locked=slam_locked)
                mppi_input = dict(mppi_input)
                mppi_input['ego_persistent'] = painted
        else:
            # Legacy array path
            mppi_input = keepouts.paint_ego(mppi_input, pose, slam_locked=slam_locked)
    except Exception as e:
        print("keepouts: paint skip %s" % e)
    
    cmd = mppi.tick(mppi_input, pose, 0.033)
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
                "using_policy_obs": using_policy_obs,
                "using_heightmap": mppi.get_debug_state().get("using_heightmap", False),
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
