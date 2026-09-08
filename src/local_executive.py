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
_planner = "vfh"  # 'vfh' | 'mppi' | 'neural_rl'
_mppi = None  # lazy MppiCostmapPlanner
_neural_rl = None  # lazy NeuralRLPolicy


def _ensure_mppi():
    global _mppi
    if _mppi is None:
        from mppi_costmap import MppiCostmapPlanner
        _mppi = MppiCostmapPlanner()
    return _mppi


def _ensure_neural_rl():
    global _neural_rl
    if _neural_rl is None:
        from neural_rl import NeuralRLPolicy
        import os
        model_path = os.environ.get('KEVIN_NEURAL_MODEL_PATH')
        fallback_planner = os.environ.get('KEVIN_NEURAL_FALLBACK', 'mppi')
        _neural_rl = NeuralRLPolicy(
            model_path=model_path,
            inference_budget_ms=5.0,
            fallback_planner=fallback_planner
        )
    return _neural_rl


def set_planner(name: str) -> None:
    """Select backend: 'vfh' (default), 'mppi', or 'neural_rl'."""
    global _planner
    name = (name or "vfh").strip().lower()
    if name not in ("vfh", "mppi", "neural_rl"):
        raise ValueError("planner must be 'vfh', 'mppi', or 'neural_rl', got %r" % name)
    with _lock:
        _planner = name
    if name == "mppi":
        _ensure_mppi()
    elif name == "neural_rl":
        _ensure_neural_rl()
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
    elif planner == "neural_rl":
        _ensure_neural_rl().set_goal(float(x), float(y))


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
    elif planner == "neural_rl":
        _ensure_neural_rl().set_wander_mode(True)


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
    elif planner == "neural_rl" and _neural_rl is not None:
        _neural_rl.cancel()


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
    elif planner == "neural_rl" and _neural_rl is not None:
        out["neural_rl"] = _neural_rl.get_debug_state()
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


def _tick_neural_rl(obs_map, pose_x, pose_y, pose_theta):
    """Neural RL backend: learned policy on ego costmap with optional policy_feed.
    
    Falls back to configured fallback planner (mppi/vfh) if neural fails.
    """
    global _dbg, _active, _mode

    neural = _ensure_neural_rl()
    if pose_x is None or pose_y is None:
        return None
    
    # Try to get policy feed (labeled heightmap) from Vision
    policy_feed = None
    slam_locked = False
    try:
        vis = tools.get_vision()
        if vis:
            policy_feed = vis.get_policy_feed()
            slam_locked = vis.slam_locked
    except Exception as e:
        print("local_executive: get_policy_feed failed: %s" % e)
    
    # Prepare obs_map (legacy fallback)
    if obs_map is None:
        import numpy as np
        from robot_config import FRAME_H, FRAME_W
        obs_map = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    # Paint keepouts onto obs_map
    pose = (float(pose_x), float(pose_y), float(pose_theta or 0.0))
    try:
        obs_map = keepouts.paint_ego(obs_map, pose, slam_locked=slam_locked)
    except Exception as e:
        print("keepouts: paint skip %s" % e)
    
    # Tick neural policy
    cmd = neural.tick(obs_map, pose, 0.033, policy_feed=policy_feed)
    
    if cmd is not None and cmd.get('source') == 'neural':
        # Neural inference succeeded
        with _lock:
            if not neural.is_active():
                _active = False
                _mode = "idle"
                _dbg = {"event": "neural_rl_inactive"}
            else:
                _dbg = {
                    "mode": _mode,
                    "planner": "neural_rl",
                    "cmd": cmd,
                    "inference_ms": cmd.get("inference_ms"),
                    "using_policy_feed": policy_feed is not None and policy_feed.get('valid', False),
                }
        return (float(cmd["fwd_mps"]), float(cmd["ang_rads"]))
    
    # Neural failed or returned None → fallback to configured planner
    fallback = neural._fallback_planner
    with _lock:
        _dbg = {
            "mode": _mode,
            "planner": "neural_rl",
            "fallback_active": True,
            "fallback_planner": fallback,
        }
    
    # Execute fallback
    if fallback == "mppi":
        # Use MPPI fallback
        try:
            mppi = _ensure_mppi()
            # Sync goal state to MPPI
            if _mode == "xy" and _goal_xy is not None:
                mppi.set_goal(_goal_xy[0], _goal_xy[1])
            elif _mode == "wander":
                mppi.set_wander_mode(True)
            else:
                mppi.cancel()
            
            # Get policy observation for MPPI
            policy_obs = None
            try:
                vis = tools.get_vision()
                if vis:
                    policy_obs = vis.get_policy_observation()
            except Exception:
                pass
            
            # Choose MPPI input
            if policy_obs is not None and policy_obs.get('metadata', {}).get('topdown_ok'):
                mppi_input = policy_obs
            else:
                mppi_input = obs_map
            
            # Paint keepouts for MPPI
            try:
                if isinstance(mppi_input, dict):
                    ego_pers = mppi_input.get('ego_persistent')
                    if ego_pers is not None:
                        painted = keepouts.paint_ego(
                            np.array(ego_pers, copy=True), pose, slam_locked=slam_locked)
                        mppi_input = dict(mppi_input)
                        mppi_input['ego_persistent'] = painted
                else:
                    mppi_input = keepouts.paint_ego(mppi_input, pose, slam_locked=slam_locked)
            except Exception as e:
                print("keepouts: paint skip in neural_rl fallback %s" % e)
            
            mppi_cmd = mppi.tick(mppi_input, pose, 0.033)
            if mppi_cmd is not None:
                return (float(mppi_cmd["fwd_mps"]), float(mppi_cmd["ang_rads"]))
        except Exception as e:
            print("neural_rl: mppi fallback failed: %s" % e)
    
    elif fallback == "vfh":
        # Use VFH fallback
        try:
            atlas, _ = tools.get_atlas()
            if atlas is None:
                return None
            
            # VFH needs the goal in the appropriate mode
            if _mode == "xy":
                if _goal_xy is None:
                    return None
                sx, sy, dist = _rolling_point_toward(
                    _goal_xy[0], _goal_xy[1], pose_x, pose_y, LOOKAHEAD_M)
                if dist < GOAL_REACHED_M:
                    clear()
                    return (0.0, 0.0)
                hdg = _world_to_robot_heading_deg(
                    sx, sy, pose_x, pose_y, pose_theta or 0.0)
                navigator.set_goal(hdg)
            elif _mode == "wander":
                import time
                global _last_wander_t
                now = time.monotonic()
                if now - _last_wander_t >= WANDER_REFRESH_S:
                    navigator.set_goal(0.0)
                    _last_wander_t = now
            else:
                return None
            
            return navigator.compute_twist(atlas)
        except Exception as e:
            print("neural_rl: vfh fallback failed: %s" % e)
    
    # All fallbacks exhausted
    return None


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
    obs_map: ego-space vis._persistent_obs (240x320) for MPPI/neural_rl path.
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
    
    if planner == "neural_rl":
        return _tick_neural_rl(obs_map, pose_x, pose_y, pose_theta)

    return _tick_vfh(atlas, pose_x, pose_y, pose_theta, mode, goal)
