"""trace_logger.py – ASPIRE-inspired execution trace logger.

Captures multimodal traces of primitive execution:
- Velocity commands (twist)
- Safety events (reflexes, throttling, stuck detection)
- Local executive decisions (goals, planning backend switches)
- Sensor snapshots (optional: costmap, height map, face detections)

Outputs:
- JSONL: one event per line, timestamped, structured
- Optional Rerun integration: visual debugging (costmaps, trajectories)

NO neural network weights shipping — only code/text skills evolved from traces.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional
import threading

_lock = threading.Lock()
_enabled = False
_log_file = None  # type: Optional[Any]
_rerun_enabled = False
_frame_count = 0


def init(log_path: str = "traces/run.jsonl", enable_rerun: bool = False) -> None:
    """Initialize trace logger with JSONL output and optional Rerun.
    
    Args:
        log_path: Path to JSONL output file (default: traces/run.jsonl)
        enable_rerun: Enable Rerun visual logging (requires rerun-sdk)
    """
    global _enabled, _log_file, _rerun_enabled
    
    with _lock:
        if _enabled:
            print("trace_logger: already initialized")
            return
            
        # Create trace directory
        log_path_obj = Path(log_path)
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Open JSONL file
        _log_file = open(log_path, "a")
        _enabled = True
        
        print(f"trace_logger: JSONL → {log_path}")
        
        # Optional Rerun initialization
        if enable_rerun:
            try:
                import rerun as rr
                rr.init("kevin_traces", spawn=False)
                _rerun_enabled = True
                print("trace_logger: Rerun enabled")
            except ImportError:
                print("trace_logger: Rerun not available (install rerun-sdk)")
                _rerun_enabled = False


def shutdown() -> None:
    """Close trace logger and flush buffers."""
    global _enabled, _log_file
    
    with _lock:
        if not _enabled:
            return
            
        if _log_file is not None:
            _log_file.close()
            _log_file = None
            
        _enabled = False
        print("trace_logger: shutdown complete")


def is_enabled() -> bool:
    """Check if trace logging is active."""
    with _lock:
        return _enabled


def log_event(event: str, **kwargs) -> None:
    """Log a structured event to JSONL trace.
    
    Args:
        event: Event type (e.g., "twist_cmd", "safety_reflex", "goal_set")
        **kwargs: Additional event data (must be JSON-serializable)
    """
    with _lock:
        if not _enabled or _log_file is None:
            return
            
        try:
            record = {
                "timestamp": time.time(),
                "event": event,
                **kwargs
            }
            _log_file.write(json.dumps(record) + "\n")
            _log_file.flush()
        except Exception as e:
            print(f"trace_logger: failed to log event {event}: {e}")


def log_twist_cmd(fwd_mps: float, ang_rads: float, safety_scaled: bool = False) -> None:
    """Log a twist command (velocity control primitive).
    
    Args:
        fwd_mps: Forward velocity (m/s)
        ang_rads: Angular velocity (rad/s)
        safety_scaled: True if safety layer modified the command
    """
    log_event(
        "twist_cmd",
        fwd_mps=round(fwd_mps, 3),
        ang_rads=round(ang_rads, 3),
        safety_scaled=safety_scaled
    )


def log_safety_event(
    reflex_type: Optional[str],
    fwd_scale: float,
    bwd_scale: float,
    ang_scale: float,
    throttled: bool
) -> None:
    """Log safety layer state and reflex activation.
    
    Args:
        reflex_type: Active reflex name (e.g., "topdown_near_field", None if inactive)
        fwd_scale: Forward velocity scale (0.0-1.0)
        bwd_scale: Backward velocity scale (0.0-1.0)
        ang_scale: Angular velocity scale (0.0-1.0)
        throttled: True if any scale < 0.95
    """
    log_event(
        "safety_state",
        reflex_type=reflex_type,
        fwd_scale=round(fwd_scale, 2),
        bwd_scale=round(bwd_scale, 2),
        ang_scale=round(ang_scale, 2),
        throttled=throttled
    )


def log_local_executive(mode: str, goal_xy: Optional[tuple], planner: str, active: bool) -> None:
    """Log local executive state change.
    
    Args:
        mode: Current mode ("idle", "xy", "wander")
        goal_xy: Target position (x, y) in world frame, or None
        planner: Planner backend ("vfh", "mppi", "neural_rl")
        active: True if executive is driving
    """
    log_event(
        "local_executive",
        mode=mode,
        goal_xy=goal_xy,
        planner=planner,
        active=active
    )


def log_stuck_detection(detected: bool, stuck_type: Optional[str], duration_s: float) -> None:
    """Log stuck state detection.
    
    Args:
        detected: True if stuck condition detected
        stuck_type: Type of stuck ("slip", "oscillation", "pinned", None)
        duration_s: How long stuck condition has persisted
    """
    log_event(
        "stuck_detection",
        detected=detected,
        stuck_type=stuck_type,
        duration_s=round(duration_s, 2)
    )


def log_skill_invocation(skill_name: str, params: dict, outcome: Optional[str] = None) -> None:
    """Log high-level skill execution.
    
    Args:
        skill_name: Skill identifier (e.g., "dog_bed_soft_approach")
        params: Skill parameters
        outcome: Result if known ("success", "failure", "in_progress", None)
    """
    log_event(
        "skill",
        skill_name=skill_name,
        params=params,
        outcome=outcome
    )


def log_costmap_snapshot(obs_map_shape: tuple, occupied_px: int, height_cm_max: Optional[int] = None) -> None:
    """Log costmap state summary (no pixel data → keeps traces small).
    
    Args:
        obs_map_shape: Shape of obstacle map (H, W)
        occupied_px: Count of occupied pixels (>= threshold)
        height_cm_max: Maximum height in height map (cm), or None
    """
    global _frame_count
    _frame_count += 1
    
    log_event(
        "costmap_snapshot",
        frame=_frame_count,
        shape=obs_map_shape,
        occupied_px=occupied_px,
        height_cm_max=height_cm_max
    )


def log_rerun_trajectory(path: list[tuple[float, float]], color: tuple[int, int, int] = (60, 120, 255)) -> None:
    """Log predicted trajectory to Rerun visual logger (if enabled).
    
    Args:
        path: List of (x, y) points in ego frame (pixels)
        color: RGB color tuple
    """
    if not _rerun_enabled:
        return
        
    try:
        import rerun as rr
        import numpy as np
        
        if len(path) < 2:
            return
            
        points = np.array(path, dtype=np.float32)
        rr.log("trajectory/predicted", rr.LineStrips2D(points, colors=color))
    except Exception as e:
        print(f"trace_logger: Rerun trajectory failed: {e}")


# Convenience: log a complete 30Hz main-loop tick
def log_tick_snapshot(
    twist_cmd: Optional[tuple[float, float]],
    safety_state: dict,
    executive_state: dict,
    obs_map_summary: Optional[dict] = None
) -> None:
    """Log a complete main-loop tick with all relevant state.
    
    Args:
        twist_cmd: (fwd_mps, ang_rads) if commanded, else None
        safety_state: dict with keys: reflex_type, fwd_scale, bwd_scale, ang_scale, throttled
        executive_state: dict with keys: mode, goal_xy, planner, active
        obs_map_summary: optional dict with keys: shape, occupied_px, height_cm_max
    """
    if twist_cmd is not None:
        log_twist_cmd(twist_cmd[0], twist_cmd[1], safety_scaled=safety_state.get("throttled", False))
    
    log_safety_event(
        safety_state.get("reflex_type"),
        safety_state.get("fwd_scale", 1.0),
        safety_state.get("bwd_scale", 1.0),
        safety_state.get("ang_scale", 1.0),
        safety_state.get("throttled", False)
    )
    
    log_local_executive(
        executive_state.get("mode", "idle"),
        executive_state.get("goal_xy"),
        executive_state.get("planner", "vfh"),
        executive_state.get("active", False)
    )
    
    if obs_map_summary is not None:
        log_costmap_snapshot(
            obs_map_summary.get("shape", (0, 0)),
            obs_map_summary.get("occupied_px", 0),
            obs_map_summary.get("height_cm_max")
        )


if __name__ == "__main__":
    # Demo: log a few trace events
    init("traces/demo.jsonl", enable_rerun=False)
    
    log_event("demo_start", robot="kevin", version="aspire_v1")
    
    log_twist_cmd(0.3, 0.1, safety_scaled=False)
    log_safety_event(None, 1.0, 1.0, 1.0, False)
    log_local_executive("xy", (2.5, 1.3), "vfh", True)
    
    time.sleep(0.5)
    
    log_safety_event("topdown_near_field", 0.0, 0.8, 1.0, True)
    log_twist_cmd(0.0, 0.0, safety_scaled=True)
    
    log_skill_invocation("dog_bed_soft_approach", {"distance_cm": 65}, outcome="in_progress")
    
    log_event("demo_end")
    
    shutdown()
    
    print("\nDemo trace written to traces/demo.jsonl")
