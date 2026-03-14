"""
tools.py - Wraps wheelbase, vision, ui, navigator. High-level tools for AI loop (sim or real).
"""

from typing import Optional

import numpy as np

try:
    import wheelbase
except ImportError:
    wheelbase = None

import vision
import ui
import navigator
import goals


# --- Shared instances (main script sets these or tools.init) ---
_wheelbase = None
_vision = None  # type: Optional[vision.Vision]
_ui = None  # type: Optional[ui.UI]
_safety_fwd = 1.0
_safety_bwd = 1.0
_safety_ang = 1.0


def set_safety_scales(fwd, bwd, ang=1.0):
    global _safety_fwd, _safety_bwd, _safety_ang
    _safety_fwd = max(0.0, min(1.0, float(fwd)))
    _safety_bwd = max(0.0, min(1.0, float(bwd)))
    _safety_ang = max(0.0, min(1.0, float(ang)))


def init(
    wheelbase_instance=None,
    vision_instance=None,
    ui_instance=None,
    *,
    rs1_serial: str = "",
    rs2_serial: str = "",
    rgb1_device_id=None,
):
    """Initialize tools with real or stub components."""
    global _wheelbase, _vision, _ui
    _wheelbase = wheelbase_instance
    _vision = vision_instance
    _ui = ui_instance
    if _vision is None and (rs1_serial or rs2_serial):
        _vision = vision.Vision(
            rs1_serial=rs1_serial or "0",
            rs2_serial=rs2_serial or "0",
            rgb1_device_id=rgb1_device_id or 0,
        )
    if _ui is None:
        _ui = ui.UI()


# --- Vision (read-only) ---

def get_frames():
    """Return (list of 3 frames 320x240 RGB, atlas 640x480, timestamp)."""
    if _vision is None:
        return [np.zeros((240, 320, 3), dtype=np.uint8)] * 3, np.zeros((480, 640, 3), dtype=np.uint8), 0.0
    return _vision.read()

def get_atlas():
    """Return (atlas 640x480, timestamp)."""
    if _vision is None:
        return np.zeros((480, 640, 3), dtype=np.uint8), 0.0
    return _vision.read_atlas()


# --- Wheelbase ---

def twist_for(forward_mps: float, angular_rads: float,
              duration_secs: float = 2.0, ramp_in_secs: float = 1.0, ramp_out_secs: float = 1.0):
    """
    Main method to control the wheelbase. Run differential drive for a set time with ramps.
    forward_mps: forward speed (m/s), angular_rads: turn rate (rad/s).
    duration_secs: how long to run; ramp_in_secs/ramp_out_secs: ramp only forward velocity
    (angular constant during ramps). New call overrides any in-progress twist_for. 10 Hz internal timer.
    """
    if _wheelbase is not None:
        _wheelbase.twist_for(forward_mps, angular_rads, duration_secs, ramp_in_secs, ramp_out_secs)

def set_wheel_vels(left_tps: float, right_tps: float):
    """Reserved for human gamepad control. Direct wheel velocities (turns/s). AI should use twist_for()."""
    if _wheelbase is not None:
        fwd = (left_tps + right_tps) / 2.0
        turn = (right_tps - left_tps) / 2.0
        if fwd > 0:
            fwd *= _safety_fwd
        elif fwd < 0:
            fwd *= _safety_bwd
        turn *= _safety_ang
        _wheelbase.set_wheel_vels(fwd - turn, fwd + turn)

def stop():
    """Immediately stop wheels and cancel any active twist_for."""
    if _wheelbase is not None:
        _wheelbase.cancel_twist_for()
        _wheelbase.stop()

def twist(forward_mps: float, angular_rads: float):
    """Instant differential drive (forward m/s, angular rad/s). Prefer twist_for() for timed moves."""
    if _wheelbase is not None:
        if forward_mps > 0:
            forward_mps *= _safety_fwd
        elif forward_mps < 0:
            forward_mps *= _safety_bwd
        angular_rads *= _safety_ang
        _wheelbase.twist(forward_mps, angular_rads)


# --- UI (user text + tool calls from agent) ---

def get_user_text() -> str:
    if _ui is None:
        return ""
    return _ui.get_user_text()

def get_pending_tool_calls():
    if _ui is None:
        return []
    return _ui.get_pending_tool_calls()


# --- Navigation (reactive obstacle avoidance) ---

def navigate(heading_deg):
    """Start navigating toward heading. 0=forward, 90=left, -90=right. None=stop."""
    if heading_deg is None:
        navigator.clear_goal()
    else:
        navigator.set_goal(float(heading_deg))

def navigate_stop():
    """Stop reactive navigation."""
    navigator.clear_goal()


# --- Visual goals (landmark-based navigation) ---

def set_visual_goals(landmarks):
    """Set ordered visual landmarks to navigate to sequentially."""
    goals.set_goals(landmarks)

def goal_reached():
    """Mark current visual goal as reached, advance to next. Returns next goal or None."""
    return goals.advance()

def get_goals_status():
    """Return current goals state for UI/prompt context."""
    return goals.get_status()

def clear_goals():
    """Cancel all visual goals."""
    goals.clear()


# --- Raw access for main loop ---
def get_vision():
    return _vision

def get_wheelbase():
    return _wheelbase

def get_ui():
    return _ui
