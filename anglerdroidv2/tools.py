"""
tools.py - Wraps wheelbase, vision, ui. High-level tools for AI loop (sim or real).
"""

from typing import Optional

import numpy as np

try:
    import wheelbase
except ImportError:
    wheelbase = None

import vision
import ui


# --- Shared instances (main script sets these or tools.init) ---
_wheelbase = None
_vision = None  # type: Optional[vision.Vision]
_ui = None  # type: Optional[ui.UI]


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
    _, atlas, ts = _vision.read()
    return atlas, ts


# --- Wheelbase ---

def set_wheel_vels(left_tps: float, right_tps: float):
    """Set wheel velocities (turns per second)."""
    if _wheelbase is not None:
        _wheelbase.set_wheel_vels(left_tps, right_tps)

def stop():
    """Stop wheels."""
    if _wheelbase is not None:
        _wheelbase.stop()

def twist(forward_mps: float, angular_rads: float):
    """Differential drive: forward m/s, angular rad/s."""
    if _wheelbase is not None:
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


# --- Raw access for main loop ---
def get_vision():
    return _vision

def get_wheelbase():
    return _wheelbase

def get_ui():
    return _ui
