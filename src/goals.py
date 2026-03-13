"""
goals.py – Visual goal tracker for landmark-based navigation.

Inspired by LM-Nav: decompose complex instructions into ordered visual landmarks,
then pursue them sequentially. Gemini extracts landmarks and confirms arrival
using live camera frames (replacing CLIP scoring + graph search).

Thread-safe: main loop reads status, Gemini thread mutates via tool calls.
"""

import threading
from typing import Optional, List


_lock = threading.Lock()
_goals = []            # type: List[str]
_current_idx = 0
_active = False
_completed = []        # type: List[str]
_on_change = None      # type: Optional[callable]


def set_goals(landmarks):
    # type: (List[str]) -> None
    """Set ordered visual landmarks to navigate to. Resets progress."""
    global _goals, _current_idx, _active, _completed
    with _lock:
        _goals = list(landmarks)
        _current_idx = 0
        _completed = []
        _active = len(_goals) > 0
    _notify()


def current_goal():
    # type: () -> Optional[str]
    """Return the current goal description, or None if done/inactive."""
    with _lock:
        if not _active or _current_idx >= len(_goals):
            return None
        return _goals[_current_idx]


def advance():
    # type: () -> Optional[str]
    """Mark current goal as reached, move to next. Returns new current goal or None if all done."""
    global _current_idx, _active
    with _lock:
        if _current_idx < len(_goals):
            _completed.append(_goals[_current_idx])
        _current_idx += 1
        if _current_idx >= len(_goals):
            _active = False
    _notify()
    return current_goal()


def clear():
    # type: () -> None
    """Cancel all goals."""
    global _goals, _current_idx, _active, _completed
    with _lock:
        _goals = []
        _current_idx = 0
        _completed = []
        _active = False
    _notify()


def is_active():
    # type: () -> bool
    with _lock:
        return _active


def get_status():
    # type: () -> dict
    """Return serializable status dict for UI display."""
    with _lock:
        return {
            "goals": _goals[:],
            "current": _current_idx,
            "active": _active,
            "completed": _completed[:],
            "current_goal": _goals[_current_idx] if _active and _current_idx < len(_goals) else None,
            "total": len(_goals),
        }


def set_on_change(callback):
    # type: (callable) -> None
    """Register a callback(status_dict) fired on every state change."""
    global _on_change
    _on_change = callback


def _notify():
    cb = _on_change
    if cb is not None:
        try:
            cb(get_status())
        except Exception:
            pass
