"""test_trace_logger.py — Unit tests for trace logger (ASPIRE execution traces).

Tests JSONL output, event logging, and shutdown without requiring hardware.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

# Import trace logger
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import trace_logger


@pytest.fixture
def temp_trace_file():
    """Create a temporary trace file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        trace_path = f.name
    
    yield trace_path
    
    # Cleanup
    if os.path.exists(trace_path):
        os.unlink(trace_path)


def test_trace_logger_init_shutdown(temp_trace_file):
    """Test trace logger initialization and shutdown."""
    # Initialize
    trace_logger.init(temp_trace_file, enable_rerun=False)
    assert trace_logger.is_enabled()
    
    # Shutdown
    trace_logger.shutdown()
    assert not trace_logger.is_enabled()


def test_trace_logger_log_event(temp_trace_file):
    """Test generic event logging."""
    trace_logger.init(temp_trace_file, enable_rerun=False)
    
    # Log a few events
    trace_logger.log_event("test_start", robot="kevin", version="v1")
    trace_logger.log_event("test_data", value=42, text="hello")
    trace_logger.log_event("test_end")
    
    trace_logger.shutdown()
    
    # Read back and verify
    with open(temp_trace_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 3
    
    # Parse first event
    event1 = json.loads(lines[0])
    assert event1["event"] == "test_start"
    assert event1["robot"] == "kevin"
    assert event1["version"] == "v1"
    assert "timestamp" in event1


def test_trace_logger_twist_cmd(temp_trace_file):
    """Test twist command logging."""
    trace_logger.init(temp_trace_file, enable_rerun=False)
    
    trace_logger.log_twist_cmd(0.3, 0.1, safety_scaled=False)
    trace_logger.log_twist_cmd(0.0, 0.0, safety_scaled=True)
    
    trace_logger.shutdown()
    
    # Verify
    with open(temp_trace_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 2
    
    event1 = json.loads(lines[0])
    assert event1["event"] == "twist_cmd"
    assert event1["fwd_mps"] == 0.3
    assert event1["ang_rads"] == 0.1
    assert event1["safety_scaled"] is False
    
    event2 = json.loads(lines[1])
    assert event2["safety_scaled"] is True


def test_trace_logger_safety_event(temp_trace_file):
    """Test safety event logging."""
    trace_logger.init(temp_trace_file, enable_rerun=False)
    
    trace_logger.log_safety_event(
        reflex_type="topdown_near_field",
        fwd_scale=0.0,
        bwd_scale=0.8,
        ang_scale=1.0,
        throttled=True
    )
    
    trace_logger.log_safety_event(
        reflex_type=None,
        fwd_scale=1.0,
        bwd_scale=1.0,
        ang_scale=1.0,
        throttled=False
    )
    
    trace_logger.shutdown()
    
    # Verify
    with open(temp_trace_file, 'r') as f:
        lines = f.readlines()
    
    event1 = json.loads(lines[0])
    assert event1["event"] == "safety_state"
    assert event1["reflex_type"] == "topdown_near_field"
    assert event1["fwd_scale"] == 0.0
    assert event1["throttled"] is True
    
    event2 = json.loads(lines[1])
    assert event2["reflex_type"] is None
    assert event2["throttled"] is False


def test_trace_logger_local_executive(temp_trace_file):
    """Test local executive state logging."""
    trace_logger.init(temp_trace_file, enable_rerun=False)
    
    trace_logger.log_local_executive(
        mode="xy",
        goal_xy=(2.5, 1.3),
        planner="vfh",
        active=True
    )
    
    trace_logger.log_local_executive(
        mode="idle",
        goal_xy=None,
        planner="neural_rl",
        active=False
    )
    
    trace_logger.shutdown()
    
    # Verify
    with open(temp_trace_file, 'r') as f:
        lines = f.readlines()
    
    event1 = json.loads(lines[0])
    assert event1["event"] == "local_executive"
    assert event1["mode"] == "xy"
    assert event1["goal_xy"] == [2.5, 1.3]
    assert event1["planner"] == "vfh"
    assert event1["active"] is True


def test_trace_logger_skill_invocation(temp_trace_file):
    """Test skill invocation logging."""
    trace_logger.init(temp_trace_file, enable_rerun=False)
    
    trace_logger.log_skill_invocation(
        skill_name="dog_bed_soft_approach",
        params={"distance_cm": 65, "height_cm": 12},
        outcome="in_progress"
    )
    
    trace_logger.log_skill_invocation(
        skill_name="stuck_recovery",
        params={"strategy": "back_out"},
        outcome="success"
    )
    
    trace_logger.shutdown()
    
    # Verify
    with open(temp_trace_file, 'r') as f:
        lines = f.readlines()
    
    event1 = json.loads(lines[0])
    assert event1["event"] == "skill"
    assert event1["skill_name"] == "dog_bed_soft_approach"
    assert event1["params"]["distance_cm"] == 65
    assert event1["outcome"] == "in_progress"


def test_trace_logger_tick_snapshot(temp_trace_file):
    """Test complete tick snapshot logging."""
    trace_logger.init(temp_trace_file, enable_rerun=False)
    
    trace_logger.log_tick_snapshot(
        twist_cmd=(0.3, 0.1),
        safety_state={
            "reflex_type": None,
            "fwd_scale": 1.0,
            "bwd_scale": 1.0,
            "ang_scale": 1.0,
            "throttled": False
        },
        executive_state={
            "mode": "xy",
            "goal_xy": (2.5, 1.3),
            "planner": "vfh",
            "active": True
        },
        obs_map_summary={
            "shape": (240, 320),
            "occupied_px": 1523,
            "height_cm_max": 35
        }
    )
    
    trace_logger.shutdown()
    
    # Verify multiple events logged
    with open(temp_trace_file, 'r') as f:
        lines = f.readlines()
    
    # Should have: twist_cmd, safety_state, local_executive, costmap_snapshot
    assert len(lines) == 4
    
    events = [json.loads(line) for line in lines]
    event_types = [e["event"] for e in events]
    
    assert "twist_cmd" in event_types
    assert "safety_state" in event_types
    assert "local_executive" in event_types
    assert "costmap_snapshot" in event_types


def test_trace_logger_disabled_logging():
    """Test that logging is no-op when disabled."""
    # Don't initialize, so logger is disabled
    assert not trace_logger.is_enabled()
    
    # These should not raise exceptions
    trace_logger.log_event("test")
    trace_logger.log_twist_cmd(0.3, 0.1)
    trace_logger.log_safety_event(None, 1.0, 1.0, 1.0, False)


def test_trace_logger_double_init(temp_trace_file):
    """Test that double init is handled gracefully."""
    trace_logger.init(temp_trace_file, enable_rerun=False)
    
    # Try to init again (should print warning but not crash)
    trace_logger.init(temp_trace_file, enable_rerun=False)
    
    assert trace_logger.is_enabled()
    
    trace_logger.shutdown()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
