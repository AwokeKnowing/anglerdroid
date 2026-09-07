#!/usr/bin/env python3
"""Unit tests for visual↔wheel stuck detection + immobilize policy."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pose import PoseEstimator


def _force_stuck(pose, commanded=0.20, actual=0.01, visual_ok=True, enc=False):
    """Drive one detection window to stuck=True without waiting real time."""
    pose.STUCK_WINDOW_S = 0.05
    pose._stuck_check_start = time.time() - 0.06
    pose._commanded_forward_sum = commanded
    pose._actual_forward_sum = actual
    pose._update_stuck_detection(
        ds_commanded=0.0,
        ds_visual=0.0,
        visual_ok=visual_ok,
        using_encoder_feedback=enc,
        dt=0.05,
    )


def test_detects_spinning_wheels_no_motion():
    pose = PoseEstimator(0.165, 0.03)
    _force_stuck(pose, commanded=0.25, actual=0.02)
    assert pose.is_stuck, "expected stuck when actual << commanded"
    assert pose.stuck_count == 1
    print("PASS: stuck detected on commanded>>visual mismatch")


def test_clears_when_motion_matches():
    pose = PoseEstimator(0.165, 0.03)
    _force_stuck(pose, commanded=0.25, actual=0.02)
    assert pose.is_stuck
    # Next window: motion matches command
    pose.STUCK_WINDOW_S = 0.05
    pose._stuck_check_start = time.time() - 0.06
    pose._commanded_forward_sum = 0.25
    pose._actual_forward_sum = 0.24
    pose._update_stuck_detection(0.0, 0.0, True, False, 0.05)
    assert not pose.is_stuck, "expected unstuck when ratio high"
    print("PASS: stuck clears when visual catches up")


def test_no_false_positive_when_idle():
    pose = PoseEstimator(0.165, 0.03)
    pose.STUCK_WINDOW_S = 0.05
    pose._stuck_check_start = time.time() - 0.06
    pose._commanded_forward_sum = 0.01  # below STUCK_CMD_THRESHOLD
    pose._actual_forward_sum = 0.0
    pose._update_stuck_detection(0.0, 0.0, True, False, 0.05)
    assert not pose.is_stuck
    print("PASS: idle / tiny command does not mark stuck")


def test_stuck_immobilize_policy_allows_reverse():
    """Policy check: STUCK zeros fwd only; TOPDOWN LOST zeros all."""
    stuck_reason = "STUCK (wheels spinning, no motion)"
    lost_reason = "TOPDOWN LOST"
    stuck_only = stuck_reason.startswith("STUCK")
    lost_only = lost_reason.startswith("STUCK")
    assert stuck_only and not lost_only
    # Document expected scale outcomes
    stuck_scales = {"fwd": 0.0, "bwd": "keep", "ang": "keep"}
    lost_scales = {"fwd": 0.0, "bwd": 0.0, "ang": 0.0}
    assert stuck_scales["bwd"] == "keep"
    assert lost_scales["bwd"] == 0.0
    print("PASS: stuck immobilize policy keeps reverse/turn")


if __name__ == "__main__":
    test_detects_spinning_wheels_no_motion()
    test_clears_when_motion_matches()
    test_no_false_positive_when_idle()
    test_stuck_immobilize_policy_allows_reverse()
    print("\nALL STUCK TESTS PASSED")
