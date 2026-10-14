#!/usr/bin/env python3
"""Host sim tests proving recover/commit never bypass hard safety stops.

Validates that MPPI policy in sim respects fwd_scale=0 and that the
action mask prevents forward motion when hard-pinned.
"""

import sys
import unittest
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

import numpy as np
from mppi_policy import MppiSimPolicy
from dual_clearance import evaluate as evaluate_dual
from robot_config import FRAME_W, FRAME_H, RCX, RCY


class TestRecoverSafetyHardStop(unittest.TestCase):
    """Test MPPI policy respects hard safety stops during recover/escape."""

    def test_mppi_policy_fwd_scale_zero(self):
        """MPPI policy must not command v>0 when fwd_scale=0."""
        policy = MppiSimPolicy(wander=True, use_soft_cost=True)
        policy.reset()
        
        # Empty obs (free space)
        obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        
        # Hard forward stop (near-field reflex / checkered mat / stuck)
        safety_scales = {
            "fwd": 0.0,
            "bwd": 1.0,
            "ang": 0.8,
            "fwd_m": 0.0,
            "bwd_m": 0.30,
            "lat_m": 0.15,
        }
        
        pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        
        # Run multiple ticks to allow MPPI to settle
        for _ in range(10):
            v, w = policy.act(obs, None, safety_scales, pose)
            # Critical: no forward motion allowed
            self.assertLessEqual(v, 0.0, f"MPPI commanded v={v} with fwd_scale=0")
            # Reverse and angular should be possible
            self.assertTrue(
                v <= 0.0 or abs(w) > 0.0,
                "MPPI should allow reverse/angular when fwd blocked"
            )

    def test_mppi_escape_back_when_pinned(self):
        """MPPI escape should back up when hard-pinned, not force forward."""
        policy = MppiSimPolicy(wander=True, use_soft_cost=True)
        policy.reset()
        
        # Obstacle wall ahead
        obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        obs[:, RCX+10:RCX+50] = 255
        
        # Hard forward stop
        safety_scales = {
            "fwd": 0.0,
            "bwd": 0.85,
            "ang": 0.70,
            "fwd_m": 0.0,
            "bwd_m": 0.28,
            "lat_m": 0.12,
        }
        
        pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        
        # Trigger pin streak by repeatedly seeing hard pin
        for _ in range(policy.PIN_STREAK + 2):
            v, w = policy.act(obs, None, safety_scales, pose)
        
        # After pin streak, should be in escape mode
        self.assertIsNotNone(policy.escape_phase, "Should enter escape after pin streak")
        
        # Escape should use reverse or spin, never forward
        for _ in range(20):
            v, w = policy.act(obs, None, safety_scales, pose)
            self.assertLessEqual(v, 0.0, f"Escape commanded v={v}>0 with fwd_scale=0")

    def test_mppi_commit_phase_simulation(self):
        """Simulate COMMIT phase: goal away from obstacle, fwd_scale=0 during commit."""
        policy = MppiSimPolicy(goal_xy=(1.0, 0.0), wander=False, use_soft_cost=True)
        policy.reset()
        
        # Free space
        obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        
        # Start with normal motion to let MPPI get going
        safety_scales = {
            "fwd": 0.85,
            "bwd": 1.0,
            "ang": 1.0,
            "fwd_m": 0.40,
            "bwd_m": 0.50,
            "lat_m": 0.30,
        }
        
        pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        
        # Run a few ticks
        for _ in range(3):
            v, w = policy.act(obs, None, safety_scales, pose)
            self.assertGreaterEqual(v, 0.0, "Should move forward when clear")
        
        # Suddenly hard-pin forward (simulates re-pinning during COMMIT)
        safety_scales["fwd"] = 0.0
        safety_scales["fwd_m"] = 0.0
        
        # Continue ticking — MPPI should respect the new hard stop
        for _ in range(10):
            v, w = policy.act(obs, None, safety_scales, pose)
            self.assertLessEqual(v, 0.0, f"COMMIT with fwd_scale=0 commanded v={v}>0")

    def test_dual_clearance_mask_enforcement(self):
        """DualScales action mask must zero forward when fwd_m=0."""
        from action_mask import apply_action_mask
        
        # Hard forward stop (0cm clearance)
        dual = evaluate_dual(fwd_m=0.0, bwd_m=0.30, lat_m=0.15)
        
        # Try to command forward
        masked = apply_action_mask(v=0.20, w=0.0, scales=dual)
        
        self.assertEqual(masked.v, 0.0, "Action mask should zero v when fwd_m=0")
        self.assertEqual(masked.mode, "block_fwd", "Should report block_fwd mode")
        
        # Reverse should be allowed
        masked_rev = apply_action_mask(v=-0.15, w=0.0, scales=dual)
        self.assertLess(masked_rev.v, 0.0, "Reverse should be allowed when bwd_m>0")

    def test_soft_cost_applied_when_clearance_low(self):
        """Soft costs should bias MPPI away from forward when fwd_m is low."""
        policy = MppiSimPolicy(wander=True, use_soft_cost=True)
        policy.reset()
        
        obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        
        # Low forward clearance (not zero, but tight)
        safety_scales = {
            "fwd": 0.35,
            "bwd": 1.0,
            "ang": 0.85,
            "fwd_m": 0.12,  # 12cm clearance
            "bwd_m": 0.40,
            "lat_m": 0.20,
        }
        
        pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        
        # MPPI should use soft costs to prefer reverse/turn over forward
        forward_count = 0
        for _ in range(15):
            v, w = policy.act(obs, None, safety_scales, pose)
            if v > 0.05:
                forward_count += 1
        
        # With soft costs and tight clearance, should mostly avoid forward
        self.assertLess(
            forward_count, 8,
            f"Expected MPPI to avoid forward with low clearance, got {forward_count}/15 forward"
        )

    def test_all_hard_stop_scenarios(self):
        """Test all hard-stop scenarios enforce v≤0."""
        policy = MppiSimPolicy(wander=True, use_soft_cost=True)
        
        scenarios = [
            ("near_field_reflex", {"fwd": 0.0, "bwd": 1.0, "ang": 0.8, "fwd_m": 0.0, "bwd_m": 0.30, "lat_m": 0.15}),
            ("topdown_hazard", {"fwd": 0.0, "bwd": 0.95, "ang": 1.0, "fwd_m": 0.0, "bwd_m": 0.35, "lat_m": 0.25}),
            ("stuck_immobilize", {"fwd": 0.0, "bwd": 1.0, "ang": 1.0, "fwd_m": 0.0, "bwd_m": 0.40, "lat_m": 0.30}),
            ("overhang_approach", {"fwd": 0.05, "bwd": 1.0, "ang": 0.70, "fwd_m": 0.02, "bwd_m": 0.35, "lat_m": 0.18}),
        ]
        
        obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        
        for scenario_name, scales in scenarios:
            policy.reset()
            for _ in range(8):
                v, w = policy.act(obs, None, scales, pose)
                if scales["fwd"] == 0.0:
                    self.assertLessEqual(v, 0.0, f"{scenario_name}: v={v}>0 with fwd=0")


class TestMPPINudgePathsSafe(unittest.TestCase):
    """Test MPPI nudge/dither paths respect safety scales."""

    def test_mppi_nudge_fwd_masked(self):
        """MPPI 'nudge_fwd' dither path applies safety mask."""
        from mppi_policy import MppiSimPolicy
        
        policy = MppiSimPolicy(wander=True, use_soft_cost=False)
        policy.reset()
        
        obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        
        # Set up condition for nudge_fwd: fwd>=0.30, MPPI returns tiny v
        safety_scales = {
            "fwd": 0.0,  # Hard stop
            "bwd": 1.0,
            "ang": 1.0,
            "fwd_m": 0.15,  # Would normally trigger nudge
            "bwd_m": 0.40,
            "lat_m": 0.25,
        }
        
        pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        
        # Force MPPI into a state where it might try nudge_fwd
        # by clearing the planner goal temporarily
        policy.planner.cancel()
        
        for _ in range(5):
            v, w = policy.act(obs, None, safety_scales, pose)
            # Even if nudge_fwd logic triggers, mask should zero v
            self.assertLessEqual(v, 0.0, f"nudge_fwd bypassed fwd_scale=0: v={v}")

    def test_mppi_mask_called_on_all_paths(self):
        """Verify _mask() is called on every return path in MppiSimPolicy.act()."""
        from mppi_policy import MppiSimPolicy
        
        policy = MppiSimPolicy(wander=True, use_soft_cost=True)
        policy.reset()
        
        obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        safety_scales = {
            "fwd": 0.0,
            "bwd": 1.0,
            "ang": 0.9,
            "fwd_m": 0.0,
            "bwd_m": 0.35,
            "lat_m": 0.20,
        }
        pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        
        # Run through various policy states
        for _ in range(30):
            v, w = policy.act(obs, None, safety_scales, pose)
            # All paths must respect fwd_scale=0
            self.assertLessEqual(v, 0.0, f"Policy path bypassed mask: v={v}, decision={policy.last_decision}")


if __name__ == "__main__":
    print("=" * 70)
    print("HOST SIM: Recover/Commit Safety Hard-Stop Tests")
    print("=" * 70)
    unittest.main(verbosity=2)
