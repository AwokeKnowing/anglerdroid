"""Unit tests for DualScales → MPPI action mask (no hardware).

Run: python -m sim.test_action_mask
"""
from __future__ import annotations

import sys
import unittest

from sim.action_mask import (
    action_feasible,
    apply_action_mask,
    classify_mode,
    mask_from_clearances,
    soft_cost_bonus,
    soft_sample_cost,
)
from sim.dual_clearance import evaluate


def _approx(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps


class TestActionMask(unittest.TestCase):
    def test_open_passthrough(self):
        s = evaluate(0.5, 0.5, 0.5)
        m = apply_action_mask(0.25, 0.4, s)
        self.assertEqual(m.mode, "open")
        self.assertTrue(_approx(m.v, 0.25) and _approx(m.w, 0.4))
        self.assertTrue(action_feasible(0.25, 0.4, s))

    def test_squeeze_slows_forward(self):
        s = evaluate(0.25, 0.5, 0.5)
        self.assertTrue(s.squeeze)
        m = apply_action_mask(0.25, 0.5, s, squeeze_v_scale=0.35)
        self.assertEqual(m.mode, "squeeze")
        self.assertTrue(_approx(m.v, 0.25 * 0.35))
        self.assertTrue(_approx(m.w, 0.5))
        self.assertGreater(soft_cost_bonus(s), 0.0)

    def test_hard_throttle_scales(self):
        s = evaluate(0.08, 0.5, 0.5)
        self.assertTrue(_approx(s.hard_fwd, 0.44))
        m = apply_action_mask(0.25, 0.0, s, squeeze_v_scale=0.35)
        self.assertEqual(m.mode, "squeeze")
        self.assertTrue(_approx(m.v, 0.25 * 0.44 * 0.35))

    def test_recover_never_forward(self):
        s = evaluate(0.01, 0.5, 0.2)
        self.assertEqual(s.hard_fwd, 0.0)
        m_fwd = apply_action_mask(0.25, 0.3, s)
        self.assertEqual(m_fwd.mode, "recover")
        self.assertEqual(m_fwd.v, 0.0)
        self.assertGreater(m_fwd.w, 0.0)
        self.assertFalse(action_feasible(0.25, 0.0, s))
        m_rev = apply_action_mask(-0.2, 0.0, s)
        self.assertEqual(m_rev.mode, "recover")
        self.assertLess(m_rev.v, 0.0)
        self.assertTrue(action_feasible(-0.2, 0.0, s))

    def test_halt_laterally_crushed(self):
        s = evaluate(0.0, 0.0, 0.0)
        m = apply_action_mask(0.25, 0.5, s)
        self.assertEqual(m.mode, "halt")
        self.assertEqual(m.v, 0.0)
        self.assertEqual(m.w, 0.0)
        self.assertFalse(action_feasible(0.1, 0.0, s))
        self.assertFalse(action_feasible(0.0, 0.2, s))

    def test_hard_zero_never_weakened(self):
        m = mask_from_clearances(0.25, 0.5, 0.0, 1.0, 1.0)
        self.assertEqual(m.mode, "recover")
        self.assertEqual(m.v, 0.0)
        self.assertEqual(m.scales.hard_fwd, 0.0)

    def test_spin_out_ang_preserved(self):
        s = evaluate(0.01, 0.5, 0.04, ang_floor=0.35)
        self.assertTrue(_approx(s.hard_ang, 0.35))
        m = apply_action_mask(0.2, 0.8, s)
        self.assertEqual(m.mode, "recover")
        self.assertEqual(m.v, 0.0)
        self.assertTrue(_approx(m.w, 0.8 * 0.35))

    def test_feasible_yaw_blocked(self):
        s = evaluate(0.5, 0.5, 0.0)
        self.assertEqual(s.hard_ang, 0.0)
        self.assertEqual(s.hard_fwd, 1.0)
        self.assertFalse(action_feasible(0.0, 0.5, s))
        self.assertTrue(action_feasible(0.1, 0.0, s))
        m = apply_action_mask(0.1, 0.5, s)
        self.assertEqual(m.w, 0.0)
        self.assertTrue(_approx(m.v, 0.1))

    def test_classify_modes(self):
        self.assertEqual(classify_mode(evaluate(1.0, 1.0, 1.0)), "open")
        self.assertEqual(classify_mode(evaluate(0.25, 0.5, 0.5)), "squeeze")
        self.assertEqual(classify_mode(evaluate(0.01, 0.5, 0.2)), "recover")
        self.assertEqual(classify_mode(evaluate(0.0, 0.0, 0.0)), "halt")

    def test_soft_sample_open_near_zero(self):
        s = evaluate(0.5, 0.5, 0.5)
        self.assertEqual(s.soft_cost_fwd(), 0.0)
        self.assertTrue(_approx(soft_sample_cost(0.2, 0.0, s), 0.0))
        self.assertTrue(_approx(soft_sample_cost(0.2, 0.3, s), 0.0))

    def test_soft_sample_fwd_blocked_prefers_reverse_spin(self):
        # Squeeze band: soft-blocked fwd, bwd/lat clear → forward costs more.
        s = evaluate(0.25, 0.5, 0.5)
        self.assertTrue(s.squeeze)
        self.assertGreater(s.soft_cost_fwd(), 0.0)
        self.assertEqual(s.soft_cost_bwd(), 0.0)
        self.assertEqual(s.soft_cost_ang(), 0.0)
        c_fwd = soft_sample_cost(0.2, 0.0, s)
        c_rev = soft_sample_cost(-0.2, 0.0, s)
        c_spin = soft_sample_cost(0.0, 0.5, s)
        self.assertGreater(c_fwd, c_rev)
        self.assertGreater(c_fwd, c_spin)
        self.assertTrue(_approx(c_rev, 0.0))
        self.assertTrue(_approx(c_spin, 0.0))
        # Directional: same scales, aggregate bonus alone would be identical.
        self.assertGreater(soft_cost_bonus(s), 0.0)

    def test_soft_sample_recover_still_costs_but_hard_unchanged(self):
        s = evaluate(0.01, 0.5, 0.2)
        self.assertEqual(s.hard_fwd, 0.0)
        self.assertEqual(classify_mode(s), "recover")
        # Soft prefer still has cost in recover band (planner hint).
        self.assertGreater(soft_sample_cost(0.2, 0.0, s), 0.0)
        # Hard mask still absolute — soft cost must not weaken it.
        m = apply_action_mask(0.25, 0.3, s)
        self.assertEqual(m.mode, "recover")
        self.assertEqual(m.v, 0.0)
        self.assertEqual(m.scales.hard_fwd, 0.0)
        self.assertFalse(action_feasible(0.25, 0.0, s))


def run():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run())
