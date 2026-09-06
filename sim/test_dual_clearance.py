"""Unit tests for soft/hard DualScales (no hardware).

Run: python -m sim.test_dual_clearance
"""
from __future__ import annotations

import sys
import unittest

from sim.dual_clearance import DualScales, clearance_scale, evaluate


def _approx(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps


class TestDualClearance(unittest.TestCase):
    def test_far_open(self):
        s = evaluate(0.5, 0.5, 0.5)
        self.assertTrue(_approx(s.soft_fwd, 1.0) and _approx(s.hard_fwd, 1.0))
        self.assertTrue(_approx(s.soft_bwd, 1.0) and _approx(s.hard_bwd, 1.0))
        self.assertTrue(_approx(s.soft_ang, 1.0) and _approx(s.hard_ang, 1.0))
        self.assertFalse(s.soft_blocked)
        self.assertTrue(s.hard_free)
        self.assertFalse(s.squeeze)
        self.assertTrue(_approx(s.soft_cost_fwd(), 0.0))

    def test_soft_band_only(self):
        s = evaluate(0.25, 0.5, 0.5)
        self.assertTrue(_approx(s.soft_fwd, 0.4))
        self.assertLess(s.soft_fwd, 1.0)
        self.assertGreaterEqual(s.hard_fwd, 0.99)
        self.assertTrue(s.soft_blocked)
        self.assertTrue(s.hard_free)
        self.assertTrue(s.squeeze is True)
        self.assertTrue(_approx(s.soft_cost_fwd(), 0.6))

    def test_hard_throttle(self):
        s = evaluate(0.08, 0.5, 0.5)
        self.assertTrue(0.0 < s.hard_fwd < 1.0)
        self.assertTrue(_approx(s.hard_fwd, 0.44))
        self.assertEqual(s.soft_fwd, 0.0)
        self.assertTrue(s.soft_blocked)
        self.assertTrue(s.hard_free)
        self.assertTrue(s.squeeze)

    def test_hard_contact(self):
        s = evaluate(0.01, 0.5, 0.5)
        self.assertEqual(s.hard_fwd, 0.0)
        self.assertTrue(s.soft_blocked)
        self.assertFalse(s.hard_free)
        self.assertFalse(s.squeeze)

    def test_pinch_nose_spin_out(self):
        s = evaluate(0.01, 0.5, 0.2, ang_floor=0.35)
        self.assertEqual(s.hard_fwd, 0.0)
        self.assertEqual(s.hard_bwd, 1.0)
        self.assertGreaterEqual(s.hard_ang, 0.35)
        s3 = evaluate(0.01, 0.5, 0.04, ang_floor=0.35)
        self.assertEqual(s3.hard_fwd, 0.0)
        self.assertGreater(s3.hard_bwd, 0.3)
        self.assertTrue(_approx(s3.hard_ang, 0.35))

    def test_laterally_crushed_no_ang_floor(self):
        s = evaluate(0.01, 0.5, 0.02, ang_floor=0.35, hard_zero=0.025)
        self.assertEqual(s.hard_fwd, 0.0)
        self.assertEqual(s.hard_ang, 0.0)
        s_eq = evaluate(0.01, 0.5, 0.025, ang_floor=0.35, hard_zero=0.025)
        self.assertEqual(s_eq.hard_ang, 0.0)

    def test_monotonic(self):
        clears = [0.0, 0.01, 0.025, 0.05, 0.08, 0.15, 0.25, 0.40, 0.5, 1.0]
        prev_sf = prev_hf = prev_sb = prev_hb = prev_sa = -1.0
        for c in clears:
            s = evaluate(c, c, c)
            self.assertGreaterEqual(s.soft_fwd, prev_sf - 1e-12)
            self.assertGreaterEqual(s.hard_fwd, prev_hf - 1e-12)
            self.assertGreaterEqual(s.soft_bwd, prev_sb - 1e-12)
            self.assertGreaterEqual(s.hard_bwd, prev_hb - 1e-12)
            self.assertGreaterEqual(s.soft_ang, prev_sa - 1e-12)
            prev_sf, prev_hf = s.soft_fwd, s.hard_fwd
            prev_sb, prev_hb = s.soft_bwd, s.hard_bwd
            prev_sa = s.soft_ang
        prev_ha = -1.0
        for c in clears:
            s = evaluate(1.0, 1.0, c)
            self.assertGreaterEqual(s.hard_ang, prev_ha - 1e-12)
            prev_ha = s.hard_ang
        prev = -1.0
        for c in clears:
            v = clearance_scale(c, 0.025, 0.15)
            self.assertGreaterEqual(v, prev - 1e-12)
            prev = v

    def test_hard_zero_never_weakened_at_contact(self):
        s = evaluate(0.0, 1.0, 1.0)
        self.assertEqual(s.hard_fwd, 0.0)
        s2 = evaluate(0.024, 1.0, 1.0)
        self.assertEqual(s2.hard_fwd, 0.0)


def run():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run())
