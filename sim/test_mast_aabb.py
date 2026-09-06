"""Unit tests for Box3D AABB mast scenarios (no hardware).

Run: python -m sim.test_mast_aabb
"""
from __future__ import annotations

import sys
import unittest

from sim.mast_aabb import MAST_CLEAR_M, MAST_TOP_Z_M, assert_all_green, run_all


class TestMastAabb(unittest.TestCase):
    def test_mast_taller_than_clearance(self):
        self.assertGreater(MAST_TOP_Z_M, MAST_CLEAR_M)
        self.assertEqual(MAST_TOP_Z_M, 0.16 + 0.70)

    def test_aabb_scenarios_green(self):
        n = assert_all_green()
        self.assertEqual(n, 5)
        results = run_all()
        by_name = {r["name"]: r for r in results}
        self.assertTrue(by_name["table_under"]["mast_hit"])
        self.assertFalse(by_name["table_under"]["chassis_hit"])
        self.assertFalse(by_name["low_bumper"]["mast_hit"])
        self.assertTrue(by_name["low_bumper"]["chassis_hit"])
        self.assertTrue(by_name["doorway_lintel"]["mast_hit"])
        self.assertFalse(by_name["open_floor"]["mast_hit"])
        self.assertFalse(by_name["high_shelf"]["mast_hit"])


def run():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run())
