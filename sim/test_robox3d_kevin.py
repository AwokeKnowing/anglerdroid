"""Focused tests for thin Kevin URDF + robox3d mast/table contact.

Skips cleanly if robox3d is not installed. Run:
  .venv/bin/python -m sim.test_robox3d_kevin
"""

from __future__ import annotations

import sys
import unittest

try:
    import robox3d  # noqa: F401

    HAS_ROBOX3D = True
except ImportError:  # pragma: no cover
    HAS_ROBOX3D = False


@unittest.skipUnless(HAS_ROBOX3D, "robox3d not installed")
class TestRobox3dKevin(unittest.TestCase):
    def test_thin_urdf_exists(self):
        from sim.robox3d_kevin import thin_urdf_path

        p = thin_urdf_path()
        self.assertTrue(p.is_file(), f"missing {p}")

    def test_load_urdf_smoke(self):
        from sim.robox3d_kevin import load_smoke

        name = load_smoke()
        self.assertTrue(name)

    def test_mast_under_table_contact(self):
        from sim.robox3d_kevin import POSE_UNDER_TABLE, mast_table_contact

        self.assertTrue(
            mast_table_contact(POSE_UNDER_TABLE),
            "expected mast/table contact when Kevin under table AABB",
        )

    def test_mast_clear_of_table(self):
        from sim.robox3d_kevin import POSE_CLEAR, mast_table_contact

        self.assertFalse(
            mast_table_contact(POSE_CLEAR),
            "expected no contact when Kevin clear of table",
        )

    def test_verify_bundle(self):
        from sim.robox3d_kevin import verify_mast_vs_table

        result = verify_mast_vs_table()
        self.assertTrue(result["ok"], result)


def run():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run())
