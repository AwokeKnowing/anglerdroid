#!/usr/bin/env python3
"""
test_loop_timing.py - Unit tests for loop timing and budget shedding.

Run with: python3 test_loop_timing.py
"""

import time
import unittest

from loop_timing import FrameBudget


class TestFrameBudget(unittest.TestCase):
    """Test frame budget and shedding logic."""
    
    def test_basic_timing(self):
        """Test basic stage timing."""
        budget = FrameBudget(budget_ms=33.3, shed_threshold=0.85)
        budget.reset_frame()
        
        # Simulate a fast stage
        with budget.stage("atlas"):
            time.sleep(0.001)  # 1ms
        
        stats = budget.get_stats()
        self.assertGreater(stats["stage_times"]["atlas"], 0.5)
        self.assertLess(stats["stage_times"]["atlas"], 5.0)
        self.assertEqual(stats["budget_ms"], 33.3)
    
    def test_critical_stages_always_run(self):
        """Critical stages should always run regardless of budget."""
        budget = FrameBudget(budget_ms=10.0, shed_threshold=0.5)
        budget.reset_frame()
        
        # Burn through the budget with a critical stage
        with budget.stage("atlas"):
            time.sleep(0.008)  # 8ms > 50% of 10ms
        
        # Critical stages should still run
        self.assertTrue(budget.should_run("atlas"))
        self.assertTrue(budget.should_run("safety"))
        self.assertTrue(budget.should_run("planner"))
    
    def test_droppable_stages_shed_when_over_budget(self):
        """Droppable stages should be skipped when budget is exceeded."""
        budget = FrameBudget(budget_ms=10.0, shed_threshold=0.5)  # 5ms threshold
        budget.reset_frame()
        
        # Burn through the budget
        with budget.stage("atlas"):
            time.sleep(0.006)  # 6ms > 5ms threshold
        
        # Droppable stages should be skipped
        self.assertFalse(budget.should_run("rerun"))
        self.assertFalse(budget.should_run("tool_calls"))
        
        # Check shed count incremented
        self.assertEqual(budget.shed_counts.get("rerun", 0), 1)
    
    def test_droppable_stages_run_when_budget_ok(self):
        """Droppable stages should run when budget is available."""
        budget = FrameBudget(budget_ms=33.3, shed_threshold=0.85)
        budget.reset_frame()
        
        # Small work - under budget
        with budget.stage("atlas"):
            time.sleep(0.002)  # 2ms << 28.3ms threshold
        
        # Droppable stages should run
        self.assertTrue(budget.should_run("rerun"))
        self.assertTrue(budget.should_run("tool_calls"))
    
    def test_accumulated_time_tracking(self):
        """Test accumulated time across multiple stages."""
        budget = FrameBudget(budget_ms=33.3, shed_threshold=0.85)
        budget.reset_frame()
        
        with budget.stage("atlas"):
            time.sleep(0.005)  # ~5ms
        
        with budget.stage("safety"):
            time.sleep(0.003)  # ~3ms
        
        stats = budget.get_stats()
        # Should be ~8ms total
        self.assertGreater(stats["accumulated_ms"], 7.0)
        self.assertLess(stats["accumulated_ms"], 15.0)
    
    def test_budget_exceeded_flag(self):
        """Test budget_exceeded() flag."""
        budget = FrameBudget(budget_ms=10.0, shed_threshold=0.5)
        budget.reset_frame()
        
        self.assertFalse(budget.budget_exceeded())
        
        with budget.stage("atlas"):
            time.sleep(0.006)  # 6ms > 5ms threshold
        
        self.assertTrue(budget.budget_exceeded())
    
    def test_frame_reset(self):
        """Test frame reset clears accumulated time."""
        budget = FrameBudget(budget_ms=33.3, shed_threshold=0.85)
        budget.reset_frame()
        
        with budget.stage("atlas"):
            time.sleep(0.005)
        
        self.assertGreater(budget.accumulated_ms, 0)
        
        # Reset should clear accumulated time
        budget.reset_frame()
        self.assertEqual(budget.accumulated_ms, 0.0)
        self.assertEqual(len(budget.stage_times), 0)
    
    def test_lifetime_stats(self):
        """Test lifetime statistics tracking."""
        budget = FrameBudget(budget_ms=33.3, shed_threshold=0.85)
        
        # Frame 1
        budget.reset_frame()
        with budget.stage("atlas"):
            time.sleep(0.001)
        
        # Frame 2
        budget.reset_frame()
        with budget.stage("atlas"):
            time.sleep(0.001)
        
        # Frame 3 - over budget
        budget.reset_frame()
        with budget.stage("atlas"):
            time.sleep(0.030)  # burn budget
        budget.should_run("rerun")  # should be shed
        
        stats = budget.get_lifetime_stats()
        self.assertEqual(stats["stage_counts"]["atlas"], 3)
        self.assertEqual(stats["shed_counts"].get("rerun", 0), 1)
    
    def test_unknown_stage_is_droppable(self):
        """Unknown stages should be treated as droppable."""
        budget = FrameBudget(budget_ms=10.0, shed_threshold=0.5)
        budget.reset_frame()
        
        # Burn budget
        with budget.stage("atlas"):
            time.sleep(0.006)
        
        # Unknown stage should be droppable
        self.assertFalse(budget.should_run("unknown_stage"))


class TestScenarios(unittest.TestCase):
    """Test realistic loop scenarios."""
    
    def test_typical_light_frame(self):
        """Test a typical frame with light work - nothing should be shed."""
        budget = FrameBudget(budget_ms=33.3, shed_threshold=0.85)
        budget.reset_frame()
        
        # Simulate typical stages
        with budget.stage("atlas"):
            time.sleep(0.005)  # 5ms
        
        self.assertTrue(budget.should_run("rerun"))
        with budget.stage("rerun"):
            time.sleep(0.008)  # 8ms
        
        with budget.stage("safety"):
            time.sleep(0.002)  # 2ms
        
        with budget.stage("planner"):
            time.sleep(0.003)  # 3ms
        
        self.assertTrue(budget.should_run("tool_calls"))
        
        stats = budget.get_stats()
        # Total ~18ms, well under 33.3ms
        self.assertLess(stats["accumulated_ms"], 25.0)
        self.assertEqual(budget.shed_counts.get("rerun", 0), 0)
    
    def test_heavy_frame_sheds_rerun(self):
        """Test a heavy frame that should shed rerun."""
        budget = FrameBudget(budget_ms=33.3, shed_threshold=0.85)
        budget.reset_frame()
        
        # Heavy atlas processing
        with budget.stage("atlas"):
            time.sleep(0.029)  # 29ms > 28.3ms threshold
        
        # Rerun should be shed
        self.assertFalse(budget.should_run("rerun"))
        self.assertEqual(budget.shed_counts["rerun"], 1)
        
        # Critical path should still run
        with budget.stage("safety"):
            time.sleep(0.001)
        
        with budget.stage("planner"):
            time.sleep(0.001)
        
        # Total accumulated should be around 31ms (under budget)
        stats = budget.get_stats()
        self.assertGreater(stats["accumulated_ms"], 28.0)
        self.assertLess(stats["accumulated_ms"], 35.0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFrameBudget))
    suite.addTests(loader.loadTestsFromTestCase(TestScenarios))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
