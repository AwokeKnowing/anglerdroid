#!/usr/bin/env python3
"""
Unit tests for CAN bus failure resilience.

Tests that CanOperationError during motion commands:
1. Does not crash the process
2. Stops wheels (best effort)
3. Disarms the wheelbase cleanly
4. Subsequent motion commands become no-ops

These tests use mocking to simulate CAN bus failures without requiring
actual hardware or a live CAN interface.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import can - tests will be skipped if not available
try:
    import can
    from can.exceptions import CanOperationError
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False
    # Mock CanOperationError for test definitions
    class CanOperationError(Exception):
        pass

@unittest.skipUnless(CAN_AVAILABLE, "python-can not available")
class TestCANResilience(unittest.TestCase):
    """Test CAN bus failure handling in WheelBase."""
    
    def setUp(self):
        """Set up test fixtures with mocked CAN bus."""
        # Mock the CAN bus and ODrive axes
        self.mock_bus = Mock(spec=can.BusABC)
        self.mock_left = Mock()
        self.mock_right = Mock()
        
        # Prevent actual CAN initialization
        self.can_patcher = patch('wheelbase.can.Bus')
        self.mock_can_bus_cls = self.can_patcher.start()
        self.mock_can_bus_cls.return_value = self.mock_bus
        
        # Mock subprocess for CAN bring-up
        self.subprocess_patcher = patch('wheelbase.subprocess.run')
        self.mock_subprocess = self.subprocess_patcher.start()
        self.mock_subprocess.return_value = Mock(stdout="state UP")
        
        # Mock ODriveAxisCAN
        self.odrive_patcher = patch('wheelbase.ODriveAxisCAN')
        self.mock_odrive_cls = self.odrive_patcher.start()
        self.mock_odrive_cls.side_effect = [self.mock_left, self.mock_right]
        
        # Mock time.sleep to speed up tests
        self.sleep_patcher = patch('wheelbase.time.sleep')
        self.mock_sleep = self.sleep_patcher.start()
        
    def tearDown(self):
        """Clean up patches."""
        self.can_patcher.stop()
        self.subprocess_patcher.stop()
        self.odrive_patcher.stop()
        self.sleep_patcher.stop()
        
    def test_set_wheel_vels_can_error(self):
        """Test that CanOperationError during set_wheel_vels is caught."""
        from wheelbase import WheelBase
        
        # Create wheelbase instance
        wb = WheelBase(can_interface="can0")
        
        # Simulate CAN failure on set_velocity
        self.mock_left.set_velocity.side_effect = CanOperationError("No such device")
        
        # Attempt to set velocities - should not crash
        wb._is_closed_loop = True  # Skip re-engage path
        wb._is_idle = False
        
        with self.assertRaises(RuntimeError) as ctx:
            wb.set_wheel_vels(0.5, 0.5)
        
        self.assertIn("CAN bus failed", str(ctx.exception))
        
        # Wheelbase should be marked as failed
        self.assertTrue(wb._can_bus_failed)
        
        # Subsequent calls should be no-ops (no exception)
        wb.set_wheel_vels(0.3, 0.3)  # Should not raise
        
    def test_twist_can_error(self):
        """Test that CanOperationError during twist is caught."""
        from wheelbase import WheelBase
        
        # Create wheelbase instance
        wb = WheelBase(can_interface="can0")
        
        # Simulate CAN failure on set_velocity
        self.mock_left.set_velocity.side_effect = CanOperationError("No such device")
        
        # Attempt twist - should not crash
        wb._is_closed_loop = True
        wb._is_idle = False
        
        with self.assertRaises(RuntimeError) as ctx:
            wb.twist(0.3, 0.0)
        
        self.assertIn("CAN bus failed", str(ctx.exception))
        self.assertTrue(wb._can_bus_failed)
        
        # Subsequent calls should be no-ops
        wb.twist(0.2, 0.1)  # Should not raise
        
    def test_twist_for_can_error(self):
        """Test that CanOperationError during twist_for loop is handled."""
        from wheelbase import WheelBase
        
        # Create wheelbase instance
        wb = WheelBase(can_interface="can0")
        
        # Start a twist_for command
        wb.twist_for(0.3, 0.0, duration_secs=2.0, ramp_in_secs=0.0, ramp_out_secs=0.0)
        
        # Wait a bit for the loop to start
        time.sleep(0.1)
        
        # Simulate CAN failure on the next set_velocity
        self.mock_left.set_velocity.side_effect = CanOperationError("Network is down")
        
        # Wait for the loop to encounter the error
        time.sleep(0.2)
        
        # Wheelbase should be marked as failed
        self.assertTrue(wb._can_bus_failed)
        
        # twist_for should have been canceled
        self.assertFalse(wb.is_twist_for_active())
        
        # Subsequent motion commands should be no-ops
        wb.twist(0.1, 0.0)  # Should not raise
        
    def test_can_error_during_reengage(self):
        """Test CanOperationError during motor re-engagement from idle."""
        from wheelbase import WheelBase
        
        # Create wheelbase instance
        wb = WheelBase(can_interface="can0")
        
        # Put wheelbase in idle state
        wb._is_closed_loop = False
        wb._is_idle = True
        
        # Simulate CAN failure during re-engage
        self.mock_left.clear_errors.side_effect = CanOperationError("Bus error")
        
        # Attempt to send non-zero velocity (triggers re-engage)
        with self.assertRaises(RuntimeError) as ctx:
            wb.set_wheel_vels(0.5, 0.5)
        
        self.assertIn("CAN bus failed", str(ctx.exception))
        self.assertIn("re-engage", str(ctx.exception))
        self.assertTrue(wb._can_bus_failed)
        
    def test_best_effort_stop_on_can_failure(self):
        """Test that best-effort stop is attempted on CAN failure."""
        from wheelbase import WheelBase
        
        # Create wheelbase instance
        wb = WheelBase(can_interface="can0")
        
        # Track calls to stop commands
        stop_calls = []
        
        def track_set_velocity(vel):
            stop_calls.append(('set_velocity', vel))
            if len(stop_calls) > 1:  # Fail after first successful call
                raise CanOperationError("Bus gone")
        
        self.mock_left.set_velocity.side_effect = track_set_velocity
        
        # Trigger a CAN failure
        wb._is_closed_loop = True
        wb._is_idle = False
        
        with self.assertRaises(RuntimeError):
            wb.set_wheel_vels(0.5, 0.5)
        
        # _handle_can_failure should have tried to stop (may fail)
        self.assertTrue(wb._can_bus_failed)
        self.assertTrue(wb._is_idle)
        self.assertFalse(wb._is_closed_loop)
        
    def test_watchdog_feeder_handles_can_error(self):
        """Test that watchdog feeder thread handles CAN errors gracefully."""
        from wheelbase import WheelBase
        
        # Create wheelbase instance
        wb = WheelBase(can_interface="can0")
        
        # Put in closed-loop state
        wb._is_closed_loop = True
        wb._is_idle = False
        
        # Simulate CAN error in feed_watchdog
        self.mock_left.feed_watchdog.side_effect = CanOperationError("Connection lost")
        
        # Let the watchdog feeder run once
        time.sleep(0.6)
        
        # Should mark as failed without crashing
        self.assertTrue(wb._can_bus_failed)
        
    def test_errno_6_osserror_simulation(self):
        """Test handling of OSError errno 6 (No such device or address)."""
        from wheelbase import WheelBase
        
        # Create wheelbase instance
        wb = WheelBase(can_interface="can0")
        
        # Simulate OSError errno 6 wrapped in CanOperationError
        error = OSError(6, "No such device or address")
        self.mock_left.set_velocity.side_effect = CanOperationError("Bus send failed", error)
        
        # Should handle gracefully
        wb._is_closed_loop = True
        wb._is_idle = False
        
        with self.assertRaises(RuntimeError) as ctx:
            wb.set_wheel_vels(0.4, 0.4)
        
        self.assertIn("CAN bus failed", str(ctx.exception))
        self.assertTrue(wb._can_bus_failed)


class TestMainLoopResilience(unittest.TestCase):
    """Test that main loop handles CAN failures gracefully."""
    
    def test_main_catches_runtime_error(self):
        """Test that main loop catches RuntimeError from CAN failures."""
        # This is a code inspection test - verify the pattern exists in main.py
        with open('src/main.py', 'r') as f:
            main_content = f.read()
        
        # Check that main.py catches RuntimeError for CAN failures
        self.assertIn('except RuntimeError as e:', main_content)
        self.assertIn('if "CAN bus failed" in str(e):', main_content)
        self.assertIn('shutting down', main_content)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
