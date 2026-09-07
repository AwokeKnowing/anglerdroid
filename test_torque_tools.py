"""test_torque_tools.py — Unit tests for ODrive torque control tools.

Tests torque commands, geometric conversions, and safety limits with mock axes.
"""

import math
import os
import sys

import pytest

# Import torque_tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torque_tools


@pytest.fixture
def mock_axes():
    """Create mock ODrive axes for testing."""
    left = torque_tools.MockODriveAxis("Left")
    right = torque_tools.MockODriveAxis("Right")
    return left, right


@pytest.fixture
def controller(mock_axes):
    """Create TorqueController with mock axes."""
    left, right = mock_axes
    return torque_tools.TorqueController(left, right)


def test_torque_controller_init(controller):
    """Test controller initialization."""
    assert controller is not None


def test_torque_controller_stop(controller, mock_axes):
    """Test emergency stop."""
    left, right = mock_axes
    
    # Set some torques
    left._torque = 0.3
    right._torque = 0.3
    
    # Stop
    controller.stop()
    
    # Should be zeroed
    assert left.get_torque() == 0.0
    assert right.get_torque() == 0.0


def test_torque_controller_set_wheel_torques(controller, mock_axes):
    """Test setting wheel torques with safety clamping."""
    left, right = mock_axes
    
    # Normal torques
    controller._set_wheel_torques(0.3, -0.2)
    assert abs(left.get_torque() - 0.3) < 1e-6
    assert abs(right.get_torque() + 0.2) < 1e-6
    
    # Clamping: exceed MAX_TORQUE_NM
    controller._set_wheel_torques(1.0, -1.0)
    assert abs(left.get_torque() - torque_tools.MAX_TORQUE_NM) < 1e-6
    assert abs(right.get_torque() + torque_tools.MAX_TORQUE_NM) < 1e-6


def test_torque_controller_micro_move_straight(controller, mock_axes):
    """Test straight line micro-move."""
    left, right = mock_axes
    
    # Forward move
    result = controller.micro_move_straight(0.05, force_nm=0.3, max_duration_s=0.1)
    
    assert result is True
    # After move, torques should be zeroed (stop called)
    assert left.get_torque() == 0.0
    assert right.get_torque() == 0.0


def test_torque_controller_micro_move_straight_backward(controller, mock_axes):
    """Test backward micro-move."""
    left, right = mock_axes
    
    # Backward move
    result = controller.micro_move_straight(-0.05, force_nm=0.3, max_duration_s=0.1)
    
    assert result is True


def test_torque_controller_micro_rotate(controller, mock_axes):
    """Test in-place rotation."""
    left, right = mock_axes
    
    # CCW rotation (positive angle)
    result = controller.micro_rotate(math.radians(15), torque_nm=0.3, max_duration_s=0.1)
    
    assert result is True
    # After rotate, torques should be zeroed
    assert left.get_torque() == 0.0
    assert right.get_torque() == 0.0


def test_torque_controller_micro_rotate_cw(controller, mock_axes):
    """Test clockwise rotation."""
    left, right = mock_axes
    
    # CW rotation (negative angle)
    result = controller.micro_rotate(math.radians(-15), torque_nm=0.3, max_duration_s=0.1)
    
    assert result is True


def test_torque_controller_force_limited_contact(controller, mock_axes):
    """Test force-limited contact task."""
    left, right = mock_axes
    
    # Forward contact
    result = controller.force_limited_contact("forward", max_force_nm=0.2, duration_s=0.1)
    
    assert result["success"] is True
    assert "contact_detected" in result
    assert "duration_s" in result


def test_torque_controller_force_limited_contact_directions(controller):
    """Test force-limited contact in all directions."""
    directions = ["forward", "backward", "left", "right"]
    
    for direction in directions:
        result = controller.force_limited_contact(direction, max_force_nm=0.2, duration_s=0.05)
        assert result["success"] is True


def test_torque_controller_force_limited_contact_invalid_direction(controller):
    """Test invalid direction raises ValueError."""
    with pytest.raises(ValueError):
        controller.force_limited_contact("up", max_force_nm=0.2, duration_s=0.1)


def test_torque_controller_impedance_control_stub(controller):
    """Test impedance control stub (not implemented)."""
    with pytest.raises(NotImplementedError):
        controller.impedance_control_stub(0.1, stiffness=10.0, damping=2.0)


def test_torque_from_force_linear():
    """Test linear force to torque conversion."""
    force_n = 10.0
    torque_nm = torque_tools.torque_from_force_linear(force_n)
    
    # torque = force * wheel_radius
    expected = force_n * torque_tools.WHEEL_RADIUS_M
    assert abs(torque_nm - expected) < 1e-6


def test_torque_from_force_angular():
    """Test angular force to torque conversion."""
    force_n = 5.0
    torque_nm = torque_tools.torque_from_force_angular(force_n)
    
    # torque = force * (wheelbase/2) * wheel_radius
    expected = force_n * (torque_tools.WHEELBASE_M / 2.0) * torque_tools.WHEEL_RADIUS_M
    assert abs(torque_nm - expected) < 1e-6


def test_estimate_contact_force():
    """Test contact force estimation from current."""
    current_a = 2.0
    force_n = torque_tools.estimate_contact_force(current_a, torque_constant=0.02, gear_ratio=1.0)
    
    # force = (current * K_t * gear_ratio) / wheel_radius
    expected = (current_a * 0.02 * 1.0) / torque_tools.WHEEL_RADIUS_M
    assert abs(force_n - expected) < 1e-6


def test_mock_odrive_axis():
    """Test mock ODrive axis."""
    axis = torque_tools.MockODriveAxis("Test")
    
    assert axis.name == "Test"
    assert axis.get_torque() == 0.0
    
    axis.set_torque(0.5)
    assert axis.get_torque() == 0.5
    
    axis.feed_watchdog()
    assert axis._watchdog_fed


def test_torque_controller_no_axes():
    """Test controller with no axes (mock mode)."""
    controller = torque_tools.TorqueController(None, None)
    
    # Should not crash
    controller.stop()
    controller.micro_move_straight(0.05, force_nm=0.3, max_duration_s=0.1)
    controller.micro_rotate(math.radians(15), torque_nm=0.3, max_duration_s=0.1)


def test_safety_limits():
    """Test safety limit constants."""
    assert torque_tools.MAX_TORQUE_NM == 0.5
    assert torque_tools.MAX_DURATION_S == 1.0
    assert torque_tools.TORQUE_RAMP_RATE == 2.0
    assert torque_tools.WATCHDOG_FEED_HZ == 20.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
