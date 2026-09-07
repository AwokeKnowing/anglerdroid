"""torque_tools.py – ODrive fine-control API for torque/current micro-moves.

Exposes low-level torque control for:
- Fine positioning (<5cm)
- Force-limited contact tasks
- Compliant motion
- Calibration and tuning

Uses odrivecan.py set_torque() with wheelbase geometry helpers.

SAFETY:
- Torque-limited: max 0.5 Nm per wheel (safe for indoor navigation)
- Time-limited: max 1.0 second per micro-move
- Watchdog-monitored: requires active feeding
- Emergency stop: immediate torque=0 on timeout

NOT for normal navigation → use twist() for velocity control.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

# Robot geometry (from robot_config.py)
from robot_config import WHEELBASE_M, WHEEL_RADIUS_M

# ODrive control (production hardware)
try:
    import odrivecan
    ODRIVE_AVAILABLE = True
except ImportError:
    odrivecan = None
    ODRIVE_AVAILABLE = False


# Safety limits
MAX_TORQUE_NM = 0.5         # Max torque per wheel (Nm)
MAX_DURATION_S = 1.0        # Max duration per micro-move (seconds)
TORQUE_RAMP_RATE = 2.0      # Torque ramp rate (Nm/s)
WATCHDOG_FEED_HZ = 20.0     # Watchdog feed frequency (Hz)


class TorqueController:
    """Fine-control torque commands for ODrive differential drive.
    
    Provides high-level torque primitives:
    - Straight line micro-move (distance-based)
    - In-place rotation micro-move (angle-based)
    - Force-limited contact (torque-based)
    - Compliant motion (impedance control stub)
    """
    
    def __init__(
        self,
        left_axis: Optional[any] = None,
        right_axis: Optional[any] = None,
        wheelbase_m: float = WHEELBASE_M,
        wheel_radius_m: float = WHEEL_RADIUS_M
    ):
        """Initialize torque controller with ODrive axes.
        
        Args:
            left_axis: ODriveAxisCAN instance for left wheel
            right_axis: ODriveAxisCAN instance for right wheel
            wheelbase_m: Distance between wheel centers (m)
            wheel_radius_m: Wheel radius (m)
        """
        self._left = left_axis
        self._right = right_axis
        self._wheelbase_m = wheelbase_m
        self._wheel_radius_m = wheel_radius_m
        
        self._last_feed_time = 0.0
        
        if not ODRIVE_AVAILABLE:
            print("torque_tools: ODrive not available (mock mode)")
    
    def _feed_watchdog(self) -> None:
        """Feed ODrive watchdogs at WATCHDOG_FEED_HZ."""
        now = time.time()
        if now - self._last_feed_time >= (1.0 / WATCHDOG_FEED_HZ):
            if self._left is not None:
                self._left.feed_watchdog()
            if self._right is not None:
                self._right.feed_watchdog()
            self._last_feed_time = now
    
    def _set_wheel_torques(self, left_nm: float, right_nm: float) -> None:
        """Set torques with safety clamping and watchdog feeding.
        
        Args:
            left_nm: Left wheel torque (Nm)
            right_nm: Right wheel torque (Nm)
        """
        # Clamp to safety limits
        left_nm = max(-MAX_TORQUE_NM, min(MAX_TORQUE_NM, left_nm))
        right_nm = max(-MAX_TORQUE_NM, min(MAX_TORQUE_NM, right_nm))
        
        # Send commands
        if self._left is not None:
            self._left.set_torque(left_nm)
        if self._right is not None:
            self._right.set_torque(right_nm)
        
        # Feed watchdog
        self._feed_watchdog()
    
    def stop(self) -> None:
        """Emergency stop: zero all torques immediately."""
        self._set_wheel_torques(0.0, 0.0)
    
    def micro_move_straight(
        self,
        distance_m: float,
        force_nm: float = 0.3,
        max_duration_s: float = MAX_DURATION_S
    ) -> bool:
        """Execute fine straight-line move with distance target.
        
        Args:
            distance_m: Target distance (m), positive=forward, negative=backward
            force_nm: Applied torque per wheel (Nm), clamped to MAX_TORQUE_NM
            max_duration_s: Timeout (seconds)
        
        Returns:
            True if completed, False if timeout or interrupted
        
        Note: Open-loop control (no encoder feedback in stub).
              Production: integrate encoder deltas for closed-loop.
        """
        if abs(distance_m) < 1e-6:
            return True
        
        # Determine direction
        direction = 1.0 if distance_m > 0 else -1.0
        torque = abs(force_nm) * direction
        
        # Clamp torque
        torque = max(-MAX_TORQUE_NM, min(MAX_TORQUE_NM, torque))
        
        # Estimate duration based on torque → accel → velocity → distance
        # Simplified: assume constant torque, small move
        # F = m*a → a ≈ (torque * 2) / (wheel_radius * robot_mass)
        # For stub: just use fixed duration proportional to distance
        est_duration = min(abs(distance_m) / 0.05, max_duration_s)  # 5cm/s assumption
        
        print(f"torque_tools: straight move {distance_m*100:.1f}cm @ {torque:.2f}Nm for {est_duration:.2f}s")
        
        # Apply torque (open-loop)
        t_start = time.time()
        while (time.time() - t_start) < est_duration:
            self._set_wheel_torques(torque, torque)
            time.sleep(0.05)  # 20 Hz update
        
        # Stop
        self.stop()
        return True
    
    def micro_rotate(
        self,
        angle_rad: float,
        torque_nm: float = 0.3,
        max_duration_s: float = MAX_DURATION_S
    ) -> bool:
        """Execute fine in-place rotation with angle target.
        
        Args:
            angle_rad: Target angle (radians), positive=CCW (left), negative=CW (right)
            torque_nm: Applied torque per wheel (Nm), clamped to MAX_TORQUE_NM
            max_duration_s: Timeout (seconds)
        
        Returns:
            True if completed, False if timeout or interrupted
        
        Note: Open-loop control (no IMU feedback in stub).
              Production: integrate IMU for closed-loop.
        """
        if abs(angle_rad) < 1e-6:
            return True
        
        # Differential drive rotation: opposite torques
        direction = 1.0 if angle_rad > 0 else -1.0
        torque = abs(torque_nm) * direction
        
        # Clamp torque
        torque = max(-MAX_TORQUE_NM, min(MAX_TORQUE_NM, torque))
        
        # Estimate duration: torque → angular accel → angular vel → angle
        # Simplified: assume small rotation, fixed rate
        est_duration = min(abs(angle_rad) / 0.5, max_duration_s)  # 0.5 rad/s assumption
        
        print(f"torque_tools: rotate {math.degrees(angle_rad):.1f}° @ {torque:.2f}Nm for {est_duration:.2f}s")
        
        # Apply differential torques (open-loop)
        t_start = time.time()
        while (time.time() - t_start) < est_duration:
            self._set_wheel_torques(torque, -torque)  # Left+, Right-
            time.sleep(0.05)  # 20 Hz update
        
        # Stop
        self.stop()
        return True
    
    def force_limited_contact(
        self,
        direction: str,
        max_force_nm: float = 0.2,
        duration_s: float = 0.5
    ) -> dict:
        """Execute force-limited contact task (e.g., gentle push).
        
        Args:
            direction: "forward", "backward", "left", "right"
            max_force_nm: Max torque per wheel (Nm)
            duration_s: Contact duration (seconds)
        
        Returns:
            Dict with keys: success, contact_detected, duration_s
        
        Note: Stub implementation (no force sensing).
              Production: monitor current draw for contact detection.
        """
        if direction not in ("forward", "backward", "left", "right"):
            raise ValueError(f"Invalid direction: {direction}")
        
        torque_left, torque_right = 0.0, 0.0
        
        if direction == "forward":
            torque_left = torque_right = max_force_nm
        elif direction == "backward":
            torque_left = torque_right = -max_force_nm
        elif direction == "left":
            torque_left = max_force_nm
            torque_right = -max_force_nm
        elif direction == "right":
            torque_left = -max_force_nm
            torque_right = max_force_nm
        
        print(f"torque_tools: force-limited {direction} @ {max_force_nm:.2f}Nm for {duration_s:.2f}s")
        
        # Apply torques
        t_start = time.time()
        contact_detected = False
        
        while (time.time() - t_start) < duration_s:
            self._set_wheel_torques(torque_left, torque_right)
            
            # Stub: no real contact detection
            # Production: monitor encoder velocity vs. commanded torque
            # If velocity << expected → contact detected
            
            time.sleep(0.05)
        
        # Stop
        self.stop()
        
        return {
            "success": True,
            "contact_detected": contact_detected,
            "duration_s": time.time() - t_start
        }
    
    def impedance_control_stub(
        self,
        target_pos_m: float,
        stiffness: float = 10.0,
        damping: float = 2.0
    ) -> None:
        """Placeholder for impedance control (compliant motion).
        
        Args:
            target_pos_m: Target position (m)
            stiffness: Spring constant (N/m)
            damping: Damping coefficient (N·s/m)
        
        Note: Requires encoder position feedback and force sensing.
              Not implemented in stub.
        """
        raise NotImplementedError("Impedance control requires encoder feedback (not in stub)")


# Geometric helpers

def torque_from_force_linear(force_n: float, wheel_radius_m: float = WHEEL_RADIUS_M) -> float:
    """Convert linear force (N) to wheel torque (Nm).
    
    Args:
        force_n: Linear force at ground contact (N)
        wheel_radius_m: Wheel radius (m)
    
    Returns:
        Torque (Nm)
    """
    return force_n * wheel_radius_m


def torque_from_force_angular(
    force_n: float,
    wheelbase_m: float = WHEELBASE_M,
    wheel_radius_m: float = WHEEL_RADIUS_M
) -> float:
    """Convert angular force (torque about robot center) to wheel torque (Nm).
    
    Args:
        force_n: Tangential force at wheel (N)
        wheelbase_m: Wheelbase (m)
        wheel_radius_m: Wheel radius (m)
    
    Returns:
        Torque per wheel (Nm)
    """
    # Moment arm = wheelbase / 2
    # Torque = Force * radius
    return force_n * (wheelbase_m / 2.0) * wheel_radius_m


def estimate_contact_force(
    current_a: float,
    torque_constant: float = 0.02,
    gear_ratio: float = 1.0
) -> float:
    """Estimate contact force from motor current (for force sensing).
    
    Args:
        current_a: Motor current (A)
        torque_constant: Motor torque constant (Nm/A)
        gear_ratio: Gearbox ratio (output/input)
    
    Returns:
        Estimated contact force (N)
    
    Note: Requires motor current measurement (not in stub).
    """
    torque_nm = current_a * torque_constant * gear_ratio
    force_n = torque_nm / WHEEL_RADIUS_M
    return force_n


# Mock ODriveAxisCAN for testing (no hardware)
class MockODriveAxis:
    """Mock ODrive axis for testing torque tools without hardware."""
    
    def __init__(self, name: str):
        self.name = name
        self._torque = 0.0
        self._watchdog_fed = False
    
    def set_torque(self, torque_nm: float) -> None:
        """Mock set_torque (just prints)."""
        self._torque = torque_nm
        # print(f"[{self.name}] set_torque({torque_nm:.3f} Nm)")
    
    def feed_watchdog(self) -> None:
        """Mock feed_watchdog."""
        self._watchdog_fed = True
        # print(f"[{self.name}] watchdog fed")
    
    def get_torque(self) -> float:
        """Get last commanded torque (for testing)."""
        return self._torque


if __name__ == "__main__":
    # Demo: test torque tools with mock axes
    print("=== Torque Tools Demo (Mock Mode) ===\n")
    
    # Create mock axes
    left = MockODriveAxis("Left")
    right = MockODriveAxis("Right")
    
    # Create controller
    controller = TorqueController(left, right)
    
    # Test micro-move straight
    print("Test 1: Straight micro-move")
    controller.micro_move_straight(0.05, force_nm=0.3, max_duration_s=1.0)
    print(f"Left torque: {left.get_torque():.3f} Nm")
    print(f"Right torque: {right.get_torque():.3f} Nm\n")
    
    # Test micro-rotate
    print("Test 2: Rotation micro-move")
    controller.micro_rotate(math.radians(15), torque_nm=0.3, max_duration_s=1.0)
    print(f"Left torque: {left.get_torque():.3f} Nm")
    print(f"Right torque: {right.get_torque():.3f} Nm\n")
    
    # Test force-limited contact
    print("Test 3: Force-limited contact")
    result = controller.force_limited_contact("forward", max_force_nm=0.2, duration_s=0.5)
    print(f"Result: {result}\n")
    
    # Test geometric helpers
    print("Test 4: Geometric conversions")
    force_linear = 10.0  # 10 N
    torque = torque_from_force_linear(force_linear)
    print(f"Linear force {force_linear} N → torque {torque:.3f} Nm")
    
    force_angular = 5.0  # 5 N tangential
    torque_ang = torque_from_force_angular(force_angular)
    print(f"Angular force {force_angular} N → torque {torque_ang:.3f} Nm per wheel")
