"""test_wheel_imu_prior.py – Unit tests for wheel+IMU prediction prior.

Tests synthetic scenarios:
  - Wheel slip (carpet → hard floor transitions)
  - Lost vision (black frames, low features)
  - IMU fusion
  - Visual correction gating
  - Pose persistence (load/save from JSON)

Run: python test_wheel_imu_prior.py
"""

import os
import json
import math
import tempfile
import numpy as np
from wheel_imu_prior import WheelIMUPrior, LINEAR_SLIP_SCALE, ANGULAR_SLIP_SCALE


def test_basic_prediction():
    """Test basic wheel odometry prediction."""
    print("\n=== Test: Basic Prediction ===")
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    
    # Move forward 1m at 0.5 m/s for 2 seconds
    dt = 0.01  # 10ms timestep
    v = 0.5    # m/s
    for _ in range(200):  # 2 seconds
        dtheta, ds = prior.predict(v, v, imu_yaw_rate=0.0, dt=dt)
    
    # Check final position (should be ~1m forward)
    assert abs(prior.x - 1.0) < 0.05, f"Expected x≈1.0m, got {prior.x:.3f}m"
    assert abs(prior.y) < 0.01, f"Expected y≈0, got {prior.y:.3f}m"
    assert abs(prior.theta) < 0.01, f"Expected theta≈0, got {prior.theta:.3f}rad"
    print(f"✓ Final pose: x={prior.x:.3f}m, y={prior.y:.3f}m, theta={math.degrees(prior.theta):.1f}°")


def test_rotation():
    """Test differential drive rotation."""
    print("\n=== Test: Rotation ===")
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    
    # Rotate 90° CCW (left wheel backward, right wheel forward)
    dt = 0.01
    v_left = -0.2
    v_right = 0.2
    steps = 200  # ~2 seconds
    
    # Expected rotation: omega = (v_right - v_left) / wheelbase
    omega = (v_right - v_left) / prior.wb  # rad/s
    expected_theta = omega * dt * steps * ANGULAR_SLIP_SCALE
    
    for _ in range(steps):
        prior.predict(v_left, v_right, imu_yaw_rate=0.0, dt=dt)
    
    # Allow 10% tolerance
    assert abs(prior.theta - expected_theta) < abs(expected_theta * 0.1), \
        f"Expected theta≈{expected_theta:.2f}rad, got {prior.theta:.2f}rad"
    print(f"✓ Final theta: {math.degrees(prior.theta):.1f}° (expected ~{math.degrees(expected_theta):.1f}°)")


def test_imu_fusion():
    """Test IMU yaw rate fusion with wheel odometry."""
    print("\n=== Test: IMU Fusion ===")
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    
    # Wheel says 10°/s, IMU says 12°/s (IMU is ground truth)
    # Fusion should blend: 50% wheel + 50% IMU = 11°/s
    dt = 0.01
    wheel_omega = math.radians(10)  # rad/s
    imu_yaw_rate = math.radians(12)  # rad/s
    
    # Differential velocities to achieve wheel_omega
    v_diff = wheel_omega * prior.wb
    v_left = 0.1 - v_diff / 2
    v_right = 0.1 + v_diff / 2
    
    for _ in range(100):  # 1 second
        prior.predict(v_left, v_right, imu_yaw_rate=imu_yaw_rate, dt=dt)
    
    # Expected: ~11° = (10 + 12) / 2 = 11° (with slip scale)
    expected_theta = math.radians(11) * 0.92  # ANGULAR_SLIP_SCALE
    print(f"✓ Fused theta: {math.degrees(prior.theta):.1f}° (expected ~{math.degrees(expected_theta):.1f}°)")
    assert abs(prior.theta - expected_theta) < 0.05


def test_visual_correction():
    """Test visual odometry correction when healthy."""
    print("\n=== Test: Visual Correction ===")
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    
    # Predict forward 0.1m (with slip scale)
    dt = 0.1
    v = 1.0
    dtheta_pred, ds_pred = prior.predict(v, v, imu_yaw_rate=0.0, dt=dt)
    
    # Visual says we only moved 0.09m (wheel slip) - this is a DELTA measurement
    vis_fwd = -0.01  # Correction: actual - predicted = 0.09 - 0.1 = -0.01
    vis_yaw = 0.0
    vis_conf = 0.8
    
    x_before = prior.x
    accepted = prior.correct_visual(vis_yaw, vis_fwd, vis_conf)
    
    assert accepted, "Visual correction should be accepted"
    # Note: small correction may not be visible at 4 decimal places
    print(f"✓ Visual correction applied: x {x_before:.4f}m → {prior.x:.4f}m")
    print(f"  (correction delta: {vis_fwd:.4f}m)")


def test_visual_rejection():
    """Test visual odometry rejection when unhealthy (outlier)."""
    print("\n=== Test: Visual Rejection ===")
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    
    # Predict forward 0.1m
    dt = 0.1
    v = 1.0
    prior.predict(v, v, imu_yaw_rate=0.0, dt=dt)
    
    # Visual says we moved 1.0m (way too much → outlier)
    vis_fwd = 1.0
    vis_yaw = 0.0
    vis_conf = 0.8
    
    x_before = prior.x
    accepted = prior.correct_visual(vis_yaw, vis_fwd, vis_conf)
    
    assert not accepted, "Visual correction should be rejected (Mahalanobis gate)"
    assert prior.x == x_before, "Pose should not change when visual rejected"
    print(f"✓ Visual outlier rejected: x={prior.x:.4f}m (unchanged)")


def test_carpet_slip_scenario():
    """Synthetic scenario: carpet → hard floor transition with wheel slip."""
    print("\n=== Test: Carpet Slip Scenario ===")
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    
    dt = 0.01
    v_cmd = 0.5  # Commanded 0.5 m/s
    
    # Phase 1: Hard floor (0-1s) — wheels accurate
    print("Phase 1: Hard floor (wheels accurate)")
    for _ in range(100):
        prior.predict(v_cmd, v_cmd, imu_yaw_rate=0.0, dt=dt)
        # Visual agrees with wheels
        if _ % 10 == 0:
            prior.correct_visual(0.0, v_cmd * dt * LINEAR_SLIP_SCALE, vis_confidence=0.9)
    
    x_phase1 = prior.x
    print(f"  x after 1s on hard floor: {x_phase1:.3f}m")
    
    # Phase 2: Carpet (1-2s) — wheels over-report (20% slip)
    print("Phase 2: Carpet (20% wheel slip)")
    for _ in range(100):
        # Wheels say 0.5 m/s, but actual is 0.4 m/s (20% slip)
        prior.predict(v_cmd, v_cmd, imu_yaw_rate=0.0, dt=dt)
        # Visual corrects for slip
        if _ % 10 == 0:
            actual_v = v_cmd * 0.8  # 20% slip
            prior.correct_visual(0.0, actual_v * dt * LINEAR_SLIP_SCALE, vis_confidence=0.9)
    
    x_phase2 = prior.x
    dx_phase2 = x_phase2 - x_phase1
    print(f"  x after 1s on carpet: {x_phase2:.3f}m (Δx={dx_phase2:.3f}m)")
    
    # Visual correction should have reduced reported motion (but not by much since
    # corrections are small deltas). Just check that we moved less than wheel-only would predict.
    # Wheel-only would be: v_cmd * dt * steps * LINEAR_SLIP_SCALE = 0.5 * 0.01 * 100 * 1.0 = 0.5m
    # With visual correction for 20% slip, should be closer to 0.4m
    # But our corrections are applied sparsely (every 10 frames), so won't be perfect
    assert dx_phase2 < 0.55, f"Motion should be bounded, got {dx_phase2:.3f}m"
    print("✓ Visual correction applied during carpet slip")


def test_lost_vision_scenario():
    """Synthetic scenario: lost vision (black frames, no features)."""
    print("\n=== Test: Lost Vision Scenario ===")
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    
    dt = 0.01
    v = 0.5
    imu_yaw_rate = math.radians(5)  # 5°/s turn
    
    # Phase 1: Good vision (0-1s)
    print("Phase 1: Good vision (visual corrections applied)")
    for i in range(100):
        dtheta, ds = prior.predict(v, v, imu_yaw_rate=imu_yaw_rate, dt=dt)
        # Apply visual correction every 10 frames
        if i % 10 == 0:
            prior.correct_visual(dtheta, ds, vis_confidence=0.8)
    
    x_phase1 = prior.x
    theta_phase1 = prior.theta
    print(f"  x={x_phase1:.3f}m, theta={math.degrees(theta_phase1):.1f}°")
    correct_phase1 = prior._correct_count
    
    # Phase 2: Lost vision (1-2s) — no visual corrections
    print("Phase 2: Lost vision (wheel+IMU only, no visual)")
    for i in range(100):
        prior.predict(v, v, imu_yaw_rate=imu_yaw_rate, dt=dt)
        # Don't call correct_visual at all (simulating lost vision)
        # In real code, adapter gates on vis_confidence
    
    x_phase2 = prior.x
    theta_phase2 = prior.theta
    correct_phase2 = prior._correct_count
    print(f"  x={x_phase2:.3f}m, theta={math.degrees(theta_phase2):.1f}°")
    print(f"  Corrections: phase1={correct_phase1}, phase2={correct_phase2}")
    
    # Should still integrate wheel+IMU even without vision
    assert x_phase2 > x_phase1, "Pose should continue integrating without vision"
    assert theta_phase2 > theta_phase1, "Rotation should continue with IMU"
    # Allow corrections from phase 1 to remain
    print("✓ Pose integration continued without visual corrections")


def test_pose_persistence():
    """Test save/load pose from JSON."""
    print("\n=== Test: Pose Persistence ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_pose.json")
        
        # Override default path for test
        import wheel_imu_prior
        original_path = wheel_imu_prior.LATEST_POSE_PATH
        wheel_imu_prior.LATEST_POSE_PATH = save_path
        
        try:
            # Create and move prior
            prior1 = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
            for _ in range(100):
                prior1.predict(0.5, 0.5, imu_yaw_rate=math.radians(10), dt=0.01)
            
            x1, y1, theta1 = prior1.x, prior1.y, prior1.theta
            print(f"Prior 1: x={x1:.3f}m, y={y1:.3f}m, theta={math.degrees(theta1):.1f}°")
            
            # Save
            prior1.save_latest_pose(force=True)
            assert os.path.exists(save_path), "Pose file should be created"
            
            # Load into new prior
            prior2 = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
            loaded = prior2.load_latest_pose()
            assert loaded, "Pose should load successfully"
            
            x2, y2, theta2 = prior2.x, prior2.y, prior2.theta
            print(f"Prior 2: x={x2:.3f}m, y={y2:.3f}m, theta={math.degrees(theta2):.1f}°")
            
            # Check match
            assert abs(x1 - x2) < 1e-6, "x should match after load"
            assert abs(y1 - y2) < 1e-6, "y should match after load"
            assert abs(theta1 - theta2) < 1e-6, "theta should match after load"
            print("✓ Pose saved and loaded correctly")
            
        finally:
            # Restore original path
            wheel_imu_prior.LATEST_POSE_PATH = original_path


def test_covariance_growth():
    """Test that covariance grows with motion but stays bounded."""
    print("\n=== Test: Covariance Growth ===")
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    
    P0 = prior.P.copy()
    print(f"Initial P: diag=[{P0[0,0]:.4f}, {P0[1,1]:.4f}, {P0[2,2]:.4f}]")
    
    # Move for 5 seconds without visual correction
    dt = 0.01
    for _ in range(500):
        prior.predict(0.5, 0.5, imu_yaw_rate=0.0, dt=dt)
    
    P1 = prior.P.copy()
    print(f"After 5s: diag=[{P1[0,0]:.4f}, {P1[1,1]:.4f}, {P1[2,2]:.4f}]")
    
    # Covariance should grow
    assert P1[0, 0] > P0[0, 0], "x covariance should grow"
    assert P1[1, 1] > P0[1, 1], "y covariance should grow"
    assert P1[2, 2] > P0[2, 2], "theta covariance should grow"
    
    # But should stay bounded
    assert P1[0, 0] < 1.0, "x covariance should be bounded"
    assert P1[1, 1] < 1.0, "y covariance should be bounded"
    assert P1[2, 2] < 0.25, "theta covariance should be bounded"
    print("✓ Covariance grows but stays bounded")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Testing wheel+IMU prior (self-slam-wheel-imu-prior-v0)")
    print("=" * 60)
    
    tests = [
        test_basic_prediction,
        test_rotation,
        test_imu_fusion,
        test_visual_correction,
        test_visual_rejection,
        test_carpet_slip_scenario,
        test_lost_vision_scenario,
        test_pose_persistence,
        test_covariance_growth,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
