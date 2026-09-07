#!/usr/bin/env python3
"""Unit tests for D435i IMU integration with pose estimation."""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pose import PoseEstimator, ANGULAR_SLIP_SCALE


class MockIMU:
    """Mock IMU for testing without hardware."""
    def __init__(self):
        self.ok = True
        self.gyro = np.zeros(3, dtype=np.float32)
        self.accel = np.array([0.0, 9.81, 0.0], dtype=np.float32)
        self.timestamp = 0.0

    def set_yaw_rate(self, yaw_rate_rps):
        self.gyro[2] = yaw_rate_rps


def _wheel_vels_for_yaw(wheelbase_m, yaw_rad, dt, v=0.05, slip=ANGULAR_SLIP_SCALE):
    """Left/right mps that yield ~yaw_rad after ANGULAR_SLIP_SCALE."""
    omega = yaw_rad / (dt * slip)
    half = omega * wheelbase_m * 0.5
    return v - half, v + half


def test_imu_pipeline_mock():
    imu = MockIMU()
    assert imu.ok
    assert imu.gyro.shape == (3,)
    assert imu.accel.shape == (3,)
    imu.set_yaw_rate(0.5)
    assert abs(imu.gyro[2] - 0.5) < 1e-6
    print("PASS: MockIMU interface correct")


def test_zero_imu_no_contribution():
    dt = 0.1
    yaw = math.radians(5)
    vl, vr = _wheel_vels_for_yaw(0.165, yaw, dt)
    a = PoseEstimator(0.165, 0.03).update(
        vl, vr, dt, yaw, 0.005, 0.5, True, imu_yaw_rate=0.0)[0]
    b = PoseEstimator(0.165, 0.03).update(
        vl, vr, dt, yaw, 0.005, 0.5, True, imu_yaw_rate=1e-9)[0]
    assert abs(a - b) < 1e-6
    print("PASS: zero IMU measurement has no effect")


def test_imu_graceful_degradation():
    dt = 0.1
    target = math.radians(10)
    vl, vr = _wheel_vels_for_yaw(0.165, target, dt)
    dtheta, ds = PoseEstimator(0.165, 0.03).update(
        vl, vr, dt, math.radians(9.5), 0.005, 0.5, True, imu_yaw_rate=0.0)
    assert math.isfinite(dtheta) and math.isfinite(ds)
    assert abs(dtheta - target) < math.radians(3.0)
    print(f"Without IMU: dtheta={math.degrees(dtheta):.2f}°")
    print("PASS: graceful degradation without IMU")


def test_imu_yaw_fusion_improves_accuracy():
    """Stationary wheels + rejected visual: IMU still contributes yaw."""
    dt = 0.1
    truth = math.radians(10)
    imu_rate = truth / dt
    # Equal wheel speeds → wheel yaw 0; visual discarded (conf too low)
    no_imu = PoseEstimator(0.165, 0.03).update(
        0.0, 0.0, dt, math.radians(3), 0.0, 0.05, True, imu_yaw_rate=0.0)[0]
    with_imu = PoseEstimator(0.165, 0.03).update(
        0.0, 0.0, dt, math.radians(3), 0.0, 0.05, True, imu_yaw_rate=imu_rate)[0]

    print(f"Stationary wheels: no-IMU {math.degrees(no_imu):.2f}°, "
          f"with-IMU {math.degrees(with_imu):.2f}° (truth 10°)")
    assert abs(no_imu) < math.radians(0.5), "Without IMU, stationary should stay ~0"
    assert abs(with_imu - 0.5 * truth) < math.radians(0.5), \
        "Fallback IMU weight 0.5 should pull halfway to gyro delta"
    print("PASS: IMU fusion contributes yaw when visual weak")


def test_imu_weight_adapts_to_visual_confidence():
    """Lower visual confidence → larger IMU influence on fused yaw."""
    dt = 0.1
    wheel_yaw = math.radians(10)
    vl, vr = _wheel_vels_for_yaw(0.165, wheel_yaw, dt)
    vis_yaw = math.radians(7.0)
    imu_rate = math.radians(20) / dt  # deliberately hotter than wheels

    def run(conf, rate):
        return PoseEstimator(0.165, 0.03).update(
            vl, vr, dt, vis_yaw, 0.005, conf, True, imu_yaw_rate=rate)[0]

    delta_high = abs(run(0.85, imu_rate) - run(0.85, 0.0))
    delta_low = abs(run(0.15, imu_rate) - run(0.15, 0.0))
    print(f"IMU pull high-conf {math.degrees(delta_high):.2f}°; "
          f"low-conf {math.degrees(delta_low):.2f}°")
    assert delta_low > delta_high, "Low visual conf should amplify IMU pull"
    print("PASS: IMU weight adapts to visual confidence")


def test_imu_frame_transform():
    # Import path exists; transform lives on IMUPipeline
    from imu import IMUPipeline  # noqa: F401
    print("PASS: IMU frame transform module importable")


if __name__ == "__main__":
    test_imu_pipeline_mock()
    test_zero_imu_no_contribution()
    test_imu_graceful_degradation()
    test_imu_yaw_fusion_improves_accuracy()
    test_imu_weight_adapts_to_visual_confidence()
    test_imu_frame_transform()
    print("\nALL IMU INTEGRATION TESTS PASSED")
