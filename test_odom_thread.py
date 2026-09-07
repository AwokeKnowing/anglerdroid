#!/usr/bin/env python3
"""test_odom_thread.py – Unit test for OdomThread pose snapshot."""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pose import PoseEstimator
from odom_thread import OdomThread


class MockWheelbase:
    def __init__(self):
        self.vl = 0.1
        self.vr = 0.1
    
    def get_wheel_velocities_mps(self):
        return self.vl, self.vr
    
    def get_encoder_health(self):
        return {'encoder_ok': True, 'age_s': 0.0}


def test_pose_snapshot():
    print("\n=== Test: OdomThread pose snapshot ===")
    pose = PoseEstimator(wheelbase_m=0.15, wheel_radius_m=0.065)
    wheelbase = MockWheelbase()
    odom_thread = OdomThread(pose, wheelbase, None, 100.0)
    odom_thread.start()
    time.sleep(0.5)
    x1, y1, theta1, t1, enc1 = odom_thread.get_pose_snapshot()
    print(f"Snapshot 1: x={x1:.4f}, y={y1:.4f} enc={enc1}")
    wheelbase.vl = 0.15
    wheelbase.vr = 0.10
    time.sleep(0.5)
    x2, y2, theta2, t2, enc2 = odom_thread.get_pose_snapshot()
    print(f"Snapshot 2: x={x2:.4f}, y={y2:.4f} enc={enc2}")
    assert enc2 is True
    odom_thread.stop()
    assert x2 > x1, f"Expected x to increase"
    assert t2 > t1, f"Expected timestamp to increase"
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    assert distance > 0.01, f"Should have moved >1cm"
    print("✓ Test passed")


if __name__ == '__main__':
    print("Testing OdomThread...")
    try:
        test_pose_snapshot()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
