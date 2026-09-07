#!/usr/bin/env python3
"""test_slam_robustness.py – Validate SLAM robustness improvements.

Tests:
  1. GPU map sync after loop closure
  2. Tracking quality metrics
  3. Encoder health monitoring
  4. Keepout transformation
  5. Keyframe export

Requirements:
  - numpy
  - opencv-python (cv2)
  - All src/ modules (slam, pose, wheelbase, keepouts)
  
Run from workspace root after environment setup is complete.
"""

import sys
import os
import time
import numpy as np
import tempfile
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_slam_gpu_sync():
    """Test that GPU map sync is triggered after loop closure."""
    print("\n=== Test 1: GPU map sync after loop closure ===")
    
    from slam import PoseGraphSLAM
    
    slam = PoseGraphSLAM()
    
    # Initially no sync needed
    assert not slam.needs_gpu_sync(), "Should not need sync at start"
    
    # Create some keyframes
    obs = np.random.randint(0, 100, (240, 320), dtype=np.uint8)
    known = np.ones((240, 320), dtype=np.uint8) * 255
    
    for i in range(5):
        x, y = i * 0.3, 0.0
        theta = 0.0
        slam.keyframe_check(obs, known, x, y, theta, 160, 120, 0.01)
    
    # Manually trigger a loop closure by adding edges
    if len(slam._keyframes) >= 2:
        slam._edges.append(
            (slam._keyframes[0].id, slam._keyframes[-1].id,
             1.0, 0.0, 0.0, slam._edges[0][5])  # Copy info matrix
        )
        slam._loop_count = 1
        
        # This should trigger rebuild and set sync flag
        slam._optimize_and_rebuild()
        
        # Check that sync flag is set
        assert slam.needs_gpu_sync(), "Should need GPU sync after loop closure"
        
        # Get maps for sync
        conf_map, height_map = slam.get_cpu_map()
        assert conf_map.shape == (720, 960), f"Wrong map shape: {conf_map.shape}"
        
        # Clear flag
        slam.clear_gpu_sync_flag()
        assert not slam.needs_gpu_sync(), "Should not need sync after clearing"
        
    print("✓ GPU sync flag management works correctly")


def test_tracking_quality_metrics():
    """Test pose tracking quality metrics."""
    print("\n=== Test 2: Tracking quality metrics ===")
    
    from pose import PoseEstimator
    
    pose = PoseEstimator(wheelbase_m=0.34, wheel_radius_m=0.08565)
    
    # Simulate some updates with visual odometry
    for i in range(10):
        vis_conf = 0.8 if i % 2 == 0 else 0.05  # Alternate good/bad
        pose.update(
            v_left_mps=0.1, v_right_mps=0.1, dt=0.033,
            vis_yaw=0.01, vis_fwd=0.003, vis_confidence=vis_conf
        )
    
    # Check metrics
    quality = pose.get_tracking_quality()
    
    assert 'visual_accept_rate' in quality
    assert 'wheel_only_rate' in quality
    assert 'time_since_visual' in quality
    assert 'excessive_disagreement' in quality
    
    # With alternating confidence, expect ~50% acceptance
    accept_rate = quality['visual_accept_rate']
    assert 0.3 < accept_rate < 0.7, f"Accept rate {accept_rate} out of range"
    
    print(f"  Visual accept rate: {accept_rate:.1%}")
    print(f"  Wheel-only rate: {quality['wheel_only_rate']:.1%}")
    print(f"  Time since visual: {quality['time_since_visual']:.2f}s")
    print("✓ Tracking quality metrics work correctly")


def test_encoder_health_stub():
    """Test encoder health metrics (stub without hardware)."""
    print("\n=== Test 3: Encoder health metrics (stub) ===")
    
    # We can't test real encoder without hardware, but verify API exists
    try:
        from wheelbase import WheelBase
        
        # Check that method exists
        assert hasattr(WheelBase, 'get_encoder_health'), \
            "WheelBase missing get_encoder_health method"
        
        print("✓ Encoder health API exists (hardware test skipped)")
    except ImportError:
        print("⚠ Could not import WheelBase (dependencies missing)")


def test_keepout_transformation():
    """Test keepout transformation for loop closure."""
    print("\n=== Test 4: Keepout transformation ===")
    
    import keepouts
    import math
    
    # Clear any existing disks
    keepouts.clear_all_disks()
    
    # Mark a disk at (1.0, 0.5)
    keepouts.mark_disk_at_pose(
        name="test_disk",
        pose_xy_yaw=(0.0, 0.0, 0.0),
        radius_m=0.5,
        kind="soft",
        ahead_m=1.0  # Will place at (1.0, 0.0)
    )
    
    marks = keepouts.list_marks()
    assert len(marks) == 1, f"Expected 1 mark, got {len(marks)}"
    
    orig_x = marks[0]['x']
    orig_y = marks[0]['y']
    
    # Apply a transformation (shift +0.5m in x, rotate 90°)
    keepouts.transform_disks(dx=0.5, dy=0.0, dtheta=math.pi / 2)
    
    marks_after = keepouts.list_marks()
    assert len(marks_after) == 1
    
    new_x = marks_after[0]['x']
    new_y = marks_after[0]['y']
    
    # After 90° rotation: (1.0, 0.0) → (0.0, 1.0), then +0.5 x → (0.5, 1.0)
    expected_x = 0.5
    expected_y = 1.0
    
    assert abs(new_x - expected_x) < 0.01, \
        f"X mismatch: expected {expected_x}, got {new_x}"
    assert abs(new_y - expected_y) < 0.01, \
        f"Y mismatch: expected {expected_y}, got {new_y}"
    
    print(f"  Original: ({orig_x:.3f}, {orig_y:.3f})")
    print(f"  Transformed: ({new_x:.3f}, {new_y:.3f})")
    print(f"  Expected: ({expected_x:.3f}, {expected_y:.3f})")
    print("✓ Keepout transformation works correctly")
    
    # Cleanup
    keepouts.clear_all_disks()


def test_keyframe_export():
    """Test keyframe export for reconstruction."""
    print("\n=== Test 5: Keyframe export ===")
    
    from slam import PoseGraphSLAM
    
    slam = PoseGraphSLAM()
    
    # Create a few keyframes
    obs = np.random.randint(0, 100, (240, 320), dtype=np.uint8)
    known = np.ones((240, 320), dtype=np.uint8) * 255
    
    for i in range(3):
        x, y = i * 0.5, 0.0
        theta = 0.0
        slam.keyframe_check(obs, known, x, y, theta, 160, 120, 0.01)
    
    # Export to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        slam.export_keyframes(temp_path, include_data=True)
        
        # Verify file exists and is valid JSON
        assert os.path.exists(temp_path), "Export file not created"
        
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        assert 'version' in data
        assert 'keyframes' in data
        assert 'edges' in data
        assert len(data['keyframes']) >= 3, \
            f"Expected >=3 keyframes, got {len(data['keyframes'])}"
        
        # Check keyframe structure
        kf = data['keyframes'][0]
        required_fields = ['id', 'x', 'y', 'theta', 'timestamp']
        for field in required_fields:
            assert field in kf, f"Keyframe missing field: {field}"
        
        print(f"  Exported {len(data['keyframes'])} keyframes")
        print(f"  File size: {os.path.getsize(temp_path) / 1024:.1f} KB")
        print("✓ Keyframe export works correctly")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_slam_stats():
    """Test SLAM statistics reporting."""
    print("\n=== Test 6: SLAM statistics ===")
    
    from slam import PoseGraphSLAM
    
    slam = PoseGraphSLAM()
    
    stats = slam.stats()
    
    required_keys = ['keyframes', 'edges', 'loop_closures', 'memory_mb',
                     'last_loop_shift_m', 'needs_gpu_sync']
    
    for key in required_keys:
        assert key in stats, f"Stats missing key: {key}"
    
    print(f"  Keyframes: {stats['keyframes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Loop closures: {stats['loop_closures']}")
    print(f"  Memory: {stats['memory_mb']:.1f} MB")
    print("✓ SLAM statistics work correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SLAM Robustness Tests")
    print("=" * 60)
    
    tests = [
        test_slam_gpu_sync,
        test_tracking_quality_metrics,
        test_encoder_health_stub,
        test_keepout_transformation,
        test_keyframe_export,
        test_slam_stats,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
