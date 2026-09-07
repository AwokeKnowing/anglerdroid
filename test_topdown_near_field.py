#!/usr/bin/env python3
"""
Unit tests for top-down near-field safety reflex.

Tests the reflex that prevents driving under tables where the floor looks clear
but the mast will crash into the underside (incident: Kevin drove under table,
RS1 was 15cm from underside, floor looked free).

Test cases:
1. Synthetic RS1 cloud with patch at 15cm → reflex triggers
2. Patch at 15cm with rear clear → reverse allowed
3. Patch at 15cm with rear blocked → reverse blocked, forward blocked
4. Patch at >40cm → no reflex trigger
5. Hand-sized close patch → reflex triggers
6. No close points → reflex does not trigger
7. Multiple close patches → reflex triggers
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vision import check_topdown_near_field, depth_topdown
from safety import SafetyGuard, build_safety_occ


def make_synthetic_cloud(z_distances, points_per_patch=100, xy_spread=0.1):
    """Create synthetic point cloud with patches at specified Z distances.
    
    Args:
        z_distances: list of Z values (metres) for each patch
        points_per_patch: number of points per patch
        xy_spread: spatial spread in X,Y (metres)
    
    Returns:
        Nx3 array (X, Y, Z)
    """
    patches = []
    for z in z_distances:
        # Random X,Y positions around center
        xy = np.random.uniform(-xy_spread, xy_spread, (points_per_patch, 2))
        z_col = np.full((points_per_patch, 1), z, dtype=np.float32)
        patch = np.hstack([xy, z_col]).astype(np.float32)
        patches.append(patch)
    
    return np.vstack(patches)


def make_floor_cloud(z_floor=0.91, grid_size=20, xy_extent=2.0):
    """Create a floor-like point cloud at specified Z distance.
    
    Args:
        z_floor: Z distance to floor (metres)
        grid_size: grid points per axis
        xy_extent: spatial extent in X,Y (metres)
    
    Returns:
        Nx3 array (X, Y, Z)
    """
    x = np.linspace(-xy_extent/2, xy_extent/2, grid_size)
    y = np.linspace(-xy_extent/2, xy_extent/2, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    points = np.zeros((grid_size * grid_size, 3), dtype=np.float32)
    points[:, 0] = xx.flatten()
    points[:, 1] = yy.flatten()
    points[:, 2] = z_floor
    
    return points


def test_near_field_detection():
    """Test 1: Near-field detection with patch at 15cm."""
    print("\n" + "="*70)
    print("Test 1: Near-field detection with patch at 15cm")
    print("="*70)
    
    # Create cloud with close patch at 15cm (0.15m)
    cloud = make_synthetic_cloud([0.15], points_per_patch=100)
    
    triggered, close_count, min_z = check_topdown_near_field(
        cloud, threshold_m=0.30, min_pixels=50)
    
    print(f"  Close patch at 0.15m:")
    print(f"    Triggered: {triggered}")
    print(f"    Close count: {close_count}")
    print(f"    Min Z: {min_z:.3f}m")
    
    assert triggered, "Should trigger with 100 points at 15cm"
    assert close_count == 100, f"Expected 100 close points, got {close_count}"
    assert abs(min_z - 0.15) < 0.01, f"Min Z should be ~0.15m, got {min_z:.3f}m"
    
    print("  ✅ PASS: Near-field reflex triggers at 15cm")


def test_no_trigger_far():
    """Test 2: No trigger with patch at >40cm."""
    print("\n" + "="*70)
    print("Test 2: No trigger with patch at >40cm")
    print("="*70)
    
    # Create cloud with patch at 50cm (0.50m) - should not trigger
    cloud = make_synthetic_cloud([0.50], points_per_patch=100)
    
    triggered, close_count, min_z = check_topdown_near_field(
        cloud, threshold_m=0.30, min_pixels=50)
    
    print(f"  Patch at 0.50m:")
    print(f"    Triggered: {triggered}")
    print(f"    Close count: {close_count}")
    print(f"    Min Z: {min_z:.3f}m")
    
    assert not triggered, "Should NOT trigger with patch at 50cm"
    assert close_count == 0, f"Expected 0 close points, got {close_count}"
    
    print("  ✅ PASS: No reflex trigger at 50cm")


def test_hand_sized_patch():
    """Test 3: Hand-sized close patch triggers."""
    print("\n" + "="*70)
    print("Test 3: Hand-sized close patch at 20cm")
    print("="*70)
    
    # Create small patch (like a hand) at 20cm with 60 points
    cloud = make_synthetic_cloud([0.20], points_per_patch=60, xy_spread=0.05)
    
    triggered, close_count, min_z = check_topdown_near_field(
        cloud, threshold_m=0.30, min_pixels=50)
    
    print(f"  Hand-sized patch (60 points) at 0.20m:")
    print(f"    Triggered: {triggered}")
    print(f"    Close count: {close_count}")
    print(f"    Min Z: {min_z:.3f}m")
    
    assert triggered, "Should trigger with 60-point hand-sized patch"
    assert close_count == 60, f"Expected 60 close points, got {close_count}"
    
    print("  ✅ PASS: Hand-sized patch triggers reflex")


def test_mixed_distances():
    """Test 4: Mixed near and far patches."""
    print("\n" + "="*70)
    print("Test 4: Mixed near (18cm) and far (90cm) patches")
    print("="*70)
    
    # Create cloud with both near (18cm) and far (90cm) patches
    # Use more points to account for border clipping
    cloud = make_synthetic_cloud([0.18, 0.90], points_per_patch=200)
    
    triggered, close_count, min_z = check_topdown_near_field(
        cloud, threshold_m=0.30, min_pixels=50)
    
    print(f"  Mixed patches (200 at 0.18m, 200 at 0.90m):")
    print(f"    Triggered: {triggered}")
    print(f"    Close count: {close_count}")
    print(f"    Min Z: {min_z:.3f}m")
    
    assert triggered, "Should trigger with near patch present"
    assert close_count >= 50, f"Expected >=50 close points after border clip, got {close_count}"
    assert abs(min_z - 0.18) < 0.01, f"Min Z should be ~0.18m, got {min_z:.3f}m"
    
    print("  ✅ PASS: Detects near patch in mixed cloud")


def test_noise_filtering():
    """Test 5: Small number of close points does not trigger (noise filter)."""
    print("\n" + "="*70)
    print("Test 5: Noise filtering (only 20 close points)")
    print("="*70)
    
    # Create cloud with only 20 close points (below min_pixels=50 threshold)
    cloud = make_synthetic_cloud([0.15], points_per_patch=20)
    
    triggered, close_count, min_z = check_topdown_near_field(
        cloud, threshold_m=0.30, min_pixels=50)
    
    print(f"  Only 20 points at 0.15m:")
    print(f"    Triggered: {triggered}")
    print(f"    Close count: {close_count}")
    
    assert not triggered, "Should NOT trigger with only 20 points (noise)"
    assert close_count == 20, f"Expected 20 close points, got {close_count}"
    
    print("  ✅ PASS: Noise filter prevents false triggers")


def test_safety_guard_near_field_stop():
    """Test 6: SafetyGuard stops forward when near-field detected."""
    print("\n" + "="*70)
    print("Test 6: SafetyGuard stops forward with near-field")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Create clear obstacle map (no floor obstacles)
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    # Update with near-field flag TRUE
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 topdown_near_field=True)
    
    print(f"  With near-field reflex:")
    print(f"    Forward scale: {guard.fwd_scale:.2f}")
    print(f"    Backward scale: {guard.bwd_scale:.2f}")
    print(f"    Angular scale: {guard.ang_scale:.2f}")
    print(f"    Near-field flag: {guard.topdown_near_field}")
    print(f"    Reason: {guard.near_field_reason}")
    
    assert guard.fwd_scale == 0.0, "Forward should be STOPPED"
    assert guard.bwd_scale > 0.0, "Backward should be ALLOWED (if clear)"
    assert guard.ang_scale > 0.0, "Angular should be ALLOWED"
    assert guard.topdown_near_field, "Near-field flag should be True"
    assert guard.near_field_reason == "topdown_near_field"
    
    print("  ✅ PASS: Forward stopped, backward/angular allowed")


def test_safety_guard_near_field_clear():
    """Test 7: SafetyGuard allows all motion when near-field clear."""
    print("\n" + "="*70)
    print("Test 7: SafetyGuard allows motion when near-field clear")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Create clear obstacle map (no floor obstacles)
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    # Update with near-field flag FALSE
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 topdown_near_field=False)
    
    print(f"  Without near-field reflex:")
    print(f"    Forward scale: {guard.fwd_scale:.2f}")
    print(f"    Backward scale: {guard.bwd_scale:.2f}")
    print(f"    Angular scale: {guard.ang_scale:.2f}")
    print(f"    Near-field flag: {guard.topdown_near_field}")
    
    assert guard.fwd_scale == 1.0, "Forward should be FULL SPEED"
    assert guard.bwd_scale == 1.0, "Backward should be FULL SPEED"
    assert guard.ang_scale == 1.0, "Angular should be FULL SPEED"
    assert not guard.topdown_near_field, "Near-field flag should be False"
    assert guard.near_field_reason is None
    
    print("  ✅ PASS: All motion allowed when clear")


def test_safety_guard_near_field_with_rear_obstacle():
    """Test 8: Near-field + rear obstacle blocks both forward and backward."""
    print("\n" + "="*70)
    print("Test 8: Near-field + rear obstacle blocks both directions")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Import coordinates from safety module
    from safety import BWD_SCAN_X0, MASK_Y0, MASK_Y1
    
    # Create obstacle map with rear obstacle
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    # Add obstacle behind robot (left of BWD_SCAN_X0 in ego frame)
    # Robot rear is at x < BWD_SCAN_X0, place obstacle close to it
    if BWD_SCAN_X0 > 5:
        obs_map[MASK_Y0:MASK_Y1, BWD_SCAN_X0-5:BWD_SCAN_X0] = 255  # Rear obstacle
    
    # Update with near-field flag TRUE and rear obstacle
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 topdown_near_field=True)
    
    print(f"  With near-field + rear obstacle:")
    print(f"    Forward scale: {guard.fwd_scale:.2f}")
    print(f"    Backward scale: {guard.bwd_scale:.2f}")
    print(f"    Angular scale: {guard.ang_scale:.2f}")
    print(f"    (BWD_SCAN_X0={BWD_SCAN_X0}, MASK_Y0={MASK_Y0}, MASK_Y1={MASK_Y1})")
    
    assert guard.fwd_scale == 0.0, "Forward should be STOPPED (near-field)"
    assert guard.bwd_scale < 0.5, "Backward should be REDUCED (obstacle)"
    # Angular might be allowed for escape spin
    
    print("  ✅ PASS: Both directions blocked appropriately")


def test_depth_topdown_with_close_objects():
    """Test 9: depth_topdown function handles close objects correctly."""
    print("\n" + "="*70)
    print("Test 9: depth_topdown processes close and far objects")
    print("="*70)
    
    # Create cloud with both floor and close overhead object
    floor = make_floor_cloud(z_floor=0.91, grid_size=15)
    overhead = make_synthetic_cloud([0.15], points_per_patch=50, xy_spread=0.1)
    cloud = np.vstack([floor, overhead])
    
    obs, known = depth_topdown(cloud, out_h=240, out_w=320)
    
    obs_px = np.count_nonzero(obs > 0)
    known_px = np.count_nonzero(known > 0)
    
    print(f"  Point cloud: {len(floor)} floor + {len(overhead)} overhead")
    print(f"  Obstacle pixels: {obs_px}")
    print(f"  Known pixels: {known_px}")
    
    # Both floor (free) and overhead (obstacle) should be marked as known
    assert known_px > 200, f"Should have substantial known coverage, got {known_px}"
    # Overhead object should create obstacles (closer than floor clip)
    assert obs_px > 0, f"Should have obstacle pixels from overhead, got {obs_px}"
    
    print("  ✅ PASS: depth_topdown handles mixed cloud correctly")


def test_empty_cloud():
    """Test 10: Empty or invalid clouds don't crash."""
    print("\n" + "="*70)
    print("Test 10: Empty cloud handling")
    print("="*70)
    
    # Empty cloud
    empty = np.zeros((0, 3), dtype=np.float32)
    triggered, close_count, min_z = check_topdown_near_field(empty)
    
    print(f"  Empty cloud:")
    print(f"    Triggered: {triggered}")
    print(f"    Close count: {close_count}")
    print(f"    Min Z: {min_z}")
    
    assert not triggered, "Empty cloud should not trigger"
    assert close_count == 0
    assert min_z == float('inf')
    
    # Cloud with all zeros
    zeros = np.zeros((100, 3), dtype=np.float32)
    triggered, close_count, min_z = check_topdown_near_field(zeros)
    
    print(f"  Zero cloud:")
    print(f"    Triggered: {triggered}")
    
    assert not triggered, "Zero cloud should not trigger"
    
    print("  ✅ PASS: Empty/invalid clouds handled gracefully")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*20 + "NEAR-FIELD REFLEX TESTS" + " "*25 + "║")
    print("╚"+"═"*68+"╝")
    
    tests = [
        test_near_field_detection,
        test_no_trigger_far,
        test_hand_sized_patch,
        test_mixed_distances,
        test_noise_filtering,
        test_safety_guard_near_field_stop,
        test_safety_guard_near_field_clear,
        test_safety_guard_near_field_with_rear_obstacle,
        test_depth_topdown_with_close_objects,
        test_empty_cloud,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  ❌ FAIL: {e}")
        except Exception as e:
            failed += 1
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED\n")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
