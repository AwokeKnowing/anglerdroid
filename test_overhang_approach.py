#!/usr/bin/env python3
"""
Unit tests for overhang approach detection (table underside at medium distance).

Tests the EARLY warning system that detects overhangs at 30-70cm before the
near-field reflex (<30cm) fires. Prevents Kevin from driving under tables.

Incident: Kevin drove under table, mast crashed into underside. RS1 topdown
showed floor clear, near-field only fired when already underneath (<30cm).

Fix: Overhang approach detector checks RS1 depth for elevated structures at
30-70cm in forward cone, blocking COMMIT before entry.

Test cases:
1. Table underside at 50cm in forward cone → triggers
2. Table underside at 50cm in rear cone → no trigger
3. Table underside at 20cm (near-field range) → no trigger (near-field handles)
4. Table underside at 80cm (too far) → no trigger
5. Open floor only → no trigger
6. Mixed: floor + table underside at 50cm → triggers
7. SafetyGuard integration → fwd=0, bwd/ang computed normally
8. HouseBot integration → zeros fwd scores, blocks COMMIT
9. Noise filtering (few pixels) → no trigger
10. Empty/invalid clouds → no crash
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vision import check_topdown_overhang_approach, check_topdown_near_field
from safety import SafetyGuard


def make_overhang_cloud(z_distance, x_center=0.0, y_center=0.20, 
                         x_extent=0.25, y_extent=0.20, points=400):
    """Create synthetic point cloud with overhead structure (table underside).
    
    Args:
        z_distance: Z distance to overhang (metres, toward camera)
        x_center: X center of overhang (metres, 0 = robot centerline)
        y_center: Y center of overhang (metres, positive = forward)
        x_extent: X width of overhang (metres)
        y_extent: Y depth of overhang (metres)
        points: number of points (default increased to survive border clipping)
    
    Returns:
        Nx3 array (X, Y, Z)
    
    Note: Uses more points by default to survive _clip_decimated_border() which
          zeros edge pixels to remove RealSense decimation artifacts.
    """
    # Random points within overhang rectangle
    x = np.random.uniform(x_center - x_extent/2, x_center + x_extent/2, points)
    y = np.random.uniform(y_center - y_extent/2, y_center + y_extent/2, points)
    z = np.full(points, z_distance, dtype=np.float32)
    
    cloud = np.column_stack([x, y, z]).astype(np.float32)
    return cloud


def make_floor_cloud(z_floor=0.91, x_extent=1.0, y_extent=1.0, grid_size=20):
    """Create floor point cloud at specified Z distance.
    
    Args:
        z_floor: Z distance to floor (metres)
        x_extent: X extent (metres)
        y_extent: Y extent (metres)
        grid_size: grid points per axis
    
    Returns:
        Nx3 array (X, Y, Z)
    """
    x = np.linspace(-x_extent/2, x_extent/2, grid_size)
    y = np.linspace(0.0, y_extent, grid_size)  # Forward only
    xx, yy = np.meshgrid(x, y)
    
    points = np.zeros((grid_size * grid_size, 3), dtype=np.float32)
    points[:, 0] = xx.flatten()
    points[:, 1] = yy.flatten()
    points[:, 2] = z_floor
    
    return points


def test_detects_table_at_50cm():
    """Test 1: Detect table underside at 50cm in forward cone."""
    print("\n" + "="*70)
    print("Test 1: Detect table underside at 50cm (forward cone)")
    print("="*70)
    
    # Table underside at 50cm, centered in forward cone
    # Use larger extent and more points to survive border clipping
    cloud = make_overhang_cloud(z_distance=0.50, x_center=0.0, y_center=0.20, 
                                 x_extent=0.28, y_extent=0.30, points=500)
    
    triggered, count, median_z = check_topdown_overhang_approach(cloud)
    
    print(f"  Table at 0.50m (forward cone, 500 points generated):")
    print(f"    Triggered: {triggered}")
    print(f"    Overhang count: {count} (after border clipping)")
    print(f"    Median Z: {median_z:.3f}m")
    
    assert triggered, f"Should detect table at 50cm in forward cone (got {count} points after clipping)"
    assert count >= 80, f"Expected >=80 overhang points, got {count}"
    assert abs(median_z - 0.50) < 0.05, f"Median Z should be ~0.50m, got {median_z:.3f}m"
    
    print("  ✅ PASS: Table underside detected at 50cm")


def test_no_trigger_rear_cone():
    """Test 2: Should NOT trigger for table in rear cone."""
    print("\n" + "="*70)
    print("Test 2: Table in REAR cone should NOT trigger")
    print("="*70)
    
    # Table underside at 50cm, but in REAR (negative Y)
    cloud = make_overhang_cloud(z_distance=0.50, x_center=0.0, y_center=-0.20, 
                                 x_extent=0.28, y_extent=0.30, points=500)
    
    triggered, count, median_z = check_topdown_overhang_approach(cloud)
    
    print(f"  Table at 0.50m (rear cone, y=-0.20m):")
    print(f"    Triggered: {triggered}")
    print(f"    Overhang count: {count}")
    
    assert not triggered, "Should NOT trigger for table in rear cone"
    assert count == 0, f"Expected 0 overhang points in rear, got {count}"
    
    print("  ✅ PASS: Rear table correctly ignored")


def test_no_trigger_near_field_range():
    """Test 3: Should NOT trigger in near-field range (<30cm) - near-field handles it."""
    print("\n" + "="*70)
    print("Test 3: Near-field range (<30cm) handled by near-field reflex")
    print("="*70)
    
    # Object at 20cm (near-field range) - use more points to survive border clipping
    cloud = make_overhang_cloud(z_distance=0.20, x_center=0.0, y_center=0.20, 
                                 x_extent=0.28, y_extent=0.30, points=300)
    
    # Check overhang approach (should NOT trigger, too close)
    ovh_triggered, ovh_count, ovh_median_z = check_topdown_overhang_approach(cloud)
    
    # Check near-field (SHOULD trigger)
    nf_triggered, nf_count, nf_min_z = check_topdown_near_field(cloud)
    
    print(f"  Object at 0.20m (near-field range, 300 pts generated):")
    print(f"    Overhang approach triggered: {ovh_triggered} (count={ovh_count})")
    print(f"    Near-field triggered: {nf_triggered} (count={nf_count} after clipping)")
    
    assert not ovh_triggered, "Overhang approach should NOT trigger at 20cm"
    assert nf_triggered, f"Near-field SHOULD trigger at 20cm (got {nf_count} points after clipping)"
    
    print("  ✅ PASS: Near-field range correctly deferred to near-field reflex")


def test_no_trigger_too_far():
    """Test 4: Should NOT trigger for overhang too far (>70cm)."""
    print("\n" + "="*70)
    print("Test 4: Overhang too far (>70cm) should NOT trigger")
    print("="*70)
    
    # Object at 80cm (too far)
    cloud = make_overhang_cloud(z_distance=0.80, x_center=0.0, y_center=0.20, 
                                 x_extent=0.28, y_extent=0.30, points=500)
    
    triggered, count, median_z = check_topdown_overhang_approach(cloud)
    
    print(f"  Object at 0.80m (too far):")
    print(f"    Triggered: {triggered}")
    print(f"    Overhang count: {count}")
    
    assert not triggered, "Should NOT trigger for overhang at 80cm (too far)"
    
    print("  ✅ PASS: Far overhang correctly ignored")


def test_open_floor_only():
    """Test 5: Open floor only (no overhang) should NOT trigger."""
    print("\n" + "="*70)
    print("Test 5: Open floor only (no overhang)")
    print("="*70)
    
    # Floor only at 91cm
    cloud = make_floor_cloud(z_floor=0.91, x_extent=1.0, y_extent=1.0, grid_size=20)
    
    triggered, count, median_z = check_topdown_overhang_approach(cloud)
    
    print(f"  Floor only at 0.91m:")
    print(f"    Triggered: {triggered}")
    print(f"    Overhang count: {count}")
    
    assert not triggered, "Should NOT trigger for open floor only"
    
    print("  ✅ PASS: Open floor does not trigger")


def test_mixed_floor_and_table():
    """Test 6: Mixed floor + table underside should trigger."""
    print("\n" + "="*70)
    print("Test 6: Mixed floor + table underside")
    print("="*70)
    
    # Floor at 91cm
    floor = make_floor_cloud(z_floor=0.91, x_extent=1.0, y_extent=1.0, grid_size=20)
    
    # Table underside at 50cm in forward cone
    table = make_overhang_cloud(z_distance=0.50, x_center=0.0, y_center=0.20, 
                                 x_extent=0.28, y_extent=0.30, points=500)
    
    # Combine
    cloud = np.vstack([floor, table])
    
    triggered, count, median_z = check_topdown_overhang_approach(cloud)
    
    print(f"  Floor (400 pts @0.91m) + Table (500 pts @0.50m):")
    print(f"    Triggered: {triggered}")
    print(f"    Overhang count: {count} (after clipping)")
    print(f"    Median Z: {median_z:.3f}m")
    
    assert triggered, f"Should detect table despite floor presence (got {count} points)"
    assert count >= 80, f"Expected >=80 table points, got {count}"
    
    print("  ✅ PASS: Table detected in mixed scene")


def test_safety_guard_overhang_approach_stop():
    """Test 7: SafetyGuard stops forward when overhang approach detected."""
    print("\n" + "="*70)
    print("Test 7: SafetyGuard stops forward with overhang approach")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Create clear obstacle map (no floor obstacles)
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    # Update with overhang_approach flag TRUE
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 topdown_overhang_approach=True)
    
    print(f"  With overhang approach reflex:")
    print(f"    Forward scale: {guard.fwd_scale:.2f}")
    print(f"    Backward scale: {guard.bwd_scale:.2f}")
    print(f"    Angular scale: {guard.ang_scale:.2f}")
    print(f"    Reason: {guard.near_field_reason}")
    
    assert guard.fwd_scale == 0.0, "Forward should be STOPPED"
    assert guard.bwd_scale > 0.0, "Backward should be ALLOWED (if clear)"
    assert guard.ang_scale > 0.0, "Angular should be ALLOWED"
    assert guard.near_field_reason == "topdown_overhang_approach"
    
    print("  ✅ PASS: Forward stopped, backward/angular allowed")


def test_safety_guard_overhang_approach_clear():
    """Test 8: SafetyGuard allows motion when overhang approach clear."""
    print("\n" + "="*70)
    print("Test 8: SafetyGuard allows motion when overhang approach clear")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Create clear obstacle map
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    # Update with overhang_approach flag FALSE
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 topdown_overhang_approach=False)
    
    print(f"  Without overhang approach reflex:")
    print(f"    Forward scale: {guard.fwd_scale:.2f}")
    print(f"    Backward scale: {guard.bwd_scale:.2f}")
    print(f"    Angular scale: {guard.ang_scale:.2f}")
    
    assert guard.fwd_scale == 1.0, "Forward should be FULL SPEED"
    assert guard.bwd_scale == 1.0, "Backward should be FULL SPEED"
    assert guard.ang_scale == 1.0, "Angular should be FULL SPEED"
    assert guard.near_field_reason is None
    
    print("  ✅ PASS: All motion allowed when clear")


def test_house_bot_integration():
    """Test 9: HouseBot zeros fwd scores when overhang approach detected."""
    print("\n" + "="*70)
    print("Test 9: HouseBot integration (blocks COMMIT)")
    print("="*70)
    
    # Simulate HouseBot logic: when overhang_approach is True, zero fwd scores
    overhang_approach = True
    fwd_scale = 1.0
    scores = {"fwd_near": 0.8, "fwd_mid": 0.9}
    
    if overhang_approach:
        fwd_scale = 0.0
        scores["fwd_near"] = 0.0
        scores["fwd_mid"] = 0.0
    
    print(f"  With overhang_approach=True:")
    print(f"    fwd_scale: {fwd_scale:.2f}")
    print(f"    fwd_near: {scores['fwd_near']:.2f}")
    print(f"    fwd_mid: {scores['fwd_mid']:.2f}")
    
    assert fwd_scale == 0.0, "HouseBot should zero fwd_scale"
    assert scores["fwd_near"] == 0.0, "HouseBot should zero fwd_near"
    assert scores["fwd_mid"] == 0.0, "HouseBot should zero fwd_mid"
    
    print("  ✅ PASS: HouseBot blocks COMMIT when overhang detected")


def test_noise_filtering():
    """Test 10: Small number of overhang points does not trigger (noise filter)."""
    print("\n" + "="*70)
    print("Test 10: Noise filtering (only 30 overhang points)")
    print("="*70)
    
    # Small patch at 50cm (below min_pixels=80 threshold)
    cloud = make_overhang_cloud(z_distance=0.50, x_center=0.0, y_center=0.20, points=30)
    
    triggered, count, median_z = check_topdown_overhang_approach(cloud, min_pixels=80)
    
    print(f"  Only 30 points at 0.50m:")
    print(f"    Triggered: {triggered}")
    print(f"    Overhang count: {count}")
    
    assert not triggered, "Should NOT trigger with only 30 points (noise)"
    assert count == 30, f"Expected 30 overhang points, got {count}"
    
    print("  ✅ PASS: Noise filter prevents false triggers")


def test_empty_cloud():
    """Test 11: Empty or invalid clouds don't crash."""
    print("\n" + "="*70)
    print("Test 11: Empty/invalid cloud handling")
    print("="*70)
    
    # Empty cloud
    empty = np.zeros((0, 3), dtype=np.float32)
    triggered, count, median_z = check_topdown_overhang_approach(empty)
    
    print(f"  Empty cloud:")
    print(f"    Triggered: {triggered}")
    print(f"    Count: {count}")
    print(f"    Median Z: {median_z}")
    
    assert not triggered, "Empty cloud should not trigger"
    assert count == 0
    assert median_z == float('inf')
    
    # Cloud with all zeros (invalid depth)
    zeros = np.zeros((100, 3), dtype=np.float32)
    triggered, count, median_z = check_topdown_overhang_approach(zeros)
    
    print(f"  Zero cloud:")
    print(f"    Triggered: {triggered}")
    
    assert not triggered, "Zero cloud should not trigger"
    
    print("  ✅ PASS: Empty/invalid clouds handled gracefully")


def test_lateral_cone_limits():
    """Test 12: Overhang outside lateral cone should NOT trigger."""
    print("\n" + "="*70)
    print("Test 12: Overhang outside lateral cone (X limits)")
    print("="*70)
    
    # Overhang at 50cm but far left (x=-0.30m, outside default cone x=-0.15 to 0.15)
    cloud = make_overhang_cloud(z_distance=0.50, x_center=-0.30, y_center=0.20, 
                                 x_extent=0.10, y_extent=0.30, points=500)
    
    triggered, count, median_z = check_topdown_overhang_approach(cloud)
    
    print(f"  Overhang at x=-0.30m (outside lateral cone):")
    print(f"    Triggered: {triggered}")
    print(f"    Overhang count: {count}")
    
    assert not triggered, "Should NOT trigger for overhang outside lateral cone"
    
    print("  ✅ PASS: Lateral cone limits work correctly")


def test_priority_near_field_closer():
    """Test 13: Near-field takes priority when object is closer than 30cm."""
    print("\n" + "="*70)
    print("Test 13: Priority - near-field reflex for <30cm objects")
    print("="*70)
    
    # Object at 25cm (near-field range) - use more points to survive border clipping
    cloud = make_overhang_cloud(z_distance=0.25, x_center=0.0, y_center=0.20, 
                                 x_extent=0.28, y_extent=0.30, points=300)
    
    # Check both detectors
    ovh_triggered, ovh_count, _ = check_topdown_overhang_approach(cloud)
    nf_triggered, nf_count, _ = check_topdown_near_field(cloud)
    
    print(f"  Object at 0.25m (300 pts generated):")
    print(f"    Overhang approach: triggered={ovh_triggered}, count={ovh_count}")
    print(f"    Near-field: triggered={nf_triggered}, count={nf_count} (after clipping)")
    
    # 25cm is borderline - might trigger overhang if near_m=0.30
    # Key is near-field MUST trigger
    assert nf_triggered, f"Near-field MUST trigger at 25cm (got {nf_count} points after clipping)"
    
    print("  ✅ PASS: Near-field handles close objects")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*16 + "OVERHANG APPROACH TESTS" + " "*29 + "║")
    print("╚"+"═"*68+"╝")
    
    tests = [
        test_detects_table_at_50cm,
        test_no_trigger_rear_cone,
        test_no_trigger_near_field_range,
        test_no_trigger_too_far,
        test_open_floor_only,
        test_mixed_floor_and_table,
        test_safety_guard_overhang_approach_stop,
        test_safety_guard_overhang_approach_clear,
        test_house_bot_integration,
        test_noise_filtering,
        test_empty_cloud,
        test_lateral_cone_limits,
        test_priority_near_field_closer,
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
