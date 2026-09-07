"""Unit tests for soft low obstacle detection (dog bed, cushions).

Tests the new check_topdown_soft_low_obstacle() detector with synthetic
depth patches representing different scenarios:
  1. Floor-only (no trip) — should NOT detect obstacle
  2. Dog-bed-like low mound (5-30cm) — should detect and trigger
  3. Table overhang (elevated, >40cm) — should NOT trigger (existing overhang path)
  4. Empty/invalid depth — fail-safe, no false clear

Run: python3 -m pytest test_soft_low_obstacle.py -v
     or: python3 test_soft_low_obstacle.py
"""

import numpy as np
import sys

# Add src to path for imports
sys.path.insert(0, 'src')

from vision import check_topdown_soft_low_obstacle, TD_FLOOR_CLIP, TD_PX_SIZE
from robot_config import FRAME_W, FRAME_H


def make_synthetic_pointcloud(z_distances, heights_cm, n_points_per_patch=200, 
                               x_range=(-0.15, 0.15), y_range=(0.3, 0.8)):
    """Create synthetic RS1 pointcloud with specified Z distances and heights.
    
    Args:
        z_distances: list of Z values (metres, toward camera)
        heights_cm: list of height values (cm above floor) corresponding to z_distances
        n_points_per_patch: number of points per (z, height) pair
        x_range: (min, max) X spread in metres (lateral)
        y_range: (min, max) Y spread in metres (forward/back, after 180° rotation)
    
    Returns:
        Nx3 pointcloud array (X, Y, Z)
    """
    assert len(z_distances) == len(heights_cm), "Must have matching z and height lists"
    
    verts = []
    for z_m, h_cm in zip(z_distances, heights_cm):
        # Z from camera: height_cm = (TD_FLOOR_CLIP - z) * 100
        # Solve for z: z = TD_FLOOR_CLIP - (h_cm / 100)
        z_actual = float(TD_FLOOR_CLIP) - (h_cm / 100.0)
        
        # Generate random points in XY spread at this Z
        n = n_points_per_patch
        x = np.random.uniform(x_range[0], x_range[1], n).astype(np.float32)
        y = np.random.uniform(y_range[0], y_range[1], n).astype(np.float32)
        z = np.full(n, z_actual, dtype=np.float32)
        
        patch = np.column_stack([x, y, z])
        verts.append(patch)
    
    if len(verts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    return np.vstack(verts).astype(np.float32)


def test_floor_only_no_trip():
    """Floor-only scene: should NOT detect obstacle (no trip)."""
    print("\n=== Test 1: Floor-only (no trip) ===")
    
    # Generate floor points: 0-2cm height (noise floor)
    z_list = [0.5, 0.6, 0.7, 0.8]
    h_list = [0.0, 1.0, 2.0, 1.5]  # Very low, floor-level
    
    verts = make_synthetic_pointcloud(z_list, h_list, n_points_per_patch=150)
    
    triggered, count, median_h = check_topdown_soft_low_obstacle(verts)
    
    print(f"  Floor points: {len(verts)}")
    print(f"  Triggered: {triggered}")
    print(f"  Low obs count: {count}")
    print(f"  Median height: {median_h:.1f} cm")
    
    assert not triggered, "Floor-only should NOT trigger soft low obstacle detection"
    print("  ✓ PASS: Floor-only does not trip")


def test_dog_bed_low_mound_trip():
    """Dog bed / cushion: low mound (5-30cm) should detect and trigger."""
    print("\n=== Test 2: Dog bed low mound (trip) ===")
    
    # Generate dog bed points: 8-25cm height at medium distance
    z_list = [0.50, 0.55, 0.60, 0.65, 0.70]
    h_list = [8.0, 15.0, 20.0, 18.0, 12.0]  # Low soft obstacle
    
    verts = make_synthetic_pointcloud(z_list, h_list, n_points_per_patch=250)
    
    triggered, count, median_h = check_topdown_soft_low_obstacle(verts)
    
    print(f"  Dog bed points: {len(verts)}")
    print(f"  Triggered: {triggered}")
    print(f"  Low obs count: {count}")
    print(f"  Median height: {median_h:.1f} cm")
    
    assert triggered, "Dog bed low mound should trigger soft low obstacle detection"
    assert 5.0 <= median_h <= 30.0, f"Median height {median_h:.1f}cm should be in 5-30cm range"
    assert count >= 100, f"Should detect enough low obstacle pixels, got {count}"
    print("  ✓ PASS: Dog bed low mound triggers detection")


def test_table_overhang_no_trip():
    """Table overhang (elevated >40cm): should NOT trigger (existing overhang path handles this)."""
    print("\n=== Test 3: Table overhang (no trip, overhang path) ===")
    
    # Generate table underside points: 45-60cm height (mast-collision zone)
    z_list = [0.40, 0.45, 0.50, 0.55]
    h_list = [45.0, 50.0, 55.0, 48.0]  # Above soft low obstacle range
    
    verts = make_synthetic_pointcloud(z_list, h_list, n_points_per_patch=200)
    
    triggered, count, median_h = check_topdown_soft_low_obstacle(verts)
    
    print(f"  Table overhang points: {len(verts)}")
    print(f"  Triggered: {triggered}")
    print(f"  Low obs count: {count}")
    print(f"  Median height: {median_h:.1f} cm")
    
    assert not triggered, "Table overhang should NOT trigger soft low obstacle (wrong height band)"
    print("  ✓ PASS: Table overhang does not trip soft low detector (overhang path handles it)")


def test_empty_invalid_depth_fail_safe():
    """Empty/invalid depth: fail-safe, no false clear."""
    print("\n=== Test 4: Empty/invalid depth (fail-safe) ===")
    
    # Test with empty pointcloud
    verts_empty = np.zeros((0, 3), dtype=np.float32)
    triggered_empty, count_empty, median_h_empty = check_topdown_soft_low_obstacle(verts_empty)
    
    print(f"  Empty cloud: triggered={triggered_empty}, count={count_empty}")
    assert not triggered_empty, "Empty depth should NOT trigger (fail-safe)"
    assert count_empty == 0, "Empty depth should have zero low obstacle count"
    
    # Test with invalid depth (all zeros)
    verts_invalid = np.zeros((500, 3), dtype=np.float32)
    triggered_invalid, count_invalid, median_h_invalid = check_topdown_soft_low_obstacle(verts_invalid)
    
    print(f"  Invalid (zero) depth: triggered={triggered_invalid}, count={count_invalid}")
    assert not triggered_invalid, "Invalid depth should NOT trigger (fail-safe)"
    assert count_invalid == 0, "Invalid depth should have zero low obstacle count"
    
    # Test with depth out of detection range (too close, in near-field zone)
    z_list_near = [0.10, 0.15, 0.20, 0.25]  # <0.35m, in near-field zone
    h_list_near = [15.0, 18.0, 20.0, 16.0]  # Would be in height range, but wrong distance
    verts_near = make_synthetic_pointcloud(z_list_near, h_list_near, n_points_per_patch=200)
    triggered_near, count_near, median_h_near = check_topdown_soft_low_obstacle(verts_near)
    
    print(f"  Near-field zone: triggered={triggered_near}, count={count_near}")
    assert not triggered_near, "Near-field zone should NOT trigger soft low (existing near-field reflex handles)"
    
    print("  ✓ PASS: Empty/invalid depth fail-safe, no false clear")


def test_mixed_scene_low_obstacle_wins():
    """Mixed scene: floor + low obstacle → should detect low obstacle."""
    print("\n=== Test 5: Mixed scene (floor + low obstacle) ===")
    
    # Mix floor and low obstacle points
    z_floor = [0.70, 0.75, 0.80, 0.85]
    h_floor = [0.0, 1.0, 2.0, 1.5]
    
    z_dog_bed = [0.50, 0.55, 0.60, 0.65]
    h_dog_bed = [12.0, 18.0, 22.0, 16.0]
    
    verts_floor = make_synthetic_pointcloud(z_floor, h_floor, n_points_per_patch=150)
    verts_dog_bed = make_synthetic_pointcloud(z_dog_bed, h_dog_bed, n_points_per_patch=250)
    
    verts_mixed = np.vstack([verts_floor, verts_dog_bed])
    
    triggered, count, median_h = check_topdown_soft_low_obstacle(verts_mixed)
    
    print(f"  Mixed scene points: {len(verts_mixed)} (floor + dog bed)")
    print(f"  Triggered: {triggered}")
    print(f"  Low obs count: {count}")
    print(f"  Median height: {median_h:.1f} cm")
    
    assert triggered, "Mixed scene with low obstacle should trigger"
    assert count >= 100, f"Should detect low obstacle pixels in mixed scene, got {count}"
    print("  ✓ PASS: Mixed scene detects low obstacle")


def test_height_band_boundaries():
    """Test boundary conditions of height detection (5cm min, 30cm max)."""
    print("\n=== Test 6: Height band boundaries ===")
    
    # Just below minimum (4cm) — should NOT trigger
    verts_below = make_synthetic_pointcloud([0.6], [4.0], n_points_per_patch=300)
    triggered_below, count_below, _ = check_topdown_soft_low_obstacle(verts_below)
    print(f"  4cm height: triggered={triggered_below}, count={count_below}")
    assert not triggered_below, "4cm (below 5cm min) should NOT trigger"
    
    # At minimum (5cm) — should trigger
    verts_min = make_synthetic_pointcloud([0.6], [5.5], n_points_per_patch=300)
    triggered_min, count_min, median_h_min = check_topdown_soft_low_obstacle(verts_min)
    print(f"  5.5cm height: triggered={triggered_min}, count={count_min}, median={median_h_min:.1f}cm")
    assert triggered_min, "5.5cm (at min threshold) should trigger"
    
    # At maximum (30cm) — should trigger
    verts_max = make_synthetic_pointcloud([0.6], [29.0], n_points_per_patch=300)
    triggered_max, count_max, median_h_max = check_topdown_soft_low_obstacle(verts_max)
    print(f"  29cm height: triggered={triggered_max}, count={count_max}, median={median_h_max:.1f}cm")
    assert triggered_max, "29cm (at max threshold) should trigger"
    
    # Just above maximum (31cm) — should NOT trigger
    verts_above = make_synthetic_pointcloud([0.6], [31.5], n_points_per_patch=300)
    triggered_above, count_above, _ = check_topdown_soft_low_obstacle(verts_above)
    print(f"  31.5cm height: triggered={triggered_above}, count={count_above}")
    assert not triggered_above, "31.5cm (above 30cm max) should NOT trigger"
    
    print("  ✓ PASS: Height band boundaries correct (5-30cm)")


def test_distance_band_boundaries():
    """Test distance detection range (near_m=0.35, far_m=1.0)."""
    print("\n=== Test 7: Distance band boundaries ===")
    
    # Too close (in near-field zone <0.35m) — should NOT trigger soft low
    verts_too_close = make_synthetic_pointcloud([0.20, 0.25, 0.30], [15.0, 18.0, 20.0], 
                                                n_points_per_patch=200)
    triggered_close, count_close, _ = check_topdown_soft_low_obstacle(verts_too_close)
    print(f"  Distance <0.35m: triggered={triggered_close}, count={count_close}")
    assert not triggered_close, "Distance <0.35m should NOT trigger (near-field zone)"
    
    # At near boundary (0.35-0.40m) — should trigger
    verts_near_edge = make_synthetic_pointcloud([0.36, 0.38, 0.40], [15.0, 18.0, 20.0],
                                                n_points_per_patch=200)
    triggered_near, count_near, _ = check_topdown_soft_low_obstacle(verts_near_edge)
    print(f"  Distance 0.36-0.40m: triggered={triggered_near}, count={count_near}")
    assert triggered_near, "Distance at near edge (0.36-0.40m) should trigger"
    
    # At far boundary (0.9-1.0m) — should trigger
    verts_far_edge = make_synthetic_pointcloud([0.90, 0.95, 0.98], [15.0, 18.0, 20.0],
                                               n_points_per_patch=200)
    triggered_far, count_far, _ = check_topdown_soft_low_obstacle(verts_far_edge)
    print(f"  Distance 0.90-0.98m: triggered={triggered_far}, count={count_far}")
    assert triggered_far, "Distance at far edge (0.90-0.98m) should trigger"
    
    # Too far (>1.0m) — should NOT trigger
    verts_too_far = make_synthetic_pointcloud([1.05, 1.10, 1.20], [15.0, 18.0, 20.0],
                                              n_points_per_patch=200)
    triggered_too_far, count_too_far, _ = check_topdown_soft_low_obstacle(verts_too_far)
    print(f"  Distance >1.0m: triggered={triggered_too_far}, count={count_too_far}")
    assert not triggered_too_far, "Distance >1.0m should NOT trigger (out of range)"
    
    print("  ✓ PASS: Distance band boundaries correct (0.35-1.0m)")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("Soft Low Obstacle Detection Unit Tests")
    print("=" * 70)
    
    tests = [
        test_floor_only_no_trip,
        test_dog_bed_low_mound_trip,
        test_table_overhang_no_trip,
        test_empty_invalid_depth_fail_safe,
        test_mixed_scene_low_obstacle_wins,
        test_height_band_boundaries,
        test_distance_band_boundaries,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
