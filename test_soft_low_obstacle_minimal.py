"""Minimal unit tests for soft low obstacle detection logic (no cv2 dependency).

Tests the soft low obstacle detector logic with synthetic depth patches.
This version tests the core detection algorithm without importing vision.py (cv2 dependency).
"""

import numpy as np
import sys

# Add src to path
sys.path.insert(0, 'src')
from robot_config import FRAME_W, FRAME_H

# Constants from vision.py
TD_FLOOR_CLIP = np.float32(0.91)
TD_PX_SIZE = 0.010  # 1cm per pixel


def check_topdown_soft_low_obstacle_standalone(verts, 
                                                 near_m=0.35, far_m=1.00,
                                                 min_height_cm=5.0, max_height_cm=30.0,
                                                 forward_strip_row_min=10, forward_strip_row_max=80,
                                                 lateral_col_margin=30,
                                                 min_pixels=100,
                                                 out_h=FRAME_H, out_w=FRAME_W,
                                                 floor_clip_m=TD_FLOOR_CLIP):
    """Standalone version of soft low obstacle detector for testing."""
    if len(verts) == 0:
        return False, 0, float('inf')
    
    # Extract coordinates
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    
    # Filter for valid depth in detection range
    valid_z = (z > 0.01) & (z >= near_m) & (z <= far_m)
    
    if not np.any(valid_z):
        return False, 0, float('inf')
    
    # Compute height above floor in cm
    height_cm = (floor_clip_m - z) * 100.0
    
    # Filter for low obstacle height band
    low_obs_mask = valid_z & (height_cm >= min_height_cm) & (height_cm <= max_height_cm)
    
    if not np.any(low_obs_mask):
        return False, 0, float('inf')
    
    # Project low obstacle points to image coordinates
    v_low_obs = verts[low_obs_mask]
    scale = np.float32(1.0 / TD_PX_SIZE)
    center = np.float32([out_w * 0.5, out_h * 0.5])
    
    # p = [col, row] in image before 180° rotation
    p = v_low_obs[:, :2] * scale + center
    cols, rows = p[:, 0], p[:, 1]
    
    # After 180° rotation
    rows_rotated = out_h - 1 - rows
    cols_rotated = out_w - 1 - cols
    
    # Check if points fall in forward strip ROI
    in_strip = (
        (rows_rotated >= forward_strip_row_min) &
        (rows_rotated <= forward_strip_row_max) &
        (cols_rotated >= lateral_col_margin) &
        (cols_rotated < out_w - lateral_col_margin)
    )
    
    low_obs_heights = height_cm[low_obs_mask][in_strip]
    
    if len(low_obs_heights) == 0:
        return False, 0, float('inf')
    
    low_obs_count = len(low_obs_heights)
    median_height_cm = float(np.median(low_obs_heights))
    
    triggered = low_obs_count >= min_pixels
    
    return triggered, low_obs_count, median_height_cm


def make_synthetic_pointcloud(z_distances, heights_cm, n_points_per_patch=200, 
                               x_range=(-0.15, 0.15), y_range=(0.3, 0.8)):
    """Create synthetic RS1 pointcloud."""
    assert len(z_distances) == len(heights_cm)
    
    verts = []
    for z_m, h_cm in zip(z_distances, heights_cm):
        z_actual = float(TD_FLOOR_CLIP) - (h_cm / 100.0)
        
        n = n_points_per_patch
        x = np.random.uniform(x_range[0], x_range[1], n).astype(np.float32)
        y = np.random.uniform(y_range[0], y_range[1], n).astype(np.float32)
        z = np.full(n, z_actual, dtype=np.float32)
        
        patch = np.column_stack([x, y, z])
        verts.append(patch)
    
    if len(verts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    return np.vstack(verts).astype(np.float32)


def test_floor_only():
    """Floor-only: should NOT trigger."""
    print("\n=== Test 1: Floor-only (no trip) ===")
    z_list = [0.5, 0.6, 0.7, 0.8]
    h_list = [0.0, 1.0, 2.0, 1.5]
    verts = make_synthetic_pointcloud(z_list, h_list, n_points_per_patch=150)
    triggered, count, median_h = check_topdown_soft_low_obstacle_standalone(verts)
    print(f"  Triggered: {triggered}, count: {count}, median_h: {median_h:.1f}cm")
    assert not triggered, "Floor-only should NOT trigger"
    print("  ✓ PASS")


def test_dog_bed():
    """Dog bed (5-30cm): should trigger."""
    print("\n=== Test 2: Dog bed low mound (trip) ===")
    z_list = [0.50, 0.55, 0.60, 0.65, 0.70]
    h_list = [8.0, 15.0, 20.0, 18.0, 12.0]
    verts = make_synthetic_pointcloud(z_list, h_list, n_points_per_patch=250)
    triggered, count, median_h = check_topdown_soft_low_obstacle_standalone(verts)
    print(f"  Triggered: {triggered}, count: {count}, median_h: {median_h:.1f}cm")
    assert triggered, "Dog bed should trigger"
    assert 5.0 <= median_h <= 30.0, f"Height {median_h:.1f}cm should be 5-30cm"
    assert count >= 100, f"Count {count} should be >= 100"
    print("  ✓ PASS")


def test_table_overhang():
    """Table overhang (>40cm): should NOT trigger."""
    print("\n=== Test 3: Table overhang (no trip) ===")
    z_list = [0.40, 0.45, 0.50, 0.55]
    h_list = [45.0, 50.0, 55.0, 48.0]
    verts = make_synthetic_pointcloud(z_list, h_list, n_points_per_patch=200)
    triggered, count, median_h = check_topdown_soft_low_obstacle_standalone(verts)
    print(f"  Triggered: {triggered}, count: {count}, median_h: {median_h:.1f}cm")
    assert not triggered, "Table overhang should NOT trigger"
    print("  ✓ PASS")


def test_empty_invalid():
    """Empty/invalid depth: fail-safe."""
    print("\n=== Test 4: Empty/invalid depth ===")
    verts_empty = np.zeros((0, 3), dtype=np.float32)
    triggered, count, _ = check_topdown_soft_low_obstacle_standalone(verts_empty)
    assert not triggered and count == 0, "Empty should not trigger"
    
    verts_invalid = np.zeros((500, 3), dtype=np.float32)
    triggered, count, _ = check_topdown_soft_low_obstacle_standalone(verts_invalid)
    assert not triggered and count == 0, "Invalid should not trigger"
    print("  ✓ PASS")


def run_tests():
    """Run all tests."""
    print("=" * 70)
    print("Soft Low Obstacle Detection Unit Tests (Minimal, no cv2)")
    print("=" * 70)
    
    tests = [test_floor_only, test_dog_bed, test_table_overhang, test_empty_invalid]
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
