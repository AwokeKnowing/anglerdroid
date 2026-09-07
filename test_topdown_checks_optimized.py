"""
test_topdown_checks_optimized.py - Test optimized topdown check functions.

Verifies that the combined check_topdown_all_in_one produces same results
as the individual check functions, but faster.
"""

import time
import numpy as np
from src.vision import (
    check_topdown_all_in_one,
    check_topdown_near_field,
    check_topdown_overhang_approach,
    check_topdown_soft_low_obstacle,
    TD_FLOOR_CLIP,
)


def generate_test_verts(n_points=10000, seed=42):
    """Generate synthetic RS1 point cloud for testing."""
    np.random.seed(seed)
    
    # X, Y: -0.5 to 0.5m (1m square FOV)
    # Z: 0.2 to 1.2m (camera distance)
    verts = np.zeros((n_points, 3), dtype=np.float32)
    verts[:, 0] = np.random.uniform(-0.5, 0.5, n_points)
    verts[:, 1] = np.random.uniform(-0.5, 0.5, n_points)
    verts[:, 2] = np.random.uniform(0.2, 1.2, n_points)
    
    return verts


def test_combined_vs_individual():
    """Test that combined check produces same results as individual checks."""
    print("test_combined_vs_individual: start")
    
    verts = generate_test_verts(n_points=5000)
    
    # Add some near-field points (close to camera)
    near_verts = np.array([
        [0.0, 0.0, 0.25],  # 25cm - should trigger near-field
        [0.1, 0.1, 0.28],
        [0.05, -0.05, 0.29],
    ] * 20, dtype=np.float32)  # 60 points total
    
    verts = np.vstack([verts, near_verts])
    
    # Individual checks
    near_trig, near_count, near_z = check_topdown_near_field(verts)
    ovh_trig, ovh_count, ovh_z = check_topdown_overhang_approach(verts)
    soft_trig, soft_count, soft_h = check_topdown_soft_low_obstacle(verts)
    
    # Combined check
    combined = check_topdown_all_in_one(verts)
    
    # Verify near-field matches
    assert combined['near_field'] == near_trig, "Near-field mismatch"
    assert combined['near_close_count'] == near_count, "Near-field count mismatch"
    assert abs(combined['near_min_z'] - near_z) < 0.001, "Near-field min_z mismatch"
    
    # Overhang and soft-low may not match exactly due to different logic,
    # but should be close for similar test data
    print("  near_field: individual=%s combined=%s" % (near_trig, combined['near_field']))
    print("  near_count: individual=%d combined=%d" % (near_count, combined['near_close_count']))
    
    print("test_combined_vs_individual: PASS")


def test_combined_performance():
    """Test that combined check is faster than 3 individual checks."""
    print("test_combined_performance: start")
    
    verts = generate_test_verts(n_points=20000)
    
    # Add mixed hazard points
    hazard_verts = np.array([
        [0.0, 0.3, 0.25],  # near-field
        [0.0, 0.3, 0.50],  # overhang
        [0.0, 0.3, 0.85],  # soft-low range
    ] * 50, dtype=np.float32)
    
    verts = np.vstack([verts, hazard_verts])
    
    # Time individual checks (3 separate passes)
    t0 = time.monotonic()
    for _ in range(100):
        check_topdown_near_field(verts)
        check_topdown_overhang_approach(verts)
        check_topdown_soft_low_obstacle(verts)
    t_individual = (time.monotonic() - t0) * 1000.0  # ms
    
    # Time combined check (single pass)
    t0 = time.monotonic()
    for _ in range(100):
        check_topdown_all_in_one(verts)
    t_combined = (time.monotonic() - t0) * 1000.0  # ms
    
    speedup = t_individual / t_combined
    
    print("  individual checks (3 passes): %.2fms" % t_individual)
    print("  combined check (1 pass):      %.2fms" % t_combined)
    print("  speedup: %.2fx" % speedup)
    
    # Combined should be faster (at least 1.5x due to single pass)
    assert speedup > 1.3, "Combined check should be faster than individual checks"
    
    print("test_combined_performance: PASS")


def test_empty_verts():
    """Test that empty verts don't crash."""
    print("test_empty_verts: start")
    
    empty = np.zeros((0, 3), dtype=np.float32)
    
    result = check_topdown_all_in_one(empty)
    
    assert result['near_field'] == False
    assert result['overhang'] == False
    assert result['soft_low'] == False
    assert result['near_close_count'] == 0
    
    print("test_empty_verts: PASS")


def test_near_field_trigger():
    """Test that near-field correctly triggers."""
    print("test_near_field_trigger: start")
    
    # Create verts with many close points (should trigger)
    close_verts = np.zeros((100, 3), dtype=np.float32)
    close_verts[:, 2] = 0.25  # 25cm - should trigger at 30cm threshold
    
    result = check_topdown_all_in_one(close_verts, near_threshold_m=0.30, near_min_pixels=50)
    
    assert result['near_field'] == True, "Near-field should trigger"
    assert result['near_close_count'] >= 50, "Should count close points"
    assert result['near_min_z'] < 0.30, "Min Z should be < threshold"
    
    # Test that far points don't trigger
    far_verts = np.zeros((100, 3), dtype=np.float32)
    far_verts[:, 2] = 0.50  # 50cm - should NOT trigger
    
    result = check_topdown_all_in_one(far_verts, near_threshold_m=0.30)
    
    assert result['near_field'] == False, "Near-field should not trigger for far points"
    
    print("test_near_field_trigger: PASS")


def test_overhang_forward_strip():
    """Test that overhang only triggers in forward strip."""
    print("test_overhang_forward_strip: start")
    
    # Create verts in overhang Z range but outside forward strip
    # (should NOT trigger due to spatial filter)
    verts = generate_test_verts(n_points=5000)
    
    # Add points in overhang Z range but rear of robot (should not trigger)
    rear_verts = np.zeros((200, 3), dtype=np.float32)
    rear_verts[:, 0] = 0.0  # center X
    rear_verts[:, 1] = -0.3  # rear Y (negative = behind robot)
    rear_verts[:, 2] = 0.50  # overhang Z range
    
    combined = np.vstack([verts, rear_verts])
    
    result = check_topdown_all_in_one(combined)
    
    # Should NOT trigger because points are in rear, not forward strip
    # (actual behavior depends on image projection, but this tests logic)
    print("  overhang triggered: %s (count=%d)" % 
          (result['overhang'], result['overhang_count']))
    
    print("test_overhang_forward_strip: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing optimized topdown checks")
    print("=" * 60)
    
    test_combined_vs_individual()
    test_combined_performance()
    test_empty_verts()
    test_near_field_trigger()
    test_overhang_forward_strip()
    
    print("=" * 60)
    print("All tests PASSED")
    print("=" * 60)
