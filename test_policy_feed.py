#!/usr/bin/env python3
"""Unit tests for policy feed export (honest ego labels + height).

Tests the export_policy_feed helper that packages ego labels + height
for neural policy consumption at 30 Hz.

Run: python3 -m pytest test_policy_feed.py -v
     or: python3 test_policy_feed.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, RCX, RCY, BODY_BOX
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.policy_feed import export_policy_feed


def test_export_layout():
    """Test that export_policy_feed returns correct layout and dtype."""
    print("\n=== Test 1: Export layout ===")
    
    # Create synthetic ego labels + height
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    # Add some data
    labels[100:120, 150:170] = CLEAR
    labels[50:70, 150:170] = OBSTACLE
    height[50:70, 150:170] = 10
    
    # Export (allocates fresh arrays)
    labels_out, height_out = export_policy_feed(labels, height)
    
    # Check shapes
    assert labels_out.shape == (FRAME_H, FRAME_W), f"Wrong labels shape: {labels_out.shape}"
    assert height_out.shape == (FRAME_H, FRAME_W), f"Wrong height shape: {height_out.shape}"
    
    # Check dtypes
    assert labels_out.dtype == np.uint8, f"Wrong labels dtype: {labels_out.dtype}"
    assert height_out.dtype == np.uint8, f"Wrong height dtype: {height_out.dtype}"
    
    # Check values preserved
    assert np.all(labels_out[100:120, 150:170] == CLEAR)
    assert np.all(labels_out[50:70, 150:170] == OBSTACLE)
    assert np.all(height_out[50:70, 150:170] == 10)
    
    print(f"  ✓ labels shape: {labels_out.shape}, dtype: {labels_out.dtype}")
    print(f"  ✓ height shape: {height_out.shape}, dtype: {height_out.dtype}")
    print("  ✓ PASS: Layout correct")


def test_zero_copy_export():
    """Test that export_policy_feed supports zero-copy to preallocated buffers."""
    print("\n=== Test 2: Zero-copy export ===")
    
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels[50:60, 100:110] = OBSTACLE
    height[50:60, 100:110] = 25
    
    # Preallocate output buffers
    labels_buf = np.full((FRAME_H, FRAME_W), 99, dtype=np.uint8)
    height_buf = np.full((FRAME_H, FRAME_W), 88, dtype=np.uint8)
    
    # Export to preallocated buffers
    labels_out, height_out = export_policy_feed(
        labels, height,
        labels_out=labels_buf,
        height_out=height_buf)
    
    # Check that returned arrays are the same objects
    assert labels_out is labels_buf, "labels_out should be same object as labels_buf"
    assert height_out is height_buf, "height_out should be same object as height_buf"
    
    # Check values copied correctly
    assert np.all(labels_out[50:60, 100:110] == OBSTACLE)
    assert np.all(height_out[50:60, 100:110] == 25)
    assert np.all(labels_out[0:10, 0:10] == 0)  # Non-obstacle region zeroed
    
    print("  ✓ Zero-copy: returned arrays are same objects as preallocated buffers")
    print("  ✓ Values copied correctly via np.copyto")
    print("  ✓ PASS: Zero-copy export works")


def test_self_wins_honesty():
    """Test that SELF label is preserved in export (honesty invariant)."""
    print("\n=== Test 3: SELF wins honesty ===")
    
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    # Fill with CLEAR everywhere
    labels[:, :] = CLEAR
    
    # Mark body box as SELF (upstream labeling enforces this)
    x0, y0, x1, y1 = BODY_BOX
    labels[y0:y1, x0:x1] = SELF
    
    # Export
    labels_out, height_out = export_policy_feed(labels, height)
    
    # SELF must remain SELF (never become CLEAR or OBSTACLE)
    assert np.all(labels_out[y0:y1, x0:x1] == SELF), "SELF must win in export"
    assert np.all(height_out[y0:y1, x0:x1] == 0), "SELF cells have no obstacle height"
    
    # Outside body should be CLEAR
    assert labels_out[0, 0] == CLEAR
    
    print(f"  ✓ SELF region ({y0}:{y1}, {x0}:{x1}) preserved as SELF")
    print("  ✓ SELF cells have height=0 (not obstacle)")
    print("  ✓ PASS: SELF wins honesty preserved")


def test_unknown_preserved():
    """Test that UNKNOWN cells remain UNKNOWN (no invented data)."""
    print("\n=== Test 4: UNKNOWN preserved ===")
    
    labels = np.full((FRAME_H, FRAME_W), UNKNOWN, dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    # Export
    labels_out, height_out = export_policy_feed(labels, height)
    
    # All cells should remain UNKNOWN
    assert np.all(labels_out == UNKNOWN), "UNKNOWN must not become CLEAR/OBSTACLE"
    assert np.all(height_out == 0), "UNKNOWN cells have no height"
    
    print("  ✓ All cells remain UNKNOWN (no invented CLEAR/OBSTACLE)")
    print("  ✓ UNKNOWN cells have height=0")
    print("  ✓ PASS: UNKNOWN honesty preserved")


def test_height_encoding():
    """Test obstacle height encoding (cm above floor, 0-100 clamped)."""
    print("\n=== Test 5: Height encoding ===")
    
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    # Low obstacle (10 cm dog bed)
    labels[50:60, 100:110] = OBSTACLE
    height[50:60, 100:110] = 10
    
    # Medium obstacle (50 cm table leg)
    labels[70:80, 100:110] = OBSTACLE
    height[70:80, 100:110] = 50
    
    # Tall obstacle (100 cm, capped)
    labels[90:100, 100:110] = OBSTACLE
    height[90:100, 100:110] = 100
    
    # Export
    labels_out, height_out = export_policy_feed(labels, height)
    
    # Check encodings
    assert np.all(height_out[50:60, 100:110] == 10), "10 cm obstacle encoded correctly"
    assert np.all(height_out[70:80, 100:110] == 50), "50 cm obstacle encoded correctly"
    assert np.all(height_out[90:100, 100:110] == 100), "100 cm obstacle (cap) encoded correctly"
    
    # CLEAR and UNKNOWN cells have no height
    labels[20:30, 100:110] = CLEAR
    labels_out, height_out = export_policy_feed(labels, height)
    assert np.all(height_out[20:30, 100:110] == 0), "CLEAR cells have height=0"
    
    print("  ✓ 10 cm obstacle: height=10")
    print("  ✓ 50 cm obstacle: height=50")
    print("  ✓ 100 cm obstacle (cap): height=100")
    print("  ✓ CLEAR cells: height=0")
    print("  ✓ PASS: Height encoding correct")


def test_label_enum_values():
    """Test that label enum values match perception contract."""
    print("\n=== Test 6: Label enum values ===")
    
    # Contract values (from perception.labels)
    assert UNKNOWN == 0, "UNKNOWN must be 0"
    assert SELF == 1, "SELF must be 1"
    assert CLEAR == 2, "CLEAR must be 2"
    assert OBSTACLE == 3, "OBSTACLE must be 3"
    
    labels = np.array([
        [UNKNOWN, SELF],
        [CLEAR, OBSTACLE]
    ], dtype=np.uint8)
    height = np.zeros((2, 2), dtype=np.uint8)
    
    labels_out, height_out = export_policy_feed(labels, height)
    
    # Check raw values
    assert labels_out[0, 0] == 0, "UNKNOWN encoded as 0"
    assert labels_out[0, 1] == 1, "SELF encoded as 1"
    assert labels_out[1, 0] == 2, "CLEAR encoded as 2"
    assert labels_out[1, 1] == 3, "OBSTACLE encoded as 3"
    
    print("  ✓ UNKNOWN=0, SELF=1, CLEAR=2, OBSTACLE=3")
    print("  ✓ Layout stable (perception contract)")
    print("  ✓ PASS: Label enum values correct")


def test_no_mutation_source():
    """Test that export does not mutate source arrays."""
    print("\n=== Test 7: No source mutation ===")
    
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels[50:60, 100:110] = OBSTACLE
    height[50:60, 100:110] = 25
    
    # Save original values
    labels_orig = labels.copy()
    height_orig = height.copy()
    
    # Export
    labels_out, height_out = export_policy_feed(labels, height)
    
    # Source arrays must not be mutated
    assert np.all(labels == labels_orig), "labels source must not be mutated"
    assert np.all(height == height_orig), "height source must not be mutated"
    
    # Output must match source
    assert np.all(labels_out == labels_orig), "labels_out must match source"
    assert np.all(height_out == height_orig), "height_out must match source"
    
    print("  ✓ Source arrays not mutated by export")
    print("  ✓ Output matches source values")
    print("  ✓ PASS: No source mutation")


def test_reuse_preallocated_buffers():
    """Test that preallocated buffers can be reused across frames (30 Hz pattern)."""
    print("\n=== Test 8: Reuse preallocated buffers ===")
    
    # Preallocate once (Vision.__init__)
    labels_buf = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_buf = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    # Frame 1
    labels1 = np.full((FRAME_H, FRAME_W), CLEAR, dtype=np.uint8)
    height1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels_out1, height_out1 = export_policy_feed(
        labels1, height1, labels_out=labels_buf, height_out=height_buf)
    
    assert labels_out1 is labels_buf
    assert np.all(labels_out1 == CLEAR)
    
    # Frame 2 (obstacle appears)
    labels2 = np.full((FRAME_H, FRAME_W), CLEAR, dtype=np.uint8)
    labels2[100:110, 100:110] = OBSTACLE
    height2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height2[100:110, 100:110] = 30
    labels_out2, height_out2 = export_policy_feed(
        labels2, height2, labels_out=labels_buf, height_out=height_buf)
    
    assert labels_out2 is labels_buf  # Same buffer object
    assert np.all(labels_out2[100:110, 100:110] == OBSTACLE)
    assert np.all(height_out2[100:110, 100:110] == 30)
    
    print("  ✓ Frame 1: exported to preallocated buffer")
    print("  ✓ Frame 2: reused same buffer object")
    print("  ✓ Values updated correctly each frame")
    print("  ✓ PASS: Preallocated buffer reuse works (30 Hz pattern)")


if __name__ == '__main__':
    print("=" * 80)
    print("POLICY FEED EXPORT UNIT TESTS")
    print("=" * 80)
    
    tests = [
        test_export_layout,
        test_zero_copy_export,
        test_self_wins_honesty,
        test_unknown_preserved,
        test_height_encoding,
        test_label_enum_values,
        test_no_mutation_source,
        test_reuse_preallocated_buffers,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 80)
    print(f"TESTS COMPLETE: {passed} passed, {failed} failed")
    print("=" * 80)
    
    sys.exit(0 if failed == 0 else 1)
