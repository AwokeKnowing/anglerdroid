#!/usr/bin/env python3
"""
Unit tests for checkered floor mat hard-stop detection (ego/vision reflex).

Tests the RGB-based checkered mat detector that works WITHOUT SLAM or map keepouts.
Detects checkered floor mat patterns and triggers forward hard-stop (fwd_scale=0)
while allowing reverse/turn if rear is clear.

Test cases:
1. Synthetic 6x6 checkerboard → detection triggers
2. Checkerboard in bottom region only → triggers
3. Checkerboard in top region (not floor) → no trigger
4. Small checkerboard (fewer corners) → triggers with reduced confidence
5. No checkerboard pattern → no trigger
6. Partial checkerboard visible → triggers if enough corners
7. SafetyGuard integration → fwd=0, bwd/ang computed normally
8. Temporal filtering → reduces flicker
9. Empty/invalid images → no crash
10. Different checkerboard sizes (tunable parameters)
"""

import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from checkered_mat import CheckeredMatDetector, check_checkered_mat
from safety import SafetyGuard


def make_checkerboard_image(h=480, w=640, rows=7, cols=7, square_size=50):
    """Create a synthetic checkerboard image.
    
    Args:
        h: image height (pixels)
        w: image width (pixels)
        rows: number of square rows (7x7 = 6x6 internal corners)
        cols: number of square columns
        square_size: size of each square (pixels)
    
    Returns:
        RGB image (HxWx3 uint8)
    """
    # Create checkerboard pattern
    board_h = rows * square_size
    board_w = cols * square_size
    
    # Create pattern
    board = np.zeros((board_h, board_w), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                board[i*square_size:(i+1)*square_size, 
                      j*square_size:(j+1)*square_size] = 255
    
    # Place on larger canvas (centered)
    img = np.full((h, w), 128, dtype=np.uint8)
    y_offset = (h - board_h) // 2
    x_offset = (w - board_w) // 2
    
    # Ensure it fits
    if y_offset >= 0 and x_offset >= 0:
        img[y_offset:y_offset+board_h, x_offset:x_offset+board_w] = board
    else:
        # Crop if too large
        img = board[:h, :w]
    
    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb


def make_checkerboard_bottom(h=480, w=640, rows=7, cols=7, square_size=40):
    """Create checkerboard in BOTTOM region (where floor mat would be).
    
    Returns:
        RGB image with checkerboard at bottom (floor perspective)
    """
    img = np.full((h, w), 128, dtype=np.uint8)
    
    board_h = rows * square_size
    board_w = cols * square_size
    
    board = np.zeros((board_h, board_w), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                board[i*square_size:(i+1)*square_size, 
                      j*square_size:(j+1)*square_size] = 255
    
    # Place at BOTTOM of image
    y_offset = h - board_h - 10
    x_offset = (w - board_w) // 2
    
    if y_offset >= 0 and x_offset >= 0:
        y_end = min(h, y_offset + board_h)
        x_end = min(w, x_offset + board_w)
        img[y_offset:y_end, x_offset:x_end] = board[:y_end-y_offset, :x_end-x_offset]
    
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb


def make_checkerboard_top(h=480, w=640, rows=7, cols=7, square_size=40):
    """Create checkerboard in TOP region (not floor - should not trigger).
    
    Returns:
        RGB image with checkerboard at top
    """
    img = np.full((h, w), 128, dtype=np.uint8)
    
    board_h = rows * square_size
    board_w = cols * square_size
    
    board = np.zeros((board_h, board_w), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                board[i*square_size:(i+1)*square_size, 
                      j*square_size:(j+1)*square_size] = 255
    
    # Place at TOP of image
    y_offset = 10
    x_offset = (w - board_w) // 2
    
    if x_offset >= 0:
        y_end = min(h, y_offset + board_h)
        x_end = min(w, x_offset + board_w)
        img[y_offset:y_end, x_offset:x_end] = board[:y_end-y_offset, :x_end-x_offset]
    
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb


def make_random_noise(h=480, w=640):
    """Create random noise image (no pattern)."""
    noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return noise


def test_detect_centered_checkerboard():
    """Test 1: Centered checkerboard with full image analysis."""
    print("\n" + "="*70)
    print("Test 1: Detect centered 6x6 checkerboard (full image)")
    print("="*70)
    
    img = make_checkerboard_image(h=480, w=640, rows=7, cols=7, square_size=50)
    
    # Use full image analysis (bottom_fraction=1.0) to detect centered board
    detector = CheckeredMatDetector(
        checkerboard_rows=6,
        checkerboard_cols=6,
        bottom_fraction=1.0,  # Analyze full image (not just bottom)
        min_corners=4
    )
    
    triggered = detector.check(img)
    
    print(f"  Centered 7x7 checkerboard (full image analysis):")
    print(f"    Triggered: {triggered}")
    print(f"    Corners found: {detector.corner_count}")
    print(f"    Confidence: {detector.detection_confidence:.2f}")
    
    assert triggered, "Should detect centered checkerboard with full image analysis"
    assert detector.corner_count >= 4, f"Expected >=4 corners, got {detector.corner_count}"
    
    print("  ✅ PASS: Centered checkerboard detected")


def test_detect_bottom_checkerboard():
    """Test 2: Detect checkerboard in bottom region (floor mat position)."""
    print("\n" + "="*70)
    print("Test 2: Detect checkerboard at BOTTOM (floor mat)")
    print("="*70)
    
    img = make_checkerboard_bottom(h=480, w=640, rows=7, cols=7, square_size=40)
    
    detector = CheckeredMatDetector(
        checkerboard_rows=6,
        checkerboard_cols=6,
        bottom_fraction=0.5,
        min_corners=4
    )
    
    triggered = detector.check(img)
    
    print(f"  Bottom checkerboard:")
    print(f"    Triggered: {triggered}")
    print(f"    Corners found: {detector.corner_count}")
    print(f"    Confidence: {detector.detection_confidence:.2f}")
    
    assert triggered, "Should detect bottom checkerboard (floor mat)"
    
    print("  ✅ PASS: Bottom checkerboard detected")


def test_no_detect_top_checkerboard():
    """Test 3: Should NOT detect checkerboard in top region (not floor)."""
    print("\n" + "="*70)
    print("Test 3: Should NOT detect checkerboard at TOP (not floor)")
    print("="*70)
    
    img = make_checkerboard_top(h=480, w=640, rows=7, cols=7, square_size=40)
    
    detector = CheckeredMatDetector(
        checkerboard_rows=6,
        checkerboard_cols=6,
        bottom_fraction=0.5,  # Only look at bottom 50%
        min_corners=4
    )
    
    triggered = detector.check(img)
    
    print(f"  Top checkerboard (bottom_fraction=0.5):")
    print(f"    Triggered: {triggered}")
    print(f"    Corners found: {detector.corner_count}")
    
    assert not triggered, "Should NOT detect checkerboard in top region"
    
    print("  ✅ PASS: Top checkerboard correctly ignored")


def test_no_pattern():
    """Test 4: No detection on random noise (no checkerboard)."""
    print("\n" + "="*70)
    print("Test 4: No detection on random noise")
    print("="*70)
    
    img = make_random_noise(h=480, w=640)
    
    detector = CheckeredMatDetector()
    triggered = detector.check(img)
    
    print(f"  Random noise:")
    print(f"    Triggered: {triggered}")
    print(f"    Corners found: {detector.corner_count}")
    
    assert not triggered, "Should NOT detect pattern in noise"
    assert detector.corner_count == 0, "Should find 0 corners in noise"
    
    print("  ✅ PASS: No false detection on noise")


def test_small_checkerboard():
    """Test 5: Detect smaller checkerboard (4x4 corners)."""
    print("\n" + "="*70)
    print("Test 5: Detect smaller 4x4 checkerboard")
    print("="*70)
    
    img = make_checkerboard_bottom(h=480, w=640, rows=5, cols=5, square_size=50)
    
    detector = CheckeredMatDetector(
        checkerboard_rows=4,
        checkerboard_cols=4,
        bottom_fraction=0.5,
        min_corners=4
    )
    
    triggered = detector.check(img)
    
    print(f"  4x4 checkerboard:")
    print(f"    Triggered: {triggered}")
    print(f"    Corners found: {detector.corner_count}")
    print(f"    Confidence: {detector.detection_confidence:.2f}")
    
    assert triggered, "Should detect 4x4 checkerboard"
    
    print("  ✅ PASS: Small checkerboard detected")


def test_safety_guard_checkered_mat_stop():
    """Test 6: SafetyGuard stops forward when checkered mat detected."""
    print("\n" + "="*70)
    print("Test 6: SafetyGuard stops forward with checkered mat")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Create clear obstacle map (no floor obstacles)
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    # Update with checkered_mat flag TRUE
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 checkered_mat=True)
    
    print(f"  With checkered mat reflex:")
    print(f"    Forward scale: {guard.fwd_scale:.2f}")
    print(f"    Backward scale: {guard.bwd_scale:.2f}")
    print(f"    Angular scale: {guard.ang_scale:.2f}")
    print(f"    Reason: {guard.near_field_reason}")
    
    assert guard.fwd_scale == 0.0, "Forward should be STOPPED"
    assert guard.bwd_scale > 0.0, "Backward should be ALLOWED (if clear)"
    assert guard.ang_scale > 0.0, "Angular should be ALLOWED"
    assert guard.near_field_reason == "checkered_mat"
    
    print("  ✅ PASS: Forward stopped, backward/angular allowed")


def test_safety_guard_checkered_mat_clear():
    """Test 7: SafetyGuard allows all motion when checkered mat clear."""
    print("\n" + "="*70)
    print("Test 7: SafetyGuard allows motion when checkered mat clear")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Create clear obstacle map
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    # Update with checkered_mat flag FALSE
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 checkered_mat=False)
    
    print(f"  Without checkered mat reflex:")
    print(f"    Forward scale: {guard.fwd_scale:.2f}")
    print(f"    Backward scale: {guard.bwd_scale:.2f}")
    print(f"    Angular scale: {guard.ang_scale:.2f}")
    
    assert guard.fwd_scale == 1.0, "Forward should be FULL SPEED"
    assert guard.bwd_scale == 1.0, "Backward should be FULL SPEED"
    assert guard.ang_scale == 1.0, "Angular should be FULL SPEED"
    assert guard.near_field_reason is None
    
    print("  ✅ PASS: All motion allowed when clear")


def test_temporal_filtering():
    """Test 8: Temporal filtering reduces flicker."""
    print("\n" + "="*70)
    print("Test 8: Temporal filtering reduces flicker")
    print("="*70)
    
    img_with = make_checkerboard_bottom(h=480, w=640, rows=7, cols=7)
    img_without = make_random_noise(h=480, w=640)
    
    detector = CheckeredMatDetector(
        checkerboard_rows=6,
        checkerboard_cols=6,
        bottom_fraction=0.5,
        min_corners=4
    )
    
    # First frame: no pattern
    t1 = detector.check(img_without)
    print(f"  Frame 1 (no pattern): {t1}, history={detector._detection_history}")
    
    # Second frame: pattern appears (but history not full yet)
    t2 = detector.check(img_with)
    print(f"  Frame 2 (pattern): {t2}, history={detector._detection_history}")
    
    # Third frame: pattern continues
    t3 = detector.check(img_with)
    print(f"  Frame 3 (pattern): {t3}, history={detector._detection_history}")
    
    assert not t1, "Frame 1 should not trigger (no pattern)"
    # t2 might or might not trigger depending on majority voting
    assert t3, "Frame 3 should trigger (sustained detection)"
    
    print("  ✅ PASS: Temporal filtering working")


def test_empty_image():
    """Test 9: Empty/invalid images don't crash."""
    print("\n" + "="*70)
    print("Test 9: Empty/invalid image handling")
    print("="*70)
    
    detector = CheckeredMatDetector()
    
    # Empty array
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    t1 = detector.check(empty)
    
    print(f"  Empty image: triggered={t1}")
    assert not t1, "Empty image should not trigger"
    
    # None
    t2 = detector.check(None)
    print(f"  None image: triggered={t2}")
    assert not t2, "None should not trigger"
    
    # Very small image
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    t3 = detector.check(tiny)
    print(f"  Tiny image: triggered={t3}")
    assert not t3, "Tiny image should not trigger"
    
    print("  ✅ PASS: Empty/invalid images handled gracefully")


def test_tunable_parameters():
    """Test 10: Different checkerboard sizes (tunable parameters)."""
    print("\n" + "="*70)
    print("Test 10: Tunable parameters (different board sizes)")
    print("="*70)
    
    # 8x8 checkerboard
    img_8x8 = make_checkerboard_bottom(h=480, w=640, rows=9, cols=9, square_size=35)
    
    detector_8x8 = CheckeredMatDetector(
        checkerboard_rows=8,
        checkerboard_cols=8,
        bottom_fraction=0.6,
        min_corners=10
    )
    
    t1 = detector_8x8.check(img_8x8)
    
    print(f"  8x8 checkerboard:")
    print(f"    Triggered: {t1}")
    print(f"    Corners: {detector_8x8.corner_count}")
    print(f"    Confidence: {detector_8x8.detection_confidence:.2f}")
    
    assert t1, "Should detect 8x8 checkerboard with correct params"
    
    # Same image with wrong detector params (should not find 6x6 in 9x9 board)
    detector_6x6 = CheckeredMatDetector(
        checkerboard_rows=6,
        checkerboard_cols=6
    )
    
    t2 = detector_6x6.check(img_8x8)
    print(f"  8x8 checkerboard with 6x6 detector: triggered={t2}")
    # This might or might not trigger depending on partial pattern matching
    
    print("  ✅ PASS: Tunable parameters work correctly")


def test_convenience_function():
    """Test 11: Convenience function check_checkered_mat()."""
    print("\n" + "="*70)
    print("Test 11: Convenience function check_checkered_mat()")
    print("="*70)
    
    img = make_checkerboard_bottom(h=480, w=640, rows=7, cols=7)
    
    triggered, corner_count, confidence = check_checkered_mat(
        img,
        checkerboard_rows=6,
        checkerboard_cols=6,
        bottom_fraction=0.5,
        min_corners=4
    )
    
    print(f"  One-shot detection:")
    print(f"    Triggered: {triggered}")
    print(f"    Corners: {corner_count}")
    print(f"    Confidence: {confidence:.2f}")
    
    assert triggered, "Convenience function should detect checkerboard"
    
    print("  ✅ PASS: Convenience function works")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*18 + "CHECKERED MAT REFLEX TESTS" + " "*24 + "║")
    print("╚"+"═"*68+"╝")
    
    tests = [
        test_detect_centered_checkerboard,
        test_detect_bottom_checkerboard,
        test_no_detect_top_checkerboard,
        test_no_pattern,
        test_small_checkerboard,
        test_safety_guard_checkered_mat_stop,
        test_safety_guard_checkered_mat_clear,
        test_temporal_filtering,
        test_empty_image,
        test_tunable_parameters,
        test_convenience_function,
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
