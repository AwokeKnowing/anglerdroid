#!/usr/bin/env python3
"""
Unit tests for topdown floor hazard hard-stop detection (ego/vision reflex).

Tests the RS1 topdown RGB-based hazard detector that works WITHOUT SLAM or map keepouts.
Detects TWO hazard types:
  1. Wood bump / threshold (edge detection)
  2. Checkered floor mat pattern (corner detection)

Triggers forward hard-stop (fwd_scale=0) while allowing reverse/turn if rear is clear.

Test cases:
1. Synthetic 6x6 checkerboard → detection triggers
2. Checkerboard in forward region → triggers
3. Checkerboard in rear region (not forward) → no trigger
4. Bump detection via edge detection → triggers
5. No pattern or bump → no trigger
6. SafetyGuard integration → fwd=0, bwd/ang computed normally
7. Temporal filtering → reduces flicker
8. Empty/invalid images → no crash
9. Different checkerboard sizes (tunable parameters)
10. Bump + checkered both present → triggers on either
"""

import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from checkered_mat import TopdownHazardDetector, check_topdown_hazard
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


def make_checkerboard_forward(h=480, w=640, rows=7, cols=7, square_size=40):
    """Create checkerboard in FORWARD region (topdown view - top of image).
    
    Note: RS1 topdown view is rotated 180°. After rotation, forward is at TOP.
    
    Returns:
        RGB image with checkerboard at top (forward region in topdown)
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
    
    # Place at TOP of image (forward region in topdown view)
    y_offset = 10
    x_offset = (w - board_w) // 2
    
    if y_offset >= 0 and x_offset >= 0:
        y_end = min(h, y_offset + board_h)
        x_end = min(w, x_offset + board_w)
        img[y_offset:y_end, x_offset:x_end] = board[:y_end-y_offset, :x_end-x_offset]
    
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb


def make_checkerboard_rear(h=480, w=640, rows=7, cols=7, square_size=40):
    """Create checkerboard in REAR region (topdown view - bottom of image).
    
    Should NOT trigger (not in forward path).
    
    Returns:
        RGB image with checkerboard at bottom (rear region in topdown)
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
    
    # Place at BOTTOM of image (rear region in topdown view)
    y_offset = h - board_h - 10
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


def make_bump_image(h=480, w=640, bump_y=100, bump_thickness=10):
    """Create image with horizontal edge (simulating wood bump/threshold).
    
    Args:
        h, w: Image dimensions
        bump_y: Y position of bump (horizontal line)
        bump_thickness: Thickness of transition edge (sharp edge for detection)
    
    Returns:
        RGB image with horizontal edge at bump_y
    """
    img = np.full((h, w), 150, dtype=np.uint8)
    
    # Create sharp dark-to-light transition (bump/threshold)
    img[:bump_y, :] = 60   # Much darker before bump (strong contrast)
    img[bump_y+bump_thickness:, :] = 230  # Much lighter after bump
    
    # Sharp transition (not too smooth, so Canny finds it easily)
    for i in range(bump_thickness):
        blend = i / bump_thickness
        img[bump_y+i, :] = int(60 * (1-blend) + 230 * blend)
    
    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb


def test_detect_centered_checkerboard():
    """Test 1: Centered checkerboard with full image analysis."""
    print("\n" + "="*70)
    print("Test 1: Detect centered 6x6 checkerboard (full image)")
    print("="*70)
    
    img = make_checkerboard_image(h=480, w=640, rows=7, cols=7, square_size=50)
    
    # Use full image analysis (forward_fraction=1.0) to detect centered board
    detector = TopdownHazardDetector(
        checkerboard_rows=6,
        checkerboard_cols=6,
        forward_fraction=1.0,  # Analyze full image
        min_corners=4
    )
    
    triggered, reason = detector.check(img)
    
    print(f"  Centered 7x7 checkerboard (full image analysis):")
    print(f"    Triggered: {triggered}")
    print(f"    Reason: {reason}")
    print(f"    Corners found: {detector.corner_count}")
    
    assert triggered, "Should detect centered checkerboard with full image analysis"
    assert reason == 'checkered', f"Expected reason='checkered', got '{reason}'"
    assert detector.corner_count >= 4, f"Expected >=4 corners, got {detector.corner_count}"
    
    print("  ✅ PASS: Centered checkerboard detected")


def test_detect_forward_checkerboard():
    """Test 2: Detect checkerboard in forward region (topdown view)."""
    print("\n" + "="*70)
    print("Test 2: Detect checkerboard in FORWARD region (topdown)")
    print("="*70)
    
    img = make_checkerboard_forward(h=480, w=640, rows=7, cols=7, square_size=40)
    
    detector = TopdownHazardDetector(
        checkerboard_rows=6,
        checkerboard_cols=6,
        forward_fraction=0.4,  # Analyze forward 40%
        min_corners=4
    )
    
    triggered, reason = detector.check(img)
    
    print(f"  Forward checkerboard:")
    print(f"    Triggered: {triggered}")
    print(f"    Reason: {reason}")
    print(f"    Corners found: {detector.corner_count}")
    
    assert triggered, "Should detect forward checkerboard in topdown view"
    assert reason == 'checkered', f"Expected reason='checkered', got '{reason}'"
    
    print("  ✅ PASS: Forward checkerboard detected")


def test_no_detect_rear_checkerboard():
    """Test 3: Should NOT detect checkerboard in rear region."""
    print("\n" + "="*70)
    print("Test 3: Should NOT detect checkerboard in REAR (not forward)")
    print("="*70)
    
    img = make_checkerboard_rear(h=480, w=640, rows=7, cols=7, square_size=40)
    
    detector = TopdownHazardDetector(
        checkerboard_rows=6,
        checkerboard_cols=6,
        forward_fraction=0.4,  # Only analyze forward 40%
        min_corners=4
    )
    
    triggered, reason = detector.check(img)
    
    print(f"  Rear checkerboard (forward_fraction=0.4):")
    print(f"    Triggered: {triggered}")
    print(f"    Corners found: {detector.corner_count}")
    
    assert not triggered, "Should NOT detect checkerboard in rear region"
    
    print("  ✅ PASS: Rear checkerboard correctly ignored")


def test_detect_bump():
    """Test 4: Detect wood bump via edge detection."""
    print("\n" + "="*70)
    print("Test 4: Detect wood bump (horizontal edge)")
    print("="*70)
    
    img = make_bump_image(h=480, w=640, bump_y=80, bump_thickness=8)
    
    detector = TopdownHazardDetector(
        forward_fraction=0.4,
        bump_edge_thresh=50
    )
    
    triggered, reason = detector.check(img)
    
    print(f"  Bump image:")
    print(f"    Triggered: {triggered}")
    print(f"    Reason: {reason}")
    print(f"    Edge count (consecutive rows): {detector.edge_count}")
    
    assert triggered, "Should detect bump via edge detection"
    assert reason == 'bump', f"Expected reason='bump', got '{reason}'"
    assert detector.edge_count >= 1, f"Expected >=1 consecutive edge rows, got {detector.edge_count}"
    
    print("  ✅ PASS: Bump detected via edge detection")


def test_no_pattern():
    """Test 5: No detection on random noise (no checkerboard or bump)."""
    print("\n" + "="*70)
    print("Test 5: No detection on random noise")
    print("="*70)
    
    img = make_random_noise(h=480, w=640)
    
    detector = TopdownHazardDetector()
    triggered, reason = detector.check(img)
    
    print(f"  Random noise:")
    print(f"    Triggered: {triggered}")
    print(f"    Corners found: {detector.corner_count}")
    print(f"    Edges found: {detector.edge_count}")
    
    assert not triggered, "Should NOT detect pattern in noise"
    
    print("  ✅ PASS: No false detection on noise")


def test_safety_guard_topdown_hazard_stop():
    """Test 6: SafetyGuard stops forward when topdown hazard detected."""
    print("\n" + "="*70)
    print("Test 6: SafetyGuard stops forward with topdown hazard")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Create clear obstacle map (no floor obstacles)
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    # Update with topdown_hazard flag TRUE
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 topdown_hazard=True)
    
    print(f"  With topdown hazard reflex:")
    print(f"    Forward scale: {guard.fwd_scale:.2f}")
    print(f"    Backward scale: {guard.bwd_scale:.2f}")
    print(f"    Angular scale: {guard.ang_scale:.2f}")
    print(f"    Reason: {guard.near_field_reason}")
    
    assert guard.fwd_scale == 0.0, "Forward should be STOPPED"
    assert guard.bwd_scale > 0.0, "Backward should be ALLOWED (if clear)"
    assert guard.ang_scale > 0.0, "Angular should be ALLOWED"
    assert guard.near_field_reason == "topdown_hazard"
    
    print("  ✅ PASS: Forward stopped, backward/angular allowed")


def test_safety_guard_topdown_hazard_clear():
    """Test 7: SafetyGuard allows all motion when topdown hazard clear."""
    print("\n" + "="*70)
    print("Test 7: SafetyGuard allows motion when topdown hazard clear")
    print("="*70)
    
    guard = SafetyGuard()
    
    # Create clear obstacle map
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    # Update with topdown_hazard flag FALSE
    guard.update(obs_map, yaw_delta=0.0, fwd_delta=0.0, 
                 topdown_hazard=False)
    
    print(f"  Without topdown hazard reflex:")
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
    
    img_with = make_checkerboard_forward(h=480, w=640, rows=7, cols=7)
    img_without = make_random_noise(h=480, w=640)
    
    detector = TopdownHazardDetector(
        checkerboard_rows=6,
        checkerboard_cols=6,
        forward_fraction=0.4,
        min_corners=4
    )
    
    # First frame: no pattern
    t1, _ = detector.check(img_without)
    print(f"  Frame 1 (no pattern): {t1}, history={detector._detection_history}")
    
    # Second frame: pattern appears (but history not full yet)
    t2, _ = detector.check(img_with)
    print(f"  Frame 2 (pattern): {t2}, history={detector._detection_history}")
    
    # Third frame: pattern continues
    t3, _ = detector.check(img_with)
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
    
    detector = TopdownHazardDetector()
    
    # Empty array
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    t1, _ = detector.check(empty)
    
    print(f"  Empty image: triggered={t1}")
    assert not t1, "Empty image should not trigger"
    
    # None
    t2, _ = detector.check(None)
    print(f"  None image: triggered={t2}")
    assert not t2, "None should not trigger"
    
    # Very small image
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    t3, _ = detector.check(tiny)
    print(f"  Tiny image: triggered={t3}")
    assert not t3, "Tiny image should not trigger"
    
    print("  ✅ PASS: Empty/invalid images handled gracefully")


def test_tunable_parameters():
    """Test 10: Different checkerboard sizes and bump thresholds (tunable parameters)."""
    print("\n" + "="*70)
    print("Test 10: Tunable parameters (different sizes/thresholds)")
    print("="*70)
    
    # 8x8 checkerboard
    img_8x8 = make_checkerboard_forward(h=480, w=640, rows=9, cols=9, square_size=25)
    
    detector_8x8 = TopdownHazardDetector(
        checkerboard_rows=8,
        checkerboard_cols=8,
        forward_fraction=0.5,
        min_corners=10
    )
    
    t1, r1 = detector_8x8.check(img_8x8)
    
    print(f"  8x8 checkerboard:")
    print(f"    Triggered: {t1}")
    print(f"    Reason: {r1}")
    print(f"    Corners: {detector_8x8.corner_count}")
    
    assert t1, "Should detect 8x8 checkerboard with correct params"
    
    # Bump with custom thresholds (make bump more prominent)
    img_bump = make_bump_image(h=480, w=640, bump_y=60, bump_thickness=10)
    
    detector_bump = TopdownHazardDetector(
        bump_edge_thresh=40,
        forward_fraction=0.3
    )
    
    t2, r2 = detector_bump.check(img_bump)
    print(f"  Bump with custom thresholds:")
    print(f"    Triggered: {t2}")
    print(f"    Reason: {r2}")
    print(f"    Consecutive edge rows: {detector_bump.edge_count}")
    
    assert t2, "Should detect bump with custom thresholds"
    assert r2 == 'bump', f"Expected bump, got {r2}"
    
    print("  ✅ PASS: Tunable parameters work correctly")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*16 + "TOPDOWN HAZARD REFLEX TESTS" + " "*25 + "║")
    print("╚"+"═"*68+"╝")
    
    tests = [
        test_detect_centered_checkerboard,
        test_detect_forward_checkerboard,
        test_no_detect_rear_checkerboard,
        test_detect_bump,
        test_no_pattern,
        test_safety_guard_topdown_hazard_stop,
        test_safety_guard_topdown_hazard_clear,
        test_temporal_filtering,
        test_empty_image,
        test_tunable_parameters,
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
