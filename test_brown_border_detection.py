#!/usr/bin/env python3
"""Unit tests for brown border detection (door mat keepout)."""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from checkered_mat import detect_brown_border, TopdownHazardDetector

def make_brown_border_image(h=240, w=320, border_width=30):
    """Create synthetic brown border on tan carpet."""
    img = np.full((h, w, 3), (200, 180, 150), dtype=np.uint8)
    hsv_brown = np.array([[[15, 100, 80]]], dtype=np.uint8)
    rgb_brown = cv2.cvtColor(hsv_brown, cv2.COLOR_HSV2RGB)[0, 0]
    
    m, bw = 20, border_width
    x0, y0, x1, y1 = m, m, w-m, h-m
    img[y0:y0+bw, x0:x1] = rgb_brown
    img[y1-bw:y1, x0:x1] = rgb_brown
    img[y0:y1, x0:x0+bw] = rgb_brown
    img[y0:y1, x1-bw:x1] = rgb_brown
    return img

def test_brown_border():
    print("\nTest: Brown border detection")
    img = make_brown_border_image(h=240, w=320, border_width=30)
    detected, score, brown_px = detect_brown_border(img)
    print(f"  Detected: {detected}, score: {score:.3f}, px: {brown_px}")
    assert detected, "Should detect brown border"
    assert score > 0.25, f"Score {score:.3f} too low"
    assert brown_px > 1000, f"Only {brown_px} brown pixels"
    print("  ✅ PASS")

def test_plain_carpet():
    print("\nTest: Plain carpet (no border)")
    img = np.full((240, 320, 3), (200, 180, 150), dtype=np.uint8)
    detected, score, brown_px = detect_brown_border(img)
    print(f"  Detected: {detected}, score: {score:.3f}, px: {brown_px}")
    assert not detected, "Should NOT detect on plain carpet"
    print("  ✅ PASS")

def test_topdown_integration():
    print("\nTest: TopdownHazardDetector integration")
    full = np.full((480, 640, 3), (200, 180, 150), dtype=np.uint8)
    hsv_brown = np.array([[[15, 100, 80]]], dtype=np.uint8)
    rgb_brown = cv2.cvtColor(hsv_brown, cv2.COLOR_HSV2RGB)[0, 0]
    
    fh = int(480 * 0.4)
    bw, m = 30, 30
    x0, y0, x1, y1 = m, m, 640-m, fh-m
    full[y0:y0+bw, x0:x1] = rgb_brown
    full[y1-bw:y1, x0:x1] = rgb_brown
    full[y0:y1, x0:x0+bw] = rgb_brown
    full[y0:y1, x1-bw:x1] = rgb_brown
    
    detector = TopdownHazardDetector(forward_fraction=0.4)
    triggered, reason = detector.check(full)
    print(f"  Triggered: {triggered}, reason: {reason}")
    assert triggered, "Should trigger"
    assert reason == 'brown_border', f"Expected 'brown_border', got '{reason}'"
    print("  ✅ PASS")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BROWN BORDER DETECTION TESTS")
    print("="*60)
    try:
        test_brown_border()
        test_plain_carpet()
        test_topdown_integration()
        print("\n✅ ALL TESTS PASSED\n")
    except AssertionError as e:
        print(f"\n❌ FAIL: {e}\n")
        sys.exit(1)
