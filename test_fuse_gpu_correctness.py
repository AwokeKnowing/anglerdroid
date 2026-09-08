"""Test GPU fuse_rs2_into_ego_gpu correctness vs CPU reference.

Synthetic data: RS1 labels + RS2 obs/known → fuse → verify GPU matches CPU.
"""
import numpy as np
import sys
import os

# Make perception module available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from perception import (
    UNKNOWN, SELF, CLEAR, OBSTACLE,
    fuse_rs2_into_ego,
    fuse_rs2_into_ego_gpu,
    GPU_AVAILABLE,
)
from robot_config import FRAME_H, FRAME_W, FOOTPRINT_BOXES, UNDER_ROBOT_BOXES


def test_fuse_gpu_vs_cpu_empty():
    """Test empty inputs: GPU should match CPU exactly."""
    h, w = FRAME_H, FRAME_W
    labels_rs1 = np.zeros((h, w), dtype=np.uint8)
    height_rs1 = np.zeros((h, w), dtype=np.uint8)
    obs2 = np.zeros((h, w), dtype=np.uint8)
    known2 = np.zeros((h, w), dtype=np.uint8)

    cpu_labels, cpu_height, cpu_metrics = fuse_rs2_into_ego(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0)

    gpu_labels, gpu_height, gpu_metrics = fuse_rs2_into_ego_gpu(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0)

    assert np.array_equal(cpu_labels, gpu_labels), "Labels mismatch on empty"
    assert np.array_equal(cpu_height, gpu_height), "Height mismatch on empty"
    assert cpu_metrics == gpu_metrics, "Metrics mismatch on empty"
    print("✓ Empty inputs: GPU matches CPU")


def test_fuse_gpu_vs_cpu_obstacles():
    """Test RS2 obstacles: GPU should match CPU."""
    h, w = FRAME_H, FRAME_W
    labels_rs1 = np.zeros((h, w), dtype=np.uint8)
    height_rs1 = np.zeros((h, w), dtype=np.uint8)
    
    # Add some RS1 clear in center
    labels_rs1[100:140, 100:140] = CLEAR
    
    # Add RS2 obstacles
    obs2 = np.zeros((h, w), dtype=np.uint8)
    obs2[50:80, 60:100] = 30  # 30 cm obstacle
    obs2[150:180, 150:180] = 50  # 50 cm obstacle
    known2 = np.zeros((h, w), dtype=np.uint8)
    known2[50:80, 60:100] = 255
    known2[150:180, 150:180] = 255

    cpu_labels, cpu_height, cpu_metrics = fuse_rs2_into_ego(
        labels_rs1, height_rs1, obs2, known2, fw_dx=10, fw_dy=5)

    gpu_labels, gpu_height, gpu_metrics = fuse_rs2_into_ego_gpu(
        labels_rs1, height_rs1, obs2, known2, fw_dx=10, fw_dy=5)

    assert np.array_equal(cpu_labels, gpu_labels), "Labels mismatch with obstacles"
    assert np.array_equal(cpu_height, gpu_height), "Height mismatch with obstacles"
    assert cpu_metrics == gpu_metrics, f"Metrics mismatch: CPU={cpu_metrics}, GPU={gpu_metrics}"
    print("✓ Obstacles: GPU matches CPU")


def test_fuse_gpu_vs_cpu_clear_in_range():
    """Test RS2 CLEAR within free_range: GPU should match CPU."""
    h, w = FRAME_H, FRAME_W
    labels_rs1 = np.zeros((h, w), dtype=np.uint8)
    height_rs1 = np.zeros((h, w), dtype=np.uint8)
    
    # RS2 CLEAR
    obs2 = np.zeros((h, w), dtype=np.uint8)
    known2 = np.zeros((h, w), dtype=np.uint8)
    known2[80:120, 80:120] = 255  # Known CLEAR region
    
    # Free range mask
    free_range = np.zeros((h, w), dtype=np.uint8)
    free_range[70:130, 70:130] = 255

    cpu_labels, cpu_height, cpu_metrics = fuse_rs2_into_ego(
        labels_rs1, height_rs1, obs2, known2, fw_dx=5, fw_dy=0,
        free_range=free_range)

    gpu_labels, gpu_height, gpu_metrics = fuse_rs2_into_ego_gpu(
        labels_rs1, height_rs1, obs2, known2, fw_dx=5, fw_dy=0,
        free_range=free_range)

    assert np.array_equal(cpu_labels, gpu_labels), "Labels mismatch with free_range"
    assert np.array_equal(cpu_height, gpu_height), "Height mismatch with free_range"
    assert cpu_metrics == gpu_metrics, f"Metrics mismatch: CPU={cpu_metrics}, GPU={gpu_metrics}"
    print("✓ CLEAR in range: GPU matches CPU")


def test_fuse_gpu_vs_cpu_cone_mask():
    """Test RS2 with cone mask: GPU should match CPU."""
    h, w = FRAME_H, FRAME_W
    labels_rs1 = np.zeros((h, w), dtype=np.uint8)
    height_rs1 = np.zeros((h, w), dtype=np.uint8)
    
    # RS2 obstacles
    obs2 = np.zeros((h, w), dtype=np.uint8)
    obs2[60:100, 60:100] = 25
    known2 = np.zeros((h, w), dtype=np.uint8)
    known2[60:100, 60:100] = 255
    
    # Cone mask (exclude some regions)
    fw_cone = np.zeros((h, w), dtype=np.uint8)
    fw_cone[50:150, 50:150] = 255

    cpu_labels, cpu_height, cpu_metrics = fuse_rs2_into_ego(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0,
        fw_cone=fw_cone)

    gpu_labels, gpu_height, gpu_metrics = fuse_rs2_into_ego_gpu(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0,
        fw_cone=fw_cone)

    assert np.array_equal(cpu_labels, gpu_labels), "Labels mismatch with cone"
    assert np.array_equal(cpu_height, gpu_height), "Height mismatch with cone"
    assert cpu_metrics == gpu_metrics, f"Metrics mismatch: CPU={cpu_metrics}, GPU={gpu_metrics}"
    print("✓ Cone mask: GPU matches CPU")


def test_fuse_gpu_vs_cpu_no_clear_over_obstacle():
    """Test RS2 CLEAR does not override RS1 OBSTACLE: GPU should match CPU."""
    h, w = FRAME_H, FRAME_W
    labels_rs1 = np.zeros((h, w), dtype=np.uint8)
    height_rs1 = np.zeros((h, w), dtype=np.uint8)
    
    # RS1 obstacle
    labels_rs1[100:120, 100:120] = OBSTACLE
    height_rs1[100:120, 100:120] = 40
    
    # RS2 tries to mark as CLEAR (should be rejected)
    obs2 = np.zeros((h, w), dtype=np.uint8)
    known2 = np.zeros((h, w), dtype=np.uint8)
    known2[100:120, 100:120] = 255  # RS2 known CLEAR
    
    free_range = np.ones((h, w), dtype=np.uint8) * 255

    cpu_labels, cpu_height, cpu_metrics = fuse_rs2_into_ego(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0,
        free_range=free_range)

    gpu_labels, gpu_height, gpu_metrics = fuse_rs2_into_ego_gpu(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0,
        free_range=free_range)

    assert np.array_equal(cpu_labels, gpu_labels), "Labels mismatch (no CLEAR over OBS)"
    assert np.array_equal(cpu_height, gpu_height), "Height mismatch (no CLEAR over OBS)"
    # RS1 obstacle should remain
    assert np.all(cpu_labels[100:120, 100:120] == OBSTACLE), "RS1 obstacle lost"
    assert cpu_metrics == gpu_metrics, f"Metrics mismatch: CPU={cpu_metrics}, GPU={gpu_metrics}"
    print("✓ No CLEAR over OBSTACLE: GPU matches CPU")


def test_fuse_gpu_vs_cpu_self_wins():
    """Test SELF painted on top: GPU should match CPU."""
    h, w = FRAME_H, FRAME_W
    labels_rs1 = np.zeros((h, w), dtype=np.uint8)
    height_rs1 = np.zeros((h, w), dtype=np.uint8)
    
    # RS2 obstacle that overlaps with SELF boxes
    obs2 = np.ones((h, w), dtype=np.uint8) * 30
    known2 = np.ones((h, w), dtype=np.uint8) * 255

    cpu_labels, cpu_height, cpu_metrics = fuse_rs2_into_ego(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0)

    gpu_labels, gpu_height, gpu_metrics = fuse_rs2_into_ego_gpu(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0)

    assert np.array_equal(cpu_labels, gpu_labels), "Labels mismatch (SELF wins)"
    assert np.array_equal(cpu_height, gpu_height), "Height mismatch (SELF wins)"
    
    # Verify SELF boxes are painted
    for x0, y0, x1, y1 in FOOTPRINT_BOXES:
        assert np.all(cpu_labels[y0:y1, x0:x1] == SELF), "SELF not painted in CPU"
        assert np.all(gpu_labels[y0:y1, x0:x1] == SELF), "SELF not painted in GPU"
        assert np.all(cpu_height[y0:y1, x0:x1] == 0), "SELF height not zero in CPU"
        assert np.all(gpu_height[y0:y1, x0:x1] == 0), "SELF height not zero in GPU"
    
    assert cpu_metrics == gpu_metrics, f"Metrics mismatch: CPU={cpu_metrics}, GPU={gpu_metrics}"
    print("✓ SELF wins: GPU matches CPU")


def test_fuse_gpu_vs_cpu_under_chassis_metric():
    """Test rs2_clear_under_pre metric counts correctly: GPU should match CPU."""
    h, w = FRAME_H, FRAME_W
    labels_rs1 = np.zeros((h, w), dtype=np.uint8)
    height_rs1 = np.zeros((h, w), dtype=np.uint8)
    
    # RS2 CLEAR under chassis
    obs2 = np.zeros((h, w), dtype=np.uint8)
    known2 = np.zeros((h, w), dtype=np.uint8)
    # Mark entire under-robot region as CLEAR from RS2
    for x0, y0, x1, y1 in UNDER_ROBOT_BOXES:
        known2[y0:y1, x0:x1] = 255

    cpu_labels, cpu_height, cpu_metrics = fuse_rs2_into_ego(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0)

    gpu_labels, gpu_height, gpu_metrics = fuse_rs2_into_ego_gpu(
        labels_rs1, height_rs1, obs2, known2, fw_dx=0, fw_dy=0)

    assert np.array_equal(cpu_labels, gpu_labels), "Labels mismatch (under chassis)"
    assert np.array_equal(cpu_height, gpu_height), "Height mismatch (under chassis)"
    assert cpu_metrics["rs2_clear_under_pre"] > 0, "Should detect false CLEAR under chassis"
    assert cpu_metrics == gpu_metrics, f"Metrics mismatch: CPU={cpu_metrics}, GPU={gpu_metrics}"
    print(f"✓ Under chassis metric: GPU matches CPU (detected {cpu_metrics['rs2_clear_under_pre']} false CLEAR)")


def run_all_tests():
    """Run all GPU fuse correctness tests."""
    if not GPU_AVAILABLE:
        print("⚠ CuPy not available, GPU tests will fall back to CPU")
        print("  (This is OK — tests verify fallback works correctly)")
        print()
    
    tests = [
        test_fuse_gpu_vs_cpu_empty,
        test_fuse_gpu_vs_cpu_obstacles,
        test_fuse_gpu_vs_cpu_clear_in_range,
        test_fuse_gpu_vs_cpu_cone_mask,
        test_fuse_gpu_vs_cpu_no_clear_over_obstacle,
        test_fuse_gpu_vs_cpu_self_wins,
        test_fuse_gpu_vs_cpu_under_chassis_metric,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed.append(test.__name__)
    
    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed")
        if GPU_AVAILABLE:
            print("  (GPU path verified)")
        else:
            print("  (CPU fallback verified)")


if __name__ == "__main__":
    run_all_tests()
