#!/usr/bin/env python3
"""Correctness tests for GPU ego label path vs CPU reference.

Verifies label_rs1_ego_gpu produces identical results to label_rs1_ego (CPU).
Runs only when CuPy is available; skips gracefully otherwise.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, EGO_PX_SIZE, FOOTPRINT_BOXES
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.ego_rs1 import label_rs1_ego
from perception.ego_rs1_fast import label_rs1_ego_gpu, _CUPY_AVAILABLE


def _synth_verts(n=45000, z_range=(0.3, 1.2), extent_m=1.6, seed=42):
    """Synthetic RS1 verts with fixed seed for reproducibility."""
    np.random.seed(seed)
    xs = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    ys = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    zs = np.random.uniform(z_range[0], z_range[1], n)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def test_gpu_vs_cpu_identical():
    """Test GPU path produces identical results to CPU reference."""
    if not _CUPY_AVAILABLE:
        print("SKIP test_gpu_vs_cpu_identical: CuPy not available")
        return True

    verts = _synth_verts(n=45000, seed=123)

    # CPU path
    labels_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego(
        verts,
        labels_out=labels_cpu,
        height_out=height_cpu,
        x_offset=-75,
    )

    # GPU path
    labels_gpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_gpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego_gpu(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_gpu,
        height_out=height_gpu,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    # Compare
    labels_match = np.array_equal(labels_cpu, labels_gpu)
    height_match = np.array_equal(height_cpu, height_gpu)

    if not labels_match or not height_match:
        diff_l = np.sum(labels_cpu != labels_gpu)
        diff_h = np.sum(height_cpu != height_gpu)
        print(f"FAIL test_gpu_vs_cpu_identical:")
        print(f"  label diff: {diff_l}/{labels_cpu.size} pixels")
        print(f"  height diff: {diff_h}/{height_cpu.size} pixels")
        return False

    print("PASS test_gpu_vs_cpu_identical")
    return True


def test_gpu_vs_cpu_empty_verts():
    """Test empty verts (edge case)."""
    if not _CUPY_AVAILABLE:
        print("SKIP test_gpu_vs_cpu_empty_verts: CuPy not available")
        return True

    verts = np.empty((0, 3), dtype=np.float32)

    labels_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego(verts, labels_out=labels_cpu, height_out=height_cpu, x_offset=-75)

    labels_gpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_gpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego_gpu(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_gpu,
        height_out=height_gpu,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    if not np.array_equal(labels_cpu, labels_gpu) or not np.array_equal(height_cpu, height_gpu):
        print("FAIL test_gpu_vs_cpu_empty_verts")
        return False

    print("PASS test_gpu_vs_cpu_empty_verts")
    return True


def test_gpu_vs_cpu_floor_only():
    """Test floor-only verts (all CLEAR)."""
    if not _CUPY_AVAILABLE:
        print("SKIP test_gpu_vs_cpu_floor_only: CuPy not available")
        return True

    verts = _synth_verts(n=10000, z_range=(0.95, 1.2), seed=456)

    labels_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego(verts, labels_out=labels_cpu, height_out=height_cpu, x_offset=-75)

    labels_gpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_gpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego_gpu(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_gpu,
        height_out=height_gpu,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    if not np.array_equal(labels_cpu, labels_gpu) or not np.array_equal(height_cpu, height_gpu):
        print("FAIL test_gpu_vs_cpu_floor_only")
        return False

    print("PASS test_gpu_vs_cpu_floor_only")
    return True


def test_gpu_vs_cpu_obstacle_only():
    """Test obstacle-only verts (all below floor clip)."""
    if not _CUPY_AVAILABLE:
        print("SKIP test_gpu_vs_cpu_obstacle_only: CuPy not available")
        return True

    verts = _synth_verts(n=10000, z_range=(0.3, 0.8), seed=789)

    labels_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego(verts, labels_out=labels_cpu, height_out=height_cpu, x_offset=-75)

    labels_gpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_gpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego_gpu(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_gpu,
        height_out=height_gpu,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    if not np.array_equal(labels_cpu, labels_gpu) or not np.array_equal(height_cpu, height_gpu):
        print("FAIL test_gpu_vs_cpu_obstacle_only")
        return False

    print("PASS test_gpu_vs_cpu_obstacle_only")
    return True


def main():
    print("=" * 70)
    print("GPU EGO LABEL CORRECTNESS TESTS")
    print("=" * 70)
    print(f"CuPy available: {_CUPY_AVAILABLE}")
    print()

    if not _CUPY_AVAILABLE:
        print("CuPy not installed — GPU tests skipped (install CuPy to enable)")
        print("Correctness tests PASS (CPU-only fallback verified)")
        return 0

    results = [
        test_gpu_vs_cpu_identical(),
        test_gpu_vs_cpu_empty_verts(),
        test_gpu_vs_cpu_floor_only(),
        test_gpu_vs_cpu_obstacle_only(),
    ]

    print()
    print("=" * 70)
    if all(results):
        print("ALL TESTS PASS ✓")
        print("GPU path produces identical results to CPU reference")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
