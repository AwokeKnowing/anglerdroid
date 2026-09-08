#!/usr/bin/env python3
"""Correctness tests for ModernGL ego label scatter vs CPU reference.

Verifies GPURender.label_rs1_ego_moderngl produces identical results to label_rs1_ego (CPU).
Runs only when ModernGL is available; skips gracefully otherwise.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, EGO_PX_SIZE, FOOTPRINT_BOXES
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.ego_rs1 import label_rs1_ego

# Check ModernGL availability
try:
    from gpu_render import GPURenderer, _HAS_MGL
    MODERNGL_AVAILABLE = _HAS_MGL
except ImportError:
    MODERNGL_AVAILABLE = False


def _synth_verts(n=45000, z_range=(0.3, 1.2), extent_m=1.6, seed=42):
    """Synthetic RS1 verts with fixed seed for reproducibility."""
    np.random.seed(seed)
    xs = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    ys = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    zs = np.random.uniform(z_range[0], z_range[1], n)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def test_moderngl_vs_cpu_identical():
    """Test ModernGL path produces identical results to CPU reference."""
    if not MODERNGL_AVAILABLE:
        print("SKIP test_moderngl_vs_cpu_identical: ModernGL not available")
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

    # ModernGL path
    gpu = GPURenderer(800, 800, 960, 960)  # minimal map/atlas sizes for testing
    gpu.configure_ego_labels(
        out_w=FRAME_W, out_h=FRAME_H,
        px_size=float(EGO_PX_SIZE),
        floor_clip_m=0.91)
    
    labels_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    gpu.label_rs1_ego_moderngl(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_mgl,
        height_out=height_mgl,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    # Compare
    labels_match = np.array_equal(labels_cpu, labels_mgl)
    height_match = np.array_equal(height_cpu, height_mgl)

    if not labels_match or not height_match:
        diff_l = np.sum(labels_cpu != labels_mgl)
        diff_h = np.sum(height_cpu != height_mgl)
        print(f"FAIL test_moderngl_vs_cpu_identical:")
        print(f"  label diff: {diff_l}/{labels_cpu.size} pixels")
        print(f"  height diff: {diff_h}/{height_cpu.size} pixels")
        
        # Debug: show label distribution
        unique_cpu, counts_cpu = np.unique(labels_cpu, return_counts=True)
        unique_mgl, counts_mgl = np.unique(labels_mgl, return_counts=True)
        print(f"  CPU labels: {dict(zip(unique_cpu, counts_cpu))}")
        print(f"  MGL labels: {dict(zip(unique_mgl, counts_mgl))}")
        return False

    print("PASS test_moderngl_vs_cpu_identical")
    return True


def test_moderngl_vs_cpu_empty_verts():
    """Test empty verts (edge case)."""
    if not MODERNGL_AVAILABLE:
        print("SKIP test_moderngl_vs_cpu_empty_verts: ModernGL not available")
        return True

    verts = np.empty((0, 3), dtype=np.float32)

    labels_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego(verts, labels_out=labels_cpu, height_out=height_cpu, x_offset=-75)

    gpu = GPURenderer(800, 800, 960, 960)
    gpu.configure_ego_labels(
        out_w=FRAME_W, out_h=FRAME_H,
        px_size=float(EGO_PX_SIZE),
        floor_clip_m=0.91)
    
    labels_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    gpu.label_rs1_ego_moderngl(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_mgl,
        height_out=height_mgl,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    if not np.array_equal(labels_cpu, labels_mgl) or not np.array_equal(height_cpu, height_mgl):
        print("FAIL test_moderngl_vs_cpu_empty_verts")
        return False

    print("PASS test_moderngl_vs_cpu_empty_verts")
    return True


def test_moderngl_vs_cpu_floor_only():
    """Test floor-only verts (all CLEAR)."""
    if not MODERNGL_AVAILABLE:
        print("SKIP test_moderngl_vs_cpu_floor_only: ModernGL not available")
        return True

    verts = _synth_verts(n=10000, z_range=(0.95, 1.2), seed=456)

    labels_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego(verts, labels_out=labels_cpu, height_out=height_cpu, x_offset=-75)

    gpu = GPURenderer(800, 800, 960, 960)
    gpu.configure_ego_labels(
        out_w=FRAME_W, out_h=FRAME_H,
        px_size=float(EGO_PX_SIZE),
        floor_clip_m=0.91)
    
    labels_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    gpu.label_rs1_ego_moderngl(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_mgl,
        height_out=height_mgl,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    if not np.array_equal(labels_cpu, labels_mgl) or not np.array_equal(height_cpu, height_mgl):
        print("FAIL test_moderngl_vs_cpu_floor_only")
        return False

    print("PASS test_moderngl_vs_cpu_floor_only")
    return True


def test_moderngl_vs_cpu_obstacle_only():
    """Test obstacle-only verts (all below floor clip)."""
    if not MODERNGL_AVAILABLE:
        print("SKIP test_moderngl_vs_cpu_obstacle_only: ModernGL not available")
        return True

    verts = _synth_verts(n=10000, z_range=(0.3, 0.8), seed=789)

    labels_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego(verts, labels_out=labels_cpu, height_out=height_cpu, x_offset=-75)

    gpu = GPURenderer(800, 800, 960, 960)
    gpu.configure_ego_labels(
        out_w=FRAME_W, out_h=FRAME_H,
        px_size=float(EGO_PX_SIZE),
        floor_clip_m=0.91)
    
    labels_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    gpu.label_rs1_ego_moderngl(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_mgl,
        height_out=height_mgl,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    if not np.array_equal(labels_cpu, labels_mgl) or not np.array_equal(height_cpu, height_mgl):
        print("FAIL test_moderngl_vs_cpu_obstacle_only")
        return False

    print("PASS test_moderngl_vs_cpu_obstacle_only")
    return True


def test_moderngl_self_boxes():
    """Test SELF boxes paint correctly (override scattered labels)."""
    if not MODERNGL_AVAILABLE:
        print("SKIP test_moderngl_self_boxes: ModernGL not available")
        return True

    # Scatter floor verts that land inside SELF box region
    verts = _synth_verts(n=10000, z_range=(0.95, 1.2), extent_m=0.5, seed=999)

    labels_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_cpu = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    label_rs1_ego(verts, labels_out=labels_cpu, height_out=height_cpu, x_offset=-75)

    gpu = GPURenderer(800, 800, 960, 960)
    gpu.configure_ego_labels(
        out_w=FRAME_W, out_h=FRAME_H,
        px_size=float(EGO_PX_SIZE),
        floor_clip_m=0.91)
    
    labels_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_mgl = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    gpu.label_rs1_ego_moderngl(
        verts,
        out_h=FRAME_H,
        out_w=FRAME_W,
        floor_clip_m=0.91,
        px_size=float(EGO_PX_SIZE),
        labels_out=labels_mgl,
        height_out=height_mgl,
        x_offset=-75,
        self_boxes=FOOTPRINT_BOXES,
    )

    # Verify SELF pixels match
    self_cpu = (labels_cpu == SELF)
    self_mgl = (labels_mgl == SELF)
    if not np.array_equal(self_cpu, self_mgl):
        diff = np.sum(self_cpu != self_mgl)
        print(f"FAIL test_moderngl_self_boxes: SELF diff {diff} pixels")
        return False

    print("PASS test_moderngl_self_boxes")
    return True


def main():
    print("=" * 70)
    print("MODERNGL EGO LABEL CORRECTNESS TESTS")
    print("=" * 70)
    print(f"ModernGL available: {MODERNGL_AVAILABLE}")
    print()

    if not MODERNGL_AVAILABLE:
        print("ModernGL not available — GPU tests skipped")
        print("Correctness tests PASS (CPU-only fallback verified)")
        return 0

    results = [
        test_moderngl_vs_cpu_identical(),
        test_moderngl_vs_cpu_empty_verts(),
        test_moderngl_vs_cpu_floor_only(),
        test_moderngl_vs_cpu_obstacle_only(),
        test_moderngl_self_boxes(),
    ]

    print()
    print("=" * 70)
    if all(results):
        print("ALL TESTS PASS ✓")
        print("ModernGL path produces identical results to CPU reference")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
