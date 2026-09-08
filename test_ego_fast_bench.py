#!/usr/bin/env python3
"""Benchmark optimized ego_rs1_fast vs original ego_rs1.label_rs1_ego."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, EGO_PX_SIZE, FOOTPRINT_BOXES
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.ego_rs1 import label_rs1_ego
from perception.ego_rs1_fast import label_rs1_ego_fast, KEVIN_GPU_SCATTER, _CUPY_AVAILABLE


def _synth_verts(n=45000, z_range=(0.3, 1.2), extent_m=1.6):
    """Synthetic RS1 verts ~mag=3 on JP6 (45k verts typical)."""
    xs = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    ys = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    zs = np.random.uniform(z_range[0], z_range[1], n)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def bench_original(verts, trials=50, warmup=5):
    """Benchmark original label_rs1_ego."""
    cam_l = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cam_h = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    for _ in range(warmup):
        label_rs1_ego(
            verts,
            labels_out=labels,
            height_out=height,
            work_labels=cam_l,
            work_height=cam_h,
            x_offset=-75,
        )

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        label_rs1_ego(
            verts,
            labels_out=labels,
            height_out=height,
            work_labels=cam_l,
            work_height=cam_h,
            x_offset=-75,
        )
        times.append((time.perf_counter() - t0) * 1000.0)

    return np.array(times), labels, height


def bench_fast_cpu(verts, trials=50, warmup=5):
    """Benchmark optimized CPU scatter."""
    cam_l = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cam_h = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    for _ in range(warmup):
        label_rs1_ego_fast(
            verts,
            out_h=FRAME_H,
            out_w=FRAME_W,
            floor_clip_m=0.91,
            px_size=float(EGO_PX_SIZE),
            labels_out=labels,
            height_out=height,
            work_labels=cam_l,
            work_height=cam_h,
            x_offset=-75,
            self_boxes=FOOTPRINT_BOXES,
            use_gpu=False,
        )

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        label_rs1_ego_fast(
            verts,
            out_h=FRAME_H,
            out_w=FRAME_W,
            floor_clip_m=0.91,
            px_size=float(EGO_PX_SIZE),
            labels_out=labels,
            height_out=height,
            work_labels=cam_l,
            work_height=cam_h,
            x_offset=-75,
            self_boxes=FOOTPRINT_BOXES,
            use_gpu=False,
        )
        times.append((time.perf_counter() - t0) * 1000.0)

    return np.array(times), labels, height


def bench_fast_gpu(verts, trials=50, warmup=5):
    """Benchmark GPU scatter (if CuPy available)."""
    if not _CUPY_AVAILABLE:
        return None, None, None

    cam_l = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cam_h = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    for _ in range(warmup):
        label_rs1_ego_fast(
            verts,
            out_h=FRAME_H,
            out_w=FRAME_W,
            floor_clip_m=0.91,
            px_size=float(EGO_PX_SIZE),
            labels_out=labels,
            height_out=height,
            work_labels=cam_l,
            work_height=cam_h,
            x_offset=-75,
            self_boxes=FOOTPRINT_BOXES,
            use_gpu=True,
        )

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        label_rs1_ego_fast(
            verts,
            out_h=FRAME_H,
            out_w=FRAME_W,
            floor_clip_m=0.91,
            px_size=float(EGO_PX_SIZE),
            labels_out=labels,
            height_out=height,
            work_labels=cam_l,
            work_height=cam_h,
            x_offset=-75,
            self_boxes=FOOTPRINT_BOXES,
            use_gpu=True,
        )
        times.append((time.perf_counter() - t0) * 1000.0)

    return np.array(times), labels, height


def print_stats(name, times_ms):
    """Print timing statistics."""
    if times_ms is None:
        print(f"  {name:24s}: NOT AVAILABLE")
        return
    print(f"  {name:24s}: "
          f"mean={np.mean(times_ms):5.2f}ms "
          f"p50={np.percentile(times_ms, 50):5.2f}ms "
          f"p95={np.percentile(times_ms, 95):5.2f}ms "
          f"max={np.max(times_ms):5.2f}ms")


def compare_results(labels_orig, labels_fast, height_orig, height_fast, name="CPU"):
    """Compare original vs optimized results for correctness."""
    labels_match = np.allclose(labels_orig, labels_fast)
    height_match = np.allclose(height_orig, height_fast)
    print(f"  {name} correctness: labels={'PASS' if labels_match else 'FAIL'} "
          f"height={'PASS' if height_match else 'FAIL'}")
    if not labels_match or not height_match:
        diff_l = np.sum(labels_orig != labels_fast)
        diff_h = np.sum(height_orig != height_fast)
        print(f"    → label diff: {diff_l}/{labels_orig.size} pixels")
        print(f"    → height diff: {diff_h}/{height_orig.size} pixels")


def main():
    print("=" * 70)
    print("EGO RS1 FAST BENCHMARK (original vs optimized)")
    print("=" * 70)
    print(f"Frame: {FRAME_H}x{FRAME_W} px_size={EGO_PX_SIZE}m")
    print(f"CuPy available: {_CUPY_AVAILABLE}")
    print(f"KEVIN_GPU_SCATTER: {KEVIN_GPU_SCATTER}")
    print()

    verts = _synth_verts(n=45000)
    print(f"Synthetic verts: {verts.shape} (typical mag=3 on JP6)")
    print()

    print("Benchmarking original label_rs1_ego (50 trials)...")
    t_orig, labels_orig, height_orig = bench_original(verts, trials=50)
    print_stats("ORIGINAL", t_orig)
    print()

    print("Benchmarking optimized CPU scatter (50 trials)...")
    t_fast_cpu, labels_fast_cpu, height_fast_cpu = bench_fast_cpu(verts, trials=50)
    print_stats("FAST (CPU)", t_fast_cpu)
    compare_results(labels_orig, labels_fast_cpu, height_orig, height_fast_cpu, "CPU")
    print()

    if _CUPY_AVAILABLE:
        print("Benchmarking optimized GPU scatter (50 trials)...")
        t_fast_gpu, labels_fast_gpu, height_fast_gpu = bench_fast_gpu(verts, trials=50)
        print_stats("FAST (GPU)", t_fast_gpu)
        compare_results(labels_orig, labels_fast_gpu, height_orig, height_fast_gpu, "GPU")
        print()
    else:
        print("GPU scatter: CuPy not available (install CuPy for GPU path)")
        print()

    print("=" * 70)
    print("SPEEDUP ANALYSIS")
    print("=" * 70)
    orig_mean = np.mean(t_orig)
    cpu_mean = np.mean(t_fast_cpu)
    speedup_cpu = orig_mean / cpu_mean if cpu_mean > 0 else 0
    print(f"  Original:    {orig_mean:.2f}ms")
    print(f"  Fast (CPU):  {cpu_mean:.2f}ms → {speedup_cpu:.2f}x speedup")
    if _CUPY_AVAILABLE and t_fast_gpu is not None:
        gpu_mean = np.mean(t_fast_gpu)
        speedup_gpu = orig_mean / gpu_mean if gpu_mean > 0 else 0
        print(f"  Fast (GPU):  {gpu_mean:.2f}ms → {speedup_gpu:.2f}x speedup")
    print()
    print("NOTE: These are host CPU/GPU timings. Orin speedups will differ.")
    print("      Algorithmic wins (sorted scatter) transfer to Orin CPU.")
    print("      GPU path requires CuPy + KEVIN_GPU_SCATTER=1 flag.")
    print("=" * 70)


if __name__ == "__main__":
    main()
