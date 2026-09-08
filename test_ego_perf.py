#!/usr/bin/env python3
"""Performance benchmarks for ego label + fuse pipeline (CONTRACT step 4 Hz reclaim).

Measures label_rs1_ego scatter + fuse_rs2_into_ego to guide optimization toward
≤20 ms Orin perception budget. Kevin hardware unreachable — synthetic tests only.
Do NOT invent Orin timings; report measured CPU times and algorithmic wins.

Usage:
    python test_ego_perf.py
    python test_ego_perf.py --trials 100 --verbose
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, EGO_PX_SIZE
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.ego_rs1 import label_rs1_ego
from perception.fuse import fuse_rs2_into_ego


def _synth_verts(n=45000, z_range=(0.3, 1.2), extent_m=1.6):
    """Synthetic RS1 verts ~mag=3 on JP6 (45k verts typical)."""
    xs = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    ys = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    zs = np.random.uniform(z_range[0], z_range[1], n)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def _synth_rs2_obs_known(h=FRAME_H, w=FRAME_W, obs_frac=0.15):
    """Synthetic RS2 obs/known maps (rotated forward depth)."""
    known = np.zeros((h, w), dtype=np.uint8)
    obs = np.zeros((h, w), dtype=np.uint8)
    n_obs = int(h * w * obs_frac)
    obs_idx = np.random.choice(h * w, n_obs, replace=False)
    obs.flat[obs_idx] = np.random.randint(10, 80, n_obs, dtype=np.uint8)
    known.flat[obs_idx] = 255
    n_clear = int(h * w * 0.25)
    clear_idx = np.random.choice(h * w, n_clear, replace=False)
    known.flat[clear_idx] = 255
    return obs, known


def bench_label_rs1_ego(verts, trials=50, warmup=5):
    """Benchmark label_rs1_ego with pre-allocated buffers."""
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


def bench_fuse_rs2_into_ego(labels_rs1, height_rs1, obs2, known2, trials=50, warmup=5):
    """Benchmark fuse_rs2_into_ego with pre-allocated buffers."""
    labels = np.empty_like(labels_rs1)
    height = np.empty_like(height_rs1)
    work_obs = np.zeros_like(obs2)
    work_known = np.zeros_like(known2)
    fw_cone = np.ones((FRAME_H, FRAME_W), dtype=np.uint8) * 255
    fw_cone[120:, :] = 0
    free_range = np.ones((FRAME_H, FRAME_W), dtype=np.uint8) * 255
    free_range[:80, :] = 0

    for _ in range(warmup):
        fuse_rs2_into_ego(
            labels_rs1,
            height_rs1,
            obs2,
            known2,
            fw_dx=57,
            fw_dy=-1,
            fw_cone=fw_cone,
            free_range=free_range,
            labels_out=labels,
            height_out=height,
            work_obs=work_obs,
            work_known=work_known,
        )

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fuse_rs2_into_ego(
            labels_rs1,
            height_rs1,
            obs2,
            known2,
            fw_dx=57,
            fw_dy=-1,
            fw_cone=fw_cone,
            free_range=free_range,
            labels_out=labels,
            height_out=height,
            work_obs=work_obs,
            work_known=work_known,
        )
        times.append((time.perf_counter() - t0) * 1000.0)

    return np.array(times), labels, height


def bench_pipeline(verts, obs2, known2, trials=50, warmup=5):
    """Benchmark full label_rs1 + fuse_rs2 pipeline."""
    cam_l = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cam_h = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels_rs1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_rs1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels_out = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_out = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    work_obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    work_known = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    fw_cone = np.ones((FRAME_H, FRAME_W), dtype=np.uint8) * 255
    fw_cone[120:, :] = 0
    free_range = np.ones((FRAME_H, FRAME_W), dtype=np.uint8) * 255
    free_range[:80, :] = 0

    for _ in range(warmup):
        label_rs1_ego(
            verts,
            labels_out=labels_rs1,
            height_out=height_rs1,
            work_labels=cam_l,
            work_height=cam_h,
            x_offset=-75,
        )
        fuse_rs2_into_ego(
            labels_rs1,
            height_rs1,
            obs2,
            known2,
            fw_dx=57,
            fw_dy=-1,
            fw_cone=fw_cone,
            free_range=free_range,
            labels_out=labels_out,
            height_out=height_out,
            work_obs=work_obs,
            work_known=work_known,
        )

    times_label = []
    times_fuse = []
    times_total = []
    for _ in range(trials):
        t0 = time.perf_counter()
        label_rs1_ego(
            verts,
            labels_out=labels_rs1,
            height_out=height_rs1,
            work_labels=cam_l,
            work_height=cam_h,
            x_offset=-75,
        )
        t1 = time.perf_counter()
        fuse_rs2_into_ego(
            labels_rs1,
            height_rs1,
            obs2,
            known2,
            fw_dx=57,
            fw_dy=-1,
            fw_cone=fw_cone,
            free_range=free_range,
            labels_out=labels_out,
            height_out=height_out,
            work_obs=work_obs,
            work_known=work_known,
        )
        t2 = time.perf_counter()
        times_label.append((t1 - t0) * 1000.0)
        times_fuse.append((t2 - t1) * 1000.0)
        times_total.append((t2 - t0) * 1000.0)

    return (
        np.array(times_label),
        np.array(times_fuse),
        np.array(times_total),
        labels_out,
        height_out,
    )


def print_stats(name, times_ms):
    """Print timing statistics."""
    print(f"  {name:20s}: "
          f"mean={np.mean(times_ms):5.2f}ms "
          f"p50={np.percentile(times_ms, 50):5.2f}ms "
          f"p95={np.percentile(times_ms, 95):5.2f}ms "
          f"max={np.max(times_ms):5.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ego label + fuse pipeline")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--verts", type=int, default=45000, help="Number of RS1 verts")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 70)
    print("EGO LABEL + FUSE PERFORMANCE BENCHMARK")
    print("  (CONTRACT step 4 CAPTURE Hz reclaim — CPU baseline)")
    print("=" * 70)
    print(f"Config: trials={args.trials} warmup={args.warmup} verts={args.verts}")
    print(f"Frame: {FRAME_H}x{FRAME_W} px_size={EGO_PX_SIZE}m")
    print()

    verts = _synth_verts(n=args.verts)
    obs2, known2 = _synth_rs2_obs_known()

    print("Benchmarking label_rs1_ego...")
    t_label, labels_rs1, height_rs1 = bench_label_rs1_ego(
        verts, trials=args.trials, warmup=args.warmup
    )
    print_stats("label_rs1_ego", t_label)
    if args.verbose:
        n_clear = np.count_nonzero(labels_rs1 == CLEAR)
        n_obs = np.count_nonzero(labels_rs1 == OBSTACLE)
        n_self = np.count_nonzero(labels_rs1 == SELF)
        print(f"    → CLEAR={n_clear} OBSTACLE={n_obs} SELF={n_self}")

    print()
    print("Benchmarking fuse_rs2_into_ego...")
    t_fuse, labels_out, height_out = bench_fuse_rs2_into_ego(
        labels_rs1, height_rs1, obs2, known2, trials=args.trials, warmup=args.warmup
    )
    print_stats("fuse_rs2_into_ego", t_fuse)
    if args.verbose:
        n_clear = np.count_nonzero(labels_out == CLEAR)
        n_obs = np.count_nonzero(labels_out == OBSTACLE)
        n_self = np.count_nonzero(labels_out == SELF)
        print(f"    → CLEAR={n_clear} OBSTACLE={n_obs} SELF={n_self}")

    print()
    print("Benchmarking full pipeline (label + fuse)...")
    t_label_p, t_fuse_p, t_total_p, labels_final, height_final = bench_pipeline(
        verts, obs2, known2, trials=args.trials, warmup=args.warmup
    )
    print_stats("pipeline:label", t_label_p)
    print_stats("pipeline:fuse", t_fuse_p)
    print_stats("pipeline:TOTAL", t_total_p)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    mean_total = np.mean(t_total_p)
    p95_total = np.percentile(t_total_p, 95)
    budget_ms = 20.0
    print(f"  Pipeline mean: {mean_total:.2f}ms  p95: {p95_total:.2f}ms")
    print(f"  Budget target: {budget_ms:.2f}ms (≤20ms for 30 Hz with headroom)")
    if p95_total <= budget_ms:
        status = "PASS ✓"
    elif mean_total <= budget_ms:
        status = "MARGINAL (p95 over budget)"
    else:
        status = "OVER BUDGET"
    print(f"  Status: {status}")
    print()
    print("NOTE: These are CPU baseline timings on host. Orin timings differ.")
    print("      Do NOT invent Orin numbers. Algorithmic wins transfer.")
    print("=" * 70)


if __name__ == "__main__":
    main()
