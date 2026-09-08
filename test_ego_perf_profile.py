#!/usr/bin/env python3
"""Detailed profiling of label_rs1_ego stages to identify bottlenecks.

Instruments each major operation in label_rs1_ego to find optimization targets.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, EGO_PX_SIZE
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE


def _synth_verts(n=45000, z_range=(0.3, 1.2), extent_m=1.6):
    """Synthetic RS1 verts ~mag=3 on JP6 (45k verts typical)."""
    xs = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    ys = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    zs = np.random.uniform(z_range[0], z_range[1], n)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def profile_label_rs1_stages(verts, trials=100):
    """Profile each stage of label_rs1_ego separately."""
    from robot_config import FOOTPRINT_BOXES

    out_h, out_w = FRAME_H, FRAME_W
    floor_clip_m = 0.91
    px_size = float(EGO_PX_SIZE)
    x_offset = -75

    times = {
        "setup": [],
        "project": [],
        "scatter_clear": [],
        "scatter_obs": [],
        "rotate": [],
        "blit": [],
        "paint_self": [],
    }

    cam_l = np.zeros((out_h, out_w), dtype=np.uint8)
    cam_h = np.zeros((out_h, out_w), dtype=np.uint8)
    labels = np.zeros((out_h, out_w), dtype=np.uint8)
    height = np.zeros((out_h, out_w), dtype=np.uint8)

    for _ in range(trials):
        t0 = time.perf_counter()
        cam_l.fill(0)
        cam_h.fill(0)
        labels.fill(0)
        height.fill(0)

        v = np.asarray(verts, dtype=np.float32)
        z = v[:, 2]
        valid = z > 0.01
        t1 = time.perf_counter()
        times["setup"].append((t1 - t0) * 1000.0)

        if np.any(valid):
            scale = np.float32(1.0 / px_size)
            center = np.float32([out_w * 0.5, out_h * 0.5])
            vv = v[valid]
            p = vv[:, :2] * scale + center
            with np.errstate(invalid="ignore"):
                ja, ia = p.astype(np.uint32).T
                ma = (ia < np.uint32(out_h)) & (ja < np.uint32(out_w))
            ia_m, ja_m = ia[ma], ja[ma]
            zv = vv[ma, 2]
            t2 = time.perf_counter()
            times["project"].append((t2 - t1) * 1000.0)

            floor = zv >= floor_clip_m
            cam_l[ia_m[floor], ja_m[floor]] = CLEAR
            t3 = time.perf_counter()
            times["scatter_clear"].append((t3 - t2) * 1000.0)

            obs = ~floor
            if np.any(obs):
                ia_o, ja_o = ia_m[obs], ja_m[obs]
                h = np.clip(
                    ((floor_clip_m - zv[obs]) * 100.0).astype(np.int32), 1, 100
                ).astype(np.uint8)
                cam_l[ia_o, ja_o] = OBSTACLE
                np.maximum.at(cam_h, (ia_o, ja_o), h)
            t4 = time.perf_counter()
            times["scatter_obs"].append((t4 - t3) * 1000.0)

        cam_l_f = cam_l[::-1, ::-1]
        cam_h_f = cam_h[::-1, ::-1]
        t5 = time.perf_counter()
        times["rotate"].append((t5 - t4) * 1000.0)

        if x_offset == 0:
            np.copyto(labels, cam_l_f)
            np.copyto(height, cam_h_f)
        else:
            h, w = cam_l_f.shape
            dx = int(x_offset)
            if dx > 0:
                if dx < w:
                    labels[:, dx:w] = cam_l_f[:, : w - dx]
                    height[:, dx:w] = cam_h_f[:, : w - dx]
            else:
                if -dx < w:
                    labels[:, : w + dx] = cam_l_f[:, -dx:]
                    height[:, : w + dx] = cam_h_f[:, -dx:]
        t6 = time.perf_counter()
        times["blit"].append((t6 - t5) * 1000.0)

        for x0, y0, x1, y1 in FOOTPRINT_BOXES:
            labels[y0:y1, x0:x1] = SELF
        t7 = time.perf_counter()
        times["paint_self"].append((t7 - t6) * 1000.0)

    return times


def print_profile(times):
    """Print profiling results."""
    print("=" * 70)
    print("DETAILED STAGE PROFILING (label_rs1_ego)")
    print("=" * 70)
    total_mean = 0.0
    for stage, t in times.items():
        arr = np.array(t)
        mean = np.mean(arr)
        p95 = np.percentile(arr, 95)
        total_mean += mean
        pct = (mean / 3.14) * 100.0  # Baseline from previous benchmark
        print(f"  {stage:16s}: mean={mean:5.3f}ms p95={p95:5.3f}ms  ({pct:4.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL':16s}: {total_mean:5.3f}ms")
    print("=" * 70)


def main():
    print("Generating synthetic verts (45k, typical mag=3)...")
    verts = _synth_verts(n=45000)
    print(f"  verts shape: {verts.shape}")
    print()
    print("Profiling label_rs1_ego stages (100 trials)...")
    times = profile_label_rs1_stages(verts, trials=100)
    print()
    print_profile(times)
    print()
    print("KEY FINDINGS:")
    arr_project = np.array(times["project"])
    arr_scatter_obs = np.array(times["scatter_obs"])
    arr_scatter_clear = np.array(times["scatter_clear"])
    print(f"  - Project (coords): {np.mean(arr_project):.3f}ms")
    print(f"  - Scatter CLEAR:    {np.mean(arr_scatter_clear):.3f}ms")
    print(f"  - Scatter OBSTACLE: {np.mean(arr_scatter_obs):.3f}ms")
    print(f"  - Total scatter:    {np.mean(arr_scatter_clear) + np.mean(arr_scatter_obs):.3f}ms")
    print()
    print("  Optimization targets:")
    print("    1. Scatter operations (fancy indexing) — largest share")
    print("    2. Projection (coordinate transform) — second")
    print("    3. GPU path (CuPy) for scatter — optional behind flag")


if __name__ == "__main__":
    main()
