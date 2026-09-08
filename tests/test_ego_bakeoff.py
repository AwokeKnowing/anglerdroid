#!/usr/bin/env python3
"""Offline Orin/host bake-off harness for ego label paths.

Times and correctness-checks three ego label paths (CPU, GPU/CuPy, ModernGL)
on identical synthetic RS1 (+optional RS2) verts:
- CPU: label_rs1_ego + fuse_rs2_into_ego
- GPU (CuPy): label_rs1_ego_gpu + fuse_rs2_into_ego_gpu (KEVIN_GPU_SCATTER / KEVIN_GPU_FUSE)
- ModernGL: label_rs1_ego_moderngl (KEVIN_MODERNGL_SCATTER)

Reports wall-clock ms per path (median over N warm runs) and asserts honesty:
- SELF core leak_px==0
- No CLEAR under chassis/self boxes
- Labels only in {UNKNOWN, SELF, CLEAR, OBSTACLE}
- GPU/ModernGL labels match CPU on SELF mask
- GPU/ModernGL do not invent CLEAR where CPU has UNKNOWN/SELF

Skips (not fails) CuPy/ModernGL sections when deps missing — host CI stays green.

Usage:
  python tests/test_ego_bakeoff.py              # Host CPU + optional GPU/ModernGL
  KEVIN_GPU_SCATTER=1 python tests/...          # On Orin: enable CuPy scatter
  KEVIN_GPU_FUSE=1 python tests/...             # On Orin: enable CuPy fuse
  KEVIN_MODERNGL_SCATTER=1 python tests/...     # On Orin: enable ModernGL scatter

See docs/perception/CONTRACT.md CAPTURE/bake-off section.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from robot_config import FRAME_H, FRAME_W, EGO_PX_SIZE, FOOTPRINT_BOXES, UNDER_ROBOT_BOXES
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.ego_rs1 import label_rs1_ego
from perception.fuse import fuse_rs2_into_ego
from perception.ego_rs1_fast import (
    label_rs1_ego_gpu,
    fuse_rs2_into_ego_gpu,
    _CUPY_AVAILABLE,
)

# Check ModernGL availability
try:
    from gpu_render import GPURenderer, _HAS_MGL
    MODERNGL_AVAILABLE = _HAS_MGL
except ImportError:
    MODERNGL_AVAILABLE = False


# ============================================================================
# Synthetic data generators
# ============================================================================

def _synth_rs1_verts(n=45000, z_range=(0.3, 1.2), extent_m=1.6, seed=42):
    """Synthetic RS1 verts (top-down, ~mag=3 on JP6)."""
    np.random.seed(seed)
    xs = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    ys = np.random.uniform(-extent_m / 2, extent_m / 2, n)
    zs = np.random.uniform(z_range[0], z_range[1], n)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def _synth_rs2_obs_known(seed=123):
    """Synthetic RS2 obs/known (forward depth)."""
    np.random.seed(seed)
    h, w = FRAME_H, FRAME_W
    obs = np.zeros((h, w), dtype=np.uint8)
    known = np.zeros((h, w), dtype=np.uint8)
    
    # Add some forward obstacles
    obs[50:80, 100:140] = 35  # 35 cm obstacle
    known[50:80, 100:140] = 255
    
    # Add some forward CLEAR
    known[120:160, 100:140] = 255
    
    return obs, known


# ============================================================================
# Honesty checks
# ============================================================================

def check_honesty(labels, height, name="Path"):
    """Assert honesty invariants on ego labels.
    
    Returns:
        dict: Metrics with leak_px, under_clear_px, invalid_labels, self_px
    """
    errors = []
    
    # 1. Labels only in {UNKNOWN, SELF, CLEAR, OBSTACLE}
    valid_labels = {UNKNOWN, SELF, CLEAR, OBSTACLE}
    unique_labels = set(np.unique(labels))
    invalid = unique_labels - valid_labels
    if invalid:
        errors.append(f"Invalid labels found: {invalid}")
    
    # 2. SELF core leak: check if any SELF leaked into obstacle/clear outside boxes
    self_mask = (labels == SELF)
    expected_self_mask = np.zeros_like(labels, dtype=bool)
    for x0, y0, x1, y1 in FOOTPRINT_BOXES:
        expected_self_mask[y0:y1, x0:x1] = True
    
    leak_mask = self_mask & ~expected_self_mask
    leak_px = np.count_nonzero(leak_mask)
    if leak_px > 0:
        errors.append(f"SELF leaked outside boxes: {leak_px} pixels")
    
    # 3. No CLEAR under chassis/self boxes
    under_clear_px = 0
    for x0, y0, x1, y1 in UNDER_ROBOT_BOXES:
        under_clear_px += np.count_nonzero(labels[y0:y1, x0:x1] == CLEAR)
    if under_clear_px > 0:
        errors.append(f"CLEAR invented under chassis: {under_clear_px} pixels")
    
    # 4. Height should be zero for SELF, CLEAR, UNKNOWN
    non_obs_mask = (labels != OBSTACLE)
    bad_height_px = np.count_nonzero(height[non_obs_mask] != 0)
    if bad_height_px > 0:
        errors.append(f"Non-zero height in non-OBSTACLE cells: {bad_height_px} pixels")
    
    metrics = {
        "leak_px": leak_px,
        "under_clear_px": under_clear_px,
        "invalid_labels": len(invalid),
        "self_px": np.count_nonzero(self_mask),
    }
    
    if errors:
        raise AssertionError(f"{name} HONESTY VIOLATION:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return metrics


def check_gpu_vs_cpu_honesty(labels_cpu, height_cpu, labels_gpu, height_gpu, name="GPU"):
    """Assert GPU/ModernGL honors CPU honesty (SELF mask match, no invented CLEAR).
    
    Returns:
        dict: Metrics with self_match, clear_invented, label_diff, height_diff
    """
    errors = []
    
    # 1. SELF mask must match CPU
    self_cpu = (labels_cpu == SELF)
    self_gpu = (labels_gpu == SELF)
    self_diff = np.count_nonzero(self_cpu != self_gpu)
    if self_diff > 0:
        errors.append(f"SELF mask mismatch vs CPU: {self_diff} pixels")
    
    # 2. GPU/ModernGL must not invent CLEAR where CPU has UNKNOWN/SELF
    cpu_not_clear = (labels_cpu == UNKNOWN) | (labels_cpu == SELF)
    gpu_clear = (labels_gpu == CLEAR)
    invented_clear = cpu_not_clear & gpu_clear
    invented_clear_px = np.count_nonzero(invented_clear)
    if invented_clear_px > 0:
        errors.append(f"Invented CLEAR where CPU has UNKNOWN/SELF: {invented_clear_px} pixels")
    
    metrics = {
        "self_match": self_diff == 0,
        "clear_invented": invented_clear_px,
        "label_diff": np.count_nonzero(labels_cpu != labels_gpu),
        "height_diff": np.count_nonzero(height_cpu != height_gpu),
    }
    
    if errors:
        raise AssertionError(f"{name} vs CPU HONESTY VIOLATION:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return metrics


# ============================================================================
# Timing helpers
# ============================================================================

def time_trials(func, trials=50, warmup=5):
    """Run function trials times after warmup, return median ms."""
    for _ in range(warmup):
        func()
    
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    
    return np.array(times)


# ============================================================================
# CPU path: label_rs1_ego + fuse_rs2_into_ego
# ============================================================================

def bench_cpu_path(verts_rs1, obs2, known2, trials=50):
    """Benchmark CPU path: label_rs1_ego + fuse_rs2_into_ego."""
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels_fused = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_fused = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    def run():
        label_rs1_ego(verts_rs1, labels_out=labels, height_out=height, x_offset=-75)
        fuse_rs2_into_ego(
            labels, height, obs2, known2, fw_dx=10, fw_dy=5,
            labels_out=labels_fused, height_out=height_fused
        )
    
    times = time_trials(run, trials=trials)
    
    # Final run for correctness check
    run()
    
    return times, labels_fused, height_fused


# ============================================================================
# GPU (CuPy) path: label_rs1_ego_gpu + fuse_rs2_into_ego_gpu
# ============================================================================

def bench_gpu_path(verts_rs1, obs2, known2, trials=50):
    """Benchmark GPU (CuPy) path: label_rs1_ego_gpu + fuse_rs2_into_ego_gpu."""
    if not _CUPY_AVAILABLE:
        return None, None, None
    
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels_fused = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_fused = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    def run():
        label_rs1_ego_gpu(
            verts_rs1,
            out_h=FRAME_H, out_w=FRAME_W,
            floor_clip_m=0.91, px_size=float(EGO_PX_SIZE),
            labels_out=labels, height_out=height,
            x_offset=-75, self_boxes=FOOTPRINT_BOXES
        )
        fuse_rs2_into_ego_gpu(
            labels, height, obs2, known2, fw_dx=10, fw_dy=5,
            labels_out=labels_fused, height_out=height_fused
        )
    
    times = time_trials(run, trials=trials)
    
    # Final run for correctness check
    run()
    
    return times, labels_fused, height_fused


# ============================================================================
# ModernGL path: label_rs1_ego_moderngl (RS1 only, no RS2 fuse yet)
# ============================================================================

def bench_moderngl_path(verts_rs1, obs2, known2, trials=50):
    """Benchmark ModernGL path: label_rs1_ego_moderngl + CPU fuse_rs2_into_ego."""
    if not MODERNGL_AVAILABLE:
        return None, None, None
    
    gpu = GPURenderer(800, 800, 960, 960)
    gpu.configure_ego_labels(
        out_w=FRAME_W, out_h=FRAME_H,
        px_size=float(EGO_PX_SIZE),
        floor_clip_m=0.91
    )
    
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels_fused = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height_fused = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    def run():
        gpu.label_rs1_ego_moderngl(
            verts_rs1,
            out_h=FRAME_H, out_w=FRAME_W,
            floor_clip_m=0.91, px_size=float(EGO_PX_SIZE),
            labels_out=labels, height_out=height,
            x_offset=-75, self_boxes=FOOTPRINT_BOXES
        )
        # Use CPU fuse (ModernGL fuse not implemented yet)
        fuse_rs2_into_ego(
            labels, height, obs2, known2, fw_dx=10, fw_dy=5,
            labels_out=labels_fused, height_out=height_fused
        )
    
    times = time_trials(run, trials=trials)
    
    # Final run for correctness check
    run()
    
    return times, labels_fused, height_fused


# ============================================================================
# Main bake-off harness
# ============================================================================

def print_table_header():
    """Print timing table header."""
    print("=" * 90)
    print(f"{'Path':<20} {'Available':<12} {'Median ms':<12} {'p95 ms':<12} {'Max ms':<12} {'Status':<10}")
    print("=" * 90)


def print_table_row(name, available, times, status="PASS"):
    """Print timing table row."""
    if not available or times is None:
        print(f"{name:<20} {'NO':<12} {'-':<12} {'-':<12} {'-':<12} {'SKIP':<10}")
    else:
        median = np.median(times)
        p95 = np.percentile(times, 95)
        max_time = np.max(times)
        print(f"{name:<20} {'YES':<12} {median:<12.2f} {p95:<12.2f} {max_time:<12.2f} {status:<10}")


def main():
    print("=" * 90)
    print("EGO LABEL PATH BAKE-OFF HARNESS")
    print("=" * 90)
    print(f"Frame: {FRAME_H}×{FRAME_W} px, EGO_PX_SIZE={EGO_PX_SIZE}m")
    print(f"CuPy available: {_CUPY_AVAILABLE}")
    print(f"ModernGL available: {MODERNGL_AVAILABLE}")
    print(f"KEVIN_GPU_SCATTER: {os.environ.get('KEVIN_GPU_SCATTER', '0')}")
    print(f"KEVIN_GPU_FUSE: {os.environ.get('KEVIN_GPU_FUSE', '0')}")
    print(f"KEVIN_MODERNGL_SCATTER: {os.environ.get('KEVIN_MODERNGL_SCATTER', '0')}")
    print()
    
    # Generate synthetic data
    print("Generating synthetic RS1 + RS2 data...")
    verts_rs1 = _synth_rs1_verts(n=45000, seed=42)
    obs2, known2 = _synth_rs2_obs_known(seed=123)
    print(f"  RS1 verts: {verts_rs1.shape} (typical mag=3 on JP6)")
    print(f"  RS2 obs/known: {obs2.shape}")
    print()
    
    # ========================================================================
    # 1. CPU path (baseline)
    # ========================================================================
    print("Benchmarking CPU path (label_rs1_ego + fuse_rs2_into_ego)...")
    times_cpu, labels_cpu, height_cpu = bench_cpu_path(verts_rs1, obs2, known2, trials=50)
    print(f"  CPU timing: median={np.median(times_cpu):.2f}ms p95={np.percentile(times_cpu, 95):.2f}ms")
    
    print("  Checking CPU honesty...")
    cpu_metrics = check_honesty(labels_cpu, height_cpu, name="CPU")
    print(f"    SELF pixels: {cpu_metrics['self_px']}")
    print(f"    Leak pixels: {cpu_metrics['leak_px']}")
    print(f"    Under-chassis CLEAR: {cpu_metrics['under_clear_px']}")
    print("    ✓ CPU honesty PASS")
    print()
    
    # ========================================================================
    # 2. GPU (CuPy) path
    # ========================================================================
    times_gpu, labels_gpu, height_gpu = None, None, None
    if _CUPY_AVAILABLE:
        print("Benchmarking GPU (CuPy) path (label_rs1_ego_gpu + fuse_rs2_into_ego_gpu)...")
        times_gpu, labels_gpu, height_gpu = bench_gpu_path(verts_rs1, obs2, known2, trials=50)
        print(f"  GPU timing: median={np.median(times_gpu):.2f}ms p95={np.percentile(times_gpu, 95):.2f}ms")
        
        print("  Checking GPU honesty...")
        gpu_metrics = check_honesty(labels_gpu, height_gpu, name="GPU")
        print(f"    SELF pixels: {gpu_metrics['self_px']}")
        
        print("  Checking GPU vs CPU consistency...")
        gpu_vs_cpu = check_gpu_vs_cpu_honesty(labels_cpu, height_cpu, labels_gpu, height_gpu, name="GPU")
        print(f"    SELF match: {gpu_vs_cpu['self_match']}")
        print(f"    Label diff: {gpu_vs_cpu['label_diff']} pixels")
        print(f"    Invented CLEAR: {gpu_vs_cpu['clear_invented']}")
        print("    ✓ GPU honesty PASS")
        print()
    else:
        print("GPU (CuPy) path: SKIP (CuPy not available)")
        print()
    
    # ========================================================================
    # 3. ModernGL path
    # ========================================================================
    times_mgl, labels_mgl, height_mgl = None, None, None
    if MODERNGL_AVAILABLE:
        print("Benchmarking ModernGL path (label_rs1_ego_moderngl + CPU fuse)...")
        times_mgl, labels_mgl, height_mgl = bench_moderngl_path(verts_rs1, obs2, known2, trials=50)
        print(f"  ModernGL timing: median={np.median(times_mgl):.2f}ms p95={np.percentile(times_mgl, 95):.2f}ms")
        
        print("  Checking ModernGL honesty...")
        mgl_metrics = check_honesty(labels_mgl, height_mgl, name="ModernGL")
        print(f"    SELF pixels: {mgl_metrics['self_px']}")
        
        print("  Checking ModernGL vs CPU consistency...")
        mgl_vs_cpu = check_gpu_vs_cpu_honesty(labels_cpu, height_cpu, labels_mgl, height_mgl, name="ModernGL")
        print(f"    SELF match: {mgl_vs_cpu['self_match']}")
        print(f"    Label diff: {mgl_vs_cpu['label_diff']} pixels")
        print(f"    Invented CLEAR: {mgl_vs_cpu['clear_invented']}")
        print("    ✓ ModernGL honesty PASS")
        print()
    else:
        print("ModernGL path: SKIP (ModernGL not available)")
        print()
    
    # ========================================================================
    # Summary table
    # ========================================================================
    print()
    print_table_header()
    print_table_row("CPU (baseline)", True, times_cpu, "PASS")
    print_table_row("GPU (CuPy)", _CUPY_AVAILABLE, times_gpu, "PASS" if times_gpu is not None else "SKIP")
    print_table_row("ModernGL", MODERNGL_AVAILABLE, times_mgl, "PASS" if times_mgl is not None else "SKIP")
    print("=" * 90)
    print()
    
    # ========================================================================
    # Speedup analysis
    # ========================================================================
    if times_cpu is not None:
        cpu_median = np.median(times_cpu)
        print("SPEEDUP vs CPU baseline:")
        if times_gpu is not None:
            gpu_median = np.median(times_gpu)
            speedup = cpu_median / gpu_median if gpu_median > 0 else 0
            print(f"  GPU (CuPy):  {gpu_median:.2f}ms → {speedup:.2f}x")
        if times_mgl is not None:
            mgl_median = np.median(times_mgl)
            speedup = cpu_median / mgl_median if mgl_median > 0 else 0
            print(f"  ModernGL:    {mgl_median:.2f}ms → {speedup:.2f}x")
        print()
    
    print("=" * 90)
    print("BAKE-OFF COMPLETE ✓")
    print("=" * 90)
    print()
    print("Next steps:")
    print("  1. On Orin: KEVIN_GPU_SCATTER=1 KEVIN_GPU_FUSE=1 python tests/test_ego_bakeoff.py")
    print("  2. On Orin: KEVIN_MODERNGL_SCATTER=1 python tests/test_ego_bakeoff.py")
    print("  3. Compare Orin timings vs host (ARM cores + GPU vs x86 CPU)")
    print("  4. See docs/perception/CONTRACT.md CAPTURE/bake-off section")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
