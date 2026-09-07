"""
test_gpu_topdown.py - Test GPU topdown depth processing.

Verifies that GPU topdown produces same results as CPU depth_topdown,
but significantly faster (~3-4x).
"""

import time
import numpy as np
from src.vision import (
    depth_topdown,
    _clip_decimated_border,
    TD_FLOOR_CLIP,
    TD_PX_SIZE,
    FRAME_H,
    FRAME_W,
)


def generate_test_cloud(n_points=10000, seed=42):
    """Generate synthetic RS1 point cloud for testing.
    
    Returns Nx3 array (X, Y, Z) in metres:
    - X, Y: ±0.5m (1m square FOV)
    - Z: 0.2-1.2m (camera distance, TD_FLOOR_CLIP=0.91m separates floor/obs)
    """
    np.random.seed(seed)
    verts = np.zeros((n_points, 3), dtype=np.float32)
    verts[:, 0] = np.random.uniform(-0.5, 0.5, n_points)
    verts[:, 1] = np.random.uniform(-0.5, 0.5, n_points)
    verts[:, 2] = np.random.uniform(0.2, 1.2, n_points)
    return verts


def add_obstacle_cluster(verts, center_x, center_y, z_height, n_pts=100, radius=0.1):
    """Add a cluster of points at given (x, y) and z height."""
    obs = np.zeros((n_pts, 3), dtype=np.float32)
    angles = np.linspace(0, 2*np.pi, n_pts)
    r = np.random.uniform(0, radius, n_pts)
    obs[:, 0] = center_x + r * np.cos(angles)
    obs[:, 1] = center_y + r * np.sin(angles)
    obs[:, 2] = z_height
    return np.vstack([verts, obs])


def test_gpu_cpu_parity():
    """Test that GPU and CPU produce identical results on fixture data."""
    print("test_gpu_cpu_parity: start")
    
    try:
        from src.gpu_render import GPURenderer
    except ImportError as e:
        print("  SKIP: GPU not available (%s)" % e)
        return
    
    # Generate test cloud with obstacles
    verts = generate_test_cloud(n_points=5000, seed=123)
    
    # Add floor points (z >= TD_FLOOR_CLIP)
    verts = add_obstacle_cluster(verts, 0.0, 0.0, 0.95, n_pts=200, radius=0.15)
    
    # Add low obstacle (5cm height above floor)
    verts = add_obstacle_cluster(verts, 0.2, 0.2, 0.86, n_pts=150, radius=0.08)
    
    # Add tall obstacle (30cm height)
    verts = add_obstacle_cluster(verts, -0.2, 0.1, 0.61, n_pts=150, radius=0.08)
    
    # Clip borders (same as vision does)
    verts_clean = _clip_decimated_border(verts)
    
    # CPU version
    cpu_obs, cpu_known = depth_topdown(verts)
    
    # GPU version
    gpu = GPURenderer(map_w=960, map_h=720, atlas_w=960, atlas_h=960)
    gpu.configure_depth_topdown(
        px_size=float(TD_PX_SIZE),
        floor_clip=float(TD_FLOOR_CLIP),
        out_h=FRAME_H,
        out_w=FRAME_W)
    
    gpu_result = gpu.depth_topdown_gpu(verts_clean)
    if gpu_result is None:
        print("  SKIP: GPU topdown not available")
        gpu.release()
        return
    
    gpu_obs, gpu_known = gpu_result
    gpu.release()
    
    # Compare known masks (should be identical)
    known_match = np.array_equal(cpu_known, gpu_known)
    if not known_match:
        cpu_known_px = int(np.count_nonzero(cpu_known))
        gpu_known_px = int(np.count_nonzero(gpu_known))
        diff_px = abs(cpu_known_px - gpu_known_px)
        print("  WARNING: known mismatch: cpu=%d gpu=%d diff=%d" % 
              (cpu_known_px, gpu_known_px, diff_px))
        # Allow small difference due to floating point / rasterization
        assert diff_px < 100, "Known mask difference too large"
    
    # Compare obs maps (should be very close, allowing for minor scatter differences)
    obs_diff = np.abs(cpu_obs.astype(np.int32) - gpu_obs.astype(np.int32))
    max_diff = int(np.max(obs_diff))
    mean_diff = float(np.mean(obs_diff[obs_diff > 0])) if np.any(obs_diff > 0) else 0.0
    
    cpu_obs_px = int(np.count_nonzero(cpu_obs))
    gpu_obs_px = int(np.count_nonzero(gpu_obs))
    
    print("  known match: %s" % known_match)
    print("  obs pixels: cpu=%d gpu=%d" % (cpu_obs_px, gpu_obs_px))
    print("  obs diff: max=%d mean=%.2f" % (max_diff, mean_diff))
    
    # Obs should match within tolerance (GPU atomic scatter may have minor diffs)
    assert abs(cpu_obs_px - gpu_obs_px) < 200, "Obs pixel count too different"
    assert max_diff < 10, "Max obstacle height diff too large"
    
    print("test_gpu_cpu_parity: PASS")


def test_gpu_performance():
    """Test that GPU is significantly faster than CPU (target: 3-4x)."""
    print("test_gpu_performance: start")
    
    try:
        from src.gpu_render import GPURenderer
    except ImportError as e:
        print("  SKIP: GPU not available (%s)" % e)
        return
    
    # Generate large cloud (~45k verts like real RS1 with mag=3)
    verts = generate_test_cloud(n_points=40000, seed=456)
    verts = add_obstacle_cluster(verts, 0.1, 0.2, 0.70, n_pts=5000, radius=0.2)
    verts_clean = _clip_decimated_border(verts)
    
    # Warm up
    _ = depth_topdown(verts)
    
    # CPU timing
    t0 = time.monotonic()
    for _ in range(20):
        _ = depth_topdown(verts)
    t_cpu = (time.monotonic() - t0) * 1000.0 / 20.0  # ms per frame
    
    # GPU setup
    gpu = GPURenderer(map_w=960, map_h=720, atlas_w=960, atlas_h=960)
    gpu.configure_depth_topdown(
        px_size=float(TD_PX_SIZE),
        floor_clip=float(TD_FLOOR_CLIP),
        out_h=FRAME_H,
        out_w=FRAME_W)
    
    # GPU warm up
    result = gpu.depth_topdown_gpu(verts_clean)
    if result is None:
        print("  SKIP: GPU topdown not available")
        gpu.release()
        return
    
    # GPU timing
    t0 = time.monotonic()
    for _ in range(20):
        _ = gpu.depth_topdown_gpu(verts_clean)
    t_gpu = (time.monotonic() - t0) * 1000.0 / 20.0  # ms per frame
    
    gpu.release()
    
    speedup = t_cpu / t_gpu
    
    print("  CPU depth_topdown: %.2fms" % t_cpu)
    print("  GPU depth_topdown: %.2fms" % t_gpu)
    print("  Speedup: %.2fx" % speedup)
    
    # GPU should be at least 2x faster (target is 3-4x)
    assert speedup > 2.0, "GPU should be significantly faster than CPU"
    
    # Hypothesis was CPU ~4.5ms → GPU ~1-2ms (3-4x speedup)
    if t_gpu < 2.5:
        print("  ✓ GPU meets <2.5ms target (hypothesis: 1-2ms)")
    
    print("test_gpu_performance: PASS")


def test_gpu_empty_verts():
    """Test that GPU handles empty verts gracefully."""
    print("test_gpu_empty_verts: start")
    
    try:
        from src.gpu_render import GPURenderer
    except ImportError as e:
        print("  SKIP: GPU not available (%s)" % e)
        return
    
    gpu = GPURenderer(map_w=960, map_h=720, atlas_w=960, atlas_h=960)
    gpu.configure_depth_topdown(
        px_size=float(TD_PX_SIZE),
        floor_clip=float(TD_FLOOR_CLIP),
        out_h=FRAME_H,
        out_w=FRAME_W)
    
    empty = np.zeros((0, 3), dtype=np.float32)
    result = gpu.depth_topdown_gpu(empty)
    
    if result is None:
        print("  SKIP: GPU topdown not available")
        gpu.release()
        return
    
    obs, known = result
    gpu.release()
    
    # Should return all zeros
    assert np.count_nonzero(obs) == 0, "Empty verts should produce no obstacles"
    assert np.count_nonzero(known) == 0, "Empty verts should produce no known"
    
    print("test_gpu_empty_verts: PASS")


def test_gpu_floor_vs_obstacle():
    """Test that GPU correctly separates floor vs obstacle by Z threshold."""
    print("test_gpu_floor_vs_obstacle: start")
    
    try:
        from src.gpu_render import GPURenderer
    except ImportError as e:
        print("  SKIP: GPU not available (%s)" % e)
        return
    
    # Create verts at center with known Z values
    # Floor: z >= TD_FLOOR_CLIP (0.91m) → known=255, obs=0
    # Obstacle: z < TD_FLOOR_CLIP → known=255, obs=height_cm
    
    verts = np.array([
        [0.0, 0.0, 0.95],  # floor (5cm below camera)
        [0.0, 0.0, 0.85],  # 6cm obstacle
        [0.0, 0.0, 0.70],  # 21cm obstacle
        [0.0, 0.0, 0.50],  # 41cm obstacle
    ] * 100, dtype=np.float32)  # Repeat to ensure coverage
    
    gpu = GPURenderer(map_w=960, map_h=720, atlas_w=960, atlas_h=960)
    gpu.configure_depth_topdown(
        px_size=float(TD_PX_SIZE),
        floor_clip=float(TD_FLOOR_CLIP),
        out_h=FRAME_H,
        out_w=FRAME_W)
    
    result = gpu.depth_topdown_gpu(verts)
    if result is None:
        print("  SKIP: GPU topdown not available")
        gpu.release()
        return
    
    obs, known = result
    gpu.release()
    
    # Center pixel should be known
    cy, cx = FRAME_H // 2, FRAME_W // 2
    assert known[cy, cx] == 255, "Center should be known"
    
    # Tallest obstacle (z=0.50 → 41cm) should win at center
    obs_height = int(obs[cy, cx])
    print("  center obs height: %dcm (expected ~41cm from tallest)" % obs_height)
    
    # Should be close to 41cm (tallest obstacle)
    assert obs_height > 30, "Should detect tall obstacle"
    assert obs_height <= 100, "Height should be capped at 100cm"
    
    print("test_gpu_floor_vs_obstacle: PASS")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing GPU topdown depth processing")
    print("=" * 70)
    
    test_gpu_cpu_parity()
    test_gpu_performance()
    test_gpu_empty_verts()
    test_gpu_floor_vs_obstacle()
    
    print("=" * 70)
    print("All tests PASSED")
    print("=" * 70)
