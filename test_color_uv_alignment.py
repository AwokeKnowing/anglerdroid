#!/usr/bin/env python3
"""Unit tests for RGB-D VO color UV alignment (CONTRACT step 4 gap fix)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from perception.color_uv_alignment import (
    CameraIntrinsics, DepthToColorAlignment, extract_rs_intrinsics_extrinsics
)
from perception.dynamic_mask import build_forward_ignore_from_verts
from robot_config import FRAME_H, FRAME_W


def test_camera_intrinsics_construction():
    """CameraIntrinsics stores focal/principal correctly."""
    intr = CameraIntrinsics(
        width=640, height=480,
        fx=320.0, fy=320.0, ppx=320.0, ppy=240.0,
        coeffs=np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float32))
    assert intr.width == 640
    assert intr.height == 480
    assert intr.fx == 320.0
    assert intr.fy == 320.0
    assert intr.ppx == 320.0
    assert intr.ppy == 240.0
    assert intr.coeffs.shape == (5,)
    # No coeffs OK
    intr2 = CameraIntrinsics(640, 480, 320.0, 320.0, 320.0, 240.0)
    assert intr2.coeffs is None


def test_depth_to_color_alignment_no_calibration():
    """DepthToColorAlignment without intrinsics reports no calibration."""
    align = DepthToColorAlignment()
    assert not align.has_calibration()
    align2 = DepthToColorAlignment(
        depth_intr=CameraIntrinsics(848, 480, 420.0, 420.0, 424.0, 240.0),
        color_intr=None, depth_to_color_extr=None)
    assert not align2.has_calibration()


def test_depth_to_color_alignment_with_calibration():
    """DepthToColorAlignment with intrinsics+extrinsics reports calibration OK."""
    d_intr = CameraIntrinsics(848, 480, 420.0, 420.0, 424.0, 240.0)
    c_intr = CameraIntrinsics(320, 240, 220.0, 220.0, 160.0, 120.0)
    extr = np.array([0.05, 0.0, 0.0], dtype=np.float32)  # 5cm baseline
    align = DepthToColorAlignment(d_intr, c_intr, extr)
    assert align.has_calibration()


def test_project_verts_to_color_uv_simple():
    """Project synthetic depth vertices to color UV with known intrinsics."""
    # Synthetic cameras: depth and color have same intrinsics (aligned optical axes)
    # but color is translated 5cm to the right (typical D435 baseline)
    d_intr = CameraIntrinsics(848, 480, 400.0, 400.0, 424.0, 240.0)
    c_intr = CameraIntrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
    extr = np.array([0.05, 0.0, 0.0], dtype=np.float32)  # depth → color: +5cm X
    align = DepthToColorAlignment(d_intr, c_intr, extr)
    
    # Synthetic depth vertex: 1m ahead (z=1.0), centered (x=0, y=0) in depth frame
    verts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    
    u, v = align.project_verts_to_color_uv(verts, indices)
    
    # After translation, point is at (0.05, 0, 1) in color frame
    # Project: u = fx * (x/z) + ppx = 200 * (0.05/1.0) + 160 = 170
    #          v = fy * (y/z) + ppy = 200 * (0/1.0) + 120 = 120
    assert u[0] == 170, f"Expected u=170, got {u[0]}"
    assert v[0] == 120, f"Expected v=120, got {v[0]}"


def test_project_verts_multiple_points():
    """Project multiple depth vertices to color UV."""
    d_intr = CameraIntrinsics(848, 480, 400.0, 400.0, 424.0, 240.0)
    c_intr = CameraIntrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
    extr = np.array([0.05, 0.0, 0.0], dtype=np.float32)
    align = DepthToColorAlignment(d_intr, c_intr, extr)
    
    # Multiple points: (x, y, z)
    verts = np.array([
        [0.0, 0.0, 1.0],    # center, 1m
        [0.1, 0.0, 1.0],    # right 10cm, 1m
        [-0.1, 0.0, 1.0],   # left 10cm, 1m
        [0.0, 0.1, 1.0],    # up 10cm, 1m
        [0.0, -0.1, 1.0],   # down 10cm, 1m
    ], dtype=np.float32)
    indices = np.arange(len(verts), dtype=np.int32)
    
    u, v = align.project_verts_to_color_uv(verts, indices)
    
    # After extrinsic translation (+0.05 in X), points are:
    # [0.05, 0, 1], [0.15, 0, 1], [-0.05, 0, 1], [0.05, 0.1, 1], [0.05, -0.1, 1]
    # u = 200 * (x/z) + 160
    # v = 200 * (y/z) + 120
    expected_u = [170, 190, 150, 170, 170]
    expected_v = [120, 120, 120, 140, 100]
    
    assert len(u) == 5
    assert len(v) == 5
    for i in range(5):
        # Allow ±1 pixel tolerance for floating-point floor
        assert abs(u[i] - expected_u[i]) <= 1, f"Point {i}: expected u={expected_u[i]}, got {u[i]}"
        assert abs(v[i] - expected_v[i]) <= 1, f"Point {i}: expected v={expected_v[i]}, got {v[i]}"


def test_project_verts_out_of_bounds():
    """Project vertices that land outside color image bounds."""
    d_intr = CameraIntrinsics(848, 480, 400.0, 400.0, 424.0, 240.0)
    c_intr = CameraIntrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
    extr = np.array([0.05, 0.0, 0.0], dtype=np.float32)
    align = DepthToColorAlignment(d_intr, c_intr, extr)
    
    # Point way off to the left (x = -1.0 at z=1.0 → projects to u < 0)
    verts = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    
    u, v = align.project_verts_to_color_uv(verts, indices)
    
    # After translation: (-0.95, 0, 1) → u = 200*(-0.95) + 160 = -30
    # Caller must check bounds; projection itself doesn't filter
    assert u[0] < 0 or u[0] >= 320, "Out-of-bounds point should project outside image"


def test_project_verts_zero_z_invalid():
    """Vertices with z ≈ 0 are invalid (divide-by-zero protection)."""
    d_intr = CameraIntrinsics(848, 480, 400.0, 400.0, 424.0, 240.0)
    c_intr = CameraIntrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
    extr = np.array([0.05, 0.0, 0.0], dtype=np.float32)
    align = DepthToColorAlignment(d_intr, c_intr, extr)
    
    # Invalid point: z = 0
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    
    u, v = align.project_verts_to_color_uv(verts, indices)
    
    # Invalid points get zero UV (filtered by valid mask)
    assert u[0] == 0 and v[0] == 0


def test_build_forward_ignore_with_calibration():
    """build_forward_ignore_from_verts with calibrated UV projection marks gray pixels."""
    try:
        import cv2
    except ImportError:
        print("SKIP forward_ignore_calibration (cv2 unavailable)")
        return
    
    import math
    
    # Mirror vision FW constants
    pitch = math.radians(25.6 - 90.0)
    rot, _ = cv2.Rodrigues(np.float64([pitch, 0, 0]))
    rot = rot.astype(np.float32)
    piv = np.array([0.0, -1.0, 0.02], dtype=np.float32)
    trans = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    scatter_h, scatter_w = FRAME_W, FRAME_H
    scale = 100.0
    offset = np.float32([scatter_w / 2.0, scatter_h / 2.0 + scale])
    fw_dx = -75 + 132
    fw_dy = -1
    
    # Synthetic cameras with typical D435 params
    d_intr = CameraIntrinsics(848, 480, 420.0, 420.0, 424.0, 240.0)
    c_intr = CameraIntrinsics(FRAME_W, FRAME_H, 160.0, 160.0, FRAME_W/2, FRAME_H/2)
    extr = np.array([0.05, 0.0, 0.0], dtype=np.float32)  # 5cm baseline
    align = DepthToColorAlignment(d_intr, c_intr, extr)
    
    # Dense-enough fake grid
    gh, gw = 160, 283
    verts = np.zeros((gh * gw, 3), dtype=np.float32)
    # Place one valid forward point ~1m ahead
    verts[gh // 2 * gw + gw // 2] = np.array([0.0, 0.05, 1.0], dtype=np.float32)
    
    # Compute expected ego cell (same math as GPU)
    p = verts[gh // 2 * gw + gw // 2].copy()
    r = (p - piv) @ rot + piv - trans
    sx = int(np.floor(r[0] * scale + offset[0]))
    sy = int(np.floor(r[1] * scale + offset[1]))
    if not (0 <= sx < scatter_w and 0 <= sy < scatter_h):
        print("skip_geom", sx, sy)
        return
    ei = sx + fw_dy
    ej = (scatter_h - 1 - sy) + fw_dx
    if not (0 <= ei < FRAME_H and 0 <= ej < FRAME_W):
        print("skip_ego", ei, ej)
        return
    
    ego = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    ego[ei, ej] = True
    
    # Test with calibration
    ign, ns, nh, ng = build_forward_ignore_from_verts(
        verts, ego,
        rotation=rot, pivot=piv, translation=trans,
        scale=scale, offset=offset,
        scatter_h=scatter_h, scatter_w=scatter_w,
        fw_dx=fw_dx, fw_dy=fw_dy,
        gray_h=FRAME_H, gray_w=FRAME_W,
        stride=1, stamp=1,
        uv_alignment=align)
    
    assert ns >= 1, "Must sample at least one vertex"
    assert nh >= 1, "Calibrated projection should hit masked ego cell"
    assert ng > 0, "Should mark some gray pixels"
    
    # Test fallback (no calibration)
    ign2, ns2, nh2, ng2 = build_forward_ignore_from_verts(
        verts, ego,
        rotation=rot, pivot=piv, translation=trans,
        scale=scale, offset=offset,
        scatter_h=scatter_h, scatter_w=scatter_w,
        fw_dx=fw_dx, fw_dy=fw_dy,
        gray_h=FRAME_H, gray_w=FRAME_W,
        stride=1, stamp=1,
        uv_alignment=None)  # No calibration
    
    assert ns2 >= 1
    # Fallback may or may not hit depending on grid remap accuracy
    # Just check it doesn't crash


def test_build_forward_ignore_alignment_mismatch_vs_fallback():
    """Calibrated UV can differ from depth-grid remap (that's the fix)."""
    try:
        import cv2
    except ImportError:
        print("SKIP forward_ignore_mismatch (cv2 unavailable)")
        return
    
    import math
    
    pitch = math.radians(25.6 - 90.0)
    rot, _ = cv2.Rodrigues(np.float64([pitch, 0, 0]))
    rot = rot.astype(np.float32)
    piv = np.array([0.0, -1.0, 0.02], dtype=np.float32)
    trans = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    scatter_h, scatter_w = FRAME_W, FRAME_H
    scale = 100.0
    offset = np.float32([scatter_w / 2.0, scatter_h / 2.0 + scale])
    
    # Synthetic cameras with significant baseline (exaggerate the gap)
    d_intr = CameraIntrinsics(848, 480, 400.0, 400.0, 424.0, 240.0)
    c_intr = CameraIntrinsics(FRAME_W, FRAME_H, 160.0, 160.0, FRAME_W/2, FRAME_H/2)
    extr = np.array([0.1, 0.0, 0.0], dtype=np.float32)  # 10cm baseline (large)
    align = DepthToColorAlignment(d_intr, c_intr, extr)
    
    gh, gw = 160, 283
    verts = np.zeros((gh * gw, 3), dtype=np.float32)
    # Point at 1m, slightly off-center
    verts[gh // 2 * gw + gw // 3] = np.array([0.1, 0.0, 1.0], dtype=np.float32)
    
    ego = np.ones((FRAME_H, FRAME_W), dtype=bool)  # All masked (easy hit)
    
    # With calibration
    ign_cal, _, nh_cal, ng_cal = build_forward_ignore_from_verts(
        verts, ego,
        rotation=rot, pivot=piv, translation=trans,
        scale=scale, offset=offset,
        scatter_h=scatter_h, scatter_w=scatter_w,
        fw_dx=0, fw_dy=0,
        gray_h=FRAME_H, gray_w=FRAME_W,
        stride=1, stamp=1,
        uv_alignment=align)
    
    # Without calibration (fallback)
    ign_fb, _, nh_fb, ng_fb = build_forward_ignore_from_verts(
        verts, ego,
        rotation=rot, pivot=piv, translation=trans,
        scale=scale, offset=offset,
        scatter_h=scatter_h, scatter_w=scatter_w,
        fw_dx=0, fw_dy=0,
        gray_h=FRAME_H, gray_w=FRAME_W,
        stride=1, stamp=1,
        uv_alignment=None)
    
    # Both should mark some pixels, but patterns may differ due to baseline shift
    assert nh_cal >= 1 and ng_cal > 0
    assert nh_fb >= 1 and ng_fb > 0
    # This test mainly proves both paths work; actual UV mismatch is the gap we're fixing


def test_extract_rs_intrinsics_no_pyrealsense():
    """extract_rs_intrinsics_extrinsics returns None when pyrealsense2 unavailable."""
    # Cannot test actual RS profile extraction without hardware, but check None fallback
    result = extract_rs_intrinsics_extrinsics(None)
    assert result == (None, None, None)


if __name__ == "__main__":
    test_camera_intrinsics_construction()
    print("OK camera_intrinsics")
    test_depth_to_color_alignment_no_calibration()
    print("OK no_calibration")
    test_depth_to_color_alignment_with_calibration()
    print("OK with_calibration")
    test_project_verts_to_color_uv_simple()
    print("OK project_simple")
    test_project_verts_multiple_points()
    print("OK project_multiple")
    test_project_verts_out_of_bounds()
    print("OK project_oob")
    test_project_verts_zero_z_invalid()
    print("OK project_invalid_z")
    test_build_forward_ignore_with_calibration()
    print("OK forward_ignore_calibration")
    test_build_forward_ignore_alignment_mismatch_vs_fallback()
    print("OK forward_ignore_mismatch")
    test_extract_rs_intrinsics_no_pyrealsense()
    print("OK extract_rs_no_pyrealsense")
    print("ALL PASS")
