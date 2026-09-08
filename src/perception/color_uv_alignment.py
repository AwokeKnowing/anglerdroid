"""Color UV alignment for RGB-D VO (CONTRACT step 4 remaining gap).

Provides proper depth→color UV projection using RealSense camera intrinsics
and extrinsics, fixing the gap where VO gray ignore previously used a simple
depth-grid remap that ignored camera calibration.

Env gate (live wire, default off): ``KEVIN_VO_COLOR_ALIGN=1`` — enables
proper rs2_project-style UV mapping when depth and color intrinsics are
available. Falls back to depth-grid remap (existing behavior) when intrinsics
unavailable or flag off.

Gap documented in CONTRACT.md line 89 / dynamic_mask.py line 228-229.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class CameraIntrinsics:
    """Camera intrinsics for depth→color UV projection.
    
    Holds focal lengths (fx, fy), principal point (ppx, ppy), and optional
    distortion coefficients. Supports rs2_project-style pixel projection.
    """
    __slots__ = ('width', 'height', 'fx', 'fy', 'ppx', 'ppy', 'coeffs')
    
    def __init__(self, width: int, height: int,
                 fx: float, fy: float, ppx: float, ppy: float,
                 coeffs: Optional[np.ndarray] = None):
        """Initialize camera intrinsics.
        
        Parameters
        ----------
        width, height : int
            Image dimensions (pixels)
        fx, fy : float
            Focal lengths (pixels)
        ppx, ppy : float
            Principal point (pixels)
        coeffs : optional (5,) float
            Distortion coefficients [k1, k2, p1, p2, k3] (Brown-Conrady model)
        """
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.ppx = float(ppx)
        self.ppy = float(ppy)
        self.coeffs = np.asarray(coeffs, dtype=np.float32) if coeffs is not None else None


class DepthToColorAlignment:
    """Depth→color UV alignment using camera calibration.
    
    When depth and color intrinsics + extrinsics are available, projects
    depth vertices to calibrated color UV coordinates. Falls back to
    depth-grid linear remap (existing behavior) when calibration unavailable.
    """
    
    def __init__(self,
                 depth_intr: Optional[CameraIntrinsics] = None,
                 color_intr: Optional[CameraIntrinsics] = None,
                 depth_to_color_extr: Optional[np.ndarray] = None):
        """Initialize depth→color alignment.
        
        Parameters
        ----------
        depth_intr : optional CameraIntrinsics
            Depth camera intrinsics
        color_intr : optional CameraIntrinsics
            Color camera intrinsics
        depth_to_color_extr : optional (3,) float
            Translation [tx, ty, tz] from depth to color camera (meters).
            Assumes aligned optical axes (typical for D435); full 6-DOF
            extrinsics (rotation+translation) can be added later if needed.
        """
        self.depth_intr = depth_intr
        self.color_intr = color_intr
        self.depth_to_color_extr = (
            np.asarray(depth_to_color_extr, dtype=np.float32).reshape(3)
            if depth_to_color_extr is not None else None)
        self._has_calibration = (
            depth_intr is not None
            and color_intr is not None
            and depth_to_color_extr is not None)
    
    def has_calibration(self) -> bool:
        """Check if proper depth→color calibration is available."""
        return self._has_calibration
    
    def project_verts_to_color_uv(
        self,
        verts: np.ndarray,
        vert_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project depth vertices to calibrated color UV coordinates.
        
        Uses rs2_project-style math: deproject depth pixel → 3D point in depth
        frame → transform to color frame (extrinsics) → project to color UV
        (color intrinsics).
        
        Parameters
        ----------
        verts : (N, 3) float32
            Depth vertices in camera frame (meters)
        vert_indices : (M,) int32
            Subset of vertex indices to project (after filtering/striding)
        
        Returns
        -------
        u : (M,) int32
            Color U (column) coordinates (pixels)
        v : (M,) int32
            Color V (row) coordinates (pixels)
        
        Notes
        -----
        Out-of-bounds coordinates (outside color image) are NOT filtered here —
        caller must check ``(0 <= u < color_width) & (0 <= v < color_height)``.
        """
        if not self._has_calibration:
            raise ValueError("project_verts_to_color_uv requires calibration")
        
        # Extract valid verts
        pts_depth = verts[vert_indices].astype(np.float32)  # (M, 3)
        
        # Transform depth frame → color frame (extrinsics)
        # For D435, depth and color optical axes are typically aligned; we only
        # translate by the stereo baseline (~0.05m in X). Full rotation can be
        # added here if needed for tilted rigs.
        pts_color = pts_depth + self.depth_to_color_extr.reshape(1, 3)
        
        # Project color-frame 3D → color UV (rs2_project)
        x, y, z = pts_color[:, 0], pts_color[:, 1], pts_color[:, 2]
        # Avoid divide-by-zero (invalid/occluded points)
        valid = (np.abs(z) > 1e-6) & np.isfinite(z)
        u_arr = np.zeros(len(x), dtype=np.int32)
        v_arr = np.zeros(len(x), dtype=np.int32)
        
        if np.any(valid):
            x_v, y_v, z_v = x[valid], y[valid], z[valid]
            # Pinhole projection: u = fx * (x/z) + ppx
            u_f = self.color_intr.fx * (x_v / z_v) + self.color_intr.ppx
            v_f = self.color_intr.fy * (y_v / z_v) + self.color_intr.ppy
            u_arr[valid] = np.floor(u_f).astype(np.int32)
            v_arr[valid] = np.floor(v_f).astype(np.int32)
        
        return u_arr, v_arr


def extract_rs_intrinsics_extrinsics(profile):
    """Extract depth/color intrinsics and extrinsics from RealSense profile.
    
    Parameters
    ----------
    profile : rs.pipeline_profile
        Active RealSense pipeline profile (from ``rs.pipeline().start(cfg)``)
    
    Returns
    -------
    depth_intr : CameraIntrinsics or None
        Depth camera intrinsics, or None if extraction failed
    color_intr : CameraIntrinsics or None
        Color camera intrinsics, or None if extraction failed
    depth_to_color_translation : (3,) ndarray or None
        Translation [tx, ty, tz] from depth to color camera (meters), or None
    
    Notes
    -----
    Requires ``pyrealsense2``. Returns ``(None, None, None)`` on any error
    (e.g. missing streams, unsupported device).
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None, None, None
    
    try:
        depth_stream = profile.get_stream(rs.stream.depth)
        color_stream = profile.get_stream(rs.stream.color)
        
        depth_vsp = depth_stream.as_video_stream_profile()
        color_vsp = color_stream.as_video_stream_profile()
        
        d_intr = depth_vsp.get_intrinsics()
        c_intr = color_vsp.get_intrinsics()
        
        # Extrinsics: depth → color
        extr = depth_stream.get_extrinsics_to(color_stream)
        translation = np.array(extr.translation, dtype=np.float32)
        
        depth_cam = CameraIntrinsics(
            d_intr.width, d_intr.height,
            d_intr.fx, d_intr.fy, d_intr.ppx, d_intr.ppy,
            coeffs=np.array(d_intr.coeffs, dtype=np.float32) if d_intr.coeffs else None)
        
        color_cam = CameraIntrinsics(
            c_intr.width, c_intr.height,
            c_intr.fx, c_intr.fy, c_intr.ppx, c_intr.ppy,
            coeffs=np.array(c_intr.coeffs, dtype=np.float32) if c_intr.coeffs else None)
        
        return depth_cam, color_cam, translation
    
    except Exception:
        return None, None, None
