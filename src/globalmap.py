"""globalmap.py – World-frame 2D occupancy grid constants and SLAM backing store.

960×720 pixels at 2 cm/px = 19.2 m × 14.4 m coverage.
World frame: x-right, y-up, origin at robot's start position.

The GPU renderer (gpu_render.py) owns the authoritative map textures.
This module provides constants, affine helpers used by SLAM, and the
CPU arrays that SLAM rebuilds into after loop closure.
"""

import math
import numpy as np

MAP_W = 960
MAP_H = 720
PX_SIZE = 0.02        # metres per pixel (2 cm)
ORIGIN_X = MAP_W // 2  # pixel 480 = world x=0
ORIGIN_Y = MAP_H // 2  # pixel 360 = world y=0

UNKNOWN_VAL = 128
FREE_THRESH = 190
OBS_THRESH = 90

EVIDENCE_STEP = 50


class GlobalMap:
    """Minimal backing store for SLAM rebuilds.  The GPU handles
    all per-frame evidence updates and rendering."""

    def __init__(self):
        self._map = np.full((MAP_H, MAP_W), UNKNOWN_VAL, dtype=np.uint8)
        self._height_map = np.zeros((MAP_H, MAP_W), dtype=np.uint8)

    @property
    def confidence_map(self):
        return self._map

    @property
    def height_map(self):
        return self._height_map

    @staticmethod
    def _forward_affine(x, y, theta, ego_cx, ego_cy, ego_px_size):
        """2×3 affine: ego pixel → global pixel."""
        ct = math.cos(theta)
        st = math.sin(theta)
        s = ego_px_size / PX_SIZE
        return np.float64([
            [ s * ct,  s * st,
              ORIGIN_X + x / PX_SIZE - ego_cx * s * ct - ego_cy * s * st],
            [-s * st,  s * ct,
              ORIGIN_Y - y / PX_SIZE + ego_cx * s * st - ego_cy * s * ct],
        ])

    @staticmethod
    def _inverse_affine(x, y, theta, ego_cx, ego_cy, ego_px_size):
        """2×3 affine: global pixel → ego pixel."""
        ct = math.cos(theta)
        st = math.sin(theta)
        r = PX_SIZE / ego_px_size
        t_gx = ORIGIN_X + x / PX_SIZE
        t_gy = ORIGIN_Y - y / PX_SIZE
        return np.float64([
            [ r * ct, -r * st,  ego_cx - r * ct * t_gx + r * st * t_gy],
            [ r * st,  r * ct,  ego_cy - r * st * t_gx - r * ct * t_gy],
        ])

    @staticmethod
    def _build_robot_mesh():
        """Build hardcoded 3D robot mesh (origin = axle centre on ground, +X fwd).

        Returns (verts, faces, colors):
          verts  — (N, 3) float32 in metres
          faces  — list of int32 arrays (polygon vertex indices per face)
          colors — (M, 3) uint8, RGB per face
        """
        vl, fl, cl = [], [], []
        R = 0.0857

        def _box(x0, y0, z0, x1, y1, z1, rgb):
            n = len(vl)
            vl.extend([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
            for q in ([n,n+3,n+2,n+1],[n+4,n+5,n+6,n+7],
                       [n,n+1,n+5,n+4],[n+2,n+3,n+7,n+6],
                       [n,n+4,n+7,n+3],[n+1,n+2,n+6,n+5]):
                fl.append(np.array(q, dtype=np.int32))
                cl.append(rgb)

        def _wheel(cx, cz, y0, y1, radius, rgb, ns=10):
            n = len(vl)
            ang = np.linspace(0, 2*np.pi, ns, endpoint=False)
            for a in ang:
                vl.append([cx+radius*np.cos(a), y0, cz+radius*np.sin(a)])
            for a in ang:
                vl.append([cx+radius*np.cos(a), y1, cz+radius*np.sin(a)])
            for i in range(ns):
                j = (i+1) % ns
                fl.append(np.array([n+i, n+ns+i, n+ns+j, n+j], dtype=np.int32))
                cl.append(rgb)
            fl.append(np.arange(n, n+ns, dtype=np.int32))
            cl.append([min(c+12, 255) for c in rgb])
            fl.append(np.arange(n+2*ns-1, n+ns-1, -1, dtype=np.int32))
            cl.append([min(c+12, 255) for c in rgb])

        z_base = 0.05
        z_top  = 0.22
        x_back = -0.15
        x_front = 0.13
        x_slope = 0.0
        y_hw = 0.16

        n = len(vl)
        vl.extend([
            [x_front, -y_hw, z_base],
            [x_back,  -y_hw, z_base],
            [x_back,  -y_hw, z_top],
            [x_slope, -y_hw, z_top],
            [x_front,  y_hw, z_base],
            [x_back,   y_hw, z_base],
            [x_back,   y_hw, z_top],
            [x_slope,  y_hw, z_top],
        ])
        body_rgb = [45, 45, 50]
        for q in ([n,n+1,n+5,n+4], [n+1,n+2,n+6,n+5],
                  [n+2,n+3,n+7,n+6], [n+3,n,n+4,n+7],
                  [n,n+3,n+2,n+1], [n+4,n+5,n+6,n+7]):
            fl.append(np.array(q, dtype=np.int32))
            cl.append(body_rgb)

        _wheel(0, R, -0.21, -0.16, R*0.97, [22, 22, 26])
        _wheel(0, R,  0.16,  0.21, R*0.97, [22, 22, 26])
        _box(-0.17, -0.015, 0.0, -0.15, 0.015, z_base, [40, 40, 45])
        _box(-0.19, -0.015, 0.0, -0.15, 0.015, 0.03, [35, 35, 40])
        _box(-0.165, -0.015, z_top, -0.135, 0.015, 1.00, [100, 105, 115])
        _box(-0.135, -0.015, 0.97, 0.015, 0.015, 1.00, [85, 90, 100])
        _box(-0.05, -0.04, 1.00, 0.04, 0.04, 1.025, [30, 55, 80])
        _box(0.015, -0.03, 0.94, 0.05, 0.03, 0.97, [30, 55, 80])
        _box(-0.14, -0.04, z_top, -0.04, 0.04, z_top + 0.025, [40, 55, 40])

        return (np.array(vl, dtype=np.float32), fl,
                np.array(cl, dtype=np.uint8))
