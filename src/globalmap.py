"""globalmap.py – World-frame 2D occupancy grid with 3-frame consensus gating.

960×720 pixels at 2 cm/px = 19.2 m × 14.4 m coverage.
World frame: x-right, y-up, origin at robot's start position.

Each cell stores a confidence byte (0–255):
  0   = strong obstacle evidence
  128 = unknown (initial)
  255 = strong free-space evidence

Observations from the ego-frame cameras are warped into global space
at 30 fps.  Only pixels where the last 3 consecutive frames agree
(all free or all obstacle) are applied to the main map, preventing
transient noise from corrupting the persistent map.
"""

import numpy as np
import cv2

MAP_W = 960
MAP_H = 720
PX_SIZE = 0.02        # metres per pixel (2 cm)
ORIGIN_X = MAP_W // 2  # pixel 480 = world x=0
ORIGIN_Y = MAP_H // 2  # pixel 360 = world y=0

UNKNOWN_VAL = 128
FREE_DISPLAY = 255
OBS_DISPLAY = 64
UNK_DISPLAY = 160

FREE_THRESH = 190
OBS_THRESH = 90

EVIDENCE_UP = 8
EVIDENCE_DOWN = 8

CONSENSUS_N = 3


class GlobalMap:
    def __init__(self):
        self._map = np.full((MAP_H, MAP_W), UNKNOWN_VAL, dtype=np.uint8)
        self._ring = [np.zeros((MAP_H, MAP_W), dtype=np.int8)
                      for _ in range(CONSENSUS_N)]
        self._ring_idx = 0
        self._count = 0

    def update(self, obs_ego, known_ego, x, y, theta,
               ego_cx, ego_cy, ego_px_size=0.01):
        """Feed one frame of ego-space observations and robot pose."""
        h, w = obs_ego.shape[:2]

        ego_obs = np.zeros((h, w), dtype=np.uint8)
        free = (known_ego > 0) & (obs_ego == 0)
        ego_obs[free] = 1
        ego_obs[obs_ego > 0] = 2

        M_fwd = self._forward_affine(x, y, theta, ego_cx, ego_cy, ego_px_size)
        global_obs = cv2.warpAffine(
            ego_obs, M_fwd, (MAP_W, MAP_H),
            flags=cv2.INTER_NEAREST, borderValue=0)

        signed = np.zeros((MAP_H, MAP_W), dtype=np.int8)
        signed[global_obs == 1] = 1
        signed[global_obs == 2] = -1

        self._ring[self._ring_idx] = signed
        self._ring_idx = (self._ring_idx + 1) % CONSENSUS_N
        self._count += 1

        if self._count < CONSENSUS_N:
            return

        r0, r1, r2 = self._ring
        all_free = (r0 == 1) & (r1 == 1) & (r2 == 1)
        all_obs = (r0 == -1) & (r1 == -1) & (r2 == -1)

        m = self._map.astype(np.int16)
        m[all_free] = np.minimum(m[all_free] + EVIDENCE_UP, 255)
        m[all_obs] = np.maximum(m[all_obs] - EVIDENCE_DOWN, 0)
        self._map = m.astype(np.uint8)

    def project_to_ego(self, x, y, theta,
                       ego_cx, ego_cy, ego_px_size, ego_h, ego_w):
        """Warp global map back to ego-centred coordinates for safety."""
        M_inv = self._inverse_affine(x, y, theta, ego_cx, ego_cy, ego_px_size)
        return cv2.warpAffine(
            self._map, M_inv, (ego_w, ego_h),
            flags=cv2.INTER_NEAREST, borderValue=UNKNOWN_VAL)

    def render(self, x, y, theta, trail_xy=None):
        """MAP_W × MAP_H BGR image with robot arrow and optional trajectory.

        trail_xy: (N, 2) float64 world [x, y] array, oldest→newest.
        """
        disp = np.full((MAP_H, MAP_W), UNK_DISPLAY, dtype=np.uint8)
        disp[self._map > FREE_THRESH] = FREE_DISPLAY
        disp[self._map < OBS_THRESH] = OBS_DISPLAY
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        if trail_xy is not None and len(trail_xy) >= 2:
            tc = (ORIGIN_X + trail_xy[:, 0] / PX_SIZE).astype(np.int32)
            tr = (ORIGIN_Y - trail_xy[:, 1] / PX_SIZE).astype(np.int32)
            keep = (tc >= 0) & (tc < MAP_W) & (tr >= 0) & (tr < MAP_H)
            pts = np.column_stack([tc[keep], tr[keep]])
            if len(pts) >= 2:
                cv2.polylines(disp, [pts.reshape(-1, 1, 2)], False,
                              (80, 140, 255), 1, cv2.LINE_AA)

        rc = int(ORIGIN_X + x / PX_SIZE)
        rr = int(ORIGIN_Y - y / PX_SIZE)
        if 6 <= rc < MAP_W - 6 and 6 <= rr < MAP_H - 6:
            length = 10
            dx = int(length * np.cos(theta))
            dy = int(-length * np.sin(theta))
            cv2.arrowedLine(disp, (rc - dx, rr - dy), (rc + dx, rr + dy),
                            (255, 180, 0), 2, tipLength=0.3)
        return disp

    # ── affine transforms ─────────────────────────────────────────

    @staticmethod
    def _forward_affine(x, y, theta, ego_cx, ego_cy, ego_px_size):
        """2×3 affine: ego pixel → global pixel (forward mapping)."""
        ct = np.cos(theta)
        st = np.sin(theta)
        s = ego_px_size / PX_SIZE

        return np.float64([
            [ s * ct,  s * st,
              ORIGIN_X + x / PX_SIZE - ego_cx * s * ct - ego_cy * s * st],
            [-s * st,  s * ct,
              ORIGIN_Y - y / PX_SIZE + ego_cx * s * st - ego_cy * s * ct],
        ])

    @staticmethod
    def _inverse_affine(x, y, theta, ego_cx, ego_cy, ego_px_size):
        """2×3 affine: global pixel → ego pixel (src→dst for warpAffine)."""
        ct = np.cos(theta)
        st = np.sin(theta)
        r = PX_SIZE / ego_px_size

        t_gx = ORIGIN_X + x / PX_SIZE
        t_gy = ORIGIN_Y - y / PX_SIZE

        return np.float64([
            [ r * ct, -r * st,  ego_cx - r * ct * t_gx + r * st * t_gy],
            [ r * st,  r * ct,  ego_cy - r * st * t_gx - r * ct * t_gy],
        ])
