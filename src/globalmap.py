"""globalmap.py – World-frame 2D occupancy grid with 3-frame consensus gating.

512×512 pixels at 4 cm/px = 20.48 m × 20.48 m coverage.
World frame: x-right, y-up, origin at robot's start position.
Displayed at 0.25 scale (128×128).

Each cell stores a confidence byte (0–255):
  0   = strong obstacle evidence
  128 = unknown (initial)
  255 = strong free-space evidence

Observations from the ego-frame cameras are warped into global space
at 30 fps.  Only pixels where the last 3 consecutive frames agree
(all free or all obstacle) are applied to the main map, preventing
transient noise from corrupting the persistent map.

The ego costmap can be reconstructed by projecting the global map
back to robot-centred coordinates each frame.
"""

import numpy as np
import cv2

SIZE = 512
PX_SIZE = 0.04        # metres per pixel
DISPLAY_SCALE = 0.25  # render at 128×128
ORIGIN = SIZE // 2    # pixel (256, 256) = world (0, 0)

UNKNOWN_VAL = 128
FREE_DISPLAY = 255
OBS_DISPLAY = 64
UNK_DISPLAY = 160

FREE_THRESH = 190     # above this → display as free
OBS_THRESH = 90       # below this → display as obstacle

EVIDENCE_UP = 8       # per consensus: nudge toward free
EVIDENCE_DOWN = 8     # per consensus: nudge toward obstacle

CONSENSUS_N = 3


class GlobalMap:
    def __init__(self):
        self._map = np.full((SIZE, SIZE), UNKNOWN_VAL, dtype=np.uint8)
        self._ring = [np.zeros((SIZE, SIZE), dtype=np.int8) for _ in range(CONSENSUS_N)]
        self._ring_idx = 0
        self._count = 0

    def update(self, obs_ego, known_ego, x, y, theta,
               ego_cx, ego_cy, ego_px_size=0.01):
        """Feed one frame of ego-space observations and robot pose.

        obs_ego   – uint8 (H, W), obstacle pixels > 0
        known_ego – uint8 (H, W), observed-area pixels > 0
        x, y      – world position (metres)
        theta     – world heading (radians, 0 = +x, CCW positive)
        ego_cx/cy – robot centre in ego image (pixels)
        ego_px_size – ego image scale (metres/pixel), default 0.01
        """
        h, w = obs_ego.shape[:2]

        ego_obs = np.zeros((h, w), dtype=np.uint8)
        free = (known_ego > 0) & (obs_ego == 0)
        ego_obs[free] = 1        # free
        ego_obs[obs_ego > 0] = 2  # obstacle

        M_fwd = self._forward_affine(x, y, theta, ego_cx, ego_cy, ego_px_size)
        global_obs = cv2.warpAffine(
            ego_obs, M_fwd, (SIZE, SIZE),
            flags=cv2.INTER_NEAREST, borderValue=0)

        signed = np.zeros((SIZE, SIZE), dtype=np.int8)
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
        """Warp the global map back to ego-centred coordinates.

        Returns uint8 (ego_h, ego_w): 0=free confirmed, 128=unknown, 255=obstacle confirmed.
        Suitable for feeding into _build_costmap as persistent_obs.
        """
        M_inv = self._inverse_affine(x, y, theta, ego_cx, ego_cy, ego_px_size)
        ego_proj = cv2.warpAffine(
            self._map, M_inv, (ego_w, ego_h),
            flags=cv2.INTER_NEAREST,
            borderValue=UNKNOWN_VAL)
        return ego_proj

    def render_display(self):
        """128×128 BGR image for imshow / atlas embedding."""
        disp = np.full((SIZE, SIZE), UNK_DISPLAY, dtype=np.uint8)
        disp[self._map > FREE_THRESH] = FREE_DISPLAY
        disp[self._map < OBS_THRESH] = OBS_DISPLAY
        small = cv2.resize(disp, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                           interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    def render_display_with_robot(self, x, y, theta):
        """128×128 BGR display with robot marker."""
        disp = self.render_display()
        s = DISPLAY_SCALE
        rc = int(ORIGIN * s + x / PX_SIZE * s)
        rr = int(ORIGIN * s - y / PX_SIZE * s)
        sz = disp.shape[0]
        if 2 <= rc < sz - 2 and 2 <= rr < sz - 2:
            length = 4
            dx = int(length * np.cos(theta))
            dy = int(-length * np.sin(theta))
            cv2.arrowedLine(disp, (rc - dx, rr - dy), (rc + dx, rr + dy),
                            (255, 180, 0), 1, tipLength=0.4)
        return disp

    # ── affine transforms ─────────────────────────────────────────

    @staticmethod
    def _forward_affine(x, y, theta, ego_cx, ego_cy, ego_px_size):
        """2×3 affine: ego pixel → global pixel (forward mapping).

        Compose: ego_px → local_metres → world_metres → global_px.
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        s = ego_px_size / PX_SIZE

        M = np.float64([
            [ s * ct,  s * st,  ORIGIN + x / PX_SIZE - ego_cx * s * ct - ego_cy * s * st],
            [-s * st,  s * ct,  ORIGIN - y / PX_SIZE + ego_cx * s * st - ego_cy * s * ct],
        ])
        return M

    @staticmethod
    def _inverse_affine(x, y, theta, ego_cx, ego_cy, ego_px_size):
        """2×3 affine: global pixel → ego pixel."""
        ct = np.cos(theta)
        st = np.sin(theta)
        r = PX_SIZE / ego_px_size  # 4.0

        t_gx = ORIGIN + x / PX_SIZE
        t_gy = ORIGIN - y / PX_SIZE

        M = np.float64([
            [ r * ct, -r * st,  ego_cx - r * ct * t_gx + r * st * t_gy],
            [ r * st,  r * ct,  ego_cy - r * st * t_gx - r * ct * t_gy],
        ])
        return M
