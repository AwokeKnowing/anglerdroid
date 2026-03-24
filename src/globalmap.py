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

# One consensus (3 agreeing frames at 30fps = 100ms) should be enough
# to flip a cell.  128 takes unknown→free or unknown→obstacle in a
# single consensus; free↔obstacle requires 2 (200ms hysteresis).
EVIDENCE_UP = 128
EVIDENCE_DOWN = 128

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

    def render(self, x, y, theta, trail_xy=None,
               fwd_scale=1.0, bwd_scale=1.0, ang_scale=1.0):
        """MAP_W × MAP_H BGR image with robot footprint and optional trajectory.

        trail_xy: (N, 2) float64 world [x, y] array, oldest→newest.
        fwd/bwd/ang_scale: 0–1 safety throttle (1=safe, 0=blocked).
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
        if 10 <= rc < MAP_W - 10 and 10 <= rr < MAP_H - 10:
            self._draw_robot(disp, rc, rr, theta,
                             fwd_scale, bwd_scale, ang_scale)
        return disp

    @staticmethod
    def _draw_robot(disp, rc, rr, theta, fwd_scale, bwd_scale, ang_scale):
        """Draw multi-rect robot at 30% opacity with safety gradient.

        Colors are written in RGB order (image is JPEG-streamed to browser).
        """
        OPACITY = 0.3
        ct, st = np.cos(theta), np.sin(theta)

        def _rot(fwd, left):
            """Robot-local (forward, left) → pixel (dcol, drow)."""
            return (fwd * ct - left * st,
                    -(fwd * st + left * ct))

        def _box(cx, cy, hw, hh, color):
            """Filled rotated box at robot-local (cx=fwd, cy=left), half-sizes."""
            corners = []
            for sx, sy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
                dc, dr = _rot(cx + sx, cy + sy)
                corners.append([int(rc + dc), int(rr + dr)])
            pts = np.array(corners, dtype=np.int32)
            overlay = disp.copy()
            cv2.fillConvexPoly(overlay, pts, color, cv2.LINE_AA)
            cv2.addWeighted(overlay, OPACITY, disp, 1.0 - OPACITY, 0, dst=disp)

        # RGB colors: blue (safe) → red (danger)
        danger = min(fwd_scale, bwd_scale, ang_scale)
        body_color = (int(255 * (1.0 - danger)),
                      int(160 * danger + 60 * (1.0 - danger)),
                      int(255 * danger))

        td = 1.0 - min(fwd_scale, bwd_scale)
        track_color = (int(180 * td),
                       int(60 + 100 * (1.0 - td)),
                       int(180 * (1.0 - td)))

        # Body: ~33×30 cm → 16.5×15 px at 2cm/px, offset 1.5cm rear
        _box(-0.75, 0, 8.25, 7.5, body_color)
        # Tracks: ~17×6 cm at 2cm/px, offset ±18cm laterally
        _box(0.75, 9, 4.25, 1.5, track_color)
        _box(0.75, -9, 4.25, 1.5, track_color)
        # Caster: ~6×3 cm at 2cm/px, 20cm rear
        _box(-10, 0, 1.5, 0.75, track_color)

        # Direction arrow (1px, fits inside body)
        adx, adr = _rot(6, 0)
        bdx, bdr = _rot(-5, 0)
        cv2.arrowedLine(disp,
                        (int(rc + bdx), int(rr + bdr)),
                        (int(rc + adx), int(rr + adr)),
                        (255, 180, 0), 1, tipLength=0.35)

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
