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

import time
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

SPRITE_SZ = 48
_SC = SPRITE_SZ // 2  # 24 — sprite centre


class GlobalMap:
    def __init__(self):
        self._map = np.full((MAP_H, MAP_W), UNKNOWN_VAL, dtype=np.uint8)
        self._ring = [None] * CONSENSUS_N  # (r0,r1,c0,c1,signed_roi) per slot
        self._ring_idx = 0
        self._count = 0
        self._spr_rgba, self._spr_lab = self._make_sprite()
        self._render_times = []
        self._update_times = []
        self._out = np.empty((MAP_H, MAP_W, 3), dtype=np.uint8)

    def update(self, obs_ego, known_ego, x, y, theta,
               ego_cx, ego_cy, ego_px_size=0.01):
        """Feed one frame of ego-space observations and robot pose.

        ROI-optimised: only warp/compare/update the bounding box of the
        observation footprint in global space (~200×200 vs 960×720).
        """
        t0 = time.monotonic()
        h, w = obs_ego.shape[:2]

        ego_obs = np.zeros((h, w), dtype=np.uint8)
        free = (known_ego > 0) & (obs_ego == 0)
        ego_obs[free] = 1
        ego_obs[obs_ego > 0] = 2

        M_fwd = self._forward_affine(x, y, theta, ego_cx, ego_cy, ego_px_size)
        t1 = time.monotonic()

        # Compute ROI: transform ego corners → global pixel bounds
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(1, 4, 2)
        gc = cv2.transform(corners, M_fwd).reshape(-1, 2)
        r0 = max(0, int(np.floor(gc[:, 1].min())))
        r1 = min(MAP_H, int(np.ceil(gc[:, 1].max())) + 1)
        c0 = max(0, int(np.floor(gc[:, 0].min())))
        c1 = min(MAP_W, int(np.ceil(gc[:, 0].max())) + 1)
        if r1 <= r0 or c1 <= c0:
            return

        # Warp only into the ROI-sized output (shift affine origin)
        M_roi = M_fwd.copy()
        M_roi[0, 2] -= c0
        M_roi[1, 2] -= r0
        roi_w, roi_h = c1 - c0, r1 - r0
        obs_roi = cv2.warpAffine(
            ego_obs, M_roi, (roi_w, roi_h),
            flags=cv2.INTER_NEAREST, borderValue=0)
        t2 = time.monotonic()

        # Encode as signed: free=+1, obs=-1, unobserved=0
        signed = np.zeros((roi_h, roi_w), dtype=np.int8)
        signed[obs_roi == 1] = 1
        signed[obs_roi == 2] = -1

        self._ring[self._ring_idx] = (r0, r1, c0, c1, signed)
        self._ring_idx = (self._ring_idx + 1) % CONSENSUS_N
        self._count += 1

        if self._count < CONSENSUS_N:
            return
        t3 = time.monotonic()

        # Consensus within intersection of all 3 ROIs
        rois = self._ring
        ir0 = max(r[0] for r in rois)
        ir1 = min(r[1] for r in rois)
        ic0 = max(r[2] for r in rois)
        ic1 = min(r[3] for r in rois)
        if ir1 <= ir0 or ic1 <= ic0:
            return

        # Extract overlapping sub-regions from each ring entry
        slices = []
        for br0, br1, bc0, bc1, s in rois:
            slices.append(s[ir0 - br0:ir1 - br0, ic0 - bc0:ic1 - bc0])
        s0, s1, s2 = slices

        all_free = (s0 == 1) & (s1 == 1) & (s2 == 1)
        all_obs = (s0 == -1) & (s1 == -1) & (s2 == -1)

        roi_map = self._map[ir0:ir1, ic0:ic1]
        m = roi_map.astype(np.int16)
        m[all_free] = np.minimum(m[all_free] + EVIDENCE_UP, 255)
        m[all_obs] = np.maximum(m[all_obs] - EVIDENCE_DOWN, 0)
        roi_map[:] = m.astype(np.uint8)
        t4 = time.monotonic()

        self._update_times.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        if len(self._update_times) % 100 == 0:
            avg = np.mean(self._update_times[-100:], axis=0) * 1000
            print("gmap update: prep=%.1fms warp=%.1fms ring=%.1fms "
                  "consensus=%.1fms total=%.1fms  roi=%dx%d" %
                  (*avg, sum(avg), roi_h, roi_w))

    def project_to_ego(self, x, y, theta,
                       ego_cx, ego_cy, ego_px_size, ego_h, ego_w):
        """Warp global map back to ego-centred coordinates for safety."""
        M_inv = self._inverse_affine(x, y, theta, ego_cx, ego_cy, ego_px_size)
        return cv2.warpAffine(
            self._map, M_inv, (ego_w, ego_h),
            flags=cv2.INTER_NEAREST, borderValue=UNKNOWN_VAL)

    def render(self, x, y, theta, trail_xy=None,
               fwd_scale=1.0, bwd_scale=1.0, ang_scale=1.0):
        """960×720 split view: left=center crop, right=4× ego zoom forward-up."""
        HALF = MAP_W // 2
        EGO_SCALE = 4.0
        EGO_RX, EGO_RY = HALF // 2, int(MAP_H * 0.80)

        rc = int(ORIGIN_X + x / PX_SIZE)
        rr = int(ORIGIN_Y - y / PX_SIZE)
        out = self._out

        # --- Left panel: colorize crop, draw trail + robot ---
        cx0 = max(0, min(rc - HALF // 2, MAP_W - HALF))
        left_raw = self._map[:, cx0:cx0 + HALF]
        left_g = np.full_like(left_raw, UNK_DISPLAY)
        left_g[left_raw > FREE_THRESH] = FREE_DISPLAY
        left_g[left_raw < OBS_THRESH] = OBS_DISPLAY
        left = cv2.cvtColor(left_g, cv2.COLOR_GRAY2BGR)

        if trail_xy is not None and len(trail_xy) >= 2:
            tc = (ORIGIN_X + trail_xy[:, 0] / PX_SIZE - cx0).astype(np.int32)
            tr = (ORIGIN_Y - trail_xy[:, 1] / PX_SIZE).astype(np.int32)
            keep = (tc >= 0) & (tc < HALF) & (tr >= 0) & (tr < MAP_H)
            pts = np.column_stack([tc[keep], tr[keep]])
            if len(pts) >= 2:
                cv2.polylines(left, [pts.reshape(-1, 1, 2)], False,
                              (80, 140, 255), 1, cv2.LINE_AA)
        self._draw_robot(left, rc - cx0, rr, theta,
                         fwd_scale, bwd_scale, ang_scale)
        out[:, :HALF] = left

        # --- Right panel: warp single-chan, colorize, draw scaled robot ---
        ct, st = np.cos(theta), np.sin(theta)
        inv_s = 1.0 / EGO_SCALE
        M_ego = np.float64([
            [ st * inv_s, -ct * inv_s,
              rc - st * EGO_RX * inv_s + ct * EGO_RY * inv_s],
            [ ct * inv_s,  st * inv_s,
              rr - ct * EGO_RX * inv_s - st * EGO_RY * inv_s],
        ])
        ego_raw = cv2.warpAffine(
            self._map, M_ego, (HALF, MAP_H),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderValue=UNKNOWN_VAL)
        ego_g = np.full_like(ego_raw, UNK_DISPLAY)
        ego_g[ego_raw > FREE_THRESH] = FREE_DISPLAY
        ego_g[ego_raw < OBS_THRESH] = OBS_DISPLAY
        right = cv2.cvtColor(ego_g, cv2.COLOR_GRAY2BGR)
        self._draw_robot(right, EGO_RX, EGO_RY, np.pi * 0.5,
                         fwd_scale, bwd_scale, ang_scale,
                         zoom=EGO_SCALE)
        out[:, HALF:] = right

        return out

    # ── sprite-based robot rendering ──────────────────────────────

    @staticmethod
    def _make_sprite():
        """Pre-render robot sprite facing right. Returns (rgba, labels).

        Labels: 0=background, 1=body, 2=track/caster, 3=arrow.
        Alpha: body/tracks=153 (60%), arrow=255 (100%).
        """
        S, C = SPRITE_SZ, _SC
        rgba = np.zeros((S, S, 4), dtype=np.uint8)
        lab = np.zeros((S, S), dtype=np.uint8)

        def _rect(x0, y0, x1, y1, a, lbl):
            cv2.rectangle(rgba, (x0, y0), (x1, y1), (0, 0, 0, a), -1)
            lab[y0:y1 + 1, x0:x1 + 1] = lbl

        _rect(C - 9, C - 7, C + 7, C + 7, 153, 1)       # body 17×15
        _rect(C - 4, C - 10, C + 4, C - 8, 153, 2)       # left track 9×3
        _rect(C - 4, C + 8, C + 4, C + 10, 153, 2)       # right track 9×3
        _rect(C - 11, C - 1, C - 9, C, 153, 2)            # caster 3×2

        cv2.arrowedLine(rgba, (C - 5, C), (C + 6, C),
                        (255, 180, 0, 255), 1, tipLength=0.35)
        lab[rgba[:, :, 3] == 255] = 3

        return rgba, lab

    def _draw_robot(self, disp, rc, rr, theta,
                    fwd_scale, bwd_scale, ang_scale, zoom=1.0):
        """Tint, rotate, and alpha-composite the pre-rendered robot sprite."""
        S, C = SPRITE_SZ, _SC
        h, w = disp.shape[:2]
        spr = self._spr_rgba.copy()

        sf = min(fwd_scale, bwd_scale, ang_scale)
        body_rgb = [int(255 * (1 - sf)),
                    int(160 * sf + 60 * (1 - sf)),
                    int(255 * sf)]
        td = 1.0 - min(fwd_scale, bwd_scale)
        trk_rgb = [int(180 * td),
                   int(60 + 100 * (1 - td)),
                   int(180 * (1 - td))]

        spr[self._spr_lab == 1, :3] = body_rgb
        spr[self._spr_lab == 2, :3] = trk_rgb

        out_s = int(S * zoom)
        out_c = out_s // 2
        M = cv2.getRotationMatrix2D((float(C), float(C)),
                                    np.degrees(theta), zoom)
        rot = cv2.warpAffine(spr, M, (out_s, out_s),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

        x0, y0 = rc - out_c, rr - out_c
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        sw = min(out_s, w - x0) - sx0
        sh = min(out_s, h - y0) - sy0
        if sw <= 0 or sh <= 0:
            return

        patch = rot[sy0:sy0 + sh, sx0:sx0 + sw]
        alpha = patch[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
        rgb = patch[:, :, :3].astype(np.float32)
        region = disp[dy0:dy0 + sh, dx0:dx0 + sw]
        np.copyto(region,
                  (region.astype(np.float32) * (1.0 - alpha) +
                   rgb * alpha).astype(np.uint8))

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
