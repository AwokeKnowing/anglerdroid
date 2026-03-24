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
        self._ring = [np.zeros((MAP_H, MAP_W), dtype=np.int8)
                      for _ in range(CONSENSUS_N)]
        self._ring_idx = 0
        self._count = 0
        self._spr_rgba, self._spr_lab = self._make_sprite()
        self._render_times = []

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
        """960×720 split view: left=center crop, right=4× ego zoom forward-up."""
        t0 = time.monotonic()
        HALF = MAP_W // 2
        EGO_SCALE = 4.0
        EGO_RX, EGO_RY = HALF // 2, int(MAP_H * 0.80)

        # coloured base map
        disp = np.full((MAP_H, MAP_W), UNK_DISPLAY, dtype=np.uint8)
        disp[self._map > FREE_THRESH] = FREE_DISPLAY
        disp[self._map < OBS_THRESH] = OBS_DISPLAY
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        t1 = time.monotonic()

        rc = int(ORIGIN_X + x / PX_SIZE)
        rr = int(ORIGIN_Y - y / PX_SIZE)

        # trail + robot on full map
        if trail_xy is not None and len(trail_xy) >= 2:
            tc = (ORIGIN_X + trail_xy[:, 0] / PX_SIZE).astype(np.int32)
            tr = (ORIGIN_Y - trail_xy[:, 1] / PX_SIZE).astype(np.int32)
            keep = (tc >= 0) & (tc < MAP_W) & (tr >= 0) & (tr < MAP_H)
            pts = np.column_stack([tc[keep], tr[keep]])
            if len(pts) >= 2:
                cv2.polylines(disp, [pts.reshape(-1, 1, 2)], False,
                              (80, 140, 255), 1, cv2.LINE_AA)
        self._draw_robot(disp, rc, rr, theta,
                         fwd_scale, bwd_scale, ang_scale)
        t2 = time.monotonic()

        # left panel: center crop
        cx0 = max(0, min(rc - HALF // 2, MAP_W - HALF))
        left = disp[:, cx0:cx0 + HALF]
        t3 = time.monotonic()

        # right panel: 4× ego zoom from rendered map
        ct, st = np.cos(theta), np.sin(theta)
        inv_s = 1.0 / EGO_SCALE
        M_ego = np.float64([
            [ st * inv_s, -ct * inv_s,
              rc - st * EGO_RX * inv_s + ct * EGO_RY * inv_s],
            [ ct * inv_s,  st * inv_s,
              rr - ct * EGO_RX * inv_s - st * EGO_RY * inv_s],
        ])
        right = cv2.warpAffine(
            disp, M_ego, (HALF, MAP_H),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderValue=(UNK_DISPLAY, UNK_DISPLAY, UNK_DISPLAY))
        t4 = time.monotonic()

        # assemble
        result = np.empty((MAP_H, MAP_W, 3), dtype=np.uint8)
        result[:, :HALF] = left
        result[:, HALF:] = right
        t5 = time.monotonic()

        self._render_times.append((t1-t0, t2-t1, t3-t2, t4-t3, t5-t4))
        if len(self._render_times) % 100 == 0:
            avg = np.mean(self._render_times[-100:], axis=0) * 1000
            print("globalmap render: colorize=%.1fms trail+robot=%.1fms "
                  "crop=%.1fms warp=%.1fms assemble=%.1fms total=%.1fms"
                  % (*avg, sum(avg)))

        return result

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
                    fwd_scale, bwd_scale, ang_scale):
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

        M = cv2.getRotationMatrix2D((float(C), float(C)),
                                    np.degrees(theta), 1.0)
        rot = cv2.warpAffine(spr, M, (S, S),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

        x0, y0 = rc - C, rr - C
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        sw = min(S, w - x0) - sx0
        sh = min(S, h - y0) - sy0
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
