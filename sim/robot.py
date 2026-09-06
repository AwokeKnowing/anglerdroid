"""Differential-drive robot with SafetyGuard-equivalent collision avoidance.

CRITICAL: hard stop when fwd_scale=0 — never allow recover/commit/escape 
to drive through obstacles. Escape yaw only when laterals allow; 
reverse only when bwd_scale>0.
"""

import math
import numpy as np
from sim.dynamics import DiffDriveDynamics


FRAME_W = 320
FRAME_H = 240
RCX = 81
RCY = 119
EGO_PX_SIZE = 0.01

ROBOT_W = 30
ROBOT_H = 42
ROBOT_CX_OFF = -78

FOOT_X0 = 56
FOOT_Y0 = 97
FOOT_X1 = 101
FOOT_Y1 = 142
FOOT_PAD_FWD = 5
FOOT_PAD_BWD = 10
FOOT_PAD_LAT = 2

MIN_CLEARANCE_PX = 1
BRAKE_START_PX = 30
OBS_THRESH = 100
# Planner soft buffer (prefer). Hard SafetyGuard still uses bare FOOT scans.
SOFT_INFLATE_PX = 10          # ~10 cm prefer-clear around obstacles
HARD_PAD_PX = 2              # tiny pad for hard contact scan only
SQUEEZE_FWD_SOFT_MAX = 0.55  # below this soft clearance → squeeze mode

# Cul-de-sac tip disk offsets (px), radius ~0.26 m
_CUL_RAD = max(3, int(0.26 / EGO_PX_SIZE))
_CUL_DISK = [
    (dx, dy)
    for dy in range(-_CUL_RAD, _CUL_RAD + 1)
    for dx in range(-_CUL_RAD, _CUL_RAD + 1)
    if dx * dx + dy * dy <= _CUL_RAD * _CUL_RAD
]

_HALF_DIAG = 26
LAT_X0 = max(0, RCX - _HALF_DIAG)
LAT_X1 = min(FRAME_W, RCX + _HALF_DIAG)
LAT_HARD_PX = 2
LAT_SAFE_PX = 9

MASK_X0 = FOOT_X0
MASK_Y0 = FOOT_Y0
MASK_X1 = FOOT_X1
MASK_Y1 = FOOT_Y1
BWD_SCAN_X0 = FOOT_X0


# Precomputed ego-frame offsets (meters) for vectorized world-to-ego warp.
_EY, _EX = np.mgrid[0:FRAME_H, 0:FRAME_W]
_DX_EGO = (_EX - RCX).astype(np.float64) * EGO_PX_SIZE
_DY_EGO = (_EY - RCY).astype(np.float64) * EGO_PX_SIZE


def _clearance_scale(clear_px):
    """S-curve 0→1 based purely on distance."""
    if clear_px <= MIN_CLEARANCE_PX:
        return 0.0
    if clear_px >= BRAKE_START_PX:
        return 1.0
    t = (clear_px - MIN_CLEARANCE_PX) / float(BRAKE_START_PX - MIN_CLEARANCE_PX)
    return t * t * (3.0 - 2.0 * t)


def soft_inflate(obs, rad=SOFT_INFLATE_PX):
    """Binary dilation for planner soft-buffer cost (not used by hard SafetyGuard)."""
    if rad <= 0:
        return (obs >= OBS_THRESH)
    try:
        import cv2
        k = 2 * int(rad) + 1
        kernel = np.ones((k, k), np.uint8)
        return cv2.dilate((obs >= OBS_THRESH).astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        # Cheap square max-pool fallback
        from numpy.lib.stride_tricks import sliding_window_view
        m = (obs >= OBS_THRESH).astype(np.uint8)
        pad = int(rad)
        mp = np.pad(m, pad, mode='constant')
        win = sliding_window_view(mp, (2 * pad + 1, 2 * pad + 1))
        return win.max(axis=(-1, -2)).astype(bool)


def soft_forward_clear_px(obs_soft, near=12, mid=55):
    """Free fraction in forward near/mid strips (ego +x). Higher = more buffer."""
    if obs_soft is None:
        return 1.0, 1.0
    h, w = obs_soft.shape
    y0, y1 = max(0, RCY - 12), min(h, RCY + 12)
    n0, n1 = FOOT_X1, min(w, FOOT_X1 + near)
    m0, m1 = min(w, FOOT_X1 + near), min(w, FOOT_X1 + mid)
    def free(a0, a1):
        patch = obs_soft[y0:y1, a0:a1]
        if patch.size == 0:
            return 0.0
        return 1.0 - float(patch.mean())
    return free(n0, n1), free(m0, m1)


def soft_heading_score(obs_soft, yaw_off_rad, horizon_m=0.9):
    """Score a candidate heading offset: fraction soft-free along a ray."""
    if obs_soft is None:
        return 1.0
    h, w = obs_soft.shape
    # Ray from FOOT_X1 along yaw_off relative to +x
    n = max(8, int(horizon_m / EGO_PX_SIZE))
    hits = 0
    free = 0
    for i in range(1, n + 1):
        x = FOOT_X1 + i * math.cos(yaw_off_rad)
        y = RCY + i * math.sin(yaw_off_rad)
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            break
        hits += 1
        if not obs_soft[iy, ix]:
            free += 1
    return (free / hits) if hits else 0.0

def cul_de_sac_escape(obs_soft, yaw_off_rad=0.0, approach_m=0.95, tip_radius_m=0.26):
    """Lookahead trap score in [0, 1]: 1 = open ahead, 0 = dead-end pocket.

    Walk along yaw_off up to approach_m but stop just before first soft hit.
    Tip uses a precomputed disk + short stub + sparse lateral fans.
    """
    if obs_soft is None:
        return 1.0
    h, w = obs_soft.shape
    cos_y = math.cos(yaw_off_rad)
    sin_y = math.sin(yaw_off_rad)
    n_max = max(8, int(approach_m / EGO_PX_SIZE))
    tip_i = n_max
    hit_wall = False
    for i in range(1, n_max + 1):
        ix = int(round(FOOT_X1 + i * cos_y))
        iy = int(round(RCY + i * sin_y))
        if not (0 <= ix < w and 0 <= iy < h) or obs_soft[iy, ix]:
            tip_i = max(1, i - 2)
            hit_wall = True
            break
    tip_x = FOOT_X1 + tip_i * cos_y
    tip_y = RCY + tip_i * sin_y

    free = hits = 0
    for dx, dy in _CUL_DISK:
        ix = int(tip_x + dx)
        iy = int(tip_y + dy)
        if 0 <= ix < w and 0 <= iy < h:
            hits += 1
            if not obs_soft[iy, ix]:
                free += 1
    disk_s = (free / hits) if hits else 0.0

    stub_free = stub_hits = 0
    for i in range(1, 21):
        ix = int(round(tip_x + i * cos_y))
        iy = int(round(tip_y + i * sin_y))
        if not (0 <= ix < w and 0 <= iy < h):
            break
        stub_hits += 1
        if not obs_soft[iy, ix]:
            stub_free += 1
    stub = (stub_free / stub_hits) if stub_hits else 0.0

    fan_free = fan_hits = 0
    for dlat in (-0.96, 0.96, -1.57, 1.57):
        cy = math.cos(yaw_off_rad + dlat)
        sy = math.sin(yaw_off_rad + dlat)
        for i in (6, 12, 18):
            ix = int(round(tip_x + i * cy))
            iy = int(round(tip_y + i * sy))
            if not (0 <= ix < w and 0 <= iy < h):
                break
            fan_hits += 1
            if not obs_soft[iy, ix]:
                fan_free += 1
    fan = (fan_free / fan_hits) if fan_hits else 0.0
    score = 0.25 * disk_s + 0.45 * stub + 0.30 * fan
    if not hit_wall:
        score = max(score, 0.85 * disk_s + 0.15)
    return float(max(0.0, min(1.0, score)))


def score_heading_with_lookahead(obs_soft, yaw_off_rad, horizon_m=0.9):
    """Blend soft ray free fraction with cul-de-sac escape (prefer open exits)."""
    ray = soft_heading_score(obs_soft, yaw_off_rad, horizon_m=horizon_m)
    esc = cul_de_sac_escape(obs_soft, yaw_off_rad, approach_m=min(1.2, horizon_m + 0.15))
    # Escape-dominant: clear ray into a pocket still loses
    return float(0.30 * ray + 0.70 * esc)





class SafetyGuard:
    """Directional collision avoidance matching src/safety.py."""
    
    def __init__(self):
        self.fwd_scale = 1.0
        self.bwd_scale = 1.0
        self.ang_scale = 1.0
        self.fwd_clear = 999
        self.bwd_clear = 999
        self.lat_clear = 999
    
    def update(self, obs_map):
        """Compute directional scales from obstacle map."""
        h_map, w_map = obs_map.shape
        y0, y1 = MASK_Y0, min(h_map, MASK_Y1)
        
        if MASK_X1 < w_map:
            strip = obs_map[y0:y1, MASK_X1:]
            col_hit = np.any(strip >= OBS_THRESH, axis=0)
            idxs = np.flatnonzero(col_hit)
            fwd_clear = int(idxs[0]) if len(idxs) > 0 else strip.shape[1]
        else:
            fwd_clear = 0
        
        if BWD_SCAN_X0 > 0:
            strip = obs_map[y0:y1, :BWD_SCAN_X0][:, ::-1]
            col_hit = np.any(strip >= OBS_THRESH, axis=0)
            idxs = np.flatnonzero(col_hit)
            bwd_clear = int(idxs[0]) if len(idxs) > 0 else strip.shape[1]
        else:
            bwd_clear = 0
        
        self.fwd_scale = _clearance_scale(fwd_clear)
        self.bwd_scale = _clearance_scale(bwd_clear)
        self.fwd_clear = fwd_clear
        self.bwd_clear = bwd_clear
        
        lx0, lx1 = LAT_X0, LAT_X1
        
        if MASK_Y0 > 0:
            strip = obs_map[:MASK_Y0, lx0:lx1][::-1, :]
            row_hit = np.any(strip >= OBS_THRESH, axis=1)
            idxs = np.flatnonzero(row_hit)
            up_clear = int(idxs[0]) if len(idxs) > 0 else strip.shape[0]
        else:
            up_clear = 0
        
        if MASK_Y1 < h_map:
            strip = obs_map[MASK_Y1:, lx0:lx1]
            row_hit = np.any(strip >= OBS_THRESH, axis=1)
            idxs = np.flatnonzero(row_hit)
            down_clear = int(idxs[0]) if len(idxs) > 0 else strip.shape[0]
        else:
            down_clear = 0
        
        min_lat = min(up_clear, down_clear)
        self.lat_clear = min_lat
        
        if min_lat <= LAT_HARD_PX:
            self.ang_scale = 0.0
        elif min_lat < LAT_SAFE_PX:
            self.ang_scale = (min_lat - LAT_HARD_PX) / float(LAT_SAFE_PX - LAT_HARD_PX)
        else:
            self.ang_scale = 1.0
        # Insect spin-out: if nose pinned but not laterally crushed, keep yaw.
        if self.fwd_scale < 0.15 and min_lat > LAT_HARD_PX:
            self.ang_scale = max(self.ang_scale, 0.35)
        if self.bwd_scale < 0.15 and min_lat > LAT_HARD_PX:
            self.ang_scale = max(self.ang_scale, 0.35)


class Robot:
    """Differential-drive robot with pose and collision detection."""
    
    def __init__(self, x_m, y_m, theta_rad, fidelity=False):
        self.x = x_m
        self.y = y_m
        self.theta = theta_rad
        self.v = 0.0
        self.w = 0.0
        self.safety = SafetyGuard()
        self.collision_count = 0
        self.recover_count = 0
        self.ego_obs = None
        self.ego_height = None
        # fidelity=True enables ~150ms cmd latency (gap D2)
        from sim.dynamics import LATENCY_S
        self.dyn = DiffDriveDynamics(latency_s=(LATENCY_S if fidelity else 0.0))
        self.fidelity = bool(fidelity)
        self.ego_soft = None
        self.soft_near = 1.0
        self.soft_mid = 1.0
        self.squeeze = False
    
    def update_ego_maps(self, world_obs, world_height):
        """Transform world maps to robot-centric ego frame (vectorized).

        Robot faces RIGHT in ego frame (x-axis points forward).
        Uses precomputed ego grid + NumPy indexing (~40x vs nested Python).
        """
        h, w = world_obs.shape
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        dx_world = _DX_EGO * cos_t - _DY_EGO * sin_t
        dy_world = _DX_EGO * sin_t + _DY_EGO * cos_t
        wx_px = np.rint((self.x + dx_world) / EGO_PX_SIZE).astype(np.int32)
        wy_px = np.rint((self.y + dy_world) / EGO_PX_SIZE).astype(np.int32)
        valid = (wx_px >= 0) & (wx_px < w) & (wy_px >= 0) & (wy_px < h)

        ego_obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        ego_height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        ego_obs[valid] = world_obs[wy_px[valid], wx_px[valid]]
        ego_height[valid] = world_height[wy_px[valid], wx_px[valid]]

        self.ego_obs = ego_obs
        self.ego_height = ego_height
        self.safety.update(ego_obs)
        # Gap: side-swipe into FOOT while fwd scan still clear — clamp if body ring dirty
        self._apply_body_ring_safety(ego_obs)
        self.ego_soft = soft_inflate(ego_obs)
        sn, sm = soft_forward_clear_px(self.ego_soft)
        self.soft_near = sn
        self.soft_mid = sm
        # Squeeze: hard still allows some forward, soft buffer already gone
        self.squeeze = (sm < SQUEEZE_FWD_SOFT_MAX) and (self.safety.fwd_scale > 0.05)
        if self.fidelity:
            self._apply_perception_noise()

    


    def _apply_perception_noise(self):
        """Fidelity: salt-pepper on ego obs + mild height noise + tiny pose drift.

        Mimics RealSense dropouts / VO jitter without poisoning SafetyGuard
        with impossible geometry. Noise is applied AFTER safety.update so hard
        stops still see clean geometry for the hard layer; soft map sees noise.
        Actually: we already called safety.update on clean map. Re-run soft only
        and optionally flip a few distant cells so MPPI costs get messier.
        """
        if self.ego_obs is None:
            return
        obs = self.ego_obs
        # Salt-pepper far from FOOT (don't invent contact under the body)
        rng = getattr(self, '_fid_rng', None)
        if rng is None:
            import numpy as np
            self._fid_rng = np.random.default_rng(0)
            rng = self._fid_rng
        import numpy as np
        mask = np.ones_like(obs, dtype=bool)
        pad = 8
        mask[max(0, FOOT_Y0 - pad):min(FRAME_H, FOOT_Y1 + pad),
             max(0, FOOT_X0 - pad):min(FRAME_W, FOOT_X1 + pad)] = False
        flip = (rng.random(obs.shape) < 0.008) & mask
        # Flip clear↔occupied sparsely
        flipped = obs.copy()
        flipped[flip] = np.where(flipped[flip] >= OBS_THRESH, 0, 255).astype(np.uint8)
        self.ego_obs = flipped
        if self.ego_height is not None:
            h = self.ego_height.astype(np.int16)
            h = np.clip(h + rng.integers(-3, 4, size=h.shape), 0, 255).astype(np.uint8)
            self.ego_height = h
        # Rebuild soft inflate from noisy obs (hard safety stays from clean)
        self.ego_soft = soft_inflate(self.ego_obs)
        sn, sm = soft_forward_clear_px(self.ego_soft)
        self.soft_near = sn
        self.soft_mid = sm
        # Tiny VO drift (gap D1) — apply occasionally
        if rng.random() < 0.15:
            self.x += float(rng.normal(0.0, 0.004))
            self.y += float(rng.normal(0.0, 0.004))
            self.theta += float(rng.normal(0.0, 0.01))


    def _apply_body_ring_safety(self, ego_obs):
        """If obstacles kiss the footprint from the side, cut motion (P0 geometry)."""
        if ego_obs is None:
            return
        foot = ego_obs[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1]
        if foot.size and foot.max() >= OBS_THRESH:
            self.safety.fwd_scale = 0.0
            self.safety.bwd_scale = min(self.safety.bwd_scale, 0.2)
            self.safety.ang_scale = min(self.safety.ang_scale, 0.3)
            return
        pad = 6
        y0, y1 = max(0, FOOT_Y0 - pad), min(ego_obs.shape[0], FOOT_Y1 + pad)
        x0, x1 = max(0, FOOT_X0 - pad), min(ego_obs.shape[1], FOOT_X1 + pad)
        ring = ego_obs[y0:y1, x0:x1].copy()
        # Clear interior FOOT so we only see the ring
        iy0, iy1 = FOOT_Y0 - y0, FOOT_Y1 - y0
        ix0, ix1 = FOOT_X0 - x0, FOOT_X1 - x0
        ring[iy0:iy1, ix0:ix1] = 0
        if ring.size and ring.max() >= OBS_THRESH:
            self.safety.fwd_scale = min(self.safety.fwd_scale, 0.35)
            self.safety.ang_scale = min(self.safety.ang_scale, 0.6)

    def check_collision(self):
        """Check if robot footprint overlaps any obstacle."""
        if self.ego_obs is None:
            return False
        
        footprint = self.ego_obs[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1]
        return np.any(footprint >= OBS_THRESH)
    
    def step(self, v_cmd, w_cmd, dt, apply_safety=True):
        """Step robot forward with commanded velocities.

        CRITICAL: Apply safety clipping BEFORE motion (default).
        Hard stop when fwd_scale=0 for forward motion.
        Hard stop when bwd_scale=0 for backward motion.
        Hard stop when ang_scale=0 for angular motion.

        apply_safety=False is ONLY for crash-hypothesis contrast runs
        (UnsafeCommitPolicy mimicking live recover override). Never use
        on HouseBotLite / production-path sims.
        """
        if apply_safety:
            v_safe = v_cmd
            w_safe = w_cmd

            if v_cmd > 0 and self.safety.fwd_scale <= 0:
                v_safe = 0.0
            elif v_cmd > 0:
                v_safe = v_cmd * self.safety.fwd_scale

            if v_cmd < 0 and self.safety.bwd_scale <= 0:
                v_safe = 0.0
            elif v_cmd < 0:
                v_safe = v_cmd * self.safety.bwd_scale

            if abs(w_cmd) > 0 and self.safety.ang_scale <= 0:
                w_safe = 0.0
            else:
                w_safe = w_cmd * self.safety.ang_scale
        else:
            v_safe = v_cmd
            w_safe = w_cmd

        # Dynamics: latency + accel + wheelbase-aware caps (gaps D1–D4)
        v_int, w_int = self.dyn.apply(v_safe, w_safe, dt)
        self.v = v_int
        self.w = w_int

        self.x += v_int * math.cos(self.theta) * dt
        self.y += v_int * math.sin(self.theta) * dt
        self.theta += w_int * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
