"""Differential-drive robot with SafetyGuard-equivalent collision avoidance.

CRITICAL: hard stop when fwd_scale=0 — never allow recover/commit/escape 
to drive through obstacles. Escape yaw only when laterals allow; 
reverse only when bwd_scale>0.
"""

import math
import numpy as np


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


class Robot:
    """Differential-drive robot with pose and collision detection."""
    
    def __init__(self, x_m, y_m, theta_rad):
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
    
    def check_collision(self):
        """Check if robot footprint overlaps any obstacle."""
        if self.ego_obs is None:
            return False
        
        footprint = self.ego_obs[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1]
        return np.any(footprint >= OBS_THRESH)
    
    def step(self, v_cmd, w_cmd, dt):
        """Step robot forward with commanded velocities.
        
        CRITICAL: Apply safety clipping BEFORE motion.
        Hard stop when fwd_scale=0 for forward motion.
        Hard stop when bwd_scale=0 for backward motion.
        Hard stop when ang_scale=0 for angular motion.
        """
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
        
        self.v = v_safe
        self.w = w_safe
        
        self.x += v_safe * math.cos(self.theta) * dt
        self.y += v_safe * math.sin(self.theta) * dt
        self.theta += w_safe * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
