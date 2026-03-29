"""safety.py – Directional collision avoidance + trajectory prediction.

Scans FORWARD and BACKWARD from the robot's physical edges (not centre)
for obstacles, computes max safe speed in each direction independently.
Moving AWAY from an obstacle is always allowed.  Trajectory prediction
for visualisation only.
"""

import math
from collections import deque
import numpy as np
import cv2

# ── Robot physics ──
VEL_RAMP_RATE = 3.0  # turns/s²
LATENCY_S = 0.15
MIN_CLEARANCE_PX = 1
BRAKE_START_PX = 30
OBS_THRESH = 100

# ── Costmap geometry ──
from robot_config import (FRAME_W, EGO_PX_SIZE, WHEEL_RADIUS_M,
                          ROBOT_W, ROBOT_H, RCX, RCY,
                          FOOT_X0, FOOT_X1, FOOT_Y0, FOOT_Y1)

PX_M = EGO_PX_SIZE
DECEL_MPS2 = VEL_RAMP_RATE * 2.0 * math.pi * WHEEL_RADIUS_M  # ≈1.61 m/s²

MASK_X0 = FOOT_X0
MASK_Y0 = FOOT_Y0
MASK_X1 = FOOT_X1
MASK_Y1 = FOOT_Y1
BWD_SCAN_X0 = FOOT_X0

# Lateral scan covers the robot's full diagonal extent so spin-corner
# collisions are caught.  half-diagonal ≈ sqrt(15²+21²) ≈ 26 px.
_HALF_DIAG = 26
LAT_X0 = max(0, RCX - _HALF_DIAG)
LAT_X1 = min(FRAME_W, RCX + _HALF_DIAG)
LAT_HARD_PX = 2    # within this → angular = 0
LAT_SAFE_PX = 9    # above this  → full angular

FPS = 30.0
DT = 1.0 / FPS
PREDICT_STEPS = 30
HISTORY_LEN = 3

PATH_COLOR = (60, 120, 255)
TRACK_FLASH = (120, 170, 255)
FLASH_HALF = 4


def _clearance_scale(clear_px):
    """S-curve 0→1 based purely on distance. No speed dependency."""
    if clear_px <= MIN_CLEARANCE_PX:
        return 0.0
    if clear_px >= BRAKE_START_PX:
        return 1.0
    t = (clear_px - MIN_CLEARANCE_PX) / float(BRAKE_START_PX - MIN_CLEARANCE_PX)
    return t * t * (3.0 - 2.0 * t)  # smoothstep


class SafetyGuard:
    def __init__(self):
        self._hist = deque(maxlen=HISTORY_LEN)
        self._fwd_scale = 1.0
        self._bwd_scale = 1.0
        self._ang_scale = 1.0
        self._throttled = False
        self._path = []
        self._tick = 0

    @property
    def is_throttled(self):
        return self._throttled

    @property
    def fwd_scale(self):
        return self._fwd_scale

    @property
    def bwd_scale(self):
        return self._bwd_scale

    @property
    def ang_scale(self):
        return self._ang_scale

    # ── per-frame update ──

    def update(self, obs_map, yaw_delta, fwd_delta):
        """Feed per-frame odometry.  Computes directional scales."""
        self._tick += 1
        self._hist.append((yaw_delta, fwd_delta))

        h_map, w_map = obs_map.shape
        y0, y1 = MASK_Y0, min(h_map, MASK_Y1)

        # ── Forward scan: from front edge rightward ──
        if MASK_X1 < w_map:
            strip = obs_map[y0:y1, MASK_X1:]
            col_hit = np.any(strip >= OBS_THRESH, axis=0)
            idxs = np.flatnonzero(col_hit)
            fwd_clear = int(idxs[0]) if len(idxs) > 0 else strip.shape[1]
        else:
            fwd_clear = 0

        # ── Backward scan: from rear edge leftward (extra caster margin) ──
        if BWD_SCAN_X0 > 0:
            strip = obs_map[y0:y1, :BWD_SCAN_X0][:, ::-1]
            col_hit = np.any(strip >= OBS_THRESH, axis=0)
            idxs = np.flatnonzero(col_hit)
            bwd_clear = int(idxs[0]) if len(idxs) > 0 else strip.shape[1]
        else:
            bwd_clear = 0

        self._fwd_scale = _clearance_scale(fwd_clear)
        self._bwd_scale = _clearance_scale(bwd_clear)
        if fwd_clear < 10 and self._tick % 5 == 0:
            print("safety: fwd_clear=%d bwd_clear=%d fwd_scale=%.2f "
                  "scan_region=[%d:%d, %d:]"
                  % (fwd_clear, bwd_clear, self._fwd_scale,
                     y0, y1, MASK_X1))

        # ── Lateral scans: above / below robot (diagonal x-extent) ──
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
        if min_lat <= LAT_HARD_PX:
            self._ang_scale = 0.0
        elif min_lat < LAT_SAFE_PX:
            self._ang_scale = (min_lat - LAT_HARD_PX) / float(LAT_SAFE_PX - LAT_HARD_PX)
        else:
            self._ang_scale = 1.0

        self._throttled = (self._fwd_scale < 0.95 or self._bwd_scale < 0.95
                           or self._ang_scale < 0.95)

        # ── Trajectory prediction (visualisation only) ──
        self._path = []
        if len(self._hist) >= 3:
            n = len(self._hist)
            omega = sum(h[0] for h in self._hist) / n * FPS
            speed = sum(h[1] for h in self._hist) / n * FPS
            x, y, th = float(RCX), float(RCY), 0.0
            for _ in range(PREDICT_STEPS):
                x += speed * math.cos(th) / PX_M * DT
                y -= speed * math.sin(th) / PX_M * DT
                th += omega * DT
                self._path.append((x, y))

    # ── drawing helpers ──

    def draw_trajectory(self, costmap):
        """Draw predicted 1-second path as a 5 px blue line."""
        if len(self._path) < 2:
            return
        pts = np.array([(int(round(x)), int(round(y)))
                        for x, y in self._path], dtype=np.int32)
        cv2.polylines(costmap, [pts], False, PATH_COLOR, 5, cv2.LINE_AA)

    def draw_wheel_flash(self, costmap):
        """Flash tracks + caster to lighter blue when throttled."""
        if not self._throttled or (self._tick // FLASH_HALF) % 2 != 0:
            return
        h = costmap.shape[0]
        tx0, tx1 = max(0, RCX - 7), RCX + 10
        costmap[max(0, RCY - 21):max(0, RCY - 15), tx0:tx1] = TRACK_FLASH
        costmap[RCY + 15:min(h, RCY + 21), tx0:tx1] = TRACK_FLASH
        costmap[RCY - 1:RCY + 2, max(0, RCX - 23):max(0, RCX - 17)] = TRACK_FLASH
