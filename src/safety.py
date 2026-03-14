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

# ── Robot physics (from wheelbase.py / odrivecan.py) ──
WHEEL_RADIUS_M = 0.08565
VEL_RAMP_RATE = 3.0  # turns/s²
DECEL_MPS2 = VEL_RAMP_RATE * 2.0 * math.pi * WHEEL_RADIUS_M  # ≈1.61 m/s²
LATENCY_S = 0.15
MIN_CLEARANCE_PX = 3
OBS_THRESH = 100

# ── Costmap geometry (must match vision.py) ──
PX_M = 0.010
ROBOT_W, ROBOT_H = 30, 42
CX_OFF = -78
CX, CY = 159, 119
RCX, RCY = CX + CX_OFF, CY

# Robot mask boundaries (same region _build_costmap clears)
MASK_X0 = max(0, RCX - 20)
MASK_Y0 = max(0, RCY - ROBOT_H // 2)
MASK_X1 = min(320, RCX + 16)
MASK_Y1 = MASK_Y0 + ROBOT_H

FPS = 30.0
DT = 1.0 / FPS
PREDICT_STEPS = 30
HISTORY_LEN = 30

PATH_COLOR = (60, 120, 255)
TRACK_FLASH = (120, 170, 255)
FLASH_HALF = 4


def _max_safe_speed(clear_px):
    """Max speed (m/s) that allows full stop within *clear_px* pixels."""
    avail = max(0.0, (clear_px - MIN_CLEARANCE_PX) * PX_M)
    if avail <= 0.0:
        return 0.0
    disc = (LATENCY_S * DECEL_MPS2) ** 2 + 2.0 * DECEL_MPS2 * avail
    return max(0.0, -LATENCY_S * DECEL_MPS2 + math.sqrt(disc))


class SafetyGuard:
    def __init__(self):
        self._hist = deque(maxlen=HISTORY_LEN)
        self._fwd_scale = 1.0
        self._bwd_scale = 1.0
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

        # ── Backward scan: from rear edge leftward ──
        if MASK_X0 > 0:
            strip = obs_map[y0:y1, :MASK_X0][:, ::-1]
            col_hit = np.any(strip >= OBS_THRESH, axis=0)
            idxs = np.flatnonzero(col_hit)
            bwd_clear = int(idxs[0]) if len(idxs) > 0 else strip.shape[1]
        else:
            bwd_clear = 0

        # ── Max safe speed in each direction ──
        v_max_fwd = _max_safe_speed(fwd_clear)
        v_max_bwd = _max_safe_speed(bwd_clear)

        # Use recent history for current speed estimate
        if len(self._hist) >= 3:
            n = len(self._hist)
            cur_speed = abs(sum(h[1] for h in self._hist) / n * FPS)
        else:
            cur_speed = 0.0

        if cur_speed > 1e-4:
            self._fwd_scale = min(1.0, v_max_fwd / cur_speed)
            self._bwd_scale = min(1.0, v_max_bwd / cur_speed)
        else:
            self._fwd_scale = 0.0 if fwd_clear <= MIN_CLEARANCE_PX else 1.0
            self._bwd_scale = 0.0 if bwd_clear <= MIN_CLEARANCE_PX else 1.0

        # Hard zero when within clearance regardless of speed estimate
        if fwd_clear <= MIN_CLEARANCE_PX:
            self._fwd_scale = 0.0
        if bwd_clear <= MIN_CLEARANCE_PX:
            self._bwd_scale = 0.0

        self._throttled = self._fwd_scale < 0.95 or self._bwd_scale < 0.95

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
