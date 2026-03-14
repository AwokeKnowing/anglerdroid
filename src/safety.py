"""safety.py – Collision avoidance + trajectory prediction.

Modular virtual sensor overlay.  Checks the robot's predicted path
against the persistent obstacle map, provides a velocity scale factor
(0-1) to prevent collisions, and draws a predicted trajectory line.
Flashes wheel tracks when throttled.
"""

import math
from collections import deque
import numpy as np
import cv2

# ── Robot physics (from wheelbase.py / odrivecan.py) ──
WHEEL_RADIUS_M = 0.08565
VEL_RAMP_RATE = 3.0  # turns/s² (ODrive setting)
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
MASK_X0 = max(0, RCX - 20)
MASK_Y0 = max(0, RCY - ROBOT_H // 2)
MASK_X1 = min(320, RCX + 16)
MASK_Y1 = MASK_Y0 + ROBOT_H

FPS = 30.0
DT = 1.0 / FPS
PREDICT_STEPS = 30
HISTORY_LEN = 30

PATH_COLOR = (60, 120, 255)
TRACK_NORMAL = (30, 60, 180)
TRACK_FLASH = (120, 170, 255)
FLASH_HALF = 4  # frames per half-cycle ≈ 3.75 Hz


class SafetyGuard:
    def __init__(self):
        self._hist = deque(maxlen=HISTORY_LEN)
        self._throttled = False
        self._scale = 1.0
        self._path = []
        self._tick = 0

    @property
    def is_throttled(self):
        return self._throttled

    @property
    def vel_scale(self):
        return self._scale

    # ── per-frame update ──

    def update(self, obs_map, yaw_delta, fwd_delta):
        """Feed per-frame odometry deltas.  Returns velocity scale 0.0-1.0."""
        self._tick += 1
        self._hist.append((yaw_delta, fwd_delta))

        if len(self._hist) < 3:
            self._scale, self._throttled, self._path = 1.0, False, []
            return 1.0

        n = len(self._hist)
        omega = sum(h[0] for h in self._hist) / n * FPS
        speed = sum(h[1] for h in self._hist) / n * FPS

        path = []
        x, y, th = float(RCX), float(RCY), 0.0
        hw, hh = ROBOT_W // 2, ROBOT_H // 2
        h_map, w_map = obs_map.shape
        first_hit = None

        for step in range(PREDICT_STEPS):
            x += speed * math.cos(th) / PX_M * DT
            y -= speed * math.sin(th) / PX_M * DT
            th += omega * DT
            path.append((x, y))

            if first_hit is not None:
                continue
            ix, iy = int(round(x)), int(round(y))
            if MASK_X0 <= ix <= MASK_X1 and MASK_Y0 <= iy <= MASK_Y1:
                continue
            bx0, by0 = max(0, ix - hw), max(0, iy - hh)
            bx1, by1 = min(w_map, ix + hw), min(h_map, iy + hh)
            if bx1 > bx0 and by1 > by0:
                if np.any(obs_map[by0:by1, bx0:bx1] >= OBS_THRESH):
                    first_hit = step

        self._path = path

        if first_hit is not None:
            dist_px = 0.0
            px, py = float(RCX), float(RCY)
            for i in range(first_hit + 1):
                dx, dy = path[i][0] - px, path[i][1] - py
                dist_px += math.sqrt(dx * dx + dy * dy)
                px, py = path[i]

            avail_m = max(0.0, (dist_px - MIN_CLEARANCE_PX) * PX_M)

            if avail_m <= 0.0:
                self._scale = 0.0
            else:
                # Max speed that still allows stopping in avail_m:
                #   v * LATENCY + v² / (2 * decel) = avail_m
                # Solve quadratic for v:
                disc = (LATENCY_S * DECEL_MPS2) ** 2 + 2.0 * DECEL_MPS2 * avail_m
                v_max = -LATENCY_S * DECEL_MPS2 + math.sqrt(disc)
                v_max = max(0.0, v_max)

                cur = abs(speed)
                if cur > 1e-4:
                    self._scale = min(1.0, v_max / cur)
                else:
                    self._scale = 1.0 if avail_m > MIN_CLEARANCE_PX * PX_M else 0.0

            self._throttled = self._scale < 0.95
        else:
            self._scale = 1.0
            self._throttled = False

        return self._scale

    # ── drawing helpers (called on the costmap image) ──

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
