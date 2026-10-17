"""5 Hz reachable-(v,w) wander on the live ego heightmap + visit doughnut.

Obstacles are dilated by a 250 mm keep-clear disk (clean turn r = 200 mm)
so wander will not enter a gap it cannot turn in. Vision throttle
(`safety_fwd`) is a planning input, not only a wheelbase failsafe: +v is
never commanded into a pin, and a significant throttle reverses along the
IMU trail past the pinch, then yaws until plus-x is clear.
"""
from __future__ import annotations

import math
import time
import numpy as np

from robot_config import (
    RCX, RCY, EGO_PX_SIZE, FRAME_W, FRAME_H, FOOTPRINT_BOXES, WANDER_CLEAR_CM,
)

N_V, N_W = 8, 8
N_PATHS = N_V * N_W
VISIT_N = 32
VISIT_CELL = 0.10
PLUS_STOP = 0.28
OBS_CM = 5.0
REPLAN_S = 0.20
# Keep-clear around the axle: 250 mm. Dilate obstacles by that disk so
# wander will not enter a gap it cannot turn in (clean turn r = 200 mm).
FOOT_R_PX = int(round(WANDER_CLEAR_CM))
THROTTLE_HALF_PX = FOOT_R_PX
SAFETY_PIN = 0.55
SAFETY_HARD = 0.25
SAFETY_CLEAR = 0.90
RECOVER_PIN_S = 0.35
REVERSE_PAST_S = 0.40
REVERSE_MAX_S = 1.20
RECOVER_CLEAR_M = 0.75
RECOVER_CLEAR_N = 15
TRAIL_BACKUP_M = 0.55
TRAIL_MIN_M = 0.18
CLEAR_KEEP = 80
GOAL_REFRESH_S = 20.0
GOAL_RANGE_M = 2.0
GOAL_NEED_M = 1.50
GOAL_SEEK_M = 1.10
GOAL_CONE_RAD = 0.30


def plus_x_throttle_boxes():
    """Axis-aligned ego boxes for plus-x / left / right stop looks.

    Origin is the axle (RCX, RCY) — midpoint between wheels — NOT the
    image/crosshair center (CROSSHAIR is ~78 px / 78 cm forward of the axle).
    """
    stop_px = int(round(PLUS_STOP / EGO_PX_SIZE))
    half = THROTTLE_HALF_PX
    boxes = []
    for yaw in (0.0, 0.70, -0.70):
        c, s = math.cos(yaw), math.sin(yaw)
        x1 = int(round(RCX + c * stop_px))
        y1 = int(round(RCY - s * stop_px))
        x0, y0 = int(RCX), int(RCY)
        boxes.append((
            max(0, min(x0, x1) - half),
            max(0, min(y0, y1) - half),
            min(FRAME_W, max(x0, x1) + half + 1),
            min(FRAME_H, max(y0, y1) + half + 1),
        ))
    return boxes


def reachable_vw(v, w, hp):
    dt = float(hp.get("replan_s", REPLAN_S))
    max_v = float(hp.get("max_v", 0.28))
    max_w = float(hp.get("max_w", 0.80))
    v_rev = float(hp.get("v_rev", -0.10))
    dv = max(float(hp.get("a_v", 0.22)) * dt, 0.015)
    dw = max(float(hp.get("a_w", 1.00)) * dt, 0.06)
    v_lo = float(np.clip(v - dv, v_rev, max_v))
    v_hi = float(np.clip(v + dv, v_rev, max_v))
    w_lo = float(np.clip(w - dw, -max_w, max_w))
    w_hi = float(np.clip(w + dw, -max_w, max_w))
    if v_hi < v_lo:
        v_lo, v_hi = v_hi, v_lo
    if w_hi < w_lo:
        w_lo, w_hi = w_hi, w_lo
    if v_hi - v_lo < 1e-4:
        v_hi = v_lo + 1e-4
    if w_hi - w_lo < 1e-4:
        w_hi = w_lo + 1e-4
    return v_lo, v_hi, w_lo, w_hi


def sample_vw(v, w, prev_vw, hp):
    v_lo, v_hi, w_lo, w_hi = reachable_vw(v, w, hp)
    vs = np.linspace(v_lo, v_hi, N_V, dtype=np.float32)
    ws = np.linspace(w_lo, w_hi, N_W, dtype=np.float32)
    vv, ww = np.meshgrid(vs, ws, indexing="ij")
    cmd = np.stack((vv.ravel(), ww.ravel()), axis=1)
    if prev_vw is not None:
        cmd[0] = np.clip(np.asarray(prev_vw, dtype=np.float32).reshape(2),
                         [v_lo, w_lo], [v_hi, w_hi])
    return cmd.astype(np.float32)


def _layers(obs_input):
    """Return (obs uint8 HxW, height uint8 HxW or None).

    Prefer the live ego layer (`ego_obs` / `ego_height`). Persistent/gmap
    is only a fallback so accumulated walls do not pin wander.
    """
    if isinstance(obs_input, dict):
        height = obs_input.get("ego_height")
        live = obs_input.get("ego_obs")
        if live is not None:
            o = np.asarray(live, dtype=np.uint8)
            obs = np.where(o > 0, np.uint8(255), np.uint8(0))
            # Binary occupancy only — do not score residual height-cm.
            return obs, None
        pers = obs_input.get("ego_persistent")
        if pers is not None:
            obs = np.asarray(pers, dtype=np.uint8)
            h = np.asarray(height, dtype=np.uint8) if height is not None else None
            return obs, h
        if height is not None:
            h = np.asarray(height, dtype=np.uint8)
            return ((h >= OBS_CM).astype(np.uint8) * 255), h
    if obs_input is None:
        return np.zeros((FRAME_H, FRAME_W), dtype=np.uint8), None
    arr = np.asarray(obs_input)
    if arr.ndim == 3:
        arr = np.max(arr, axis=2)
    if arr.ndim != 2:
        return np.zeros((FRAME_H, FRAME_W), dtype=np.uint8), None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr, None


def _ego_rc(x, y, x0, y0, c0, s0):
    dx = x - np.float32(x0)
    dy = y - np.float32(y0)
    dx_e = dx * c0 + dy * s0
    dy_e = -dx * s0 + dy * c0
    inv = np.float32(1.0 / EGO_PX_SIZE)
    cols = (RCX + dx_e * inv).astype(np.int32)
    rows = (RCY - dy_e * inv).astype(np.int32)
    return rows, cols


def _gather_h(rows, cols, hmap, h, w):
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    idx = np.clip(rows * w + cols, 0, h * w - 1)
    val = hmap[idx].astype(np.float32)
    return ok, np.where(ok, val, np.float32(0.0))


class InsectPlanner:
    """Wander / goto_xy via 64 reachable (v,w) scored 5 s out."""

    def __init__(self, hp=None):
        self.hp = {
            "replan_s": REPLAN_S,
            "score_s": 5.0,
            "score_dt": 0.20,
            "max_v": 0.12,
            "max_w": 0.40,
            "v_rev": -0.10,
            "a_v": 0.22,
            "a_w": 1.00,
            "plus_x_stop": PLUS_STOP,
            "prim_progress": 0.75,
            "prim_stuck": 2.60,
            "curve_wall_coef": 0.70,
            "visit_coef": 0.55,
            "visit_hit_coef": 0.70,
            "visit_fwd_coef": 0.20,
            "visit_hot": 8.0,
            "goal_coef": 1.80,
            "goal_face_coef": 0.55,
            "goal_reach_m": 0.45,
            "spin_coef": 0.18,
            "unstuck": 0.0,
            "unstuck_clear_px": 0.50,
            "unstuck_hold_s": 0.40,
        }
        if hp:
            self.hp.update(hp)
        self._active = False
        self._wander = False
        self._goal = None
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self._prev = None
        self._last_replan = 0.0
        self.visit = np.zeros((VISIT_N, VISIT_N), dtype=np.float32)
        self._gx = None
        self._gy = None
        self._in_unstuck = False
        self._clear_n = 0
        self._unstuck = None
        self._plus_x = 8.0
        self._debug_open = (8.0, 8.0, 8.0)
        self._debug = {"tick_ms": 0.0, "prim_ms": 0.0, "hot": 0.0, "throt": 0.0}
        self._pose = (0.0, 0.0, 0.0)
        self._curve_world = np.zeros((0, 2), dtype=np.float32)
        self._trail_world = []
        self._clear_trail = []
        self._last_clear_xy = None
        self._recover = False
        self._recover_mode = ""
        self._pin_s = 0.0
        self._recover_t = 0.0
        self._past_s = 0.0
        self._recover_clear_n = 0
        self._reverse_xy = None
        self._reverse_src = ""
        self._goal_auto = False
        self._last_goal_t = 0.0
        self._seeking = False
        self._goal_blocked_s = 0.0
        self._slow_until = 0.0
        self._ban_th = None
        self._ban_until = 0.0
        self._last_recover_done = 0.0
        self._turn_sign = 0.0
        self._turn_t0 = 0.0
        self._mode_name = ""
        self._last_status_t = 0.0
        self._goal_open_s = 0.0

    def set_goal(self, x, y, auto=False):
        """Drive toward world (x, y). auto=True means the 20 s wander picker."""
        self._goal = (float(x), float(y))
        self._goal_auto = bool(auto)
        self._last_goal_t = time.monotonic()
        self._seeking = False
        self._active = True
        print("InsectPlanner: goal (%.2f, %.2f) auto=%s wander=%s"
              % (x, y, self._goal_auto, self._wander), flush=True)

    def set_wander_mode(self, enabled: bool):
        self._wander = bool(enabled)
        if enabled:
            self._goal = None
            self._goal_auto = True
            self._last_goal_t = 0.0
            self._seeking = False
            self._active = True
            print("InsectPlanner: wander ON (20 s / 2 m clear-path goals)", flush=True)
        else:
            self._active = False
            print("InsectPlanner: wander OFF", flush=True)

    def cancel(self):
        self._active = False
        self._wander = False
        self._goal = None
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self._prev = None
        self._in_unstuck = False
        self._clear_n = 0
        self._recover = False
        self._recover_mode = ""
        self._pin_s = 0.0
        self._recover_t = 0.0
        self._past_s = 0.0
        self._recover_clear_n = 0
        self._reverse_xy = None
        self._reverse_src = ""
        self._clear_trail = []
        self._last_clear_xy = None
        self._goal_auto = False
        self._seeking = False
        self._goal_blocked_s = 0.0
        self.visit[:] = 0
        self._gx = self._gy = None
        if self._unstuck is not None:
            self._unstuck.reset()
        print("InsectPlanner: cancelled", flush=True)

    def is_active(self):
        return self._active

    def get_debug_state(self):
        d = dict(self._debug)
        d.update(cmd_fwd=self._cmd_v, cmd_ang=self._cmd_w, wander=self._wander,
                 unstuck=self._in_unstuck, plus_x=self._plus_x, goal=self._goal,
                 open=self._debug_open,
                 recover=self._recover_mode if self._recover else "",
                 reverse_src=self._reverse_src if self._recover else "",
                 seeking=self._seeking, left_px=self._debug_open[1],
                 right_px=self._debug_open[2], mode=self._mode_name)
        return d

    def _append_trail(self, x, y):
        if self._trail_world:
            px, py = self._trail_world[-1]
            if (float(x) - px) ** 2 + (float(y) - py) ** 2 < 0.015 ** 2:
                return
        self._trail_world.append((float(x), float(y)))
        if len(self._trail_world) > 400:
            self._trail_world = self._trail_world[-300:]

    def _world_xy_to_ego(self, xy):
        """World (x,y) samples → ego (col, row) from axle RCX,RCY at last pose."""
        if xy is None or len(xy) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        arr = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        x0, y0, th0 = self._pose
        c0, s0 = np.float32(math.cos(th0)), np.float32(math.sin(th0))
        r, c = _ego_rc(arr[:, 0], arr[:, 1], x0, y0, c0, s0)
        return np.stack((c, r), axis=1).astype(np.int32)

    def paint_ego_overlay(self, overlay):
        """Path (cyan) + odom trail (gold) + axle (green). Origin = wheel axle."""
        import cv2
        if overlay is None or overlay.size == 0:
            return
        h, w = overlay.shape[:2]
        def _poly(pts, color, thick):
            if pts is None or len(pts) < 2:
                return
            prev = None
            for col, row in pts:
                p = (int(col), int(row))
                if not (0 <= p[0] < w and 0 <= p[1] < h):
                    prev = None
                    continue
                if prev is not None:
                    cv2.line(overlay, prev, p, color, thick, cv2.LINE_AA)
                prev = p
        _poly(self._world_xy_to_ego(self._trail_world), (240, 200, 60, 255), 1)
        _poly(self._world_xy_to_ego(self._curve_world), (80, 220, 255, 255), 2)
        if self._reverse_xy is not None:
            tgt = self._world_xy_to_ego([self._reverse_xy])
            if len(tgt):
                cv2.circle(overlay, (int(tgt[0, 0]), int(tgt[0, 1])), 6,
                           (255, 0, 180, 255), 2, cv2.LINE_AA)
        if self._goal is not None:
            g = self._world_xy_to_ego([self._goal])
            if len(g):
                cv2.circle(overlay, (int(g[0, 0]), int(g[0, 1])), 7,
                           (255, 180, 40, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (int(RCX), int(RCY)), 4, (40, 255, 80, 255), -1, cv2.LINE_AA)
        cv2.line(overlay, (int(RCX), int(RCY)), (int(RCX) + 16, int(RCY)),
                 (40, 255, 80, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _inflate_occ(obs, rad_px=FOOT_R_PX):
        """Dilate obstacles by the wander keep-clear disk, then clear self."""
        import cv2
        occ = (np.asarray(obs, dtype=np.uint8) > 0).astype(np.uint8) * np.uint8(255)
        r = int(max(0, rad_px))
        if r > 0:
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            kern = ((xx * xx + yy * yy) <= r * r).astype(np.uint8)
            occ = cv2.dilate(occ, kern, iterations=1)
        for x0, y0, x1, y1 in FOOTPRINT_BOXES:
            occ[y0:y1, x0:x1] = 0
        return occ

    def _note_clear(self, x, y, plus_x, safety_fwd):
        """Remember poses that were actually free — more useful than a crawl trail."""
        if self._recover:
            return
        if plus_x < PLUS_STOP or float(safety_fwd) < SAFETY_CLEAR:
            return
        self._last_clear_xy = (float(x), float(y))
        if self._clear_trail:
            px, py = self._clear_trail[-1]
            if (float(x) - px) ** 2 + (float(y) - py) ** 2 < 0.02 ** 2:
                return
        self._clear_trail.append((float(x), float(y)))
        if len(self._clear_trail) > CLEAR_KEEP:
            self._clear_trail = self._clear_trail[-60:]

    @staticmethod
    def _useful_backup(points, x, y, need=TRAIL_BACKUP_M, min_d=TRAIL_MIN_M):
        """Farthest useful history point; None if the path is too short to steer on."""
        best = None
        best_d = 0.0
        seq = points[:-1] if len(points) > 1 else points
        for px, py in reversed(seq):
            d = math.hypot(px - x, py - y)
            if d >= need:
                return (float(px), float(py)), d
            if d > best_d:
                best_d, best = d, (float(px), float(py))
        if best is not None and best_d >= min_d:
            return best, best_d
        return None, 0.0

    def _backup_xy(self):
        """Trail if it spans enough, else last-clear pose, else None (straight reverse)."""
        x, y, _th = self._pose
        tgt, _d = self._useful_backup(self._trail_world, x, y)
        if tgt is not None:
            return tgt, "trail"
        tgt, _d = self._useful_backup(self._clear_trail, x, y)
        if tgt is not None:
            return tgt, "clear"
        if self._last_clear_xy is not None:
            cx, cy = self._last_clear_xy
            if math.hypot(cx - x, cy - y) >= TRAIL_MIN_M:
                return (float(cx), float(cy)), "clear"
        return None, "straight"

    def _reverse_cmd(self):
        """Reverse along useful history, or straight body-back if history is junk."""
        v_rev = float(self.hp.get("v_rev", -0.10))
        max_w = float(self.hp.get("max_w", 0.80))
        tgt, src = self._backup_xy()
        self._reverse_xy = tgt
        self._reverse_src = src
        if tgt is None:
            return v_rev, 0.0
        x, y, th = self._pose
        bearing = math.atan2(tgt[1] - y, tgt[0] - x)
        err = bearing - (th + math.pi)
        while err > math.pi:
            err -= 2.0 * math.pi
        while err < -math.pi:
            err += 2.0 * math.pi
        w = float(np.clip(2.4 * err, -max_w, max_w))
        return v_rev, w

    def _maybe_recover(self, plus_x, left_px, right_px, dt, safety_fwd=1.0):
        """Throttle/plus-x pin → reverse along IMU trail, then yaw until clear.

        Never creeps +v into the pin. Vision throttle (`safety_fwd`) and the
        planner's own plus-x look are the same signal: the last plan was bad.
        """
        stop = float(self.hp.get("plus_x_stop", PLUS_STOP))
        dt = float(dt) if dt else 0.033
        safety_fwd = 1.0 if safety_fwd is None else float(safety_fwd)
        hard_pin = safety_fwd < SAFETY_HARD
        map_pin = plus_x < stop
        # Safety flicker at 0.4–0.55 with a still-open plus-x is not a pin.
        soft_pin = map_pin or (safety_fwd < SAFETY_PIN and plus_x < 0.50)
        if hard_pin or soft_pin:
            self._pin_s += dt
        else:
            self._pin_s = 0.0

        if not self._recover:
            start = (hard_pin and self._pin_s >= 0.03) or (
                soft_pin and self._pin_s >= RECOVER_PIN_S)
            if start:
                self._recover = True
                self._recover_mode = "reverse"
                self._recover_t = 0.0
                self._past_s = 0.0
                self._recover_clear_n = 0
                _tgt, self._reverse_src = self._backup_xy()
                self._reverse_xy = _tgt
                print("InsectPlanner: recover reverse %s plus_x=%.2f "
                      "safety=%.2f L=%.2f R=%.2f"
                      % (self._reverse_src or "straight", plus_x, safety_fwd,
                         left_px, right_px), flush=True)
                self._turn_sign = 0.0
            else:
                return False

        self._recover_t += dt

        if self._recover_mode == "reverse":
            lifted = (safety_fwd >= SAFETY_CLEAR) and (plus_x >= stop)
            if lifted:
                self._past_s += dt
            else:
                self._past_s = 0.0
            if self._past_s >= REVERSE_PAST_S or self._recover_t >= REVERSE_MAX_S:
                self._recover_mode = "yaw"
                self._past_s = 0.0
                self._recover_clear_n = 0
                print("InsectPlanner: recover yaw after reverse plus_x=%.2f "
                      "safety=%.2f" % (plus_x, safety_fwd), flush=True)
            if self._recover_mode == "reverse":
                v, w = self._reverse_cmd()
                self._cmd_v, self._cmd_w = float(v), float(w)
                self._prev = (self._cmd_v, self._cmd_w)
                return True

        # Yaw in place until heading is open. Do not creep forward.
        # Lock turn direction — L/R looks flicker and were flipping ±0.7 every tick.
        prefer = 1.0 if left_px >= right_px else -1.0
        wsign = self._sticky_turn(prefer)
        if plus_x >= RECOVER_CLEAR_M and safety_fwd >= SAFETY_CLEAR:
            self._recover_clear_n += 1
            if self._recover_clear_n >= RECOVER_CLEAR_N:
                now = time.monotonic()
                bounced = (self._last_recover_done
                           and (now - self._last_recover_done) < 8.0)
                print("InsectPlanner: recover done plus_x=%.2f safety=%.2f%s"
                      % (plus_x, safety_fwd, " bounce→seek" if bounced else ""),
                      flush=True)
                x, y, th = self._pose
                self._ban_th = th
                self._ban_until = now + 8.0
                self._slow_until = now + 2.5
                self._last_recover_done = now
                self._recover = False
                self._recover_mode = ""
                self._recover_t = 0.0
                self._past_s = 0.0
                self._reverse_xy = None
                self._recover_clear_n = 0
                if bounced:
                    self._goal = None
                    self._seeking = True
                    self._last_goal_t = 0.0
                return False
            self._cmd_v = 0.0
            self._prev = (self._cmd_v, self._cmd_w)
            return True
        self._recover_clear_n = 0
        self._reverse_xy = None
        self._cmd_v = 0.0
        self._cmd_w = wsign * 0.40
        self._prev = (self._cmd_v, self._cmd_w)
        return True

    def _emit(self, plus_x, safety_fwd=1.0):
        """Never command +v into throttle; commit speed to remaining clearance."""
        safety_fwd = 1.0 if safety_fwd is None else float(safety_fwd)
        if self._cmd_v > 0.0:
            # Hard pin only: map-stop or safety slam. Soft safety (0.4–0.7)
            # with an open plus-x must not freeze in place.
            if plus_x < PLUS_STOP or safety_fwd < 0.25:
                self._cmd_v = 0.0
            elif safety_fwd < 0.70 and plus_x < 0.90:
                self._cmd_v = 0.0
            elif plus_x < 0.55:
                self._cmd_v = min(self._cmd_v, 0.06)
            elif plus_x < 0.90:
                self._cmd_v = min(self._cmd_v, 0.10)
            elif plus_x < 1.30:
                self._cmd_v = min(self._cmd_v, 0.14)
            if safety_fwd < 0.85:
                self._cmd_v = min(self._cmd_v, 0.10)
            if time.monotonic() < self._slow_until:
                self._cmd_v = min(self._cmd_v, 0.08)
            self._prev = (self._cmd_v, self._cmd_w)
        lo, le, ri = self._debug_open
        if self._recover:
            mode = "recover-" + (self._recover_mode or "?")
        elif self._seeking:
            mode = "seek"
        else:
            mode = "drive"
        self._status(mode, plus_x, le, ri, safety_fwd)
        return {"fwd_mps": self._cmd_v, "ang_rads": self._cmd_w}

    def _ensure_unstuck(self):
        if self._unstuck is None:
            from unstuck_policy import UnstuckPolicy
            self._unstuck = UnstuckPolicy(hp=self.hp)
        return self._unstuck

    def _openings(self, height, obs):
        """plus-x and left/right clearance from the axle cell, metres."""
        from unstuck_policy import _look_px
        fwd = _look_px(height, obs, RCX, RCY, 0.0, GOAL_RANGE_M)
        left = _look_px(height, obs, RCX, RCY, 0.70, 1.2)
        right = _look_px(height, obs, RCX, RCY, -0.70, 1.2)
        return fwd, left, right

    def _look(self, fat, yaw_img, dist_m):
        from unstuck_policy import _look_px
        return _look_px(fat, fat, RCX, RCY, yaw_img, dist_m)

    def _heading_banned(self, yaw_abs):
        now = time.monotonic()
        if self._ban_th is None or now > self._ban_until:
            return False
        err = yaw_abs - self._ban_th
        while err > math.pi:
            err -= 2.0 * math.pi
        while err < -math.pi:
            err += 2.0 * math.pi
        return abs(err) < 0.70

    def _sticky_turn(self, prefer_sign):
        """Hold one yaw sign for ~1.2 s so noisy L/R cannot flip every tick."""
        now = time.monotonic()
        prefer = 1.0 if prefer_sign >= 0.0 else -1.0
        if self._turn_sign == 0.0 or (now - self._turn_t0) >= 1.20:
            if prefer != self._turn_sign:
                print("insect: turn_dir %+g plus_x=%.2f" % (prefer, self._plus_x),
                      flush=True)
            self._turn_sign = prefer
            self._turn_t0 = now
        return self._turn_sign

    def _status(self, mode, plus_x, left_px, right_px, safety_fwd):
        now = time.monotonic()
        changed = mode != self._mode_name
        if changed or (now - self._last_status_t) >= 1.0:
            g = self._goal
            gs = "none" if g is None else "(%.2f,%.2f)" % (g[0], g[1])
            print("insect: %s plus_x=%.2f L=%.2f R=%.2f safety=%.2f "
                  "v=%.2f w=%.2f goal=%s"
                  % (mode, plus_x, left_px, right_px, safety_fwd,
                     self._cmd_v, self._cmd_w, gs), flush=True)
            self._mode_name = mode
            self._last_status_t = now

    def _best_forward(self, fat):
        """Best heading in a forward fan with a clear ~2 m corridor, or None."""
        x, y, th = self._pose
        best = None
        for deg in range(-50, 55, 10):
            yaw_img = deg * math.pi / 180.0
            look = self._look(fat, yaw_img, GOAL_RANGE_M)
            if look < GOAL_SEEK_M:
                continue
            yaw_abs = th + yaw_img
            if self._heading_banned(yaw_abs):
                continue
            score = look - 0.35 * abs(yaw_img)
            if best is None or score > best[0]:
                best = (score, look, yaw_img)
        if best is None:
            return None, -1.0
        return best[2], best[1]

    def _maybe_refresh_goal(self, fat, plus_x, left_px, right_px, dt):
        """20 s / 2 m goal if a clear corridor exists; else turn in place."""
        if not self._wander or not self._goal_auto:
            self._seeking = False
            return False
        x, y, th = self._pose
        now = time.monotonic()
        if self._goal is not None:
            gx, gy = self._goal
            br = math.atan2(gy - y, gx - x) - th
            while br > math.pi:
                br -= 2.0 * math.pi
            while br < -math.pi:
                br += 2.0 * math.pi
            look_g = self._look(fat, br, GOAL_RANGE_M)
            if look_g < 0.60:
                self._goal_blocked_s += dt
                if self._goal_blocked_s >= 0.40:
                    print("InsectPlanner: drop blocked goal look=%.2f" % look_g,
                          flush=True)
                    self._goal = None
                    self._last_goal_t = 0.0
                    self._goal_blocked_s = 0.0
            else:
                self._goal_blocked_s = 0.0
            if self._goal is not None and (now - self._last_goal_t) < GOAL_REFRESH_S:
                self._seeking = False
                return False
        yaw_img, look = self._best_forward(fat)
        # Need a stable *forward* corridor, not a 1-frame side glance.
        if (yaw_img is None or look < GOAL_NEED_M or abs(yaw_img) > GOAL_CONE_RAD
                or plus_x < 1.20):
            self._goal_open_s = 0.0
            self._seeking = True
            self._goal = None
            if yaw_img is not None:
                prefer = 1.0 if yaw_img > 0.0 else -1.0
            else:
                prefer = 1.0 if left_px >= right_px else -1.0
            self._cmd_v = 0.0
            self._cmd_w = self._sticky_turn(prefer) * 0.35
            self._prev = (self._cmd_v, self._cmd_w)
            return True
        if plus_x < 1.50:
            self._goal_open_s += dt
            if self._goal_open_s < 0.25:
                self._seeking = True
                self._cmd_v = 0.0
                self._cmd_w = 0.0
                self._prev = (self._cmd_v, self._cmd_w)
                return True
        gx = x + GOAL_RANGE_M * math.cos(th + yaw_img)
        gy = y + GOAL_RANGE_M * math.sin(th + yaw_img)
        self.set_goal(gx, gy, auto=True)
        self._seeking = False
        self._turn_sign = 0.0
        self._goal_open_s = 0.0
        return False

    def _maybe_unstuck(self, plus_x, left_px, right_px, dt):
        if float(self.hp.get("unstuck", 0.0)) < 0.5:
            self._in_unstuck = False
            self._clear_n = 0
            return False
        stop = float(self.hp.get("plus_x_stop", PLUS_STOP))
        clear_px = float(self.hp.get("unstuck_clear_px", 0.50))
        hold_n = int(round(float(self.hp.get("unstuck_hold_s", 0.40)) / max(float(dt), 0.033)))
        if plus_x < stop:
            if not self._in_unstuck:
                self._ensure_unstuck().reset()
            self._in_unstuck = True
            self._clear_n = 0
            return True
        if self._in_unstuck:
            if plus_x >= clear_px:
                self._clear_n += 1
                if self._clear_n >= max(3, hold_n):
                    self._in_unstuck = False
                    self._clear_n = 0
                    return False
                return True
            self._clear_n = 0
            return True
        return False

    def _burn_visit(self, x, y, dt):
        gx = int(round(float(x) / VISIT_CELL))
        gy = int(round(float(y) / VISIT_CELL))
        if self._gx is None:
            self._gx, self._gy = gx, gy
        dgx, dgy = gx - self._gx, gy - self._gy
        self._gx, self._gy = gx, gy
        if dgx or dgy:
            out = np.zeros_like(self.visit)
            h, w = self.visit.shape
            dcol, drow = -dgx, dgy
            sr0, sr1 = max(0, -drow), min(h, h - drow)
            dr0 = max(0, drow)
            sc0, sc1 = max(0, -dcol), min(w, w - dcol)
            dc0 = max(0, dcol)
            if sr1 > sr0 and sc1 > sc0:
                out[dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = (
                    self.visit[sr0:sr1, sc0:sc1])
            self.visit = out
        amp = float(dt) / 0.10
        cy = cx = VISIT_N // 2
        for di, dj, a in (
            (0, 0, 1.0),
            (-1, 0, 0.35), (1, 0, 0.35), (0, -1, 0.35), (0, 1, 0.35),
            (-1, -1, 0.15), (-1, 1, 0.15), (1, -1, 0.15), (1, 1, 0.15),
        ):
            r, c = cy + di, cx + dj
            if 0 <= r < VISIT_N and 0 <= c < VISIT_N:
                self.visit[r, c] += np.float32(a * amp)

    def _visit_heat(self, x, y, x0, y0):
        n = int(x.shape[0])
        cy = cx = VISIT_N // 2
        col = np.rint(cx + (x - np.float32(x0)) / VISIT_CELL).astype(np.int32)
        row = np.rint(cy - (y - np.float32(y0)) / VISIT_CELL).astype(np.int32)
        ok = (row >= 0) & (row < VISIT_N) & (col >= 0) & (col < VISIT_N)
        r = np.clip(row, 0, VISIT_N - 1)
        c = np.clip(col, 0, VISIT_N - 1)
        return np.where(ok, self.visit[r, c], np.float32(0.0)).astype(np.float32)

    def _score(self, cmd, pose, obs, height):
        hp = self.hp
        n = int(cmd.shape[0])
        dt = np.float32(hp.get("score_dt", 0.20))
        steps = max(8, int(round(float(hp.get("score_s", 5.0)) / float(dt))))
        stop = np.float32(hp.get("plus_x_stop", PLUS_STOP))
        hot_s = np.float32(max(float(hp.get("visit_hot", 8.0)), 1e-3))
        x0, y0, th0 = float(pose[0]), float(pose[1]), float(pose[2])
        c0, s0 = np.float32(math.cos(th0)), np.float32(math.sin(th0))
        x = np.full(n, np.float32(x0))
        y = np.full(n, np.float32(y0))
        yaw = np.full(n, np.float32(th0))
        v_cmd, w_cmd = cmd[:, 0], cmd[:, 1]
        path = np.zeros(n, dtype=np.float32)
        throt = np.zeros(n, dtype=np.float32)
        coll = np.zeros(n, dtype=np.float32)
        close = np.zeros(n, dtype=np.float32)
        hot = np.zeros(n, dtype=np.float32)
        hot_hit = np.zeros(n, dtype=np.float32)
        min_px = np.full(n, np.float32(8.0))
        hh, ww = obs.shape[:2]
        occ = np.asarray(obs, dtype=np.uint8) > 0
        hflat = occ.ravel().astype(np.uint8) * np.uint8(40)
        steps_n = steps
        curves = np.zeros((n, steps_n + 1, 2), dtype=np.float32)
        curves[:, 0, 0] = x
        curves[:, 0, 1] = y
        k = 0
        for _ in range(steps):
            look = stop
            fx = x + np.cos(yaw).astype(np.float32) * look
            fy = y + np.sin(yaw).astype(np.float32) * look
            r, c = _ego_rc(fx, fy, x0, y0, c0, s0)
            ok, hv = _gather_h(r, c, hflat, hh, ww)
            pin = ok & (hv >= OBS_CM)
            throt += pin.astype(np.float32)
            v = np.where(pin, np.float32(0.0), v_cmd)
            w = w_cmd
            nx = x + v * np.cos(yaw).astype(np.float32) * dt
            ny = y + v * np.sin(yaw).astype(np.float32) * dt
            nyaw = yaw + w * dt
            br, bc = _ego_rc(nx, ny, x0, y0, c0, s0)
            bok, bh = _gather_h(br, bc, hflat, hh, ww)
            blocked = bok & (bh >= OBS_CM)
            dist = np.hypot(nx - x, ny - y)
            moved = (~blocked) & (dist > np.float32(0.002))
            path += np.where(moved, dist, np.float32(0.0))
            x = np.where(moved, nx, x)
            y = np.where(moved, ny, y)
            yaw = np.where(moved, nyaw, yaw)
            yaw = np.where((~moved) & (np.abs(w) > 0.04), nyaw, yaw)
            coll += blocked.astype(np.float32)
            close += np.clip(bh / np.float32(40.0), 0.0, 2.0)
            heat = self._visit_heat(x, y, x0, y0)
            hot += np.tanh(heat / hot_s)
            hot_hit += (heat >= hot_s).astype(np.float32)
            min_px = np.minimum(min_px, np.where(pin, np.float32(0.10), look))
            k += 1
            curves[:, k, 0] = x
            curves[:, k, 1] = y
        throt /= np.float32(steps)
        coll /= np.float32(steps)
        close /= np.float32(steps)
        hot /= np.float32(steps)
        hot_hit /= np.float32(steps)
        fx = x + np.cos(yaw).astype(np.float32) * np.float32(1.0)
        fy = y + np.sin(yaw).astype(np.float32) * np.float32(1.0)
        r, c = _ego_rc(fx, fy, x0, y0, c0, s0)
        ok, hv = _gather_h(r, c, hflat, hh, ww)
        px_end = np.where(ok & (hv >= OBS_CM), np.float32(0.15), np.float32(1.5))
        heat_end = self._visit_heat(x, y, x0, y0)
        prog = np.float32(hp.get("prim_progress", 0.55))
        stuck = np.float32(hp.get("prim_stuck", 2.60))
        wall = np.float32(hp.get("curve_wall_coef", 0.95))
        vcoef = np.float32(hp.get("visit_coef", 1.20))
        vhit = np.float32(hp.get("visit_hit_coef", 1.20))
        vend = np.float32(hp.get("visit_fwd_coef", 0.28))
        spin = np.float32(hp.get("spin_coef", 0.08))
        max_w = np.float32(max(float(hp.get("max_w", 0.80)), 1e-6))
        score = (
            prog * path
            + np.float32(0.18) * np.minimum(px_end, 2.0)
            + np.float32(0.10) * np.minimum(min_px, 2.0)
            + vend * (np.float32(1.0) - np.tanh(heat_end / hot_s))
            - wall * close
            - stuck * throt
            - np.float32(2.0) * coll
            - vcoef * hot
            - vhit * hot_hit
            - spin * (np.abs(w_cmd) / max_w)
        )
        if self._goal is not None:
            gx, gy = self._goal
            d0 = math.hypot(gx - x0, gy - y0)
            dend = np.hypot(gx - x, gy - y)
            score += np.float32(hp.get("goal_coef", 1.15)) * (np.float32(d0) - dend)
            br = np.arctan2(np.float32(gy) - y, np.float32(gx) - x)
            face = np.cos(yaw - br)
            score += np.float32(hp.get("goal_face_coef", 0.35)) * face * (np.float32(1.0) - throt)
        score = np.where((path < 0.12) & (throt > 0.40), score - 1.6, score)
        return score.astype(np.float32), path, throt, hot, curves

    def _unstuck_obs(self, height, obs, plus_x, left_px, right_px, bearing):
        from unstuck_policy import crop_kevin_ego
        occ = crop_kevin_ego(height, obs)
        stop = float(self.hp.get("plus_x_stop", PLUS_STOP))
        max_v = max(float(self.hp.get("max_v", 0.28)), 1e-6)
        max_w = max(float(self.hp.get("max_w", 0.80)), 1e-6)
        vec = np.array([
            np.float32(np.clip(self._cmd_v / max_v, -1.0, 1.0)),
            np.float32(np.clip(self._cmd_w / max_w, -1.0, 1.0)),
            np.float32(min(1.0, plus_x / 2.0)),
            np.float32(min(1.0, left_px / 1.2)),
            np.float32(min(1.0, right_px / 1.2)),
            np.float32(1.0 if plus_x < stop else 0.0),
            np.float32(math.sin(bearing)),
            np.float32(math.cos(bearing)),
        ], dtype=np.float32)
        return occ, vec

    def tick(self, obs_input, pose, dt, safety_fwd=1.0, pose_ok=True):
        if not self._active:
            return None
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        self._pose = (x, y, th)
        pose_ok = bool(pose_ok)
        # Frozen/unlocked pose makes world xy a "drive forever" heading.
        if not pose_ok and self._goal is not None:
            print("InsectPlanner: drop world goal (pose unlocked)", flush=True)
            self._goal = None
            self._last_goal_t = 0.0
            self._goal_auto = bool(self._wander)
        self._append_trail(x, y)
        self._burn_visit(x, y, float(dt) if dt else 0.033)
        now = time.monotonic()
        obs, _height = _layers(obs_input)
        fat = self._inflate_occ(obs)
        plus_x, left_px, right_px = self._openings(fat, fat)
        # Safety already scales the twist. Do not fake a plus-x wall from
        # extras RGB or we only yaw in place.
        self._plus_x = float(plus_x)
        self._debug_open = (float(plus_x), float(left_px), float(right_px))
        dt = float(dt) if dt else 0.033
        self._note_clear(x, y, plus_x, 1.0 if safety_fwd is None else float(safety_fwd))
        bearing = 0.0
        if self._goal is not None:
            gx, gy = self._goal
            dist_g = math.hypot(gx - x, gy - y)
            bearing = math.atan2(gy - y, gx - x) - th
            while bearing > math.pi:
                bearing -= 2.0 * math.pi
            while bearing < -math.pi:
                bearing += 2.0 * math.pi
            if dist_g <= float(self.hp.get("goal_reach_m", 0.45)):
                print("InsectPlanner: arrived (%.2f, %.2f)" % (gx, gy), flush=True)
                self._goal = None
                self._last_goal_t = 0.0
                bearing = 0.0
                if not self._wander:
                    self._cmd_v, self._cmd_w = 0.0, 0.0
                    self._active = False
                    return None
                # Fall through — wander picks the next 2 m goal.
        if float(self.hp.get("unstuck", 0.0)) >= 0.5 and self._maybe_unstuck(
                plus_x, left_px, right_px, dt):
            pol = self._ensure_unstuck()
            img, vec = self._unstuck_obs(fat, fat, plus_x, left_px, right_px, bearing)
            v, w, src = pol.act(
                plus_x, left_px, right_px, self._cmd_v, self._cmd_w,
                img=img, vec=vec, goal_bearing=bearing)
            self._cmd_v, self._cmd_w = float(v), float(w)
            self._debug = {
                "tick_ms": 0.0, "prim_ms": 0.0, "hot": 0.0,
                "throt": 1.0 if plus_x < PLUS_STOP else 0.0,
                "plus_x": plus_x, "unstuck": True, "unstuck_src": src,
                "live": True, "goal": self._goal,
            }
            return self._emit(plus_x, safety_fwd)
        if self._maybe_recover(plus_x, left_px, right_px, dt, safety_fwd):
            self._debug = {
                "tick_ms": 0.0, "prim_ms": 0.0, "hot": 0.0,
                "throt": 1.0 if plus_x < PLUS_STOP else 0.0,
                "plus_x": plus_x, "recover": self._recover_mode,
                "live": True, "goal": self._goal, "safety_fwd": safety_fwd,
            }
            return self._emit(plus_x, safety_fwd)
        if pose_ok and self._maybe_refresh_goal(fat, plus_x, left_px, right_px, dt):
            self._debug = {
                "tick_ms": 0.0, "prim_ms": 0.0, "hot": 0.0, "throt": 0.0,
                "plus_x": plus_x, "seek": True, "live": True, "goal": None,
                "safety_fwd": safety_fwd,
            }
            return self._emit(plus_x, safety_fwd)
        due = (now - self._last_replan) >= float(self.hp.get("replan_s", REPLAN_S)) - 1e-4
        if not due and self._prev is not None:
            return self._emit(plus_x, safety_fwd)
        t0 = time.perf_counter()
        hp = dict(self.hp)
        if time.monotonic() < self._slow_until:
            hp["max_v"] = min(float(hp.get("max_v", 0.22)), 0.10)
        cmd = sample_vw(self._cmd_v, self._cmd_w, self._prev, hp)
        score, path, throt, hot, curves = self._score(cmd, (x, y, th), fat, None)
        if self._prev is not None:
            score = score.copy()
            score[0] += np.float32(0.07)
        i = int(np.argmax(score))
        self._cmd_v = float(cmd[i, 0])
        self._cmd_w = float(cmd[i, 1])
        self._curve_world = curves[i]
        self._prev = (self._cmd_v, self._cmd_w)
        self._last_replan = now
        ms = (time.perf_counter() - t0) * 1e3
        self._debug = {
            "tick_ms": ms,
            "prim_ms": ms,
            "hot": float(hot[i]),
            "throt": float(throt[i]),
            "path5": float(path[i]),
            "score": float(score[i]),
            "n": N_PATHS,
            "live": True,
            "safety_fwd": safety_fwd,
            "goal": self._goal,
        }
        return self._emit(plus_x, safety_fwd)
