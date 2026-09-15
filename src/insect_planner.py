"""5 Hz reachable-(v,w) wander on the live ego heightmap + visit doughnut.

Same controller as sim/explore_prim.py, but obstacles come from the 320×240
ego map (no privileged house boxes, no global costmap). Scoring runs at 5 Hz;
30 Hz ticks just hold the last command. Capture/map thread is untouched.

Safety reflexes in vision/wheelbase still scale or zero the twist.
"""
from __future__ import annotations

import math
import time
import numpy as np

from robot_config import RCX, RCY, EGO_PX_SIZE, FRAME_W, FRAME_H

N_V, N_W = 8, 8
N_PATHS = N_V * N_W
VISIT_N = 32
VISIT_CELL = 0.10
PLUS_STOP = 0.28
OBS_CM = 5.0
REPLAN_S = 0.20


def reachable_vw(v, w, hp):
    dt = float(hp.get("replan_s", REPLAN_S))
    max_v = float(hp.get("max_v", 0.28))
    max_w = float(hp.get("max_w", 0.80))
    v_rev = float(hp.get("v_rev", -0.10))
    dv = max(float(hp.get("a_v", 0.70)) * dt, 0.05)
    dw = max(float(hp.get("a_w", 2.40)) * dt, 0.20)
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
    """Return (obs uint8 HxW, height uint8 HxW or None)."""
    if isinstance(obs_input, dict):
        height = obs_input.get("ego_height")
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
            "max_v": 0.28,
            "max_w": 0.80,
            "v_rev": -0.10,
            "a_v": 0.70,
            "a_w": 2.40,
            "plus_x_stop": PLUS_STOP,
            "prim_progress": 0.55,
            "prim_stuck": 2.60,
            "curve_wall_coef": 0.95,
            "visit_coef": 1.20,
            "visit_hit_coef": 1.20,
            "visit_fwd_coef": 0.28,
            "visit_hot": 8.0,
            "goal_coef": 0.45,
            "spin_coef": 0.08,
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
        self._debug = {"tick_ms": 0.0, "prim_ms": 0.0, "hot": 0.0, "throt": 0.0}

    def set_goal(self, x, y):
        self._goal = (float(x), float(y))
        self._wander = False
        self._active = True
        print("InsectPlanner: goal (%.2f, %.2f)" % (x, y), flush=True)

    def set_wander_mode(self, enabled: bool):
        self._wander = bool(enabled)
        if enabled:
            self._goal = None
            self._active = True
            print("InsectPlanner: wander ON (5 Hz reachable v,w)", flush=True)
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
        self.visit[:] = 0
        self._gx = self._gy = None
        print("InsectPlanner: cancelled", flush=True)

    def is_active(self):
        return self._active

    def get_debug_state(self):
        d = dict(self._debug)
        d.update(cmd_fwd=self._cmd_v, cmd_ang=self._cmd_w, wander=self._wander)
        return d

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
        if height is not None:
            hflat = np.asarray(height, dtype=np.uint8).ravel()
        else:
            hflat = (np.asarray(obs, dtype=np.uint8) > 100).astype(np.uint8).ravel() * np.uint8(40)
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
            score += np.float32(hp.get("goal_coef", 0.45)) * (np.float32(d0) - dend)
        score = np.where((path < 0.12) & (throt > 0.40), score - 1.6, score)
        return score.astype(np.float32), path, throt, hot

    def tick(self, obs_input, pose, dt):
        if not self._active:
            return None
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        self._burn_visit(x, y, float(dt) if dt else 0.033)
        now = time.monotonic()
        due = (now - self._last_replan) >= float(self.hp.get("replan_s", REPLAN_S)) - 1e-4
        if not due and self._prev is not None:
            return {"fwd_mps": self._cmd_v, "ang_rads": self._cmd_w}
        t0 = time.perf_counter()
        obs, height = _layers(obs_input)
        cmd = sample_vw(self._cmd_v, self._cmd_w, self._prev, self.hp)
        score, path, throt, hot = self._score(cmd, (x, y, th), obs, height)
        if self._prev is not None:
            score = score.copy()
            score[0] += np.float32(0.07)
        i = int(np.argmax(score))
        self._cmd_v = float(cmd[i, 0])
        self._cmd_w = float(cmd[i, 1])
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
        }
        return {"fwd_mps": self._cmd_v, "ang_rads": self._cmd_w}
