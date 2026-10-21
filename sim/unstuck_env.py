#!/usr/bin/env python3
"""Cheap CPU unstuck gym: pin-spawn, raster ego, no Newton / no 20h GPU.

Actor sees a 48×80 occupancy crop (same FOV Kevin will crop from the 1 cm
ego heightmap) plus a 8-D vec. Reward is open plus-x and hold it clear.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_SIM = Path(__file__).resolve().parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

import explore_prim as prim  # noqa: E402
from rand_house import make_layout  # noqa: E402
from unstuck_net import BACK_M, FWD_M, IMG_H, IMG_W, LAT_M, VEC_DIM  # noqa: E402

PX = (BACK_M + FWD_M) / float(IMG_W)
RCX = BACK_M / PX
RCY = LAT_M / PX
assert abs(IMG_H * PX - 2.0 * LAT_M) < 1e-6


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def raster_ego(boxes, x, y, yaw, radius=4.0):
    """Occupancy 48×80, axle at (RCX, RCY), +x col+, +y (left) row-."""
    packed = prim.pack_boxes(boxes, x, y, radius=radius)
    cols, rows = np.meshgrid(np.arange(IMG_W, dtype=np.float32),
                             np.arange(IMG_H, dtype=np.float32))
    dx_e = (cols - np.float32(RCX)) * np.float32(PX)
    dy_e = (np.float32(RCY) - rows) * np.float32(PX)
    c = np.float32(math.cos(yaw))
    s = np.float32(math.sin(yaw))
    wx = np.float32(x) + c * dx_e - s * dy_e
    wy = np.float32(y) + s * dx_e + c * dy_e
    if packed.shape[0] == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.float32)
    d = prim.clearance_batch(wx.ravel(), wy.ravel(), packed).reshape(IMG_H, IMG_W)
    return (d < np.float32(0.025)).astype(np.float32)


class UnstuckEnv:
    def __init__(self, hp=None, seed=0):
        self.hp = {
            "dt": 0.10, "ep_steps": 80, "max_v": 0.28, "max_w": 0.80,
            "v_rev": -0.10, "a_v": 0.70, "a_w": 2.40, "replan_s": 0.20,
            "plus_x_stop": 0.28, "unstuck_px_lo": 0.10, "unstuck_px_hi": 0.24,
            "unstuck_clear_px": 0.50, "unstuck_hold_s": 0.40, "coll_done": 40,
            "rebuild_every": 12, "goal_p": 0.55,
        }
        if hp:
            self.hp.update(hp)
        self.rng = random.Random(int(seed))
        self.np_rng = np.random.RandomState(int(seed) + 7)
        self.ep = 0
        self.boxes = []
        self.spawns = []
        self._rebuild()
        self.x = self.y = self.yaw = 0.0
        self.plus_x = 8.0
        self.exe_v = self.exe_w = 0.0
        self.last_act = np.zeros(2, dtype=np.float32)
        self.clear_s = 0.0
        self.prev_plus_x = 9.0
        self.goal = None
        self.steps = 0
        self.path_m = 0.0
        self.collisions = 0
        self.coll_streak = 0
        self.cut_reason = ""

    def _rebuild(self):
        layout = make_layout(self.rng.randint(0, 1_000_000))
        self.boxes = list(layout["walls"]) + list(layout["furniture"])
        self.spawns = list(layout["spawns"]) or [(0.0, 0.0)]

    def _packed(self, x=None, y=None):
        return prim.pack_boxes(
            self.boxes,
            self.x if x is None else x,
            self.y if y is None else y,
            radius=5.5,
        )

    def _px(self, x=None, y=None, yaw=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        yaw = self.yaw if yaw is None else yaw
        xx = np.array([x], dtype=np.float32)
        yy = np.array([y], dtype=np.float32)
        th = np.array([yaw], dtype=np.float32)
        return float(prim.plus_x_batch(xx, yy, th, self._packed(x, y), nose=0.15)[0])

    def _ok(self, x, y, yaw, pad=0.08):
        xx = np.array([x], dtype=np.float32)
        yy = np.array([y], dtype=np.float32)
        th = np.array([yaw], dtype=np.float32)
        return bool(prim.foot_ok_batch(xx, yy, th, self._packed(x, y), pad=pad)[0])

    def _look(self, yaw_off, dist=1.2):
        th = self.yaw + yaw_off
        xx = np.array([self.x], dtype=np.float32)
        yy = np.array([self.y], dtype=np.float32)
        aa = np.array([th], dtype=np.float32)
        return float(prim.plus_x_batch(xx, yy, aa, self._packed(self.x, self.y), nose=0.15)[0])

    def _pick_pin(self):
        """Free pose, then walk the heading until plus-x is in the throttle band."""
        stop = float(self.hp["plus_x_stop"])
        lo = float(self.hp["unstuck_px_lo"])
        hi = min(float(self.hp["unstuck_px_hi"]), stop - 0.02)
        opts = list(self.spawns)
        self.rng.shuffle(opts)
        fallback = None
        for _ in range(80):
            if opts:
                bx, by = opts.pop()
            else:
                bx = self.rng.uniform(-2.6, 2.6)
                by = self.rng.uniform(-6.2, 6.2)
            x = bx + self.rng.uniform(-0.35, 0.35)
            y = by + self.rng.uniform(-0.35, 0.35)
            if not self._ok(x, y, 0.0, pad=0.22):
                continue
            yaws = [self.rng.uniform(-math.pi, math.pi)]
            yaws.extend(-math.pi + (2.0 * math.pi * i) / 10.0 for i in range(10))
            for yaw in yaws:
                if not self._ok(x, y, yaw, pad=0.20):
                    continue
                px = self._px(x, y, yaw)
                if not (0.25 <= px <= 5.5):
                    continue
                c, s = math.cos(yaw), math.sin(yaw)
                # Walk forward until the nose is in [lo, hi].
                walked = False
                for dist in (i * 0.06 for i in range(0, int(px / 0.06) + 1)):
                    nx = x + c * dist
                    ny = y + s * dist
                    if not self._ok(nx, ny, yaw, pad=0.16):
                        break
                    px2 = self._px(nx, ny, yaw)
                    if lo <= px2 <= hi:
                        self.x, self.y, self.yaw = nx, ny, yaw
                        return True
                    if fallback is None and 0.08 <= px2 <= stop:
                        fallback = (nx, ny, yaw)
                    walked = True
                if not walked:
                    continue
        if fallback is not None:
            self.x, self.y, self.yaw = fallback
            return True
        return False

    def _pick_goal(self):
        if self.rng.random() >= float(self.hp.get("goal_p", 0.55)):
            self.goal = None
            return
        for _ in range(24):
            gx = self.rng.uniform(-2.6, 2.6)
            gy = self.rng.uniform(-6.2, 6.2)
            if math.hypot(gx - self.x, gy - self.y) < 1.4:
                continue
            if self._ok(gx, gy, 0.0, pad=0.28):
                self.goal = (gx, gy)
                return
        self.goal = None

    def _observe(self):
        occ = raster_ego(self.boxes, self.x, self.y, self.yaw)
        self.plus_x = self._px()
        left = self._look(0.70)
        right = self._look(-0.70)
        stop = float(self.hp["plus_x_stop"])
        bearing = 0.0
        if self.goal is not None:
            bearing = _wrap(math.atan2(self.goal[1] - self.y, self.goal[0] - self.x) - self.yaw)
        max_v = max(float(self.hp["max_v"]), 1e-6)
        max_w = max(float(self.hp["max_w"]), 1e-6)
        vec = np.array([
            np.float32(np.clip(self.exe_v / max_v, -1.0, 1.0)),
            np.float32(np.clip(self.exe_w / max_w, -1.0, 1.0)),
            np.float32(min(1.0, self.plus_x / 2.0)),
            np.float32(min(1.0, left / 1.2)),
            np.float32(min(1.0, right / 1.2)),
            np.float32(1.0 if self.plus_x < stop else 0.0),
            np.float32(math.sin(bearing)),
            np.float32(math.cos(bearing)),
        ], dtype=np.float32)
        assert vec.shape[0] == VEC_DIM
        img = occ[None, :, :].astype(np.float32)
        return {"img": img, "vec": vec, "plus_x": self.plus_x,
                "left": left, "right": right, "bearing": bearing}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(int(seed))
        self.ep += 1
        if self.ep == 1 or (self.ep % int(self.hp.get("rebuild_every", 12))) == 0:
            self._rebuild()
        self.exe_v = self.exe_w = 0.0
        self.last_act[:] = 0
        self.steps = 0
        self.path_m = 0.0
        self.collisions = 0
        self.coll_streak = 0
        self.clear_s = 0.0
        self.cut_reason = ""
        for _ in range(8):
            if self._pick_pin():
                px = self._px()
                if 0.08 <= px <= 0.30:
                    break
            self._rebuild()
        else:
            if not self._pick_pin():
                self.x, self.y, self.yaw = self.spawns[0][0], self.spawns[0][1], 0.0
        self._pick_goal()
        obs = self._observe()
        self.prev_plus_x = float(self.plus_x)
        return obs, {"plus_x": self.plus_x, "goal": self.goal}

    def step(self, action):
        hp = self.hp
        dt = float(hp["dt"])
        max_v = float(hp["max_v"])
        max_w = float(hp["max_w"])
        v_rev = float(hp["v_rev"])
        stop = float(hp["plus_x_stop"])
        clear_px = float(hp["unstuck_clear_px"])
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:2], -1.0, 1.0)
        v = float(np.clip(a[0] * max_v, v_rev, max_v))
        w = float(np.clip(a[1] * max_w, -max_w, max_w))
        v_lo, v_hi, w_lo, w_hi = prim.reachable_vw(self.exe_v, self.exe_w, hp)
        v = float(np.clip(v, v_lo, v_hi))
        w = float(np.clip(w, w_lo, w_hi))
        prev_px = float(self.prev_plus_x)
        if float(self.plus_x) < stop:
            v = 0.0
        nx = self.x + v * math.cos(self.yaw) * dt
        ny = self.y + v * math.sin(self.yaw) * dt
        nyaw = _wrap(self.yaw + w * dt)
        collided = False
        moved = False
        if abs(v) > 0.01 or abs(w) > 0.02:
            if self._ok(nx, ny, nyaw, pad=0.06):
                dist = math.hypot(nx - self.x, ny - self.y)
                if dist > 0.002:
                    self.path_m += dist
                    self.x, self.y, self.yaw = nx, ny, nyaw
                    moved = True
                elif abs(w) > 0.04:
                    self.yaw = nyaw
                else:
                    collided = True
                    w = 0.0
            elif abs(w) > 0.04 and self._ok(self.x, self.y, nyaw, pad=0.04):
                self.yaw = nyaw
            else:
                collided = True
                w = 0.0
        if collided:
            self.collisions += 1
            self.coll_streak += 1
        else:
            self.coll_streak = 0
        self.exe_v, self.exe_w = float(v), float(w)
        self.last_act[:] = a
        self.steps += 1
        obs = self._observe()
        px = float(self.plus_x)
        throttled = px < stop
        if throttled:
            self.clear_s = 0.0
        else:
            self.clear_s += dt
        dpx = px - prev_px
        rew = 0.90 * float(np.clip(dpx / 0.35, -1.0, 1.0))
        if throttled:
            rew += 0.16 * abs(w) / max(max_w, 1e-6)
            if abs(w) < 0.08:
                rew -= 0.22
            if self.goal is not None:
                br = float(obs["bearing"])
                rew += 0.12 * math.sin(br) * np.sign(w)
        else:
            rew += 0.32
        if collided:
            rew -= 0.40
        done = False
        win = False
        self.cut_reason = ""
        if self.clear_s >= float(hp["unstuck_hold_s"]) and px >= clear_px:
            done = True
            win = True
            rew += 1.70
            self.cut_reason = "clear"
        elif self.coll_streak >= int(hp["coll_done"]):
            done = True
            rew -= 0.55
            self.cut_reason = "crash"
        elif self.steps >= int(hp["ep_steps"]):
            done = True
            self.cut_reason = "time"
        self.prev_plus_x = px
        info = {
            "plus_x": px, "v": v, "w": w, "collided": collided,
            "throttled": 1.0 if throttled else 0.0, "clear_s": self.clear_s,
            "unstuck": 1.0 if win else 0.0, "cut": self.cut_reason,
            "path_m": self.path_m, "goal": self.goal, "done": done,
            "win": win, "x": self.x, "y": self.y,
        }
        return obs, float(rew), bool(done), info
