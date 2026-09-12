#!/usr/bin/env python3
"""Insect-explore gym around Newton: raw depth + ego crop + audio.

Actor never sees room names, map waypoints, or fused maps. Privileged
coverage is eval-only. Kinematic step has plus-x / box stop, no XYZ slips.

5 Hz: sample 64 (v,w) in the 0.2 s reachable set from current speed, score a
5 s imagined rollout (plus-x throttle = stuck), execute the winner for 0.2 s.
Episodes 30 s. Random spawn yaw. No sticky curve follow.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

_SIM = Path(__file__).resolve().parent
_REPO = _SIM.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visit_wander import (  # noqa: E402
    FLOOR_X0, FLOOR_X1, FLOOR_Y0, FLOOR_Y1, ROOMS, room_of, _point_in_boxes,
)
from house_mesh import plus_x_against_boxes  # noqa: E402
import explore_prim as prim  # noqa: E402

# Opening punched in the y≈2.43 partition (and far-north lip if present).
DOOR_GAPS = (
    dict(x0=-0.55, x1=0.95, y0=2.18, y1=2.72),
    dict(x0=-0.55, x1=0.95, y0=5.28, y1=5.78),
)

SPAWN_XY = (
    (-2.20, -5.10), (1.10, -5.40), (0.20, -4.40),
    (-2.30, -0.70), (0.05, 0.15), (2.20, -1.40),
    (-2.05, 3.10), (1.70, 3.20), (0.10, 6.15),
    (-2.40, -3.40), (2.30, -4.80),
)

def _wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _bezier_pt(ctrl, t):
    t = float(np.clip(t, 0.0, 1.0))
    u = 1.0 - t
    p0, p1, p2, p3 = ctrl
    return (u ** 3) * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + (t ** 3) * p3


def _bezier_poly(ctrl, n=20):
    ts = np.linspace(0.0, 1.0, int(n))
    return np.stack([_bezier_pt(ctrl, t) for t in ts], axis=0)


def _aabb_clearance(px, py, boxes):
    """Signed distance to the nearest yawed AABB. Negative if inside."""
    best = 9.0
    for b in boxes:
        yaw = float(b.get("yaw", 0.0) or 0.0)
        cb, sb = math.cos(yaw), math.sin(yaw)
        dx = float(px) - float(b["cx"])
        dy = float(py) - float(b["cy"])
        lx = cb * dx + sb * dy
        ly = -sb * dx + cb * dy
        ax = abs(lx) - float(b["hx"])
        ay = abs(ly) - float(b["hy"])
        if ax <= 0.0 and ay <= 0.0:
            d = max(ax, ay)
        else:
            d = math.hypot(max(ax, 0.0), max(ay, 0.0))
        if d < best:
            best = d
    return float(best)


def _bezier_tangent(ctrl, t=1.0):
    """Unit dB/dt of a cubic Bézier. t=1 is the arrival heading 3(P3−P2)."""
    t = float(np.clip(t, 0.0, 1.0))
    u = 1.0 - t
    p0, p1, p2, p3 = ctrl
    d = (3.0 * u * u * (p1 - p0)
         + 6.0 * u * t * (p2 - p1)
         + 3.0 * t * t * (p3 - p2))
    n = float(np.hypot(d[0], d[1]))
    if n < 1e-5:
        d = p3 - p2
        n = float(np.hypot(d[0], d[1]))
    if n < 1e-5:
        d = p3 - p0
        n = float(np.hypot(d[0], d[1]))
    if n < 1e-5:
        return np.array((1.0, 0.0), dtype=np.float64)
    return d / n


def _aabb_ray_hit(ox, oy, dx, dy, boxes, inflate=0.0):
    """First hit distance along a 2D ray vs inflated yawed AABBs. 0 if inside."""
    best = 9e9
    for b in boxes:
        yaw = float(b.get("yaw", 0.0) or 0.0)
        cb, sb = math.cos(yaw), math.sin(yaw)
        px = float(ox) - float(b["cx"])
        py = float(oy) - float(b["cy"])
        lx = cb * px + sb * py
        ly = -sb * px + cb * py
        vx = cb * float(dx) + sb * float(dy)
        vy = -sb * float(dx) + cb * float(dy)
        hx = float(b["hx"]) + inflate
        hy = float(b["hy"]) + inflate
        if abs(lx) <= hx and abs(ly) <= hy:
            return 0.0
        tmin, tmax = -1e9, 1e9
        miss = False
        for orig, vel, h in ((lx, vx, hx), (ly, vy, hy)):
            if abs(vel) < 1e-12:
                if abs(orig) > h:
                    miss = True
                    break
                continue
            t1 = (-h - orig) / vel
            t2 = (h - orig) / vel
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmax < tmin:
                miss = True
                break
        if miss or tmax < 0.0:
            continue
        hit = tmin if tmin > 0.0 else 0.0
        if hit < best:
            best = hit
    return float(best)


def _continue_wall_cost(ctrl, boxes, hp):
    """Penalize an arrival heading that would hit if driven 3 s further.

    Tangent at P3, raycast. t_hit >= curve_cont_s → 0. Else
    coef * (1 − t_hit / horizon). Parallel-to-wall or open-ahead is free;
    head-on into a wall at P3 scores ~coef.
    """
    if ctrl is None or len(ctrl) < 4 or not boxes:
        return 0.0, None
    coef = float(hp.get("curve_cont_coef", 0.90))
    if coef <= 0.0:
        return 0.0, None
    horizon_s = float(hp.get("curve_cont_s", 3.0))
    max_v = max(float(hp.get("max_v", 0.28)), 1e-3)
    inflate = float(hp.get("curve_hit_m", 0.14))
    p3 = np.asarray(ctrl[-1], dtype=np.float64)
    tan = _bezier_tangent(ctrl, 1.0)
    d_hit = _aabb_ray_hit(float(p3[0]), float(p3[1]), float(tan[0]), float(tan[1]),
                          boxes, inflate)
    horizon_m = max_v * horizon_s
    vis_d = horizon_m if (not math.isfinite(d_hit) or d_hit > 1e8) else min(d_hit, horizon_m)
    ray = (p3, p3 + tan * vis_d)
    if not math.isfinite(d_hit) or d_hit > 1e8:
        return 0.0, ray
    t_hit = d_hit / max_v
    if t_hit >= horizon_s:
        return 0.0, ray
    return float(coef * (1.0 - t_hit / max(horizon_s, 1e-3))), ray


def _poly_wall_cost(poly, boxes, hp):
    """Penalize a committed path that hugs or intersects walls.

    clear_m is the desired centerline clearance (robot radius + margin).
    A path down a corridor middle scores ~0; a path into a wall scores ~1+.
    """
    if poly is None or len(poly) < 2 or not boxes:
        return 0.0
    safe = float(hp.get("curve_clear", 0.42))
    coef = float(hp.get("curve_wall_coef", 0.55))
    hit_coef = float(hp.get("curve_hit_coef", 0.80))
    graze = float(hp.get("curve_hit_m", 0.14))
    close_sum = 0.0
    n_hit = 0
    n = len(poly)
    min_d = 9.0
    end_d = 9.0
    for i, p in enumerate(poly):
        d = _aabb_clearance(float(p[0]), float(p[1]), boxes)
        close_sum += max(0.0, 1.0 - d / max(safe, 1e-3))
        if d < graze:
            n_hit += 1
        if d < min_d:
            min_d = d
        if i == n - 1:
            end_d = d
    mean_close = close_sum / n
    worst = max(0.0, 1.0 - min_d / max(safe, 1e-3))
    end_close = max(0.0, 1.0 - end_d / max(safe, 1e-3))
    # Worst point + endpoint matter more than the open approach, so a
    # curve that only kisses the wall at P3 is still expensive.
    return (
        coef * (0.25 * mean_close + 0.40 * worst + 0.35 * end_close)
        + hit_coef * (n_hit / n)
    )


DEPTH_H, DEPTH_W = 48, 80
EGO_H, EGO_W = 48, 64
RS1_Z = 0.86
EGO_X0, EGO_X1 = 0.05, 2.45
EGO_Y0, EGO_Y1 = -1.28, 1.28

# World-aligned visit window: 32×32 @ 10 cm into an imaginary global heatmap.
# Robot occupies the center cell. Translation slides the window; yaw does not
# rotate it. Heat that leaves the 3.2 m view is forgotten.
VISIT_N = 32
VISIT_M = 3.2
VISIT_CELL = VISIT_M / VISIT_N
VISIT_CX = VISIT_N // 2
VISIT_CY = VISIT_N // 2
TRACK_M = 0.23  # Kevin wheel track; |v_l − v_r| = |w| * TRACK_M


def _shift_forget(a, dcol, drow):
    """Translate a heatmap; incoming edge is 0, outgoing heat is dropped."""
    out = np.zeros_like(a)
    h, w = a.shape
    src_r0 = max(0, -drow)
    src_r1 = min(h, h - drow)
    dst_r0 = max(0, drow)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    src_c0 = max(0, -dcol)
    src_c1 = min(w, w - dcol)
    dst_c0 = max(0, dcol)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = a[src_r0:src_r1, src_c0:src_c1]
    return out


def _box_in_gaps(b, gaps=DOOR_GAPS) -> bool:
    cx, cy = float(b["cx"]), float(b["cy"])
    for g in gaps:
        if g["x0"] <= cx <= g["x1"] and g["y0"] <= cy <= g["y1"]:
            return True
    return False


def _xy_in_gaps(x, y, gaps=DOOR_GAPS) -> bool:
    for g in gaps:
        if g["x0"] <= x <= g["x1"] and g["y0"] <= y <= g["y1"]:
            return True
    return False


def _quat_rot(q, v):
    x, y, z, w = [float(t) for t in q]
    qv = np.array([x, y, z], dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    uv = np.cross(qv, vv)
    uuv = np.cross(qv, uv)
    return vv + 2.0 * (w * uv + uuv)


def mask_door_depth(d, origin, quat, dx, dy, open_door: bool):
    """Zero visual hits whose world XY sits in an open door AABB."""
    if not open_door:
        return d
    d = np.asarray(d, dtype=np.float32)
    if d.size == 0:
        return d
    out = d.copy()
    ok = np.isfinite(out) & (out > 0.04)
    if not np.any(ok):
        return out
    rdx = np.asarray(dx, dtype=np.float32)
    rdy = np.asarray(dy, dtype=np.float32)
    rdz = np.full_like(rdx, -1.0)
    nrm = np.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
    nrm = np.maximum(nrm, 1e-6)
    lx, ly, lz = rdx / nrm, rdy / nrm, rdz / nrm
    look = _quat_rot(quat, np.array([0.0, 0.0, -1.0]))
    wx = _quat_rot(quat, np.stack([lx.ravel(), ly.ravel(), lz.ravel()], axis=1))
    wx = wx.reshape(d.shape + (3,))
    along = np.abs(wx[..., 0] * look[0] + wx[..., 1] * look[1] + wx[..., 2] * look[2])
    along = np.maximum(along, 0.15)
    t = out / along.astype(np.float32)
    ox, oy = float(origin[0]), float(origin[1])
    hx = ox + t * wx[..., 0]
    hy = oy + t * wx[..., 1]
    hole = ok & (
        ((hx >= -0.55) & (hx <= 0.95) & (hy >= 2.18) & (hy <= 2.72))
        | ((hx >= -0.55) & (hx <= 0.95) & (hy >= 5.28) & (hy <= 5.78))
    )
    out[hole] = 0.0
    return out


def _resize(img, w, h):
    import cv2
    return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)


def ego_from_rs1(verts, h=EGO_H, w=EGO_W):
    """Body-frame height crop from RS1 verts. Camera X = -body X, rs_y = +body Y."""
    grid = np.zeros((h, w), dtype=np.float32)
    v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    if v.shape[0] < 8:
        return grid
    ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 0.04)
    if not np.any(ok):
        return grid
    bx = -v[ok, 0]
    by = v[ok, 1]
    hz = np.clip(RS1_Z - v[ok, 2], 0.0, 1.6)
    ix = ((bx - EGO_X0) / (EGO_X1 - EGO_X0) * w).astype(np.int32)
    iy = ((by - EGO_Y0) / (EGO_Y1 - EGO_Y0) * h).astype(np.int32)
    keep = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    if not np.any(keep):
        return grid
    # last-write max via reverse sort
    ix, iy, hz = ix[keep], iy[keep], hz[keep]
    order = np.argsort(hz)
    grid[iy[order], ix[order]] = hz[order]
    return grid


def _seg_blocked(ax, ay, bx, by, boxes, n=7) -> bool:
    for i in range(1, n):
        t = i / float(n)
        px = ax + (bx - ax) * t
        py = ay + (by - ay) * t
        if _point_in_boxes(px, py, boxes, pad=0.02):
            return True
    return False


class InsectEnv:
    """One Kevin, Warp RS1/RS2 raycast on the 3090, kinematic (v,w)."""

    def __init__(self, hparams: dict, seed: int = 1):
        os.environ.setdefault("DISPLAY", ":1")
        self.hp = dict(hparams)
        self.rng = random.Random(int(seed))
        self.np_rng = np.random.default_rng(int(seed) + 9)
        self.device = str(self.hp.get("device", "cuda:0"))
        self._build(int(seed))
        self.t_sim = 0.0
        self.steps = 0
        self.ep = 0
        self.trail = []
        self.last_obs = None
        self.last_info = {}
        self.cov = np.zeros((self.cov_h, self.cov_w), dtype=np.uint8)
        self.rooms_seen = set()
        self.path_m = 0.0
        self.collisions = 0
        self.coll_streak = 0
        self.yaw_only_streak = 0
        self.door_open = False
        self.last_act = np.zeros(4, dtype=np.float32)
        self.still_s = 0.0
        self.jerk_sum = 0.0
        self.disp_win = 0.0
        self.turn_win = 0.0
        self.spin_win = 0.0
        self._reset_pi()
        self._reset_plant()
        self._reset_cost()
        self._reset_visit()
        self._reset_curve()
        self.talkers = []
        print(
            "insect 5Hz (v,w) plant=%.2f rand_house=%.2f replan=%.2fs score=%.1fs"
            % (float(self.hp.get("plant", 0.0)), float(self.hp.get("rand_house", 1.0)),
               float(self.hp.get("replan_s", 0.20)), float(self.hp.get("score_s", 5.0))),
            flush=True,
        )
        self.chat = ""
        self.build_t = time.time()

    def _build(self, seed: int):
        import warp as wp
        import newton
        from newton_kevin import (
            add_house, add_random_obstacles, build_kevin, DEFAULT_STL,
        )
        from newton_drive import SimRsCameras, _kin_ok, _nearby_boxes, _yaw_quat
        from sim_people import SimPeople
        from rand_house import make_layout, boxes_to_mesh, add_boxes_to_builder

        self._kin_ok = _kin_ok
        self._nearby_boxes = _nearby_boxes
        self._yaw_quat = _yaw_quat

        wp.init()
        builder = newton.ModelBuilder()
        use_rand = float(self.hp.get("rand_house", 1.0)) >= 0.5
        self.layout_rooms = ()
        self.spawn_xy = list(SPAWN_XY)
        caster_verts, caster_faces = None, None
        clutter = str(self.hp.get("clutter", "random") or "random")
        if use_rand:
            layout = make_layout(int(seed))
            self.layout_rooms = tuple(layout["rooms"])
            self.spawn_xy = list(layout["spawns"]) or list(SPAWN_XY)
            self.mins, self.maxs = layout["mins"], layout["maxs"]
            add_boxes_to_builder(builder, layout["walls"], "wall")
            add_boxes_to_builder(builder, layout["furniture"], "furn")
            self.wall_all = list(layout["walls"])
            self.furniture = list(layout["furniture"])
            self.clutter_spec = "rand_house"
            caster_verts, caster_faces = boxes_to_mesh(
                layout["walls"] + layout["furniture"]
            )
            spawn = self.spawn_xy[0]
            self.people = SimPeople(seed=int(seed))
            folk = []
            for name, xy in layout["people"][:4]:
                folk.append({
                    "name": name, "xy": xy, "r": 0.30, "mood": True,
                    "cool": 0.0, "chat_left": 0.0, "said": "",
                })
            if folk:
                self.people.folk = folk
            build_kevin(builder, start_xy=spawn, yaw=0.0)
            print(
                "rand house rooms=%d walls=%d furn=%d people=%d"
                % (len(self.layout_rooms), len(self.wall_all),
                   len(self.furniture), len(self.people.folk)),
                flush=True,
            )
        else:
            stl = Path(DEFAULT_STL)
            mins, maxs = add_house(builder, stl)
            self.mins, self.maxs = mins, maxs
            spawn = (float(mins[0]) + 1.2, float(mins[1]) + 1.2)
            self.people = SimPeople(seed=int(seed))
            add_random_obstacles(
                builder, mins, maxs,
                seed=int(seed),
                static=True,
                clutter=clutter,
                spawn=spawn,
                people_xy=[p["xy"] for p in self.people.folk],
            )
            build_kevin(builder, start_xy=spawn, yaw=0.0)
            caster_verts = getattr(add_house, "verts", None)
            caster_faces = getattr(add_house, "faces", None)
            self.wall_all = list(getattr(add_house, "wall_boxes", []) or [])
            self.furniture = list(getattr(add_random_obstacles, "placed", []) or [])
            self.clutter_spec = str(getattr(add_random_obstacles, "spec", clutter))
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.cams = SimRsCameras(self.model, caster_verts, caster_faces)
        cell = 0.40
        self.cov_w = int(round((FLOOR_X1 - FLOOR_X0) / cell))
        self.cov_h = int(round((FLOOR_Y1 - FLOOR_Y0) / cell))
        self.cov_cell = cell
        print(
            "insect env walls=%d furniture=%d clutter=%s caster=%s %dx%d"
            % (
                len(self.wall_all), len(self.furniture), self.clutter_spec,
                self.cams.caster is not None, self.cams.h, self.cams.w,
            ),
            flush=True,
        )

    def kin_boxes(self):
        walls = self.wall_all
        if self.door_open:
            walls = [b for b in walls if not _box_in_gaps(b)]
        return list(walls) + list(self.furniture)

    def _stamp_cov(self, x, y):
        ix = int((float(x) - FLOOR_X0) / self.cov_cell)
        iy = int((float(y) - FLOOR_Y0) / self.cov_cell)
        if 0 <= ix < self.cov_w and 0 <= iy < self.cov_h:
            if self.cov[iy, ix] == 0:
                self.cov[iy, ix] = 1
                return True
        return False

    def coverage_frac(self) -> float:
        n = int(self.cov.size)
        return float(self.cov.sum()) / max(1, n)

    def _reset_pi(self):
        dt = float(self.hp.get("dt", 0.10))
        win = max(8, int(round(float(self.hp.get("pi_window_s", 2.0)) / max(dt, 1e-3))))
        self.pi_hist = deque(maxlen=win)
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_th = 0.0
        self.disp_win = 0.0
        self.turn_win = 0.0
        self.spin_win = 0.0
        self.path_win = 0.0
        self.straight = 1.0
        self.w_ema = 0.0
        self.odom_path = 0.0

    def _pi_step(self, v, w, dt):
        """Wheel+IMU path integration. Straightness = chord / odom path."""
        self.odom_th += float(w) * dt
        self.odom_x += float(v) * math.cos(self.odom_th) * dt
        self.odom_y += float(v) * math.sin(self.odom_th) * dt
        self.odom_path += abs(float(v)) * dt
        self.pi_hist.append((self.odom_x, self.odom_y, self.odom_th, self.odom_path))
        self.w_ema = 0.85 * self.w_ema + 0.15 * abs(float(w))
        x0, y0, th0, p0 = self.pi_hist[0]
        self.disp_win = math.hypot(self.odom_x - x0, self.odom_y - y0)
        self.turn_win = abs(self.odom_th - th0)
        self.path_win = max(0.0, self.odom_path - p0)
        self.straight = self.disp_win / (self.path_win + 0.08)
        self.spin_win = self.turn_win / (self.disp_win + 0.12)
        return self.disp_win, self.turn_win, self.spin_win

    def _reset_plant(self):
        self.plant_mode = "cruise"
        self.plant_turn = 0.0
        self.plant_path = 0.0
        self.plant_lock = 0.0
        self.plant_fails = 0

    def _reset_cost(self):
        self.goal_yaw = float(getattr(self, "yaw", 0.0))
        self.stale_s = 0.0
        self.blocked_s = 0.0
        self.cost_sum = 0.0
        self.cut_reason = ""
        self.dw_sum = 0.0
        self.dv_sum = 0.0
        self.open_n = 0
        self.open_turn_sum = 0.0
        self.open_speed_sum = 0.0
        self.heading_err_sum = 0.0
        self._prev_v = 0.0
        self._prev_w = 0.0

    def _visit_gxgy(self):
        x = float(getattr(self, "x", 0.0))
        y = float(getattr(self, "y", 0.0))
        return int(round(x / VISIT_CELL)), int(round(y / VISIT_CELL))

    def _reset_visit(self):
        self.visit = np.zeros((VISIT_N, VISIT_N), dtype=np.float32)
        self._visit_gx, self._visit_gy = self._visit_gxgy()
        self._burn_visit()

    def _burn_visit(self):
        """Stamp the occupied world cell (always the window center)."""
        cy, cx = VISIT_CY, VISIT_CX
        blob = (
            (0, 0, 1.0),
            (-1, 0, 0.35), (1, 0, 0.35), (0, -1, 0.35), (0, 1, 0.35),
            (-1, -1, 0.15), (-1, 1, 0.15), (1, -1, 0.15), (1, 1, 0.15),
        )
        n = VISIT_N
        for di, dj, a in blob:
            y, x = cy + di, cx + dj
            if 0 <= y < n and 0 <= x < n:
                self.visit[y, x] += a

    def _visit_step(self):
        """Slide the 3.2 m window over a world-axis heatmap; yaw does not rotate it."""
        gx, gy = self._visit_gxgy()
        dgx = gx - self._visit_gx
        dgy = gy - self._visit_gy
        self._visit_gx, self._visit_gy = gx, gy
        if dgx or dgy:
            # Robot moved +dgx/+dgy world cells → heat shifts opposite.
            # Image: +x right (col+), +y up (row-).
            self.visit = _shift_forget(self.visit, -dgx, dgy)
        self._burn_visit()

    def _visit_here(self):
        return float(self.visit[VISIT_CY, VISIT_CX])

    def _visit_ahead(self, meters):
        """Heat along current heading on the world-aligned window."""
        col = VISIT_CX + (float(meters) * math.cos(self.yaw)) / VISIT_CELL
        row = VISIT_CY - (float(meters) * math.sin(self.yaw)) / VISIT_CELL
        r, c = int(round(row)), int(round(col))
        if 0 <= r < VISIT_N and 0 <= c < VISIT_N:
            return float(self.visit[r, c])
        return 0.0

    def _visit_ch(self):
        return np.clip(self.visit / 18.0, 0.0, 1.0).astype(np.float32)

    def _reset_curve(self):
        self.curve_pts = None
        self.curve_poly = None
        self.curve_beads = None
        self.curve_got = np.zeros(3, dtype=np.uint8)
        self.curve_t0 = 0.0
        self.curve_chord = 0.0
        self.curve_range = 0.0
        self.curve_next = 0
        self.curve_wins = 0
        self.curve_losses = 0
        self.curve_n = 0
        self.chord_win_sum = 0.0
        self.beads_sum = 0.0
        self.seg_done = False
        self.need_curve = True
        self.last_seg = ""
        self.seg_score = 0.0
        self.curve_wall_c = 0.0
        self.curve_cont_c = 0.0
        self.curve_cont_ray = None
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.exe_v = 0.0
        self.exe_w = 0.0
        self._prim_vw = None
        self.prim_alts = []
        self.prim_ms = 0.0
        self.prim_throt = 0.0
        self.prim_hot = 0.0
        self.prim_hot_hit = 0.0

    def _actor_vw(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:2], -1.0, 1.0)
        max_v = float(self.hp.get("max_v", 0.28))
        max_w = float(self.hp.get("max_w", 0.80))
        v_rev = float(self.hp.get("v_rev", -0.10))
        return (
            float(np.clip(a[0] * max_v, v_rev, max_v)),
            float(np.clip(a[1] * max_w, -max_w, max_w)),
        )

    def _score_boxes(self):
        boxes = list(self.kin_boxes())
        folk = getattr(getattr(self, "people", None), "folk", None) or ()
        for p in folk:
            boxes.append({
                "cx": float(p["xy"][0]), "cy": float(p["xy"][1]),
                "yaw": 0.0, "hx": 0.35, "hy": 0.35,
            })
        return boxes

    def _replan(self, action):
        """5 Hz: 64 reachable (v,w), 5 s score, pick max."""
        t0 = time.perf_counter()
        hp = self.hp
        box = prim.pack_boxes(self._score_boxes(), self.x, self.y)
        actor_vw = self._actor_vw(action)
        cmd = prim.sample_vw(self.exe_v, self.exe_w, actor_vw, self._prim_vw, hp)
        score, path, throt, trail, hot, hot_hit = prim.score_vw(
            cmd[:, 0], cmd[:, 1], self.x, self.y, self.yaw, box, hp,
            visit=self.visit,
        )
        score = score.copy()
        if self._prim_vw is not None:
            score[1] += np.float32(0.07)
        i = int(np.argmax(score))
        self.cmd_v = float(cmd[i, 0])
        self.cmd_w = float(cmd[i, 1])
        self._prim_vw = (self.cmd_v, self.cmd_w)
        self.curve_poly = trail[i]
        order = np.argsort(score)
        self.prim_alts = [trail[int(j)] for j in order[-8:-1][::-1]]
        self.prim_ms = (time.perf_counter() - t0) * 1e3
        self.prim_throt = float(throt[i])
        self.prim_hot = float(hot[i])
        self.prim_hot_hit = float(hot_hit[i])
        self.seg_score = float(score[i])
        self.curve_chord = float(path[i])
        self.curve_wall_c = 0.0
        self.curve_cont_c = float(throt[i])
        self.curve_t0 = self.t_sim
        self.need_curve = False
        self.seg_done = False
        max_v = max(float(hp.get("max_v", 0.28)), 1e-6)
        max_w = max(float(hp.get("max_w", 0.80)), 1e-6)
        self.last_act[0] = np.float32(np.clip(self.cmd_v / max_v, -1.0, 1.0))
        self.last_act[1] = np.float32(np.clip(self.cmd_w / max_w, -1.0, 1.0))
        self.last_act[2] = 0.0
        self.last_act[3] = 0.0
        self.curve_n += 1
        self.chord_win_sum += float(path[i])
        if path[i] >= 0.50 and throt[i] < 0.30:
            self.curve_wins += 1
            self.last_seg = "win"
        else:
            self.curve_losses += 1
            self.last_seg = "throt" if throt[i] > 0.35 else "short"
        return True

    def _commit_curve(self, action):
        return self._replan(action)

    def _curve_collect(self):
        if self.curve_beads is None:
            return 0
        rad = float(self.hp.get("curve_bead_r", 0.32))
        n_hit = 0
        while self.curve_next < 3:
            bx, by = self.curve_beads[self.curve_next]
            if math.hypot(self.x - bx, self.y - by) > rad:
                break
            self.curve_got[self.curve_next] = 1
            self.curve_next += 1
            n_hit += 1
        return n_hit

    def _track_curve(self, max_v, max_w, stop):
        """Pure-pursuit on the committed Bézier. plus-x stop still wins."""
        poly = self.curve_poly
        if poly is None or len(poly) < 2:
            return 0.0, 0.0
        d2 = (poly[:, 0] - self.x) ** 2 + (poly[:, 1] - self.y) ** 2
        i0 = int(np.argmin(d2))
        look = float(self.hp.get("curve_look", 0.45))
        acc = 0.0
        i1 = i0
        while i1 + 1 < len(poly) and acc < look:
            acc += float(np.hypot(poly[i1 + 1, 0] - poly[i1, 0],
                                  poly[i1 + 1, 1] - poly[i1, 1]))
            i1 += 1
        gx, gy = float(poly[i1, 0]), float(poly[i1, 1])
        err = _wrap_pi(math.atan2(gy - self.y, gx - self.x) - self.yaw)
        w = float(np.clip(2.6 * err, -max_w, max_w))
        v = max_v * max(0.0, math.cos(err)) ** 2
        if i0 >= len(poly) - 2:
            v *= 0.55
        if self.plus_x < stop:
            v = 0.0
        return float(v), float(w)

    def _try_move(self, v, w, dt, max_v, max_w):
        boxes = self.kin_boxes()
        near = self._nearby_boxes(self.x, self.y, boxes)
        nx = self.x + v * math.cos(self.yaw) * dt
        ny = self.y + v * math.sin(self.yaw) * dt
        nyaw = self.yaw + w * dt
        collided = False
        moved = False
        yaw_only = False
        yaw_cap = int(self.hp.get("yaw_only_max", 40))
        for p in self.people.folk:
            if math.hypot(nx - p["xy"][0], ny - p["xy"][1]) < 0.48 and v > 0.02:
                collided = True
                v = 0.0
                nx, ny = self.x, self.y
                break
        if not collided and (abs(v) > 0.01 or abs(w) > 0.02):
            if self._kin_ok(nx, ny, nyaw, near, pad=0.06):
                dist = math.hypot(nx - self.x, ny - self.y)
                if dist > 0.002:
                    self.path_m += dist
                    self.plant_path = float(getattr(self, "plant_path", 0.0)) + dist
                    if dist > 0.04:
                        self.plant_fails = 0
                    self.x, self.y, self.yaw = nx, ny, nyaw
                    moved = True
                    self.yaw_only_streak = 0
                elif abs(w) > 0.04 and self.yaw_only_streak < yaw_cap:
                    self.yaw = nyaw
                    self.yaw_only_streak += 1
                    yaw_only = True
                elif abs(w) > 0.04:
                    collided = True
                    w = 0.0
            elif (
                abs(w) > 0.04
                and self.yaw_only_streak < yaw_cap
                and self._kin_ok(self.x, self.y, nyaw, near, pad=0.04)
            ):
                self.yaw = nyaw
                self.yaw_only_streak += 1
                yaw_only = True
            else:
                collided = True
                w = 0.0
        if collided:
            self.collisions += 1
            self.coll_streak += 1
        elif not yaw_only:
            self.coll_streak = 0
        return v, w, collided, moved, yaw_only

    def _room_name(self, x, y):
        for name, x0, x1, y0, y1 in getattr(self, "layout_rooms", ()) or ():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return room_of(x, y)

    def _insect_plant(self, v, w_cmd, plus_block, dt, max_v):
        """Go straight. Discrete turns on contact or after ~2 m.

        Continuous yaw is a gyroscope. Actor still chooses speed and the
        sign of occasional reorients. No waypoints / room names.
        """
        if float(self.hp.get("plant", 1.0)) < 0.5:
            return v, w_cmd
        if not hasattr(self, "plant_mode"):
            self._reset_plant()
        self.plant_lock = max(0.0, self.plant_lock - dt)
        turn_w = 0.80

        if self.plant_mode == "turn":
            sgn = 1.0 if self.plant_turn >= 0.0 else -1.0
            w = sgn * turn_w
            v = 0.0
            self.plant_turn -= w * dt
            if self.plant_turn * sgn <= 0.05:
                self.plant_mode = "cruise"
                self.plant_turn = 0.0
                self.plant_path = 0.0
                self.plant_lock = 1.5
                return 0.16, 0.0
            return v, w

        v = float(np.clip(max(v, 0.12), 0.12, max_v))
        blocked = bool(plus_block) or self.coll_streak >= 1
        far = self.plant_path >= float(self.hp.get("reorient_m", 2.0))
        want = blocked or (
            self.plant_lock <= 0.0 and far and abs(w_cmd) > 0.22
        )
        if want:
            if abs(w_cmd) > 0.08:
                sgn = 1.0 if w_cmd >= 0.0 else -1.0
            else:
                sgn = 1.0 if self.rng.random() < 0.5 else -1.0
            if blocked:
                self.plant_fails = int(getattr(self, "plant_fails", 0)) + 1
            mag = self.rng.uniform(0.70, 1.25) if blocked else self.rng.uniform(0.40, 0.70)
            if self.plant_fails >= 3:
                mag = math.pi * 0.85
                sgn = 1.0 if self.rng.random() < 0.5 else -1.0
                self.plant_fails = 0
            self.plant_turn = sgn * mag
            self.plant_mode = "turn"
            self.plant_path = 0.0
            return 0.0, sgn * turn_w
        return v, 0.0

    def _pick_spawn(self):
        boxes = self.kin_boxes()
        opts = list(getattr(self, "spawn_xy", None) or SPAWN_XY)
        self.rng.shuffle(opts)
        pad = 0.34
        fallback = None
        for _ in range(96):
            if opts:
                bx, by = opts.pop()
            else:
                bx = self.rng.uniform(FLOOR_X0 + 1.0, FLOOR_X1 - 1.0)
                by = self.rng.uniform(FLOOR_Y0 + 1.0, FLOOR_Y1 - 1.0)
            x = bx + self.rng.uniform(-0.55, 0.55)
            y = by + self.rng.uniform(-0.55, 0.55)
            yaw = self.rng.uniform(-math.pi, math.pi)
            if not self._kin_ok(x, y, yaw, boxes, pad=pad):
                continue
            crowded = False
            for p in self.people.folk:
                if math.hypot(x - float(p["xy"][0]), y - float(p["xy"][1])) < 0.75:
                    crowded = True
                    break
            if crowded:
                continue
            px, crossed, _ = plus_x_against_boxes((x, y), yaw, boxes, nose_x=0.15)
            px = float(px) if math.isfinite(float(px)) else 0.0
            if crossed or px < 0.55:
                if fallback is None:
                    fallback = (x, y, yaw)
                continue
            return x, y, yaw
        if fallback is not None:
            return fallback
        fb = (getattr(self, "spawn_xy", None) or SPAWN_XY)[0]
        return fb[0], fb[1], 0.0

    def _audio(self, x, y, yaw):
        energy = 0.0
        ipd = 0.0
        toward = 0.0
        self.chat = ""
        self.talkers = []
        boxes = self.kin_boxes()
        c, s = math.cos(yaw), math.sin(yaw)
        for p in self.people.folk:
            if float(p.get("chat_left", 0.0)) <= 0.0 and not p.get("talking"):
                continue
            px, py = float(p["xy"][0]), float(p["xy"][1])
            dx, dy = px - x, py - y
            dist = math.hypot(dx, dy) + 0.25
            occ = 0.18 if _seg_blocked(x, y, px, py, boxes) else 1.0
            e = occ / (dist * dist)
            bearing = math.atan2(dy, dx)
            rel = bearing - yaw
            while rel > math.pi:
                rel -= 2.0 * math.pi
            while rel < -math.pi:
                rel += 2.0 * math.pi
            energy += e
            ipd += math.sin(rel) * e
            toward += math.cos(rel) * e
            self.talkers.append(p["name"])
            if p.get("said"):
                self.chat = p["said"]
        if energy > 1e-6:
            ipd /= energy
            toward /= energy
        return (
            np.float32(min(1.5, energy)),
            np.float32(np.clip(ipd, -1.0, 1.0)),
            np.float32(np.clip(toward, -1.0, 1.0)),
        )

    def _tick_talkers(self, dt):
        for p in self.people.folk:
            p["talking"] = bool(p.get("talking"))
            left = float(p.get("chat_left", 0.0))
            if left > 0.0:
                p["chat_left"] = left - dt
                p["talking"] = True
                if p["chat_left"] <= 0.0:
                    p["talking"] = False
                    p["said"] = ""
        if self.rng.random() < 0.012:
            p = self.rng.choice(self.people.folk)
            p["talking"] = True
            p["chat_left"] = self.rng.uniform(6.0, 18.0)
            p["said"] = "%s: hey over here" % p["name"]

    def _observe(self):
        pos = (self.x, self.y, 0.0)
        quat = self._yaw_quat(self.yaw)
        self.cams.wall_boxes = self.kin_boxes()
        verts1, verts2, _, _ = self.cams.grab(self.state, pos, quat)
        d1 = np.asarray(self.cams.last_d1, dtype=np.float32)
        d2 = np.asarray(self.cams.last_d2, dtype=np.float32)
        o1 = getattr(self.cams, "last_o1", pos)
        o2 = getattr(self.cams, "last_o2", pos)
        q1 = quat
        # RS1/RS2 camera quats are baked into grab; mask with body yaw is a
        # coarse hole but the AABB test on hit XY is what matters.
        d1 = mask_door_depth(d1, o1, q1, self.cams.dx, self.cams.dy, self.door_open)
        d2 = mask_door_depth(d2, o2, quat, self.cams.dx, self.cams.dy, self.door_open)
        noise = float(self.hp.get("depth_noise", 0.0) or 0.0)
        if noise > 1e-6:
            d2 = d2 * (1.0 + noise * self.np_rng.standard_normal(d2.shape).astype(np.float32))
            drop = self.np_rng.random(d2.shape) < min(0.04, noise)
            d2 = np.where(drop, 0.0, d2).astype(np.float32)
        plus_x, crossed, _ = plus_x_against_boxes(
            (self.x, self.y), self.yaw, self.kin_boxes(), nose_x=0.15,
        )
        self.plus_x = float(plus_x) if math.isfinite(float(plus_x)) else 8.0
        self.crossed = bool(crossed)
        depth = np.clip(d2, 0.0, 8.0) / 8.0
        depth = _resize(depth, DEPTH_W, DEPTH_H).astype(np.float32)
        ego = ego_from_rs1(verts1)
        ego_n = np.clip(ego / 1.2, 0.0, 1.0)
        ego_in = _resize(ego_n, DEPTH_W, DEPTH_H).astype(np.float32)
        vch = _resize(self._visit_ch(), DEPTH_W, DEPTH_H).astype(np.float32)
        stacked = np.stack([depth, ego_in, vch], axis=0)
        ae, ipd, toward = self._audio(self.x, self.y, self.yaw)
        replan_s = max(1e-3, float(self.hp.get("replan_s", 0.20)))
        seg_frac = float(np.clip((self.t_sim - self.curve_t0) / replan_s, 0.0, 1.0))
        beads = float(1.0 - getattr(self, "prim_throt", 0.0))
        vec = np.array(
            [
                self.last_act[0], self.last_act[1], ae, ipd, toward,
                np.float32(min(1.0, self.plus_x / 2.0)),
                np.float32(seg_frac),
                np.float32(beads),
                np.float32(min(1.0, self.curve_chord / 4.0)),
                np.float32(min(1.0, self._visit_here() / 18.0)),
            ],
            dtype=np.float32,
        )
        self.last_depth = depth
        self.last_ego = ego_n
        self.last_d2 = d2
        obs = {"img": stacked, "vec": vec}
        return obs, verts1, verts2

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(int(seed))
        self.ep += 1
        self.door_open = self.rng.random() < float(self.hp.get("door_open_p", 0.55))
        self._reset_visit()
        self._reset_curve()
        obs = None
        for _ in range(24):
            self.x, self.y, self.yaw = self._pick_spawn()
            obs, _, _ = self._observe()
            if float(self.plus_x) >= 0.55:
                break
        self.steps = 0
        self.t_sim = 0.0
        self.trail = [(self.x, self.y)]
        self.cov[:] = 0
        self.rooms_seen = set()
        self.path_m = 0.0
        self.collisions = 0
        self.last_act[:] = 0
        self.still_s = 0.0
        self.jerk_sum = 0.0
        self.coll_streak = 0
        self.yaw_only_streak = 0
        self._reset_pi()
        self._reset_plant()
        self._reset_cost()
        self._reset_visit()
        self._reset_curve()
        self.goal_yaw = self.yaw
        for p in self.people.folk:
            p["talking"] = False
            p["chat_left"] = 0.0
            p["said"] = ""
        if self.rng.random() < 0.65:
            p = self.rng.choice(self.people.folk)
            p["talking"] = True
            p["chat_left"] = self.rng.uniform(8.0, 22.0)
            p["said"] = "%s: hey over here" % p["name"]
        self._stamp_cov(self.x, self.y)
        self.rooms_seen.add(self._room_name(self.x, self.y))
        obs, _, _ = self._observe()
        self.last_obs = obs
        info = self._info(0.0, False)
        self.last_info = info
        return obs, info

    def _info(self, rew, done, extra=None):
        info = {
            "x": self.x, "y": self.y, "yaw": self.yaw,
            "plus_x": self.plus_x, "crossed": self.crossed,
            "door_open": self.door_open,
            "rooms": ",".join(sorted(self.rooms_seen)),
            "room": self._room_name(self.x, self.y),
            "n_rooms": len(self.rooms_seen),
            "coverage": self.coverage_frac(),
            "cov_cells": int(self.cov.sum()),
            "path_m": self.path_m,
            "cover_per_m": float(self.cov.sum()) / max(self.path_m, 0.40),
            "collisions": self.collisions,
            "reward": rew,
            "t_sim": self.t_sim,
            "chat": self.chat,
            "talkers": ",".join(self.talkers),
            "clutter": self.clutter_spec,
            "jerk": self.jerk_sum / max(1, self.steps),
            "visit_here": self._visit_here(),
            "visit_ahead": self._visit_ahead(0.6),
            "dw_mean": self.dw_sum / max(1, self.steps),
            "dv_mean": self.dv_sum / max(1, self.steps),
            "open_turn": self.open_turn_sum / max(1, self.open_n),
            "open_speed": self.open_speed_sum / max(1, self.open_n),
            "heading_err": self.heading_err_sum / max(1, self.steps),
            "cost": self.cost_sum,
            "cut": self.cut_reason,
            "stale_s": self.stale_s,
            "smooth_score": 1.0 / (1.0 + 4.0 * (self.jerk_sum / max(1, self.steps))
                                    + 1.2 * (self.dw_sum / max(1, self.steps))),
            "wide_score": float(self.cov.sum()) * max(1, len(self.rooms_seen)) / 12.0,
            "smooth_now": float(self.hp.get("smooth_now", self.hp.get("smooth_coef", 0.02))),
            "disp": self.disp_win,
            "turn": self.turn_win,
            "spin": self.spin_win,
            "straight": self.straight,
            "path_win": self.path_win,
            "done": done,
            "seg_done": bool(getattr(self, "seg_done", False)),
            "need_curve": bool(getattr(self, "need_curve", True)),
            "curve_chord": float(getattr(self, "curve_chord", 0.0)),
            "curve_beads": int(getattr(self, "curve_got", np.zeros(3)).sum()),
            "curve_wins": int(getattr(self, "curve_wins", 0)),
            "curve_losses": int(getattr(self, "curve_losses", 0)),
            "seg_score": float(getattr(self, "seg_score", 0.0)),
            "last_seg": getattr(self, "last_seg", ""),
            "mean_chord": float(getattr(self, "chord_win_sum", 0.0)) / max(1, int(getattr(self, "curve_n", 0))),
            "mean_beads": float(getattr(self, "beads_sum", 0.0)) / max(1, int(getattr(self, "curve_n", 0))),
            "win_frac": float(getattr(self, "curve_wins", 0)) / max(1, int(getattr(self, "curve_n", 0))),
            "curve_wall": float(getattr(self, "curve_wall_c", 0.0)),
            "curve_cont": float(getattr(self, "curve_cont_c", 0.0)),
            "prim_ms": float(getattr(self, "prim_ms", 0.0)),
            "prim_hot": float(getattr(self, "prim_hot", 0.0)),
            "prim_hot_hit": float(getattr(self, "prim_hot_hit", 0.0)),
            "exe_v": float(getattr(self, "exe_v", 0.0)),
            "exe_w": float(getattr(self, "exe_w", 0.0)),
        }
        if extra:
            info.update(extra)
        return info

    def step(self, action):
        hp = self.hp
        if float(hp.get("curve_mode", 1.0)) >= 0.5:
            return self._step_curve(action)
        return self._step_vw(action)

    def _step_curve(self, action):
        hp = self.hp
        dt = float(hp.get("dt", 0.10))
        max_v = float(hp.get("max_v", 0.32))
        max_w = float(hp.get("max_w", 0.80))
        stop = float(hp.get("plus_x_stop", 0.28))
        replan_s = float(hp.get("replan_s", 0.20))
        replanned = False
        if self.need_curve or (self.t_sim - self.curve_t0) >= replan_s - 1e-9:
            self._replan(action)
            replanned = True
        v, w = float(self.cmd_v), float(self.cmd_w)
        if float(getattr(self, "plus_x", 9.0)) < stop:
            v = 0.0
        path0 = float(self.path_m)
        v, w, collided, moved, yaw_only = self._try_move(v, w, dt, max_v, max_w)
        self.exe_v, self.exe_w = float(v), float(w)
        dist = max(0.0, float(self.path_m) - path0)
        self.t_sim += dt
        self.steps += 1
        self._pi_step(v, w, dt)
        self._visit_step()
        self._tick_talkers(dt)
        gained = 0
        if moved:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 400:
                self.trail = self.trail[-300:]
            if self._stamp_cov(self.x, self.y):
                gained = 1
            self.rooms_seen.add(self._room_name(self.x, self.y))
            self.still_s = 0.0
        else:
            self.still_s += dt
        obs, _, _ = self._observe()
        self.last_obs = obs
        elapsed = self.t_sim - self.curve_t0
        crash_cut = self.coll_streak >= int(hp.get("coll_done", 22))
        self.need_curve = elapsed >= replan_s - 1e-9
        self.seg_done = bool(self.need_curve or crash_cut)
        rew = float(hp.get("cover_coef", 0.4)) * gained * 0.15
        rew += float(hp.get("prim_step", 0.45)) * dist
        if replanned:
            rew += 0.05 * float(np.tanh(self.seg_score))
        if crash_cut:
            rew -= 0.4
        chord_n = float(np.clip(self.curve_chord / 4.0, 0.0, 1.0))
        done = False
        self.cut_reason = ""
        if crash_cut:
            done = True
            self.cut_reason = "crash"
        elif self.steps >= int(hp.get("ep_steps", 300)):
            done = True
            self.cut_reason = "time"
        self.cost_sum -= rew
        openness = float(np.clip((self.plus_x - 0.12) / 1.10, 0.0, 1.0))
        if openness > 0.55:
            self.open_n += 1
            self.open_turn_sum += abs(w)
            self.open_speed_sum += max(0.0, v)
        info = self._info(rew, done, {
            "v": v, "w": w, "collided": collided, "still": 1.0 if abs(v) < 0.04 else 0.0,
            "openness": openness, "gained": gained, "plant": "off",
            "seg_elapsed": elapsed, "chord_n": chord_n,
            "curve_wall": float(getattr(self, "curve_wall_c", 0.0)),
            "curve_cont": float(getattr(self, "prim_throt", 0.0)),
            "prim_ms": float(getattr(self, "prim_ms", 0.0)),
        })
        self.last_info = info
        return obs, float(rew), bool(done), info

    def _step_vw(self, action):
        hp = self.hp
        dt = float(hp.get("dt", 0.10))
        max_v = float(hp.get("max_v", 0.32))
        max_w = float(hp.get("max_w", 1.15))
        stop = float(hp.get("plus_x_stop", 0.28))
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a_cmd = np.clip(a[:2], -1.0, 1.0).astype(np.float32)
        prev = np.array(self.last_act, dtype=np.float32)
        ema = float(hp.get("action_ema_now", hp.get("action_ema", 0.0)) or 0.0)
        ema = float(np.clip(ema, 0.0, 0.85))
        if ema > 1e-6:
            a_cmd = (1.0 - ema) * a_cmd + ema * prev
        plant_on = float(hp.get("plant", 1.0)) >= 0.5
        yaw_lpf = 0.0 if plant_on else float(hp.get("yaw_lpf", 0.62) or 0.0)
        yaw_lpf = float(np.clip(yaw_lpf, 0.0, 0.92))
        if yaw_lpf > 1e-6:
            a_cmd[1] = (1.0 - yaw_lpf) * a_cmd[1] + yaw_lpf * prev[1]
        jerk = float(np.sum((a_cmd - prev) ** 2))
        self.jerk_sum += jerk
        v = float(a_cmd[0] * max_v)
        w_cmd = float(a_cmd[1] * max_w)
        plus_block = self.plus_x < stop
        if plus_block and v > 0.0:
            v = 0.0
        v, w = self._insect_plant(v, w_cmd, plus_block, dt, max_v)
        w = float(np.clip(w, -max(max_w, 0.85), max(max_w, 0.85)))
        boxes = self.kin_boxes()
        near = self._nearby_boxes(self.x, self.y, boxes)
        nx = self.x + v * math.cos(self.yaw) * dt
        ny = self.y + v * math.sin(self.yaw) * dt
        nyaw = self.yaw + w * dt
        collided = False
        moved = False
        yaw_only = False
        yaw_cap = int(hp.get("yaw_only_max", 40))
        for p in self.people.folk:
            if math.hypot(nx - p["xy"][0], ny - p["xy"][1]) < 0.48 and v > 0.02:
                collided = True
                v = 0.0
                nx, ny = self.x, self.y
                break
        if not collided and (abs(v) > 0.01 or abs(w) > 0.02):
            if self._kin_ok(nx, ny, nyaw, near, pad=0.06):
                dist = math.hypot(nx - self.x, ny - self.y)
                if dist > 0.002:
                    self.path_m += dist
                    self.plant_path = float(getattr(self, "plant_path", 0.0)) + dist
                    if dist > 0.04:
                        self.plant_fails = 0
                    self.x, self.y, self.yaw = nx, ny, nyaw
                    moved = True
                    self.yaw_only_streak = 0
                elif abs(w) > 0.04 and self.yaw_only_streak < yaw_cap:
                    self.yaw = nyaw
                    self.yaw_only_streak += 1
                    yaw_only = True
                elif abs(w) > 0.04:
                    collided = True
                    w = 0.0
            elif (
                abs(w) > 0.04
                and self.yaw_only_streak < yaw_cap
                and self._kin_ok(self.x, self.y, nyaw, near, pad=0.04)
            ):
                # Brief in-place yaw to look for a gap — not a pirouette.
                self.yaw = nyaw
                self.yaw_only_streak += 1
                yaw_only = True
            else:
                collided = True
                w = 0.0
        if collided:
            self.collisions += 1
            self.coll_streak += 1
        elif not yaw_only:
            self.coll_streak = 0
        self.last_act[0] = v / max(max_v, 1e-6)
        self.last_act[1] = w / max(max_w, 1e-6)
        self.t_sim += dt
        self.steps += 1
        self._pi_step(v, w, dt)
        self._visit_step()
        self._tick_talkers(dt)
        gained = 0
        if moved:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 400:
                self.trail = self.trail[-300:]
            if self._stamp_cov(self.x, self.y):
                gained = 1
            self.rooms_seen.add(self._room_name(self.x, self.y))
            self.still_s = 0.0
        else:
            self.still_s += dt

        obs, _, _ = self._observe()
        self.last_obs = obs
        toward = float(obs["vec"][4])
        audio_e = float(obs["vec"][2])
        still = 1.0 if abs(v) < 0.04 else 0.0
        st = float(np.clip(self.straight, 0.0, 1.2))
        stop_d = float(hp.get("plus_x_stop", 0.28))
        openness = float(np.clip((self.plus_x - 0.12) / 1.10, 0.0, 1.0))
        dead = 1.0 - openness
        if self.plus_x < stop_d or collided:
            self.blocked_s += dt
        else:
            if self.blocked_s > 0.55:
                self.goal_yaw = self.yaw
            self.blocked_s = 0.0
        heading_err = abs(_wrap_pi(self.yaw - self.goal_yaw))
        if gained:
            self.stale_s = 0.0
        else:
            self.stale_s += dt
        self.dw_sum += abs(w - self._prev_w)
        self.dv_sum += abs(v - self._prev_v)
        self._prev_v, self._prev_w = v, w
        if openness > 0.55:
            self.open_n += 1
            self.open_turn_sum += abs(w)
            self.open_speed_sum += max(0.0, v)
        self.heading_err_sum += heading_err

        turn_c = float(hp.get("turn_coef", 0.35)) * (abs(w) / max(max_w, 1e-6)) * openness
        ang_c = float(hp.get("ang_coef", 0.10)) * (abs(w) * TRACK_M) / max(max_v, 1e-6)
        slow_c = float(hp.get("slow_coef", 0.25)) * max(0.0, 0.82 * max_v - max(0.0, v)) / max(max_v, 1e-6) * openness
        stale_c = float(hp.get("stale_coef", 0.12)) * min(self.stale_s / 6.0, 1.6)
        crash_c = float(hp.get("collision_coef", 5.5)) * (1.0 if collided else 0.0)
        if self.coll_streak >= 2:
            crash_c *= 1.0 + 0.15 * min(self.coll_streak, 12)
        stuck_c = float(hp.get("stuck_coef", 0.70)) * still
        look_m = float(hp.get("look_m", 1.50))
        look_c = (
            float(hp.get("look_coef", 0.55))
            * max(0.0, v) / max(max_v, 1e-6)
            * max(0.0, 1.0 - self.plus_x / max(look_m, 1e-3)) ** 2
        )
        here = self._visit_here()
        ahead = 0.5 * self._visit_ahead(0.40) + 0.5 * self._visit_ahead(0.80)
        visit_c = float(hp.get("visit_coef", 0.35)) * math.tanh(here / 10.0)
        visit_fwd_c = (
            float(hp.get("visit_fwd_coef", 0.28))
            * max(0.0, v) / max(max_v, 1e-6)
            * math.tanh(ahead / 8.0)
        )
        commit_c = float(hp.get("commit_coef", 0.20)) * min(heading_err / math.pi, 1.0) * openness
        jerk_c = float(hp.get("jerk_coef", 0.04)) * jerk
        cost = (
            turn_c + ang_c + slow_c + stale_c + crash_c + stuck_c
            + look_c + visit_c + visit_fwd_c + commit_c + jerk_c
        )
        self.cost_sum += cost
        cover_b = float(hp.get("cover_coef", 1.4)) * gained
        audio_b = float(hp.get("audio_coef", 0.20)) * audio_e * max(0.0, toward) * max(0.0, v) * openness
        rew = cover_b + audio_b - cost
        if self.crossed or self.plus_x < 0.10:
            extra_c = 1.4
            self.cost_sum += extra_c
            rew -= extra_c

        ep_steps = int(hp.get("ep_steps", 600))
        coll_done = int(hp.get("coll_done", 22))
        cost_cut = float(hp.get("cost_cut", 80.0))
        cost_cut_min_s = float(hp.get("cost_cut_min_s", 6.0))
        done = False
        if self.coll_streak >= coll_done:
            done = True
            self.cut_reason = "crash"
        elif self.t_sim >= cost_cut_min_s and self.cost_sum >= cost_cut:
            done = True
            self.cut_reason = "cost"
        elif self.steps >= ep_steps:
            done = True
            self.cut_reason = "time"
        info = self._info(rew, done, {
            "v": v, "w": w, "collided": collided, "still": still,
            "openness": openness, "heading_err_now": heading_err, "gained": gained,
            "cost_step": cost, "turn_c": turn_c, "stale_c": stale_c,
            "look_c": look_c, "visit_c": visit_c, "ang_c": ang_c,
            "plant": "off", "straight": st, "dead": dead,
        })
        self.last_info = info
        return obs, float(rew), bool(done), info

    def people_draw(self):
        return self.people.folk
