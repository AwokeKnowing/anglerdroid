#!/usr/bin/env python3
"""64 (v,w) primitives: reachable in 0.2 s from current speed, scored to 5 s.

Everything is batched NumPy. One Python loop is the 5 s time axis (50 ticks).
"""
from __future__ import annotations

import numpy as np

FOOT = np.array(
    (
        (0.15, 0.0), (0.15, 0.16), (0.15, -0.16),
        (0.0, 0.16), (0.0, -0.16), (0.0, 0.0),
        (-0.18, 0.0), (-0.26, 0.0),
    ),
    dtype=np.float32,
)
FLOOR_X0, FLOOR_X1 = np.float32(-3.55), np.float32(3.75)
FLOOR_Y0, FLOOR_Y1 = np.float32(-7.45), np.float32(7.45)
N_V, N_W = 8, 8
N_PATHS = N_V * N_W
TRAIL_KEEP = 11
VISIT_CELL = np.float32(0.10)


def pack_boxes(boxes, x, y, radius=5.5):
    if not boxes:
        return np.zeros((0, 6), dtype=np.float32)
    x, y, radius = float(x), float(y), float(radius)
    rows = []
    for b in boxes:
        cx, cy = float(b["cx"]), float(b["cy"])
        hx, hy = float(b["hx"]), float(b["hy"])
        if abs(cx - x) + abs(cy - y) > radius + hx + hy:
            continue
        yaw = float(b.get("yaw", 0.0) or 0.0)
        rows.append((cx, cy, hx, hy, np.cos(yaw), np.sin(yaw)))
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def plus_x_batch(x, y, yaw, box, nose=0.15, inflate=0.01):
    n = int(x.shape[0])
    if box.shape[0] == 0:
        return np.full(n, np.float32(8.0), dtype=np.float32)
    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    return _ray_hit(x + c * np.float32(nose), y + s * np.float32(nose), c, s, box, inflate)


def _ray_hit(ox, oy, dx, dy, box, inflate):
    cx, cy = box[:, 0], box[:, 1]
    hx, hy = box[:, 2] + inflate, box[:, 3] + inflate
    cb, sb = box[:, 4], box[:, 5]
    px = ox[:, None] - cx[None, :]
    py = oy[:, None] - cy[None, :]
    lx = cb[None, :] * px + sb[None, :] * py
    ly = -sb[None, :] * px + cb[None, :] * py
    vx = cb[None, :] * dx[:, None] + sb[None, :] * dy[:, None]
    vy = -sb[None, :] * dx[:, None] + cb[None, :] * dy[:, None]
    inside = (np.abs(lx) <= hx[None, :]) & (np.abs(ly) <= hy[None, :])
    tmin = np.full(lx.shape, np.float32(-1.0e9))
    tmax = np.full(lx.shape, np.float32(1.0e9))
    miss = np.zeros(lx.shape, dtype=bool)
    for orig, vel, h in ((lx, vx, hx[None, :]), (ly, vy, hy[None, :])):
        par = np.abs(vel) < 1.0e-12
        miss |= par & (np.abs(orig) > h)
        den = np.where(par, np.float32(1.0), vel)
        t1 = (-h - orig) / den
        t2 = (h - orig) / den
        ta = np.minimum(t1, t2)
        tb = np.maximum(t1, t2)
        tmin = np.where(par, tmin, np.maximum(tmin, ta))
        tmax = np.where(par, tmax, np.minimum(tmax, tb))
        miss |= tmax < tmin
    miss |= tmax < 0.0
    hit = np.where(tmin > 0.0, tmin, np.float32(0.0))
    hit = np.where(inside, np.float32(0.0), hit)
    hit = np.where(miss & (~inside), np.float32(9.0e9), hit)
    return np.clip(np.min(hit, axis=1), 0.0, 8.0).astype(np.float32)


def clearance_batch(px, py, box):
    n = int(px.shape[0])
    if box.shape[0] == 0:
        return np.full(n, np.float32(9.0), dtype=np.float32)
    cx, cy, hx, hy, cb, sb = box.T
    dx = px[:, None] - cx[None, :]
    dy = py[:, None] - cy[None, :]
    lx = cb[None, :] * dx + sb[None, :] * dy
    ly = -sb[None, :] * dx + cb[None, :] * dy
    ax = np.abs(lx) - hx[None, :]
    ay = np.abs(ly) - hy[None, :]
    inside = (ax <= 0.0) & (ay <= 0.0)
    d = np.where(inside, np.maximum(ax, ay),
                 np.sqrt(np.maximum(ax, 0.0) ** 2 + np.maximum(ay, 0.0) ** 2))
    return np.min(d, axis=1).astype(np.float32)


def foot_ok_batch(x, y, yaw, box, pad=0.06):
    in_floor = (
        (x >= FLOOR_X0 + 0.20) & (x <= FLOOR_X1 - 0.20)
        & (y >= FLOOR_Y0 + 0.20) & (y <= FLOOR_Y1 - 0.20)
    )
    if box.shape[0] == 0:
        return in_floor
    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    fx, fy = FOOT[:, 0], FOOT[:, 1]
    wx = x[:, None] + c[:, None] * fx[None, :] - s[:, None] * fy[None, :]
    wy = y[:, None] + s[:, None] * fx[None, :] + c[:, None] * fy[None, :]
    n = int(x.shape[0])
    d = clearance_batch(wx.reshape(-1), wy.reshape(-1), box).reshape(n, FOOT.shape[0])
    return in_floor & np.all(d >= pad, axis=1)


def visit_heat_batch(x, y, x0, y0, visit):
    """World-aligned 3.2 m window: robot at center, +x col+, +y row-."""
    n = int(x.shape[0])
    if visit is None or visit.size == 0:
        return np.zeros(n, dtype=np.float32)
    h, w = visit.shape
    cy, cx = h // 2, w // 2
    col = np.rint(cx + (x - np.float32(x0)) / VISIT_CELL).astype(np.int32)
    row = np.rint(cy - (y - np.float32(y0)) / VISIT_CELL).astype(np.int32)
    ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    r = np.clip(row, 0, h - 1)
    c = np.clip(col, 0, w - 1)
    return np.where(ok, visit[r, c], np.float32(0.0)).astype(np.float32)


def reachable_vw(v, w, hp):
    """Axis-aligned (v,w) box reachable in replan_s from current speed."""
    dt = float(hp.get("replan_s", 0.20))
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


def sample_vw(v, w, actor_vw, prev_vw, hp):
    """8×8 grid on the 0.2 s reachable set. Actor + last winner occupy 0,1."""
    v_lo, v_hi, w_lo, w_hi = reachable_vw(v, w, hp)
    vs = np.linspace(v_lo, v_hi, N_V, dtype=np.float32)
    ws = np.linspace(w_lo, w_hi, N_W, dtype=np.float32)
    vv, ww = np.meshgrid(vs, ws, indexing="ij")
    cmd = np.stack((vv.ravel(), ww.ravel()), axis=1)
    av = np.clip(np.asarray(actor_vw, dtype=np.float32).reshape(2), [v_lo, w_lo], [v_hi, w_hi])
    cmd[0] = av
    if prev_vw is not None:
        cmd[1] = np.clip(np.asarray(prev_vw, dtype=np.float32).reshape(2),
                         [v_lo, w_lo], [v_hi, w_hi])
    return cmd.astype(np.float32)


def rollout_5s(v_cmd, w_cmd, x0, y0, yaw0, box, hp, visit=None):
    """Hold each (v,w) for score_s (5 s). plus-x zeros v (stuck), yaw may continue."""
    n = int(v_cmd.shape[0])
    dt = np.float32(hp.get("score_dt", hp.get("dt", 0.10)))
    horizon = float(hp.get("score_s", 5.0))
    steps = max(8, int(round(horizon / float(dt))))
    stop = np.float32(hp.get("plus_x_stop", 0.28))
    hot_s = np.float32(max(float(hp.get("visit_hot", 8.0)), 1e-3))
    x = np.full(n, np.float32(x0))
    y = np.full(n, np.float32(y0))
    yaw = np.full(n, np.float32(yaw0))
    path = np.zeros(n, dtype=np.float32)
    throt = np.zeros(n, dtype=np.float32)
    coll = np.zeros(n, dtype=np.float32)
    close = np.zeros(n, dtype=np.float32)
    hot = np.zeros(n, dtype=np.float32)
    hot_hit = np.zeros(n, dtype=np.float32)
    min_px = np.full(n, np.float32(8.0))
    keep = TRAIL_KEEP
    stride = max(1, steps // (keep - 1))
    trail = np.zeros((n, keep, 2), dtype=np.float32)
    trail[:, 0, 0] = x
    trail[:, 0, 1] = y
    k_save = 1
    safe = np.float32(max(float(hp.get("curve_clear", 0.42)), 1e-3))
    for t in range(steps):
        px = plus_x_batch(x, y, yaw, box)
        min_px = np.minimum(min_px, px)
        pin = px < stop
        throt += pin.astype(np.float32)
        v = np.where(pin, np.float32(0.0), v_cmd)
        w = w_cmd
        c = np.cos(yaw).astype(np.float32)
        s = np.sin(yaw).astype(np.float32)
        nx = x + v * c * dt
        ny = y + v * s * dt
        nyaw = yaw + w * dt
        ok = foot_ok_batch(nx, ny, nyaw, box, pad=0.06)
        dist = np.hypot(nx - x, ny - y)
        moved = ok & (dist > np.float32(0.002))
        path += np.where(moved, dist, np.float32(0.0))
        x = np.where(moved, nx, x)
        y = np.where(moved, ny, y)
        yaw = np.where(moved, nyaw, yaw)
        yaw_ok = foot_ok_batch(x, y, nyaw, box, pad=0.04)
        yaw_only = (~moved) & (np.abs(w) > 0.04) & yaw_ok
        yaw = np.where(yaw_only, nyaw, yaw)
        coll += ((~moved) & (~yaw_only)).astype(np.float32)
        d = clearance_batch(x, y, box)
        close += np.clip(1.0 - d / safe, 0.0, 2.0)
        heat = visit_heat_batch(x, y, x0, y0, visit)
        hot += np.tanh(heat / hot_s)
        hot_hit += (heat >= hot_s).astype(np.float32)
        if k_save < keep and ((t + 1) % stride == 0 or t + 1 == steps):
            trail[:, k_save, 0] = x
            trail[:, k_save, 1] = y
            k_save += 1
    throt /= np.float32(steps)
    coll /= np.float32(steps)
    close /= np.float32(steps)
    hot /= np.float32(steps)
    hot_hit /= np.float32(steps)
    if k_save < keep:
        trail[:, k_save:] = trail[:, k_save - 1:k_save]
    px_end = plus_x_batch(x, y, yaw, box)
    heat_end = visit_heat_batch(x, y, x0, y0, visit)
    return path, throt, min_px, coll, close, px_end, trail, hot, hot_hit, heat_end


def score_vw(v_cmd, w_cmd, x, y, yaw, box, hp, visit=None):
    path, throt, min_px, coll, close, px_end, trail, hot, hot_hit, heat_end = rollout_5s(
        v_cmd, w_cmd, x, y, yaw, box, hp, visit=visit,
    )
    prog = np.float32(hp.get("prim_progress", 0.55))
    stuck = np.float32(hp.get("prim_stuck", 2.60))
    wall = np.float32(hp.get("curve_wall_coef", 0.95))
    vcoef = np.float32(hp.get("visit_coef", 0.50))
    vhit = np.float32(hp.get("visit_hit_coef", 1.20))
    vend = np.float32(hp.get("visit_fwd_coef", 0.28))
    hot_s = np.float32(max(float(hp.get("visit_hot", 8.0)), 1e-3))
    # Hot visit cells are a doughnut collision: stay-on-loop loses to a cooler exit.
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
    )
    score = np.where((path < 0.12) & (throt > 0.40), score - 1.6, score)
    return score.astype(np.float32), path, throt, trail, hot, hot_hit
