#!/usr/bin/env python3
"""CPU house layouts for rl_nav.

Newton's casita / rand_house look right because rooms are wall-anchored
furniture kits (bed, couch, desk, table) with robot-wide aisles, not one
floating box. This generator keeps that, adds a few topologies, and
rejects a plan unless Kevin can reach every visitable sector.

Planning numbers (residential guides + IRC + Newton kits):
- door clear 0.90–1.12 m (IRC ~0.81; Kevin needs ~0.70 to pass)
- primary aisle ≥ 0.80 m, bed-side ≥ 0.60 m
- min room 2.2 m (same as rand_house)
- free floor after furniture ~45–58% (Newton "rooms full, aisles left")
"""
from __future__ import annotations

import math
import random

import numpy as np

try:
    from . import kernels as K
except ImportError:
    import kernels as K  # noqa: E402

WALL_T = K.WALL_T
WALL_H = K.WALL_H
MAX_B = K.MAX_B
SEC_N = K.SEC_N
SEC_NX = K.SEC_NX
SEC_NY = K.SEC_NY
FLOOR_X0, FLOOR_X1 = K.FLOOR_X0, K.FLOOR_X1
FLOOR_Y0, FLOOR_Y1 = K.FLOOR_Y0, K.FLOOR_Y1
MIN_ROOM = 2.20
DOOR = (0.92, 1.12)
AISLE = 0.80
BODY_PAD = 0.26
CELL = 0.16
WALK_LO, WALK_HI = 0.32, 0.70
# kit count, irregular extras, top-up walk target, score walk band
_CLUTTER = (
    ((1, 2, 2, 3), (0, 1, 1), 0.62, (0.54, 0.70)),
    ((2, 3, 3, 4), (1, 2, 2), 0.54, (0.46, 0.62)),
    ((3, 4, 4, 5), (2, 2, 3), 0.46, (0.38, 0.54)),
    ((4, 5, 5, 6), (2, 3, 3, 4), 0.38, (0.32, 0.48)),
)
KIND = {
    "wall": 1.0, "couch": 2.0, "bed": 3.0, "table": 4.0, "desk": 5.0, "chair": 6.0,
    "oval": 7.0, "poly": 8.0,
}

# hx, hy as half-extents. Matches room_clutter / casita scale.
_FURN = {
    "bed": (0.40, 0.70, 0.28),
    "couch": (0.72, 0.30, 0.40),
    "desk": (0.50, 0.24, 0.38),
    "table": (0.46, 0.42, 0.36),
    "chair": (0.28, 0.28, 0.42),
}


def _box(cx, cy, hx, hy, kind, hz=None, yaw=0.0, sides=0):
    if hz is None:
        hz = 0.5 * WALL_H if kind <= 1.0 else 0.36
    cz = float(sides) if int(kind) >= 8 else hz
    return np.array(
        [cx, cy, cz, max(hx, 0.03), max(hy, 0.03), hz, float(yaw), float(kind)],
        dtype=np.float32,
    )


def _wall_rect(x0, x1, y0, y1):
    return _box(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * abs(x1 - x0), 0.5 * abs(y1 - y0), 1.0)


def _vert_gap(x, y0, y1, g0, g1):
    t = WALL_T
    lo, hi = max(y0, min(g0, g1)), min(y1, max(g0, g1))
    out = []
    if lo - y0 > 0.35:
        out.append(_wall_rect(x - t, x + t, y0, lo))
    if y1 - hi > 0.35:
        out.append(_wall_rect(x - t, x + t, hi, y1))
    return out


def _horz_gap(y, x0, x1, g0, g1):
    t = WALL_T
    lo, hi = max(x0, min(g0, g1)), min(x1, max(g0, g1))
    out = []
    if lo - x0 > 0.35:
        out.append(_wall_rect(x0, lo, y - t, y + t))
    if x1 - hi > 0.35:
        out.append(_wall_rect(hi, x1, y - t, y + t))
    return out


def _keep_v(x, g0, g1, depth=0.48):
    return (x - depth, x + depth, min(g0, g1) - 0.12, max(g0, g1) + 0.12)


def _keep_h(y, g0, g1, depth=0.48):
    return (min(g0, g1) - 0.12, max(g0, g1) + 0.12, y - depth, y + depth)


def _in_keep(cx, cy, hx, hy, keeps, pad=0.04):
    for x0, x1, y0, y1 in keeps:
        if (cx + hx + pad) >= x0 and (cx - hx - pad) <= x1 and (cy + hy + pad) >= y0 and (cy - hy - pad) <= y1:
            return True
    return False


def _aabb_hit(a, b, gap):
    r1 = math.hypot(float(a[3]), float(a[4]))
    r2 = math.hypot(float(b[3]), float(b[4]))
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])) < (r1 + r2 + gap)


def _envelope(x0, x1, y0, y1):
    return [
        _wall_rect(x0, x1, y0, y0 + WALL_T),
        _wall_rect(x0, x1, y1 - WALL_T, y1),
        _wall_rect(x0, x0 + WALL_T, y0, y1),
        _wall_rect(x1 - WALL_T, x1, y0, y1),
    ]


def _door(rng, lo, hi):
    dw = rng.uniform(*DOOR)
    span = hi - lo - dw - 1.10
    if span < 0.05:
        mid = 0.5 * (lo + hi)
        return mid - 0.5 * dw, mid + 0.5 * dw, dw
    mid = lo + 0.55 + 0.5 * dw + rng.random() * span
    return mid - 0.5 * dw, mid + 0.5 * dw, dw


def _bsp(rng, rect, depth, walls, rooms, keeps):
    x0, x1, y0, y1 = rect
    w, h = x1 - x0, y1 - y0
    can_x = w >= 2.0 * MIN_ROOM + 0.25
    can_y = h >= 2.0 * MIN_ROOM + 0.25
    if depth <= 0 or (not can_x and not can_y):
        rooms.append((x0, x1, y0, y1))
        return
    # Prefer splitting the long axis so rooms stay squarish (real plans).
    split_x = can_x if (can_x and not can_y) else (can_x and (w >= h * 0.92 or not can_y))
    if split_x:
        cut = rng.uniform(x0 + MIN_ROOM, x1 - MIN_ROOM)
        g0, g1, _ = _door(rng, y0, y1)
        walls.extend(_vert_gap(cut, y0, y1, g0, g1))
        keeps.append(_keep_v(cut, g0, g1))
        _bsp(rng, (x0, cut, y0, y1), depth - 1, walls, rooms, keeps)
        _bsp(rng, (cut, x1, y0, y1), depth - 1, walls, rooms, keeps)
    else:
        cut = rng.uniform(y0 + MIN_ROOM, y1 - MIN_ROOM)
        g0, g1, _ = _door(rng, x0, x1)
        walls.extend(_horz_gap(cut, x0, x1, g0, g1))
        keeps.append(_keep_h(cut, g0, g1))
        _bsp(rng, (x0, x1, y0, cut), depth - 1, walls, rooms, keeps)
        _bsp(rng, (x0, x1, cut, y1), depth - 1, walls, rooms, keeps)


def _topo_bsp(rng, x0, x1, y0, y1):
    walls, rooms, keeps = [], [], []
    _bsp(rng, (x0, x1, y0, y1), rng.choice((3, 3, 4, 4, 5)), walls, rooms, keeps)
    return walls, rooms, keeps


def _topo_spine(rng, x0, x1, y0, y1):
    """Central hall, rooms off both sides — the usual house circulation spine."""
    walls, rooms, keeps = [], [], []
    hw = rng.uniform(1.15, 1.40)
    hx0 = rng.uniform(x0 + MIN_ROOM, x1 - MIN_ROOM - hw)
    hx1 = hx0 + hw
    n_l = 2 if (y1 - y0) < 9.0 else rng.choice((2, 3))
    n_r = 2 if (y1 - y0) < 9.0 else rng.choice((2, 3))

    def _split_side(xa, xb, n, wall_x, into_hall):
        ys = [y0]
        for k in range(1, n):
            ys.append(y0 + (y1 - y0) * (k / n) + rng.uniform(-0.35, 0.35))
        ys.append(y1)
        ys = sorted(ys)
        for i in range(n):
            a, b = ys[i], ys[i + 1]
            if b - a < MIN_ROOM:
                continue
            rooms.append((xa, xb, a, b))
            if i > 0:
                g0, g1, _ = _door(rng, xa + 0.15, xb - 0.15)
                walls.extend(_horz_gap(a, xa, xb, g0, g1))
                keeps.append(_keep_h(a, g0, g1, 0.40))
            gy0, gy1, _ = _door(rng, a, b)
            walls.extend(_vert_gap(wall_x, a, b, gy0, gy1))
            keeps.append(_keep_v(wall_x, gy0, gy1))

    _split_side(x0, hx0, n_l, hx0, True)
    _split_side(hx1, x1, n_r, hx1, True)
    rooms.append((hx0, hx1, y0, y1))
    return walls, rooms, keeps


def _topo_stagger(rng, x0, x1, y0, y1):
    """Offset T-walls so the only through-path jogs (S-corridor)."""
    walls, rooms, keeps = [], [], []
    xa = x0 + (x1 - x0) * rng.uniform(0.30, 0.40)
    xb = x0 + (x1 - x0) * rng.uniform(0.58, 0.70)
    if xb < xa + 1.15:
        xb = xa + 1.15
    ym = y0 + (y1 - y0) * rng.uniform(0.42, 0.58)
    gy0, gy1, _ = _door(rng, y0, ym)
    walls.extend(_vert_gap(xa, y0, ym, gy0, gy1))
    keeps.append(_keep_v(xa, gy0, gy1))
    gy0, gy1, _ = _door(rng, ym, y1)
    walls.extend(_vert_gap(xb, ym, y1, gy0, gy1))
    keeps.append(_keep_v(xb, gy0, gy1))
    # Horizontal: one door in the offset band so the S is the only crossing.
    band0, band1 = min(xa, xb), max(xa, xb)
    g0, g1, _ = _door(rng, band0, band1)
    walls.extend(_horz_gap(ym, x0, x1, g0, g1))
    keeps.append(_keep_h(ym, g0, g1))
    rooms.extend((
        (x0, xa, y0, ym),
        (xa, x1, y0, ym),
        (x0, xb, ym, y1),
        (xb, x1, ym, y1),
    ))
    return walls, rooms, keeps


def _topo_lwing(rng, x0, x1, y0, y1):
    """Big living wing + L-hall + two or three bedrooms."""
    walls, rooms, keeps = [], [], []
    if rng.random() < 0.5:
        ys = y0 + (y1 - y0) * rng.uniform(0.40, 0.48)
        hall = rng.uniform(1.20, 1.40)
        rooms.append((x0, x1, y0, ys))
        g0, g1, _ = _door(rng, x0 + 0.8, x1 - 0.8)
        walls.extend(_horz_gap(ys, x0, x1, g0, g1))
        keeps.append(_keep_h(ys, g0, g1))
        hx1 = x0 + hall
        rooms.append((x0, hx1, ys, y1))
        n = 3
        span = x1 - hx1
        xs = [hx1 + span * (k / n) for k in range(n + 1)]
        for i in range(n):
            a, b = xs[i], xs[i + 1]
            rooms.append((a, b, ys, y1))
            gy0, gy1, _ = _door(rng, ys, y1)
            walls.extend(_vert_gap(a, ys, y1, gy0, gy1))
            keeps.append(_keep_v(a, gy0, gy1))
    else:
        xs = x0 + (x1 - x0) * rng.uniform(0.40, 0.50)
        hall = rng.uniform(1.20, 1.40)
        rooms.append((x0, xs, y0, y1))
        g0, g1, _ = _door(rng, y0 + 0.8, y1 - 0.8)
        walls.extend(_vert_gap(xs, y0, y1, g0, g1))
        keeps.append(_keep_v(xs, g0, g1))
        hy1 = y0 + hall
        rooms.append((xs, x1, y0, hy1))
        n = 3
        span = y1 - hy1
        ys = [hy1 + span * (k / n) for k in range(n + 1)]
        for i in range(n):
            a, b = ys[i], ys[i + 1]
            rooms.append((xs, x1, a, b))
            gx0, gx1, _ = _door(rng, xs, x1)
            walls.extend(_horz_gap(a, xs, x1, gx0, gx1))
            keeps.append(_keep_h(a, gx0, gx1))
    return walls, rooms, keeps


def _topo_cross(rng, x0, x1, y0, y1):
    """Plus-shaped hall, four corner rooms."""
    walls, rooms, keeps = [], [], []
    xa = x0 + (x1 - x0) * rng.uniform(0.34, 0.42)
    xb = x0 + (x1 - x0) * rng.uniform(0.58, 0.66)
    ya = y0 + (y1 - y0) * rng.uniform(0.34, 0.42)
    yb = y0 + (y1 - y0) * rng.uniform(0.58, 0.66)
    if xb < xa + 1.15:
        xb = xa + 1.15
    if yb < ya + 1.15:
        yb = ya + 1.15
    for xw, ylo, yhi in ((xa, y0, ya), (xa, yb, y1), (xb, y0, ya), (xb, yb, y1)):
        if yhi - ylo < MIN_ROOM * 0.85:
            continue
        g0, g1, _ = _door(rng, ylo, yhi)
        walls.extend(_vert_gap(xw, ylo, yhi, g0, g1))
        keeps.append(_keep_v(xw, g0, g1))
    for yw, xlo, xhi in ((ya, x0, xa), (ya, xb, x1), (yb, x0, xa), (yb, xb, x1)):
        if xhi - xlo < MIN_ROOM * 0.85:
            continue
        g0, g1, _ = _door(rng, xlo, xhi)
        walls.extend(_horz_gap(yw, xlo, xhi, g0, g1))
        keeps.append(_keep_h(yw, g0, g1))
    rooms.extend((
        (x0, xa, y0, ya), (xb, x1, y0, ya),
        (x0, xa, yb, y1), (xb, x1, yb, y1),
        (xa, xb, y0, y1),
        (x0, x1, ya, yb),
    ))
    return walls, rooms, keeps


def _topo_galley(rng, x0, x1, y0, y1):
    """Long corridor, rooms all on one side."""
    walls, rooms, keeps = [], [], []
    hall = rng.uniform(1.15, 1.40)
    n = rng.choice((3, 4, 4, 5))
    if rng.random() < 0.5:
        hx1 = x0 + hall
        rooms.append((x0, hx1, y0, y1))
        ys = [y0 + (y1 - y0) * (k / n) for k in range(n + 1)]
        for i in range(n):
            a, b = ys[i], ys[i + 1]
            rooms.append((hx1, x1, a, b))
            if i > 0:
                g0, g1, _ = _door(rng, hx1 + 0.15, x1 - 0.15)
                walls.extend(_horz_gap(a, hx1, x1, g0, g1))
                keeps.append(_keep_h(a, g0, g1, 0.40))
            gy0, gy1, _ = _door(rng, a, b)
            walls.extend(_vert_gap(hx1, a, b, gy0, gy1))
            keeps.append(_keep_v(hx1, gy0, gy1))
    else:
        hy1 = y0 + hall
        rooms.append((x0, x1, y0, hy1))
        xs = [x0 + (x1 - x0) * (k / n) for k in range(n + 1)]
        for i in range(n):
            a, b = xs[i], xs[i + 1]
            rooms.append((a, b, hy1, y1))
            if i > 0:
                g0, g1, _ = _door(rng, hy1 + 0.15, y1 - 0.15)
                walls.extend(_vert_gap(a, hy1, y1, g0, g1))
                keeps.append(_keep_v(a, g0, g1, 0.40))
            gx0, gx1, _ = _door(rng, a, b)
            walls.extend(_horz_gap(hy1, a, b, gx0, gx1))
            keeps.append(_keep_h(hy1, gx0, gx1))
    return walls, rooms, keeps


def _topo_ring(rng, x0, x1, y0, y1):
    """Loop around a center room."""
    walls, rooms, keeps = [], [], []
    m = rng.uniform(1.45, 1.75)
    ix0, ix1 = x0 + m, x1 - m
    iy0, iy1 = y0 + m, y1 - m
    if ix1 - ix0 < MIN_ROOM or iy1 - iy0 < MIN_ROOM:
        return _topo_stagger(rng, x0, x1, y0, y1)
    for xw in (ix0, ix1):
        g0, g1, _ = _door(rng, iy0, iy1)
        walls.extend(_vert_gap(xw, iy0, iy1, g0, g1))
        keeps.append(_keep_v(xw, g0, g1))
    for yw in (iy0, iy1):
        g0, g1, _ = _door(rng, ix0, ix1)
        walls.extend(_horz_gap(yw, ix0, ix1, g0, g1))
        keeps.append(_keep_h(yw, g0, g1))
    rooms.extend((
        (x0, x1, y0, iy0),
        (x0, x1, iy1, y1),
        (x0, ix0, iy0, iy1),
        (ix1, x1, iy0, iy1),
        (ix0, ix1, iy0, iy1),
    ))
    return walls, rooms, keeps


def _topo_ushape(rng, x0, x1, y0, y1):
    """Open living court with rooms on three sides."""
    walls, rooms, keeps = [], [], []
    hall = rng.uniform(1.15, 1.40)
    if rng.random() < 0.5:
        ys = y0 + (y1 - y0) * rng.uniform(0.38, 0.48)
        rooms.append((x0, x1, y0, ys))
        g0, g1, _ = _door(rng, x0 + 0.6, x1 - 0.6)
        walls.extend(_horz_gap(ys, x0, x1, g0, g1))
        keeps.append(_keep_h(ys, g0, g1))
        lx1 = x0 + hall
        rx0 = x1 - hall
        rooms.append((x0, lx1, ys, y1))
        rooms.append((rx0, x1, ys, y1))
        n = rng.choice((2, 2, 3))
        span = rx0 - lx1
        xs = [lx1 + span * (k / n) for k in range(n + 1)]
        for i in range(n):
            a, b = xs[i], xs[i + 1]
            rooms.append((a, b, ys, y1))
            gy0, gy1, _ = _door(rng, ys, y1)
            walls.extend(_vert_gap(a, ys, y1, gy0, gy1))
            keeps.append(_keep_v(a, gy0, gy1))
            if i == n - 1:
                gy0, gy1, _ = _door(rng, ys, y1)
                walls.extend(_vert_gap(b, ys, y1, gy0, gy1))
                keeps.append(_keep_v(b, gy0, gy1))
    else:
        xs = x0 + (x1 - x0) * rng.uniform(0.38, 0.48)
        rooms.append((x0, xs, y0, y1))
        g0, g1, _ = _door(rng, y0 + 0.6, y1 - 0.6)
        walls.extend(_vert_gap(xs, y0, y1, g0, g1))
        keeps.append(_keep_v(xs, g0, g1))
        by1 = y0 + hall
        ty0 = y1 - hall
        rooms.append((xs, x1, y0, by1))
        rooms.append((xs, x1, ty0, y1))
        n = rng.choice((2, 2, 3))
        span = ty0 - by1
        ys = [by1 + span * (k / n) for k in range(n + 1)]
        for i in range(n):
            a, b = ys[i], ys[i + 1]
            rooms.append((xs, x1, a, b))
            gx0, gx1, _ = _door(rng, xs, x1)
            walls.extend(_horz_gap(a, xs, x1, gx0, gx1))
            keeps.append(_keep_h(a, gx0, gx1))
            if i == n - 1:
                gx0, gx1, _ = _door(rng, xs, x1)
                walls.extend(_horz_gap(b, xs, x1, gx0, gx1))
                keeps.append(_keep_h(b, gx0, gx1))
    return walls, rooms, keeps


def _room_kind(rect):
    x0, x1, y0, y1 = rect
    area = (x1 - x0) * (y1 - y0)
    if area < 9.0:
        return "bed"
    if area < 16.0:
        return "den"
    return "live"


def _kit_for(kind, rng):
    if kind == "bed":
        return [("bed",) + _FURN["bed"], ("desk",) + _FURN["desk"]]
    if kind == "den":
        return [("desk",) + _FURN["desk"], ("chair",) + _FURN["chair"], ("table",) + _FURN["table"]]
    bits = [("couch",) + _FURN["couch"], ("table",) + _FURN["table"], ("chair",) + _FURN["chair"]]
    if rng.random() < 0.55:
        bits.append(("couch", 0.55, 0.28, 0.40))
    if rng.random() < 0.40:
        bits.append(("desk",) + _FURN["desk"])
    return bits


def _place_on_wall(rng, rect, hx, hy, keeps, placed, flush=0.06):
    x0, x1, y0, y1 = rect
    walls = ["s", "n", "w", "e"]
    rng.shuffle(walls)
    for wall in walls:
        if wall in ("s", "n"):
            if (x1 - x0) < 2.0 * (AISLE + hx) + 0.3:
                continue
            cx = rng.uniform(x0 + AISLE + hx, x1 - AISLE - hx)
            cy = (y0 + flush + hy) if wall == "s" else (y1 - flush - hy)
        else:
            if (y1 - y0) < 2.0 * (AISLE + hy) + 0.3:
                continue
            cy = rng.uniform(y0 + AISLE + hy, y1 - AISLE - hy)
            cx = (x0 + flush + hx) if wall == "w" else (x1 - flush - hx)
        # Jog along the wall so two pieces make an S, not a straight aisle.
        if wall in ("s", "n"):
            cx += rng.uniform(-0.25, 0.25) * min(1.0, (x1 - x0) * 0.15)
            cx = min(max(cx, x0 + AISLE + hx), x1 - AISLE - hx)
        else:
            cy += rng.uniform(-0.25, 0.25) * min(1.0, (y1 - y0) * 0.15)
            cy = min(max(cy, y0 + AISLE + hy), y1 - AISLE - hy)
        yaw = (0.0 if wall in ("s", "n") else 0.5 * math.pi) + rng.uniform(-0.22, 0.22)
        cand = _box(cx, cy, hx, hy, 2.0, yaw=yaw)
        if _in_keep(cx, cy, hx, hy, keeps):
            continue
        if any(_aabb_hit(cand, p, AISLE) for p in placed):
            continue
        return cand
    return None


def _furnish(rng, rooms, keeps, kit_ns=(3, 3, 4, 4, 5), irreg_ns=(1, 2, 2, 3)):
    out = []
    for rec in rooms:
        x0, x1, y0, y1 = rec
        if (x1 - x0) < 2.4 or (y1 - y0) < 2.4:
            continue
        kit = _kit_for(_room_kind(rec), rng)
        rng.shuffle(kit)
        n = min(len(kit), rng.choice(kit_ns))
        for name, hx, hy, hz in kit[:n]:
            if name in ("table", "chair"):
                # Island / pull-up: offset from walls, leaves a bent path.
                pad = AISLE + max(hx, hy)
                if x1 - x0 < 2 * pad + 0.4 or y1 - y0 < 2 * pad + 0.4:
                    continue
                cx = rng.uniform(x0 + pad, x1 - pad)
                cy = rng.uniform(y0 + pad, y1 - pad)
                # Bias off-center so the aisle curves around it.
                cx += 0.22 * math.copysign(1.0, rng.random() - 0.5)
                cy += 0.22 * math.copysign(1.0, rng.random() - 0.5)
                cx = min(max(cx, x0 + pad), x1 - pad)
                cy = min(max(cy, y0 + pad), y1 - pad)
                cand = _box(cx, cy, hx, hy, KIND[name], hz, yaw=rng.uniform(0.0, 6.2832))
                if _in_keep(cx, cy, hx, hy, keeps):
                    continue
                if any(_aabb_hit(cand, p, AISLE) for p in out):
                    continue
                out.append(cand)
            else:
                got = _place_on_wall(rng, rec, hx, hy, keeps, out)
                if got is None:
                    continue
                got[2] = hz
                got[5] = hz
                got[7] = KIND[name]
                out.append(got)
        out.extend(_irregular_in_room(rng, rec, keeps, out, irreg_ns))
    return out


def _irregular_in_room(rng, rect, keeps, placed, irreg_ns=(1, 2, 2, 3)):
    """Ovals + convex n-gons at random yaw — closer to RealSense blobs."""
    x0, x1, y0, y1 = rect
    extra = []
    n_try = rng.choice(irreg_ns)
    for _ in range(n_try):
        oval = rng.random() < 0.55
        if oval:
            hx = rng.uniform(0.22, 0.42)
            hy = rng.uniform(0.18, 0.38)
            kind, sides = KIND["oval"], 0
        else:
            hx = rng.uniform(0.24, 0.48)
            hy = rng.uniform(0.20, 0.40)
            kind, sides = KIND["poly"], rng.choice((5, 5, 6, 7))
        pad = AISLE + max(hx, hy)
        if x1 - x0 < 2 * pad + 0.35 or y1 - y0 < 2 * pad + 0.35:
            continue
        ok = None
        for _att in range(8):
            cx = rng.uniform(x0 + pad, x1 - pad)
            cy = rng.uniform(y0 + pad, y1 - pad)
            yaw = rng.uniform(0.0, 6.2832)
            cand = _box(cx, cy, hx, hy, kind, 0.34, yaw=yaw, sides=sides)
            if _in_keep(cx, cy, hx, hy, keeps):
                continue
            if any(_aabb_hit(cand, p, AISLE) for p in placed + extra):
                continue
            ok = cand
            break
        if ok is not None:
            extra.append(ok)
    return extra


def _shape_hit(x, y, b, pad):
    hx, hy = float(b[3]), float(b[4])
    if hx < 1e-4:
        return False
    yaw = float(b[6])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - float(b[0]), y - float(b[1])
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    kind = float(b[7])
    hx, hy = hx + pad, hy + pad
    if hx < 1e-5 or hy < 1e-5:
        return False
    if kind < 6.5:
        return abs(lx) <= hx and abs(ly) <= hy
    ux, uy = lx / hx, ly / hy
    if kind < 7.5:
        return ux * ux + uy * uy <= 1.0
    n = int(round(float(b[2])))
    n = min(7, max(5, n))
    lim = math.cos(math.pi / n)
    best = -1e9
    for k in range(n):
        a = (2.0 * math.pi * k + math.pi) / n
        d = ux * math.cos(a) + uy * math.sin(a)
        if d > best:
            best = d
    return best <= lim


def _occ_grid(boxes):
    xs = np.arange(FLOOR_X0 + 0.5 * CELL, FLOOR_X1, CELL, dtype=np.float32)
    ys = np.arange(FLOOR_Y0 + 0.5 * CELL, FLOOR_Y1, CELL, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    occ = np.zeros(gx.shape, dtype=np.uint8)
    pad = BODY_PAD
    for b in boxes:
        hx, hy = float(b[3]), float(b[4])
        if hx < 1e-4:
            continue
        rad = math.hypot(hx, hy) + pad
        near = (np.abs(gx - float(b[0])) <= rad) & (np.abs(gy - float(b[1])) <= rad)
        if not np.any(near):
            continue
        yaw = float(b[6])
        c, s = math.cos(yaw), math.sin(yaw)
        dx = gx[near] - float(b[0])
        dy = gy[near] - float(b[1])
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        ah, ay = hx + pad, hy + pad
        kind = float(b[7])
        if kind < 6.5:
            hit = (np.abs(lx) <= ah) & (np.abs(ly) <= ay)
        elif kind < 7.5:
            hit = (lx / ah) ** 2 + (ly / ay) ** 2 <= 1.0
        else:
            n = min(7, max(5, int(round(float(b[2])))))
            lim = math.cos(math.pi / n)
            ux, uy = lx / ah, ly / ay
            best = np.full(ux.shape, -1e9, dtype=np.float32)
            for k in range(n):
                a = (2.0 * math.pi * k + math.pi) / n
                best = np.maximum(best, ux * math.cos(a) + uy * math.sin(a))
            hit = best <= lim
        chunk = occ[near]
        chunk[hit] = 1
        occ[near] = chunk
    return occ, xs, ys


def _flood(occ, xs, ys, sx, sy):
    ix = int(np.argmin(np.abs(xs - sx)))
    iy = int(np.argmin(np.abs(ys - sy)))
    if occ[iy, ix]:
        found = False
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < occ.shape[1] and 0 <= jy < occ.shape[0] and not occ[jy, jx]:
                    ix, iy = jx, jy
                    found = True
                    break
            if found:
                break
        if not found:
            return None
    seen = np.zeros_like(occ, dtype=np.uint8)
    q = [(ix, iy)]
    seen[iy, ix] = 1
    head = 0
    h, w = occ.shape
    while head < len(q):
        x, y = q[head]
        head += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if seen[ny, nx] or occ[ny, nx]:
                continue
            seen[ny, nx] = 1
            q.append((nx, ny))
    return seen


def _sec_center(s):
    ix = s % SEC_NX
    iy = s // SEC_NX
    cx = FLOOR_X0 + (ix + 0.5) * ((FLOOR_X1 - FLOOR_X0) / float(SEC_NX))
    cy = FLOOR_Y0 + (iy + 0.5) * ((FLOOR_Y1 - FLOOR_Y0) / float(SEC_NY))
    return cx, cy


def _sec_valid_gpu(s, boxes):
    cx, cy = _sec_center(s)
    for b in boxes:
        if float(b[3]) < 1e-4:
            continue
        if _shape_hit(cx, cy, b, 0.12):
            return False, []
    return True, [(cx, cy)]


def _reachable_xy(seen, xs, ys, x, y):
    if seen is None:
        return False
    ix = int(np.argmin(np.abs(xs - x)))
    iy = int(np.argmin(np.abs(ys - y)))
    return bool(seen[iy, ix])


def _score(boxes, rooms):
    occ, xs, ys = _occ_grid(boxes)
    interior = occ.size
    walk = 1.0 - float(occ.mean())
    spawn = None
    rng_pts = []
    for rec in rooms:
        cx = 0.5 * (rec[0] + rec[1])
        cy = 0.5 * (rec[2] + rec[3])
        rng_pts.append((cx, cy))
    for cx, cy in rng_pts + [(0.0, 0.0)]:
        ix = int(np.argmin(np.abs(xs - cx)))
        iy = int(np.argmin(np.abs(ys - cy)))
        if not occ[iy, ix]:
            spawn = (float(xs[ix]), float(ys[iy]))
            break
    if spawn is None:
        free = np.argwhere(occ == 0)
        if len(free) == 0:
            return None
        iy, ix = free[len(free) // 2]
        spawn = (float(xs[ix]), float(ys[iy]))
    seen = _flood(occ, xs, ys, spawn[0], spawn[1])
    if seen is None:
        return None
    n_valid = 0
    n_ok = 0
    for s in range(SEC_N):
        ok, pts = _sec_valid_gpu(s, boxes)
        if not ok:
            continue
        n_valid += 1
        if any(_reachable_xy(seen, xs, ys, px, py) for px, py in pts):
            n_ok += 1
    if n_ok < 80:
        return None
    if n_valid > 0 and n_ok < 0.88 * n_valid:
        return None
    if not (WALK_LO <= walk <= WALK_HI):
        return None
    reach_rooms = 0
    for rec in rooms:
        cx = 0.5 * (rec[0] + rec[1])
        cy = 0.5 * (rec[2] + rec[3])
        if _reachable_xy(seen, xs, ys, cx, cy):
            reach_rooms += 1
        else:
            # room center on furniture is fine if any cell in the room is reached
            x0, x1, y0, y1 = rec
            hit = False
            for iy, y in enumerate(ys):
                if y < y0 or y > y1:
                    continue
                for ix, x in enumerate(xs):
                    if x < x0 or x > x1:
                        continue
                    if seen[iy, ix]:
                        hit = True
                        break
                if hit:
                    break
            if not hit:
                return None
            reach_rooms += 1
    return {
        "spawn": spawn,
        "walk": walk,
        "n_sec": n_ok,
        "n_rooms": reach_rooms,
        "interior": interior,
    }


def _top_up(rng, rooms, keeps, walls, furn, want=0.52):
    """Add extra chairs/tables until walkable area is near half the floor."""
    boxes, n = _pack(walls, furn)
    occ, _, _ = _occ_grid(boxes[:n])
    walk = 1.0 - float(occ.mean())
    extras = list(furn)
    tries = 0
    while walk > want + 0.03 and tries < 24:
        tries += 1
        rec = rooms[rng.randrange(len(rooms))]
        name = rng.choice(("chair", "table", "couch"))
        hx, hy, hz = _FURN[name] if name in _FURN else _FURN["chair"]
        got = _place_on_wall(rng, rec, hx, hy, keeps, extras)
        if got is None:
            continue
        got[2] = hz
        got[5] = hz
        got[7] = KIND[name]
        extras.append(got)
        boxes, n = _pack(walls, extras)
        occ, _, _ = _occ_grid(boxes[:n])
        walk = 1.0 - float(occ.mean())
    return extras


def _pack(walls, furn):
    all_b = list(walls) + list(furn)
    if len(all_b) > MAX_B:
        all_b = all_b[:MAX_B]
    arr = np.zeros((MAX_B, 8), dtype=np.float32)
    for i, b in enumerate(all_b):
        arr[i] = b
    return arr, len(all_b)


def make_layout(seed: int):
    rng = random.Random(int(seed) * 10007 + 17)
    x0, x1 = FLOOR_X0 + 0.04, FLOOR_X1 - 0.04
    y0, y1 = FLOOR_Y0 + 0.04, FLOOR_Y1 - 0.04
    ix0, ix1 = x0 + WALL_T, x1 - WALL_T
    iy0, iy1 = y0 + WALL_T, y1 - WALL_T
    TOPOS = (
        _topo_bsp, _topo_spine, _topo_stagger, _topo_lwing,
        _topo_cross, _topo_galley, _topo_ring, _topo_ushape,
    )
    topo = rng.choice(TOPOS)
    extra, rooms, keeps = topo(rng, ix0, ix1, iy0, iy1)
    rooms = [r for r in rooms if (r[1] - r[0]) > 1.4 and (r[3] - r[2]) > 1.4]
    if len(rooms) < 3:
        return None
    walls = _envelope(x0, x1, y0, y1) + extra
    kit_ns, irreg_ns, want, _band = rng.choice(_CLUTTER)
    furn = _top_up(
        rng, rooms, keeps, walls,
        _furnish(rng, rooms, keeps, kit_ns, irreg_ns),
        want=want,
    )
    boxes, n = _pack(walls, furn)
    meta = _score(boxes[:n], rooms)
    if meta is None:
        return None
    meta["topo"] = topo.__name__
    meta["nbox"] = n
    meta["n_rooms_draw"] = len(rooms)
    return {"boxes": boxes, "nbox": n, "xy": meta["spawn"], "meta": meta}


def make_good_layout(seed: int, tries: int = 48):
    for t in range(int(tries)):
        got = make_layout(int(seed) + t * 7919)
        if got is not None:
            return got
    # Last resort: empty rooms, still connected (always valid).
    rng = random.Random(int(seed) + 3)
    x0, x1 = FLOOR_X0 + 0.04, FLOOR_X1 - 0.04
    y0, y1 = FLOOR_Y0 + 0.04, FLOOR_Y1 - 0.04
    extra, rooms, keeps = _topo_stagger(rng, x0 + WALL_T, x1 - WALL_T, y0 + WALL_T, y1 - WALL_T)
    walls = _envelope(x0, x1, y0, y1) + extra
    boxes, n = _pack(walls, [])
    meta = _score(boxes[:n], rooms) or {
        "spawn": (0.0, 0.0), "walk": 0.7, "n_sec": 12, "n_rooms": len(rooms),
    }
    meta["topo"] = "fallback_stagger"
    meta["nbox"] = n
    return {"boxes": boxes, "nbox": n, "xy": meta["spawn"], "meta": meta}


def build_pool(n: int = 128, seed: int = 7):
    boxes = np.zeros((n, MAX_B, 8), dtype=np.float32)
    nbox = np.zeros((n,), dtype=np.int32)
    xy = np.zeros((n, 2), dtype=np.float32)
    walks, secs, rooms, topos = [], [], [], []
    for i in range(n):
        lay = make_good_layout(int(seed) + i * 104729)
        boxes[i] = lay["boxes"]
        nbox[i] = lay["nbox"]
        xy[i, 0] = lay["xy"][0]
        xy[i, 1] = lay["xy"][1]
        m = lay["meta"]
        walks.append(m["walk"])
        secs.append(m["n_sec"])
        rooms.append(m["n_rooms"])
        topos.append(m.get("topo", "?"))
    from collections import Counter
    print(
        "house pool n=%d walk=%.2f±%.2f sec=%.1f rooms=%.1f topos=%s"
        % (
            n, float(np.mean(walks)), float(np.std(walks)),
            float(np.mean(secs)), float(np.mean(rooms)),
            dict(Counter(topos)),
        ),
        flush=True,
    )
    return boxes, nbox, xy
