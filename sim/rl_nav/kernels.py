#!/usr/bin/env python3
"""Warp kernels for a map-free Kevin explorer.

Sensing matches live ego: RS1 top-down rectangle union RS2 80° / 2.5 m cone.
Objective is fog-of-war floor coverage: ~0.4 m tiles stamped in a body
disk with wall LOS (no paint through walls). Not coin-chasing, not FOV
flood. Ego is occupancy only. Persistence is a 64-byte memory. Cover
fraction is in the vector — never painted on the ego raster.

Throttle air-gap is 30% of live SafetyGuard (70% smaller vs touching):
50% v then hard stop, then footprint hit = crash.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

MAX_B = 64
MAX_M = 4
MEM_N = 64
N_PRIM = 64
N_V = 8
N_W = 8
# Live 320×240 @ 1 cm → 80×60 @ 4 cm (hard to tell from the bag downsample).
EGO_H = 60
EGO_W = 80
N_FOOT = 8
N_SUB = 4
# Fog tiles (~0.40 m). Painted in a body disk with wall LOS — not FOV,
# not 2.4 m view cells (those were farmed by weaving a column edge).
SEC_NX = 18
SEC_NY = 37
SEC_N = SEC_NX * SEC_NY
PAINT_R = 0.90
PAINT_WIN = 7
# vec: v, w, plus_x, t, cover, v_scale, w_scale. No goal / hint bearing.
VEC_N = 7

FLOOR_X0, FLOOR_X1 = -3.55, 3.75
FLOOR_Y0, FLOOR_Y1 = -7.45, 7.45
WALL_T = 0.08
WALL_H = 2.15
TRACK_M = 0.34
NOSE = 0.15
V_MAX = 0.25
W_MAX = 0.80
V_REV = -0.20
A_MAX = 1.61
ALPHA_MAX = 4.0

# Live BRAKE_START ≈ 0.30 m. Old 8 cm only fired on the crash frame.
# Policy sees the mark; motion scales the same way as Kevin fwd/bwd/ang.
THROT_M = 0.28
STOP_M = 0.04
THROT_SIDE_M = 0.18
STOP_SIDE_M = 0.04
THROT_BWD_M = 0.20
STOP_BWD_M = 0.04
THROT_MARK = 0.20
MOVER_R = 0.28
BODY_R = 0.18

# RS1 rectangle in body metres (x forward, y left). Live bbox after
# TD_X_OFFSET; painting uses the bag-baked FOV stencil, not this AABB.
RS1_X0, RS1_X1 = -0.71, 1.58
RS1_Y0, RS1_Y1 = -1.10, 1.09
RS2_RANGE = 2.50
RS2_HALF = math.radians(40.0)
# Look knobs are runtime (look[0..10]) so tuning does not recompile Warp.
# 0 unused  1 unused  2 obs_thick  3 unused  4 unused
# 5 unused  6 close_hit  7 close_blob  8 unused
# 9 carpet_p (doc only; collision uses BLOB_P)  10 unused
# Real furniture is solid. Floor in FOV is clear. No 1 px dirt.
# Rare blobs (~2.5 s life, then gone). Same hash in paint + foot_ok +
# throttle rays. Show/hide across frames is intended — policy should
# slow down and use the 64-byte memory, not ignore red.
LOOK_KNOB_N = 11
LOOK_CLEAN = np.array(
    [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.40, 0.28, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
# Rare (~0.6% of 24 cm cells). Life ~1.4 s then a new draw — not a carpet.
# Never inside BLOB_NEAR_M of the axle: a pop under the hull was an instant
# crash and was ending every episode in a few metres.
BLOB_P = 0.006
BLOB_CELL = 0.24
BLOB_HZ = 0.70
BLOB_NEAR_M = 0.80
# Pets/people: never spawn inside this of the axle. They may walk closer.
MOVER_NEAR_M = 1.00
# Drift, not a chase. Kevin cannot dodge these; they are temp obstacles
# that mill a bit. 0.18–0.53 m/s was faster than the bot.
MOVER_SPD_LO = 0.035
MOVER_SPD_HI = 0.070
# Inside this, scale vel down to a crawl (almost parked near the hull).
MOVER_SLOW_M = 1.20
MOVER_SLOW_MIN = 0.10
LOOK_KNOBS = np.array(
    [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.40, 0.28, 0.0, BLOB_P, 0.0],
    dtype=np.float32,
)
FOV_HOLE = float(LOOK_KNOBS[0])
FOV_EDGE_HOLE = float(LOOK_KNOBS[1])
OBS_THICK_M = float(LOOK_KNOBS[2])
OBS_HOLE = float(LOOK_KNOBS[3])
HIT_JITTER_M = float(LOOK_KNOBS[4])
HIT_JITTER_SLOW_M = float(LOOK_KNOBS[5])
CLOSE_HIT_M = float(LOOK_KNOBS[6])
CLOSE_BLOB_M = float(LOOK_KNOBS[7])
CONE_WIGGLE = float(LOOK_KNOBS[8])

# Ego raster window = live axle (81,119) @ 1 cm, downsampled 4×.
EGO_X0, EGO_X1 = -0.81, 2.39
EGO_Y0, EGO_Y1 = -1.20, 1.20
EGO_PX = 0.04

DT_POLICY = 0.20
# Tour budget stays view-sized (18 hops × ~2.46 m / 0.25), not fog-tile count.
_VIEW_SX = (FLOOR_X1 - FLOOR_X0) / 3.0
_VIEW_SY = (FLOOR_Y1 - FLOOR_Y0) / 6.0
SEC_PITCH = 0.5 * (_VIEW_SX + _VIEW_SY)
T_TOUR_S = 17.0 * SEC_PITCH / V_MAX
# Discovery: ~438 free 0.40 m tiles, walk ≈ 0.56 → ~61 m².
# Hamiltonian lawnmower ≥ 175 m → 12 min at V_MAX=0.25. 20 min is enough
# to get the big rooms and still have leftover time — not to vacuum
# every last tile. Last-tile jackpot is off; paint tapers after 70%.
EP_S = 1200.0
EP_STEPS = int(round(EP_S / DT_POLICY))
SEC_SX = (FLOOR_X1 - FLOOR_X0) / float(SEC_NX)
SEC_SY = (FLOOR_Y1 - FLOOR_Y0) / float(SEC_NY)
PAINT_N = max(1.0, math.pi * PAINT_R * PAINT_R / (SEC_SX * SEC_SY))
PAINT_REW = 1.0 / PAINT_N
# Full credit until this cover fraction, then fade to PAINT_TAIL.
COVER_GAIN = 0.70
PAINT_TAIL = 0.25


@wp.func
def uhash(x: wp.uint32) -> wp.uint32:
    x = wp.uint32(x) ^ (wp.uint32(x) >> wp.uint32(16))
    x = x * wp.uint32(0x7FEB352D)
    x = x ^ (x >> wp.uint32(15))
    x = x * wp.uint32(0x846CA68B)
    x = x ^ (x >> wp.uint32(16))
    return x


@wp.func
def frand(seed: int, k: int) -> float:
    h = uhash(wp.uint32(seed) + wp.uint32(k) * wp.uint32(0x9E3779B9))
    return float(h) * (1.0 / 4294967295.0)


@wp.func
def wrap_pi(a: float) -> float:
    two = 6.283185307179586
    pi = 3.141592653589793
    return a - two * wp.floor((a + pi) / two)


@wp.func
def put_box(
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
    i: int,
    cx: float,
    cy: float,
    cz: float,
    hx: float,
    hy: float,
    hz: float,
    yaw: float,
    kind: float,
) -> int:
    if i < 0 or i >= MAX_B:
        return i
    boxes[e, i, 0] = wp.float32(cx)
    boxes[e, i, 1] = wp.float32(cy)
    boxes[e, i, 2] = wp.float32(cz)
    boxes[e, i, 3] = wp.float32(hx)
    boxes[e, i, 4] = wp.float32(hy)
    boxes[e, i, 5] = wp.float32(hz)
    boxes[e, i, 6] = wp.float32(yaw)
    boxes[e, i, 7] = wp.float32(kind)
    return i + 1


@wp.func
def wall_rect(
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
    i: int,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> int:
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = wp.max(0.5 * wp.abs(x1 - x0), 0.03)
    hy = wp.max(0.5 * wp.abs(y1 - y0), 0.03)
    return put_box(boxes, e, i, cx, cy, 0.5 * WALL_H, hx, hy, 0.5 * WALL_H, 0.0, 1.0)


@wp.func
def vert_wall_gap(
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
    i: int,
    x: float,
    y0: float,
    y1: float,
    g0: float,
    g1: float,
) -> int:
    t = WALL_T
    lo = wp.max(y0, wp.min(g0, g1))
    hi = wp.min(y1, wp.max(g0, g1))
    if lo - y0 > 0.35:
        i = wall_rect(boxes, e, i, x - t, x + t, y0, lo)
    if y1 - hi > 0.35:
        i = wall_rect(boxes, e, i, x - t, x + t, hi, y1)
    return i


@wp.func
def horz_wall_gap(
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
    i: int,
    y: float,
    x0: float,
    x1: float,
    g0: float,
    g1: float,
) -> int:
    t = WALL_T
    lo = wp.max(x0, wp.min(g0, g1))
    hi = wp.min(x1, wp.max(g0, g1))
    if lo - x0 > 0.35:
        i = wall_rect(boxes, e, i, x0, lo, y - t, y + t)
    if x1 - hi > 0.35:
        i = wall_rect(boxes, e, i, hi, x1, y - t, y + t)
    return i


@wp.func
def shape_inside_local(lx: float, ly: float, hx: float, hy: float, kind: float, sides: int) -> int:
    if hx < 1.0e-5 or hy < 1.0e-5:
        return 0
    if kind < 6.5:
        if wp.abs(lx) <= hx and wp.abs(ly) <= hy:
            return 1
        return 0
    ux = lx / hx
    uy = ly / hy
    if kind < 7.5:
        if ux * ux + uy * uy <= 1.0:
            return 1
        return 0
    n = sides
    if n < 5:
        n = 5
    if n > 7:
        n = 7
    lim = wp.cos(3.14159265359 / float(n))
    best = float(-1.0e9)
    for k in range(7):
        if k >= n:
            break
        a = (6.28318530718 * float(k) + 3.14159265359) / float(n)
        d = ux * wp.cos(a) + uy * wp.sin(a)
        if d > best:
            best = d
    if best <= lim:
        return 1
    return 0


@wp.func
def aabb_inside(
    px: float, py: float, boxes: wp.array3d(dtype=wp.float32), e: int, pad: float
) -> int:
    for b in range(MAX_B):
        hx0 = float(boxes[e, b, 3])
        if hx0 <= 0.0:
            continue
        cx = float(boxes[e, b, 0])
        cy = float(boxes[e, b, 1])
        hy0 = float(boxes[e, b, 4])
        yaw = float(boxes[e, b, 6])
        kind = float(boxes[e, b, 7])
        sides = int(wp.rint(float(boxes[e, b, 2])))
        cb = wp.cos(yaw)
        sb = wp.sin(yaw)
        dx = px - cx
        dy = py - cy
        lx = cb * dx + sb * dy
        ly = -sb * dx + cb * dy
        if shape_inside_local(lx, ly, hx0 + pad, hy0 + pad, kind, sides) == 1:
            return 1
    return 0


@wp.func
def foot_ok(
    x: float, y: float, yaw: float, boxes: wp.array3d(dtype=wp.float32), e: int, pad: float, t: float
) -> int:
    if x < FLOOR_X0 + 0.20 or x > FLOOR_X1 - 0.20 or y < FLOOR_Y0 + 0.20 or y > FLOOR_Y1 - 0.20:
        return 0
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    fx = wp.float32(0.15)
    fy = wp.float32(0.0)
    for k in range(N_FOOT):
        if k == 0:
            fx = 0.15
            fy = 0.0
        elif k == 1:
            fx = 0.15
            fy = 0.16
        elif k == 2:
            fx = 0.15
            fy = -0.16
        elif k == 3:
            fx = 0.0
            fy = 0.16
        elif k == 4:
            fx = 0.0
            fy = -0.16
        elif k == 5:
            fx = 0.0
            fy = 0.0
        elif k == 6:
            fx = -0.18
            fy = 0.0
        else:
            fx = -0.26
            fy = 0.0
        wx = x + c * fx - s * fy
        wy = y + s * fx + c * fy
        if aabb_inside(wx, wy, boxes, e, pad) == 1:
            return 0
        if blob_occ(wx, wy, e, t, x, y) == 1:
            return 0
    return 1


@wp.func
def slab1(orig: float, vel: float, h: float, tmin: float, tmax: float):
    if wp.abs(vel) < 1.0e-12:
        if wp.abs(orig) > h:
            return float(1.0), float(0.0)
        return tmin, tmax
    t1 = (-h - orig) / vel
    t2 = (h - orig) / vel
    lo = wp.min(t1, t2)
    hi = wp.max(t1, t2)
    return wp.max(tmin, lo), wp.min(tmax, hi)


@wp.func
def ray_shape_local(
    lx: float, ly: float, vx: float, vy: float, hx: float, hy: float, kind: float, sides: int
) -> float:
    if hx < 1.0e-5 or hy < 1.0e-5:
        return float(9.0e9)
    if kind < 6.5:
        if wp.abs(lx) <= hx and wp.abs(ly) <= hy:
            return 0.0
        tmin = float(-1.0e9)
        tmax = float(1.0e9)
        tmin, tmax = slab1(lx, vx, hx, tmin, tmax)
        tmin, tmax = slab1(ly, vy, hy, tmin, tmax)
        if tmax < tmin or tmax < 0.0:
            return float(9.0e9)
        if tmin < 0.0:
            return 0.0
        return tmin
    ox = lx / hx
    oy = ly / hy
    dvx = vx / hx
    dvy = vy / hy
    if kind < 7.5:
        if ox * ox + oy * oy <= 1.0:
            return 0.0
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (ox * dvx + oy * dvy)
        c = ox * ox + oy * oy - 1.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0 or a < 1.0e-12:
            return float(9.0e9)
        sd = wp.sqrt(disc)
        t0 = (-b - sd) / (2.0 * a)
        t1 = (-b + sd) / (2.0 * a)
        hit = t0
        if hit < 0.0:
            hit = t1
        if hit < 0.0:
            return float(9.0e9)
        return hit
    n = sides
    if n < 5:
        n = 5
    if n > 7:
        n = 7
    lim = wp.cos(3.14159265359 / float(n))
    if shape_inside_local(lx, ly, hx, hy, 8.0, n) == 1:
        return 0.0
    tmin = float(-1.0e9)
    tmax = float(1.0e9)
    miss = int(0)
    for k in range(7):
        if k >= n:
            break
        a = (6.28318530718 * float(k) + 3.14159265359) / float(n)
        nx = wp.cos(a)
        ny = wp.sin(a)
        denom = nx * dvx + ny * dvy
        num = lim - (nx * ox + ny * oy)
        if wp.abs(denom) < 1.0e-12:
            if nx * ox + ny * oy > lim:
                miss = 1
            continue
        t = num / denom
        if denom < 0.0:
            if t > tmin:
                tmin = t
        else:
            if t < tmax:
                tmax = t
    if miss == 1 or tmax < tmin or tmax < 0.0:
        return float(9.0e9)
    if tmin < 0.0:
        return 0.0
    return tmin


@wp.func
def ray_circle(
    ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, r: float
) -> float:
    px = ox - cx
    py = oy - cy
    if px * px + py * py <= r * r:
        return 0.0
    bb = 2.0 * (px * dx + py * dy)
    cc = px * px + py * py - r * r
    disc = bb * bb - 4.0 * cc
    if disc < 0.0:
        return float(9.0e9)
    sd = wp.sqrt(disc)
    t0 = (-bb - sd) * 0.5
    t1 = (-bb + sd) * 0.5
    hit = t0
    if hit < 0.0:
        hit = t1
    if hit < 0.0:
        return float(9.0e9)
    return hit


@wp.func
def ray_aabb_2d(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
    inflate: float,
) -> float:
    best = float(9.0e9)
    for b in range(MAX_B):
        hx0 = float(boxes[e, b, 3])
        if hx0 <= 0.0:
            continue
        hx = hx0 + inflate
        hy = float(boxes[e, b, 4]) + inflate
        cx = float(boxes[e, b, 0])
        cy = float(boxes[e, b, 1])
        yaw = float(boxes[e, b, 6])
        kind = float(boxes[e, b, 7])
        sides = int(wp.rint(float(boxes[e, b, 2])))
        cb = wp.cos(yaw)
        sb = wp.sin(yaw)
        px = ox - cx
        py = oy - cy
        lx = cb * px + sb * py
        ly = -sb * px + cb * py
        vx = cb * dx + sb * dy
        vy = -sb * dx + cb * dy
        hit = ray_shape_local(lx, ly, vx, vy, hx, hy, kind, sides)
        if hit < best:
            best = hit
    return best


@wp.func
def body_xy(dx: float, dy: float, yaw: float):
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return lx, ly


@wp.func
def in_rs1(lx: float, ly: float) -> int:
    if lx >= RS1_X0 and lx <= RS1_X1 and ly >= RS1_Y0 and ly <= RS1_Y1:
        return 1
    return 0


@wp.func
def in_rs2(lx: float, ly: float) -> int:
    d = wp.sqrt(lx * lx + ly * ly)
    if d > RS2_RANGE or d < 1.0e-4:
        return 0
    if wp.abs(wp.atan2(ly, lx)) <= RS2_HALF:
        return 1
    return 0


@wp.func
def ephemeral_blob(
    wx: float, wy: float, e: int, t: float, cell_m: float, life_hz: float, p: float, salt: int
) -> int:
    if p <= 0.0:
        return 0
    inv = 1.0 / cell_m
    cx = int(wp.floor(wx * inv))
    cy = int(wp.floor(wy * inv))
    life = int(t * life_hz)
    seed = e * 10007 + cx * 1009 + cy * 917 + life * 13 + salt
    if frand(seed, 31) >= p:
        return 0
    ux = wx * inv - float(cx)
    uy = wy * inv - float(cy)
    jx = 0.32 + 0.36 * frand(seed, 41)
    jy = 0.32 + 0.36 * frand(seed, 43)
    dx = ux - jx
    dy = uy - jy
    if dx * dx + dy * dy < 0.30:
        return 1
    return 0


@wp.func
def blob_occ(wx: float, wy: float, e: int, t: float, rx: float, ry: float) -> int:
    dx = wx - rx
    dy = wy - ry
    if dx * dx + dy * dy < BLOB_NEAR_M * BLOB_NEAR_M:
        return 0
    return ephemeral_blob(wx, wy, e, t, BLOB_CELL, BLOB_HZ, BLOB_P, 11)


@wp.func
def in_self_body(lx: float, ly: float) -> int:
    # One hull covering body + 10 cm wheels + mast. Not a CAD wheel schematic.
    if lx >= -0.16 and lx <= 0.20 and wp.abs(ly) <= 0.22:
        return 1
    return 0


@wp.func
def gap_scale(d: float, stop_m: float, throt_m: float) -> float:
    if d <= stop_m:
        return 0.0
    if d >= throt_m:
        return 1.0
    return (d - stop_m) / (throt_m - stop_m)


@wp.func
def body_clear(
    px: float, py: float, yaw: float, lx: float, ly: float, ux: float, uy: float,
    boxes: wp.array3d(dtype=wp.float32), e: int, t: float,
) -> float:
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    wx = px + c * lx - s * ly
    wy = py + s * lx + c * ly
    rdx = c * ux - s * uy
    rdy = s * ux + c * uy
    d = ray_aabb_2d(wx, wy, rdx, rdy, boxes, e, 0.0)
    for i in range(20):
        s_m = 0.04 * float(i + 1)
        if s_m >= d:
            break
        if blob_occ(wx + rdx * s_m, wy + rdy * s_m, e, t, px, py) == 1:
            d = s_m
            break
    if d > 8.0:
        d = 8.0
    return d


@wp.func
def world_occ(
    wx: float,
    wy: float,
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
    mx: wp.array2d(dtype=wp.float32),
    my: wp.array2d(dtype=wp.float32),
    n_movers: int,
) -> int:
    if aabb_inside(wx, wy, boxes, e, 0.0) == 1:
        return 1
    for m in range(MAX_M):
        if m >= n_movers:
            break
        dx = wx - float(mx[e, m])
        dy = wy - float(my[e, m])
        if dx * dx + dy * dy <= MOVER_R * MOVER_R:
            return 1
    return 0


@wp.func
def in_fov(lx: float, ly: float) -> int:
    if in_rs1(lx, ly) == 1:
        return 1
    return in_rs2(lx, ly)


@wp.func
def clip_vw(v: float, w: float, vmax: float, wmax: float):
    # 40 lb: a little less yaw at speed, not a hard wheel diamond.
    # Old |v|+|w|·track/2 with budget=vmax left zero yaw at vmax — too big.
    # At vmax keep 85% of wmax; slow down to get the rest (or crash).
    vv = wp.clamp(v, V_REV, vmax)
    ww = wp.clamp(w, -wmax, wmax)
    denom = vmax
    if denom < 1.0e-6:
        denom = 1.0e-6
    frac = wp.abs(vv) / denom
    if frac > 1.0:
        frac = 1.0
    w_allow = wmax * (1.0 - 0.15 * frac)
    ww = wp.clamp(ww, -w_allow, w_allow)
    return vv, ww


@wp.func
def sec_center(s: int):
    nx = float(SEC_NX)
    ny = float(SEC_NY)
    ix = float(s % SEC_NX)
    iy = float(s / SEC_NX)
    cx = FLOOR_X0 + (ix + 0.5) * ((FLOOR_X1 - FLOOR_X0) / nx)
    cy = FLOOR_Y0 + (iy + 0.5) * ((FLOOR_Y1 - FLOOR_Y0) / ny)
    return cx, cy


@wp.func
def sec_id(x: float, y: float) -> int:
    fx = (x - FLOOR_X0) / (FLOOR_X1 - FLOOR_X0)
    fy = (y - FLOOR_Y0) / (FLOOR_Y1 - FLOOR_Y0)
    ix = int(fx * float(SEC_NX))
    iy = int(fy * float(SEC_NY))
    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0
    if ix >= SEC_NX:
        ix = SEC_NX - 1
    if iy >= SEC_NY:
        iy = SEC_NY - 1
    return iy * SEC_NX + ix


@wp.func
def sec_is_valid(s: int, boxes: wp.array3d(dtype=wp.float32), e: int) -> int:
    cx, cy = sec_center(s)
    if aabb_inside(cx, cy, boxes, e, 0.12) == 1:
        return 0
    return 1


@wp.func
def clear_to(
    x: float,
    y: float,
    tx: float,
    ty: float,
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
) -> int:
    dx = tx - x
    dy = ty - y
    glen = wp.sqrt(dx * dx + dy * dy)
    if glen < 0.10:
        return 1
    ux = dx / glen
    uy = dy / glen
    hit = ray_aabb_2d(x, y, ux, uy, boxes, e, 0.0)
    if hit < glen - 0.05:
        return 0
    return 1


@wp.func
def paint_disk(
    x: float,
    y: float,
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
    seen: wp.array2d(dtype=wp.int32),
    valid: wp.array2d(dtype=wp.int32),
) -> int:
    nnew = int(0)
    r2 = PAINT_R * PAINT_R
    sid = sec_id(x, y)
    ix0 = sid % SEC_NX
    iy0 = (sid - ix0) / SEC_NX
    for dj in range(-PAINT_WIN, PAINT_WIN + 1):
        for di in range(-PAINT_WIN, PAINT_WIN + 1):
            ix = ix0 + di
            iy = iy0 + dj
            if ix < 0 or iy < 0 or ix >= SEC_NX or iy >= SEC_NY:
                continue
            s = iy * SEC_NX + ix
            if valid[e, s] == 0 or seen[e, s] != 0:
                continue
            cx, cy = sec_center(s)
            dx = cx - x
            dy = cy - y
            if dx * dx + dy * dy > r2:
                continue
            if clear_to(x, y, cx, cy, boxes, e) == 0:
                continue
            seen[e, s] = 1
            nnew = nnew + 1
    return nnew


@wp.func
def can_see(
    x: float,
    y: float,
    yaw: float,
    tx: float,
    ty: float,
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
) -> int:
    dx = tx - x
    dy = ty - y
    lx, ly = body_xy(dx, dy, yaw)
    if in_fov(lx, ly) == 0:
        return 0
    glen = wp.sqrt(dx * dx + dy * dy)
    if glen < 0.08:
        return 1
    ux = dx / glen
    uy = dy / glen
    hit = ray_aabb_2d(x, y, ux, uy, boxes, e, 0.0)
    if hit < glen - 0.06:
        return 0
    return 1


@wp.kernel
def k_reset(
    mask: wp.array(dtype=wp.int32),
    seed0: int,
    ep: wp.array(dtype=wp.int32),
    boxes: wp.array3d(dtype=wp.float32),
    nbox: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w: wp.array(dtype=wp.float32),
    mx: wp.array2d(dtype=wp.float32),
    my: wp.array2d(dtype=wp.float32),
    mvx: wp.array2d(dtype=wp.float32),
    mvy: wp.array2d(dtype=wp.float32),
    n_movers: wp.array(dtype=wp.int32),
    mem: wp.array2d(dtype=wp.float32),
    steps: wp.array(dtype=wp.int32),
    t_sim: wp.array(dtype=wp.float32),
    path_m: wp.array(dtype=wp.float32),
    hits: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    cut: wp.array(dtype=wp.int32),
    v_scale: wp.array(dtype=wp.float32),
    a_scale: wp.array(dtype=wp.float32),
    w_scale: wp.array(dtype=wp.float32),
    delay_n: wp.array(dtype=wp.int32),
    held_v: wp.array(dtype=wp.float32),
    held_w: wp.array(dtype=wp.float32),
    plus_x: wp.array(dtype=wp.float32),
    seen: wp.array2d(dtype=wp.int32),
    valid: wp.array2d(dtype=wp.int32),
    n_free: wp.array(dtype=wp.int32),
    n_movers_hp: int,
    stage: int,
):
    e = wp.tid()
    if mask[e] == 0:
        return
    seed = seed0 + e * 10007 + int(ep[e]) * 9176 + 13 + stage * 41
    # boxes / nbox are filled by env from the CPU house pool.
    sx = float(x[e])
    sy = float(y[e])
    syaw = (frand(seed, 200) - 0.5) * 6.28318530718
    found = int(0)
    if foot_ok(sx, sy, 0.0, boxes, e, 0.34, 0.0) == 1:
        found = 2
    for k in range(24):
        if found >= 2:
            break
        tx = FLOOR_X0 + 0.55 + (FLOOR_X1 - FLOOR_X0 - 1.10) * frand(seed, 210 + k)
        ty = FLOOR_Y0 + 0.55 + (FLOOR_Y1 - FLOOR_Y0 - 1.10) * frand(seed, 240 + k)
        tyaw = (frand(seed, 270 + k) - 0.5) * 6.28318530718
        if foot_ok(tx, ty, tyaw, boxes, e, 0.34, 0.0) == 1:
            px = ray_aabb_2d(tx, ty, wp.cos(tyaw), wp.sin(tyaw), boxes, e, 0.01)
            if px >= 0.55:
                sx = tx
                sy = ty
                syaw = tyaw
                found = 2
            elif found == 0:
                sx = tx
                sy = ty
                syaw = tyaw
                found = 1
    x[e] = wp.float32(sx)
    y[e] = wp.float32(sy)
    best_px = float(-1.0)
    best_yaw = float(syaw)
    for k in range(16):
        tyaw = -3.14159265359 + (6.28318530718 * float(k) / 16.0)
        opx = ray_aabb_2d(sx, sy, wp.cos(tyaw), wp.sin(tyaw), boxes, e, 0.01)
        if opx > best_px:
            best_px = opx
            best_yaw = tyaw
    yaw[e] = wp.float32(wrap_pi(best_yaw))
    v[e] = wp.float32(0.0)
    w[e] = wp.float32(0.0)

    nf = int(0)
    for s in range(SEC_N):
        seen[e, s] = 0
        valid[e, s] = 0
        if sec_is_valid(s, boxes, e) == 0:
            continue
        valid[e, s] = 1
        nf = nf + 1
    n_free[e] = nf
    hits[e] = paint_disk(sx, sy, boxes, e, seen, valid)

    nm = n_movers_hp
    if nm < 0:
        nm = 0
    if nm > MAX_M:
        nm = MAX_M
    for m in range(MAX_M):
        mx[e, m] = wp.float32(0.0)
        my[e, m] = wp.float32(0.0)
        mvx[e, m] = wp.float32(0.0)
        mvy[e, m] = wp.float32(0.0)
    placed = int(0)
    near2 = MOVER_NEAR_M * MOVER_NEAR_M
    for m in range(MAX_M):
        if m >= nm:
            break
        found = int(0)
        tx = float(0.0)
        ty = float(0.0)
        for k in range(48):
            px = FLOOR_X0 + 0.60 + (FLOOR_X1 - FLOOR_X0 - 1.20) * frand(seed, 700 + m * 48 + k)
            py = FLOOR_Y0 + 0.60 + (FLOOR_Y1 - FLOOR_Y0 - 1.20) * frand(seed, 900 + m * 48 + k)
            dx = px - sx
            dy = py - sy
            if dx * dx + dy * dy < near2:
                continue
            if foot_ok(px, py, 0.0, boxes, e, 0.28, 0.0) == 1:
                tx = px
                ty = py
                found = 1
                break
        if found == 0:
            continue
        spd = MOVER_SPD_LO + (MOVER_SPD_HI - MOVER_SPD_LO) * frand(seed, 720 + m)
        ang = 6.28318530718 * frand(seed, 730 + m)
        mx[e, placed] = wp.float32(tx)
        my[e, placed] = wp.float32(ty)
        mvx[e, placed] = wp.float32(spd * wp.cos(ang))
        mvy[e, placed] = wp.float32(spd * wp.sin(ang))
        placed = placed + 1
    n_movers[e] = placed

    for k in range(MEM_N):
        mem[e, k] = wp.float32(0.0)
    steps[e] = 0
    t_sim[e] = wp.float32(0.0)
    path_m[e] = wp.float32(0.0)
    done[e] = 0
    cut[e] = 0
    # Per-episode speed cap so memory cannot lock to one v_max (live policies vary).
    # 0.40–1.20 × 0.25 m/s → 0.10–0.30 m/s. Observed v is still v/V_MAX (absolute).
    v_scale[e] = wp.float32(0.40 + 0.80 * frand(seed, 800))
    a_scale[e] = wp.float32(0.70 + 0.60 * frand(seed, 801))
    w_scale[e] = wp.float32(0.65 + 0.60 * frand(seed, 802))
    delay_n[e] = int(frand(seed, 803) * 2.0)
    held_v[e] = wp.float32(0.0)
    held_w[e] = wp.float32(0.0)
    plus_x[e] = wp.float32(8.0)
    ep[e] = ep[e] + 1


@wp.kernel
def k_step(
    prim: wp.array(dtype=wp.int32),
    write: wp.array2d(dtype=wp.float32),
    boxes: wp.array3d(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w: wp.array(dtype=wp.float32),
    mx: wp.array2d(dtype=wp.float32),
    my: wp.array2d(dtype=wp.float32),
    mvx: wp.array2d(dtype=wp.float32),
    mvy: wp.array2d(dtype=wp.float32),
    n_movers: wp.array(dtype=wp.int32),
    mem: wp.array2d(dtype=wp.float32),
    steps: wp.array(dtype=wp.int32),
    t_sim: wp.array(dtype=wp.float32),
    path_m: wp.array(dtype=wp.float32),
    hits: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    cut: wp.array(dtype=wp.int32),
    v_scale: wp.array(dtype=wp.float32),
    a_scale: wp.array(dtype=wp.float32),
    w_scale: wp.array(dtype=wp.float32),
    delay_n: wp.array(dtype=wp.int32),
    held_v: wp.array(dtype=wp.float32),
    held_w: wp.array(dtype=wp.float32),
    plus_x: wp.array(dtype=wp.float32),
    throt_f: wp.array(dtype=wp.float32),
    throt_b: wp.array(dtype=wp.float32),
    throt_l: wp.array(dtype=wp.float32),
    throt_r: wp.array(dtype=wp.float32),
    rew: wp.array(dtype=wp.float32),
    seen: wp.array2d(dtype=wp.int32),
    valid: wp.array2d(dtype=wp.int32),
    n_free: wp.array(dtype=wp.int32),
    crash_coef: float,
    stall_coef: float,
):
    e = wp.tid()
    if done[e] != 0:
        rew[e] = wp.float32(0.0)
        return
    dt = DT_POLICY / float(N_SUB)
    vmax = V_MAX * float(v_scale[e])
    wmax = W_MAX * float(w_scale[e])
    av = A_MAX * float(a_scale[e])
    aw = ALPHA_MAX * float(a_scale[e])
    pv = float(v[e])
    pw = float(w[e])
    dv = wp.max(av * DT_POLICY, 0.05)
    dw = wp.max(aw * DT_POLICY, 0.20)
    v_lo = wp.clamp(pv - dv, V_REV, vmax)
    v_hi = wp.clamp(pv + dv, V_REV, vmax)
    w_lo = wp.clamp(pw - dw, -wmax, wmax)
    w_hi = wp.clamp(pw + dw, -wmax, wmax)
    if v_hi < v_lo:
        tmp = v_lo
        v_lo = v_hi
        v_hi = tmp
    if w_hi < w_lo:
        tmp = w_lo
        w_lo = w_hi
        w_hi = tmp
    a = int(prim[e])
    if a < 0:
        a = 0
    if a >= N_PRIM:
        a = N_PRIM - 1
    iv = a / N_W
    iw = a - iv * N_W
    fv = float(iv) / float(N_V - 1)
    fw = float(iw) / float(N_W - 1)
    v_cmd = v_lo + fv * (v_hi - v_lo)
    w_cmd = w_lo + fw * (w_hi - w_lo)
    v_cmd, w_cmd = clip_vw(v_cmd, w_cmd, vmax, wmax)
    if delay_n[e] > 0:
        old_v = float(held_v[e])
        old_w = float(held_w[e])
        held_v[e] = wp.float32(v_cmd)
        held_w[e] = wp.float32(w_cmd)
        v_cmd = old_v
        w_cmd = old_w
    else:
        held_v[e] = wp.float32(v_cmd)
        held_w[e] = wp.float32(w_cmd)

    px = float(x[e])
    py = float(y[e])
    pyaw = float(yaw[e])
    tnow = float(t_sim[e])
    r = float(0.0)
    crashed = int(0)
    for _s in range(N_SUB):
        c = wp.cos(pyaw)
        s = wp.sin(pyaw)
        cf = body_clear(px, py, pyaw, 0.19, 0.0, 1.0, 0.0, boxes, e, tnow)
        cb = body_clear(px, py, pyaw, -0.16, 0.0, -1.0, 0.0, boxes, e, tnow)
        cl = body_clear(px, py, pyaw, 0.03, 0.22, 0.0, 1.0, boxes, e, tnow)
        cr = body_clear(px, py, pyaw, 0.03, -0.22, 0.0, -1.0, boxes, e, tnow)
        plus_x[e] = wp.float32(cf)
        stop_f = wp.max(STOP_M, 1.25 * vmax * DT_POLICY)
        stop_b = wp.max(STOP_BWD_M, 1.25 * vmax * DT_POLICY)
        stop_s = wp.max(STOP_SIDE_M, 1.25 * wmax * 0.12)
        throt_f_m = wp.max(THROT_M, stop_f + (THROT_M - STOP_M))
        throt_b_m = wp.max(THROT_BWD_M, stop_b + (THROT_BWD_M - STOP_BWD_M))
        throt_s_m = wp.max(THROT_SIDE_M, stop_s + (THROT_SIDE_M - STOP_SIDE_M))
        sf = gap_scale(cf, stop_f, throt_f_m)
        sb = gap_scale(cb, stop_b, throt_b_m)
        sl = gap_scale(cl, stop_s, throt_s_m)
        sr = gap_scale(cr, stop_s, throt_s_m)
        throt_f[e] = wp.float32(1.0 - sf)
        throt_b[e] = wp.float32(1.0 - sb)
        throt_l[e] = wp.float32(1.0 - sl)
        throt_r[e] = wp.float32(1.0 - sr)
        vt = v_cmd
        wt = w_cmd
        if vt > 0.0:
            vt = vt * sf
        elif vt < 0.0:
            vt = vt * sb
        if wt > 0.0:
            wt = wt * sl
        elif wt < 0.0:
            wt = wt * sr
        max_dv = av * dt
        max_dw = aw * dt
        if wp.abs(vt - pv) > max_dv:
            if vt > pv:
                pv = pv + max_dv
            else:
                pv = pv - max_dv
        else:
            pv = vt
        if wp.abs(wt - pw) > max_dw:
            if wt > pw:
                pw = pw + max_dw
            else:
                pw = pw - max_dw
        else:
            pw = wt
        nx = px + pv * c * dt
        ny = py + pv * s * dt
        nyaw = wrap_pi(pyaw + pw * dt)
        if foot_ok(nx, ny, nyaw, boxes, e, 0.04, tnow) == 0:
            crashed = 1
            pv = 0.0
            pw = 0.0
            break
        dist = wp.sqrt((nx - px) * (nx - px) + (ny - py) * (ny - py))
        path_m[e] = path_m[e] + wp.float32(dist)
        px = nx
        py = ny
        pyaw = nyaw
        nm = int(n_movers[e])
        for m in range(MAX_M):
            if m >= nm:
                continue
            dx0 = px - float(mx[e, m])
            dy0 = py - float(my[e, m])
            d2 = dx0 * dx0 + dy0 * dy0
            slow = float(1.0)
            lim = MOVER_SLOW_M * MOVER_SLOW_M
            if d2 < lim:
                d = wp.sqrt(d2)
                u = d / MOVER_SLOW_M
                slow = MOVER_SLOW_MIN + (1.0 - MOVER_SLOW_MIN) * u
            ox = float(mx[e, m]) + float(mvx[e, m]) * dt * slow
            oy = float(my[e, m]) + float(mvy[e, m]) * dt * slow
            if aabb_inside(ox, oy, boxes, e, MOVER_R) == 1:
                mvx[e, m] = wp.float32(-float(mvx[e, m]))
                mvy[e, m] = wp.float32(-float(mvy[e, m]))
            else:
                mx[e, m] = wp.float32(ox)
                my[e, m] = wp.float32(oy)
            dx = px - float(mx[e, m])
            dy = py - float(my[e, m])
            if dx * dx + dy * dy < (MOVER_R + BODY_R) * (MOVER_R + BODY_R):
                crashed = 1
    x[e] = wp.float32(px)
    y[e] = wp.float32(py)
    yaw[e] = wp.float32(pyaw)
    v[e] = wp.float32(pv)
    w[e] = wp.float32(pw)
    nnew = int(0)
    if crashed == 0:
        nnew = paint_disk(px, py, boxes, e, seen, valid)
    nf0 = int(n_free[e])
    if nf0 < 1:
        nf0 = 1
    frac = float(hits[e]) / float(nf0)
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    mult = float(1.0)
    if frac > COVER_GAIN:
        u = (frac - COVER_GAIN) / (1.0 - COVER_GAIN)
        if u > 1.0:
            u = 1.0
        mult = 1.0 - (1.0 - PAINT_TAIL) * u
    hits[e] = hits[e] + nnew
    r = r + float(nnew) * wp.float32(PAINT_REW) * mult
    stall = float(0.0)
    if crashed == 0 and pv < 0.06:
        stall = stall_coef
    r = r - stall
    nf = int(n_free[e])
    left = nf - int(hits[e])
    if left < 0:
        left = 0
    steps[e] = steps[e] + 1
    t_sim[e] = t_sim[e] + wp.float32(DT_POLICY)
    if crashed == 1:
        r = r - crash_coef
        done[e] = 1
        cut[e] = 2
    elif steps[e] >= EP_STEPS:
        done[e] = 1
        cut[e] = 3
        if left == 0:
            cut[e] = 1
    rew[e] = wp.float32(r)
    for k in range(MEM_N):
        wr = float(write[e, k])
        wr = wp.clamp(wr, -1.0, 1.0)
        old = float(mem[e, k])
        nxt = 0.80 * old + 0.20 * wr
        q = wp.round(nxt * 127.0) / 127.0
        mem[e, k] = wp.float32(wp.clamp(q, -1.0, 1.0))


@wp.kernel
def k_obs(
    boxes: wp.array3d(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    mx: wp.array2d(dtype=wp.float32),
    my: wp.array2d(dtype=wp.float32),
    n_movers: wp.array(dtype=wp.int32),
    mem: wp.array2d(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w: wp.array(dtype=wp.float32),
    plus_x: wp.array(dtype=wp.float32),
    throt_f: wp.array(dtype=wp.float32),
    throt_b: wp.array(dtype=wp.float32),
    throt_l: wp.array(dtype=wp.float32),
    throt_r: wp.array(dtype=wp.float32),
    t_sim: wp.array(dtype=wp.float32),
    hits: wp.array(dtype=wp.int32),
    n_free: wp.array(dtype=wp.int32),
    v_scale: wp.array(dtype=wp.float32),
    w_scale: wp.array(dtype=wp.float32),
    fov: wp.array2d(dtype=wp.int32),
    look: wp.array(dtype=wp.float32),
    ego: wp.array3d(dtype=wp.float32),
    vec: wp.array2d(dtype=wp.float32),
):
    e, row, col = wp.tid()
    rx = float(x[e])
    ry = float(y[e])
    ryaw = float(yaw[e])
    fx = EGO_X0 + (EGO_X1 - EGO_X0) * ((float(col) + 0.5) / float(EGO_W))
    fy = EGO_Y1 - (EGO_Y1 - EGO_Y0) * ((float(row) + 0.5) / float(EGO_H))
    # Look like bag rebuild 80×60: paint only inside the baked FOV silhouette
    # (RS1 house + RS2 cone). No z16 deproject. UNKNOWN=0 CLEAR=0.5 OBS=1.
    lab = wp.float32(0.0)
    if in_self_body(fx, fy) == 1:
        mark = wp.float32(0.0)
        if fx >= 0.02 and float(throt_f[e]) > 0.08:
            mark = wp.float32(THROT_MARK)
        if fx <= 0.02 and float(throt_b[e]) > 0.08:
            mark = wp.float32(THROT_MARK)
        if fy >= 0.08 and float(throt_l[e]) > 0.08:
            mark = wp.float32(THROT_MARK)
        if fy <= -0.08 and float(throt_r[e]) > 0.08:
            mark = wp.float32(THROT_MARK)
        ego[e, row, col] = mark
    else:
        c = wp.cos(ryaw)
        s = wp.sin(ryaw)
        wx = rx + c * fx - s * fy
        wy = ry + s * fx + c * fy
        nm = int(n_movers[e])
        see = int(fov[row, col])
        t = float(t_sim[e])
        if see == 1:
            if world_occ(wx, wy, boxes, e, mx, my, nm) == 1:
                lab = wp.float32(1.0)
            else:
                lab = wp.float32(0.5)
        glen = wp.sqrt(fx * fx + fy * fy)
        ang = wp.atan2(fy, fx)
        in_cone = int(0)
        if glen > 0.05 and glen <= float(RS2_RANGE) and wp.abs(ang) <= float(RS2_HALF):
            in_cone = 1
        if in_cone == 1:
            hit = float(9.0e9)
            if glen > 1.0e-4:
                ux = (c * fx - s * fy) / glen
                uy = (s * fx + c * fy) / glen
                hit = ray_aabb_2d(rx, ry, ux, uy, boxes, e, 0.0)
                for m in range(MAX_M):
                    if m >= nm:
                        break
                    hm = ray_circle(rx, ry, ux, uy, float(mx[e, m]), float(my[e, m]), MOVER_R)
                    if hm < hit:
                        hit = hm
                for bi in range(20):
                    s_m = 0.04 * float(bi + 1)
                    if s_m >= hit:
                        break
                    if blob_occ(rx + ux * s_m, ry + uy * s_m, e, t, rx, ry) == 1:
                        hit = s_m
                        break
            thick = float(look[2])
            if hit < float(look[6]):
                thick = float(look[7]) + (float(look[6]) - hit) * 0.35
            if glen + 0.04 < hit:
                if lab < 0.25:
                    lab = wp.float32(0.5)
            elif glen < hit + thick:
                lab = wp.float32(1.0)
            else:
                lab = wp.float32(0.0)
        if lab > 0.25 and lab < 0.75:
            if blob_occ(wx, wy, e, t, rx, ry) == 1:
                lab = wp.float32(1.0)
        ego[e, row, col] = lab
    if row == 0 and col == 0:
        nf = float(n_free[e])
        if nf < 1.0:
            nf = 1.0
        vec[e, 0] = v[e] / wp.float32(V_MAX)
        vec[e, 1] = w[e] / wp.float32(W_MAX)
        vec[e, 2] = plus_x[e] / wp.float32(2.0)
        vec[e, 3] = t_sim[e] / wp.float32(EP_S)
        vec[e, 4] = float(hits[e]) / nf
        vec[e, 5] = v_scale[e]
        vec[e, 6] = w_scale[e]
        for k in range(MEM_N):
            vec[e, VEC_N + k] = mem[e, k]


def pinhole_unused():
    return np.zeros(1, dtype=np.float32)
