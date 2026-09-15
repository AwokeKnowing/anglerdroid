#!/usr/bin/env python3
"""Warp kernels for a fully GPU-resident insect vec-env.

No Newton, no triangle mesh, no per-env Python. Houses are yawed AABBs.
Depth is a pinhole AABB slab test. Dynamics, Bézier commit/track, visit,
and rewards are one launch over N.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

MAX_B = 32
POLY_N = 24
DEPTH_H = 48
DEPTH_W = 80
EGO_H = 48
EGO_W = 64
VISIT_N = 32
VISIT_M = 3.2
VISIT_CELL = VISIT_M / VISIT_N
TRACK_M = 0.23
RS1_Z = 0.86
RS2_Z = 0.97
RS2_NOSE = 0.05
PITCH = math.radians(25.6)
VFOV = math.radians(58.0)
EGO_X0, EGO_X1 = 0.05, 2.45
EGO_Y0, EGO_Y1 = -1.28, 1.28
FLOOR_X0, FLOOR_X1 = -3.55, 3.75
FLOOR_Y0, FLOOR_Y1 = -7.45, 7.45
COV_CELL = 0.40
COV_W = int(round((FLOOR_X1 - FLOOR_X0) / COV_CELL))
COV_H = int(round((FLOOR_Y1 - FLOOR_Y0) / COV_CELL))
WALL_T = 0.08
WALL_H = 2.15
MAX_RAY = 8.0

# Footprint samples in body (x forward, y left), matching plus_x_against_boxes.
N_FOOT = 8


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
):
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
):
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
):
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
):
    t = WALL_T
    lo = wp.max(x0, wp.min(g0, g1))
    hi = wp.min(x1, wp.max(g0, g1))
    if lo - x0 > 0.35:
        i = wall_rect(boxes, e, i, x0, lo, y - t, y + t)
    if x1 - hi > 0.35:
        i = wall_rect(boxes, e, i, hi, x1, y - t, y + t)
    return i


@wp.func
def box_clearance(px: float, py: float, boxes: wp.array3d(dtype=wp.float32), e: int) -> float:
    best = float(9.0)
    for b in range(MAX_B):
        hx = float(boxes[e, b, 3])
        if hx <= 0.0:
            continue
        cy = float(boxes[e, b, 1])
        cx = float(boxes[e, b, 0])
        hy = float(boxes[e, b, 4])
        yaw = float(boxes[e, b, 6])
        cb = wp.cos(yaw)
        sb = wp.sin(yaw)
        dx = px - cx
        dy = py - cy
        lx = cb * dx + sb * dy
        ly = -sb * dx + cb * dy
        ax = wp.abs(lx) - hx
        ay = wp.abs(ly) - hy
        d = float(0.0)
        if ax <= 0.0 and ay <= 0.0:
            d = wp.max(ax, ay)
        else:
            d = wp.sqrt(wp.max(ax, 0.0) * wp.max(ax, 0.0) + wp.max(ay, 0.0) * wp.max(ay, 0.0))
        if d < best:
            best = d
    return best


@wp.func
def aabb_inside(px: float, py: float, boxes: wp.array3d(dtype=wp.float32), e: int, pad: float) -> int:
    for b in range(MAX_B):
        hx = float(boxes[e, b, 3])
        if hx <= 0.0:
            continue
        cx = float(boxes[e, b, 0])
        cy = float(boxes[e, b, 1])
        hy = float(boxes[e, b, 4])
        yaw = float(boxes[e, b, 6])
        cb = wp.cos(yaw)
        sb = wp.sin(yaw)
        dx = px - cx
        dy = py - cy
        lx = cb * dx + sb * dy
        ly = -sb * dx + cb * dy
        if wp.abs(lx) <= hx + pad and wp.abs(ly) <= hy + pad:
            return 1
    return 0


@wp.func
def height_at(px: float, py: float, boxes: wp.array3d(dtype=wp.float32), e: int) -> float:
    h = float(0.0)
    for b in range(MAX_B):
        hx = float(boxes[e, b, 3])
        if hx <= 0.0:
            continue
        cx = float(boxes[e, b, 0])
        cy = float(boxes[e, b, 1])
        hy = float(boxes[e, b, 4])
        yaw = float(boxes[e, b, 6])
        cb = wp.cos(yaw)
        sb = wp.sin(yaw)
        dx = px - cx
        dy = py - cy
        lx = cb * dx + sb * dy
        ly = -sb * dx + cb * dy
        if wp.abs(lx) <= hx and wp.abs(ly) <= hy:
            top = float(boxes[e, b, 2]) + float(boxes[e, b, 5])
            if top > h:
                h = top
    return h


@wp.func
def foot_ok(x: float, y: float, yaw: float, boxes: wp.array3d(dtype=wp.float32), e: int, pad: float) -> int:
    if x < FLOOR_X0 + 0.20 or x > FLOOR_X1 - 0.20 or y < FLOOR_Y0 + 0.20 or y > FLOOR_Y1 - 0.20:
        return 0
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    # 8 body samples
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
    return 1


@wp.func
def slab1(orig: float, vel: float, h: float, tmin: float, tmax: float):
    """One AABB slab. miss → tmin=1, tmax=0."""
    if wp.abs(vel) < 1.0e-12:
        if wp.abs(orig) > h:
            return float(1.0), float(0.0)
        return tmin, tmax
    t1 = (-h - orig) / vel
    t2 = (h - orig) / vel
    lo = wp.min(t1, t2)
    hi = wp.max(t1, t2)
    ntmin = wp.max(tmin, lo)
    ntmax = wp.min(tmax, hi)
    return ntmin, ntmax


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
        cb = wp.cos(yaw)
        sb = wp.sin(yaw)
        px = ox - cx
        py = oy - cy
        lx = cb * px + sb * py
        ly = -sb * px + cb * py
        vx = cb * dx + sb * dy
        vy = -sb * dx + cb * dy
        if wp.abs(lx) <= hx and wp.abs(ly) <= hy:
            return 0.0
        tmin = float(-1.0e9)
        tmax = float(1.0e9)
        tmin, tmax = slab1(lx, vx, hx, tmin, tmax)
        tmin, tmax = slab1(ly, vy, hy, tmin, tmax)
        if tmax < tmin or tmax < 0.0:
            continue
        hit = tmin
        if hit < 0.0:
            hit = float(0.0)
        if hit < best:
            best = hit
    return best


@wp.func
def ray_aabb_3d(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    boxes: wp.array3d(dtype=wp.float32),
    e: int,
) -> float:
    best = float(MAX_RAY)
    for b in range(MAX_B):
        hx = float(boxes[e, b, 3])
        if hx <= 0.0:
            continue
        hy = float(boxes[e, b, 4])
        hz = float(boxes[e, b, 5])
        cx = float(boxes[e, b, 0])
        cy = float(boxes[e, b, 1])
        cz = float(boxes[e, b, 2])
        yaw = float(boxes[e, b, 6])
        cb = wp.cos(yaw)
        sb = wp.sin(yaw)
        px = ox - cx
        py = oy - cy
        pz = oz - cz
        lx = cb * px + sb * py
        ly = -sb * px + cb * py
        lz = pz
        vx = cb * dx + sb * dy
        vy = -sb * dx + cb * dy
        vz = dz
        tmin = float(0.0)
        tmax = float(MAX_RAY)
        tmin, tmax = slab1(lx, vx, hx, tmin, tmax)
        tmin, tmax = slab1(ly, vy, hy, tmin, tmax)
        tmin, tmax = slab1(lz, vz, hz, tmin, tmax)
        if tmax < tmin or tmax < 0.0:
            continue
        hit = tmin
        if hit < 0.02:
            continue
        if hit < best:
            best = hit
    return best


@wp.func
def bezier_xy(
    p0x: float, p0y: float, p1x: float, p1y: float,
    p2x: float, p2y: float, p3x: float, p3y: float, t: float,
):
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3.0 * u * u * t
    b2 = 3.0 * u * t * t
    b3 = t * t * t
    return b0 * p0x + b1 * p1x + b2 * p2x + b3 * p3x, b0 * p0y + b1 * p1y + b2 * p2y + b3 * p3y


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
    visit: wp.array3d(dtype=wp.float32),
    cov: wp.array3d(dtype=wp.uint8),
    path_m: wp.array(dtype=wp.float32),
    coll: wp.array(dtype=wp.int32),
    coll_streak: wp.array(dtype=wp.int32),
    yaw_only: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    t_sim: wp.array(dtype=wp.float32),
    need_curve: wp.array(dtype=wp.int32),
    curve_got: wp.array2d(dtype=wp.int32),
    curve_next: wp.array(dtype=wp.int32),
    curve_chord: wp.array(dtype=wp.float32),
    curve_t0: wp.array(dtype=wp.float32),
    curve_n: wp.array(dtype=wp.int32),
    curve_wins: wp.array(dtype=wp.int32),
    curve_losses: wp.array(dtype=wp.int32),
    beads_sum: wp.array(dtype=wp.float32),
    chord_win_sum: wp.array(dtype=wp.float32),
    last_act: wp.array2d(dtype=wp.float32),
    rooms_seen: wp.array(dtype=wp.int32),
    visit_gx: wp.array(dtype=wp.int32),
    visit_gy: wp.array(dtype=wp.int32),
    ctrl: wp.array3d(dtype=wp.float32),
    poly: wp.array3d(dtype=wp.float32),
    beads: wp.array3d(dtype=wp.float32),
    wall_c: wp.array(dtype=wp.float32),
    cont_c: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.int32),
    seg_done: wp.array(dtype=wp.int32),
    cut: wp.array(dtype=wp.int32),
    cov_cells: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    if mask[e] == 0:
        return
    seed = seed0 + e * 10007 + int(ep[e]) * 9176 + 13
    for b in range(MAX_B):
        boxes[e, b, 3] = wp.float32(0.0)
        boxes[e, b, 4] = wp.float32(0.0)
        boxes[e, b, 7] = wp.float32(0.0)
    x0 = FLOOR_X0 + 0.04
    x1 = FLOOR_X1 - 0.04
    y0 = FLOOR_Y0 + 0.04
    y1 = FLOOR_Y1 - 0.04
    i = int(0)
    i = wall_rect(boxes, e, i, x0, x1, y0, y0 + WALL_T)
    i = wall_rect(boxes, e, i, x0, x1, y1 - WALL_T, y1)
    i = wall_rect(boxes, e, i, x0, x0 + WALL_T, y0, y1)
    i = wall_rect(boxes, e, i, x1 - WALL_T, x1, y0, y1)
    dw = 0.88 + 0.30 * frand(seed, 1)
    cutx0 = x0 + (x1 - x0) * (0.28 + 0.14 * frand(seed, 2))
    cutx1 = x0 + (x1 - x0) * (0.56 + 0.14 * frand(seed, 3))
    if cutx1 < cutx0 + 2.15:
        cutx1 = cutx0 + 2.15
    cuty = y0 + (y1 - y0) * (0.42 + 0.16 * frand(seed, 4))
    midy0 = y0 + 0.70 + (cuty - y0 - 1.4) * frand(seed, 5)
    midy1 = cuty + 0.70 + (y1 - cuty - 1.4) * frand(seed, 6)
    midx = x0 + 0.70 + (x1 - x0 - 1.4) * frand(seed, 7)
    i = vert_wall_gap(boxes, e, i, cutx0, y0, y1, midy0 - 0.5 * dw, midy0 + 0.5 * dw)
    i = vert_wall_gap(boxes, e, i, cutx1, y0, y1, midy1 - 0.5 * dw, midy1 + 0.5 * dw)
    i = horz_wall_gap(boxes, e, i, cuty, x0, x1, midx - 0.5 * dw, midx + 0.5 * dw)
    # six rooms from the two x-cuts and one y-cut; furniture + people
    xs = wp.float32(0.0)
    xe = wp.float32(0.0)
    ys = wp.float32(0.0)
    ye = wp.float32(0.0)
    for r in range(6):
        if r % 3 == 0:
            xs = x0 + WALL_T
            xe = cutx0
        elif r % 3 == 1:
            xs = cutx0
            xe = cutx1
        else:
            xs = cutx1
            xe = x1 - WALL_T
        if r < 3:
            ys = y0 + WALL_T
            ye = cuty
        else:
            ys = cuty
            ye = y1 - WALL_T
        if frand(seed, 20 + r) < 0.72 and (xe - xs) > 1.6 and (ye - ys) > 1.6:
            hx = 0.22 + 0.30 * frand(seed, 40 + r)
            hy = 0.22 + 0.40 * frand(seed, 50 + r)
            pad = 0.55
            cx = xs + pad + hx + (xe - xs - 2.0 * (pad + hx)) * frand(seed, 60 + r)
            cy = ys + pad + hy + (ye - ys - 2.0 * (pad + hy)) * frand(seed, 70 + r)
            hz = 0.28 + 0.20 * frand(seed, 80 + r)
            if xe - xs > 2.0 * (pad + hx) + 0.2 and ye - ys > 2.0 * (pad + hy) + 0.2:
                i = put_box(boxes, e, i, cx, cy, hz, hx, hy, hz, 0.0, 2.0)
        if r < 2 and frand(seed, 90 + r) < 0.55:
            pcx = 0.5 * (xs + xe) + (frand(seed, 100 + r) - 0.5) * 0.6
            pcy = 0.5 * (ys + ye) + (frand(seed, 110 + r) - 0.5) * 0.6
            i = put_box(boxes, e, i, pcx, pcy, 0.85, 0.30, 0.30, 0.85, 0.0, 3.0)
    nbox[e] = i

    sx = float(0.5 * (x0 + x1))
    sy = float(0.5 * (y0 + y1))
    syaw = float((frand(seed, 200) - 0.5) * 6.28318530718)
    found = int(0)
    for k in range(12):
        rr = k % 6
        if rr % 3 == 0:
            xs = x0 + WALL_T
            xe = cutx0
        elif rr % 3 == 1:
            xs = cutx0
            xe = cutx1
        else:
            xs = cutx1
            xe = x1 - WALL_T
        if rr < 3:
            ys = y0 + WALL_T
            ye = cuty
        else:
            ys = cuty
            ye = y1 - WALL_T
        tx = 0.5 * (xs + xe) + (frand(seed, 210 + k) - 0.5) * 0.8
        ty = 0.5 * (ys + ye) + (frand(seed, 230 + k) - 0.5) * 0.8
        tyaw = (frand(seed, 250 + k) - 0.5) * 6.28318530718
        if found < 2 and foot_ok(tx, ty, tyaw, boxes, e, 0.34) == 1:
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
    for r in range(VISIT_N):
        for c in range(VISIT_N):
            visit[e, r, c] = wp.float32(0.0)
    visit[e, VISIT_N // 2, VISIT_N // 2] = wp.float32(1.0)
    visit_gx[e] = int(wp.rint(sx / VISIT_CELL))
    visit_gy[e] = int(wp.rint(sy / VISIT_CELL))
    for r in range(COV_H):
        for c in range(COV_W):
            cov[e, r, c] = wp.uint8(0)
    path_m[e] = wp.float32(0.0)
    coll[e] = 0
    coll_streak[e] = 0
    yaw_only[e] = 0
    steps[e] = 0
    t_sim[e] = wp.float32(0.0)
    need_curve[e] = 1
    curve_next[e] = 0
    curve_chord[e] = wp.float32(0.0)
    curve_t0[e] = wp.float32(0.0)
    curve_n[e] = 0
    curve_wins[e] = 0
    curve_losses[e] = 0
    beads_sum[e] = wp.float32(0.0)
    chord_win_sum[e] = wp.float32(0.0)
    rooms_seen[e] = 1
    wall_c[e] = wp.float32(0.0)
    cont_c[e] = wp.float32(0.0)
    done[e] = 0
    seg_done[e] = 0
    cut[e] = 0
    cov_cells[e] = 0
    ep[e] = ep[e] + 1
    for k in range(4):
        last_act[e, k] = wp.float32(0.0)
        ctrl[e, k, 0] = wp.float32(0.0)
        ctrl[e, k, 1] = wp.float32(0.0)
    for k in range(3):
        curve_got[e, k] = 0
        beads[e, k, 0] = wp.float32(0.0)
        beads[e, k, 1] = wp.float32(0.0)
    for k in range(POLY_N):
        poly[e, k, 0] = wp.float32(0.0)
        poly[e, k, 1] = wp.float32(0.0)


@wp.kernel
def k_step(
    act: wp.array2d(dtype=wp.float32),
    boxes: wp.array3d(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    visit: wp.array3d(dtype=wp.float32),
    visit_tmp: wp.array3d(dtype=wp.float32),
    cov: wp.array3d(dtype=wp.uint8),
    path_m: wp.array(dtype=wp.float32),
    coll: wp.array(dtype=wp.int32),
    coll_streak: wp.array(dtype=wp.int32),
    yaw_only: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    t_sim: wp.array(dtype=wp.float32),
    need_curve: wp.array(dtype=wp.int32),
    curve_got: wp.array2d(dtype=wp.int32),
    curve_next: wp.array(dtype=wp.int32),
    curve_chord: wp.array(dtype=wp.float32),
    curve_t0: wp.array(dtype=wp.float32),
    curve_n: wp.array(dtype=wp.int32),
    curve_wins: wp.array(dtype=wp.int32),
    curve_losses: wp.array(dtype=wp.int32),
    beads_sum: wp.array(dtype=wp.float32),
    chord_win_sum: wp.array(dtype=wp.float32),
    last_act: wp.array2d(dtype=wp.float32),
    visit_gx: wp.array(dtype=wp.int32),
    visit_gy: wp.array(dtype=wp.int32),
    ctrl: wp.array3d(dtype=wp.float32),
    poly: wp.array3d(dtype=wp.float32),
    beads: wp.array3d(dtype=wp.float32),
    wall_c: wp.array(dtype=wp.float32),
    cont_c: wp.array(dtype=wp.float32),
    plus_x: wp.array(dtype=wp.float32),
    rew: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.int32),
    seg_done: wp.array(dtype=wp.int32),
    need_out: wp.array(dtype=wp.int32),
    cut: wp.array(dtype=wp.int32),
    cov_cells: wp.array(dtype=wp.int32),
    rooms_seen: wp.array(dtype=wp.int32),
    v_out: wp.array(dtype=wp.float32),
    w_out: wp.array(dtype=wp.float32),
    dt: float,
    max_v: float,
    max_w: float,
    plus_stop: float,
    rmin: float,
    rmax: float,
    lat_max: float,
    bead_r: float,
    look: float,
    curve_s: float,
    curve_bead: float,
    cover_coef: float,
    curve_win: float,
    curve_lose: float,
    clear_m: float,
    wall_coef: float,
    hit_coef: float,
    hit_m: float,
    cont_s: float,
    cont_coef: float,
    coll_done: int,
    ep_steps: int,
    yaw_cap: int,
    cov_cell: float,
):
    e = wp.tid()
    px = float(x[e])
    py = float(y[e])
    pyaw = float(yaw[e])
    committed = int(0)
    if need_curve[e] != 0:
        a0 = wp.clamp(float(act[e, 0]), -1.0, 1.0)
        a1 = wp.clamp(float(act[e, 1]), -1.0, 1.0)
        a2 = wp.clamp(float(act[e, 2]), -1.0, 1.0)
        a3 = wp.clamp(float(act[e, 3]), -1.0, 1.0)
        last_act[e, 0] = wp.float32(a0)
        last_act[e, 1] = wp.float32(a1)
        last_act[e, 2] = wp.float32(a2)
        last_act[e, 3] = wp.float32(a3)
        r = rmin + 0.5 * (a0 + 1.0) * (rmax - rmin)
        # a1 = heading offset of the chord (the missing "turn"). Laterals
        # a2/a3 are shape in that rotated frame. P3 sits on the new heading.
        yoff = a1 * 1.57079632679
        c = wp.cos(pyaw + yoff)
        s = wp.sin(pyaw + yoff)
        p0x = px
        p0y = py
        p1x = px + c * (r / 3.0) - s * (a2 * lat_max)
        p1y = py + s * (r / 3.0) + c * (a2 * lat_max)
        p2x = px + c * (2.0 * r / 3.0) - s * (a3 * lat_max)
        p2y = py + s * (2.0 * r / 3.0) + c * (a3 * lat_max)
        p3x = px + c * r
        p3y = py + s * r
        ctrl[e, 0, 0] = wp.float32(p0x)
        ctrl[e, 0, 1] = wp.float32(p0y)
        ctrl[e, 1, 0] = wp.float32(p1x)
        ctrl[e, 1, 1] = wp.float32(p1y)
        ctrl[e, 2, 0] = wp.float32(p2x)
        ctrl[e, 2, 1] = wp.float32(p2y)
        ctrl[e, 3, 0] = wp.float32(p3x)
        ctrl[e, 3, 1] = wp.float32(p3y)
        chord = wp.sqrt((p3x - p0x) * (p3x - p0x) + (p3y - p0y) * (p3y - p0y))
        curve_chord[e] = wp.float32(chord)
        curve_t0[e] = t_sim[e]
        curve_next[e] = 0
        curve_got[e, 0] = 0
        curve_got[e, 1] = 0
        curve_got[e, 2] = 0
        for k in range(POLY_N):
            t = float(k) / float(POLY_N - 1)
            bx, by = bezier_xy(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t)
            poly[e, k, 0] = wp.float32(bx)
            poly[e, k, 1] = wp.float32(by)
        t1x, t1y = bezier_xy(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, 1.0 / 3.0)
        t2x, t2y = bezier_xy(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, 2.0 / 3.0)
        beads[e, 0, 0] = wp.float32(t1x)
        beads[e, 0, 1] = wp.float32(t1y)
        beads[e, 1, 0] = wp.float32(t2x)
        beads[e, 1, 1] = wp.float32(t2y)
        beads[e, 2, 0] = wp.float32(p3x)
        beads[e, 2, 1] = wp.float32(p3y)
        close_sum = float(0.0)
        n_hit = int(0)
        min_d = float(9.0)
        end_d = float(9.0)
        for k in range(POLY_N):
            d = box_clearance(float(poly[e, k, 0]), float(poly[e, k, 1]), boxes, e)
            close_sum = close_sum + wp.max(0.0, 1.0 - d / wp.max(clear_m, 1.0e-3))
            if d < hit_m:
                n_hit = n_hit + 1
            if d < min_d:
                min_d = d
            if k == POLY_N - 1:
                end_d = d
        mean_close = close_sum / float(POLY_N)
        worst = wp.max(0.0, 1.0 - min_d / wp.max(clear_m, 1.0e-3))
        end_close = wp.max(0.0, 1.0 - end_d / wp.max(clear_m, 1.0e-3))
        wc = wall_coef * (0.25 * mean_close + 0.40 * worst + 0.35 * end_close) + hit_coef * (
            float(n_hit) / float(POLY_N)
        )
        wall_c[e] = wp.float32(wc)
        tdx = 3.0 * (p3x - p2x)
        tdy = 3.0 * (p3y - p2y)
        tn = wp.sqrt(tdx * tdx + tdy * tdy)
        if tn < 1.0e-5:
            tdx = p3x - p0x
            tdy = p3y - p0y
            tn = wp.sqrt(tdx * tdx + tdy * tdy)
        if tn < 1.0e-5:
            tdx = 1.0
            tdy = 0.0
            tn = 1.0
        tdx = tdx / tn
        tdy = tdy / tn
        d_hit = ray_aabb_2d(p3x, p3y, tdx, tdy, boxes, e, hit_m)
        cc = 0.0
        if cont_coef > 0.0 and d_hit < 1.0e8:
            t_hit = d_hit / wp.max(max_v, 1.0e-3)
            if t_hit < cont_s:
                cc = cont_coef * (1.0 - t_hit / wp.max(cont_s, 1.0e-3))
        cont_c[e] = wp.float32(cc)
        need_curve[e] = 0
        committed = 1

    # plus-x along heading
    c = wp.cos(pyaw)
    s = wp.sin(pyaw)
    nx_n = px + c * 0.15
    ny_n = py + s * 0.15
    pxd = ray_aabb_2d(nx_n, ny_n, c, s, boxes, e, 0.01)
    if pxd > 8.0:
        pxd = 8.0
    plus_x[e] = wp.float32(pxd)

    # pure pursuit
    best_i = int(0)
    best_d = float(1.0e9)
    for k in range(POLY_N):
        dx = float(poly[e, k, 0]) - px
        dy = float(poly[e, k, 1]) - py
        d2 = dx * dx + dy * dy
        if d2 < best_d:
            best_d = d2
            best_i = k
    acc = float(0.0)
    i1 = int(best_i)
    for _k in range(POLY_N):
        if i1 + 1 < POLY_N and acc < look:
            ax = float(poly[e, i1 + 1, 0]) - float(poly[e, i1, 0])
            ay = float(poly[e, i1 + 1, 1]) - float(poly[e, i1, 1])
            acc = acc + wp.sqrt(ax * ax + ay * ay)
            i1 = i1 + 1
    gx = float(poly[e, i1, 0])
    gy = float(poly[e, i1, 1])
    err = wrap_pi(wp.atan2(gy - py, gx - px) - pyaw)
    w = wp.clamp(2.6 * err, -max_w, max_w)
    v = max_v * wp.max(0.0, wp.cos(err)) * wp.max(0.0, wp.cos(err))
    if best_i >= POLY_N - 2:
        v = v * 0.55
    if pxd < plus_stop:
        v = float(0.0)

    nx = px + v * c * dt
    ny = py + v * s * dt
    nyaw = wrap_pi(pyaw + w * dt)
    collided = int(0)
    moved = int(0)
    yaw_only_now = int(0)
    if wp.abs(v) > 0.01 or wp.abs(w) > 0.02:
        if foot_ok(nx, ny, nyaw, boxes, e, 0.06) == 1:
            dist = wp.sqrt((nx - px) * (nx - px) + (ny - py) * (ny - py))
            if dist > 0.002:
                path_m[e] = path_m[e] + wp.float32(dist)
                px = nx
                py = ny
                pyaw = nyaw
                moved = 1
                yaw_only[e] = 0
            elif wp.abs(w) > 0.04 and yaw_only[e] < yaw_cap:
                pyaw = nyaw
                yaw_only[e] = yaw_only[e] + 1
                yaw_only_now = 1
            elif wp.abs(w) > 0.04:
                collided = 1
                w = float(0.0)
        elif wp.abs(w) > 0.04 and yaw_only[e] < yaw_cap and foot_ok(px, py, nyaw, boxes, e, 0.04) == 1:
            pyaw = nyaw
            yaw_only[e] = yaw_only[e] + 1
            yaw_only_now = 1
        else:
            collided = 1
            w = float(0.0)
    if collided == 1:
        coll[e] = coll[e] + 1
        coll_streak[e] = coll_streak[e] + 1
    elif yaw_only_now == 0:
        coll_streak[e] = 0

    x[e] = wp.float32(px)
    y[e] = wp.float32(py)
    yaw[e] = wp.float32(pyaw)
    v_out[e] = wp.float32(v)
    w_out[e] = wp.float32(w)
    t_sim[e] = t_sim[e] + wp.float32(dt)
    steps[e] = steps[e] + 1

    # visit slide (world-aligned)
    gx2 = int(wp.rint(px / VISIT_CELL))
    gy2 = int(wp.rint(py / VISIT_CELL))
    dgx = gx2 - visit_gx[e]
    dgy = gy2 - visit_gy[e]
    visit_gx[e] = gx2
    visit_gy[e] = gy2
    dcol = -dgx
    drow = dgy
    for r in range(VISIT_N):
        for col in range(VISIT_N):
            sr = r - drow
            sc = col - dcol
            val = float(0.0)
            if dgx == 0 and dgy == 0:
                val = float(visit[e, r, col])
            elif sr >= 0 and sr < VISIT_N and sc >= 0 and sc < VISIT_N:
                val = float(visit[e, sr, sc])
            visit_tmp[e, r, col] = wp.float32(val)
    for r in range(VISIT_N):
        for col in range(VISIT_N):
            visit[e, r, col] = visit_tmp[e, r, col]
    # stamp center blob
    cy = VISIT_N // 2
    cx = VISIT_N // 2
    visit[e, cy, cx] = visit[e, cy, cx] + wp.float32(1.0)
    if cy > 0:
        visit[e, cy - 1, cx] = visit[e, cy - 1, cx] + wp.float32(0.35)
    if cy + 1 < VISIT_N:
        visit[e, cy + 1, cx] = visit[e, cy + 1, cx] + wp.float32(0.35)
    if cx > 0:
        visit[e, cy, cx - 1] = visit[e, cy, cx - 1] + wp.float32(0.35)
    if cx + 1 < VISIT_N:
        visit[e, cy, cx + 1] = visit[e, cy, cx + 1] + wp.float32(0.35)

    gained = int(0)
    if moved == 1:
        ix = int(wp.floor((px - FLOOR_X0) / cov_cell))
        iy = int(wp.floor((py - FLOOR_Y0) / cov_cell))
        if ix >= 0 and ix < COV_W and iy >= 0 and iy < COV_H:
            if int(cov[e, iy, ix]) == 0:
                cov[e, iy, ix] = wp.uint8(1)
                cov_cells[e] = cov_cells[e] + 1
                gained = 1
        # room id from 3x2 grid
        rx = 0
        if px > (FLOOR_X0 + FLOOR_X1) * 0.5:
            rx = 1
        if px > FLOOR_X0 + (FLOOR_X1 - FLOOR_X0) * 0.72:
            rx = 2
        ry = 0
        if py > 0.0:
            ry = 1
        bit = 1 << (ry * 3 + rx)
        if (rooms_seen[e] & bit) == 0:
            rooms_seen[e] = rooms_seen[e] | bit

    # beads
    hits = int(0)
    nxt = int(curve_next[e])
    more = int(1)
    for _k in range(3):
        if more == 1 and nxt < 3:
            bx = float(beads[e, nxt, 0])
            by = float(beads[e, nxt, 1])
            dd = wp.sqrt((px - bx) * (px - bx) + (py - by) * (py - by))
            if dd <= bead_r:
                curve_got[e, nxt] = 1
                nxt = nxt + 1
                hits = hits + 1
            else:
                more = 0
    curve_next[e] = nxt

    elapsed = float(t_sim[e]) - float(curve_t0[e])
    win = int(0)
    if nxt >= 3:
        win = 1
    timeout = int(0)
    if elapsed >= curve_s:
        timeout = 1
    if float(plus_x[e]) < plus_stop and elapsed >= 1.0 and nxt == 0:
        timeout = 1
    crash_cut = int(0)
    if coll_streak[e] >= coll_done:
        crash_cut = 1
    seg = int(0)
    if win == 1 or timeout == 1 or crash_cut == 1:
        seg = 1
    seg_done[e] = seg
    rew_acc = float(curve_bead) * float(hits)
    rew_acc = rew_acc + cover_coef * float(gained) * 0.15
    if committed == 1:
        rew_acc = rew_acc - float(wall_c[e]) - float(cont_c[e])
    chord_n = wp.clamp(float(curve_chord[e]) / 4.0, 0.0, 1.0)
    if seg == 1:
        curve_n[e] = curve_n[e] + 1
        beads_sum[e] = beads_sum[e] + wp.float32(float(curve_got[e, 0] + curve_got[e, 1] + curve_got[e, 2]))
        if win == 1:
            rew_acc = rew_acc + curve_win * chord_n
            curve_wins[e] = curve_wins[e] + 1
            chord_win_sum[e] = chord_win_sum[e] + curve_chord[e]
            cut[e] = 1
        else:
            rew_acc = rew_acc - curve_lose
            curve_losses[e] = curve_losses[e] + 1
            if crash_cut == 1:
                cut[e] = 2
            else:
                cut[e] = 3
        need_curve[e] = 1
    need_out[e] = need_curve[e]
    ep_done = int(0)
    if crash_cut == 1:
        ep_done = 1
        cut[e] = 2
    elif steps[e] >= ep_steps:
        ep_done = 1
        cut[e] = 3
    done[e] = ep_done
    rew[e] = wp.float32(rew_acc)


@wp.kernel
def k_rs2(
    boxes: wp.array3d(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    dx: wp.array2d(dtype=wp.float32),
    dy: wp.array2d(dtype=wp.float32),
    depth: wp.array3d(dtype=wp.float32),
):
    e, row, col = wp.tid()
    cyaw = wp.cos(float(yaw[e]))
    syaw = wp.sin(float(yaw[e]))
    ox = float(x[e]) + cyaw * RS2_NOSE
    oy = float(y[e]) + syaw * RS2_NOSE
    oz = RS2_Z
    lx = float(dx[row, col])
    ly = float(dy[row, col])
    lz = -1.0
    inv = 1.0 / wp.sqrt(lx * lx + ly * ly + lz * lz)
    lx = lx * inv
    ly = ly * inv
    lz = lz * inv
    sp = wp.sin(PITCH)
    cp = wp.cos(PITCH)
    # R_rs2 * cam, then yaw. See newton_drive SimRsCameras.
    bx = sp * ly + (-cp) * lz
    by = -lx
    bz = cp * ly + sp * lz
    wx = cyaw * bx - syaw * by
    wy = syaw * bx + cyaw * by
    wz = bz
    t_box = ray_aabb_3d(ox, oy, oz, wx, wy, wz, boxes, e)
    t_floor = float(MAX_RAY)
    if wz < -1.0e-5:
        t_floor = (0.0 - oz) / wz
        if t_floor < 0.04:
            t_floor = float(MAX_RAY)
    t = t_box
    if t_floor < t:
        t = t_floor
    if t >= MAX_RAY:
        depth[e, row, col] = wp.float32(0.0)
    else:
        # optical depth along look (-cam Z)
        lookx = cyaw * cp
        looky = syaw * cp
        lookz = -sp
        along = wp.abs(wx * lookx + wy * looky + wz * lookz)
        if along < 0.15:
            along = 0.15
        depth[e, row, col] = wp.float32(wp.clamp(t * along, 0.0, MAX_RAY))


@wp.kernel
def k_obs(
    boxes: wp.array3d(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    depth: wp.array3d(dtype=wp.float32),
    visit: wp.array3d(dtype=wp.float32),
    last_act: wp.array2d(dtype=wp.float32),
    plus_x: wp.array(dtype=wp.float32),
    t_sim: wp.array(dtype=wp.float32),
    curve_t0: wp.array(dtype=wp.float32),
    curve_got: wp.array2d(dtype=wp.int32),
    curve_chord: wp.array(dtype=wp.float32),
    img: wp.array4d(dtype=wp.float32),
    vec: wp.array2d(dtype=wp.float32),
    curve_s: float,
):
    e, row, col = wp.tid()
    d = float(depth[e, row, col])
    img[e, 0, row, col] = wp.float32(wp.clamp(d / MAX_RAY, 0.0, 1.0))
    # ego height: sample world XY for this output pixel (48x80 from 48x64 range)
    u = (float(col) + 0.5) / float(DEPTH_W)
    v = (float(row) + 0.5) / float(DEPTH_H)
    bx = EGO_X0 + u * (EGO_X1 - EGO_X0)
    by = EGO_Y0 + v * (EGO_Y1 - EGO_Y0)
    c = wp.cos(float(yaw[e]))
    s = wp.sin(float(yaw[e]))
    wx = float(x[e]) + c * bx - s * by
    wy = float(y[e]) + s * bx + c * by
    h = height_at(wx, wy, boxes, e)
    img[e, 1, row, col] = wp.float32(wp.clamp(h / 1.2, 0.0, 1.0))
    vr = v * float(VISIT_N - 1)
    vc = u * float(VISIT_N - 1)
    r0 = int(wp.clamp(vr, 0.0, float(VISIT_N - 1)))
    c0 = int(wp.clamp(vc, 0.0, float(VISIT_N - 1)))
    vis = float(visit[e, r0, c0])
    img[e, 2, row, col] = wp.float32(wp.clamp(vis / 18.0, 0.0, 1.0))
    if row == 0 and col == 0:
        beads = float(curve_got[e, 0] + curve_got[e, 1] + curve_got[e, 2]) / 3.0
        seg_frac = wp.clamp((float(t_sim[e]) - float(curve_t0[e])) / wp.max(curve_s, 1.0e-3), 0.0, 1.0)
        here = float(visit[e, VISIT_N // 2, VISIT_N // 2])
        cc = wp.cos(float(yaw[e]))
        ss = wp.sin(float(yaw[e]))
        pxd = ray_aabb_2d(
            float(x[e]) + cc * 0.15, float(y[e]) + ss * 0.15, cc, ss, boxes, e, 0.01,
        )
        if pxd > 8.0:
            pxd = 8.0
        plus_x[e] = wp.float32(pxd)
        vec[e, 0] = last_act[e, 0]
        vec[e, 1] = last_act[e, 1]
        vec[e, 2] = wp.float32(0.0)
        vec[e, 3] = wp.float32(0.0)
        vec[e, 4] = wp.float32(0.0)
        vec[e, 5] = wp.float32(wp.clamp(pxd / 2.0, 0.0, 1.0))
        vec[e, 6] = wp.float32(seg_frac)
        vec[e, 7] = wp.float32(beads)
        vec[e, 8] = wp.float32(wp.clamp(float(curve_chord[e]) / 4.0, 0.0, 1.0))
        vec[e, 9] = wp.float32(wp.clamp(here / 18.0, 0.0, 1.0))


def pinhole_dxdy(h=DEPTH_H, w=DEPTH_W, vfov=VFOV):
    aspect = float(w) / float(h)
    th = math.tan(vfov * 0.5)
    py, px = np.mgrid[0:h, 0:w]
    u = (px.astype(np.float32) + 0.5) / float(w) - 0.5
    v = (py.astype(np.float32) + 0.5) / float(h) - 0.5
    dx = u * (2.0 * th * aspect)
    dy = -v * (2.0 * th)
    return dx.astype(np.float32), dy.astype(np.float32)
