#!/usr/bin/env python3
"""Procedural house-like floorplans for insect-explore (not the casita STL)."""
from __future__ import annotations

import math
import random

import numpy as np

from visit_wander import FLOOR_X0, FLOOR_X1, FLOOR_Y0, FLOOR_Y1

WALL_T = 0.08
WALL_H = 2.15
DOOR_W = (0.88, 1.18)
MIN_ROOM = 2.15


def _wall_box(x0, x1, y0, y1, z0=0.0, z1=WALL_H):
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)
    return {
        "cx": float(cx), "cy": float(cy), "cz": float(cz),
        "hx": max(0.5 * abs(x1 - x0), 0.03),
        "hy": max(0.5 * abs(y1 - y0), 0.03),
        "hz": max(0.5 * abs(z1 - z0), 0.05),
        "yaw": 0.0,
    }


def _vert_wall(x, y0, y1, gap=None):
    t = WALL_T
    if gap is None:
        return [_wall_box(x - t, x + t, y0, y1)]
    g0, g1 = gap
    g0, g1 = max(y0, min(g0, g1)), min(y1, max(g0, g1))
    out = []
    if g0 - y0 > 0.35:
        out.append(_wall_box(x - t, x + t, y0, g0))
    if y1 - g1 > 0.35:
        out.append(_wall_box(x - t, x + t, g1, y1))
    return out


def _horz_wall(y, x0, x1, gap=None):
    t = WALL_T
    if gap is None:
        return [_wall_box(x0, x1, y - t, y + t)]
    g0, g1 = gap
    g0, g1 = max(x0, min(g0, g1)), min(x1, max(g0, g1))
    out = []
    if g0 - x0 > 0.35:
        out.append(_wall_box(x0, g0, y - t, y + t))
    if x1 - g1 > 0.35:
        out.append(_wall_box(g1, x1, y - t, y + t))
    return out


def _bsp(rng, rect, depth, walls, rooms):
    x0, x1, y0, y1 = rect
    w, h = x1 - x0, y1 - y0
    can_x = w >= 2.0 * MIN_ROOM + 0.2
    can_y = h >= 2.0 * MIN_ROOM + 0.2
    if depth <= 0 or (not can_x and not can_y):
        rooms.append((x0, x1, y0, y1))
        return
    split_x = can_x if (can_x and not can_y) else (can_x and (w >= h or not can_y))
    dw = rng.uniform(*DOOR_W)
    if split_x:
        cut = rng.uniform(x0 + MIN_ROOM, x1 - MIN_ROOM)
        mid = rng.uniform(y0 + 0.55 + 0.5 * dw, y1 - 0.55 - 0.5 * dw)
        walls.extend(_vert_wall(cut, y0, y1, gap=(mid - 0.5 * dw, mid + 0.5 * dw)))
        _bsp(rng, (x0, cut, y0, y1), depth - 1, walls, rooms)
        _bsp(rng, (cut, x1, y0, y1), depth - 1, walls, rooms)
    else:
        cut = rng.uniform(y0 + MIN_ROOM, y1 - MIN_ROOM)
        mid = rng.uniform(x0 + 0.55 + 0.5 * dw, x1 - 0.55 - 0.5 * dw)
        walls.extend(_horz_wall(cut, x0, x1, gap=(mid - 0.5 * dw, mid + 0.5 * dw)))
        _bsp(rng, (x0, x1, y0, cut), depth - 1, walls, rooms)
        _bsp(rng, (x0, x1, cut, y1), depth - 1, walls, rooms)


def _furniture_in_room(rng, rect, n):
    x0, x1, y0, y1 = rect
    pad = 0.55
    out = []
    for _ in range(n):
        hx = rng.uniform(0.22, 0.55)
        hy = rng.uniform(0.22, 0.70)
        cx = rng.uniform(x0 + pad + hx, x1 - pad - hx) if x1 - x0 > 2 * (pad + hx) + 0.2 else None
        cy = rng.uniform(y0 + pad + hy, y1 - pad - hy) if y1 - y0 > 2 * (pad + hy) + 0.2 else None
        if cx is None or cy is None:
            continue
        hz = rng.uniform(0.28, 0.48)
        out.append({
            "cx": float(cx), "cy": float(cy), "cz": float(hz),
            "hx": float(hx), "hy": float(hy), "hz": float(hz),
            "yaw": 0.0,
        })
    return out


def boxes_to_mesh(boxes):
    """Indexed triangle mesh for VisualMeshCaster."""
    verts = []
    faces = []

    def add_box(b):
        c, s = math.cos(float(b.get("yaw", 0.0))), math.sin(float(b.get("yaw", 0.0)))
        cx, cy, cz = float(b["cx"]), float(b["cy"]), float(b["cz"])
        hx, hy, hz = float(b["hx"]), float(b["hy"]), float(b["hz"])
        base = len(verts)
        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    lx = (-hx if ix == 0 else hx)
                    ly = (-hy if iy == 0 else hy)
                    lz = (-hz if iz == 0 else hz)
                    verts.append((cx + c * lx - s * ly, cy + s * lx + c * ly, cz + lz))
        def i(ix, iy, iz):
            return base + ix * 4 + iy * 2 + iz
        quads = (
            (i(0, 0, 0), i(1, 0, 0), i(1, 1, 0), i(0, 1, 0)),
            (i(0, 0, 1), i(0, 1, 1), i(1, 1, 1), i(1, 0, 1)),
            (i(0, 0, 0), i(0, 0, 1), i(1, 0, 1), i(1, 0, 0)),
            (i(0, 1, 0), i(1, 1, 0), i(1, 1, 1), i(0, 1, 1)),
            (i(0, 0, 0), i(0, 1, 0), i(0, 1, 1), i(0, 0, 1)),
            (i(1, 0, 0), i(1, 0, 1), i(1, 1, 1), i(1, 1, 0)),
        )
        for a, b, c, d in quads:
            faces.append((a, b, c))
            faces.append((a, c, d))

    for b in boxes:
        add_box(b)
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def make_layout(seed: int):
    rng = random.Random(int(seed) + 17)
    x0, x1 = FLOOR_X0 + 0.04, FLOOR_X1 - 0.04
    y0, y1 = FLOOR_Y0 + 0.04, FLOOR_Y1 - 0.04
    walls = []
    walls.extend(_horz_wall(y0, x0, x1))
    walls.extend(_horz_wall(y1, x0, x1))
    walls.extend(_vert_wall(x0, y0, y1))
    walls.extend(_vert_wall(x1, y0, y1))
    rooms = []
    depth = rng.choice((3, 3, 4, 4, 5))
    _bsp(rng, (x0 + WALL_T, x1 - WALL_T, y0 + WALL_T, y1 - WALL_T), depth, walls, rooms)
    furniture = []
    spawns = []
    people = []
    named = []
    for i, rec in enumerate(rooms):
        rx0, rx1, ry0, ry1 = rec
        named.append(("r%d" % i, rx0, rx1, ry0, ry1))
        n_f = rng.choice((0, 1, 1, 2, 2, 3))
        furniture.extend(_furniture_in_room(rng, rec, n_f))
        cx, cy = 0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1)
        spawns.append((cx, cy))
        if rng.random() < 0.45 and (rx1 - rx0) > 1.6 and (ry1 - ry0) > 1.6:
            people.append((
                rng.choice(("Ana", "Marco", "Priya", "Jules", "Sam", "Nia")),
                (cx + rng.uniform(-0.4, 0.4), cy + rng.uniform(-0.4, 0.4)),
            ))
    return {
        "walls": walls,
        "furniture": furniture,
        "rooms": named,
        "spawns": spawns,
        "people": people[:4] or [("Ana", spawns[0] if spawns else (0.0, 0.0))],
        "mins": (x0, y0, 0.0),
        "maxs": (x1, y1, WALL_H),
    }


def add_boxes_to_builder(builder, boxes, prefix):
    import warp as wp
    import newton
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.4, gap=0.0)
    for i, b in enumerate(boxes):
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(b.get("yaw", 0.0)))
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                p=wp.vec3(float(b["cx"]), float(b["cy"]), float(b["cz"])),
                q=q,
            ),
            hx=float(b["hx"]),
            hy=float(b["hy"]),
            hz=float(b["hz"]),
            cfg=cfg,
            color=(0.86, 0.80, 0.70) if prefix == "wall" else (0.50, 0.38, 0.28),
            label="%s_%d" % (prefix, i),
        )
