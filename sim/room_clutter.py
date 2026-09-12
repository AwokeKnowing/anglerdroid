"""Casita furniture: couches, desks, tables, beds.

Real houses leave vacuum-width aisles, not open gym floor. Each room gets a
kit of wall-anchored pieces; density picks how many. Door columns, spawn, and
people stay clear so Kevin can still tour.
"""
from __future__ import annotations

import math
import random

# Measured SW→center gap is east of the partition end (~x=-1.27), not x=-1.15.
# Keep a robot-wide column through every interior opening.
DOOR_CORRIDORS = (
    (-1.05, 0.12, -3.50, -2.15),   # SW portal
    (-0.35, 0.55, -3.95, -3.15),   # south → center
    (-1.50, -0.50, -0.55, 0.55),   # west_mid → center
    (0.80, 1.60, -0.50, 0.55),     # center → east
    (-0.45, 0.55, 0.80, 1.65),     # into north
    (-0.40, 0.55, 5.15, 5.85),     # far north
)

# kind, cx, cy, hx, hy, hz  (yaw=0, world XY)
# Sized like real furniture; placed to leave ≥ ~0.55 m aisles.
_KITS = {
    "sw": (
        ("bed", -2.98, -6.72, 0.40, 0.58, 0.28),
        ("desk", -1.78, -7.08, 0.52, 0.24, 0.38),
        ("couch", -2.62, -3.12, 0.70, 0.30, 0.40),
        ("chair", -1.62, -4.55, 0.30, 0.30, 0.42),
    ),
    "south": (
        ("couch", 0.55, -7.08, 1.05, 0.30, 0.40),
        ("table", 0.25, -5.72, 0.52, 0.42, 0.38),
        ("desk", 3.32, -6.25, 0.26, 0.72, 0.38),
        ("chair", 1.55, -4.35, 0.30, 0.30, 0.42),
        ("couch", 2.55, -3.92, 0.85, 0.28, 0.40),
    ),
    "west_mid": (
        ("desk", -3.18, -0.85, 0.24, 0.70, 0.38),
        ("couch", -2.55, 0.55, 0.80, 0.30, 0.40),
        ("table", -1.85, -1.55, 0.42, 0.42, 0.38),
    ),
    "center": (
        ("table", 0.48, -0.35, 0.50, 0.50, 0.38),
        ("couch", 0.88, 1.15, 0.72, 0.28, 0.40),
        ("chair", -0.15, 0.85, 0.28, 0.28, 0.42),
    ),
    "east_s": (
        ("couch", 2.95, -2.15, 0.55, 0.85, 0.40),
        ("desk", 3.32, -0.85, 0.24, 0.55, 0.38),
        ("table", 1.95, -1.55, 0.40, 0.40, 0.38),
    ),
    "east_mid": (
        ("desk", 3.30, 0.85, 0.26, 0.58, 0.38),
        ("chair", 2.15, 1.15, 0.30, 0.30, 0.42),
    ),
    "north_w": (
        ("couch", -2.55, 1.85, 0.90, 0.30, 0.40),
        ("table", -1.55, 3.55, 0.48, 0.48, 0.38),
        ("desk", -3.18, 4.35, 0.24, 0.65, 0.38),
        ("chair", -0.85, 2.55, 0.28, 0.28, 0.42),
    ),
    "north_e": (
        ("couch", 2.95, 2.25, 0.55, 0.80, 0.40),
        ("table", 1.15, 3.15, 0.48, 0.42, 0.38),
        ("chair", 2.55, 4.85, 0.30, 0.30, 0.42),
    ),
    "far_north": (
        ("couch", -1.55, 6.95, 1.10, 0.30, 0.40),
        ("table", 1.55, 6.45, 0.50, 0.42, 0.38),
    ),
}

_COLORS = {
    "couch": (0.42, 0.28, 0.22, 1.0),
    "bed": (0.55, 0.48, 0.62, 1.0),
    "desk": (0.45, 0.32, 0.18, 1.0),
    "table": (0.62, 0.48, 0.28, 1.0),
    "chair": (0.28, 0.42, 0.38, 1.0),
    "box": (0.70, 0.40, 0.22, 1.0),
}

_DENSITY_N = {
    "empty": 0.0,
    "sparse": 0.35,
    "medium": 0.60,
    "dense": 0.85,
    "packed": 1.0,
}
_DENSITY_NAMES = ("sparse", "medium", "dense", "packed")


def in_door_corridor(x, y, hx=0.0, hy=0.0, pad=0.08):
    for x0, x1, y0, y1 in DOOR_CORRIDORS:
        if (x0 - hx - pad) <= x <= (x1 + hx + pad) and (y0 - hy - pad) <= y <= (y1 + hy + pad):
            return True
    return False


def _aabb_hit(a, b, gap):
    return (
        abs(a["cx"] - b["cx"]) < (a["hx"] + b["hx"] + gap)
        and abs(a["cy"] - b["cy"]) < (a["hy"] + b["hy"] + gap)
    )


def _ok(piece, spawn, people_xy, placed, aisle=0.16):
    x, y = piece["cx"], piece["cy"]
    hx, hy = piece["hx"], piece["hy"]
    if in_door_corridor(x, y, hx, hy, pad=0.10):
        return False
    if spawn is not None and math.hypot(x - spawn[0], y - spawn[1]) < 0.85 + max(hx, hy):
        return False
    for px, py in people_xy or ():
        if math.hypot(x - px, y - py) < 0.70 + max(hx, hy):
            return False
    for q in placed:
        if _aabb_hit(piece, q, aisle):
            return False
    return True


def _piece(kind, cx, cy, hx, hy, hz, yaw=0.0):
    rgba = _COLORS.get(kind, _COLORS["box"])
    return {
        "kind": kind,
        "cx": float(cx),
        "cy": float(cy),
        "cz": float(hz),
        "yaw": float(yaw),
        "hx": float(hx),
        "hy": float(hy),
        "hz": float(hz),
        "rgba": rgba,
    }


def pick_density(spec, seed):
    raw = str(spec or "random").strip().lower()
    if raw in ("random", "auto"):
        name = random.Random(int(seed) + 41).choice(_DENSITY_NAMES)
        return name, _DENSITY_N[name]
    if raw in _DENSITY_N:
        return raw, _DENSITY_N[raw]
    try:
        v = float(raw)
    except ValueError:
        return "medium", _DENSITY_N["medium"]
    if v >= 1.5:
        return "n=%d" % int(v), v
    if v > 1.0:
        return "packed", 1.0
    return "custom", max(0.0, min(1.0, v))


def layout_furniture(seed=3, spec="random", spawn=None, people_xy=None, per_room=True):
    """Return obstacle dicts. per_room=True rolls a density for each room."""
    rng = random.Random(int(seed) + 7)
    house_name, house_frac = pick_density(spec, seed)
    placed = []
    room_notes = []
    for room, kit in _KITS.items():
        if per_room and str(spec).lower() in ("random", "auto"):
            name = rng.choice(_DENSITY_NAMES)
            frac = _DENSITY_N[name]
        else:
            name, frac = house_name, house_frac
        if isinstance(frac, float) and frac > 1.5:
            n = int(frac)
        else:
            n = int(round(len(kit) * float(frac)))
        n = max(0, min(len(kit), n))
        order = list(kit)
        rng.shuffle(order)
        got = 0
        for kind, cx, cy, hx, hy, hz in order:
            if got >= n:
                break
            jx = rng.uniform(-0.08, 0.08)
            jy = rng.uniform(-0.08, 0.08)
            p = _piece(kind, cx + jx, cy + jy, hx, hy, hz)
            if not _ok(p, spawn, people_xy, placed):
                p = _piece(kind, cx, cy, hx, hy, hz)
                if not _ok(p, spawn, people_xy, placed):
                    continue
            placed.append(p)
            got += 1
        if name == "packed":
            for _ in range(2):
                cx = rng.uniform(-3.1, 3.3)
                cy = rng.uniform(-6.9, 6.9)
                hx = rng.uniform(0.22, 0.34)
                hy = rng.uniform(0.22, 0.34)
                p = _piece("box", cx, cy, hx, hy, rng.uniform(0.28, 0.42))
                if _ok(p, spawn, people_xy, placed, aisle=0.50):
                    placed.append(p)
                    got += 1
        room_notes.append("%s:%s/%d" % (room, name, got))
    layout_furniture.spec = house_name if str(spec).lower() not in ("random", "auto") else "random"
    layout_furniture.rooms = ",".join(room_notes)
    layout_furniture.n = len(placed)
    return placed


def add_furniture_shapes(builder, pieces, static=True):
    import warp as wp
    import newton

    cfg = newton.ModelBuilder.ShapeConfig(density=80.0, mu=0.7)
    for i, p in enumerate(pieces):
        x, y, hz = p["cx"], p["cy"], p["hz"]
        rgba = p.get("rgba", _COLORS["box"])
        color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
        if static:
            body = -1
            xf = wp.transform(p=wp.vec3(x, y, hz), q=wp.quat_identity())
        else:
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, y, hz), q=wp.quat_identity()),
                label="furniture_%d" % i,
            )
            xf = None
        builder.add_shape_box(
            body,
            xform=xf,
            hx=p["hx"],
            hy=p["hy"],
            hz=p["hz"],
            cfg=cfg,
            color=color,
            label="%s_%d" % (p.get("kind", "box"), i),
        )
