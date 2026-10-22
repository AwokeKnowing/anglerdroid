"""Resolved world (house) configuration: the ONE resolver for which house mesh every consumer loads (James 2026-10-21).

Input: worlds/<world>/world.json   world = env KEVIN_WORLD (default casita).  Env KEVIN_WALLS_STL overrides the mesh path
(experiments / rollback).  Derived from the mesh itself (never hardcoded in consumers):
  wall_top_z      = z extent of the walls STL after the frame normalisation (bbox min at the origin)
  ceiling_z       = wall_top_z + ceiling_above_wall_top_m     (ceiling plane: sim ceiling slab underside, maploc map)
  lamp_z          = wall_top_z - lamp_below_wall_top_m        (sim ceiling lamps)
  extent_xy       = (W, H) plan size, offset = -bbox min (sim frame translation)
Pure stdlib (the robot venvs have no trimesh; system python has no numpy).  CLI: python -m src.world_params show [--json]
"""
import json, os, sys, hashlib, struct

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)


class WorldConfigError(SystemExit):
    def __init__(self, msg): super().__init__("WORLD CONFIG INVALID: " + msg)


def config_path(world=None, env=None):
    env = os.environ if env is None else env
    return os.path.join(REPO, "worlds", world or env.get("KEVIN_WORLD") or "casita", "world.json")


def _abs(p): return p if os.path.isabs(p) else os.path.join(REPO, p)


def stl_bounds(path):
    """(min xyz, max xyz) of a binary STL (mm files scaled to m like every consumer: span > 100 -> x 0.001)."""
    raw = open(path, "rb").read()
    if raw[:5].lower() == b"solid" and b"\x00" not in raw[:84]:
        raise WorldConfigError("ASCII STL not supported: %s" % path)
    n = int.from_bytes(raw[80:84], "little"); lo = [float("inf")] * 3; hi = [float("-inf")] * 3
    for rec in struct.iter_unpack("<12f2x", raw[84:84 + 50 * n]):
        for k in range(3, 12):
            a = k % 3; v = rec[k]
            if v < lo[a]: lo[a] = v
            if v > hi[a]: hi[a] = v
    if max(h - l for h, l in zip(hi, lo)) > 100: lo, hi = [x * 0.001 for x in lo], [x * 0.001 for x in hi]
    return lo, hi


def resolve(world=None, env=None):
    env = os.environ if env is None else env
    cp = config_path(world, env)
    try: cfg = json.load(open(cp))
    except FileNotFoundError: raise WorldConfigError("missing %s" % cp)
    stl = env.get("KEVIN_WALLS_STL") or cfg["walls_stl"]
    src = "env KEVIN_WALLS_STL" if env.get("KEVIN_WALLS_STL") else "input: %s.walls_stl" % os.path.relpath(cp, REPO)
    p = _abs(stl)
    if not os.path.exists(p): raise WorldConfigError("walls STL missing: %s" % p)
    lo, hi = stl_bounds(p); ext = [h - l for h, l in zip(hi, lo)]; top = float(ext[2])
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
    R = {
        "world": dict(value=cfg.get("world", world or "casita"), units="", source="input: " + os.path.relpath(cp, REPO)),
        "walls_stl": dict(value=p, units="path", source=src),
        "walls_stl_md5": dict(value=md5, units="", source="derived: md5(walls_stl)[:12]"),
        "walls_offset": dict(value=[float(-x) for x in lo], units="m", source="derived: -bbox_min(walls_stl)"),
        "extent_xy": dict(value=[float(ext[0]), float(ext[1])], units="m", source="derived: bbox size x, y"),
        "wall_top_z": dict(value=top, units="m", source="derived: bbox size z (floor = wall bottom = 0)"),
        "ceiling_z": dict(value=top + float(cfg["ceiling_above_wall_top_m"]), units="m", source="derived: wall_top_z + ceiling_above_wall_top_m"),
        "lamp_z": dict(value=top - float(cfg["lamp_below_wall_top_m"]), units="m", source="derived: wall_top_z - lamp_below_wall_top_m"),
    }
    return R


def values(R): return {k: v["value"] for k, v in R.items()}


V = values(resolve())


def walls_stl(): return V["walls_stl"]


if __name__ == "__main__":
    R = resolve()
    if "--json" in sys.argv: print(json.dumps(R, indent=1))
    else:
        for k, v in R.items(): print("%-15s %-60s %s" % (k, v["value"], v["source"]))
