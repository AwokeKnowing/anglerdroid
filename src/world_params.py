"""Resolved world (house) configuration: the ONE resolver for which house mesh every consumer loads (James 2026-10-21).

Input: worlds/<world>/world.json   world = env KEVIN_WORLD (default casita).  Env KEVIN_WALLS_STL overrides the mesh path
(experiments / rollback).  Derived from the mesh itself (never hardcoded in consumers):
  wall_top_z      = z extent of the walls STL after the frame normalisation (bbox min at the origin)
  ceiling_z       = wall_top_z + ceiling_above_wall_top_m     (ceiling plane: sim ceiling slab underside, maploc map)
  lamp_z          = wall_top_z - lamp_below_wall_top_m        (sim ceiling lamps)
  extent_xy       = (W, H) plan size, offset = -bbox min (sim frame translation)
Counters (2026-10-21, James "add the counters to the model"): world.json "counters" names a counters.json of parameterised
solid volumes (footprint + front side + top height + slab / overhang / toe kick) kept SEPARATE from the walls STL (which is
never edited).  counter_parts() expands them to world-frame boxes / cylinders (sim collision + visuals + 2D GT raster);
house_stl() = walls STL + those solids as one binary STL in run/ (gitignored, cached by md5) for mesh consumers (maploc,
casita_fit, lineloc via LL_STL).  Env KEVIN_COUNTERS=0 disables them, KEVIN_COUNTERS=<path> overrides the file.
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
    cj = env.get("KEVIN_COUNTERS")
    if cj == "0": cpath, csrc = None, "env KEVIN_COUNTERS=0 (disabled)"
    elif cj: cpath, csrc = _abs(cj), "env KEVIN_COUNTERS"
    elif cfg.get("counters"): cpath, csrc = _abs(cfg["counters"]), "input: %s.counters" % os.path.relpath(cp, REPO)
    else: cpath, csrc = None, "none (world.json has no counters)"
    if cpath and not os.path.exists(cpath): raise WorldConfigError("counters file missing: %s" % cpath)
    parts = counter_parts_from(json.load(open(cpath))) if cpath else []
    R["counters_json"] = dict(value=cpath, units="path", source=csrc)
    R["counters_md5"] = dict(value=hashlib.md5(open(cpath, "rb").read()).hexdigest()[:12] if cpath else "", units="", source="derived: md5(counters_json)[:12]")
    R["counter_parts"] = dict(value=len(parts), units="", source="derived: counter_parts() (boxes + cylinders)")
    R["counter_top_z_max"] = dict(value=max([p["z1"] for p in parts], default=0.0), units="m", source="derived: max part top")
    return R


# ---------------------------------------------------------------- floor zones (pure stdlib)
def floor_zones(world=None, env=None):
    """world.json "floor_zones" -> dict(default=<material>, zones=[dict(name, material, poly=[[x, y], ...])]) in the world
    frame; validated (>= 3 vertices, positive area = counter-clockwise).  Worlds without the block: everything 'carpet'."""
    cfg = json.load(open(config_path(world, env)))
    F = cfg.get("floor_zones") or {"default": "carpet", "zones": []}
    for z in F.get("zones", []):
        P = z["poly"]
        if len(P) < 3: raise WorldConfigError("floor zone %s: < 3 vertices" % z["name"])
        a = sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P))) / 2
        if a <= 0: raise WorldConfigError("floor zone %s: polygon must be counter-clockwise (area %.3f)" % (z["name"], a))
    return {"default": F.get("default", "carpet"), "zones": [dict(name=z["name"], material=z["material"], poly=[list(map(float, q)) for q in z["poly"]]) for z in F.get("zones", [])]}


def floor_material_at(x, y, F=None):
    """material of the floor at world (x, y): first zone containing the point (even-odd test), else the default."""
    F = F or floor_zones()
    for z in F["zones"]:
        P = z["poly"]; inside = False
        for i in range(len(P)):
            (x0, y0), (x1, y1) = P[i], P[i - 1]
            if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0: inside = not inside
        if inside: return z["material"]
    return F["default"]


# ---------------------------------------------------------------- counters (pure stdlib)
def _num(c, d, k, default=None): return c[k] if k in c else d.get(k, default)


def counter_parts_from(C):
    """counters.json dict -> list of world-frame parts: dict(kind box|cyl, name, role, cx, cy, hx, hy (box half sizes along
    the frame's s/t), r (cyl), yaw (rad), z0, z1, mat)."""
    import math
    D = C.get("defaults", {}); F = C.get("frames", {"world": {"origin": [0, 0], "yaw_deg": 0}}); out = []
    for c in C["counters"]:
        fr = F[c.get("frame", "world")]; ox, oy = fr["origin"]; yw = math.radians(fr.get("yaw_deg", 0.0)); co, si = math.cos(yw), math.sin(yw)
        def W(s, t): return ox + co * s - si * t, oy + si * s + co * t
        def box(s0, s1, t0, t1, z0, z1, mat, role):
            if s1 - s0 < 1e-6 or t1 - t0 < 1e-6 or z1 - z0 < 1e-6: return
            cx, cy = W((s0 + s1) / 2, (t0 + t1) / 2)
            out.append(dict(kind="box", name=c["name"], role=role, cx=cx, cy=cy, hx=(s1 - s0) / 2, hy=(t1 - t0) / 2, yaw=yw, z0=z0, z1=z1, mat=mat))
        s0, s1 = c["s"]; t0, t1 = c["t"]; top = float(c["top_z"]); th = float(_num(c, D, "top_thick", 0.0))
        ov = float(_num(c, D, "overhang", 0.0)); ob = float(_num(c, D, "overhang_back", 0.0)); tk = _num(c, D, "toe_kick")
        mt, mb = _num(c, D, "mat_top", "dark"), _num(c, D, "mat_body", "white"); f = c["front"]
        if f not in ("s0", "s1", "t0", "t1"): raise WorldConfigError("counter %s: front must be s0|s1|t0|t1" % c["name"])
        def inset(a, front_d, back_d):     # move the front side in by front_d, the opposite side in by back_d
            b = list(a); k = {"s0": (0, 1), "s1": (1, 0), "t0": (2, 3), "t1": (3, 2)}[f]; sg = [1, -1, 1, -1]
            b[k[0]] += sg[k[0]] * front_d; b[k[1]] += sg[k[1]] * back_d; return b
        fp = [s0, s1, t0, t1]
        if th > 0: box(*fp, top - th, top, mt, "top")
        body = inset(fp, ov, ob); zb = top - th
        if tk:
            box(*body, float(tk["h"]), zb, mb, "carcass"); box(*inset(body, float(tk["d"]), 0.0), 0.0, float(tk["h"]), "dark", "toe kick")
        else: box(*body, 0.0, zb, mb, "carcass" if th > 0 else "solid")
        for e in c.get("extra", []):
            m = e.get("mat", mb); z0, z1 = e["z"]
            if "r" in e:
                cx, cy = W(*e["c"]); out.append(dict(kind="cyl", name=c["name"], role=e.get("role", "extra"), cx=cx, cy=cy, r=float(e["r"]), yaw=0.0, z0=z0, z1=z1, mat=m))
            else: box(e["s"][0], e["s"][1], e["t"][0], e["t"][1], z0, z1, m, e.get("role", "extra"))
    return out


def counter_parts():
    p = V["counters_json"]
    return counter_parts_from(json.load(open(p))) if p else []


def _part_tris(p, nseg=48):
    import math
    z0, z1 = p["z0"], p["z1"]
    if p["kind"] == "box":
        co, si = math.cos(p["yaw"]), math.sin(p["yaw"])
        ring = [(p["cx"] + co * a * p["hx"] - si * b * p["hy"], p["cy"] + si * a * p["hx"] + co * b * p["hy"]) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    else:
        ring = [(p["cx"] + p["r"] * math.cos(2 * math.pi * k / nseg), p["cy"] + p["r"] * math.sin(2 * math.pi * k / nseg)) for k in range(nseg)]
    n = len(ring); T = []; cb = (sum(x for x, _ in ring) / n, sum(y for _, y in ring) / n)
    for k in range(n):       # CCW ring: outward side quads, fan caps (convex)
        (ax, ay), (bx, by) = ring[k], ring[(k + 1) % n]
        T += [((ax, ay, z0), (bx, by, z0), (bx, by, z1)), ((ax, ay, z0), (bx, by, z1), (ax, ay, z1))]
        T += [((cb[0], cb[1], z1), (ax, ay, z1), (bx, by, z1)), ((cb[0], cb[1], z0), (bx, by, z0), (ax, ay, z0))]
    return T


def house_stl():
    """walls STL + counter solids as one binary STL (the walls file's own frame and units), cached in run/."""
    w = V["walls_stl"]
    if not V["counters_json"]: return w
    out = os.path.join(REPO, "run", "house_%s_%s.stl" % (V["walls_stl_md5"], V["counters_md5"]))
    if os.path.exists(out): return out
    raw = open(w, "rb").read(); n = int.from_bytes(raw[80:84], "little")
    off = V["walls_offset"]                                   # world = raw (m) + off
    unit = 1000.0 if _raw_span(raw, n) > 100 else 1.0
    recs = []
    for p in counter_parts():
        for tri in _part_tris(p):
            P = [((x - off[0]) * unit, (y - off[1]) * unit, (z - off[2]) * unit) for x, y, z in tri]
            ux, uy, uz = (P[1][i] - P[0][i] for i in range(3)); vx, vy, vz = (P[2][i] - P[0][i] for i in range(3))
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx; L = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
            recs.append(struct.pack("<12fH", nx / L, ny / L, nz / L, *P[0], *P[1], *P[2], 0))
    os.makedirs(os.path.dirname(out), exist_ok=True); tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"house = walls + counters (src/world_params.py)".ljust(80, b" ")); f.write((n + len(recs)).to_bytes(4, "little"))
        f.write(raw[84:84 + 50 * n]); f.write(b"".join(recs))
    os.replace(tmp, out); return out


def _raw_span(raw, n):
    lo = [float("inf")] * 3; hi = [float("-inf")] * 3
    for rec in struct.iter_unpack("<12f2x", raw[84:84 + 50 * n]):
        for k in range(3, 12):
            a = k % 3
            if rec[k] < lo[a]: lo[a] = rec[k]
            if rec[k] > hi[a]: hi[a] = rec[k]
    return max(h - l for h, l in zip(hi, lo))


def values(R): return {k: v["value"] for k, v in R.items()}


V = values(resolve())


def walls_stl(): return V["walls_stl"]


if __name__ == "__main__":
    if sys.argv[1:2] == ["house_stl"]: print(house_stl()); sys.exit(0)
    if sys.argv[1:2] == ["counters"]:
        for p in counter_parts(): print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}))
        sys.exit(0)
    R = resolve()
    if "--json" in sys.argv: print(json.dumps(R, indent=1))
    else:
        for k, v in R.items(): print("%-15s %-60s %s" % (k, v["value"], v["source"]))
