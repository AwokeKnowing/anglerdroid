"""Resolved world config (src/world_params.py): every consumer loads the ONE house mesh named in worlds/casita/world.json,
heights are derived from that mesh, KEVIN_WALLS_STL overrides it, a missing mesh fails closed, and no tool can overwrite
the committed models.   python tests/test_world_params.py   (or pytest)"""
import os, subprocess, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import world_params as WP

ZFIX = os.path.join(REPO, "sim", "casitahouse31-walls-laser-zfix.stl")
OLD = os.path.join(REPO, "sim", "casitahouse-walls-fixed.stl")
ENV = {k: v for k, v in os.environ.items() if k not in ("KEVIN_WALLS_STL", "KEVIN_WORLD", "ML_STL", "LL_STL", "KEVIN_COUNTERS")}


def test_default_is_zfix_and_heights_derived():
    V = WP.values(WP.resolve(env=ENV))
    assert V["walls_stl"] == ZFIX
    assert abs(V["wall_top_z"] - 2.1897) < 1e-3 and abs(V["ceiling_z"] - (V["wall_top_z"] + 0.01)) < 1e-9
    assert abs(V["lamp_z"] - (V["wall_top_z"] - 0.13)) < 1e-9
    assert abs(V["extent_xy"][0] - 7.2283) < 1e-3 and abs(V["extent_xy"][1] - 14.7579) < 1e-3


def test_override_old_model_reproduces_old_constants():
    V = WP.values(WP.resolve(env=dict(ENV, KEVIN_WALLS_STL=OLD)))
    assert V["walls_stl"] == OLD
    assert abs(V["wall_top_z"] - 2.28) < 1e-4 and abs(V["ceiling_z"] - 2.29) < 1e-4 and abs(V["lamp_z"] - 2.15) < 1e-4


def test_missing_mesh_fails_closed():
    try: WP.resolve(env=dict(ENV, KEVIN_WALLS_STL="/nonexistent/walls.stl"))
    except SystemExit as e: assert "WORLD CONFIG INVALID" in str(e); return
    raise AssertionError("missing mesh accepted")


def _py(code, **env):
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=dict(ENV, **env), capture_output=True, text=True, timeout=120)


def test_consumers_resolve_the_world_mesh():
    code = ("import sys; sys.path[:0]=['sim','nav/maploc','tools/lineloc','tools/casita_fit']\n"
            "import house_mesh, lineloc, camgeo, maploc\n"
            "print(house_mesh.FIXED_STL); print(house_mesh.REPAIRED_STL); print(lineloc.world_stl()); print(camgeo.world_stl()); print(maploc.world_stl())\n")
    r = _py(code); assert r.returncode == 0, r.stderr
    fixed, repaired, ll, cg, ml = r.stdout.split()
    assert fixed == ll == ZFIX, r.stdout          # walls-only consumers (lineloc: rebuild its map with LL_STL=house_stl)
    assert cg == ml and os.path.basename(cg).startswith("house_") and os.path.dirname(cg) == os.path.join(REPO, "run"), r.stdout
    assert os.path.dirname(repaired) == os.path.join(REPO, "run") and not repaired.endswith(("zfix.stl", "fixed.stl"))
    r = _py(code, KEVIN_WALLS_STL=OLD); assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == OLD



def test_counters_resolved_and_solid():
    V = WP.values(WP.resolve(env=ENV))
    assert V["counters_json"] == os.path.join(REPO, "worlds", "casita", "counters.json") and V["counter_parts"] > 0
    parts = WP.counter_parts_from(__import__("json").load(open(V["counters_json"])))
    names = {p["name"] for p in parts}
    assert {"passthrough", "kitchen_east", "kitchen_north", "island_base", "island_bar"} <= names
    for p in parts:
        assert p["z1"] > p["z0"] >= 0 and p["z1"] <= V["wall_top_z"] and p["kind"] in ("box", "cyl")
        if p["kind"] == "box": assert p["hx"] > 0 and p["hy"] > 0
    pt = [p for p in parts if p["name"] == "passthrough"]
    assert abs(max(p["z1"] for p in pt) - 0.802) < 1e-9 and min(p["cy"] - p["hy"] for p in pt) == 4.481   # top front edge
    V0 = WP.values(WP.resolve(env=dict(ENV, KEVIN_COUNTERS="0")))
    assert V0["counters_json"] is None and V0["counter_parts"] == 0


def test_house_stl_keeps_the_walls_frame():
    p = WP.house_stl(); assert p != WP.walls_stl() and os.path.dirname(p) == os.path.join(REPO, "run")
    lo, hi = WP.stl_bounds(p); lo0, hi0 = WP.stl_bounds(WP.walls_stl())
    assert all(abs(a - b) < 1e-6 for a, b in zip(lo + hi, lo0 + hi0))          # counters inside the walls bbox: same frame
    n = lambda f: int.from_bytes(open(f, "rb").read()[80:84], "little")
    assert n(p) > n(WP.walls_stl()) and open(p, "rb").read()[84:84 + 50 * n(WP.walls_stl())] == open(WP.walls_stl(), "rb").read()[84:84 + 50 * n(WP.walls_stl())]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
