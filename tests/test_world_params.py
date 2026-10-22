"""Resolved world config (src/world_params.py): every consumer loads the ONE house mesh named in worlds/casita/world.json,
heights are derived from that mesh, KEVIN_WALLS_STL overrides it, a missing mesh fails closed, and no tool can overwrite
the committed models.   python tests/test_world_params.py   (or pytest)"""
import os, subprocess, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import world_params as WP

ZFIX = os.path.join(REPO, "sim", "casitahouse31-walls-laser-zfix.stl")
OLD = os.path.join(REPO, "sim", "casitahouse-walls-fixed.stl")
ENV = {k: v for k, v in os.environ.items() if k not in ("KEVIN_WALLS_STL", "KEVIN_WORLD", "ML_STL", "LL_STL")}


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
            "import house_mesh, lineloc, camgeo\n"
            "print(house_mesh.FIXED_STL); print(house_mesh.REPAIRED_STL); print(lineloc.world_stl()); print(camgeo.world_stl())\n")
    r = _py(code); assert r.returncode == 0, r.stderr
    fixed, repaired, ll, cg = r.stdout.split()
    assert fixed == ll == cg == ZFIX, r.stdout
    assert os.path.dirname(repaired) == os.path.join(REPO, "run") and not repaired.endswith(("zfix.stl", "fixed.stl"))
    r = _py(code, KEVIN_WALLS_STL=OLD); assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == OLD


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
