"""Resolved robot config (src/robot_params.py): soft_v3_2 reproduces James's approved a4224de numbers, stock gives back the
pre-tyre numbers, every consumer sees the resolved values, and a bad config refuses to start (fail closed).
python tests/test_robot_params.py   (or pytest)"""
import json, math, os, subprocess, sys, tempfile
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ["KEVIN_ROBOT_PARAMS_NOLOAD"] = "1"          # this test resolves explicitly; consumers are checked in subprocesses
import robot_params as RP

CLEAN = {k: v for k, v in os.environ.items() if k not in ("KEVIN_WHEELS", "KEVIN_TIRE", "KEVIN_VEL_RAMP", "KEVIN_CALIB", "KEVIN_ROBOT",
                                                          "KEVIN_ROBOT_CONFIG", "KEVIN_ROBOT_PARAMS_NOLOAD")}
CLEAN["KEVIN_CALIB"] = "/nonexistent/calibration.json"   # identity calibration (the approved numbers assume none)
SIL12_DECK = [[-0.27371, -0.01022], [-0.1995, -0.15], [-0.08009, -0.20998], [0.08319, -0.20998], [0.12372, -0.15418],
              [0.12629, -0.13818], [0.12629, 0.13818], [0.12372, 0.15418], [0.08319, 0.20998], [-0.08009, 0.20998],
              [-0.1995, 0.15], [-0.27371, 0.01022]]      # pre-tyre FOOTPRINT_SAFE (hull of SIL12 + deck corners)
SOFT_FP = [[-0.27371, -0.01022], [-0.1995, -0.15], [-0.1038, -0.2033], [-0.08009, -0.20998], [0.08319, -0.20998], [0.1038, -0.2033],
           [0.12372, -0.15418], [0.12629, -0.13818], [0.12629, 0.13818], [0.12372, 0.15418], [0.1038, 0.2033], [0.08319, 0.20998],
           [-0.08009, 0.20998], [-0.1038, 0.2033], [-0.1995, 0.15], [-0.27371, 0.01022]]   # a4224de: + tyre corners (+-0.1038, +-0.2033)


def V(w): return RP.values(RP.resolve(wheels=w, env=CLEAN))


def close(a, b, tol=1e-9): return abs(a - b) <= tol


def same_poly(a, b): return len(a) == len(b) and all(close(p[0], q[0]) and close(p[1], q[1]) for p, q in zip(a, b))


def test_soft_matches_a4224de():
    s = V("soft_v3_2"); k = 0.08565 / 0.10065           # a4224de: stock turns * R_stock / R_soft
    assert close(s["vel_limit_tps"], 4.0 * k) and round(s["vel_limit_tps"], 3) == 3.404
    assert close(s["vel_ramp_tps2"], 3.0 * k) and round(s["vel_ramp_tps2"], 3) == 2.553
    assert close(s["vel_ramp_max_tps2"], 5.0 * k) and round(s["vel_ramp_max_tps2"], 3) == 4.255
    assert close(s["cliff_z0_m"], 0.961) and close(s["td_floor_clip_m"], 0.925)
    assert same_poly(s["footprint_safe"], SOFT_FP), s["footprint_safe"]
    assert close(s["rs1_height_m"], 0.962595) and close(s["rs2_height_m"], 0.482863) and close(s["mast_rgb_height_m"], 0.99115)
    assert abs(s["rs1_height_m"] - 0.9626) < 5e-5 and abs(s["rs2_height_m"] - 0.4829) < 5e-5 and abs(s["mast_rgb_height_m"] - 0.9912) < 5e-5
    assert close(s["wheel_r_m"], 0.10065) and close(s["axle_height_m"], 0.10065) and close(s["wheel_diameter_cm"], 20.13)
    assert close(s["a_brake_mps2"], 3.0 * 2 * math.pi * 0.08565) and s["n_spokes"] == 20 and close(s["spoke_ripple_m"], 0.0002)


def test_stock_gives_old_numbers():
    s = V("stock")
    assert close(s["vel_limit_tps"], 4.0) and close(s["vel_ramp_tps2"], 3.0) and close(s["vel_ramp_max_tps2"], 5.0)
    assert close(s["cliff_z0_m"], 0.946) and close(s["td_floor_clip_m"], 0.91) and close(s["td_floor_clip_vision_m"], 0.947595 - 0.041)
    assert same_poly(s["footprint_safe"], SIL12_DECK), s["footprint_safe"]
    assert close(s["rs1_height_m"], 0.947595) and close(s["rs2_height_m"], 0.467863) and close(s["mast_rgb_height_m"], 0.97615)
    assert close(s["wheel_r_m"], 0.08565) and close(s["axle_height_m"], 0.08565) and s["spoke_ripple_m"] == 0.0
    assert close(s["r_circ_m"], math.hypot(0.27371, 0.01022)) and close(s["half_w_m"], 0.20998)


def test_selection_and_overrides():
    assert V(None)["wheels"] == json.load(open(os.path.join(REPO, "robots", "kevin001", "robot.json")))["wheels"] == "soft_v3_2"
    assert RP.values(RP.resolve(env=dict(CLEAN, KEVIN_WHEELS="stock")))["wheels"] == "stock"
    assert RP.values(RP.resolve(env=dict(CLEAN, KEVIN_TIRE="soft")))["wheels"] == "soft_v3_2"          # legacy alias
    r = RP.resolve(env=dict(CLEAN, KEVIN_VEL_RAMP="2.0")); assert r["vel_ramp_tps2"]["value"] == 2.0 and "KEVIN_VEL_RAMP" in r["vel_ramp_tps2"]["source"]
    for e in RP.resolve(wheels="soft_v3_2", env=CLEAN).values(): assert e["source"].startswith(("input", "derived")), e


def _refuses(**kw):
    try: RP.resolve(**kw)
    except RP.RobotConfigError as e: return str(e)
    raise AssertionError("accepted a bad config: %s" % {k: (v if k != "env" else {e: v[e] for e in v if e.startswith("KEVIN")}) for k, v in kw.items()})


def test_fail_closed():
    assert "disagree" in _refuses(env=dict(CLEAN, KEVIN_WHEELS="stock", KEVIN_TIRE="soft_v3_2"))
    assert "unknown wheels" in _refuses(wheels="nope", env=CLEAN)
    assert "missing" in _refuses(env=dict(CLEAN, KEVIN_ROBOT_CONFIG="/nonexistent/robot.json"))
    C = json.load(open(os.path.join(REPO, "robots", "kevin001", "robot.json")))
    for mut in (lambda c: c["drive"].pop("vel_limit_mps"), lambda c: c["drive"].update(vel_ramp_mps2=float("nan")),
                lambda c: c["perception"].update(cliff_z0_ref_m="x"), lambda c: c["drive"].update(vel_limit_mps=50.0)):
        c = json.loads(json.dumps(C)); mut(c)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(c, f)
        _refuses(env=dict(CLEAN, KEVIN_ROBOT_CONFIG=f.name)); os.unlink(f.name)
    # a consumer process (the safety core module) must not import with a bad config
    p = subprocess.run([sys.executable, "-c", "import sys; sys.path[:0]=['src','.']; import nav.safety_core"], cwd=REPO,
                       env=dict(CLEAN, KEVIN_ROBOT_CONFIG="/nonexistent/robot.json"), capture_output=True, text=True)
    assert p.returncode != 0 and "ROBOT CONFIG INVALID" in p.stderr, p.stderr[-500:]


PROBE = r"""
import sys, json; sys.path[:0] = ['src', '.', 'nav/maploc']
import robot_params as RP, twin_geometry as TG, robot_config as RC, tire_profile as TP
import nav.safety_core as SC, nav.perceive as PE, nav.safety as NS
import nav.kplan.core as KC
from perception import ego_rs1, fast_ego80
import maploc
print(json.dumps(dict(hash=RP.HASH, wheels=RP.V['wheels'], tg_ramp=TG.VEL_RAMP_TPS2, tg_lim=TG.VEL_LIMIT_TPS, sc=[SC.RAMP_NORMAL, SC.RAMP_MAX],
  cliff=PE.Ego.CLIFF_Z0, fp=TG.FOOTPRINT_SAFE.tolist(), sc_fp=SC.FOOTPRINT_SAFE.tolist(), ns_abrake=NS.A_BRAKE_MEAS, rc_d=RC.WHEEL_DIAMETER_CM,
  tp=[TP.NAME, TP.RIDE_DZ_M], clip=[float(ego_rs1.TD_FLOOR_CLIP), float(fast_ego80.TD_FLOOR_CLIP)], fwh=fast_ego80.FW_CAM_H,
  h=[TG.height('rs1_depth'), TG.height('rs2_depth'), TG.height('rgb')], ml=maploc.AXLE_H, kc=KC.Limits.a_brake)))
"""


def test_consumers_read_resolved():
    for w in ("stock", "soft_v3_2"):
        p = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO, env=dict(CLEAN, KEVIN_WHEELS=w, KEVIN_PLAN_OWNS_CLEARANCE="1"), capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-1500:]
        o = json.loads(p.stdout.strip().splitlines()[-1]); s = V(w)
        assert o["wheels"] == w and "[robot_params] robot kevin001 wheels %s" % w in p.stderr and "hash %s" % o["hash"] in p.stderr
        assert o["tg_ramp"] == s["vel_ramp_tps2"] and o["tg_lim"] == s["vel_limit_tps"] and o["sc"] == [s["vel_ramp_tps2"], s["vel_ramp_max_tps2"]]
        assert o["cliff"] == s["cliff_z0_m"] and same_poly(o["fp"], s["footprint_safe"]) and same_poly(o["sc_fp"], s["footprint_safe"])
        assert close(o["ns_abrake"], s["a_brake_mps2"]) and close(o["rc_d"], s["wheel_diameter_cm"]) and o["tp"] == [w, s["ride_dz_m"]]
        assert all(close(c, s["td_floor_clip_m"], 1e-6) for c in o["clip"]) and close(o["fwh"], s["rs2_height_m"])
        assert o["h"] == [s["rs1_height_m"], s["rs2_height_m"], s["mast_rgb_height_m"]] and o["ml"] == s["axle_height_m"]
        if o["kc"] is not None: assert close(o["kc"], s["a_brake_mps2"])
        print("consumers %s ok: hash %s" % (w, o["hash"]))
    wb = open(os.path.join(REPO, "src", "wheelbase.py")).read()      # WheelBase needs CAN: check it sets the resolved turns
    assert 'ramp = float(_RP.V["vel_ramp_tps2"])' in wb and 'axis.set_vel_limit(float(_RP.V["vel_limit_tps"]))' in wb


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
