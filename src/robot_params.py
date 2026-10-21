"""Resolved robot configuration: ONE resolver, one flat table, every process reads only this (James 2026-10-21).

Inputs (nothing else holds a tyre- or robot-dependent number):
  robots/<robot>/robot.json   robot = env KEVIN_ROBOT (default kevin001); KEVIN_ROBOT_CONFIG = explicit path.
                              "wheels": the wheel/tyre profile name (src/tire_profiles.json), plus rim drive targets,
                              floor references and the body outline.  Env KEVIN_WHEELS overrides "wheels"
                              (KEVIN_TIRE = legacy alias; both set and different = error).
  src/tire_profiles.json      wheel profiles (loaded/free radius, ride_dz, tread, spokes, sim contact ...)
  src/twin_geometry.json      twin frames per profile (scripts/gen_twin_geometry.py)
  src/calibration.json        optional measured corrections (KEVIN_CALIB path; identity when absent)
  env KEVIN_VEL_RAMP          optional normal ramp override (turns/s^2), as before

resolve() -> {key: {"value", "units", "source"}}; source is "input: <file>.<path>" or "derived: <formula>".
V (plain values) and HASH are computed once at import.  Invalid / missing config raises RobotConfigError at import:
every consumer (safety core included) refuses to start - fail closed.  The resolved table is written to
run/resolved_config_<hash>.json (+ run/resolved_config.json = latest) and logged once per process (stderr).

CLI:  python -m src.robot_params show [--wheels X] [--robot R] [--diff stock] [--json]
"""
import copy, hashlib, json, math, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
RUN_DIR = os.environ.get("KEVIN_RUN_DIR", os.path.join(REPO, "run"))


class RobotConfigError(SystemExit):
    """bad or missing robot configuration: the process must not start (SystemExit so nothing catches it by accident)."""
    def __init__(self, msg): super().__init__("ROBOT CONFIG INVALID (refusing to start): " + msg)


def _load(path, what):
    try:
        with open(path) as f: return json.load(f)
    except FileNotFoundError: raise RobotConfigError("%s missing: %s" % (what, path))
    except Exception as e: raise RobotConfigError("%s unreadable: %s (%s)" % (what, path, e))


def _num(x, name, lo, hi):
    try: v = float(x)
    except Exception: raise RobotConfigError("%s = %r is not a number" % (name, x))
    if not (math.isfinite(v) and lo <= v <= hi): raise RobotConfigError("%s = %r outside [%g, %g]" % (name, v, lo, hi))
    return v


def _hull(P):
    P = sorted(map(tuple, P))
    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0: h.pop()
            h.append(p)
        return h
    lo, up = half(P), half(P[::-1]); return [list(p) for p in lo[:-1] + up[:-1]]


def config_path(robot=None, env=None):
    env = os.environ if env is None else env
    if robot is None and env.get("KEVIN_ROBOT_CONFIG"): return env["KEVIN_ROBOT_CONFIG"]
    return os.path.join(REPO, "robots", robot or env.get("KEVIN_ROBOT") or "kevin001", "robot.json")


def wheels_name(base_wheels, wheels=None, env=None):
    """selected profile name (aliases resolved) and its source string."""
    env = os.environ if env is None else env
    TP = _load(os.path.join(_HERE, "tire_profiles.json"), "tire profiles")
    if wheels is not None: n, src = wheels, "argument"
    else:
        ew, et = env.get("KEVIN_WHEELS", "").strip(), env.get("KEVIN_TIRE", "").strip()
        a = TP["aliases"]
        if ew and et and a.get(ew, ew) != a.get(et, et):
            raise RobotConfigError("KEVIN_WHEELS=%s and KEVIN_TIRE=%s disagree" % (ew, et))
        if ew: n, src = ew, "env KEVIN_WHEELS"
        elif et: n, src = et, "env KEVIN_TIRE (legacy)"
        else: n, src = base_wheels, "robot.json wheels"
    if not n: raise RobotConfigError("no wheels selected (robot.json \"wheels\")")
    n = TP["aliases"].get(n, n)
    if n not in TP["profiles"]: raise RobotConfigError("unknown wheels %r; profiles: %s" % (n, list(TP["profiles"])))
    return n, src, TP


def resolve(robot=None, wheels=None, env=None):
    env = os.environ if env is None else env
    cp = config_path(robot, env); C = _load(cp, "robot config")
    rc = os.path.relpath(cp, REPO)
    name, wsrc, TP = wheels_name(C.get("wheels"), wheels, env)
    P = TP["profiles"][name]; tp = "tire_profiles.%s" % name
    G = _load(os.path.join(_HERE, "twin_geometry.json"), "twin geometry")["profiles"].get(name)
    if G is None: raise RobotConfigError("src/twin_geometry.json has no profile %r (run scripts/gen_twin_geometry.py)" % name)
    calp = env.get("KEVIN_CALIB", os.path.join(_HERE, "calibration.json"))
    CAL = _load(calp, "calibration") if os.path.exists(calp) else {}
    T = {}
    def put(k, v, units, source): T[k] = dict(value=v, units=units, source=source)

    put("robot", C.get("robot"), "", "input: %s robot" % rc)
    put("config_file", rc, "", "input")
    put("wheels", name, "", "input: %s (%s)" % (wsrc, rc if wsrc.startswith("robot") else "override"))
    put("twin_urdf", P["twin_urdf"], "", "input: %s.twin_urdf" % tp)
    put("profile_suffix", P["suffix"], "", "input: %s.suffix" % tp)
    put("calibration_file", os.path.relpath(calp, REPO) if CAL else None, "", "input: KEVIN_CALIB / src/calibration.json (absent = identity)")

    # ---- wheel ---------------------------------------------------------------------------------------------------
    r_load = _num(P["wheel_r_loaded_m"], tp + ".wheel_r_loaded_m", 0.03, 0.3)
    r_free = _num(P.get("wheel_r_free_m", r_load), tp + ".wheel_r_free_m", r_load, 0.3)
    dz = _num(P["ride_dz_m"], tp + ".ride_dz_m", -0.05, 0.1)
    rs = _num(CAL.get("wheel_r_scale", 1.0), "calibration.wheel_r_scale", 0.8, 1.2)
    ts = _num(CAL.get("track_scale", 1.0), "calibration.track_scale", 0.8, 1.2)
    if abs(float(G["wheel_r_m"]) - r_load) > 1e-9 or abs(float(G["axle_height_m"]) - r_load) > 1e-9 or abs(float(G["ride_dz_m"]) - dz) > 1e-9:
        raise RobotConfigError("src/twin_geometry.json [%s] does not match %s (regenerate with scripts/gen_twin_geometry.py)" % (name, tp))
    put("wheel_r_loaded_m", r_load, "m", "input: %s.wheel_r_loaded_m" % tp)
    put("wheel_r_free_m", r_free, "m", "input: %s.wheel_r_free_m" % tp)
    put("ride_dz_m", dz, "m", "input: %s.ride_dz_m (chassis + every sensor rise vs stock)" % tp)
    put("caster_spacer_m", P.get("caster_spacer_m"), "m", "input: %s.caster_spacer_m" % tp)
    put("wheel_mass_kg", P.get("wheel_mass_kg"), "kg", "input: %s.wheel_mass_kg" % tp)
    wr = r_load * rs
    put("wheel_r_m", wr, "m", "derived: wheel_r_loaded_m * calibration.wheel_r_scale (%g)" % rs)
    put("wheel_diameter_cm", 200.0 * wr, "cm", "derived: 200 * wheel_r_m")
    put("track_m", float(G["track_m"]) * ts, "m", "derived: twin_geometry.%s.track_m * calibration.track_scale (%g)" % (name, ts))
    put("axle_height_m", r_load, "m", "derived: = wheel_r_loaded_m (base_link = axle; twin_geometry axle_height_m checked equal)")

    # ---- drive limits (ODrive takes turns) -------------------------------------------------------------------------
    D = C.get("drive") or {}
    circ = 2 * math.pi * wr
    for k, lo, hi, u in (("vel_limit_mps", 0.1, 5.0, "m/s"), ("vel_ramp_mps2", 0.1, 10.0, "m/s^2"), ("vel_ramp_max_mps2", 0.1, 10.0, "m/s^2")):
        put(k, _num(D.get(k), "drive." + k, lo, hi), u, "input: %s drive.%s" % (rc, k))
    put("vel_limit_tps", T["vel_limit_mps"]["value"] / circ, "turns/s", "derived: vel_limit_mps / (2 pi wheel_r_m)")
    if env.get("KEVIN_VEL_RAMP"):
        put("vel_ramp_tps2", _num(env["KEVIN_VEL_RAMP"], "KEVIN_VEL_RAMP", 0.1, 20.0), "turns/s^2", "input: env KEVIN_VEL_RAMP (override)")
    else:
        put("vel_ramp_tps2", T["vel_ramp_mps2"]["value"] / circ, "turns/s^2", "derived: vel_ramp_mps2 / (2 pi wheel_r_m)")
    put("vel_ramp_max_tps2", T["vel_ramp_max_mps2"]["value"] / circ, "turns/s^2", "derived: vel_ramp_max_mps2 / (2 pi wheel_r_m)")
    put("a_brake_mps2", T["vel_ramp_tps2"]["value"] * circ, "m/s^2", "derived: vel_ramp_tps2 * 2 pi wheel_r_m (planner stop model)")

    # ---- sensor heights (twin frame z above the axle + calibration dz + axle height) --------------------------------
    for key, fr in (("rs1_height_m", "rs1_depth"), ("rs2_height_m", "rs2_depth"), ("mast_rgb_height_m", "rgb")):
        f = G["frames"].get(fr)
        if f is None: raise RobotConfigError("twin_geometry.%s has no frame %s" % (name, fr))
        cdz = float(((CAL.get("frames") or {}).get(fr) or {}).get("dxyz", [0, 0, 0])[2])
        put(key, float(f["xyz"][2]) + cdz + r_load, "m",
            "derived: twin_geometry.%s.frames.%s.xyz.z%s + axle_height_m" % (name, fr, " + calibration dz %g" % cdz if cdz else ""))
    put("rs2_pitch_down_deg", float(G["rs2_pitch_down_deg"]), "deg", "input: twin_geometry.%s.rs2_pitch_down_deg" % name)

    # ---- perception floor references ----------------------------------------------------------------------------------
    Q = C.get("perception") or {}
    cz = _num(Q.get("cliff_z0_ref_m"), "perception.cliff_z0_ref_m", 0.3, 2.0)
    fc = _num(Q.get("td_floor_clip_ref_m"), "perception.td_floor_clip_ref_m", 0.3, 2.0)
    oh = _num(Q.get("td_obstacle_h_m"), "perception.td_obstacle_h_m", 0.005, 0.2)
    put("cliff_z0_ref_m", cz, "m", "input: %s perception.cliff_z0_ref_m (stock)" % rc)
    put("cliff_z0_m", cz + dz, "m", "derived: cliff_z0_ref_m + ride_dz_m (expected RS1 depth to the floor)")
    put("td_floor_clip_ref_m", fc, "m", "input: %s perception.td_floor_clip_ref_m (stock)" % rc)
    put("td_floor_clip_m", fc + dz, "m", "derived: td_floor_clip_ref_m + ride_dz_m (perception ego_rs1/fast_ego80 RS1 floor reject)")
    put("td_obstacle_h_m", oh, "m", "input: %s perception.td_obstacle_h_m" % rc)
    put("td_floor_clip_vision_m", T["rs1_height_m"]["value"] - oh, "m", "derived: rs1_height_m - td_obstacle_h_m (src/vision.py RS1 floor reject)")

    # ---- footprint ---------------------------------------------------------------------------------------------------
    B = C.get("body") or {}
    try:
        sil = [[float(a), float(b)] for a, b in B["sil12"]]; deck = [[float(a), float(b)] for a, b in B.get("deck_corners", [])]
    except Exception: raise RobotConfigError("%s body.sil12 / deck_corners invalid" % rc)
    if len(sil) < 3: raise RobotConfigError("body.sil12 needs >= 3 points")
    ty = P.get("tread_y_out_m")
    tyre = [] if ty is None else [[sx * r_free, sy * _num(ty, tp + ".tread_y_out_m", 0.05, 0.5)] for sx in (-1, 1) for sy in (-1, 1)]
    put("sil12", sil, "m (x, y) axle frame", "input: %s body.sil12" % rc)
    put("tread_y_out_m", ty, "m", "input: %s.tread_y_out_m (null = tyre inside SIL12)" % tp)
    put("footprint_tyre_corners", tyre, "m (x, y)", "derived: (+-wheel_r_free_m, +-tread_y_out_m)" if tyre else "derived: none (tread_y_out_m null)")
    fp = _hull(sil + deck + tyre)
    put("footprint_safe", fp, "m (x, y) axle frame", "derived: convex hull(sil12 + body.deck_corners + footprint_tyre_corners)")
    sm = _num(B.get("safety_margin_m"), "body.safety_margin_m", 0.0, 0.2)
    put("safety_margin_m", sm, "m", "input: %s body.safety_margin_m" % rc)
    rc_ = max(math.hypot(x, y) for x, y in fp); hw = max(abs(y) for _, y in fp)
    put("r_circ_m", rc_, "m", "derived: max |footprint_safe| (spin disc)")
    put("half_w_m", hw, "m", "derived: max |y| of footprint_safe")
    put("r_rotate_m", rc_ + sm, "m", "derived: r_circ_m + safety_margin_m")
    put("r_pass_m", hw + sm, "m", "derived: half_w_m + safety_margin_m")

    # ---- sim -----------------------------------------------------------------------------------------------------------
    put("n_spokes", int(P.get("n_spokes", 0) or 0), "", "input: %s.n_spokes" % tp)
    put("spoke_ripple_m", float(P.get("spoke_ripple_m", 0.0) or 0.0), "m", "input: %s.spoke_ripple_m (estimate; emu KEVIN_EMU_RIPPLE overrides)" % tp)
    put("mj_tire_solref", P.get("mj_tire_solref"), "", "input: %s.mj_tire_solref" % tp)
    put("mj_tire_solimp", P.get("mj_tire_solimp"), "", "input: %s.mj_tire_solimp" % tp)
    put("tire_friction", P.get("tire_friction"), "", "input: %s.tire_friction" % tp)
    return T


def values(T): return {k: v["value"] for k, v in T.items()}


def config_hash(T):
    return hashlib.sha256(json.dumps(values(T), sort_keys=True, default=float).encode()).hexdigest()[:12]


def dump(T, h=None):
    """write run/resolved_config_<hash>.json and run/resolved_config.json (latest); returns the hashed path (None if unwritable)."""
    h = h or config_hash(T)
    try:
        os.makedirs(RUN_DIR, exist_ok=True); body = json.dumps(dict(hash=h, resolved=T), indent=1, default=float)
        p = os.path.join(RUN_DIR, "resolved_config_%s.json" % h)
        for q in (p, os.path.join(RUN_DIR, "resolved_config.json")):
            tmp = "%s.%d.tmp" % (q, os.getpid())
            with open(tmp, "w") as f: f.write(body)
            os.replace(tmp, q)
        return p
    except OSError as e:
        print("[robot_params] WARN cannot write %s (%s)" % (RUN_DIR, e), file=sys.stderr); return None


def save_with(directory):
    """copy the resolved table into a recording directory (robot_config.json)."""
    with open(os.path.join(directory, "robot_config.json"), "w") as f: json.dump(dict(hash=HASH, resolved=RESOLVED), f, indent=1, default=float)


def _fmt(v, units=""):
    if isinstance(v, float): return "%.6g" % v
    if isinstance(v, list) and v and isinstance(v[0], list): return "[" + ", ".join("(%s)" % ", ".join("%.4f" % a for a in p) for p in v) + "]"
    return json.dumps(v) if not isinstance(v, str) else v


def _same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def table(T, diff=None, width=44):
    """text table; values longer than `width` go on an indented line below their row."""
    rows, extra = [], {}
    for k, e in T.items():
        if diff is not None and k in diff and _same(diff[k]["value"], e["value"]): continue
        cells = [k, _fmt(e["value"])] + ([_fmt(diff[k]["value"]) if k in diff else "-"] if diff is not None else []) + [e["units"], e["source"]]
        long = [(i, c) for i, c in enumerate(cells[1:-2], 1) if len(c) > width]
        for i, c in long: extra.setdefault(len(rows), []).append(("value" if i == 1 else "other", c)); cells[i] = "(below)"
        rows.append(cells)
    if not rows: return "(no differences)"
    hdr = ["key", "value"] + (["other"] if diff is not None else []) + ["units", "source"]
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    out = ["  ".join(str(c).ljust(w[i]) for i, c in enumerate(hdr)).rstrip()]
    for n, r in enumerate(rows):
        out.append("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip())
        for lab, c in extra.get(n, []): out.append("    %s: %s" % (lab, c))
    return "\n".join(out)


RESOLVED = None; V = {}; HASH = None; DUMP = None
if __name__ != "__main__" and not os.environ.get("KEVIN_ROBOT_PARAMS_NOLOAD"):
    RESOLVED = resolve(); V = values(RESOLVED); HASH = config_hash(RESOLVED); DUMP = dump(RESOLVED, HASH)
    print("[robot_params] robot %s wheels %s (%s) hash %s dump %s" % (V["robot"], V["wheels"], RESOLVED["wheels"]["source"], HASH, DUMP),
          file=sys.stderr, flush=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m src.robot_params", description="resolved robot configuration")
    ap.add_argument("cmd", nargs="?", default="show", choices=["show"])
    ap.add_argument("--robot"); ap.add_argument("--wheels"); ap.add_argument("--diff", metavar="WHEELS", help="show only values that differ from these wheels")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    T = resolve(a.robot, a.wheels); h = config_hash(T)
    if a.json:
        out = dict(hash=h, resolved=T)
        if a.diff: out["diff_against"] = dict(wheels=a.diff, resolved=resolve(a.robot, a.diff))
        print(json.dumps(out, indent=1, default=float)); return
    print("robot %s  wheels %s  hash %s  (%s)" % (T["robot"]["value"], T["wheels"]["value"], h, T["config_file"]["value"]))
    if a.diff:
        O = resolve(a.robot, a.diff); print("differences vs wheels=%s (hash %s); 'other' = %s value\n" % (O["wheels"]["value"], config_hash(O), a.diff))
        print(table(T, O))
    else: print(table(T))


if __name__ == "__main__":
    main()
