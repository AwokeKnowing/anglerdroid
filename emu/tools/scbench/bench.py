"""Fast offline safety bench: runs nav.safety_core.decide (+ nav.governor.governor) and the prototype
emu.tools.scbench.proto_decel on synthetic scenes, open-loop (one decision) and closed-loop (30 Hz core, 1 ms plant with
command latency + actuator lag + per-wheel ramp, like the ODrive).  Seconds, no sim.
  python -m emu.tools.scbench.bench [scene ...]   (PYTHONPATH=.:src)
Perception is ideal (true points every tick, no frame age) unless --age S (motion-compensation error model: points
observed S ago, compensated with the commanded instead of the actual twist)."""
from __future__ import annotations
import argparse, collections, json, math, sys, time
import numpy as np
from nav import safety_core as S
from nav import governor as GOV
from nav import limits as LIM
RULES = ("core", "wired", "proto")      # core = safety_core with SC_DECEL off (bands/brake/escape); wired = SC_DECEL on
from emu.tools.scbench import proto_decel as PD

HW = float(np.abs(S.FOOTPRINT_SAFE[:, 1]).max()); XF = float(S.FOOTPRINT_SAFE[:, 0].max()); XR = float(S.FOOTPRINT_SAFE[:, 0].min())
DT_PLANT = 0.001; LAG = PD.LAG; T_CMD = S.T_CMD; A_UP = S.BRAKE[S.RAMP_NORMAL][1]


class StubBands:
    """core bands without an Ego: front strip from the footprint front edge, rear strip at the footprint rear edge."""
    def __init__(self): self.hw, self.x_front, self.x_rear = HW, XF + 0.005, XR - 0.005
    front = S.Bands.front; rear = S.Bands.rear; rects = S.Bands.rects


def line(p0, p1, step=0.01):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); n = max(2, int(np.linalg.norm(p1 - p0) / step) + 1)
    return np.linspace(p0, p1, n)


def post(c, r=0.015, n=12):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False); return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]


def const(v, w): return lambda t, pose: (v, w)


# scene: world points, start pose, planner policy, duration, success test(pose, log) -> bool, what 'correct' means
SCENES = collections.OrderedDict()
def scene(name, pts, policy, T, goal, expect, start=(0.0, 0.0, 0.0), escape=False):
    SCENES[name] = dict(P=np.vstack(pts), policy=policy, T=T, goal=goal, expect=expect, start=np.array(start, float), escape=escape)

_wedge = [post((0.11, 0.15), 0.006), line((-0.45, HW + 0.015), (0.45, HW + 0.015)), line((0.45, HW + 0.015), (0.45, 0.6))]
scene("wedge_spin_in", _wedge, const(0.0, 1.57), 2.0, lambda p, L: L["deepen"] <= 0.002,
      "in contact (leg inside fp, edge 1.5 cm left): full-speed CCW spin sweeps the front-left corner into the edge -> must be rejected")
scene("wedge_reverse_out", _wedge, const(-0.12, 0.0), 2.0, lambda p, L: p[0] <= -0.2 and L["deepen"] <= 0.002,
      "same wedge: straight reverse at planner speed increases clearance -> must pass untouched and exit (governor in escape mode)", escape=True)
scene("wall_follow_4.5cm", [line((-0.5, HW + 0.045), (4.0, HW + 0.045))], const(0.20, 0.0), 10.0, lambda p, L: p[0] >= 1.8,
      "wall 4.5 cm to the left, 0.20 m/s straight -> untouched, 2 m in 10 s")
scene("wall_follow_2.5cm", [line((-0.5, HW + 0.025), (4.0, HW + 0.025))], const(0.15, 0.0), 10.0, lambda p, L: p[0] >= 1.3,
      "hugging inside the margin (2.5 cm), 0.15 m/s parallel -> no clearance lost, should pass")
scene("wall_drift_2deg", [line((-0.5, HW + 0.045), (4.0, HW + 0.045))], const(0.20, 0.0), 10.0, lambda p, L: L["sd_min"] >= S.MARGIN - 0.005,
      "4.5 cm wall, heading 2 deg into it, raw 0.20 m/s (no planner) -> core must stop before the margin", start=(0, 0, math.radians(2)))
scene("squeeze_gap+4cm", [post((0.8, HW + 0.04 + 0.015)), post((0.8, -HW - 0.04 - 0.015))], const(0.10, 0.0), 14.0, lambda p, L: p[0] >= 1.2,
      "gap = width + 2x4 cm (> margin each side), 0.10 m/s -> pass through")
scene("squeeze_gap+2cm", [post((0.8, HW + 0.02 + 0.015)), post((0.8, -HW - 0.02 - 0.015))], const(0.10, 0.0), 12.0, lambda p, L: L["sd_min"] >= 0.0,
      "gap = width + 2x2 cm (< margin each side): blocked by design (margin), must not touch")
for v in (0.10, 0.30, 0.45):
    scene("headon_%.2f" % v, [line((1.2, -0.6), (1.2, 0.6))], const(v, 0.0), 14.0, lambda p, L: L["sd_min"] >= S.MARGIN - 0.005,
          "raw %.2f m/s at a wall (no planner) -> rest gap >= margin" % v)
scene("spin_leg_r0.30", [post((0.30, 0.0))], const(0.0, 1.57), 2.0, lambda p, L: L["sd_min"] >= S.MARGIN - 0.005,
      "leg 30 cm ahead of the axle (< 34.5 cm rear-corner radius), spin 1.57 -> stop before margin")
scene("spin_leg_r0.42", [post((0.42, 0.0))], const(0.0, 1.57), 4.0, lambda p, L: abs(p[2]) >= math.pi and L["interv"] == 0,
      "leg 42 cm ahead (rear corners clear by 6 cm), spin 1.57 -> untouched, > 180 deg")
scene("band_creep_4.5cm", [line((XF + 0.045, -0.4), (XF + 0.045, 0.4))], const(0.03, 0.0), 3.0, lambda p, L: L["sd_min"] >= S.MARGIN - 0.005 and p[0] >= 0.005,
      "box face 4.5 cm ahead (inside the 5 cm band), 0.03 m/s docking creep -> may close to the margin, not further")
_PC = LIM.PLAN_CLEARANCE; _PA = LIM.PERCEPTION_ALLOWANCE
_core_clean = lambda L: L["interv"] == 0 and L["sd_min"] >= S.MARGIN
scene("plan_wall_follow", [line((-0.5, HW + _PC + _PA + 0.005), (4.0, HW + _PC + _PA + 0.005))], const(0.25, 0.0), 10.0,
      lambda p, L: p[0] >= 2.0 and _core_clean(L),
      "PLANNER keeps margin+allowance (+under-read, +5 mm) beside a wall, 0.25 m/s: core never intervenes, 2 m")
scene("plan_wall_4.5cm", [line((-0.5, HW + 0.045), (4.0, HW + 0.045))], const(0.25, 0.0), 10.0, lambda p, L: _core_clean(L),
      "wall 4.5 cm (< margin + allowance): the planner side may slow or stop; the core must not have to intervene")
scene("plan_wall_drift", [line((-0.5, HW + _PC + _PA + 0.005), (4.0, HW + _PC + _PA + 0.005))], const(0.20, 0.0), 10.0, lambda p, L: _core_clean(L),
      "starts where the planner keeps it (margin + allowance + under-read + 5 mm), heading 2 deg into the wall, raw 0.20 m/s: "
      "the planner side must stop it at its zone, the core must not have to act", start=(0, 0, math.radians(2)))
scene("plan_gap_4cm", [post((0.8, HW + 0.04 + 0.015)), post((0.8, -HW - 0.04 - 0.015))], const(0.10, 0.0), 14.0, lambda p, L: _core_clean(L),
      "gap 4 cm/side (< margin + allowance): the planner side declines (stops short); the core must not have to act")
scene("plan_gap_pass", [post((0.8, HW + _PC + _PA + 0.005 + 0.015)), post((0.8, -HW - _PC - _PA - 0.005 - 0.015))], const(0.10, 0.0), 14.0,
      lambda p, L: p[0] >= 1.2 and _core_clean(L),
      "narrowest gap that should pass: margin + allowance + under-read allowance + 5 mm per side -> through, core untouched")
scene("arc_fast_wall", [line((1.0, -1.0), (1.0, 1.5))], const(0.45, 0.6), 4.0, lambda p, L: L["sd_min"] >= S.MARGIN - 0.005,
      "0.45 m/s arc (w 0.6) toward a wall -> stop before margin (per-wheel ramp vs modelled arc stop)")


def to_robot(P, pose):
    return S.to_body(P, pose[0], pose[1], pose[2])


AGE_ROUTE = (0.052, 0.012, 0.033, 0.116)     # measured frame age on the route (p50 ~52 ms, p99 75-87, max 95-116): mean, sd, min, max


def age_sampler(spec, rng):
    """None -> ideal; float -> fixed age (s); 'route' -> clipped normal fitted to the measured route frame ages."""
    if spec in (None, 0, 0.0, "0"): return lambda: 0.0
    if spec == "route": m, sdv, lo, hi = AGE_ROUTE; return lambda: float(np.clip(rng.normal(m, sdv), lo, hi))
    a = float(spec); return lambda: a


# perception under-read model (measured live, emu/tools/scbench/noiseprobe.py 2026-10-20): every point is seen closer to the
# footprint along the normal of its nearest edge; per tick, independently for the core and the brain/governor:
# side edges u ~ N(0.78, 0.35) cm clipped [0, 1.87] (moving, n 1083), front/rear edges u ~ N(0.10, 0.03) cm (static)
PERC = dict(none=None, measured=dict(side=(0.0078, 0.0035, 0.0187), front=(0.0010, 0.0003, 0.0016)))
PERC.update(pt_sim=dict(pt="sim"), pt_real=dict(pt="real", subpix=0.2), pt_real_sun=dict(pt="real", subpix=0.4),
            pt_static=dict(pt="sim", sig=0.0), pt_noise=dict(pt="sim", static=0.0), pt_noise01=dict(pt="sim", static=0.0, sig=0.0001), pt_static_n01=dict(pt="sim", sig=0.0001),
            grid_sim=dict(q=0.0008), grid_real=dict(q=0.0021), grid_sun=dict(q=0.0042), grid_still=dict(q=0.0))
# per-POINT models (pointnoise.py, live sim 2026-10-20, still + drifting beside a wall, core viz snapshots):
#   side (RS1 mast ego grid): each world point under-reads by 0 or 1.01 cm (1 cm grid quantisation, ~half the points),
#   STATIC (same every snapshot); per-point temporal sd median 0.01-0.02 cm, p90 0.06-0.08 cm.  Front (RS2 live): ~0.
#   pt_sim : that static pattern + per tick N(0, 0.8 mm) per point (the p90 temporal sd, rounded up).
#   pt_real: same static pattern + per tick the emulator's own D4xx stereo model at the TOP of its domain-rand range
#            (emu/depthcam.py: RMS = Z^2 subpix / (f B), subpix 0.2 instead of 0.08), projected on the wall normal:
#            RS1 (mast z 0.862 m, looking down): sigma_lat = Z r subpix/(f B), r = horizontal offset from the mast axis;
#            RS2 (front): sigma = Z^2 subpix/(f B), Z = distance from RS2.  Applied fresh every tick to every point
#            (an upper bound: a real ego grid would average it).  ASSUMED model, not measured on the real bot.
_F_PX, _B = 240.0 / math.tan(math.radians(29.0)), 0.050       # D435 depth 848x480, VFOV 58 deg; baseline 50 mm
def _grid_q(Pw, sig, off, rng):
    """ego-grid model (pointnoise.py: the core's side points sit on 1 cm cell corners -> under-read 0 / 1.01 cm, static
    for a still or parallel-drifting robot, temporal sd ~0.01-0.08 cm): each world point gets the depth noise N(0, sig)
    (fresh per tick), then is reported at its 1 cm world-grid cell corner (floor, grid offset 'off')."""
    Q = Pw + sig * rng.standard_normal(Pw.shape)
    return (np.floor((Q - off) / 0.01) * 0.01 + off).astype(np.float32)
_STATIC = {}
def _static(n):
    if n not in _STATIC: _STATIC[n] = np.where(np.random.default_rng(12345 + n).random(n) < 0.5, 0.0101, 0.0)
    return _STATIC[n]
def perceive(P, model, rng):
    if model is None or not len(P): return P
    if "pt" in model:
        D = P @ S.N_.T - S.C_; e = D.argmax(1); n = S.N_[e]; side = np.abs(n[:, 1]) > 0.7
        u = np.where(side, _static(len(P)), 0.0) * model.get("static", 1.0)
        if model["pt"] == "sim": sig = np.full(len(P), model.get("sig", 0.0008))
        else:
            sub = model.get("subpix", 0.2); r = np.hypot(P[:, 0] + 0.007, P[:, 1]); Z1 = 0.862
            z2 = np.hypot(P[:, 0] + 0.130, P[:, 1])
            sig = np.where(side, Z1 * r * sub / (_F_PX * _B), z2 * z2 * sub / (_F_PX * _B))
        u = u + sig * rng.standard_normal(len(P))
        return (P - u[:, None] * n).astype(np.float32)
    D = P @ S.N_.T - S.C_; e = D.argmax(1); n = S.N_[e]; side = np.abs(n[:, 1]) > 0.7
    us = float(np.clip(rng.normal(*model["side"][:2]), 0, model["side"][2])); uf = float(np.clip(rng.normal(*model["front"][:2]), 0, model["front"][2]))
    # per-point spread within a tick (ASSUMED, not measured separately): each point under-reads u * U(0.6, 1), so the
    # closest point still matches the measured per-tick min under-read while points along a wall flicker by up to 40 %
    k = rng.uniform(0.6, 1.0, len(P)) if model.get("spread", True) else 1.0
    return (P - (np.where(side, us, uf) * k)[:, None] * n).astype(np.float32)


def run(name, rule, use_gov, age=None, seed=0, scenes=None, odo=0.0, perc=None):
    sc = (scenes or SCENES)[name]; Pw = sc["P"]; pose = sc["start"].copy(); bands = StubBands()
    S.SC_DECEL = GOV.SC_DECEL = (rule == "wired"); rule_ = rule; rule = "core" if rule in ("core", "wired") else rule
    rng = np.random.default_rng(seed); age_f = age_sampler(age, rng); hist = collections.deque(maxlen=400)   # (t, pose) 1 ms
    vl = vr = 0.0; q = collections.deque()                  # (t_apply, vl_tgt, vr_tgt, decel)
    t = 0.0; t_next = 0.0; v_m = w_m = 0.0; tgt = (0.0, 0.0, A_UP)
    log = dict(codes=collections.Counter(), interv=0, ticks=0, sd_min=np.inf, deepen=0.0, peak_dec=0.0, first_code=None)
    lab = PD.clusters(Pw); nc = lab.max() + 1
    q_off = rng.uniform(0, 0.01, 2)                         # wall position within the 1 cm ego grid, per seed
    def cl_min(s): return np.array([s[lab == c].min() for c in range(nc)])
    c_start = cl_min(S.sd(to_robot(Pw, pose))); prev_wheels = (0.0, 0.0)
    # wired+gov runs the real command path: governor -> CmdLimiter (hw.drive) -> core, whose LimitCheck counts planner bugs
    climit = LIM.CmdLimiter(S.TRACK_M) if (use_gov and rule_ == "wired") else None; chk = LIM.LimitCheck(S.TRACK_M); log["plan_bugs"] = 0; log["gov_interv"] = 0
    while t < sc["T"] - 1e-9:
        if t >= t_next - 1e-9:
            t_next += S.T_TICK; log["ticks"] += 1
            ag = age_f()
            if ag > 0 and hist:
                # what the core sees: points captured 'ag' ago (pose then), moved into NOW with the measured twist (as the core)
                k = max(0, len(hist) - 1 - int(round(ag / DT_PLANT))); P = S.motion_comp(to_robot(Pw, hist[k][1]), v_m * (1 + odo), w_m * (1 + odo), ag)
            else: P = to_robot(Pw, pose)
            v_c, w_c = sc["policy"](t, pose)
            if perc is not None and "q" in perc:      # ego-grid model on WORLD points, independent draws for governor and core
                Pg, P = (to_robot(_grid_q(Pw, perc["q"], q_off, rng), pose) for _ in range(2))
            else: Pg = perceive(P, perc, rng); P = perceive(P, perc, rng)
            if use_gov:
                v0_, w0_ = v_c, w_c
                if rule == "core": v_c, w_c = GOV.governor(Pg, v_c, w_c, v_m, w_m, bands.rects(), escape=sc.get("escape", False))[:2]
                else: v_c, w_c = PD.governor(P, v_c, w_c)[:2]
                if abs(v_c - v0_) > 1e-6 or abs(w_c - w0_) > 1e-6: log["gov_interv"] += 1
            if climit is not None: v_c, w_c = climit(v_c, w_c, t)
            if use_gov and rule_ == "wired": chk(v_c, w_c, t); log["plan_bugs"] = chk.bugs
            if ag > S.AGE_MAX_S:
                vo, wo, code = 0.0, 0.0, "inop_age"; dec = S.BRAKE[S.RAMP_MAX][1] if rule == "core" else PD.A_LONG
            elif rule == "core":
                vo, wo, code, ramp = S.decide(P, bands, v_c, w_c, v_m, w_m)[:4]
                dec = S.A_LONG if abs(ramp - S.RAMP_DESIGN) < 1e-9 else float(np.interp(ramp, [S.RAMP_NORMAL, S.RAMP_MAX], [S.BRAKE[S.RAMP_NORMAL][1], S.BRAKE[S.RAMP_MAX][1]]))
            else:
                vo, wo, code, dec = PD.decide(P, None, v_c, w_c, v_m, w_m)[:4]
            log["codes"][code] += 1
            if log["first_code"] is None: log["first_code"] = code
            if (abs(vo - v_c) > 1e-6 or abs(wo - w_c) > 1e-6): log["interv"] += 1
            q.append((t + T_CMD + LAG, vo - wo * S.TRACK_M / 2, vo + wo * S.TRACK_M / 2, dec))
        while q and q[0][0] <= t + 1e-9: tgt = q.popleft()[1:]
        # per-wheel ramp (ODrive): accelerate at the normal rate, decelerate at the commanded brake decel
        for i, (cur, goal) in enumerate(((vl, tgt[0]), (vr, tgt[1]))):
            lim = (tgt[2] if abs(goal) < abs(cur) or goal * cur < 0 else A_UP) * DT_PLANT
            nv = cur + max(-lim, min(lim, goal - cur))
            if i == 0: vl = nv
            else: vr = nv
        log["peak_dec"] = max(log["peak_dec"], max(abs(vl) < abs(prev_wheels[0]) and (abs(prev_wheels[0]) - abs(vl)) / DT_PLANT or 0,
                                                   abs(vr) < abs(prev_wheels[1]) and (abs(prev_wheels[1]) - abs(vr)) / DT_PLANT or 0))
        prev_wheels = (vl, vr)
        v_m, w_m = 0.5 * (vl + vr), (vr - vl) / S.TRACK_M
        pose[0] += v_m * math.cos(pose[2] + w_m * DT_PLANT / 2) * DT_PLANT; pose[1] += v_m * math.sin(pose[2] + w_m * DT_PLANT / 2) * DT_PLANT
        pose[2] += w_m * DT_PLANT; t += DT_PLANT; hist.append((t, pose.copy()))
        if int(round(t / DT_PLANT)) % 5 == 0:
            s = S.sd(to_robot(Pw, pose)); log["sd_min"] = min(log["sd_min"], float(s.min()))
            log["deepen"] = max(log["deepen"], float((np.minimum(c_start, S.MARGIN) - cl_min(s)).max()))  # worst per-obstacle clearance lost below min(start, margin)
    log["sd_end"] = float(S.sd(to_robot(Pw, pose)).min()); log["pose"] = pose.round(3).tolist()
    log["ok"] = bool(sc["goal"](pose, log)); log["contact"] = log["sd_min"] < 0.0
    return log


def open_loop(name, rule):
    sc = SCENES[name]; P = to_robot(sc["P"], sc["start"]); v, w = sc["policy"](0.0, sc["start"])
    S.SC_DECEL = GOV.SC_DECEL = (rule == "wired")
    if rule in ("core", "wired"): return S.decide(P, StubBands(), v, w, 0.0, 0.0)[2]
    return PD.decide(P, None, v, w, 0.0, 0.0)[2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scenes", nargs="*"); ap.add_argument("--json")
    ap.add_argument("--age", default=None, help="frame-age model: seconds, or 'route' (measured distribution)"); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--perc", default="none", choices=list(PERC), help="perception under-read model (measured: live noiseprobe)")
    ap.add_argument("--odo", type=float, default=0.0, help="relative error of the twist used for motion compensation (-0.1: under-reads 10%%)")
    a = ap.parse_args()
    names = a.scenes or list(SCENES); out = {}
    t0 = time.time()
    hdr = "%-18s | %-34s | %-34s | %-34s" % ("scene", "OLD core rule (+gov)", "WIRED safety_core SC_DECEL (+gov)", "PROTOTYPE (+gov)")
    print(hdr); print("-" * len(hdr))
    for n in names:
        row = {}
        for rule in RULES:
            for g in (False, True):
                Ls = [run(n, rule, g, a.age, sd_, odo=a.odo, perc=PERC[a.perc]) for sd_ in range(a.seeds)]; L = min(Ls, key=lambda x: (x["ok"], x["sd_min"]))   # worst seed
                L["n_ok"] = sum(x["ok"] for x in Ls); L["open"] = open_loop(n, rule); row["%s%s" % (rule, "+gov" if g else "")] = L
        def cell(L, Lg):
            return "%s%s %-5s int%3d sd%+5.1f dp%4.1f | gov %s" % ("PASS" if L["ok"] else "FAIL", "" if a.seeds == 1 else "%d/%d" % (L["n_ok"], a.seeds), L["open"][:5], L["interv"], 100 * L["sd_min"], 100 * L["deepen"], "PASS" if Lg["ok"] else "FAIL")
        print("%-18s | %-34s | %-34s | %-34s | bugs %d | wired+gov: core int %d gov int %d sd %+.1f x %.2f" % (n, cell(row["core"], row["core+gov"]), cell(row["wired"], row["wired+gov"]), cell(row["proto"], row["proto+gov"]), row["wired+gov"]["plan_bugs"], row["wired+gov"]["interv"], row["wired+gov"]["gov_interv"], 100 * row["wired+gov"]["sd_min"], row["wired+gov"]["pose"][0]))
        out[n] = {k: {kk: (dict(vv) if isinstance(vv, collections.Counter) else vv) for kk, vv in L.items()} for k, L in row.items()}
        out[n]["expect"] = SCENES[n]["expect"]
    print("age model: %s, odometry error %+.0f%%, perception %s, seeds %d (worst seed shown); PLAN_CLEARANCE %.3f (margin %.3f + allowance %.3f)" % (a.age, 100 * a.odo, a.perc, a.seeds, LIM.PLAN_CLEARANCE, S.MARGIN, LIM.PERCEPTION_ALLOWANCE))
    print("(%.1f s)  cell: goal PASS/FAIL, first-tick code, interventions (ticks), min sd cm, max per-obstacle clearance lost below min(start, margin) cm | with governor" % (time.time() - t0))
    if a.json: json.dump(out, open(a.json, "w"), indent=1, default=float)
