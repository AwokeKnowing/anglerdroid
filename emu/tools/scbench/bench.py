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
def scene(name, pts, policy, T, goal, expect, start=(0.0, 0.0, 0.0)):
    SCENES[name] = dict(P=np.vstack(pts), policy=policy, T=T, goal=goal, expect=expect, start=np.array(start, float))

_wedge = [post((0.11, 0.15), 0.006), line((-0.45, HW + 0.015), (0.45, HW + 0.015)), line((0.45, HW + 0.015), (0.45, 0.6))]
scene("wedge_spin_in", _wedge, const(0.0, 1.57), 2.0, lambda p, L: L["deepen"] <= 0.002,
      "in contact (leg inside fp, edge 1.5 cm left): full-speed CCW spin sweeps the front-left corner into the edge -> must be rejected")
scene("wedge_reverse_out", _wedge, const(-0.12, 0.0), 2.0, lambda p, L: p[0] <= -0.2 and L["deepen"] <= 0.002,
      "same wedge: straight reverse at planner speed increases clearance -> must pass untouched and exit")
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
scene("arc_fast_wall", [line((1.0, -1.0), (1.0, 1.5))], const(0.45, 0.6), 4.0, lambda p, L: L["sd_min"] >= S.MARGIN - 0.005,
      "0.45 m/s arc (w 0.6) toward a wall -> stop before margin (per-wheel ramp vs modelled arc stop)")


def to_robot(P, pose):
    return S.to_body(P, pose[0], pose[1], pose[2])


def run(name, rule, use_gov, age=0.0):
    sc = SCENES[name]; Pw = sc["P"]; pose = sc["start"].copy(); bands = StubBands()
    vl = vr = 0.0; q = collections.deque()                  # (t_apply, vl_tgt, vr_tgt, decel)
    t = 0.0; t_next = 0.0; v_m = w_m = 0.0; tgt = (0.0, 0.0, A_UP)
    log = dict(codes=collections.Counter(), interv=0, ticks=0, sd_min=np.inf, deepen=0.0, peak_dec=0.0, first_code=None)
    lab = PD.clusters(Pw); nc = lab.max() + 1
    def cl_min(s): return np.array([s[lab == c].min() for c in range(nc)])
    c_start = cl_min(S.sd(to_robot(Pw, pose))); prev_wheels = (0.0, 0.0)
    while t < sc["T"] - 1e-9:
        if t >= t_next - 1e-9:
            t_next += S.T_TICK; log["ticks"] += 1
            P = to_robot(Pw, pose)
            v_c, w_c = sc["policy"](t, pose)
            if use_gov:
                if rule == "core": v_c, w_c = GOV.governor(P, v_c, w_c, v_m, w_m, bands.rects())[:2]
                else: v_c, w_c = PD.governor(P, v_c, w_c)[:2]
            if rule == "core":
                vo, wo, code, ramp = S.decide(P, bands, v_c, w_c, v_m, w_m)[:4]
                dec = float(np.interp(ramp, [S.RAMP_NORMAL, S.RAMP_MAX], [S.BRAKE[S.RAMP_NORMAL][1], S.BRAKE[S.RAMP_MAX][1]]))
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
        pose[2] += w_m * DT_PLANT; t += DT_PLANT
        if int(round(t / DT_PLANT)) % 5 == 0:
            s = S.sd(to_robot(Pw, pose)); log["sd_min"] = min(log["sd_min"], float(s.min()))
            log["deepen"] = max(log["deepen"], float((np.minimum(c_start, S.MARGIN) - cl_min(s)).max()))  # worst per-obstacle clearance lost below min(start, margin)
    log["sd_end"] = float(S.sd(to_robot(Pw, pose)).min()); log["pose"] = pose.round(3).tolist()
    log["ok"] = bool(sc["goal"](pose, log)); log["contact"] = log["sd_min"] < 0.0
    return log


def open_loop(name, rule):
    sc = SCENES[name]; P = to_robot(sc["P"], sc["start"]); v, w = sc["policy"](0.0, sc["start"])
    if rule == "core": return S.decide(P, StubBands(), v, w, 0.0, 0.0)[2]
    return PD.decide(P, None, v, w, 0.0, 0.0)[2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scenes", nargs="*"); ap.add_argument("--json"); a = ap.parse_args()
    names = a.scenes or list(SCENES); out = {}
    t0 = time.time()
    hdr = "%-18s | %-34s | %-34s" % ("scene", "CURRENT core (+gov)", "PROTOTYPE decel rule (+gov)")
    print(hdr); print("-" * len(hdr))
    for n in names:
        row = {}
        for rule in ("core", "proto"):
            for g in (False, True):
                L = run(n, rule, g); L["open"] = open_loop(n, rule); row["%s%s" % (rule, "+gov" if g else "")] = L
        def cell(L, Lg):
            return "%s %-5s int%3d sd%+5.1f dp%4.1f | gov %s" % ("PASS" if L["ok"] else "FAIL", L["open"][:5], L["interv"], 100 * L["sd_min"], 100 * L["deepen"], "PASS" if Lg["ok"] else "FAIL")
        print("%-18s | %-34s | %-34s" % (n, cell(row["core"], row["core+gov"]), cell(row["proto"], row["proto+gov"])))
        out[n] = {k: {kk: (dict(vv) if isinstance(vv, collections.Counter) else vv) for kk, vv in L.items()} for k, L in row.items()}
        out[n]["expect"] = SCENES[n]["expect"]
    print("(%.1f s)  cell: goal PASS/FAIL, first-tick code, interventions (ticks), min sd cm, max per-obstacle clearance lost below min(start, margin) cm | with governor" % (time.time() - t0))
    if a.json: json.dump(out, open(a.json, "w"), indent=1, default=float)
