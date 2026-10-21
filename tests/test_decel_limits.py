"""Designed-decel rule + ONE accel config (nav/limits.py), python tests/test_decel_limits.py [n]
1. property: with SC_DECEL, a command the governor passes is never zeroed by the core's decide() on the same points
   (governor: planner limits = safety / PLAN_MULT, PLAN_MULT x lead; core: safety limits)
2. the command output stage (CmdLimiter) never emits a command outside the PLANNER limits, for random command streams,
   and the core's LimitCheck (SAFETY limits) therefore never counts a planner bug on its output
3. LimitCheck counts an out-of-range raw stream (step 0 -> 0.5 m/s; lateral 0.4 m/s x 3 rad/s) as planner bugs
4. planner clearance (PLAN_CLEARANCE = margin + measured perception allowance): config relations, and the governor can
   never pass a command that takes the robot from outside PLAN_CLEARANCE to inside it (exact sd along its own
   trajectory, densely resampled; 2 mm tolerance for the 2 mm sweep cell merge, which errs safe)"""
import os, sys
os.environ["KEVIN_SC_DECEL"] = "1"; os.environ.setdefault("KEVIN_PLAN_CLEARANCE", "1")
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path[:0] = [R, os.path.join(R, "src")]
import nav  # noqa
from nav import safety_core as S, governor as G, limits as LIM
from twin_geometry import TRACK_M


class _B:
    def front(self, P): return False
    def rear(self, P): return False


def prop(n, seed=0):
    rng = np.random.default_rng(seed); st = dict(n=0, gov_moves=0, disagree=0, inside=0)
    for _ in range(n):
        P = []
        while len(P) < rng.integers(1, 30):
            q = rng.uniform(-0.7, 0.7, 2)
            if S.sd(q[None])[0] > -0.03: P.append(q)
        P = np.array(P); st["inside"] += int(S.sd(P).min() < S.MARGIN)
        v = rng.choice([0.0, rng.uniform(-0.25, 0.6)]); w = rng.uniform(-2.0, 2.0); vm = v * rng.uniform(0, 1); wm = w * rng.uniform(0, 1)
        gv, gw = G.governor(P, v, w, vm, wm)[:2]; st["n"] += 1
        if abs(gv) < 1e-3 and abs(gw) < 1e-3: continue
        st["gov_moves"] += 1
        o = S.decide(P, _B(), gv, gw, vm, wm)
        if not (abs(o[0] - gv) < 1e-9 and abs(o[1] - gw) < 1e-9): st["disagree"] += 1
    return st


def limiter(n, seed=1):
    rng = np.random.default_rng(seed); L = LIM.CmdLimiter(TRACK_M); Cp = LIM.LimitCheck(TRACK_M, LIM.PLAN_A_LONG, LIM.PLAN_A_LAT, tol=1e-6)
    Cs = LIM.LimitCheck(TRACK_M); t = 100.0
    for i in range(n):
        t += rng.choice([1 / 30, 1 / 14, 0.005, 0.2, 0.5])
        v, w = (0.0, 0.0) if rng.random() < 0.1 else (rng.uniform(-0.3, 0.8), rng.uniform(-3, 3))
        vo, wo = L(v, w, t); Cp(vo, wo, t); Cs(vo, wo, t)
    return dict(n=n, clamped=L.n_clamped, plan_violations=Cp.bugs, safety_bugs=Cs.bugs)


def rawbugs():
    C = LIM.LimitCheck(TRACK_M); t = 10.0
    for v, w in [(0, 0), (0.5, 0), (0.5, 0), (0.4, 3.0)]:
        t += 1 / 14; C(v, w, t)
    return C.bugs


def plan_clearance(n, seed=2):
    import twin_geometry as TG
    cfg = S.MARGIN <= LIM.PLAN_CLEARANCE <= LIM.MPPI_DEAD and abs(LIM.PLAN_CLEARANCE - S.MARGIN - LIM.PERCEPTION_ALLOWANCE) < 1e-9 \
        and min(LIM.GLOBAL_INFLATE) >= TG.HALF_W + LIM.PLAN_CLEARANCE - 1e-9
    rng = np.random.default_rng(seed); st = dict(cfg=bool(cfg), n=0, outside=0, passed=0, entered=0, worst_mm=None)
    for _ in range(n):
        y0 = rng.uniform(0.03, 0.10) * rng.choice([-1, 1]); hw = float(np.abs(S.FOOTPRINT_SAFE[:, 1]).max())
        xs = np.arange(-0.6, 1.2, 0.01); P = [np.c_[xs, np.full_like(xs, np.sign(y0) * (hw + abs(y0)))]]       # a wall beside
        if rng.random() < 0.5: P.append(np.c_[np.full(40, rng.uniform(0.2, 0.9)), rng.uniform(-0.3, 0.3, 40)])  # + something ahead
        P = np.vstack(P).astype(np.float32) + rng.normal(0, 0.002, (len(np.vstack(P)), 2)).astype(np.float32)
        sd0 = float(S.sd(P).min()); st["n"] += 1
        if sd0 < LIM.PLAN_CLEARANCE: continue
        st["outside"] += 1
        v = rng.uniform(-0.2, 0.6); w = rng.uniform(-1.5, 1.5) * rng.choice([0, 1]); gv, gw = G.governor(P, v, w, v * 0.8, w * 0.8)[:2]
        if abs(gv) < 1e-3 and abs(gw) < 1e-3: continue
        st["passed"] += 1
        poses = S.brake_traj(gv, gw, LIM.PLAN_MULT * S.T_R, LIM.PLAN_A_LONG, LIM.PLAN_A_LAT)[0]
        fine = np.vstack([poses[0]] + [np.linspace(poses[i], poses[i + 1], 6)[1:] for i in range(len(poses) - 1)])
        mn = min(float(S.sd(S.to_body(P, *q)).min()) for q in fine)
        lost = LIM.PLAN_CLEARANCE - mn
        st["worst_mm"] = round(1e3 * lost, 2) if st["worst_mm"] is None else max(st["worst_mm"], round(1e3 * lost, 2))
        if lost > 0.002: st["entered"] += 1
    return st


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    a = prop(n); b = limiter(20000); c = rawbugs(); d = plan_clearance(n)
    print("property", a); print("limiter", b); print("raw stream planner bugs", c, "(expect 2)")
    print("planner clearance", d, "PLAN_CLEARANCE %.3f = margin %.3f + allowance %.3f; MPPI_DEAD %.3f; GLOBAL_INFLATE %s" % (LIM.PLAN_CLEARANCE, S.MARGIN, LIM.PERCEPTION_ALLOWANCE, LIM.MPPI_DEAD, LIM.GLOBAL_INFLATE))
    print("limits: safety %.2f / %.2f m/s^2, planner %.2f / %.2f (mult %.2f)" % (LIM.SAFETY_A_LONG, LIM.SAFETY_A_LAT, LIM.PLAN_A_LONG, LIM.PLAN_A_LAT, LIM.PLAN_MULT))
    ok = a["disagree"] == 0 and b["plan_violations"] == 0 and b["safety_bugs"] == 0 and c == 2 and d["cfg"] and d["entered"] == 0 and d["passed"] > 0
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
