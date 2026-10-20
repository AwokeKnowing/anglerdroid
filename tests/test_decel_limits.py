"""Designed-decel rule + ONE accel config (nav/limits.py), python tests/test_decel_limits.py [n]
1. property: with SC_DECEL, a command the governor passes is never zeroed by the core's decide() on the same points
   (governor: planner limits = safety / PLAN_MULT, PLAN_MULT x lead; core: safety limits)
2. the command output stage (CmdLimiter) never emits a command outside the PLANNER limits, for random command streams,
   and the core's LimitCheck (SAFETY limits) therefore never counts a planner bug on its output
3. LimitCheck counts an out-of-range raw stream (step 0 -> 0.5 m/s; lateral 0.4 m/s x 3 rad/s) as planner bugs"""
import os, sys
os.environ["KEVIN_SC_DECEL"] = "1"
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


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    a = prop(n); b = limiter(20000); c = rawbugs()
    print("property", a); print("limiter", b); print("raw stream planner bugs", c, "(expect 2)")
    print("limits: safety %.2f / %.2f m/s^2, planner %.2f / %.2f (mult %.2f)" % (LIM.SAFETY_A_LONG, LIM.SAFETY_A_LAT, LIM.PLAN_A_LONG, LIM.PLAN_A_LAT, LIM.PLAN_MULT))
    ok = a["disagree"] == 0 and b["plan_violations"] == 0 and b["safety_bugs"] == 0 and c == 2
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
