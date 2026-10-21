"""MPPI may only reverse into space the RS1 currently observes free (World.edt_rev, mppi kernels).  GPU test.
python tests/test_reverse_observed.py

A) open room, wall 0.35 m ahead, goal 1.5 m behind: MPPI turns in place and drives forward (never reverses when the rear
   is unobserved; at most a few ticks when it is observed free).
B) dead-end corridor 0.50 m wide (too narrow to spin), goal 1.2 m behind:
   rear observed free -> it reverses out;  rear NOT observed -> it never reverses (and nav reports no progress)."""
import math, os, sys
import numpy as np, torch
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nav", "kplan"))]
os.environ.setdefault("KPLAN_GLOBAL", "fast")
import planner as KP, fastpath as FP
from core import integrate_v

N = 600
def world(corridor):
    occ = np.zeros((N, N), bool); occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[:, 400:410] = True                                             # wall / dead end at x = 4.0 m
    if corridor: occ[:275, 150:400] = True; occ[325:, 150:400] = True        # 0.50 m corridor y 2.75..3.25, x 1.5..4.0
    return occ

def run(occ, rear_observed, goal, steps=360):
    rev = np.zeros_like(occ) if rear_observed else np.ones_like(occ)
    yy, xx = np.mgrid[0:N, 0:N]; rev[np.hypot(xx - 365, yy - 300) < 34] = False   # the body + a few cm around it is always observed (ego SELF + CLEAR)
    W = KP.World(occ, rev_block=rev); NG = FP.FastNewGoal(W, 1)
    x, y, th, v, w = 3.65, 3.0, 0.0, 0.0, 0.0
    p = FP.plan_path(NG.m, np.array([x, y]), np.asarray(goal), eps=2.0); assert p is not None
    NG.M.set_paths(np.array([[*goal, 0.0]]), [p]); vs = []; dmin = 9.0
    for _ in range(steps):
        u, _i = NG.G.plan(torch.tensor([[x, y, th, v, w]], device="cuda")); a, al = u[0].tolist()
        v = integrate_v(v, a, 1 / 30, -0.2, 0.6); w = float(np.clip(w + al / 30, -2.5, 2.5))
        th += w / 30; x += v * math.cos(th) / 30; y += v * math.sin(th) / 30
        vs.append(v if math.hypot(x - 3.65, y - 3.0) > 0.05 else 0.0)        # reversing inside the observed disc is fine
        dmin = min(dmin, math.hypot(x - goal[0], y - goal[1]))
    return np.array(vs), dmin

for name, corr, goal in (("A open room", False, (2.15, 3.0)), ("B dead-end corridor", True, (2.45, 3.0))):
    vo, do = run(world(corr), True, goal); vu, du = run(world(corr), False, goal)
    print("%-20s rear observed: reverse steps %3d, closest %.2f m | rear unobserved: reverse steps %3d, closest %.2f m" % (name, (vo < -0.02).sum(), do, (vu < -0.02).sum(), du))
    assert (vu < -0.02).sum() == 0, "reversed into unobserved space"
    if not corr: assert do < 0.5 and du < 0.5 and (vo < -0.02).mean() < 0.05, "open room: turn in place + drive forward"
    else: assert (vo < -0.02).sum() > 0 and do < 0.4, "observed-free rear: reversing out is allowed"
print("PASS")
