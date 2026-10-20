"""Footprint escape search (brain side; nav/safety_core.py is not touched).

Problem (L15 kitchen stall, 2026-10-20, emu/tools/stuckdiag): Kevin wedged nose-in between a counter corner and a chair
(front-right point 6.4 cm from the footprint, rear 0.34 m).  The governor allows only rotation; axle clearance 0.287 m
< R_ROTATE 0.309 so _face_path will not spin; 99.5 % of MPPI's 384 rollouts die on the footprint/reverse rule, so MPPI
returns u = 0 and the trail retreat (an MPPI goal) stalls the same way.  An SE(2) lattice search on the same points finds a
5-step escape (right 50 deg, back 8 cm) in 209 expansions.

Here: best-first search over in-place rotations (10 deg) and short straight moves (4 cm fwd / rev) until the axle has room
to spin (R_ROTATE).  A primitive is feasible only if
  * the governor (nav/governor.py: the safety core's own rules x2 with the core's bands) passes it AT its execution speed
    (CREEP_V / CREEP_W) from the primitive's start pose, with full scale -- so execution is not vetoed by the brain's
    speed limiter (same rule) or the core, and
  * the true footprint keeps >= CLEAR m from every point over the whole primitive (sampled every 1 cm / 2.5 deg).
Points: the governor's own point set (RS1 ego + RS2 + map memory, robot frame) at the stuck tick, put in the nav frame.
Cost: path length with reversing 2x and rotation 0.2 m/rad, + a switch penalty (fewer direction changes).
Returns a list of segments [("rot", dtheta) | ("lin", metres)] or None (with a reason)."""
from __future__ import annotations
import heapq, math, time
import numpy as np
import twin_geometry as TG
from nav.governor import governor
from nav.safety_core import sd

CREEP_V = 0.06; CREEP_W = 0.5; STEP_M = 0.04; STEP_A = math.radians(10); CLEAR = 0.02; GOAL_R = TG.R_ROTATE + 0.01
MOVES = (("lin", STEP_M), ("lin", -STEP_M), ("rot", STEP_A), ("rot", -STEP_A))


def to_robot(P, p):
    c, s = math.cos(p[2]), math.sin(p[2]); d = P - np.asarray(p[:2], float)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def _step(p, kind, a):
    return (p[0] + a * math.cos(p[2]), p[1] + a * math.sin(p[2]), p[2]) if kind == "lin" else (p[0], p[1], p[2] + a)


def plan(Pw, pose, bands=None, max_exp=4000, max_s=3.0, goal_r=GOAL_R, reach=0.6):
    """Pw (N,2) obstacle points, nav frame; pose (x, y, th).  -> (segments | None, info dict)."""
    t0 = time.monotonic(); x0, y0 = pose[0], pose[1]
    Pw = np.asarray(Pw, float)[:, :2]
    from nav.governor import R_FP
    Pw = Pw[np.hypot(Pw[:, 0] - x0, Pw[:, 1] - y0) < reach + R_FP + 0.25] if len(Pw) else Pw   # all a sweep within reach can touch
    if not len(Pw): return [], dict(why="no points", exp=0)
    axle = lambda p: float(np.hypot(Pw[:, 0] - p[0], Pw[:, 1] - p[1]).min())
    if axle(pose) >= goal_r: return [], dict(why="already free to spin", exp=0, axle=axle(pose))

    def feasible(p, kind, a):
        for k in (4, 2, 1, 3):                                            # cheap sweep test first (end pose first)
            if sd(to_robot(Pw, _step(p, kind, a * k / 4))).min() < CLEAR: return None
        v, w = (math.copysign(CREEP_V, a), 0.0) if kind == "lin" else (0.0, math.copysign(CREEP_W, a))
        if governor(to_robot(Pw, p), v, w, bands=bands)[4] != "ok": return None
        return _step(p, kind, a)

    key = lambda p: (round(p[0] / 0.02), round(p[1] / 0.02), round(((p[2] + math.pi) % (2 * math.pi)) / STEP_A) % 36)
    Q = [(0.0, 0, tuple(pose), (), None)]; seen = {key(pose)}; n = 0; tie = 0
    while Q and n < max_exp and time.monotonic() - t0 < max_s:
        c, _, p, path, last = heapq.heappop(Q); n += 1
        if axle(p) >= goal_r:
            segs = []
            for kind, a in path:
                if segs and segs[-1][0] == kind and segs[-1][1] * a > 0: segs[-1] = (kind, segs[-1][1] + a)
                else: segs.append((kind, a))
            return segs, dict(why="found", exp=n, cost=round(c, 3), end=[round(float(v), 3) for v in p], ms=round((time.monotonic() - t0) * 1e3), axle0=round(axle(pose), 3))
        if math.hypot(p[0] - x0, p[1] - y0) > reach: continue
        for kind, a in MOVES:
            q = feasible(p, kind, a)
            if q is None: continue
            k = key(q)
            if k in seen: continue
            seen.add(k); tie += 1
            dc = (abs(a) * (2.0 if a < 0 else 1.0)) if kind == "lin" else 0.2 * abs(a)
            sw = 0.03 if last is not None and last != (kind, a > 0) else 0.0
            heapq.heappush(Q, (c + dc + sw, tie, q, path + ((kind, a),), (kind, a > 0)))
    return None, dict(why="no escape within %d expansions / %.1f s" % (n, time.monotonic() - t0), exp=n, axle0=round(axle(pose), 3))
