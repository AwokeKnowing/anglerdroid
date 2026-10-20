"""Planner-side speed governor (brain), aligned with the safety core so the two cannot disagree.

The safety core (nav/safety_core.decide, unchanged) passes a command only if (a) it does not translate toward an
occupied hard band (5 cm in front of / behind the footprint), and (b) its sweep stays clear for the stopping
distance, or -- only when ALREADY inside the margin (sd0 < MARGIN) -- the motion gets no closer anywhere (escape).
Anything else is zeroed.

The previous governor allowed 'gets no closer' at any distance and ignored the bands, so it passed commands the
core then zeroed; the planner, seeing them 'allowed', kept sending them: a deadlock (stuck dump 131010: brain
v=-0.03 w=-0.72 'ok', core output 0, wheels still for 15 s among chair legs).  Here every candidate must pass the
core's own rules with factor x the stopping distance and the core's own band rectangles, on the brain's points
(RS1 ego + RS2 + map memory: a superset of the core's RS1 ego + RS2).  On identical points a command it passes, the
core passes (larger window => smaller 'fr'-cap and 'mn'; same sd0 and escape test; property test: 0/3000 random
disagreements vs 53/3000 for the old rule); the 2x covers measured-speed and one-tick point-age differences.  Candidates, most to least of the planner's intent: the command scaled 1 .. 0.1, then the rotation
alone (the core checks a pure rotation without the translation bands), then stop.  It only ever reduces.
With SC_DECEL (sim default, see safety_core) every candidate is tested with the core's designed-decel rule instead, with
factor x the reaction lead time."""
from __future__ import annotations
import numpy as np
from nav import limits as LIM
from nav.safety_core import sd, free_along, free_rotation, need_dist, need_angle, MARGIN, SC_DECEL, T_R, brake_traj, decel_violates, decel_prep, traj_travel
from twin_geometry import FOOTPRINT_SAFE

R_FP = float(np.hypot(FOOTPRINT_SAFE[:, 0], FOOTPRINT_SAFE[:, 1]).max())   # footprint radius about the axle

SCALES = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1)


def _band(P, rect):
    x0, y0, x1, y1 = rect
    return bool(len(P)) and bool(((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)).any())


def governor(P, v, w, v_meas=0.0, w_meas=0.0, bands=None, factor=2.0):
    """P (N,2) robot frame (brain points U core points); bands: core band rects [[front], [rear]] (x0, y0, x1, y1)
    or None.  Returns (v, w, scale, need2, why)."""
    sd0 = float(sd(P).min()) if len(P) else np.inf
    # exact pruning (cost only): a sweep of arclength s keeps the footprint within s + R_FP of the axle, so points farther
    # than that + MARGIN can neither stop it nor be 'closest'.  Same sampling steps as the core.
    rP = np.hypot(P[:, 0], P[:, 1]) if len(P) else np.zeros(0)
    near = lambda reach: P[rP <= reach + R_FP + MARGIN + 0.02]
    bf = bands is not None and _band(P, bands[0]); br = bands is not None and _band(P, bands[1])

    def ok(vs, ws):
        if abs(vs) >= 1e-3:
            sign = 1 if vs > 0 else -1
            if (sign > 0 and bf) or (sign < 0 and br): return False, 0.0
            ve = max(abs(vs), abs(v_meas) if v_meas * vs > 0 else 0.0); need2 = factor * (need_dist(ve) + MARGIN)
            fr, mn = free_along(near(need2), sign, ws / vs, need2)
        elif abs(ws) >= 1e-3:
            we = max(abs(ws), abs(w_meas) if w_meas * ws > 0 else 0.0); need2 = factor * need_angle(we)
            fr, mn = free_rotation(near(0.0), 1 if ws > 0 else -1, need2)
        else: return True, 0.0
        return fr >= need2 or (sd0 < MARGIN and mn >= sd0 - 1e-3), need2

    if SC_DECEL:
        # mirror of the core's designed-decel rule (safety_core.decide_decel) with the PLANNER accel limits (safety / PLAN_MULT,
        # nav/limits.py) and PLAN_MULT x the reaction lead, so the planner slows before the core would have to act; same
        # margin, no band veto (the core has none in this mode)
        factor = LIM.PLAN_MULT; aL, aT = LIM.PLAN_A_LONG, LIM.PLAN_A_LAT
        sdP = sd(P) if len(P) else np.zeros(0); PREP = {}
        def ok(vs, ws):
            if abs(vs) < 1e-3 and abs(ws) < 1e-3: return True, 0.0
            ve = vs if not (v_meas * vs > 0 and abs(v_meas) > abs(vs)) else v_meas
            we = ws if not (w_meas * ws > 0 and abs(w_meas) > abs(ws)) else w_meas
            poses, t_stop = brake_traj(ve, we, factor * T_R, aL, aT)
            if not len(P): return True, t_stop
            if "p" not in PREP:     # candidates once, for the largest travel of this call (full command, or measured if faster)
                big = brake_traj(max(abs(v), abs(v_meas)), max(abs(w), abs(w_meas)), factor * T_R, aL, aT)[0]
                PREP["p"] = decel_prep(P, sdP, max(traj_travel(big), traj_travel(poses)), MARGIN)
            if PREP["p"] is None: return True, t_stop
            return not decel_violates(P, sdP, poses, MARGIN, PREP["p"])[0], t_stop
    if abs(v) < 1e-3 and abs(w) < 1e-3: return 0.0, 0.0, 1.0, 0.0, "idle"
    for s in SCALES:
        g, n2 = ok(v * s, w * s)
        if g: return v * s, w * s, s, n2, "ok" if s == 1.0 else "scaled"
    if abs(v) >= 1e-3 and abs(w) >= 1e-3:
        for s in SCALES:
            g, n2 = ok(0.0, w * s)
            if g: return 0.0, w * s, s, n2, "rotate_only"
    return 0.0, 0.0, 0.0, 0.0, "stop"
