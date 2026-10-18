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
alone (the core checks a pure rotation without the translation bands), then stop.  It only ever reduces."""
from __future__ import annotations
import numpy as np
from nav.safety_core import sd, free_along, free_rotation, need_dist, need_angle, MARGIN

SCALES = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1)


def _band(P, rect):
    x0, y0, x1, y1 = rect
    return bool(len(P)) and bool(((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)).any())


def governor(P, v, w, v_meas=0.0, w_meas=0.0, bands=None, factor=2.0):
    """P (N,2) robot frame (brain points U core points); bands: core band rects [[front], [rear]] (x0, y0, x1, y1)
    or None.  Returns (v, w, scale, need2, why)."""
    sd0 = float(sd(P).min()) if len(P) else np.inf
    bf = bands is not None and _band(P, bands[0]); br = bands is not None and _band(P, bands[1])

    def ok(vs, ws):
        if abs(vs) >= 1e-3:
            sign = 1 if vs > 0 else -1
            if (sign > 0 and bf) or (sign < 0 and br): return False, 0.0
            ve = max(abs(vs), abs(v_meas) if v_meas * vs > 0 else 0.0); need2 = factor * (need_dist(ve) + MARGIN)
            fr, mn = free_along(P, sign, ws / vs, need2)
        elif abs(ws) >= 1e-3:
            we = max(abs(ws), abs(w_meas) if w_meas * ws > 0 else 0.0); need2 = factor * need_angle(we)
            fr, mn = free_rotation(P, 1 if ws > 0 else -1, need2)
        else: return True, 0.0
        return fr >= need2 or (sd0 < MARGIN and mn >= sd0 - 1e-3), need2

    if abs(v) < 1e-3 and abs(w) < 1e-3: return 0.0, 0.0, 1.0, 0.0, "idle"
    for s in SCALES:
        g, n2 = ok(v * s, w * s)
        if g: return v * s, w * s, s, n2, "ok" if s == 1.0 else "scaled"
    if abs(v) >= 1e-3 and abs(w) >= 1e-3:
        for s in SCALES:
            g, n2 = ok(0.0, w * s)
            if g: return 0.0, w * s, s, n2, "rotate_only"
    return 0.0, 0.0, 0.0, 0.0, "stop"
