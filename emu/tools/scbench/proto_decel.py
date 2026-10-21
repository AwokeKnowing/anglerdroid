"""PROTOTYPE (not wired in): designed-deceleration intervention rule (James t610-t612), standalone, same inputs/outputs
as nav.safety_core.decide so emu/tools/scbench can run both on identical scenes.

Rule: pass (v, w) untouched unless, were it passed this tick, a stop at the DESIGNED decel (friction ellipse 0.2 g long /
0.1 g lat, wheel-rim limited) after the reaction lead (one tick + T_CMD + actuator lag) would bring the footprint inside
MARGIN of any point (per point: no point already inside MARGIN may get closer).  Then: zero, braking at the designed decel.
The stop is modelled as a proportional (v, w) decay along the commanded arc (both wheels reach 0 together)."""
from __future__ import annotations
import math
import numpy as np
from nav.safety_core import N_, C_, sd, MARGIN, T_TICK, T_CMD, TRACK_M, R_FOOT

G = 9.81
A_LONG, A_LAT = 0.2 * G, 0.1 * G
LAG = 0.0987                 # measured actuator lag of the ramp-3 stop fit (ASSUMED equal at the design ramp; measure)
T_R = T_TICK + T_CMD + LAG   # reaction lead: command keeps running this long before decel starts
DT = 0.005
CODES = ("ok", "idle", "brake", "collide", "escape", "envelope")


def brake_traj(v, w, t_lead=T_R):
    """poses (x, y, th) along: hold (v, w) for t_lead, then a designed-decel stop.  Returns (poses (S, 3), t_stop, envelope)."""
    x = y = th = 0.0; poses = [(0.0, 0.0, 0.0)]; t = 0.0; env = False
    rim = abs(v) + abs(w) * TRACK_M / 2                       # fastest wheel rim speed per unit of (v, w) scale
    u = 1.0
    def step(uv, dt):
        nonlocal x, y, th
        vv, ww = v * uv, w * uv
        x += vv * math.cos(th + ww * dt / 2) * dt; y += vv * math.sin(th + ww * dt / 2) * dt; th += ww * dt
    ds_max = 0.01; da_max = math.radians(1.0)
    n_hold = max(1, int(math.ceil(max(abs(v) * t_lead / ds_max, abs(w) * t_lead / da_max, t_lead / 0.02))))
    for _ in range(n_hold): step(1.0, t_lead / n_hold); poses.append((x, y, th))
    t = t_lead
    while u > 0 and rim > 1e-6:
        alat = abs(v * w) * u * u
        f = 1.0 - (alat / A_LAT) ** 2
        if f <= 0.01: f = 0.01; env = True                     # outside the lateral envelope: brake on the boundary
        r = A_LONG * math.sqrt(f) / rim                       # du/dt (rim decel = r * rim <= A_LONG*sqrt(..))
        du = min(u, r * DT); dt = du / r
        step(u - du / 2, dt); u -= du; t += dt
        if (len(poses) and (math.hypot(x - poses[-1][0], y - poses[-1][1]) >= 0.005 or abs(th - poses[-1][2]) >= math.radians(0.5))) or u <= 0:
            poses.append((x, y, th))
    return np.array(poses), t, env


def _sd_matrix(P, poses):
    x, y, th = poses[:, 0, None], poses[:, 1, None], poses[:, 2, None]
    c, s = np.cos(th), np.sin(th); dx, dy = P[None, :, 0] - x, P[None, :, 1] - y
    L = np.stack([c * dx + s * dy, -s * dx + c * dy], -1)
    return (L @ N_.T - C_).max(-1)                            # (S, N)


TOL = 1e-4                   # numeric only: a 1 mm tolerance lets a slow drift ratchet 1 mm/tick into the margin (bench: wall_drift)
LINK = 0.03                  # points closer than this belong to the same obstacle


def clusters(P, link=LINK):
    """connected components of the point set (grid union-find, cell = link) -> label per point."""
    if not len(P): return np.zeros(0, int)
    key = np.floor(P / link).astype(np.int64); cells = {}
    for i, k in enumerate(map(tuple, key)): cells.setdefault(k, []).append(i)
    par = {k: k for k in cells}
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for (cx, cy) in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (cx + dx, cy + dy)
                if nb in cells: par[f(nb)] = f((cx, cy))
    roots = {}; lab = np.empty(len(P), int)
    for k, idx in cells.items(): lab[idx] = roots.setdefault(f(k), len(roots))
    return lab


def violates(P, poses, margin=MARGIN):
    """per OBSTACLE (connected cluster): its clearance along the trajectory must stay >= min(its clearance now, margin).
    (Per point is too strict: sliding along a parallel wall moves individual wall points closer while the wall's
    clearance is unchanged.  Global, as today's core, is too loose: one point deep inside lets every other obstacle
    come as close.)  -> (violates, min sd along, worst cluster)"""
    if not len(P): return False, np.inf, -1
    reach = float(np.hypot(poses[:, 0], poses[:, 1]).max()) + R_FOOT + margin + 0.02
    keep = np.hypot(P[:, 0], P[:, 1]) <= reach
    if not keep.any(): return False, np.inf, -1
    Q = P[keep]; M = _sd_matrix(Q, poses); lab = clusters(Q); nc = lab.max() + 1
    cm = np.full((len(poses), nc), np.inf)
    for c in range(nc): cm[:, c] = M[:, lab == c].min(1)
    bad = cm < np.minimum(cm[0], margin)[None, :] - TOL
    return bool(bad.any()), float(M.min()), int(bad.any(0).argmax()) if bad.any() else -1


def decide(P, bands, v_c, w_c, v_m, w_m, margin=MARGIN, t_lead=T_R):
    """same signature/returns as nav.safety_core.decide: (v_out, w_out, code, ramp(=design decel m/s^2), free, need, sd0, bf, br)."""
    sd0 = float(sd(P).min()) if len(P) else np.inf
    if abs(v_c) < 1e-3 and abs(w_c) < 1e-3: return 0.0, 0.0, "idle", A_LONG, np.nan, 0.0, sd0, False, False
    # effective motion: the command, or the measured twist where it is faster in the same direction
    v_e = v_c if not (v_m * v_c > 0 and abs(v_m) > abs(v_c)) else v_m
    w_e = w_c if not (w_m * w_c > 0 and abs(w_m) > abs(w_c)) else w_m
    poses, t_stop, env = brake_traj(v_e, w_e, t_lead)
    bad, mn, _ = violates(P, poses, margin)
    if bad: return 0.0, 0.0, ("collide" if sd0 < margin else "brake"), A_LONG, mn, t_stop, sd0, False, False
    return v_c, w_c, ("envelope" if env else "escape" if sd0 < margin else "ok"), A_LONG, mn, t_stop, sd0, False, False


def governor(P, v, w, factor=2.0, scales=(1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1)):
    """planner side: largest scale that the core rule passes with factor x the reaction lead (so the planner slows
    before the core would have to act); same margin as the core (the planner's own path margin is the planner's
    business, not the governor's: a governor margin of 2x blocks the 4 cm squeeze); then rotate-only; then stop."""
    if abs(v) < 1e-3 and abs(w) < 1e-3: return 0.0, 0.0, 1.0, 0.0, "idle"
    for s in scales:
        o = decide(P, None, v * s, w * s, 0.0, 0.0, margin=MARGIN, t_lead=factor * T_R)
        if o[2] in ("ok", "escape", "envelope"): return v * s, w * s, s, 0.0, "ok" if s == 1 else "scaled"
    if abs(v) >= 1e-3 and abs(w) >= 1e-3:
        for s in scales:
            o = decide(P, None, 0.0, w * s, 0.0, 0.0, margin=MARGIN, t_lead=factor * T_R)
            if o[2] in ("ok", "escape", "envelope"): return 0.0, w * s, s, 0.0, "rotate_only"
    return 0.0, 0.0, 0.0, 0.0, "stop"
