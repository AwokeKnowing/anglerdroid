"""Safety core: dedicated deterministic 30 Hz loop, separate process from the planner/nav brain (2026-10-17 design,
James; PROPOSAL branch safety-core-loop-2026-10-17, sim only, unmerged).

Per RS1 frame (the 30 Hz clock), in its own process (no GIL shared with the planner), pinned to one CPU, SCHED_FIFO
when the OS allows it (see docs/safety_core.md):
  1. operability: RS1 and RS2 frame age (time.monotonic() at decision - capture) <= AGE_MAX_S (0.10 s) and no missed
     30 Hz cadence (frame interval <= CADENCE_MAX_S) and RS1 view not occluded -> else STOP.
  2. deterministic transform: RS1 top-down ego labels (nav.perceive.Ego, same code as the brain) + RS2 forward cone
     (nav.perceive.Rs2Cloud, all heights 3 cm .. 1.0 m) collapsed to 2-D, then moved into the robot frame of NOW
     using the measured wheel-odometry twist over each frame's age.
  3. hard bands: 5 cm in front of the masked footprint and 5 cm at the rear edge of what RS1 sees behind it.  Any
     obstacle in a band -> no motion toward it (v toward it = 0, exactly; no creep).
  4. last-resort brake: the commanded (or measured, if faster) motion is swept along its arc for the distance needed
     to still stop if this tick passes it (one tick + command latency + measured braking distance at MAX braking).  If
     that sweep enters the margin, command zero now, with the vel ramp raised just as far as needed (normal ramp 3 ..
     max 5 turns/s^2) to stop before the obstacle.  No gradual throttling: a command passes unchanged or is zeroed.
The core is the only writer to the wheels (it owns WheelBase); the brain sends desired (v, w) over nav.bus and reads
wheel state back.  A brain command older than CMD_STALE_S, or a dead parent process, -> zero.

Measured (sim, emu.tools.brake_test, twin ODrive model, carpet): stop distance from command-zero to rest
  ramp 3: 0.31 m/s 0.050 m, 0.444 0.089, 0.609 0.135  -> d = 0.0987 v + v^2 / (2 * 2.47)
  ramp 5: 0.31 m/s 0.040 m, 0.444 0.068, 0.609 0.100  -> d = 0.0926 v + v^2 / (2 * 4.26)   pitch <= 6.4 deg, no skid
  ramp 8: 0.609 m/s 0.083 m but wheel skid 0.24 m/s, pitch 8.1 deg -> ramp 5 taken as MAX (sim; real bot: verify).
"""
from __future__ import annotations
import gc, json, math, os, sys, time, traceback
import numpy as np

import nav  # noqa (src on path)
from twin_geometry import FOOTPRINT_SAFE, SAFETY_MARGIN, WHEEL_R_M, TRACK_M

HZ = 30.0; T_TICK = 1.0 / HZ
AGE_MAX_S = float(os.environ.get("KEVIN_SC_AGE_MAX", "0.10"))
CADENCE_MAX_S = float(os.environ.get("KEVIN_SC_CADENCE_MAX", str(1.5 / HZ)))    # a dropped frame = missed cadence
PAIR_WAIT_S = float(os.environ.get("KEVIN_SC_PAIR_WAIT", "0.010"))   # max wait for the RS2 frame paired with this RS1 frame when it is due
CMD_STALE_S = 0.3
ERR_MAX = 30          # consecutive failing ticks (1 s) before the core zeroes the wheels and exits loudly
T_CMD = 0.010                         # core decision -> CAN frame on the wire (assumed; measured: see log t_sent - t_dec)
import robot_params as _RP                     # resolved robot config: plain numbers, read once here; missing/invalid = import fails = no core
# 2026-10-21 PARAMETER (James approval a4224de): ramps in turns follow the wheels, rim m/s^2 kept: stock 3.0 / 5.0, soft_v3_2 2.553 / 4.255
RAMP_NORMAL, RAMP_MAX = float(_RP.V["vel_ramp_tps2"]), float(_RP.V["vel_ramp_max_tps2"])
BRAKE = {RAMP_NORMAL: (0.0987, 2.47), RAMP_MAX: (0.0926, 4.26)}     # (lag s, decel m/s^2), fits above
BAND = 0.05; MARGIN = SAFETY_MARGIN; H_MAX = 1.0
STATE_FIELDS = ["t_dec", "period", "age1", "age2", "v_cmd", "w_cmd", "v_m", "w_m", "v_out", "w_out", "code", "ramp", "free",
                "need", "sd0", "band_f", "band_r", "n_pts", "rt", "cpu", "t_sent", "cap1", "cap2", "brakes", "plan_bugs"]
CODES = ["ok", "idle", "inop_age", "inop_cadence", "no_depth", "band_front", "band_rear", "brake", "cmd_stale", "escape", "collide"]

# ---- designed-deceleration rule (James 2026-10-20, t610-t613; docs/safety_time_to_intervene.md, emu/tools/scbench) ----
# Pass a command untouched unless, were it passed this tick, a stop at the DESIGNED decel (friction ellipse 0.2 g long /
# 0.1 g lat, wheel-rim limited) after the reaction lead (one tick + T_CMD + actuator lag) would bring the footprint
# inside MARGIN of any obstacle; inside the margin, no obstacle (connected point cluster) may lose clearance.  Then zero,
# braking at the designed decel.  No speed-independent band veto.  SIM ONLY by default: on when KEVIN_HAL=sim
# (KEVIN_SC_DECEL=0/1 overrides); elsewhere the rule above (bands + brake + escape) is unchanged.
SC_DECEL = os.environ.get("KEVIN_SC_DECEL", "1" if os.environ.get("KEVIN_HAL") == "sim" else "0") == "1"
from nav import limits as LIM                   # ONE accel config (nav/limits.py): the core uses the SAFETY limits
A_LONG, A_LAT = LIM.SAFETY_A_LONG, LIM.SAFETY_A_LAT
LAG_D = 0.0987                                  # actuator lag of the measured ramp-3 stop fit (ASSUMED at the design ramp; measure)
T_R = T_TICK + T_CMD + LAG_D                    # reaction lead: the command keeps running this long before decel starts
RAMP_DESIGN = A_LONG / (2 * math.pi * WHEEL_R_M)    # ODrive vel_ramp_rate (turns/s^2) giving A_LONG at the rim (3.65 nominal)
D_TOL = 1e-4                                    # numeric only (a 1 mm tolerance ratchets a slow drift into the margin)
D_LINK = 0.03                                   # points closer than this belong to one obstacle
D_CL_BAND = 0.5                                 # obstacles are labelled within this distance of the footprint


def d_stop(v, ramp):
    lag, a = BRAKE[ramp] if ramp in BRAKE else BRAKE[RAMP_MAX]
    v = abs(v); return v * lag + v * v / (2 * a)


def need_dist(v, ramp=RAMP_MAX):
    """distance travelled if this tick lets |v| through and the next tick has to stop it: one tick + command latency,
    then the measured stop at 'ramp'."""
    return abs(v) * (T_TICK + T_CMD) + d_stop(v, ramp)


def _edges(poly):
    P = poly; Q = np.roll(P, -1, 0); e = Q - P
    n = np.stack([e[:, 1], -e[:, 0]], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
    if (n @ P.mean(0) - (n * P).sum(1)).max() > 0: n = -n
    return n, (n * P).sum(1)


N_, C_ = _edges(FOOTPRINT_SAFE)


N32, C32 = N_.T.astype(np.float32), C_.astype(np.float32)


def sd(P):
    """signed distance of each point to the footprint (float32 matmul: ~3x cheaper than float64 on 6-10k points)"""
    return (np.asarray(P, np.float32) @ N32 - C32).max(1) if len(P) else np.zeros(0, np.float32)


def pose_on_arc(s, sign, kappa):
    """pose after arclength s (>= 0) along an arc of curvature kappa, direction sign (+1 fwd / -1 back)."""
    if abs(kappa) < 1e-6: return sign * s, 0.0, 0.0
    th = sign * s * kappa
    return math.sin(th) / kappa, (1 - math.cos(th)) / kappa, th


def to_body(P, x, y, th):
    c, s = math.cos(th), math.sin(th); d = P - (x, y)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


R_FOOT = float(np.hypot(FOOTPRINT_SAFE[:, 0], FOOTPRINT_SAFE[:, 1]).max())


def _sweep(P, poses):
    """min signed distance of points P to the footprint at each pose (x, y, th) -> (len(poses),), vectorised."""
    x, y, th = poses[:, 0, None], poses[:, 1, None], poses[:, 2, None]
    c, s = np.cos(th), np.sin(th); dx, dy = P[None, :, 0] - x, P[None, :, 1] - y
    L = np.stack([c * dx + s * dy, -s * dx + c * dy], -1)                       # (S, N, 2)
    return (L @ N_.T - C_).max(-1).min(-1)


def _first_hit(m, steps):
    hit = np.nonzero((m < MARGIN) & (steps > 0))[0]
    return (float(steps[hit[0]]) if len(hit) else float(steps[-1])), float(m[:hit[0] + 1].min() if len(hit) else m.min())


def free_along(P, sign, kappa, max_s, step=0.01):
    """first arclength where the footprint comes within MARGIN of any point (max_s if none), and min sd up to it."""
    if not len(P): return max_s, np.inf
    P = P[np.hypot(P[:, 0], P[:, 1]) < R_FOOT + max_s + MARGIN + 0.02]
    if not len(P): return max_s, np.inf
    steps = np.arange(0.0, max_s + 1e-9, step); steps = steps if steps[-1] >= max_s - 1e-9 else np.r_[steps, max_s]
    poses = np.array([pose_on_arc(s, sign, kappa) for s in steps])
    return _first_hit(_sweep(P, poses), steps)


def free_rotation(P, sign, max_a, step=0.02):
    if not len(P): return max_a, np.inf
    P = P[np.hypot(P[:, 0], P[:, 1]) < R_FOOT + MARGIN + 0.02]
    if not len(P): return max_a, np.inf
    steps = np.arange(0.0, max_a + 1e-9, step); steps = steps if steps[-1] >= max_a - 1e-9 else np.r_[steps, max_a]
    poses = np.stack([np.zeros_like(steps), np.zeros_like(steps), sign * steps], 1)
    return _first_hit(_sweep(P, poses), steps)


def alpha_max(ramp=RAMP_MAX):
    return BRAKE[ramp][1] / (TRACK_M / 2)


def need_angle(w, ramp=RAMP_MAX):
    lag, _ = BRAKE[ramp]; w = abs(w); return w * (T_TICK + T_CMD + lag) + w * w / (2 * alpha_max(ramp))


def motion_comp(P, v, w, dt):
    """points observed dt ago -> current robot frame, robot having moved along (v, w) for dt (wheel odometry)."""
    if not len(P) or dt <= 0 or (abs(v) < 1e-4 and abs(w) < 1e-4): return P
    th = w * dt
    if abs(w) > 1e-6: x, y = v / w * math.sin(th), v / w * (1 - math.cos(th))
    else: x, y = v * dt, 0.0
    return to_body(P, x, y, th)


class Bands:
    """front band: BAND ahead of the masked footprint; rear band: BAND at the rear edge of what RS1 sees behind the
    masked footprint (learned from which ego pixels return depth).  Width = footprint half-width (wheels)."""
    def __init__(self, ego):
        X, Y, body = ego.X, ego.Y, ego.body
        self.hw = float(np.abs(FOOTPRINT_SAFE[:, 1]).max())
        cols = np.abs(Y) <= self.hw
        self.x_front = float(X[body & cols].max()) + 0.005           # mask front (+ half px)
        self.x_body_rear = float(X[body & cols].min()) - 0.005
        self.x_rear = self.x_body_rear                               # updated from the visibility statistics
        self._seen = np.zeros(X.shape, np.float32); self._n = 0; self.X, self.Y, self.cols = X, Y, cols
    def learn(self, lab):
        from perception.labels import CLEAR, OBSTACLE
        if self._n >= 60: return
        self._seen += ((lab == CLEAR) | (lab == OBSTACLE)); self._n += 1
        if self._n in (10, 30, 60):
            vis = (self._seen / self._n > 0.5) & self.cols & (self.X < self.x_body_rear)
            if vis.any(): self.x_rear = float(self.X[vis].max()) + 0.005
    def front(self, P):
        return bool(len(P)) and bool(((P[:, 0] >= self.x_front) & (P[:, 0] < self.x_front + BAND) & (np.abs(P[:, 1]) <= self.hw)).any())
    def rear(self, P):
        return bool(len(P)) and bool(((P[:, 0] <= self.x_rear) & (P[:, 0] > self.x_rear - BAND) & (np.abs(P[:, 1]) <= self.hw)).any())
    def rects(self):
        return [[self.x_front, -self.hw, self.x_front + BAND, self.hw], [self.x_rear - BAND, -self.hw, self.x_rear, self.hw]]


D_STEP = 0.01                                   # sweep sample: every footprint point moves <= 1 cm between poses ...
D_STEP_MIN = 0.002                              # ... finer for slow commands (<= a quarter of the lead travel, >= 2 mm), so
                                                # the end rounding stays small; depends on the command only, not the horizon


def brake_traj(v, w, t_lead=None, a_long=None, a_lat=None):
    """poses (S, 3) of: hold (v, w) for t_lead, then a designed-decel stop along the same arc ((v, w) decay together;
    rim decel <= a_long * sqrt(1 - (a_lat_now / a_lat)^2)).  The whole path is the constant-(v, w) arc for an
    'equivalent time' tau in [0, t_lead + D] (D = integral of the decay), sampled on a FIXED grid tau_k = k * dtau
    (footprint travel D_STEP per sample, end rounded up to the grid): two horizons of the same command (core and
    governor) are sampled at the same poses, the longer one at a superset.  -> (poses, t_stop)"""
    t_lead = T_R if t_lead is None else t_lead; a_long = A_LONG if a_long is None else a_long; a_lat = A_LAT if a_lat is None else a_lat
    rim = abs(v) + abs(w) * TRACK_M / 2
    if rim < 1e-9: return np.zeros((1, 3)), 0.0
    u = 1.0; D = 0.0; t = t_lead; DT = 0.004; lat0 = abs(v * w)
    if lat0 < 0.1 * a_lat:                                          # lateral never binds: closed form
        T = rim / a_long; D = T / 2; t += T
    else:
        while u > 0:
            r = a_long * math.sqrt(max(1.0 - (lat0 * u * u / a_lat) ** 2, 0.01)) / rim; du = min(u, r * DT); h = du / r
            D += (u - du / 2) * h; u -= du; t += h
    tau_end = t_lead + D; rate = abs(v) + abs(w) * R_FOOT; dtau = min(D_STEP, max(D_STEP_MIN, rate * T_R / 4)) / rate
    tau = np.arange(0, int(math.ceil(tau_end / dtau - 1e-9)) + 1) * dtau
    th = w * tau
    if abs(w) < 1e-9: x, y = v * tau, np.zeros_like(tau)
    else: x, y = v / w * np.sin(th), v / w * (1 - np.cos(th))
    return np.stack([x, y, th], 1), t


def _sd_matrix_T(P, poses):
    """signed distance of every point to the footprint at every pose -> (N, S).  One float32 matmul with the footprint
    normals rotated per pose (n . R^T (p - t) = (R n) . p - (R n) . t), edge-major so the max over the 12 edges runs
    on contiguous (N, S) blocks."""
    th = poses[:, 2]; c, s_ = np.cos(th)[None, :], np.sin(th)[None, :]
    Mx = N_[:, 0, None] * c - N_[:, 1, None] * s_; My = N_[:, 0, None] * s_ + N_[:, 1, None] * c          # (E, S)
    off = (Mx * poses[None, :, 0] + My * poses[None, :, 1] + C_[:, None]).astype(np.float32).ravel()
    W = np.stack([Mx.ravel(), My.ravel()], 0).astype(np.float32)                                          # (2, E*S)
    D = np.asarray(P, np.float32) @ W; D -= off; S_ = len(poses)
    out = D[:, :S_].copy()
    for e in range(1, len(N_)): np.maximum(out, D[:, e * S_:(e + 1) * S_], out=out)
    return out


try:
    from scipy.ndimage import label as _nd_label
except Exception:                       # pragma: no cover
    _nd_label = None
_K8 = np.ones((3, 3), bool)


def _clusters(P, link=D_LINK):
    """connected components (cells of size link, 8-connected) -> label per point"""
    key = np.floor(P / link).astype(np.int64); k0 = key.min(0); key -= k0; shp = key.max(0) + 1
    if _nd_label is not None and shp[0] * shp[1] <= 250000:
        g = np.zeros(tuple(shp), bool); g[key[:, 0], key[:, 1]] = True
        L, _ = _nd_label(g, structure=_K8); return L[key[:, 0], key[:, 1]]
    cells, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel(); n = len(cells)
    idx = {k: i for i, k in enumerate(zip(cells[:, 0].tolist(), cells[:, 1].tolist()))}; par = list(range(n))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for (cx, cy), i in idx.items():
        for dx, dy in ((1, -1), (1, 0), (1, 1), (0, 1)):
            j = idx.get((cx + dx, cy + dy))
            if j is not None:
                ra, rb = f(i), f(j)
                if ra != rb: par[ra] = rb
    root = np.array([f(i) for i in range(n)]); return root[inv]


D_CELL = 0.002                                  # points are merged to the centres of D_CELL cells before the sweep (a box
D_CELL_ERR = D_CELL / math.sqrt(2)              # face is ~10k RS2 points stacked in xy, ~5x fewer cells); |sd error| <= this,
                                                # added to the margin (conservative) -- well under the depth noise (5-8 mm)


def decel_prep(P, sdP, travel, margin=MARGIN):
    """candidate cells (those that can get within margin when every footprint point moves <= travel), grouped by
    obstacle -> (Q (n, 2) cell centres, group starts) or None"""
    reach = travel + margin + 0.005 + D_CELL_ERR
    if not (sdP <= reach).any(): return None
    # obstacles are labelled on a FIXED band around the footprint (not on this trajectory's candidates), so the core and
    # the governor (longer horizon -> more candidates) see the same obstacles; an obstacle's closest point is always a
    # candidate when any of its points is, and its other points cannot reach the margin on this trajectory
    B = P[sdP <= max(D_CL_BAND, reach)]; k = np.floor(B / D_CELL).astype(np.int64)
    u = np.unique((k[:, 0] + (1 << 20)) * (1 << 21) + (k[:, 1] + (1 << 20)))
    C = (np.stack([u // (1 << 21) - (1 << 20), u % (1 << 21) - (1 << 20)], 1) + 0.5).astype(np.float32) * np.float32(D_CELL)
    sdC = sd(C); cand = sdC <= reach
    if not cand.any(): return None
    lab = _clusters(C)[cand]; Q = C[cand]; o = np.argsort(lab, kind="stable"); ls = lab[o]
    return Q[o], np.r_[0, np.nonzero(np.diff(ls))[0] + 1]


def traj_travel(poses):
    return float((np.hypot(poses[:, 0], poses[:, 1]) + np.abs(poses[:, 2]) * R_FOOT).max())


def decel_violates(P, sdP, poses, margin=MARGIN, prep=None):
    """per obstacle: clearance along 'poses' must stay >= min(its clearance now, margin).  Only points that can get
    within margin on this trajectory are evaluated (|motion of any footprint point| <= |dxy| + |dth| R_FOOT).
    prep: decel_prep() for a travel >= this trajectory's (reused across candidate commands).  -> (violates, min sd)"""
    if not len(P): return False, np.inf
    if prep is None: prep = decel_prep(P, sdP, traj_travel(poses), margin)
    if prep is None: return False, float(sdP.min())
    Q, st = prep[0], prep[1]; M = _sd_matrix_T(Q, poses); cm = np.minimum.reduceat(M, st, axis=0)       # (obstacles, S)
    return bool((cm < np.minimum(cm[:, :1], margin + D_CELL_ERR) - D_TOL).any()), float(M.min()) - D_CELL_ERR


def decide_decel(P, bands, v_c, w_c, v_m, w_m, margin=MARGIN, t_lead=None, sdP=None):
    """designed-decel rule; same returns as decide().  Called by decide() when SC_DECEL (idle/coast handled there)."""
    if sdP is None: sdP = sd(P) if len(P) else np.zeros(0)
    sd0 = float(sdP.min()) if len(P) else np.inf
    bf, br = (bands.front(P), bands.rear(P)) if bands is not None else (False, False)     # logged only (no veto)
    v_e = v_m if (v_m * v_c > 0 and abs(v_m) > abs(v_c)) else v_c
    w_e = w_m if (w_m * w_c > 0 and abs(w_m) > abs(w_c)) else w_c
    poses, t_stop = brake_traj(v_e, w_e, t_lead)
    bad, mn = decel_violates(P, sdP, poses, margin)
    if bad: return 0.0, 0.0, ("collide" if sd0 < margin else "brake"), RAMP_DESIGN, mn, t_stop, sd0, bf, br
    return v_c, w_c, ("escape" if sd0 < margin else "ok"), RAMP_NORMAL, mn, t_stop, sd0, bf, br


def decide(P, bands, v_c, w_c, v_m, w_m):
    """pure function of the tick's inputs -> (v_out, w_out, code, ramp, free, need, sd0, band_f, band_r)."""
    sdP = sd(P) if len(P) else None; sd0 = float(sdP.min()) if len(P) else np.inf
    if SC_DECEL and (abs(v_c) >= 1e-3 or abs(w_c) >= 1e-3): return decide_decel(P, bands, v_c, w_c, v_m, w_m, sdP=sdP)
    bf, br = bands.front(P), bands.rear(P)
    moving = abs(v_m) > 0.02 or abs(w_m) > 0.05
    if abs(v_c) < 1e-3 and abs(w_c) < 1e-3:
        if not moving: return 0.0, 0.0, "idle", RAMP_NORMAL, np.nan, 0.0, sd0, bf, br
        # coasting to the zero command: is the normal ramp enough?
        if abs(v_m) > 0.02:
            fr, _ = free_along(P, 1 if v_m > 0 else -1, w_m / v_m if abs(v_m) > 1e-3 else 0.0, need_dist(v_m, RAMP_NORMAL) + 0.05)
            ramp = _ramp_for(fr, v_m)
            return 0.0, 0.0, "brake" if ramp > RAMP_NORMAL else "idle", ramp, fr, need_dist(v_m, RAMP_NORMAL), sd0, bf, br
        return 0.0, 0.0, "idle", RAMP_NORMAL, np.nan, 0.0, sd0, bf, br
    if abs(v_c) >= 1e-3:
        sign = 1 if v_c > 0 else -1
        if (sign > 0 and bf) or (sign < 0 and br):
            return 0.0, 0.0, "band_front" if sign > 0 else "band_rear", _brake_ramp(P, v_m, w_m), np.nan, 0.0, sd0, bf, br
        v_e = max(abs(v_c), abs(v_m) if v_m * v_c > 0 else 0.0); kappa = w_c / v_c
        need = need_dist(v_e)
        fr, mn = free_along(P, sign, kappa, need + 0.02)
        if fr >= need: return v_c, w_c, "ok", RAMP_NORMAL, fr, need, sd0, bf, br
        # already inside the margin: a motion that never gets closer anywhere is an escape, allowed
        if sd0 < MARGIN and mn >= sd0 - 1e-3: return v_c, w_c, "escape", RAMP_NORMAL, fr, need, sd0, bf, br
        return 0.0, 0.0, "brake", _brake_ramp(P, v_m, w_m), fr, need, sd0, bf, br
    sign = 1 if w_c > 0 else -1; w_e = max(abs(w_c), abs(w_m) if w_m * w_c > 0 else 0.0)
    need = need_angle(w_e); fa, mn = free_rotation(P, sign, need + 0.02)
    if fa >= need: return v_c, w_c, "ok", RAMP_NORMAL, fa, need, sd0, bf, br
    if sd0 < MARGIN and mn >= sd0 - 1e-3: return v_c, w_c, "escape", RAMP_NORMAL, fa, need, sd0, bf, br
    return 0.0, 0.0, "brake", RAMP_MAX, fa, need, sd0, bf, br


def _ramp_for(avail, v):
    """smallest ramp (normal .. max) whose measured stop distance (+ command latency) fits in 'avail'."""
    v = abs(v); dn, dm = v * T_CMD + d_stop(v, RAMP_NORMAL), v * T_CMD + d_stop(v, RAMP_MAX)
    if avail >= dn: return RAMP_NORMAL
    if avail <= dm or dn <= dm: return RAMP_MAX
    return RAMP_NORMAL + (RAMP_MAX - RAMP_NORMAL) * (dn - avail) / (dn - dm)


def _brake_ramp(P, v_m, w_m):
    if abs(v_m) <= 0.02: return RAMP_MAX if abs(w_m) > 0.05 else RAMP_NORMAL
    fr, _ = free_along(P, 1 if v_m > 0 else -1, w_m / v_m, need_dist(v_m, RAMP_NORMAL) + 0.05)
    return _ramp_for(fr, v_m)                    # fr already stops MARGIN short of the obstacle


def governor(P, v, w, v_meas=0.0, factor=2.0):
    """PLANNER side (brain): largest scale of (v, w) whose sweep keeps factor x the core's threshold (need + margin)
    clear, so the core rarely fires.  Returns (v, w, scale, need2)."""
    sd0 = float(sd(P).min()) if len(P) else np.inf
    for s in (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.0):
        vs, ws = v * s, w * s
        if abs(vs) >= 1e-3:
            ve = max(abs(vs), abs(v_meas) if v_meas * vs > 0 else 0.0); need2 = factor * (need_dist(ve) + MARGIN)
            fr, mn = free_along(P, 1 if vs > 0 else -1, ws / vs, need2, step=0.02)
        elif abs(ws) >= 1e-3:
            need2 = factor * need_angle(ws); fr, mn = free_rotation(P, 1 if ws > 0 else -1, need2, step=0.04)
        else: return 0.0, 0.0, s, 0.0
        # clear for 2x the threshold, or a motion that gets no closer to anything (leaving a tight spot)
        if fr >= need2 or mn >= sd0 - 1e-3: return vs, ws, s, need2
    return 0.0, 0.0, 0.0, 0.0


class FrameClock:
    """camera stamp -> monotonic (sim stamps are monotonic; other domains: latency-floor offset)."""
    def __init__(self): self.off = None; self.t_off = 0.0
    def capture(self, stamp, t_recv):
        d = t_recv - stamp
        if 0.0 <= d < 2.0: return stamp
        if self.off is None or d < self.off or abs(d - self.off) > 1.0 or t_recv - self.t_off > 10.0: self.off, self.t_off = d, t_recv
        return min(stamp + self.off, t_recv)


def _realtime(cpu):
    got = []
    try: os.sched_setaffinity(0, {cpu}); got.append("cpu%d" % cpu)
    except Exception as e: got.append("affinity failed (%s)" % e.__class__.__name__)
    try: os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(os.environ.get("KEVIN_SC_RTPRIO", "50")))); got.append("SCHED_FIFO")
    except Exception as e:
        got.append("SCHED_FIFO denied (%s)" % e.__class__.__name__)
        try: os.nice(-10); got.append("nice -10")
        except Exception: got.append("nice unchanged")
    try:
        import resource; got.append("rtprio limit %s" % (resource.getrlimit(resource.RLIMIT_RTPRIO),))
    except Exception: pass
    return ", ".join(got)


def core_main(rs1_serial, rs2_intr, rs2_shape, parent_pid, log_path=None):
    """entry point of the safety core process (multiprocessing 'spawn')."""
    cpu = int(os.environ.get("KEVIN_SC_CPU", str((os.cpu_count() or 1) - 1)))
    rt = _realtime(cpu); print("safety_core: pid %d %s" % (os.getpid(), rt), flush=True)
    print("safety_core: robot config %s wheels %s hash %s dump %s | ramp %.4f/%.4f turns/s^2" % (_RP.V["robot"], _RP.V["wheels"], _RP.HASH, _RP.DUMP, RAMP_NORMAL, RAMP_MAX), flush=True)
    from nav import bus
    from nav.perceive import Ego, Rs2Cloud
    B = bus.open_all(rs2_shape, create=False)
    from cameras import RSCamera                                  # the core owns RS1 (top-down): its frames never
    cam = RSCamera(rs1_serial, compute_pointcloud=True)           # wait on the brain process
    rs1_intr = cam.depth_intrinsics()
    B["rs1info"].write(f=np.array([rs1_intr[k] for k in ("fx", "fy", "ppx", "ppy", "width", "height")], float), set=1)
    import wheelbase
    cwd = os.getcwd(); os.chdir(nav.SRC)
    try: wb = wheelbase.WheelBase()
    finally: os.chdir(cwd)
    ego = Ego(); ego.intr = rs1_intr; rs2 = Rs2Cloud(rs2_intr); bands = Bands(ego)
    clk1, clk2 = FrameClock(), FrameClock()
    EB = B.get("ego") if os.environ.get("KEVIN_SC_EGO_PUB", "1") == "1" else None; nfr = 0   # sc_ego: labels for the brain (nav.ego_shared)
    LOG = open(log_path, "a", buffering=1 << 16) if log_path else None
    gc.collect(); gc.freeze(); gc.disable()                       # no GC pauses inside the loop (refcounting still frees)
    s1 = s2 = 0; cap1p = None; cap2 = None; cap2p = None; P2 = np.zeros((0, 2), np.float32); r1 = {}; r2 = {}; n2 = 0; lat2 = None
    ramp_now = RAMP_NORMAL; t_prev = None; brakes = 0; viz_t = 0.0; ps = 0; n_stuck = 0; errs = 0
    lchk = LIM.LimitCheck(TRACK_M)                                 # brain commands beyond the SAFETY accel limits = planner bug
    while True:
        try:
            if os.getppid() != parent_pid: wb.twist(0.0, 0.0); wb.shutdown(); return     # brain gone -> stop, exit
            fs = cam.wait_frameset(int(CADENCE_MAX_S * 1e3))           # next RS1 frame, or wake up and stop
            t_recv = time.monotonic(); new1 = fs is not None and cam.ingest(fs)
            if new1:
                cam.publish(); B["rs1"].write(verts=cam.verts, n=len(cam.verts), stamp=cam.depth_stamp, t_recv=t_recv)
            t_dec = time.monotonic()
            vl, vr = wb.get_wheel_velocities_mps(); v_m = 0.5 * (vl + vr); w_m = (vr - vl) / TRACK_M
            cur = wb.get_motor_currents()
            B["wheel"].write(vl=vl, vr=vr, t=t_dec, il=cur[0] if cur else np.nan, ir=cur[1] if cur else np.nan, iage=cur[2] if cur else np.nan, batt=float(getattr(wb, "_batt_pct", 0) or 0))
            sq, c = B["cmd"].try_read()
            if sq is None: v_c = w_c = 0.0; t_c = -1e9; n_stuck += 1      # cmd writer paused mid-write -> treated as stale
            else: v_c, w_c, t_c = float(c["v"][0]), float(c["w"][0]), float(c["t"][0])
            code = None
            if t_dec - t_c > CMD_STALE_S: v_c = w_c = 0.0; code_c = "cmd_stale"
            else:
                code_c = None
                if lchk(v_c, w_c, t_c) and (lchk.bugs <= 3 or lchk.bugs % 100 == 0):
                    print("safety_core: PLANNER BUG #%d: command (v, w, t) %s exceeds the safety accel limits (%.2f / %.2f m/s^2)" % (lchk.bugs, lchk.last, A_LONG, A_LAT), flush=True)
            if new1:
                tA = time.monotonic(); cap1 = clk1.capture(float(cam.depth_stamp), t_recv)
                per = None if cap1p is None else cap1 - cap1p; cap1p = cap1
                ego.update(cam.verts); P1 = ego.obstacle_xy(max_n=1500); valid = ego.valid_fraction(); bands.learn(ego.lab); tB = time.monotonic()
                nfr += 1
                if EB is not None:     # publish what was just computed (seqlock single writer: never waits on readers)
                    lm = ego.lab_map
                    EB.write(lab=ego.lab, hgt=ego.hgt, lab_map=ego.lab if lm is None else lm, has_map=0 if lm is None else 1, obs_mask=ego.obs_mask,
                             occluded=1 if getattr(ego, "occluded", False) else 0, stamp=cam.depth_stamp, t_cap=cap1, t_pub=time.monotonic(),
                             frame=nfr, rs1_seq=B["rs1"].seq)
            else:
                per = t_dec - (cap1p or t_dec); P1 = None; valid = 0.0; tA = tB = t_dec
            # RS2: of the frames not yet used, take the one captured nearest to this RS1 frame (the two cameras run
            # unsynchronised; taking simply the newest skips a frame whenever RS1 is handled late, and misses one when RS2
            # lands just after the read).  If the RS2 frame paired with this RS1 frame is due within PAIR_WAIT_S (expected
            # delivery = previous capture + 1/HZ + measured RS2 delivery latency), wait for it.
            ref = cap1p if new1 else None; t_end = time.monotonic() + PAIR_WAIT_S; best = None
            while True:
                if B["rs2"].seq != s2:
                    sq, rr = B["rs2"].try_read(out=r2 if r2 else None)
                    if sq is None:                                  # writer paused mid-write: no RS2 this tick (stale frame;
                        r2 = {}; best = None; n_stuck += 1          # age2 grows -> inop_age -> zero).  Used to raise and kill the core.
                        if n_stuck == 1 or n_stuck % 30 == 0: print("safety_core: rs2 bus writer stuck -> stale (%d)" % n_stuck, flush=True)
                        break
                    s2, r2 = sq, rr; best = None
                    nw = int(r2["n"][0]); j = (nw - 1) % bus.RS2_RING
                    d = float(r2["t_recv"][j]) - clk2.capture(float(r2["stamp"][j]), float(r2["t_recv"][j]))
                    lat2 = d if lat2 is None else 0.9 * lat2 + 0.1 * d
                if best is None and r2:
                    nw = int(r2["n"][0])
                    for k in range(max(n2 + 1, nw - bus.RS2_RING + 1), nw + 1):
                        j = (k - 1) % bus.RS2_RING; c2 = clk2.capture(float(r2["stamp"][j]), float(r2["t_recv"][j]))
                        if best is None or ref is None or abs(c2 - ref) <= abs(best[2] - ref): best = (k, j, c2)
                due = (ref is not None and cap2p is not None and lat2 is not None and (best is None or best[2] < ref - 0.5 / HZ)
                       and cap2p + 1.0 / HZ <= ref + 0.5 / HZ and cap2p + 1.0 / HZ + lat2 <= t_end)
                if not due or time.monotonic() >= t_end: break
                time.sleep(0.0005)
            if best is not None:
                n2, j, cap2 = best
                per2 = None if cap2p is None else cap2 - cap2p; cap2p = cap2
                sp = B["rs2pose"].try_read()[1]
                if sp is not None and int(sp["set"][0]): rs2.set_pose(sp["T"], sp["R"].reshape(3, 3))
                P2, _ = rs2.points(r2["depth"][j], h_max=H_MAX)
            else: per2 = None
            tC = time.monotonic()                                          # frame age AT DECISION (after the transform)
            age1 = (tC - cap1p) if cap1p is not None else np.inf; age2 = (tC - cap2) if cap2 is not None else np.inf
            if not new1 or age1 > AGE_MAX_S or age2 > AGE_MAX_S:
                code = "inop_age"
            elif (per is not None and per > CADENCE_MAX_S) or (per2 is not None and per2 > CADENCE_MAX_S):
                code = "inop_cadence"
            elif valid < 0.05:
                code = "no_depth"
            free = need = np.nan; sd0 = np.nan; bf = br = False; npts = 0; P = np.zeros((0, 2), np.float32)
            if code is not None:
                v_o = w_o = 0.0; ramp = RAMP_MAX if (abs(v_m) > 0.02 or abs(w_m) > 0.05) else RAMP_NORMAL
            else:
                P = np.concatenate([motion_comp(P1, v_m, w_m, age1), motion_comp(P2, v_m, w_m, age2)]) if len(P2) else motion_comp(P1, v_m, w_m, age1)
                npts = len(P)
                v_o, w_o, code, ramp, free, need, sd0, bf, br = decide(P, bands, v_c, w_c, v_m, w_m)
                if code_c and code == "idle": code = code_c
            if ramp != ramp_now:                                       # brake ramp first, then the zero command
                with wb.bus_lock:
                    for ax in (wb.left, wb.right): ax.write_property_nowait('axis0.controller.config.vel_ramp_rate', float(ramp))
                ramp_now = ramp
            if code in ("brake", "collide", "band_front", "band_rear", "inop_age", "inop_cadence", "no_depth") and (abs(v_c) > 1e-3 or abs(w_c) > 1e-3): brakes += 1
            tD = time.monotonic(); wb.twist(v_o, w_o); t_sent = time.monotonic()
            period = None if t_prev is None else t_dec - t_prev; t_prev = t_dec
            f = [t_dec, period if period is not None else np.nan, age1, age2, v_c, w_c, v_m, w_m, v_o, w_o, CODES.index(code) if code in CODES else -1,
                 ramp, free, need, sd0, bf, br, npts, 0, cpu, t_sent, cap1p or np.nan, cap2 or np.nan, brakes, lchk.bugs]
            B["state"].write(f=np.array(f, np.float64))
            if new1 and t_dec - viz_t > 0.2 and len(P):                                # 5 Hz snapshot for the viewer overlay
                k = min(len(P), 2000); B["viz"].write(pts=P[:k].astype(np.float32), n=np.array([k]), bands=np.array(bands.rects(), np.float64).ravel(), t=np.array([t_dec]))
                viz_t = t_dec
            if LOG is not None:
                LOG.write(json.dumps([round(x, 4) if isinstance(x, float) and math.isfinite(x) else (None if isinstance(x, float) else x) for x in
                                      [t_dec, period, age1, age2, v_c, w_c, v_m, w_m, v_o, w_o, code, ramp, free, need, sd0, bf, br, npts, t_sent, per, per2, valid, t_dec - t_recv, tB - tA, tC - tB, tD - tC, t_sent - tD]]) + "\n")
                ps += 1
                if ps % 30 == 0: LOG.flush()
            errs = 0
        except Exception:
            # never die silently: zero the wheels, report, keep looping; persistent failure -> zero + exit loudly (nav
            # sees the core dead, stops all goals and restarts it)
            errs += 1
            try: wb.twist(0.0, 0.0)
            except Exception: pass
            print("safety_core: tick error %d (wheels zeroed):\n%s" % (errs, traceback.format_exc()), flush=True)
            if errs >= ERR_MAX:
                print("safety_core: %d consecutive tick errors -> wheels zero, EXIT" % errs, flush=True)
                try: wb.twist(0.0, 0.0); wb.shutdown()
                except Exception: pass
                sys.exit(3)
            time.sleep(T_TICK)


def _read_rs1(blk, out):
    """seqlock read of only the n valid verts (the block holds MAX_VERTS)."""
    for _ in range(200):
        s0 = int(blk.hdr[0])
        if s0 & 1: time.sleep(0.0001); continue
        n = int(blk.arr["n"][0])
        if "verts" not in out or len(out["verts"]) != n: out["verts"] = np.empty((n, 3), np.float32)
        np.copyto(out["verts"], blk.arr["verts"][:n]); out["stamp"] = blk.arr["stamp"].copy(); out["t_recv"] = blk.arr["t_recv"].copy()
        if int(blk.hdr[0]) == s0: return s0, n, out
    raise TimeoutError("bus rs1: writer stuck")


def _wait(blk, last, timeout):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        s = int(blk.hdr[0])
        if s != last and not (s & 1): return True
        time.sleep(0.0003)
    return False
