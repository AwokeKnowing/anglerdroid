"""Collision-only safety envelope (replaces the old always-on slow-down/reflex logic).

Each tick, the commanded (v, w) is swept forward for its own stopping distance (reaction + braking at a_brake)
with the real footprint (twin SIL12 12-gon + margin).  If the sweep hits a fresh depth point, the command is
scaled down (keeping curvature) to the largest scale that is safe; nothing else is ever touched, so normal
driving with clearance passes through unchanged.  No fresh valid depth -> immobilised (v = w = 0).

Staleness (2026-10-17, approved for sim test, branch safety-staleness-now): frame age is measured against the
current time (time.monotonic() when the command is computed), not against the frame's own stamp (old: now=t_frame ->
age 0 always, so the 0.25 s stale check never fired and Kevin drove on 0.25-0.71 s old frames).  The point cloud
describes the world at capture time, so the robot has already moved for 'age' seconds on it: the age is added to the
reaction time of the stopping sweep (the envelope grows with frame age).  Frames older than STALE_S (0.5 s) ->
immobilised.  Stopping model (a_brake 1.5, t_react 0.10) and footprint are unchanged.
"""
from __future__ import annotations
import math, time
import numpy as np

import nav  # noqa (src on path)
from twin_geometry import VEL_RAMP_TPS2 as _TG_VR, SIL12, FOOTPRINT_SAFE, SAFETY_MARGIN, WHEEL_R_M                    # twin footprint polygon (+ sim deck corners)

import os

# MEASURED stopping model (2026-10-17 sim stop test, emu/tools/stoptest; superseded proposal branch
# safety-measured-model-on-pr62-2026-10-17), active only with KEVIN_SAFETY_MEASURED=1 (default OFF):
#   braking: ODrive vel_ramp 3 turns/s^2 (src/wheelbase.py) -> 3 * 2 pi * 0.08565 = 1.614 m/s^2 at the wheel
#   T_ACT 0.04 s: stop command -> wheel decel (CAN/UDP 0-7 ms + velocity-loop lag ~31 ms)
#   reaction per tick = (now - frame capture) + one frame interval + T_ACT; stale > 0.25 s
A_BRAKE_MEAS = _TG_VR * 2 * math.pi * WHEEL_R_M
T_ACT = 0.04
MEASURED = os.environ.get("KEVIN_SAFETY_MEASURED", "0") == "1"


def _edges(poly):
    """Convex polygon (CCW/CW) -> outward unit normals n and offsets c, inside iff max(n.p - c) <= 0."""
    P = poly; Q = np.roll(P, -1, 0); e = Q - P
    n = np.stack([e[:, 1], -e[:, 0]], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
    if (n @ P.mean(0) - (n * P).sum(1)).max() > 0: n = -n            # make outward
    return n, (n * P).sum(1)


STALE_S = 0.5 if os.environ.get("KEVIN_SAFETY_STALE_NOW", "0") == "1" else 0.25   # PR #62: 0.25 s; the staleness-now variant: 0.5 s


class FrameClock:
    """Frame capture stamp -> host time.monotonic() domain.  The sim stamps (hal/sim/pyrealsense2) are already
    monotonic; a real RealSense get_timestamp() may be in another domain (hardware / global = system clock).  If
    0 <= t_recv - stamp < SAME_DOMAIN_S the stamp is taken as monotonic (exact age).  Otherwise the capture time is
    stamp + offset, offset = smallest (t_recv - stamp) seen in the last WINDOW_S (transport latency floor; re-based if
    the stamp clock jumps by > 1 s).  Either way capture <= t_recv, so the age is never below the host-observed age."""
    SAME_DOMAIN_S = 2.0; WINDOW_S = 10.0
    def __init__(self): self.off = None; self.t_off = 0.0; self.domain = None
    def capture(self, stamp, t_recv):
        if stamp is None: return None
        d = t_recv - stamp
        if 0.0 <= d < self.SAME_DOMAIN_S: self.domain = "monotonic"; return stamp
        self.domain = "mapped"
        if self.off is None or d < self.off or abs(d - self.off) > 1.0 or t_recv - self.t_off > self.WINDOW_S:
            self.off, self.t_off = d, t_recv                     # new latency floor / clock jump / window expired
        return min(stamp + self.off, t_recv)


class Envelope:
    def __init__(self, margin=SAFETY_MARGIN, a_brake=1.5, t_react=0.10, stale_s=STALE_S, min_valid=0.05, reach=2.0):
        self.n, self.c = _edges(FOOTPRINT_SAFE); self.margin = margin; self.a = a_brake; self.tr = t_react
        self.stale = stale_s; self.min_valid = min_valid; self.reach = reach
        self.measured = MEASURED; self._t_prev = None; self.frame_dt = 1 / 30
        if self.measured: self.a = A_BRAKE_MEAS; self.t_act = T_ACT; self.tr = T_ACT; self.stale = 0.25
        self.interventions = 0; self.immobilised = 0; self.ticks = 0; self._was = False
        self.last = dict(state="ok", scale=1.0)

    def _sd(self, P):                     # signed distance (approx, exact on edge faces) of points to the footprint
        return (P @ self.n.T - self.c).max(1)

    def _sweep_min(self, P, v, w, dist):
        """min signed distance of P to footprint swept along arc (v,w) for arclength 'dist' (+ rotation if w)."""
        T = dist / max(abs(v), 1e-3) if abs(v) > 1e-3 else 0.0
        if abs(v) <= 1e-3:                # in-place rotation: sweep the rotation it would take to stop
            T = getattr(self, "_tr_now", self.tr) + abs(w) / 6.0
        k = max(2, int(math.ceil(max(dist, abs(w) * T * 0.35) / 0.03)) + 1)
        ts = np.linspace(0, T, k); m = np.inf; self._pen = 0.0
        for t in ts:
            th = w * t
            if abs(w) > 1e-6: x, y = v / w * math.sin(th), v / w * (1 - math.cos(th))
            else: x, y = v * t, 0.0
            c, s = math.cos(th), math.sin(th); d = P - (x, y)
            L = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
            sd = self._sd(L); m = min(m, float(sd.min()))
            self._pen = max(self._pen, float(np.maximum(self.margin - sd, 0).sum()))   # total penetration along sweep
        return m

    def filter(self, v, w, pts, depth_t, valid_frac, now=None, v_meas=0.0, w_meas=0.0):
        """pts: (N,2) robot-frame obstacle points (fresh RS1 ego + RS2); v_meas/w_meas: measured (odometry) twist.
        The sweep length uses the larger of commanded and measured speed in the commanded direction: the drive's
        velocity loop lags/overshoots (emulated ODrive: ~0.6 s to start, 20 % overshoot), so what matters is how fast
        the robot is actually going.  Returns (v, w, info)."""
        now = time.monotonic() if now is None else now; self.ticks += 1
        age = None if depth_t is None else max(now - depth_t, 0.0)          # frame age NOW (depth_t: capture, monotonic)
        if self.measured and depth_t is not None:                           # KEVIN_SAFETY_MEASURED: measured reaction for this tick
            if self._t_prev is not None and depth_t > self._t_prev: self.frame_dt = min(depth_t - self._t_prev, self.stale)
            self._t_prev = depth_t; self.tr = age + self.frame_dt + self.t_act
        if depth_t is None or age > self.stale or valid_frac < self.min_valid:
            self.immobilised += 1; self.last = dict(state="no_depth", scale=0.0, age=age)
            return 0.0, 0.0, self.last
        tr = self._tr_now = self.tr if self.measured else self.tr + age     # reaction: base + time already driven blind (measured: includes age)
        if pts is None or len(pts) == 0 or (abs(v) < 1e-3 and abs(w) < 1e-3):
            self._mark(False); self.last = dict(state="ok", scale=1.0, age=age); return v, w, self.last
        P = pts[np.hypot(pts[:, 0], pts[:, 1]) < self.reach]
        if len(P) > 1500: P = P[np.random.default_rng(0).choice(len(P), 1500, replace=False)]
        if len(P) == 0:
            self._mark(False); self.last = dict(state="ok", scale=1.0, age=age); return v, w, self.last
        sd0 = self._sd(P); now_sd = float(sd0.min()); pen0 = float(np.maximum(self.margin - sd0, 0).sum())
        if now_sd < 0.10:                 # tight: no fast spins (a 0.31 m-wide box sweeps its corners quickly)
            w = float(np.clip(w, -0.8, 0.8))
        for s in (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.0):
            vs, ws = v * s, w * s
            ve = max(abs(vs), abs(v_meas) if v_meas * v >= 0 else 0.0)
            dist = ve * tr + ve * ve / (2 * self.a)
            if s > 0: m = self._sweep_min(P, math.copysign(max(abs(vs), 1e-3), v) if abs(v) > 1e-3 else 0.0, ws, dist)
            else:      # commanding zero still coasts the measured speed for its stopping distance
                m = self._sweep_min(P, v_meas, 0.0, abs(v_meas) * tr + v_meas ** 2 / (2 * self.a)) if abs(v_meas) > 0.02 else now_sd
            # safe if the sweep keeps the margin, or (already inside it) the motion does not get any closer
            # safe if the sweep keeps the margin, or (already inside it) no part of the footprint penetrates MORE
            # anywhere along the sweep (total penetration; the old min-only test let rotations push corners in)
            if m >= self.margin or (now_sd < self.margin and s > 0 and m >= now_sd - 1e-3 and self._pen <= pen0 + 1e-4) or s == 0:
                hit = s < 1.0; self._mark(hit)
                self.last = dict(state="limited" if hit else "ok", scale=s, clearance=now_sd, age=age)
                return vs, ws, self.last
        self._mark(True); self.last = dict(state="blocked", scale=0.0, clearance=now_sd, age=age)
        return 0.0, 0.0, self.last

    def _mark(self, hit):
        if hit and not self._was: self.interventions += 1
        self._was = hit
