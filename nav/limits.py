"""ONE acceleration-limit config for the whole stack (James 2026-10-20, t614).

  safety core   : brakes at, and judges commands against, the SAFETY limits (friction ellipse 0.2 g longitudinal at
                  the wheel rim / 0.1 g lateral): KEVIN_A_LONG_G, KEVIN_A_LAT_G
  planner side  : governor, every plan generator and the command output stage use the SAFETY limits divided by the
                  margin multiple KEVIN_PLAN_ACCEL_MULT (default 2 -> 0.1 g / 0.05 g; may later go toward 1.1)

The command output stage (nav.hw.HW.drive -> CmdLimiter) is the only path from any planner to the wheels, so no plan
can emit an out-of-range command: it rate-limits each wheel's rim speed to PLAN_A_LONG and caps |v w| at PLAN_A_LAT.
The safety core independently counts every brain command that exceeds the SAFETY limits as a planner bug
(STATE_FIELDS 'plan_bugs', shown in /api/state core.plan_bugs)."""
from __future__ import annotations
import math, os, time

G = 9.81
SAFETY_A_LONG = float(os.environ.get("KEVIN_A_LONG_G", "0.2")) * G       # m/s^2, wheel rim
SAFETY_A_LAT = float(os.environ.get("KEVIN_A_LAT_G", "0.1")) * G         # m/s^2, body lateral (v * w)
PLAN_MULT = max(1.0, float(os.environ.get("KEVIN_PLAN_ACCEL_MULT", "2.0")))
PLAN_A_LONG = SAFETY_A_LONG / PLAN_MULT
PLAN_A_LAT = SAFETY_A_LAT / PLAN_MULT
PLAN_LIMIT = os.environ.get("KEVIN_PLAN_LIMIT", "1" if os.environ.get("KEVIN_HAL") == "sim" else "0") == "1"   # output clamp: sim default (Orin untouched)
STALE_S = 0.3                       # older last command: the core has zeroed the wheels, start from rest


def wheels(v, w, track):
    return v - w * track / 2, v + w * track / 2


class CmdLimiter:
    """per-command clamp to the PLANNER limits: lateral |v w| <= a_lat (w reduced), then each wheel's rim speed may
    change by at most a_long * dt since the previous command (both wheels scaled together so the curvature is kept)."""
    def __init__(self, track, a_long=None, a_lat=None):
        self.track = track; self.a_long = PLAN_A_LONG if a_long is None else a_long; self.a_lat = PLAN_A_LAT if a_lat is None else a_lat
        self.v = self.w = 0.0; self.t = None; self.n_clamped = 0

    def __call__(self, v, w, t=None):
        t = time.monotonic() if t is None else t
        v, w = float(v), float(w); v0, w0 = v, w
        if abs(v * w) > self.a_lat and abs(v) > 1e-6: w = math.copysign(self.a_lat / abs(v), w)
        if self.t is None or t - self.t > STALE_S: pv = pw = 0.0; dt = 1.0 / 30
        else: pv, pw = self.v, self.w; dt = max(t - self.t, 1e-3)
        (l0, r0), (l1, r1) = wheels(pv, pw, self.track), wheels(v, w, self.track)
        dmax = self.a_long * dt; d = max(abs(l1 - l0), abs(r1 - r0))
        k = min(1.0, dmax / d) if d > 0 else 1.0                         # step from the previous command toward the target
        lat = lambda k_: abs((pv + (v - pv) * k_) * (pw + (w - pw) * k_)) <= self.a_lat * (1 + 1e-9)
        if not lat(k):            # |v w| is quadratic along the step: back off to the largest k that keeps it (k = 0 does)
            lo, hi = 0.0, k
            for _ in range(30):
                m = 0.5 * (lo + hi)
                if lat(m): lo = m
                else: hi = m
            k = lo
        v, w = pv + (v - pv) * k, pw + (w - pw) * k
        if abs(v - v0) > 1e-9 or abs(w - w0) > 1e-9: self.n_clamped += 1
        self.v, self.w, self.t = v, w, t
        return v, w


class LimitCheck:
    """safety core side: does the brain's command stream stay within the SAFETY limits?  (tolerance for timestamp noise)"""
    def __init__(self, track, a_long=SAFETY_A_LONG, a_lat=SAFETY_A_LAT, tol=0.05):
        self.track, self.a_long, self.a_lat, self.tol = track, a_long, a_lat, tol; self.prev = None; self.bugs = 0; self.last = None

    def __call__(self, v, w, t):
        """-> True when (v, w) at t is a planner bug (counted)."""
        if self.prev is not None and t == self.prev[2]: return False           # same command re-read
        bug = abs(v * w) > self.a_lat * (1 + self.tol) + 1e-6
        if self.prev is not None and 0 < t - self.prev[2] <= STALE_S:
            (l0, r0), (l1, r1) = wheels(self.prev[0], self.prev[1], self.track), wheels(v, w, self.track)
            dt = t - self.prev[2]; bug |= max(abs(l1 - l0), abs(r1 - r0)) > self.a_long * dt * (1 + self.tol) + 1e-3
        elif self.prev is None or t - self.prev[2] > STALE_S:
            l1, r1 = wheels(v, w, self.track); bug |= max(abs(l1), abs(r1)) > self.a_long / 30 * (1 + self.tol) + 1e-3   # from rest
        self.prev = (v, w, t)
        if bug: self.bugs += 1; self.last = (round(v, 3), round(w, 3), round(t, 3))
        return bug
