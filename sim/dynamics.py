"""Higher-fidelity differential-drive dynamics for Kevin sim.

Closes sim↔reality gaps D1–D4 (ramp, latency, wheelbase, speed caps).
Safety clipping still happens in Robot.step *before* dynamics integrate.
"""

from __future__ import annotations

import math
from collections import deque

WHEELBASE_M = 0.34
WHEEL_RADIUS_M = 0.08565
V_MAX = 0.25
W_MAX = 0.80
V_MIN = -0.05
A_MAX = 1.61
ALPHA_MAX = 4.0
LATENCY_S = 0.15
CTRL_HZ = 30.0


class DiffDriveDynamics:
    """Command delay + accel-limited unicycle with wheelbase-aware ω."""

    def __init__(self, latency_s=LATENCY_S, dt_nominal=1.0 / CTRL_HZ):
        self.latency_s = float(latency_s)
        self.dt_nominal = float(dt_nominal)
        n = max(0, int(round(self.latency_s / max(1e-6, self.dt_nominal))))
        self.use_delay = n > 0
        self._delay = deque([(0.0, 0.0)] * n, maxlen=n) if self.use_delay else deque()
        self.v = 0.0
        self.w = 0.0

    def reset(self, v=0.0, w=0.0):
        self.v, self.w = float(v), float(w)
        if self.use_delay:
            n = self._delay.maxlen or 0
            self._delay = deque([(0.0, 0.0)] * n, maxlen=n)

    def clip_cmd(self, v_cmd, w_cmd):
        v = max(V_MIN, min(V_MAX, float(v_cmd)))
        w = max(-W_MAX, min(W_MAX, float(w_cmd)))
        half_L = WHEELBASE_M * 0.5
        lim = abs(v) + abs(w) * half_L
        if lim > V_MAX and lim > 1e-9:
            s = V_MAX / lim
            v *= s
            w *= s
        return v, w

    def apply(self, v_cmd, w_cmd, dt):
        v_c, w_c = self.clip_cmd(v_cmd, w_cmd)
        if self.use_delay:
            self._delay.append((v_c, w_c))
            v_c, w_c = self._delay.popleft()
        dv = v_c - self.v
        dw = w_c - self.w
        max_dv = A_MAX * dt
        max_dw = ALPHA_MAX * dt
        if abs(dv) > max_dv:
            dv = math.copysign(max_dv, dv)
        if abs(dw) > max_dw:
            dw = math.copysign(max_dw, dw)
        self.v += dv
        self.w += dw
        return self.v, self.w
