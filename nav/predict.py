"""Latency-compensated plan start state (James 2026-10-18).

A plan's command only acts after (planning time + command latency).  Planning from the last MEASURED state therefore
plans for a robot that has already moved on.  CmdPredictor forward-integrates the measured state (odometry pose +
wheel speeds at the frame stamp) through the commands already issued but not yet executed, up to the time the new
command will take effect, with the wheelbase model of the twin: each wheel's velocity setpoint follows the commanded
wheel speed (zero-order hold, delayed by the command latency) limited by the ODrive vel_ramp_rate (kevin_drive
ODriveAxis: vel_setpoint += clip(input_vel - vel_setpoint, +-ramp*dt)); the wheel speed follows the setpoint.

Nothing here is tuned:
  * command latency d = least-squares fit, online, of that same model to the measured wheel speeds over the last
    WIN_S seconds (grid over 0..D_MAX), smoothed with the median of the recent fits;
  * planning time = EWMA of the measured cycle compute times (cycle start -> command issued);
  * ramp = the safety core's current vel_ramp_rate (it reports it), else KEVIN_VEL_RAMP (twin default 3 turns/s^2).
eval(): every prediction is scored against the measured state at its effect time (position, yaw, v, w), together
with the error the UNcompensated start state (latest measurement) would have had -> before/after in one run."""
from __future__ import annotations
import collections, math, threading
import numpy as np
import twin_geometry as TG

TWO_PI_R = 2 * math.pi * TG.WHEEL_R_M
WIN_S, D_MAX, D_STEP, SIM_DT = 4.0, 0.30, 0.005, 0.005


def _wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


class CmdPredictor:
    def __init__(self, ramp_tps=3.0):
        self.cmds = collections.deque(maxlen=4000)       # (t_issue, v, w)
        self.meas = collections.deque(maxlen=600)        # (t, vl, vr, x, y, th, t_w): odom pose at frame stamp t, wheel speeds read at t_w
        self.ramp_tps = ramp_tps; self.lat = 0.08; self._fits = collections.deque(maxlen=9); self._last_fit = 0.0
        self.t_plan = 0.004; self.period = 0.033
        self.pending = collections.deque(maxlen=400)    # predictions awaiting their effect time
        self.err = collections.defaultdict(lambda: collections.deque(maxlen=3000))
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- inputs
    def issued(self, t, v, w):
        with self.lock: self.cmds.append((t, float(v), float(w)))

    def measured(self, t, vl, vr, pose, t_w=None):
        t_w = t if t_w is None else max(float(t_w), t)
        with self.lock:
            if self.meas and t_w <= self.meas[-1][6]: t_w = self.meas[-1][6] + 1e-4     # same wheel sample read twice
            self.meas.append((t, float(vl), float(vr), float(pose[0]), float(pose[1]), float(pose[2]), t_w))
        self._score(t)

    def cycle(self, t_compute, period):
        self.t_plan = 0.9 * self.t_plan + 0.1 * t_compute
        if period is not None: self.period = 0.9 * self.period + 0.1 * min(period, 0.5)

    # ---------------------------------------------------------------- model
    def _a_wheel(self, ramp_tps=None): return (ramp_tps or self.ramp_tps) * TWO_PI_R          # m/s^2 per wheel

    def _cmd_at(self, C, t):
        """zero-order hold of the issued command at time t (C sorted by time)."""
        i = np.searchsorted(C[:, 0], t, side="right") - 1
        return (0.0, 0.0) if i < 0 else (C[i, 1], C[i, 2])

    def _sim(self, C, t0, t1, sl, sr, d, a, pose=None, t_hold=None):
        """wheel setpoints sl, sr (m/s) from t0 to t1 under commands C delayed by d; integrates the pose if given.
        t_hold: the wheel speeds are a measurement taken at t_hold >= t0, so they are held (not ramped) until then."""
        x, y, th = pose if pose is not None else (0.0, 0.0, 0.0); half = TG.TRACK_M / 2
        t = t0; n = max(1, int(math.ceil((t1 - t0) / SIM_DT))); h = (t1 - t0) / n
        for _ in range(n):
            if t_hold is None or t >= t_hold:
                v_c, w_c = self._cmd_at(C, t - d)
                tl, tr = v_c - w_c * half, v_c + w_c * half; st = a * h
                sl += max(-st, min(st, tl - sl)); sr += max(-st, min(st, tr - sr))
            v, w = 0.5 * (sl + sr), (sr - sl) / TG.TRACK_M
            if pose is not None:
                tm = th + 0.5 * w * h; x += v * h * math.cos(tm); y += v * h * math.sin(tm); th += w * h
            t += h
        return sl, sr, (x, y, _wrap(th))

    def fit_latency(self):
        """least-squares command latency: the model's wheel speeds vs the measured ones over the last WIN_S seconds."""
        with self.lock:
            M = np.array(self.meas, float); C = np.array(self.cmds, float)
        if len(M) < 30 or len(C) < 10: return None
        M = M[M[:, 6] > M[-1, 6] - WIN_S][:, [6, 1, 2]]     # wheel speeds on their own read clock
        if len(M) < 30: return None
        Cw = C[C[:, 0] > M[0, 0] - D_MAX - 0.1]
        # excitation: the measured wheel speeds must change by much more than their noise, else every delay fits
        if len(Cw) < 5 or max(np.ptp(M[:, 1]), np.ptp(M[:, 2])) < 0.10: return None
        C = C[C[:, 0] > M[0, 0] - D_MAX - 1.0]; a = self._a_wheel()
        best = None; E = []
        for d in np.arange(0.0, D_MAX + 1e-9, D_STEP * 2):
            sl, sr = M[0, 1], M[0, 2]; e = 0.0
            for k in range(1, len(M)):
                sl, sr, _ = self._sim(C, M[k - 1, 0], M[k, 0], sl, sr, d, a)
                e += (sl - M[k, 1]) ** 2 + (sr - M[k, 2]) ** 2
            E.append(e)
            if best is None or e < best[0]: best = (e, d)
        E = np.array(E); self.fit_info = dict(d=round(best[1], 3), sse_min=round(float(E.min()), 4), sse_d0=round(float(E[0]), 4), sse_max=round(float(E.max()), 4), n=len(M))
        if E.max() <= 0 or (E.max() - E.min()) / E.max() < 0.2: return None      # flat: the window does not determine d
        self._fits.append(best[1]); self.lat = float(np.median(self._fits))
        return best[1]

    def predict(self, t_eff, ramp_tps=None):
        """state expected at t_eff: (x, y, th) odom frame, v, w, plus the start measurement used."""
        with self.lock:
            if not self.meas: return None
            m = self.meas[-1]; C = np.array(self.cmds, float) if self.cmds else np.zeros((0, 3))
        if not len(C): C = np.zeros((1, 3))
        sl, sr, pose = self._sim(C, m[0], max(t_eff, m[0]), m[1], m[2], self.lat, self._a_wheel(ramp_tps), pose=m[3:6], t_hold=m[6])
        return pose, 0.5 * (sl + sr), (sr - sl) / TG.TRACK_M, m

    # ---------------------------------------------------------------- scoring
    def remember(self, t_eff, pred, m):
        """pred = (pose, v, w) used as the plan's start state (or None = uncompensated); m = the measurement it came from."""
        with self.lock: self.pending.append((t_eff, pred, m))

    def _score(self, t_now):
        with self.lock:
            M = np.array(self.meas, float) if len(self.meas) > 2 else None
            todo = []
            while self.pending and self.pending[0][0] <= t_now: todo.append(self.pending.popleft())
        if M is None: return
        for t_eff, pred, m in todo:
            if t_eff < M[0, 0]: continue
            if t_eff > M[-1, 6]: continue
            i = int(np.searchsorted(M[:, 0], t_eff)); i = min(max(i, 1), len(M) - 1)
            f = (t_eff - M[i - 1, 0]) / max(M[i, 0] - M[i - 1, 0], 1e-6)
            act = M[i - 1] + f * (M[i] - M[i - 1]); act[5] = M[i - 1, 5] + f * _wrap(M[i, 5] - M[i - 1, 5])
            j = int(np.searchsorted(M[:, 6], t_eff)); j = min(max(j, 1), len(M) - 1)      # speeds on the wheel clock
            g = (t_eff - M[j - 1, 6]) / max(M[j, 6] - M[j - 1, 6], 1e-6)
            vl_a, vr_a = M[j - 1, 1] + g * (M[j, 1] - M[j - 1, 1]), M[j - 1, 2] + g * (M[j, 2] - M[j - 1, 2])
            va, wa = 0.5 * (vl_a + vr_a), (vr_a - vl_a) / TG.TRACK_M
            raw = (m[3:6], 0.5 * (m[1] + m[2]), (m[2] - m[1]) / TG.TRACK_M)
            for tag, p in (("pred", pred), ("raw", raw)):
                if p is None: continue
                (x, y, th), v, w = p
                self.err[tag + "_pos"].append(math.hypot(x - act[3], y - act[4])); self.err[tag + "_yaw"].append(abs(math.degrees(_wrap(th - act[5]))))
                self.err[tag + "_v"].append(abs(v - va)); self.err[tag + "_w"].append(abs(w - wa))
            self.err["sample_to_effect"].append(t_eff - m[0]); self.err["wheel_age_at_frame"].append(m[6] - m[0])

    def summary(self):
        out = dict(fit=getattr(self, "fit_info", None), n_fits=len(self._fits), lat_ms=round(self.lat * 1e3, 1), t_plan_ms=round(self.t_plan * 1e3, 2), period_ms=round(self.period * 1e3, 1))
        for k, q in list(self.err.items()):
            if q: a = np.array(q); out[k] = [round(float(np.mean(a)), 4), round(float(np.percentile(a, 95)), 4), len(a)]
        return out
