"""Latency-compensated plan start state (James 2026-10-18).

A plan's command only acts after (planning time + command latency).  Planning from the last MEASURED state therefore
plans for a robot that has already moved on.  CmdPredictor forward-integrates the freshest measured state through
the commands the wheels have been / will be given, up to the time the new command takes effect, with the wheelbase
model of the twin: each wheel's velocity setpoint follows the commanded wheel speed (zero-order hold, delayed by the
transport latency) limited by the ODrive vel_ramp_rate (kevin_drive ODriveAxis: vel_setpoint += clip(input_vel -
vel_setpoint, +-ramp*dt)); the wheel speed follows the setpoint.

The command history is what the SAFETY CORE SENT to the wheels (v_out, w_out, ramp, t_sent from its state), not what
the brain asked for: the core governs/brakes (and goes inop on late frames) and owns the ramp rate, so brain commands
are not what the wheels execute.  (v1 integrated the brain commands: under load the core's limiting made that worse
than no prediction at all, and the latency fit ran into its bound trying to explain it.)

Effect time of a new command: the core forwards it at its first tick after the command is issued (ticks every core
period after its last t_sent), then the transport latency d.  Nothing is tuned:
  * d = least-squares fit, online, of the model to the measured wheel speeds (grid 0..D_MAX), median of recent fits;
  * planning time = EWMA of measured cycle compute time; core period = EWMA of the core's measured tick period;
  * ramp = per tick from the core's own report.
Every plan cycle is scored against the measured state at its effect time, together with the error the uncompensated
start state (latest measurement) would have had (paired, same cycles)."""
from __future__ import annotations
import collections, json, math, os, threading, time
import numpy as np
import twin_geometry as TG

TWO_PI_R = 2 * math.pi * TG.WHEEL_R_M
WIN_S, D_MAX, D_STEP, SIM_DT = 4.0, 0.30, 0.005, 0.005


def _wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


class CmdPredictor:
    def __init__(self, ramp_tps=3.0):
        self.out = collections.deque(maxlen=4000)        # (t_sent, v_out, w_out, ramp_tps): what the core sent the wheels
        self.cmds = collections.deque(maxlen=4000)       # (t_issue, v, w): brain commands (logged / fallback only)
        self.meas = collections.deque(maxlen=600)        # (t, vl, vr, x, y, th, t_w): odom pose at frame stamp t, wheel speeds read at t_w
        self.ramp_tps = ramp_tps; self.lat = 0.02; self._fits = collections.deque(maxlen=9)
        self.t_plan = 0.004; self.period = 0.033; self.core_period = 1 / 30
        self.pending = collections.deque(maxlen=400)
        self.err = collections.defaultdict(lambda: collections.deque(maxlen=3000))
        self.lock = threading.Lock()
        p = os.environ.get("KEVIN_PRED_LOG"); self._log = open(p, "a", buffering=1 << 16) if p else None

    def _w(self, *row):
        if self._log is not None: self._log.write(json.dumps([round(x, 5) if isinstance(x, float) else x for x in row]) + "\n")

    # ---------------------------------------------------------------- inputs
    def issued(self, t, v, w):
        with self.lock: self.cmds.append((t, float(v), float(w))); self._w("c", t, float(v), float(w))

    def core(self, t_sent, v_out, w_out, ramp=None, period=None):
        """one safety-core tick (its reported state); repeated ticks are ignored."""
        if t_sent is None or v_out is None or w_out is None: return
        with self.lock:
            if self.out and t_sent <= self.out[-1][0] + 1e-6: return
            self.out.append((float(t_sent), float(v_out), float(w_out), float(ramp or self.ramp_tps)))
            if period is not None and 0.005 < period < 0.5: self.core_period = 0.95 * self.core_period + 0.05 * period
            self._w("o", float(t_sent), float(v_out), float(w_out), float(ramp or self.ramp_tps))

    def measured(self, t, vl, vr, pose, t_w=None):
        t_w = t if t_w is None else max(float(t_w), t)
        with self.lock:
            if self.meas and t_w <= self.meas[-1][6]: t_w = self.meas[-1][6] + 1e-4     # same wheel sample read twice
            self.meas.append((t, float(vl), float(vr), float(pose[0]), float(pose[1]), float(pose[2]), t_w))
            self._w("m", t, float(vl), float(vr), float(pose[0]), float(pose[1]), float(pose[2]), t_w)
        self._score(t)

    def cycle(self, t_compute, period):
        self.t_plan = 0.9 * self.t_plan + 0.1 * t_compute
        if period is not None: self.period = 0.9 * self.period + 0.1 * min(period, 0.5)

    # ---------------------------------------------------------------- model
    def effect_time(self, t_now):
        """when a command issued after planning (t_now + t_plan) acts: the core's next tick, plus the transport latency."""
        with self.lock: ts = self.out[-1][0] if self.out else None
        t_iss = t_now + self.t_plan; P = self.core_period
        if ts is None: return t_iss + P + self.lat
        k = max(1, math.ceil((t_iss - ts) / P)); return ts + k * P + self.lat

    @staticmethod
    def _hold(O, t):
        i = np.searchsorted(O[:, 0], t, side="right") - 1
        return O[max(i, 0)] if i >= 0 else np.array([t, 0.0, 0.0, O[0, 3] if len(O) else 3.0])

    def _sim(self, O, t0, t1, sl, sr, d, pose=None):
        """wheel setpoints sl, sr (m/s) from t0 to t1 under core outputs O (t, v, w, ramp) delayed by d; integrates
        the pose if given."""
        x, y, th = pose if pose is not None else (0.0, 0.0, 0.0); half = TG.TRACK_M / 2
        t = t0; n = max(1, int(math.ceil((t1 - t0) / SIM_DT))) if t1 > t0 else 0; h = (t1 - t0) / max(n, 1)
        for _ in range(n):
            _, v_c, w_c, rp = self._hold(O, t - d)
            tl, tr = v_c - w_c * half, v_c + w_c * half; st = rp * TWO_PI_R * h
            sl += max(-st, min(st, tl - sl)); sr += max(-st, min(st, tr - sr))
            v, w = 0.5 * (sl + sr), (sr - sl) / TG.TRACK_M
            if pose is not None:
                tm = th + 0.5 * w * h; x += v * h * math.cos(tm); y += v * h * math.sin(tm); th += w * h
            t += h
        return sl, sr, (x, y, _wrap(th))

    def fit_latency(self):
        """least-squares transport latency: the model's wheel speeds (driven by the core's outputs) vs the measured
        ones over the last WIN_S s, vectorised over all candidate delays."""
        with self.lock:
            M = np.array(self.meas, float); O = np.array(self.out, float)
        if len(M) < 30 or len(O) < 10: return None
        M = M[M[:, 6] > M[-1, 6] - WIN_S][:, [6, 1, 2]]
        if len(M) < 30 or max(np.ptp(M[:, 1]), np.ptp(M[:, 2])) < 0.10: return None   # no excitation
        O = O[O[:, 0] > M[0, 0] - D_MAX - 1.0]
        if len(O) < 5: return None
        D = np.arange(0.0, D_MAX + 1e-9, D_STEP * 2); half = TG.TRACK_M / 2
        T = np.arange(M[0, 0], M[-1, 0] + SIM_DT, SIM_DT)
        idx = np.searchsorted(O[:, 0], T[None, :] - D[:, None], side="right") - 1
        ok = idx >= 0; j = np.maximum(idx, 0)
        Vc = np.where(ok, O[j, 1], 0.0); Wc = np.where(ok, O[j, 2], 0.0); ST = O[j, 3] * TWO_PI_R * SIM_DT
        TL, TR = Vc - Wc * half, Vc + Wc * half
        SL = np.empty_like(TL); SR = np.empty_like(TR); sl = np.full(len(D), M[0, 1]); sr = np.full(len(D), M[0, 2])
        for k in range(T.size):
            st = ST[:, k]; sl = sl + np.clip(TL[:, k] - sl, -st, st); sr = sr + np.clip(TR[:, k] - sr, -st, st); SL[:, k] = sl; SR[:, k] = sr
        kk = np.clip(np.round((M[1:, 0] - T[0]) / SIM_DT).astype(int), 0, T.size - 1)
        E = ((SL[:, kk] - M[None, 1:, 1]) ** 2 + (SR[:, kk] - M[None, 1:, 2]) ** 2).sum(1)
        ib = int(np.argmin(E))
        self.fit_info = dict(d=round(float(D[ib]), 3), rms=round(float(math.sqrt(E[ib] / (2 * len(kk)))), 4), rms_d0=round(float(math.sqrt(E[0] / (2 * len(kk)))), 4), n=len(M))
        if E.max() <= 0 or (E.max() - E.min()) / E.max() < 0.2: return None             # flat: the window does not determine d
        if ib == len(D) - 1: return None                                                # at the bound: not a minimum
        self._fits.append(float(D[ib])); self.lat = float(np.median(self._fits)); self._w("f", time.monotonic(), float(D[ib]), self.lat)
        return float(D[ib])

    def predict(self, t_eff, fresh=None):
        """state expected at t_eff: ((x, y, th) odom frame, v, w, m).  m = latest frame measurement (pose at m[0]);
        fresh = (t_w, vl, vr), a newer wheel-speed reading: the pose is carried from m[0] to t_w on the measured speeds,
        then the model runs from the measured speeds at t_w."""
        with self.lock:
            if not self.meas: return None
            m = self.meas[-1]; O = np.array(self.out, float) if self.out else np.zeros((0, 4))
        t_w, vl, vr = (m[6], m[1], m[2]) if fresh is None or fresh[0] <= m[6] else fresh
        x, y, th = m[3:6]; h = max(t_w - m[0], 0.0); v, w = 0.5 * (vl + vr), (vr - vl) / TG.TRACK_M
        tm = th + 0.5 * w * h; x += v * h * math.cos(tm); y += v * h * math.sin(tm); th = _wrap(th + w * h)
        if not len(O): O = np.array([[0.0, v, w, self.ramp_tps]])
        sl, sr, pose = self._sim(O, t_w, max(t_eff, t_w), vl, vr, self.lat, pose=(x, y, th))
        r = pose, 0.5 * (sl + sr), (sr - sl) / TG.TRACK_M, m
        self._w("p", time.monotonic(), t_eff, m[0], t_w, pose[0], pose[1], pose[2], r[1], r[2])
        return r

    # ---------------------------------------------------------------- scoring
    def remember(self, t_eff, pred, m):
        with self.lock: self.pending.append((t_eff, pred, m))

    def _score(self, t_now):
        with self.lock:
            M = np.array(self.meas, float) if len(self.meas) > 2 else None
            todo = []
            while self.pending and self.pending[0][0] <= t_now: todo.append(self.pending.popleft())
        if M is None: return
        for t_eff, pred, m in todo:
            if t_eff < M[0, 0] or t_eff > M[-1, 6]: continue
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
        out = dict(fit=getattr(self, "fit_info", None), n_fits=len(self._fits), lat_ms=round(self.lat * 1e3, 1), t_plan_ms=round(self.t_plan * 1e3, 2),
                   period_ms=round(self.period * 1e3, 1), core_period_ms=round(self.core_period * 1e3, 1))
        if self._log is not None: self._log.flush()
        for k, q in list(self.err.items()):
            if q: a = np.array(q); out[k] = [round(float(np.mean(a)), 4), round(float(np.percentile(a, 50)), 4), round(float(np.percentile(a, 95)), 4), len(a)]
        return out
