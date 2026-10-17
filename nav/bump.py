"""Contact ('bump') detector: motor current + IMU vs a model of normal driving.  ADDITIVE safety layer.

Why: the depth cameras do not see everything (the rear, low or thin things, glass, the dog).  When Kevin touches
something, the wheels have to push against it, so the motors draw current that free driving would not need, and
the body decelerates in a way the drive command did not ask for.  This module turns both into a calibrated
anomaly test.  It never relaxes anything else: it can only stop the robot.

Per wheel k (robot-forward sense), the motor current follows rigid-body drive physics:
    Kt*I_k = r*(m*a_k + b*v_k + c*sgn(v_k) + scrub*|w|*sgn(v_k)) + r*F_ext,k
so I_k = theta_k . phi_k + e_k with phi_k = [1, sgn(v_k), v_k, a_k, sgn(v_k)*|w|].  v_k and a_k are the measured wheel
speed and its derivative, and w is the measured yaw rate.  theta is identified online by recursive least squares
(forgetting 0.9995, i.e. ~1 min memory at 30 Hz) from normal driving, and the residual e_k ~ external force.  Its
noise sigma_k is a running RMS of in-model residuals.  Robust: samples with |z| > 4 do not update the model or
sigma, so a slow push cannot be learned away.

IMU: base-frame specific force (D435i accel through twin IMU_R).  The slow tilt/gravity part is removed with a 5 s
EWMA, and the expected acceleration is the measured wheel-mean derivative (forward) and v*w (lateral).  The residual
noise sigma_a is a running RMS, as above.

Detection (per 30 Hz tick).  All thresholds are in units of the learned noise, so nothing is a hand constant in
amperes or m/s^2:
  current:   z_k = e_k/sigma_k opposing the motion (motor pushing harder in its direction of travel) >= Z_I on
             N_I = 3 consecutive ticks (100 ms).  Z_I = 5: Gaussian per-tick tail 3e-7, cubed for 3 ticks.
  stall:     commanded wheel speed > 3 * encoder noise, measured < 30 % of it, for 0.3 s, with z_k >= 3.
  IMU:       the residual r_a = IMU specific force - rigid-body acceleration implied by the encoders (lever arm
             included), tick mean.  It does NOT trip on its own: the encoders decelerate with the body on a hit, so
             r_a measures IMU/encoder disagreement (timing, slip, chassis rock), and in sim + on carpet its lateral
             tail is heavy (caster scrub, thresholds).  Tried as a corroborated trigger (z_A>=8 with a 2-3 sigma
             current residual): still false positives and a wrong-sign direction on a frontal hit, so it is used
             only to set the side of the contact when it agrees with the travel direction (|z_A| >= 4 and its
             reaction opposes travel), and it is logged in every event for real-robot validation.
Until ARM_N moving samples have been seen, the model is not trusted: sigma is the prior (1 A, 1 m/s^2), so only
gross events trip.  The calibration persists in ~/.kevin/bump_cal_<hal>.json.

Direction: the IMU residual is the reaction push, and the contact is opposite it (robot frame).  Without a clear
IMU residual, the direction of travel is used, with the side taken from which wheel's current is anomalous.
"""
from __future__ import annotations
import json, math, os, time
import numpy as np

Z_I, N_I, Z_STALL, T_STALL = 5.0, 3, 3.0, 0.3
_lp = os.environ.get("KEVIN_BUMP_LOG"); LOG = open(_lp, "a", buffering=1) if _lp else None
ARM_N = 600                         # 20 s of moving samples
LAM = 0.9995
REFRACT_S = 1.5


class _RLS:
    def __init__(s, n, p0=100.0):
        s.th = np.zeros(n); s.P = np.eye(n) * p0
    def predict(s, phi): return float(s.th @ phi)
    def update(s, phi, y, lam=LAM):
        Pp = s.P @ phi; k = Pp / (lam + phi @ Pp)
        s.th = s.th + k * (y - s.th @ phi); s.P = (s.P - np.outer(k, Pp)) / lam
        if np.trace(s.P) > 1e6: s.P *= 1e6 / np.trace(s.P)          # bounded covariance (no blow-up when idle)


class BumpDetector:
    def __init__(self, track, cal_path=None, imu_xy=None):
        self.track = float(track)
        if imu_xy is None:
            try:
                import twin_geometry as TG; imu_xy = (float(TG.IMU_T[0]), float(TG.IMU_T[1]))
            except Exception: imu_xy = (0.0, 0.0)
        self.rx, self.ry = imu_xy                                     # IMU lever arm from the axle (rigid-body terms)
        self.prev_w = None
        self.rls = [_RLS(5), _RLS(5)]
        self.var = [1.0, 1.0]; self.var_a = [1.0, 1.0]; self.var_v = 1e-4
        self.n_moving = 0; self.prev = None; self.g_bias = None
        self.run_I = [0, 0]; self.stall_t = [None, None]; self.t_refr = -1e9
        self.events = []; self.last = None; self.stats = dict(ticks=0, events=0, max_zI=0.0, max_zA=0.0)
        self.cal_path = cal_path or os.path.expanduser("~/.kevin/bump_cal_%s.json" % os.environ.get("KEVIN_HAL", "real"))
        self._saved = time.monotonic(); self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self):
        try:
            d = json.load(open(self.cal_path))
            for k in range(2): self.rls[k].th = np.array(d["th"][k]); self.rls[k].P = np.eye(5) * 1.0
            self.var = list(d["var"]); self.var_a = list(d["var_a"]); self.var_v = d.get("var_v", self.var_v); self.n_moving = int(d["n"])
        except Exception: pass
    def save(self):
        try:
            os.makedirs(os.path.dirname(self.cal_path), exist_ok=True)
            json.dump(dict(th=[r.th.tolist() for r in self.rls], var=self.var, var_a=self.var_a, var_v=self.var_v, n=self.n_moving), open(self.cal_path, "w"))
        except Exception: pass

    @property
    def armed(self): return self.n_moving >= ARM_N

    def reset(self, t):
        """discontinuity (relocation in sim, wheels lifted...): restart differences, refractory from t."""
        self.prev = None; self.prev_w = None; self.t_refr = t; self.run_I = [0, 0]; self.stall_t = [None, None]

    def sig(self, k): return math.sqrt(self.var[k]) if self.armed else 1.0
    def sig_a(self, k): return math.sqrt(self.var_a[k]) if self.armed else 1.0

    # ---------------------------------------------------------------- per tick
    def update(self, t, cmd, wheel, gyro_z, currents, accel):
        """t s; cmd (v, w) last command sent; wheel (vl, vr) measured m/s; gyro_z rad/s (or None); currents
        (il, ir) A robot-forward or None; accel (N,3) base-frame specific force since last tick or None.
        Returns an event dict (and latches it in self.last) or None."""
        self.stats["ticks"] += 1
        if LOG:                                                         # KEVIN_BUMP_LOG=path: raw per-tick rows for offline model fits
            Am = np.asarray(accel, float)[:, :2].mean(0).tolist() if accel is not None and len(accel) else [None, None]
            LOG.write(json.dumps([round(t, 4), cmd[0], cmd[1], wheel[0], wheel[1], gyro_z, *(list(currents) if currents is not None else [None, None])[:2], *Am]) + "\n")
        vw = np.array(wheel, float); w = gyro_z if gyro_z is not None else (vw[1] - vw[0]) / self.track
        if self.prev is None: self.prev = (t, vw); return None
        dt = max(1e-3, t - self.prev[0]); aw = (vw - self.prev[1]) / dt; self.prev = (t, vw)
        al = 0.0 if self.prev_w is None else (w - self.prev_w) / dt; self.prev_w = w
        v, wc = cmd; u = np.array([v - wc * self.track / 2, v + wc * self.track / 2])
        moving = bool(np.abs(u).max() > 0.02 or np.abs(vw).max() > 0.02)
        refr = t - self.t_refr < REFRACT_S
        # ---- IMU residual (forward, lateral)
        za = np.zeros(2); ra = np.zeros(2)
        if accel is not None and len(accel):
            A = np.asarray(accel, float)[:, :2]
            if self.g_bias is None: self.g_bias = A.mean(0)
            # rigid body at the IMU lever arm r: a = a_axle + alpha x r - w^2 r  (planar)
            exp = np.array([aw.mean() - al * self.ry - w * w * self.rx, vw.mean() * w + al * self.rx - w * w * self.ry])
            R = A - self.g_bias - exp; ra = R.mean(0)                      # compared at the control rate (tick mean)
            za = np.array([ra[0] / self.sig_a(0), ra[1] / self.sig_a(1)])
            if not refr and np.abs(za).max() < 4:
                a_ = min(1.0, dt / 5.0); self.g_bias = (1 - a_) * self.g_bias + a_ * (A.mean(0) - exp)
                for i in range(2): self.var_a[i] += min(1.0, dt / 10.0) * (float(ra[i] ** 2) - self.var_a[i])
            self.stats["max_zA"] = max(self.stats["max_zA"], float(np.abs(za).max()))
        # ---- current residual per wheel (opposing motion => positive 'push')
        zI = np.zeros(2); push = np.zeros(2); eI = np.zeros(2)
        if currents is not None:
            for k in range(2):
                s = math.copysign(1.0, u[k] if abs(u[k]) > 0.01 else (vw[k] if abs(vw[k]) > 0.01 else 0.0)) if (abs(u[k]) > 0.01 or abs(vw[k]) > 0.01) else 0.0
                phi = np.array([1.0, s, vw[k], aw[k], s * abs(w)])
                e = currents[k] - self.rls[k].predict(phi); eI[k] = e; zI[k] = e / self.sig(k); push[k] = zI[k] * s
                if moving and not refr and abs(zI[k]) < 4:
                    self.rls[k].update(phi, currents[k])
                    self.var[k] += min(1.0, dt / 10.0) * (max(e * e, 0.05 ** 2) - self.var[k])     # 0.05 A: current-sense resolution floor
            self.stats["max_zI"] = max(self.stats["max_zI"], float(np.abs(zI).max()))
        if moving and not refr: self.n_moving += 1
        if np.abs(u).max() == 0.0 and not refr:                         # commanded still: encoder noise level
            self.var_v += min(1.0, dt / 10.0) * (float(np.mean(vw ** 2)) - self.var_v)
        if self.armed and time.monotonic() - self._saved > 30: self.save(); self._saved = time.monotonic()
        if refr: return None
        # ---- tests
        why = None
        for k in range(2):
            self.run_I[k] = self.run_I[k] + 1 if push[k] >= Z_I else 0
        if max(self.run_I) >= N_I: why = "motor current %+.1f A over model (%.0f sigma)" % (eI[int(np.argmax(push))], push.max())
        for k in range(2):
            stalled = abs(u[k]) > 3 * math.sqrt(self.var_v) + 0.03 and abs(vw[k]) < 0.3 * abs(u[k]) and push[k] >= Z_STALL
            if stalled:
                self.stall_t[k] = self.stall_t[k] or t
                if t - self.stall_t[k] >= T_STALL and why is None: why = "wheel stall (%s commanded %.2f m/s, measured %.2f)" % ("LR"[k], u[k], vw[k])
            else: self.stall_t[k] = None
        zam = float(np.abs(za).max())
        if why is None: return None
        # ---- direction (robot frame): travel direction, side from the one-sided wheel load or a consistent IMU
        fwd = v if abs(v) > 0.01 else float(vw.mean())
        ang = 0.0 if fwd >= 0 else math.pi
        sgn = 1.0 if fwd >= 0 else -1.0
        if zam >= 4 and ra[0] * sgn < 0:                               # reaction opposes travel: take its side
            ang = math.atan2(-ra[1], -ra[0])
        elif abs(push[0] - push[1]) > 1.5:                            # one-sided load: contact on that side
            side = 1.0 if push[0] > push[1] else -1.0                 # left wheel loaded -> contact on the left
            ang += side * (math.pi / 4 if fwd >= 0 else -math.pi / 4)
        ev = dict(t=t, why=why, angle=wrap(ang), dir=direction_name(ang), zI=[round(float(z), 1) for z in zI], zA=round(zam, 1),
                  v_cmd=round(float(v), 3), w_cmd=round(float(wc), 3), armed=self.armed)
        self.last = ev; self.events.append(ev); del self.events[:-20]; self.stats["events"] += 1
        self.t_refr = t; self.run_I = [0, 0]; self.stall_t = [None, None]
        return ev

    def state(self):
        return dict(armed=self.armed, n=self.n_moving, sigma_I=[round(math.sqrt(v), 3) for v in self.var], sigma_a=[round(math.sqrt(v), 3) for v in self.var_a],
                    last=self.last, stats=self.stats)


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


def direction_name(a):
    names = ["front", "front-left", "left", "rear-left", "rear", "rear-right", "right", "front-right"]
    return names[int(round(wrap(a) / (math.pi / 4))) % 8]


def contact_points(footprint, ang, half_width=math.radians(35), out=0.02, step=0.02):
    """Footprint boundary points whose bearing from the axle is within +-half_width of the contact angle, pushed
    'out' metres outward: where to mark the unseen obstacle in the map (robot frame)."""
    P = np.asarray(footprint, float); Q = np.roll(P, -1, 0); pts = []
    for a, b in zip(P, Q):
        n = max(2, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(0, 1, n, endpoint=False): pts.append(a + t * (b - a))
    pts = np.array(pts); bear = np.arctan2(pts[:, 1], pts[:, 0])
    sel = pts[np.abs(np.vectorize(wrap)(bear - ang)) <= half_width]
    if not len(sel): sel = pts[[int(np.argmin(np.abs(np.vectorize(wrap)(bear - ang))))]]
    r = np.linalg.norm(sel, axis=1, keepdims=True)
    return (sel * (1 + out / np.maximum(r, 1e-6))).astype(np.float32)
