"""Per-camera floor-tilt estimator (offline-evaluated; NOT wired into nav or the safety core).
Each frame: a FIXED set of pixels (chosen once from the nominal mount: rays that hit the floor in range) is projected with
the camera's static extrinsic; the floor-band points are fitted with a trimmed plane h = a x + b y + c (constant cost:
at most N_PIX points, 3 passes).  Two gates, both at the one chosen number ALPHA (nav.lslam.model):
  1. planarity: residual variance vs the camera's measured depth/floor noise sigma_n, chi2(n-3)  -> furniture/clutter
  2. innovation: fitted (pitch, roll) vs the filter prediction with its covariance, chi2(2)
A failed frame is dropped (the filter keeps predicting from the gyro = 'hold the last value' plus the measured rotation).
Filter per camera and axis, state [b, d]: b = static mount offset (constant: q_b = 0, i.e. a running weighted mean = the
long low-pass), d = dynamic tilt (body pitch + mast sway): d' = exp(-dt/tau) d + omega dt, an Ornstein-Uhlenbeck process
whose sigma_d / tau are MEASURED (train data), omega = gyro rate about that axis (gyro noise measured at rest).
Measurement z = b + d with R = the fit's own parameter covariance s^2 (A^T A)^-1."""
import math
import numpy as np
from scipy.stats import chi2
from nav.lslam.model import ALPHA

N_PIX = 600          # fixed per-frame point budget (cost bound, not a weight)


def fixed_pixels(K, R, T, axle_h, H, W, rmin, rmax, n=N_PIX, seed=0):
    """pixel indices whose NOMINAL ray meets the floor at range rmin..rmax (base frame), n of them, fixed for the run."""
    v, u = np.mgrid[0:H, 0:W]; d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3) @ R.T
    hz = T[2] + axle_h; s = -hz / np.where(d[:, 2] < -1e-6, d[:, 2], np.nan); x = T[0] + s * d[:, 0]; y = T[1] + s * d[:, 1]
    ok = np.flatnonzero(np.isfinite(s) & (np.hypot(x, y) > rmin) & (np.hypot(x, y) < rmax))
    return np.sort(np.random.default_rng(seed).choice(ok, min(n, len(ok)), replace=False))


def fit(P, axle_h, band=0.10):
    """P: base-frame points (n,3).  Returns (a, b, c, s2, cov2x2(a,b), n) of the trimmed plane on the floor band, or None."""
    h = P[:, 2] + axle_h; k = np.abs(h) < band
    if k.sum() < 30: return None                       # degeneracy guard for a 3-parameter fit
    A = np.c_[P[k, 0], P[k, 1], np.ones(k.sum())]; z = h[k]
    for _ in range(3):
        c, *_ = np.linalg.lstsq(A, z, rcond=None); e = A @ c - z; med = np.median(np.abs(e))
        keep = np.abs(e) < max(3 * 1.4826 * med, 1e-3)
        if keep.sum() < 30: return None
        A, z = A[keep], z[keep]
    c, *_ = np.linalg.lstsq(A, z, rcond=None); e = A @ c - z; n = len(z); s2 = float(e @ e / max(n - 3, 1))
    cov = s2 * np.linalg.inv(A.T @ A)[:2, :2]
    return float(c[0]), float(c[1]), float(c[2]), s2, cov, n, int(k.sum())


class TiltKF:
    """two axes (pitch, roll) x [b, d].  pitch = atan(a), roll = -atan(b) of the base-frame floor plane."""
    def __init__(self, sigma_n2, sig_d, tau, sg2, b0_var=math.radians(3) ** 2, alpha=ALPHA):
        self.sn2 = sigma_n2; self.sd2 = np.asarray(sig_d, float) ** 2; self.tau = np.asarray(tau, float); self.sg2 = np.asarray(sg2, float)
        self.x = np.zeros((2, 2)); self.P = np.array([np.diag([b0_var, sd]) for sd in self.sd2]); self.alpha = alpha
        self.g_inn = chi2.ppf(1 - alpha, 2); self.n_acc = 0; self.n_rej = [0, 0, 0]

    def predict(self, dt, w_pitch, w_roll):
        if dt <= 0: return
        for i, w in enumerate((w_pitch, w_roll)):
            a = math.exp(-dt / self.tau[i]); F = np.array([[1, 0], [0, a]])
            Q = np.diag([0.0, self.sd2[i] * (1 - a * a) + self.sg2[i] * dt * dt])
            self.x[i] = F @ self.x[i] + [0, w * dt]; self.P[i] = F @ self.P[i] @ F.T + Q

    def tilt(self): return self.x[:, 0] + self.x[:, 1]     # camera tilt (offset + dynamic): what levelling uses
    def tilt_var(self): return np.array([p[0, 0] + p[1, 1] + 2 * p[0, 1] for p in self.P])

    def update(self, f):
        """f = fit(...) or None.  Returns 'ok' | reason of rejection."""
        if f is None: self.n_rej[0] += 1; return "nofit"
        a, b, c, s2, cov, n, nb = f
        if (n - 3) * s2 / self.sn2 > chi2.ppf(1 - self.alpha, n - 3): self.n_rej[1] += 1; return "planarity"
        z = np.array([math.atan(a), -math.atan(b)]); J = np.diag([1 / (1 + a * a), -1 / (1 + b * b)]); Rz = J @ cov @ J.T
        H = np.array([1.0, 1.0]); y = z - self.tilt(); S = np.diag([H @ self.P[i] @ H for i in range(2)]) + Rz
        if y @ np.linalg.solve(S, y) > self.g_inn: self.n_rej[2] += 1; return "innovation"
        for i in range(2):                                   # per-axis scalar update (cross-axis fit covariance ignored)
            Si = H @ self.P[i] @ H + Rz[i, i]; Kk = self.P[i] @ H / Si
            self.x[i] = self.x[i] + Kk * y[i]; self.P[i] = (np.eye(2) - np.outer(Kk, H)) @ self.P[i]
        self.n_acc += 1; return "ok"
