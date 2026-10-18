"""Lean 2D SLAM: wheel+gyro odometry (30 Hz) + log-odds occupancy map (2 cm) + correlative scan matching
against the map's likelihood field (async, ~5 Hz) -> map-frame pose.  No ROS, numpy + OpenCV only.

  pose_map = T_map_odom o pose_odom ;  the matcher only ever nudges T_map_odom (bounded, smoothed)."""
from __future__ import annotations
import math, os, threading, time
from collections import deque
import numpy as np
import cv2


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def transform(P, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.stack([pose[0] + c * P[:, 0] - s * P[:, 1], pose[1] + s * P[:, 0] + c * P[:, 1]], 1)


_WLOG = open(os.environ["KEVIN_WALL_LOG"], "a") if os.environ.get("KEVIN_WALL_LOG") else None


class Odom:
    """Dead reckoning: v from wheel encoders, yaw from the D435i gyro with online self-calibration (bench/slam/kslam):
    a 3-state EKF [yaw, gyro scale error, gyro bias].  ZUPT (wheels stopped, nothing commanded) observes the bias;
    wall directions in RS2 depth (Manhattan frame of the house, learned online, observe_walls) observe yaw, which makes
    the scale observable.  Noise values are kslam's (D435i gyro / wheel), not tuned here.  No gyro -> wheel difference.
    KEVIN_ODOM_EKF=0: the previous gyro-minus-EMA-bias integration (fixed scale)."""
    QG, QS, QB = 0.004 ** 2, 1e-6 ** 2, 2e-5 ** 2           # per second: yaw-rate noise, scale-error / bias random walk
    def __init__(self, track=None):
        self.pose = (0.0, 0.0, 0.0); self.v = 0.0; self.w = 0.0; self.t = None
        import twin_geometry as TG; self.track = track or TG.TRACK_M
        self.hist = deque(maxlen=300); self.bias = 0.0; self._still = 0
        self.ekf = os.environ.get("KEVIN_ODOM_EKF", "1") == "1"
        self.scale = 0.0; self.P = np.diag([1e-6, 0.02 ** 2, 0.003 ** 2]); self._gz = 0.0
        self.psi0 = None; self._psi_acc = []; self.n_wall = 0; self.n_wall_rej = 0; self.n_zupt = 0

    def _upd(self, H, r, R):
        S = float(H @ self.P @ H) + R; K = self.P @ H / S
        x, y, th = self.pose; d = K * r
        self.pose = (x, y, wrap(th + d[0])); self.scale += d[1]; self.bias += d[2]
        self.P = self.P - np.outer(K, H) @ self.P
        return d[0]

    def update(self, t, vl, vr, gyro_w=None, commanded_zero=False):
        v = 0.5 * (vl + vr); wd = (vr - vl) / self.track
        dt = 0.0 if self.t is None else min(max(t - self.t, 0.0), 0.6)
        if gyro_w is not None:
            # standstill = nothing commanded and wheels below 1 cm/s (realistic encoder jitter exceeds 2 mm/s)
            still = commanded_zero and abs(vl) < 0.01 and abs(vr) < 0.01
            self._still = self._still + 1 if still else 0
            if not self.ekf:
                if self._still > 15: self.bias += 0.05 * (gyro_w - self.bias); self.n_zupt += 1
                w = 0.0 if self._still > 15 else gyro_w - self.bias
            else:
                w = gyro_w * (1.0 + self.scale) - self.bias; self._gz = gyro_w
                if dt > 0:
                    F = np.array([[1.0, gyro_w * dt, -dt], [0, 1, 0], [0, 0, 1]])
                    self.P = F @ self.P @ F.T + np.diag([self.QG * dt, self.QS * dt, self.QB * dt])
                if self._still > 15:                         # ZUPT: true yaw rate 0 -> observes bias (and scale x rate)
                    self._upd(np.array([0.0, gyro_w, -1.0]), 0.0 - w, self.QG * 10); self.n_zupt += 1
                    w = 0.0
        else: w = wd
        if self.t is not None:
            x, y, th = self.pose; thm = th + 0.5 * w * dt   # gyro rate = mean over the gap (hw.yaw_rate)
            self.pose = (x + v * dt * math.cos(thm), y + v * dt * math.sin(thm), wrap(th + w * dt))
        self.t, self.v, self.w = t, v, w
        self.hist.append((t, self.pose))
        return self.pose

    def observe_walls(self, t_cap, z4, n):
        """Manhattan heading (kslam): z4 = mean exp(4i*a) of wall-normal angles a in the BASE frame of a depth frame
        captured at t_cap, from n wall pixels.  Learns the house's wall direction psi0 in the first frames, then
        corrects yaw (mod 90 deg) with a 3-sigma gate; skipped while turning fast (blur / latency)."""
        if not self.ekf or n < 150: return None
        Rl = abs(z4); th_c = self.at(t_cap)[2]; phi = (math.atan2(z4.imag, z4.real) / 4) + th_c   # world wall angle
        if _WLOG: _WLOG.write("%.4f %.4f %.5f %.5f %.3f %d %.4f %s\n" % (t_cap, self.t or 0, th_c, phi, Rl, n, self.w, self.psi0)); _WLOG.flush()
        if abs(self.w) > 0.8: self.n_wall_rej += 1; return None   # turning: wall directions unreliable (measured live:
        # |w| < 0.8 rad/s within 0.7 deg of GT p95, |w| > 1.2 rad/s ~39 deg off) -> neither learn psi0 nor correct
        if self.psi0 is None:
            # the house's wall direction in the map frame: learn it only while the yaw itself is still well known
            # (sigma < 2 deg) and barely turning, from 5 mutually consistent frames (|mean exp(4i phi)| > 0.9).
            # Learning it mid-tour (live 07:30: yaw sigma tens of degrees after LightNav's +-2.5 rad/s turns) gave a
            # wrong psi0 and the first correction then rotated the pose 34 deg.
            if Rl > 0.5 and n > 400 and abs(self.w) < 0.3 and self.P[0, 0] < math.radians(2) ** 2:
                self._psi_acc.append(np.exp(4j * phi)); del self._psi_acc[:-5]
                m = np.mean(self._psi_acc)
                if len(self._psi_acc) == 5 and abs(m) > 0.9: self.psi0 = float(np.angle(m) / 4)
            return None
        if Rl <= 0.45: self.n_wall_rej += 1; return None
        q = math.pi / 2; r = -(((phi - self.psi0) + q / 2) % q - q / 2)
        sig = math.radians(1.5) / math.sqrt(min(n, 4000) / 500.0) / Rl
        S = self.P[0, 0] + sig ** 2
        # a mod-90 deg measurement is only usable while the prior picks the branch unambiguously: the 5-sigma gate
        # (r^2/S < 25) must lie inside +-45 deg, i.e. sqrt(S) < 9 deg.  Beyond that, any wall frame "fits" some branch.
        if S > math.radians(9) ** 2 or r * r / S >= 25.0: self.n_wall_rej += 1; return None
        self.n_wall += 1; return self._upd(np.array([1.0, 0.0, 0.0]), r, sig ** 2)

    def at(self, t):
        for tt, p in reversed(self.hist):
            if tt <= t: return p
        return self.hist[0][1] if self.hist else self.pose


# Free-space carving (observed-free cells of the CURRENT depth frames clear remembered obstacles).
# CARVE_L: log-odds drop per frame that sees through a cell (> one hit, +0.9, so a contradicted cell clears in a few
# frames even from saturation: 6.0 -> < 0.7 in 4 frames, ~0.8 s at the 5 Hz map rate).
# CARVE_TOL: height slack = vertical spacing of neighbouring stride-8 RS2 rays at ~2 m (~4 cm).
# CARVE_BAND: the obstacle height band used for cells with no 3-D evidence (RS2 obstacle band 3..95 cm).
CARVE_L, CARVE_TOL, CARVE_BAND = 1.5, 0.04, (0.03, 0.95)


def carve_rays(rays, res=0.02, hmax=0.95, max_lh=3.0):
    """Free space seen by depth rays.  rays: [(origin(3), E(N,3), floor(N))] in the robot frame, z = height above the
    floor; E = measured returns (no-return pixels are unknown and carve nothing).  Each ray is sampled every <= res
    horizontally up to its return minus the depth noise margin (D435: 2 cells + 0.012 z^2; floor returns: 1 cell),
    keeping 0..hmax m and <= max_lh m horizontally from the lens (beyond, the noise margin is > 15 cm and the rays are
    > 6 cm apart: carving gets patchy, and the cost grows with the ray length).  Returns (cells(M,2) robot frame, fmin(M), fmax(M)): per res cell the height range crossed."""
    X, Y, H = [], [], []
    for o, E, fl in rays:
        E = np.asarray(E, np.float32)
        if not len(E): continue
        o = np.asarray(o, np.float32); d = E - o; rng = np.linalg.norm(d, axis=1); lh = np.hypot(d[:, 0], d[:, 1])
        marg = np.where(fl, res, 2 * res + 0.012 * rng * rng)
        te = np.clip(np.minimum((rng - marg) / np.maximum(rng, 1e-6), max_lh / np.maximum(lh, 1e-6)), 0.0, 1.0).astype(np.float32)
        ns = np.ceil(lh * te / res).astype(np.int32); sel = ns > 0
        d, te, ns = d[sel], te[sel], ns[sel]
        if not len(ns): continue
        rid = np.repeat(np.arange(len(ns), dtype=np.int32), ns)
        j = np.arange(1, len(rid) + 1, dtype=np.int32) - np.repeat(np.cumsum(ns, dtype=np.int32) - ns, ns)
        t = j.astype(np.float32) * np.repeat((te / ns).astype(np.float32), ns)               # samples 1..ns of each ray, last one at te
        h = o[2] + t * np.repeat(d[:, 2], ns); keep = (h >= 0.0) & (h <= hmax)
        t, h = t[keep], h[keep]; ks = np.flatnonzero(keep)
        X.append(o[0] + t * np.repeat(d[:, 0], ns)[ks]); Y.append(o[1] + t * np.repeat(d[:, 1], ns)[ks]); H.append(h)
    if not X: return np.zeros((0, 2), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    ix = np.floor(np.concatenate(X) * (1.0 / res)).astype(np.int32); iy = np.floor(np.concatenate(Y) * (1.0 / res)).astype(np.int32)
    H = np.concatenate(H).astype(np.float32, copy=False); x0, y0 = int(ix.min()), int(iy.min()); w = int(ix.max()) - x0 + 1; hgt = int(iy.max()) - y0 + 1
    k = ((iy - y0) * w + (ix - x0)).astype(np.intp)                       # dense local window (intp: ufunc.at fast path)
    fmin = np.full(w * hgt, np.inf, np.float32); fmax = np.full(w * hgt, -np.inf, np.float32)
    np.minimum.at(fmin, k, H); np.maximum.at(fmax, k, H)
    u = np.flatnonzero(fmin <= fmax)
    C = np.stack([u % w + x0, u // w + y0], 1)
    return ((C + 0.5) * res).astype(np.float32), fmin[u], fmax[u]


class GridMap:
    """Log-odds occupancy, 2 cm (5 cm cells cap scan matching at ~4 cm; 2 cm gives ~2 cm, measured). origin = world coords of cell (0,0) corner."""
    def __init__(self, size_m=40.0, res=0.02, origin=(-20.0, -20.0)):
        self.res = res; self.n = int(size_m / res); self.origin = np.array(origin, np.float64)
        self.L = np.zeros((self.n, self.n), np.float32); self.seen = np.zeros((self.n, self.n), bool)
        self.lock = threading.Lock(); self.version = 0
        self.lf = np.zeros((self.n, self.n), np.float32)
        # obstacle-evidence height band per cell (m above floor; inf/-inf = no 3-D evidence, e.g. loaded map)
        self.Hlo = np.full((self.n, self.n), np.inf, np.float16); self.Hhi = np.full((self.n, self.n), -np.inf, np.float16)
        self.n_carved = 0

    def rc(self, P):
        ij = np.floor((P - self.origin) / self.res).astype(np.int32)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] < self.n)
        return ij[ok, 1], ij[ok, 0]

    def integrate(self, pose, obs_r, free_r, obs_h=None, carve=None, w_hit=0.9):
        """obs_r/free_r: robot-frame points (obs_h: their heights above floor, NaN = unknown).
        carve = carve_rays(...) of THIS frame: cells crossed by depth rays with no return there.  A cell is carved only
        where this frame's rays pass through the whole height band in which the cell's obstacle evidence was seen
        (a removed box clears; a wall is never crossed; a table top clears only if rays pass at table height);
        cells hit in this frame and cells out of view are untouched.  w_hit=0: carve only (stationary frames)."""
        with self.lock:
            Lr, Lo, Hi = self.L.ravel(), self.Hlo.ravel(), self.Hhi.ravel()
            if len(free_r):
                r, c = self.rc(transform(free_r, pose)); np.subtract.at(self.L, (r, c), 0.25); self.seen[r, c] = True
            hit = np.zeros(0, np.int64)
            if len(obs_r):
                k, ok = self._keys(transform(obs_r, pose))
                hit = np.unique(k)                                   # one hit per cell per scan
                if w_hit: Lr[hit] += w_hit; self.seen.ravel()[hit] = True
                if obs_h is not None and w_hit:
                    h = np.asarray(obs_h, np.float32)[ok]; m = np.isfinite(h)
                    np.minimum.at(Lo, k[m], h[m].astype(np.float16)); np.maximum.at(Hi, k[m], h[m].astype(np.float16))
            if carve is not None and len(carve[0]):
                k, ok = self._keys(transform(carve[0], pose)); fmin, fmax = carve[1][ok], carve[2][ok]
                o = np.argsort(k, kind="stable"); k, fmin, fmax = k[o], fmin[o], fmax[o]
                uk, i0 = np.unique(k, return_index=True)
                if len(uk):
                    fmin, fmax = np.minimum.reduceat(fmin, i0), np.maximum.reduceat(fmax, i0)
                    lo, hi = Lo[uk].astype(np.float32), Hi[uk].astype(np.float32); none = lo > hi
                    lo[none], hi[none] = CARVE_BAND                  # no height evidence: needs the whole obstacle band
                    m = (fmin <= lo + CARVE_TOL) & (fmax >= hi - CARVE_TOL) & ~np.isin(uk, hit)
                    kk = uk[m]; Lr[kk] -= CARVE_L; self.seen.ravel()[kk] = True
                    self.n_carved = int(((Lr[kk] + CARVE_L) > 0.7).sum())     # cells that were occupied before this frame
            np.clip(self.L, -4.0, 6.0, out=self.L)
            if carve is not None and len(carve[0]) and len(uk):
                gone = kk[Lr[kk] <= 0.0]; Lo[gone] = np.inf; Hi[gone] = -np.inf   # evidence fully contradicted: forget its band
            self.version += 1

    def _keys(self, Pw):
        ij = np.floor((Pw - self.origin) / self.res).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] < self.n)
        return ij[ok, 1] * self.n + ij[ok, 0], ok

    def occupied(self, thr=0.7):
        return self.L > thr

    def refresh_likelihood(self, sigma=0.03):
        occ = self.occupied().astype(np.uint8)
        d = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 3) * self.res
        self.lf = np.exp(-(d * d) / (2 * sigma * sigma)).astype(np.float32)
        return int(occ.sum())


class Matcher:
    """Correlative scan matching (Olson 2009 style) on a bilinearly-interpolated likelihood field, coarse-to-fine,
    with a weak odometry prior (ties -> the odometry guess, never toward an arbitrary search corner)."""
    def __init__(self, grid: GridMap, lin=0.12, lin_step=0.03, ang=math.radians(4), ang_step=math.radians(1.0), prior=0.02):
        self.g = grid; self.lin, self.ls, self.ang, self.As, self.prior = lin, lin_step, ang, ang_step, prior

    def _score(self, P, guess, dxs, dys, dths):
        g = self.g; lf = g.lf; out = np.empty((len(dths), len(dxs), len(dys)), np.float32)
        for k, dth in enumerate(dths):
            th = guess[2] + dth; c, s = math.cos(th), math.sin(th)
            W = np.stack([c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1]], 1)
            base = (W + np.array(guess[:2]) - g.origin) / g.res - 0.5                 # continuous cell coords (centres)
            fx = base[None, None, :, 0] + (dxs / g.res)[:, None, None]; fy = base[None, None, :, 1] + (dys / g.res)[None, :, None]
            x0 = np.floor(fx); y0 = np.floor(fy); ax = (fx - x0).astype(np.float32); ay = (fy - y0).astype(np.float32)
            x0 = np.clip(x0.astype(np.int32), 0, g.n - 2); y0 = np.clip(y0.astype(np.int32), 0, g.n - 2)
            v = (lf[y0, x0] * (1 - ax) * (1 - ay) + lf[y0, x0 + 1] * ax * (1 - ay) + lf[y0 + 1, x0] * (1 - ax) * ay + lf[y0 + 1, x0 + 1] * ax * ay)
            out[k] = v.mean(-1)
        # odometry prior
        pen = self.prior * (np.hypot(dxs[:, None], dys[None, :])[None] / 0.1 + np.abs(dths)[:, None, None] / math.radians(4))
        return out, out - pen

    def match(self, P, guess):
        d = np.arange(-self.lin, self.lin + 1e-9, self.ls); a = np.arange(-self.ang, self.ang + 1e-9, self.As)
        raw, sc = self._score(P, guess, d, d, a)
        k, i, j = np.unravel_index(np.argmax(sc), sc.shape)
        s0 = float(raw[len(a) // 2, len(d) // 2, len(d) // 2]); med = float(np.median(raw))
        # refine: 1/3 steps around the coarse best
        c = (guess[0] + d[i], guess[1] + d[j], guess[2] + a[k])
        f = np.array([-1, 0, 1]) * self.ls / 3; fa = np.array([-1, 0, 1]) * self.As / 3
        raw2, _ = self._score(P, c, f, f, fa)
        k2, i2, j2 = np.unravel_index(np.argmax(raw2), raw2.shape)
        best = (c[0] + f[i2], c[1] + f[j2], wrap(c[2] + fa[k2])); bs = float(raw2[k2, i2, j2])
        return best, bs, med, s0


class Slam:
    """Owns odom->map correction. Call step() each control tick; the matcher+mapper run in their own thread."""
    def __init__(self, grid: GridMap | None = None, match=True):
        self.grid = grid or GridMap(); self.odom = Odom(); self.T = (0.0, 0.0, 0.0)
        self.matcher = Matcher(self.grid); self.do_match = match; self.frozen = False
        self._job = None; self._cv = threading.Condition(); self.stats = dict(matches=0, accepted=0, ms=0.0, score=0.0)
        self._run = True; threading.Thread(target=self._loop, daemon=True, name="slam").start()
        import os
        self.rec = [] if os.environ.get("KEVIN_SLAM_REC") else None
        if self.rec is not None:
            import atexit, pickle
            atexit.register(lambda: pickle.dump(self.rec, open(os.environ["KEVIN_SLAM_REC"], "wb")))

    def pose(self):
        return compose(self.T, self.odom.pose)

    def relocalize(self, P, step=0.10, n_ang=72, keep=40):
        """Global localization on a loaded map: score the scan P (robot frame) at every free 10 cm cell x 5 deg
        heading on the likelihood field, then refine the best `keep` candidates with the CSM matcher.
        Returns (pose, score, margin over the 2nd-best distinct pose)."""
        g = self.grid; t0 = time.perf_counter()
        if len(P) > 300: P = P[np.random.default_rng(0).choice(len(P), 300, replace=False)]
        f = int(round(step / g.res)); seen = g.seen[::f, ::f] & (g.L[::f, ::f] < 0)
        iy, ix = np.nonzero(seen)
        X = g.origin[0] + (ix * f + 0.5) * g.res; Y = g.origin[1] + (iy * f + 0.5) * g.res
        best = []
        for th in np.linspace(-math.pi, math.pi, n_ang, endpoint=False):
            c, s = math.cos(th), math.sin(th)
            Wx = c * P[:, 0] - s * P[:, 1]; Wy = s * P[:, 0] + c * P[:, 1]
            sc = np.zeros(len(X), np.float32)
            for k0 in range(0, len(X), 4000):
                cx = ((X[k0:k0 + 4000, None] + Wx[None] - g.origin[0]) / g.res).astype(np.int32).clip(0, g.n - 1)
                cy = ((Y[k0:k0 + 4000, None] + Wy[None] - g.origin[1]) / g.res).astype(np.int32).clip(0, g.n - 1)
                sc[k0:k0 + 4000] = g.lf[cy, cx].mean(1)
            j = np.argsort(sc)[-keep:]
            best += [(float(sc[i]), float(X[i]), float(Y[i]), float(th)) for i in j]
        best.sort(reverse=True); cands = []
        for sc0, x, y, th in best[:keep]:
            est, sc, _, _ = self.matcher.match(P, (x, y, th))
            est, sc, _, _ = self.matcher.match(P, est)
            cands.append((sc, est))
        cands.sort(key=lambda c: -c[0]); top = cands[0]
        other = [c for c in cands[1:] if math.hypot(c[1][0] - top[1][0], c[1][1] - top[1][1]) > 0.5 or abs(wrap(c[1][2] - top[1][2])) > 0.5]
        margin = top[0] - (other[0][0] if other else 0.0)
        self.stats["reloc_ms"] = (time.perf_counter() - t0) * 1e3
        return top[1], top[0], margin

    def set_pose(self, x, y, th):
        self.T = compose((x, y, th), inverse(self.odom.pose))

    def submit(self, t, obs_r, free_r, match_r=None, obs_h=None, rays=None):
        """Latest scan (robot-frame obstacle + free points, stamp t; match_r = points used for scan matching,
        default obs_r). Dropped if the worker is busy."""
        with self._cv:
            self._job = (t, self.odom.at(t), obs_r, free_r, obs_r if match_r is None else match_r, obs_h, rays); self._cv.notify()
        if self.rec is not None: self.rec.append(self._job)

    def _loop(self):
        last_lf = 0.0; last_int = None; last_int_t = 0.0
        while self._run:
            with self._cv:
                while self._job is None: self._cv.wait()
                t, podom, obs, free, mpts, obs_h, rays = self._job; self._job = None
            if self.frozen: continue                                        # loaded map, not yet relocalized
            t0 = time.perf_counter()
            pmap = compose(self.T, podom)
            moved = last_int is None or math.hypot(podom[0] - last_int[0], podom[1] - last_int[1]) > 0.04 \
                or abs(wrap(podom[2] - last_int[2])) > math.radians(2.5)
            if not moved and time.monotonic() - last_int_t < 3.0:
                continue                                                  # static: no re-integration (prevents self-reinforcing creep)
            mpts = mpts[np.hypot(mpts[:, 0], mpts[:, 1]) < 3.0]           # far D435 points are too noisy to match on
            if self.do_match and self.grid.version > 40 and len(mpts) > 40 and moved:   # 40 scans of odom-only warm-up
                P = mpts[np.random.default_rng().choice(len(mpts), 1000, replace=False)] if len(mpts) > 1000 else mpts
                est, sc, med, s0 = self.matcher.match(P, pmap); self.stats["matches"] += 1; self.stats["score"] = sc
                d = (est[0] - pmap[0], est[1] - pmap[1], wrap(est[2] - pmap[2]))
                small = math.hypot(d[0], d[1]) < 0.06 and abs(d[2]) < math.radians(1.5)
                # gated, partial correction: CSM only trims odometry (which is good to ~1 %); a wrong match can
                # never move the pose by more than a few mm per scan, so one bad map region cannot run away
                why = "low" if sc <= 0.5 else "flat" if sc <= med + 0.04 else "nogain" if sc <= (s0 or 0) + 0.02 else \
                    "big" if not small and sc <= (s0 or 0) + 0.05 else None
                if why: self.stats["rej_" + why] = self.stats.get("rej_" + why, 0) + 1
                if why is None:
                    # small: partial (k=0.3); large but clearly better (+0.05): clamped step (2 cm, 0.5 deg) so
                    # accumulated gyro-yaw drift during long maneuvers is pulled back gradually instead of locked out
                    k = 0.3; dx, dy, dth = k * d[0], k * d[1], k * d[2]
                    if not small:
                        n = math.hypot(dx, dy); f = min(1.0, 0.02 / n) if n > 1e-9 else 1.0
                        dx, dy = dx * f, dy * f; dth = max(-math.radians(0.5), min(math.radians(0.5), dth))
                        self.stats["clamped"] = self.stats.get("clamped", 0) + 1
                    pmap = (pmap[0] + dx, pmap[1] + dy, wrap(pmap[2] + dth))
                    self.T = compose(pmap, inverse(podom)); self.stats["accepted"] += 1
            self.grid.integrate(pmap, obs, free, obs_h, carve_rays(rays, self.grid.res) if rays else None); last_int = podom; last_int_t = time.monotonic()
            if time.monotonic() - last_lf > 0.4:
                self.grid.refresh_likelihood(); last_lf = time.monotonic()
            self.stats["ms"] = (time.perf_counter() - t0) * 1e3
