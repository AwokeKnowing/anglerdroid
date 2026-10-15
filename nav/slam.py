"""Lean 2D SLAM: wheel+gyro odometry (30 Hz) + log-odds occupancy map (2 cm) + correlative scan matching
against the map's likelihood field (async, ~5 Hz) -> map-frame pose.  No ROS, numpy + OpenCV only.

  pose_map = T_map_odom o pose_odom ;  the matcher only ever nudges T_map_odom (bounded, smoothed)."""
from __future__ import annotations
import math, threading, time
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


class Odom:
    """Dead reckoning: v from wheel encoders, yaw rate from the D435i gyro (wheel difference as fallback)."""
    def __init__(self, track=None):
        self.pose = (0.0, 0.0, 0.0); self.v = 0.0; self.w = 0.0; self.t = None
        import twin_geometry as TG; self.track = track or TG.TRACK_M
        self.hist = deque(maxlen=300); self.bias = 0.0; self._still = 0

    def update(self, t, vl, vr, gyro_w=None, commanded_zero=False):
        v = 0.5 * (vl + vr); wd = (vr - vl) / self.track
        if gyro_w is not None:
            # zero-velocity update: wheels stopped and nothing commanded -> learn gyro bias (EMA, ~2 s)
            # standstill = nothing commanded and wheels below 1 cm/s.  (Was 2 mm/s: under the realistic encoder model the
            # velocity jitter alone exceeds that, so the bias was never learned -> gyro heading drifted at the raw bias,
            # 0.17 deg/s in the hall A/B.)
            if commanded_zero and abs(vl) < 0.01 and abs(vr) < 0.01:
                self._still += 1
                if self._still > 15: self.bias += 0.05 * (gyro_w - self.bias); self.n_zupt = getattr(self, 'n_zupt', 0) + 1
            else: self._still = 0
            w = 0.0 if self._still > 15 else gyro_w - self.bias
        else: w = wd
        if self.t is not None:
            dt = min(max(t - self.t, 0.0), 0.6); x, y, th = self.pose; thm = th + 0.5 * w * dt   # gyro rate = mean over the gap (hw.yaw_rate)
            self.pose = (x + v * dt * math.cos(thm), y + v * dt * math.sin(thm), wrap(th + w * dt))
        self.t, self.v, self.w = t, v, w
        self.hist.append((t, self.pose))
        return self.pose

    def at(self, t):
        for tt, p in reversed(self.hist):
            if tt <= t: return p
        return self.hist[0][1] if self.hist else self.pose


class GridMap:
    """Log-odds occupancy, 2 cm (5 cm cells cap scan matching at ~4 cm; 2 cm gives ~2 cm, measured). origin = world coords of cell (0,0) corner."""
    def __init__(self, size_m=40.0, res=0.02, origin=(-20.0, -20.0)):
        self.res = res; self.n = int(size_m / res); self.origin = np.array(origin, np.float64)
        self.L = np.zeros((self.n, self.n), np.float32); self.seen = np.zeros((self.n, self.n), bool)
        self.lock = threading.Lock(); self.version = 0
        self.lf = np.zeros((self.n, self.n), np.float32)

    def rc(self, P):
        ij = np.floor((P - self.origin) / self.res).astype(np.int32)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] < self.n)
        return ij[ok, 1], ij[ok, 0]

    def integrate(self, pose, obs_r, free_r):
        with self.lock:
            if len(free_r):
                r, c = self.rc(transform(free_r, pose)); np.subtract.at(self.L, (r, c), 0.25); self.seen[r, c] = True
            if len(obs_r):
                r, c = self.rc(transform(obs_r, pose))
                k = np.unique(r.astype(np.int64) * self.n + c)          # one hit per cell per scan
                self.L.ravel()[k] += 0.9; self.seen.ravel()[k] = True
            np.clip(self.L, -4.0, 6.0, out=self.L); self.version += 1

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

    def submit(self, t, obs_r, free_r, match_r=None):
        """Latest scan (robot-frame obstacle + free points, stamp t; match_r = points used for scan matching,
        default obs_r). Dropped if the worker is busy."""
        with self._cv:
            self._job = (t, self.odom.at(t), obs_r, free_r, obs_r if match_r is None else match_r); self._cv.notify()
        if self.rec is not None: self.rec.append(self._job)

    def _loop(self):
        last_lf = 0.0; last_int = None; last_int_t = 0.0
        while self._run:
            with self._cv:
                while self._job is None: self._cv.wait()
                t, podom, obs, free, mpts = self._job; self._job = None
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
            self.grid.integrate(pmap, obs, free); last_int = podom; last_int_t = time.monotonic()
            if time.monotonic() - last_lf > 0.4:
                self.grid.refresh_likelihood(); last_lf = time.monotonic()
            self.stats["ms"] = (time.perf_counter() - t0) * 1e3
