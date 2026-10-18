"""slam2 - lean 2D SLAM (new, 2026-10-13).  Same public API as nav.slam.Slam so run.py swaps it in (KEVIN_SLAM=2).

Why: the old 2D "morph matching" (src/gpu_render odom_gpu + src/slam.py thumbnails) nearly worked because it
matched DENSE, morphologically-closed obstacle images (holes/speckle filled -> wide, smooth score basin) instead of
sparse noisy points against a sharp likelihood field. It failed on frame-to-frame drift and had no real map.
slam2 keeps the dense/smooth idea, adds a map:

  scans  : ego obstacle points from BOTH depth cameras (RS1 top-down ego + RS2 forward), robot frame, 5 Hz
  odom   : wheel v + gyro yaw (nav.slam.Odom) = prediction
  front  : scan -> keyframe map. Keyframe hit-count grid (4 cm) -> morph close -> distance transform ->
           smooth field exp(-d^2/2s^2) at two scales (coarse s=12 cm search, fine s=4 cm refine).
           Brute-force GPU (torch) correlative search over (x, y, theta) around the odom prediction + weak prior.
  back   : keyframes every 20 cm / 8 deg; loop closure on revisits (old keyframe within 2 m, >4 m of path ago):
           wide search of the recent local cloud vs the old keyframes' submap; 2D pose graph (Gauss-Newton,
           scipy sparse) re-optimizes all keyframes; planning map rebuilt in place.
  output : map<-odom correction T (pose_map = T o pose_odom), planning GridMap (2 cm log-odds) for kplan/global.
Async: submit() never blocks the 30 Hz loop; a worker thread does matching/insert/closure."""
from __future__ import annotations
import json, math, os, threading, time
import numpy as np, cv2
from nav.slam import Odom, GridMap, wrap, compose, inverse, transform, carve_rays

try:
    import torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:                                   # pragma: no cover
    torch = None; DEV = None

RES = 0.04                      # matching grid (m/cell)
WIN = 10.0                      # half-size of the field window uploaded per match (m)
KF_D, KF_A = 0.20, math.radians(8)
MAXR = 6.0                      # scan points within 6 m (long context beats Z^2 noise; measured)
NPTS = 2000
MIN_CNT = 1                     # keyframe hits for a cell to count as occupied
CLOSE = True
PRIOR_W = 0.05
GAIN = 0.0                      # accept a match only if it beats the odom prediction by this much
SIG = (0.12, 0.04)
DEG_R = 0.06; DEG_MIN = 15.0; DEG_TH = 0.004   # curvature gates (score/m^2; score drop for +-2 deg)
LOOP_MIN = 0.65
MAX_W = 0.15                    # rad/s: no match/insert while turning faster (camera/odom time skew)
LOST_SC = 0.6
MATCH_TEST = os.environ.get("KEVIN_SLAM2_TEST", "chi2")   # chi2 (standard innovation test + Kalman gain) | gate (old)
CHI2_99_3 = 11.34
try: ODOM_ALPHA = json.load(open(os.path.join(os.path.dirname(__file__), "..", "src", "odom_noise.json")))["alpha"]
except Exception: ODOM_ALPHA = [0.07, 0.0, 0.00025, 0.0011]
K = 1.0                         # fraction of each scan correction applied
PRIOR = (0.15, math.radians(6))


def _voxel(P, v=0.04, n=NPTS, seed=0):
    if len(P) == 0: return P
    k = np.unique(np.floor(P / v).astype(np.int64), axis=0, return_index=True)[1]; P = P[k]
    if len(P) > n: P = P[np.random.default_rng(seed).choice(len(P), n, replace=False)]
    return P.astype(np.float32)


class KF:
    __slots__ = ("odom", "pose", "obs", "free", "path", "carve")
    def __init__(self, odom, pose, obs, free, path, carve=None):
        self.odom, self.pose, self.obs, self.free, self.path, self.carve = odom, pose, obs, free, path, carve


class Field:
    """Keyframe hit counts -> closed occupancy -> smooth distance fields (two scales)."""
    def __init__(self, size_m=40.0, origin=(-20.0, -20.0)):
        self.n = int(size_m / RES); self.origin = np.array(origin, float)
        self.cnt = np.zeros((self.n, self.n), np.uint16); self.dirty = True; self.f = {}

    def add(self, pose, obs, sign=1):
        P = transform(obs, pose); ij = np.floor((P - self.origin) / RES).astype(np.int64)
        ok = (ij >= 0).all(1) & (ij < self.n).all(1); k = np.unique(ij[ok, 1] * self.n + ij[ok, 0])
        c = self.cnt.ravel(); c[k] = np.clip(c[k].astype(np.int32) + sign, 0, 60000).astype(np.uint16); self.dirty = True

    def fields(self):
        if self.dirty:
            occ = (self.cnt >= MIN_CNT).astype(np.uint8)
            if CLOSE: occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))      # "morph": fill holes/gaps
            d = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3) * RES
            self.f = {s: np.exp(-(d * d) / (2 * s * s)).astype(np.float32) for s in SIG}; self.dirty = False
        return self.f


def _score(field_t, org, P_t, poses):
    """Mean field value of points P (N,2) under each pose (M,3) -> (M,).  Bilinear via grid_sample."""
    c, s = torch.cos(poses[:, 2:3]), torch.sin(poses[:, 2:3])
    X = poses[:, 0:1] + c * P_t[None, :, 0] - s * P_t[None, :, 1]
    Y = poses[:, 1:2] + s * P_t[None, :, 0] + c * P_t[None, :, 1]
    H, W = field_t.shape[-2:]
    gx = ((X - org[0]) / RES - 0.5) / (W - 1) * 2 - 1; gy = ((Y - org[1]) / RES - 0.5) / (H - 1) * 2 - 1   # cell centres
    g = torch.stack([gx, gy], -1)[None]                                   # (1,M,N,2)
    v = torch.nn.functional.grid_sample(field_t, g, mode="bilinear", padding_mode="zeros", align_corners=True)
    return v[0, 0].mean(1)


def search(fieldmaps, forg, P, guess, lin, lstep, ang, astep, prior=None, refine=True):
    prior = prior or PRIOR
    """Brute-force correlative search (coarse field), then fine refinement.  Returns (pose, score, score_at_guess)."""
    x0, y0, t0 = guess
    # crop a window around the guess
    i0 = int((x0 - WIN - forg[0]) / RES); j0 = int((y0 - WIN - forg[1]) / RES); m = int(2 * WIN / RES)
    n = fieldmaps[SIG[0]].shape[0]; i0 = min(max(i0, 0), n - m); j0 = min(max(j0, 0), n - m)
    org = torch.tensor([forg[0] + i0 * RES, forg[1] + j0 * RES], device=DEV, dtype=torch.float32)
    Pt = torch.from_numpy(P).to(DEV)
    out = []
    for s, L, LS, A, AS in ((SIG[0], lin, lstep, ang, astep), (SIG[1], lstep, lstep / 4, astep, astep / 4)):
        F = torch.from_numpy(np.ascontiguousarray(fieldmaps[s][j0:j0 + m, i0:i0 + m])).to(DEV)[None, None]
        dx = torch.arange(-L, L + 1e-9, LS, device=DEV); da = torch.arange(-A, A + 1e-9, AS, device=DEV)
        gx, gy, ga = torch.meshgrid(dx, dx, da, indexing="ij")
        cand = torch.stack([gx.ravel() + x0, gy.ravel() + y0, ga.ravel() + t0], 1).float()
        sc = _score(F, org, Pt, cand)
        dd = (cand[:, 0] - guess[0]) ** 2 + (cand[:, 1] - guess[1]) ** 2
        pen = PRIOR_W * (dd / prior[0] ** 2 + (cand[:, 2] - guess[2]) ** 2 / prior[1] ** 2)
        k = int(torch.argmax(sc - pen)); x0, y0, t0 = [float(v) for v in cand[k]]
        s_g = float(_score(F, org, Pt, torch.tensor([guess], device=DEV, dtype=torch.float32))[0])
        out.append((float(sc[k]), s_g))
        if not refine: break
    # degeneracy (corridors, single walls): curvature of the fine score around the peak; only move the pose along
    # directions the scan actually constrains, keep the odom prediction along flat ones.
    r = DEG_R; u = np.array([[math.cos(a), math.sin(a)] for a in np.arange(0, math.pi * 2, math.pi / 4)])
    probe = [[x0 + r * c, y0 + r * s_, t0] for c, s_ in u] + [[x0, y0, t0 + math.radians(2)], [x0, y0, t0 - math.radians(2)]]
    v = _score(F, org, Pt, torch.tensor(probe, device=DEV, dtype=torch.float32)).cpu().numpy(); best = out[-1][0]
    d = np.maximum(best - v[:8], 0) / (0.5 * r * r)
    A = np.stack([u[:, 0] ** 2, 2 * u[:, 0] * u[:, 1], u[:, 1] ** 2], 1); h = np.linalg.lstsq(A, d, rcond=None)[0]
    w, V = np.linalg.eigh(np.array([[h[0], h[1]], [h[1], h[2]]]))
    dxy = np.array([x0 - guess[0], y0 - guess[1]]); keep = V[:, w > DEG_MIN]
    dxy = keep @ (keep.T @ dxy)
    dth = wrap(t0 - guess[2]) if (best - v[8:].max()) > DEG_TH else 0.0
    search.last_eig = w.tolist()
    # Laplace approximation (Olson 2009): per-point log-likelihood curvature ~ score curvature / score, points treated
    # as independent -> information matrix of the match (xy from the fitted Hessian, theta from the +-2 deg probes)
    n = float(len(P)); bs = max(best, 1e-3)
    Hxy = np.array([[h[0], h[1]], [h[1], h[2]]]); hth = max(2 * best - v[8] - v[9], 0.0) / math.radians(2) ** 2
    Info = np.zeros((3, 3)); Info[:2, :2] = n * Hxy / bs; Info[2, 2] = n * hth / bs
    search.last_raw = (x0, y0, t0); search.last_info = Info
    return (guess[0] + dxy[0], guess[1] + dxy[1], wrap(guess[2] + dth)), out[-1][0], out[-1][1]


def optimize(poses, edges, fixed=0, iters=10):
    """2D pose graph, Gauss-Newton. edges: (i, j, z=(dx,dy,dth) of j in i, info diag (3,))."""
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    X = np.array(poses, float); n = len(X)
    for _ in range(iters):
        rows, cols, vals = [], [], []; b = np.zeros(3 * n); H_diag = {}
        Hd = {}
        def addH(a, bb, M):
            for r in range(3):
                for c in range(3):
                    rows.append(3 * a + r); cols.append(3 * bb + c); vals.append(M[r, c])
        for i, j, z, inf in edges:
            xi, xj = X[i], X[j]; c, s = math.cos(xi[2]), math.sin(xi[2])
            R_T = np.array([[c, s], [-s, c]]); dt = xj[:2] - xi[:2]
            e = np.r_[R_T @ dt - z[:2], wrap(xj[2] - xi[2] - z[2])]
            dR = np.array([[-s, c], [-c, -s]])
            A = np.zeros((3, 3)); A[:2, :2] = -R_T; A[:2, 2] = dR @ dt; A[2, 2] = -1
            B = np.zeros((3, 3)); B[:2, :2] = R_T; B[2, 2] = 1
            Om = np.diag(inf)
            addH(i, i, A.T @ Om @ A); addH(i, j, A.T @ Om @ B); addH(j, i, B.T @ Om @ A); addH(j, j, B.T @ Om @ B)
            b[3 * i:3 * i + 3] += A.T @ Om @ e; b[3 * j:3 * j + 3] += B.T @ Om @ e
        addH(fixed, fixed, np.eye(3) * 1e9)
        H = sp.csr_matrix((vals, (rows, cols)), shape=(3 * n, 3 * n))
        dx = sla.spsolve(H.tocsc(), -b); X += dx.reshape(n, 3); X[:, 2] = (X[:, 2] + np.pi) % (2 * np.pi) - np.pi
        if np.abs(dx).max() < 1e-4: break
    return [tuple(p) for p in X]


class Slam2:
    def __init__(self, grid: GridMap | None = None, match=True):
        self.grid = grid or GridMap(); self.odom = Odom(); self.T = (0.0, 0.0, 0.0)
        self.field = Field(); self.kfs: list[KF] = []; self.edges = []; self.do_match = match and torch is not None
        self.frozen = False; self.stats = dict(matches=0, accepted=0, ms=0.0, score=0.0, kfs=0, loops=0, loop_ms=0.0, opt_ms=0.0)
        self._job = None; self._cv = threading.Condition(); self._run = True; self._lock = threading.Lock()
        self.rec = None; self._path = 0.0; self._last_pm = None; self._last_loop_kf = -100
        if torch is not None:             # CUDA/kernel warm-up now, not on the first scan (it held the GIL ~0.6 s)
            f = {sg: np.zeros((int(2 * WIN / RES) + 2,) * 2, np.float32) for sg in SIG}
            search(f, np.array([-WIN, -WIN]), np.zeros((10, 2), np.float32), (0.0, 0.0, 0.0), 0.24, 0.04, 0.07, 0.017)
        threading.Thread(target=self._loop, daemon=True).start()

    # -- API shared with nav.slam.Slam
    def pose(self): return compose(self.T, self.odom.pose)
    def set_pose(self, x, y, th): self.T = compose((x, y, th), inverse(self.odom.pose))
    def submit(self, t, obs_r, free_r, match_r=None, obs_h=None, rays=None):
        """rays: current depth returns [(origin, E, floor)] for free-space carving (nav.slam.carve_rays)."""
        with self._cv: self._job = (t, self.odom.at(t), obs_r, free_r, obs_h, rays); self._cv.notify()
    def relocalize(self, P, step=0.10, n_ang=72, keep=40):
        """Global search over the whole field (used after map load)."""
        self.seed_from_grid()
        F = self.field.fields(); t0 = time.perf_counter(); P = _voxel(np.asarray(P, np.float32), n=800)
        occ = np.argwhere(self.field.cnt > 0)
        if len(occ) == 0: return (0, 0, 0), 0.0, 0.0
        lo = self.field.origin + occ.min(0)[::-1] * RES; hi = self.field.origin + occ.max(0)[::-1] * RES
        best = []
        for cx in np.arange(lo[0], hi[0], 1.0):
            for cy in np.arange(lo[1], hi[1], 1.0):
                for th in np.arange(-math.pi, math.pi, math.radians(15)):
                    p, s, _ = search(F, self.field.origin, P, (cx, cy, th), 0.5, 0.1, math.radians(7.5), math.radians(2.5), prior=(1e3, 1e3))
                    best.append((s, p))
        best.sort(key=lambda b: -b[0]); top = best[0]
        other = [b for b in best[1:] if math.hypot(b[1][0] - top[1][0], b[1][1] - top[1][1]) > 0.5]
        self.stats["reloc_ms"] = (time.perf_counter() - t0) * 1e3
        return top[1], top[0], top[0] - (other[0][0] if other else 0.0)

    def reset_to_grid(self):
        """Map loaded: forget this session's keyframes, match against the loaded map until relocalized."""
        with self._lock:
            self.kfs = []; self.edges = []; self.field = Field(); self._path = 0.0; self._last_loop_kf = -100
        self.seed_from_grid()

    def seed_from_grid(self):
        """After a map load (planning GridMap only), rebuild the matching field from its confident cells."""
        if self.field.cnt.any() or not (self.grid.L > 1.5).any(): return
        r, c = np.nonzero(self.grid.L > 1.5)
        P = np.stack([self.grid.origin[0] + (c + 0.5) * self.grid.res, self.grid.origin[1] + (r + 0.5) * self.grid.res], 1)
        self.field.add((0.0, 0.0, 0.0), P.astype(np.float32)); self.seeded = True

    # -- worker
    def _loop(self):
        while self._run:
            with self._cv:
                while self._job is None: self._cv.wait()
                t, podom, obs, free, obs_h, rays = self._job; self._job = None
            if self.frozen: continue
            try: self._step(t, podom, obs, free, obs_h, rays)
            except Exception as e: print("slam2 error: %r" % e)

    def _Pcov(self, podom):
        """Covariance of the odometry prediction (map frame), propagated step by step with the standard
        probabilistic-robotics odometry motion model: var_rot = a1 rot^2 + a2 trans^2, var_trans = a3 trans^2 + a4 rot^2.
        alphas are fitted once from recorded sim odometry vs ground truth (~/.kevin/emu/fit_odom.py -> src/odom_noise.json)."""
        P, q = getattr(self, "_Pc", None), getattr(self, "_Pc_odom", None)
        if P is None: P, q = np.diag([1e-6, 1e-6, 1e-8]), podom
        c, s_ = math.cos(q[2]), math.sin(q[2]); dx, dy = podom[0] - q[0], podom[1] - q[1]
        tr = math.hypot(dx, dy); rot = abs(wrap(podom[2] - q[2])); a1, a2, a3, a4 = ODOM_ALPHA
        vt, vr = a3 * tr * tr + a4 * rot * rot, a1 * rot * rot + a2 * tr * tr
        F = np.array([[1, 0, -dy], [0, 1, dx], [0, 0, 1.0]])
        G = np.array([[c, 0], [s_, 0], [0, 1.0]])
        P = F @ P @ F.T + G @ np.diag([vt, vr]) @ G.T
        self._Pc, self._Pc_odom = P, podom
        return P

    def _step(self, t, podom, obs, free, obs_h=None, rays=None):
        t0 = time.perf_counter(); self._carve = None
        pm = compose(self.T, podom)
        if MATCH_TEST == "chi2": self._Pcov(podom)        # propagate prediction covariance every odom step
        lp = getattr(self, "_lastjob", None); self._lastjob = (t, podom)
        if lp is not None and t > lp[0] and abs(wrap(podom[2] - lp[1][2])) / (t - lp[0]) > MAX_W:
            self.stats["skip_turn"] = self.stats.get("skip_turn", 0) + 1; return     # turning: scans skewed/smeared

        obs = np.asarray(obs, np.float32); P = _voxel(obs[np.hypot(obs[:, 0], obs[:, 1]) < MAXR]) if len(obs) else obs
        if self.do_match and (len(self.kfs) >= 3 or getattr(self, 'seeded', False)) and len(P) > 60:
            est, sc, s_g = search(self.field.fields(), self.field.origin, P, pm, 0.24, 0.04, math.radians(4), math.radians(1))
            raw, info = search.last_raw, search.last_info
            # odometry prior: how far may the scan pull us?  Wheel odom is good to ~3 % of distance + slip events,
            # so the allowed correction grows with odometry travelled since the last accepted match (and resets on it).
            la = getattr(self, "_last_acc", None) or podom
            dd = math.hypot(podom[0] - la[0], podom[1] - la[1]) + 0.3 * abs(wrap(podom[2] - la[2]))
            gate_xy, gate_th = 0.06 + 0.08 * dd, math.radians(2.5) + 0.05 * dd
            if sc < LOST_SC:              # odom jumped (wheel slip / contact): wider re-acquire around the prediction
                r = min(1.0, 0.25 + 0.15 * dd)                 # widen with uncertainty, not blindly to 1 m (maze aliasing)
                e2, s2, _ = search(self.field.fields(), self.field.origin, P, pm, r, 0.08, math.radians(12), math.radians(2), prior=(max(0.3, r), math.radians(10)))
                self.stats["wide"] = self.stats.get("wide", 0) + 1
                if s2 > sc + 0.1: est, sc = e2, s2; raw, info = search.last_raw, search.last_info
            self.stats["matches"] += 1; self.stats["score"] = round(sc, 3)
            if MATCH_TEST == "chi2":
                # standard EKF-style check: innovation (match - odom prediction) vs predicted covariance
                # S = P_odom + I_match^-1; reject outside the 99 % chi-square region (3 dof), else Kalman update.
                Po = self._Pcov(podom); d = np.array([raw[0] - pm[0], raw[1] - pm[1], wrap(raw[2] - pm[2])])
                R = np.linalg.pinv(info + 1e-9 * np.eye(3)) if np.linalg.cond(info + 1e-9 * np.eye(3)) < 1e12 else np.eye(3) * 1e6
                R = R + np.diag([1e-6, 1e-6, 1e-8])
                S = Po + R; m2 = float(d @ np.linalg.solve(S, d)); self.stats["chi2"] = round(m2, 2)
                if m2 < CHI2_99_3:
                    Kg = Po @ np.linalg.inv(S); u = Kg @ d
                    pm = (pm[0] + u[0], pm[1] + u[1], wrap(pm[2] + u[2])); self._Pc = (np.eye(3) - Kg) @ Po
                    self._last_acc = podom; self.T = compose(pm, inverse(podom)); self.stats["accepted"] += 1
                else:
                    self.stats["rej_chi2"] = self.stats.get("rej_chi2", 0) + 1
            else:
                dxy = math.hypot(est[0] - pm[0], est[1] - pm[1]); dth = abs(wrap(est[2] - pm[2]))
                if dxy > gate_xy or dth > gate_th:               # implausible vs odometry: aliasing (maze, door rows)
                    self.stats["gated"] = self.stats.get("gated", 0) + 1; sc = 0.0
                if sc > 0.35 and sc >= s_g + GAIN:
                    self._last_acc = podom
                    d = (est[0] - pm[0], est[1] - pm[1], wrap(est[2] - pm[2]))
                    pm = (pm[0] + K * d[0], pm[1] + K * d[1], wrap(pm[2] + K * d[2]))    # low-pass: odom between, scan pulls
                    self.T = compose(pm, inverse(podom)); self.stats["accepted"] += 1
        # planning map: every scan that moved (free space needs many rays; keyframes alone left it 'unknown')
        lg = getattr(self, "_last_int", None)
        if lg is None or math.hypot(podom[0] - lg[0], podom[1] - lg[1]) > 0.04 or abs(wrap(podom[2] - lg[2])) > math.radians(2.5):
            tc = time.perf_counter(); carve = carve_rays(rays, self.grid.res) if rays else None
            self.grid.integrate(pm, obs, free, obs_h, carve); self._last_int = podom
            self._carve = carve; self.stats["carve_ms"] = round((time.perf_counter() - tc) * 1e3, 1)
            self.stats["carved"] = self.grid.n_carved; self.stats["carved_tot"] = self.stats.get("carved_tot", 0) + self.grid.n_carved
        elif rays:      # standing still: no new evidence weight (a parked scan would saturate), but what the cameras see
            # through NOW is free -> carve (this frame's hits only shield their cells)
            tc = time.perf_counter(); self.grid.integrate(pm, obs, np.zeros((0, 2), np.float32), obs_h, carve_rays(rays, self.grid.res), w_hit=0.0)
            self.stats["carve_ms"] = round((time.perf_counter() - tc) * 1e3, 1)
            self.stats["carved"] = self.grid.n_carved; self.stats["carved_tot"] = self.stats.get("carved_tot", 0) + self.grid.n_carved
        last = self.kfs[-1] if self.kfs else None
        if last is None or math.hypot(podom[0] - last.odom[0], podom[1] - last.odom[1]) > KF_D or abs(wrap(podom[2] - last.odom[2])) > KF_A:
            self._insert(podom, pm, P, _voxel(np.asarray(free, np.float32), 0.06, 1500, 1) if len(free) else free,
                         getattr(self, "_carve", None))
        self.stats["ms"] = round((time.perf_counter() - t0) * 1e3, 2)

    def _insert(self, podom, pm, obs, free, carve=None):
        if self.kfs:
            a = self.kfs[-1]; self._path += math.hypot(podom[0] - a.odom[0], podom[1] - a.odom[1])
            z = compose(inverse(a.pose), pm); d = math.hypot(z[0], z[1])
            inf = np.array([1 / (0.01 + 0.03 * d) ** 2] * 2 + [1 / (0.005 + 0.02 * abs(z[2]) + 0.01 * d) ** 2])
            self.edges.append((len(self.kfs) - 1, len(self.kfs), z, inf))
        self.kfs.append(KF(podom, pm, obs, free, self._path, carve)); self.stats["kfs"] = len(self.kfs)
        self.field.add(pm, obs)
        if len(self.kfs) - self._last_loop_kf > 10: self._try_loop()

    def _try_loop(self):
        n = len(self.kfs) - 1; cur = self.kfs[n]
        cands = [i for i in range(n - 30) if cur.path - self.kfs[i].path > 4.0
                 and math.hypot(self.kfs[i].pose[0] - cur.pose[0], self.kfs[i].pose[1] - cur.pose[1]) < 2.0]
        if not cands: return
        t0 = time.perf_counter()
        j = min(cands, key=lambda i: math.hypot(self.kfs[i].pose[0] - cur.pose[0], self.kfs[i].pose[1] - cur.pose[1]))
        sub = Field()
        for k in range(max(0, j - 20), min(n - 30, j + 20)): sub.add(self.kfs[k].pose, self.kfs[k].obs)
        # recent local cloud expressed in the current keyframe frame
        Q = np.concatenate([transform(self.kfs[k].obs, compose(inverse(cur.pose), self.kfs[k].pose)) for k in range(max(0, n - 8), n + 1)])
        Q = _voxel(Q.astype(np.float32), 0.04, 1500)
        est, sc, s_g = search(sub.fields(), sub.origin, Q, cur.pose, 0.6, 0.06, math.radians(10), math.radians(1.5), prior=(1.0, math.radians(20)))
        self.stats["loop_ms"] = round((time.perf_counter() - t0) * 1e3, 1); self.stats["loop_score"] = round(sc, 3)
        if sc < LOOP_MIN: return
        corr = math.hypot(est[0] - cur.pose[0], est[1] - cur.pose[1])
        if corr > 0.15 + 0.03 * (cur.path - self.kfs[j].path): return     # implausible vs odom drift (~3 %)
        z = compose(inverse(self.kfs[j].pose), est)
        self.edges.append((j, n, z, np.array([1 / 0.03 ** 2, 1 / 0.03 ** 2, 1 / math.radians(1.0) ** 2])))
        self._last_loop_kf = n; self.stats["loops"] += 1
        t1 = time.perf_counter()
        new = optimize([k.pose for k in self.kfs], self.edges)
        self.stats["opt_ms"] = round((time.perf_counter() - t1) * 1e3, 1)
        moved = max(math.hypot(a[0] - k.pose[0], a[1] - k.pose[1]) for a, k in zip(new, self.kfs))
        self.stats["loop_shift_m"] = round(moved, 3)
        for p, k in zip(new, self.kfs): k.pose = p
        self.T = compose(self.kfs[-1].pose, inverse(self.kfs[-1].odom))
        self._rebuild()

    def _rebuild(self):
        f = Field(); g = GridMap(res=self.grid.res, size_m=self.grid.n * self.grid.res, origin=tuple(self.grid.origin))
        g.Hlo[:] = self.grid.Hlo; g.Hhi[:] = self.grid.Hhi       # KF scans are 2-D: keep the live height evidence (shift ~cm)
        for k in self.kfs:
            f.add(k.pose, k.obs)
            for _ in range(3): g.integrate(k.pose, k.obs, k.free, None, k.carve)   # KFs ~1/4 of scans: weight them up
        self.field = f
        with self.grid.lock:
            self.grid.L[:] = g.L; self.grid.seen[:] = g.seen; self.grid.Hlo[:] = g.Hlo; self.grid.Hhi[:] = g.Hhi; self.grid.version += 1
