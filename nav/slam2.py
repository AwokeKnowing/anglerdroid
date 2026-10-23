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
STILL_SKIP = os.environ.get("KEVIN_SLAM2_STILL_SKIP", "1") == "1"
STILL_D, STILL_A = 0.04, math.radians(1.0)   # = the matcher's fine search step (search(..., 0.04 m, ..., 1 deg)): smaller motion is unresolvable
# (skips the scan MATCH only; KEVIN_SLAM2_STILL_SKIP=0 matches every job)
MATCH_TEST = os.environ.get("KEVIN_SLAM2_TEST", "chi2")   # chi2 (standard innovation test + Kalman gain) | gate (old)
CHI2_99_3 = 11.34
try: ODOM_ALPHA = json.load(open(os.path.join(os.path.dirname(__file__), "..", "src", "odom_noise.json")))["alpha"]
except Exception: ODOM_ALPHA = [0.07, 0.0, 0.00025, 0.0011]
K = 1.0                         # fraction of each scan correction applied
PRIOR = (0.15, math.radians(6))
# ---- lifelong mode (KEVIN_SLAM2_LIFE=1, 2026-10-22; design Page 'Kevin unified long-term SLAM', bench/slam/life24.py)
# The default mode grows without bound: a keyframe every 20 cm / 8 deg forever, a full GN over all of them and a full
# field + planning-map re-render on every closure.  Lifelong mode, all standard and with the literature's defaults:
#  * reduced pose graph (Johannsson et al. 2013): at a keyframe trigger, if a node already exists within NODE_D /
#    NODE_A of the current pose, no node is added -- the chain re-anchors to it with one odometry edge (one per node
#    pair) -> node count bounded by the AREA visited, not by time;
#  * node refresh (short/long-term memory, Biber & Duckett 2005, the simplest form): a re-anchored node close to the
#    current view takes the newest scan (old scan removed from the field), so moved furniture leaves the map;
#  * robust loop edges: Dynamic Covariance Scaling (Agarwal et al. 2013), Phi = 1 (the paper's default);
#  * warm-started vectorised GN (bounded by the node count), and on a closure only the nodes that moved are re-added
#    to the matching field (the planning map re-renders only if a node moved > REMAP_D);
#  * save()/load() of the graph (persistence across restarts).
LIFE = os.environ.get("KEVIN_SLAM2_LIFE", "0") == "1"
NODE_D, NODE_A = 0.5, math.radians(30)
DCS_PHI = 1.0
REFRESH_D, REFRESH_A, REFRESH_S = 0.25, math.radians(15), 60.0
REMAP_D = 0.05
# ---- asynchronous loop closure (KEVIN_SLAM2_ASYNC_LC=1, 2026-10-22; LIFE mode).  The loop-closure search + pose-graph
# solve + planning-map re-render ran inside the per-scan update (Orin: 115-170 ms on a loop-search update).
# Standard back-end/front-end split (e.g. Cartographer, ORB-SLAM's loop thread): at a node insert the main update only
# snapshots the graph and hands it to a worker thread; the front end keeps matching against the current field.  When the
# worker is done, a later update applies the result: the loop edge is added, solved nodes take their new poses, nodes added
# since the snapshot are carried by the rigid correction of the node they grew from (the submit-time anchor), the matching
# field is patched incrementally (moved nodes only) and re-rendered lazily, the pre-rendered planning map is swapped in.
# One job in flight; node inserts while it runs do not start another search.
# KEVIN_SLAM2_ASYNC_LAG=N > 0: deterministic variant for tests/benches -- apply exactly N updates after the submit (waits
# for the worker if needed), so results do not depend on thread timing.
# Default ON (bench/slam/orin_bench.py replays, 2026-10-22): LAG=1 reproduces the synchronous trajectory (max 1.9 cm /
# 0.7 deg for the one update in between, final 0; same nodes/loops); free-thread runs differ from sync by less than i777
# vs Orin sync does; Orin loop-search updates 135 -> 30 ms mean, updates > 100 ms 28 -> 0.  KEVIN_SLAM2_ASYNC_LC=0 = old path.
ASYNC_LC = os.environ.get("KEVIN_SLAM2_ASYNC_LC", "1") == "1"
ASYNC_LAG = int(os.environ.get("KEVIN_SLAM2_ASYNC_LAG", "0"))
_TL = threading.local()            # per-thread search() outputs (the loop worker searches concurrently with the front end)


def _voxel(P, v=0.04, n=NPTS, seed=0):
    if len(P) == 0: return P
    q = np.floor(P / v).astype(np.int64)
    # == np.unique(q, axis=0, return_index=True)[1] (rows in lexicographic order, first occurrence of each) via a single
    # mixed-radix int64 key, which preserves that order: identical indices, ~5-10x faster than the row-wise unique
    lo = q.min(0); span = q.max(0) - lo + 1
    if q.ndim == 2 and float(np.prod(span.astype(np.float64))) < 2.0 ** 62:
        key = q[:, 0] - lo[0]
        for c in range(1, q.shape[1]): key = key * span[c] + (q[:, c] - lo[c])
        k = np.unique(key, return_index=True)[1]
    else: k = np.unique(q, axis=0, return_index=True)[1]
    P = P[k]
    if len(P) > n: P = P[np.random.default_rng(seed).choice(len(P), n, replace=False)]
    return P.astype(np.float32)


class KF:
    __slots__ = ("odom", "pose", "obs", "free", "path", "carve", "t", "t_obs")
    def __init__(self, odom, pose, obs, free, path, carve=None, t=0.0):
        self.odom, self.pose, self.obs, self.free, self.path, self.carve = odom, pose, obs, free, path, carve
        self.t = self.t_obs = t


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
    search.last_raw = (x0, y0, t0); search.last_info = Info; _TL.raw, _TL.info = (x0, y0, t0), Info
    return (guess[0] + dxy[0], guess[1] + dxy[1], wrap(guess[2] + dth)), out[-1][0], out[-1][1]


def optimize(poses, edges, fixed=0, iters=10, robust=None, tol=1e-4):
    """2D pose graph, Gauss-Newton. edges: (i, j, z=(dx,dy,dth) of j in i, info diag (3,)).
    robust: bool per edge -> Dynamic Covariance Scaling (Agarwal 2013): weight s^2, s = min(1, 2 Phi / (Phi + chi2)).
    Vectorised (numpy + scipy sparse).  Returns poses (and sets optimize.last_scale = per-edge s)."""
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    X = np.array(poses, float); n = len(X)
    if not edges: return [tuple(p) for p in X]
    I = np.array([e[0] for e in edges]); J = np.array([e[1] for e in edges])
    Z = np.array([e[2] for e in edges], float); INF = np.array([e[3] for e in edges], float)
    RB = np.zeros(len(edges), bool) if robust is None else np.asarray(robust, bool)
    sc = np.ones(len(edges))
    for _ in range(iters):
        xi, xj = X[I], X[J]; c, s_ = np.cos(xi[:, 2]), np.sin(xi[:, 2]); dt = xj[:, :2] - xi[:, :2]
        e = np.stack([c * dt[:, 0] + s_ * dt[:, 1] - Z[:, 0], -s_ * dt[:, 0] + c * dt[:, 1] - Z[:, 1],
                      (xj[:, 2] - xi[:, 2] - Z[:, 2] + np.pi) % (2 * np.pi) - np.pi], 1)
        chi2 = (e * e * INF).sum(1)
        sc = np.where(RB, np.minimum(1.0, 2 * DCS_PHI / (DCS_PHI + chi2)), 1.0)
        W = INF * (sc * sc)[:, None]
        m = len(I); A = np.zeros((m, 3, 3)); B = np.zeros((m, 3, 3))
        A[:, 0, 0] = -c; A[:, 0, 1] = -s_; A[:, 0, 2] = -s_ * dt[:, 0] + c * dt[:, 1]
        A[:, 1, 0] = s_; A[:, 1, 1] = -c; A[:, 1, 2] = -c * dt[:, 0] - s_ * dt[:, 1]; A[:, 2, 2] = -1
        B[:, 0, 0] = c; B[:, 0, 1] = s_; B[:, 1, 0] = -s_; B[:, 1, 1] = c; B[:, 2, 2] = 1
        AtW = A.transpose(0, 2, 1) * W[:, None, :]; BtW = B.transpose(0, 2, 1) * W[:, None, :]
        blocks = [(I, I, AtW @ A), (I, J, AtW @ B), (J, I, BtW @ A), (J, J, BtW @ B)]
        rr, cc, vv = [], [], []
        o = np.arange(3)
        for a_, b_, M in blocks:
            rr.append((3 * a_[:, None, None] + o[None, :, None]).repeat(3, 2).ravel())
            cc.append((3 * b_[:, None, None] + o[None, None, :]).repeat(3, 1).ravel()); vv.append(M.ravel())
        rr.append(3 * fixed + o); cc.append(3 * fixed + o); vv.append(np.full(3, 1e9))
        H = sp.csc_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(3 * n, 3 * n))
        b = np.zeros(3 * n)
        np.add.at(b, (3 * I[:, None] + o).ravel(), np.einsum("mij,mj->mi", AtW, e).ravel())
        np.add.at(b, (3 * J[:, None] + o).ravel(), np.einsum("mij,mj->mi", BtW, e).ravel())
        dx = sla.spsolve(H, -b); X += dx.reshape(n, 3); X[:, 2] = (X[:, 2] + np.pi) % (2 * np.pi) - np.pi
        if np.abs(dx).max() < tol: break
    optimize.last_scale = sc
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
        try:                              # solver warm-up too (scipy.sparse import + first factorisation: ~0.3 s on the 1st closure)
            _i = np.full(3, 1e4)
            optimize([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [(0, 1, (1, 0, 0), _i), (1, 2, (1, 0, 0), _i), (0, 2, (2, 0, 0), _i)], robust=[False, False, True])
        except Exception as e: print("slam2 solver warm-up: %r" % e)
        self.async_lc, self.lc_lag = ASYNC_LC, ASYNC_LAG; self._lc = None; self._lc_gen = 0; self._nstep = 0; self.lc_log = []
        self._lc_stream = torch.cuda.Stream() if (torch is not None and DEV == "cuda") else None
        self._hp_stream = torch.cuda.Stream(priority=-3) if (self._lc_stream is not None and self.async_lc) else None   # max priority (CUDA clamps)
        # KEVIN_SLAM_REC=<file>: every job the worker PROCESSES (the exact replay input; submit() drops jobs while busy),
        # plus T before it, streamed as pickle records by a writer thread (nav exits via os._exit: no atexit dump)
        self._recq = None
        if os.environ.get("KEVIN_SLAM_REC"):
            import collections, pickle
            self._recq = collections.deque(); f = open(os.environ["KEVIN_SLAM_REC"], "ab")
            def _w():
                while True:
                    while self._recq: pickle.dump(self._recq.popleft(), f, protocol=4)
                    f.flush(); time.sleep(0.2)
            threading.Thread(target=_w, daemon=True, name="slamrec").start()
        threading.Thread(target=self._loop, daemon=True, name="slam2").start()

    # -- API shared with nav.slam.Slam
    def pose(self): return compose(self.T, self.odom.pose)
    def set_pose(self, x, y, th): self.T = compose((x, y, th), inverse(self.odom.pose))
    def submit(self, t, obs_r, free_r, match_r=None, obs_h=None, rays=None):
        """rays: current depth returns [(origin, E, floor)] for free-space carving (nav.slam.carve_rays)."""
        with self._cv:
            st = self.stats; st["sub"] = st.get("sub", 0) + 1
            if self._job is not None: st["drop"] = st.get("drop", 0) + 1    # worker still busy: latest job wins
            self._job = (t, self.odom.at(t), obs_r, free_r, obs_h, rays); self._cv.notify()
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
            self.kfs = []; self.edges = []; self.field = Field(); self._path = 0.0; self._last_loop_kf = -100; self._lc_gen += 1
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
            if self._recq is not None: self._recq.append((t, podom, obs, free, obs_h, rays, tuple(self.T)))
            self.stats["steps"] = self.stats.get("steps", 0) + 1
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

    def _step(self, *a, **kw):
        if self._hp_stream is None: return self._step_body(*a, **kw)
        # async loop closure: the per-update path runs on a high-priority CUDA stream so the worker's big loop-search
        # kernels (low-priority stream) do not halve its throughput (Orin, measured: match 54 -> 31 ms under a busy worker).
        # Event-ordered against the default stream both ways, so callers/other threads see the usual ordering.
        dflt = torch.cuda.default_stream(); self._hp_stream.wait_stream(dflt)
        try:
            with torch.cuda.stream(self._hp_stream): return self._step_body(*a, **kw)
        finally: dflt.wait_stream(self._hp_stream)

    def _step_body(self, t, podom, obs, free, obs_h=None, rays=None):
        t0 = time.perf_counter(); self._carve = None; self._nstep += 1
        if self._lc is not None: self._lc_poll()            # async loop closure finished: apply before predicting
        pm = compose(self.T, podom)
        if MATCH_TEST == "chi2": self._Pcov(podom)        # propagate prediction covariance every odom step
        lp = getattr(self, "_lastjob", None); self._lastjob = (t, podom)
        if lp is not None and t > lp[0] and abs(wrap(podom[2] - lp[1][2])) / (t - lp[0]) > MAX_W:
            self.stats["skip_turn"] = self.stats.get("skip_turn", 0) + 1; return     # turning: scans skewed/smeared
        # standing still: a scan from (within the matcher's resolution of) the last matched pose re-measures the same
        # scene through the same noise -- a match adds no independent information (repeated Kalman updates on correlated
        # scans would make the pose overconfident).  Gate the MATCH on the odometry delta since the last match.  The map
        # update below still runs: while parked it only clears what the cameras see through now (w_hit 0), which is how
        # a person/phantom that left the view is removed (L15 22:22: stuck 60 s with the whole step skipped).
        q = getattr(self, "_last_proc", None)
        still = STILL_SKIP and q is not None and math.hypot(podom[0] - q[0], podom[1] - q[1]) < STILL_D and abs(wrap(podom[2] - q[2])) < STILL_A
        if still: self.stats["skip_still"] = self.stats.get("skip_still", 0) + 1
        else: self._last_proc = podom

        obs = np.asarray(obs, np.float32); P = _voxel(obs[np.hypot(obs[:, 0], obs[:, 1]) < MAXR]) if len(obs) else obs
        if not still and self.do_match and (len(self.kfs) >= 3 or getattr(self, 'seeded', False)) and len(P) > 60:
            est, sc, s_g = search(self.field.fields(), self.field.origin, P, pm, 0.24, 0.04, math.radians(4), math.radians(1))
            raw, info = _TL.raw, _TL.info
            # odometry prior: how far may the scan pull us?  Wheel odom is good to ~3 % of distance + slip events,
            # so the allowed correction grows with odometry travelled since the last accepted match (and resets on it).
            la = getattr(self, "_last_acc", None) or podom
            dd = math.hypot(podom[0] - la[0], podom[1] - la[1]) + 0.3 * abs(wrap(podom[2] - la[2]))
            gate_xy, gate_th = 0.06 + 0.08 * dd, math.radians(2.5) + 0.05 * dd
            if sc < LOST_SC:              # odom jumped (wheel slip / contact): wider re-acquire around the prediction
                r = min(1.0, 0.25 + 0.15 * dd)                 # widen with uncertainty, not blindly to 1 m (maze aliasing)
                e2, s2, _ = search(self.field.fields(), self.field.origin, P, pm, r, 0.08, math.radians(12), math.radians(2), prior=(max(0.3, r), math.radians(10)))
                self.stats["wide"] = self.stats.get("wide", 0) + 1
                if s2 > sc + 0.1: est, sc = e2, s2; raw, info = _TL.raw, _TL.info
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
        if LIFE:
            tr = getattr(self, "_trig", None)
            if tr is None or math.hypot(podom[0] - tr[0], podom[1] - tr[1]) > KF_D or abs(wrap(podom[2] - tr[2])) > KF_A:
                self._trig = podom
                self._life_trigger(t, podom, pm, P, _voxel(np.asarray(free, np.float32), 0.06, 1500, 1) if len(free) else free,
                                   getattr(self, "_carve", None))
        else:
            last = self.kfs[-1] if self.kfs else None
            if last is None or math.hypot(podom[0] - last.odom[0], podom[1] - last.odom[1]) > KF_D or abs(wrap(podom[2] - last.odom[2])) > KF_A:
                self._insert(podom, pm, P, _voxel(np.asarray(free, np.float32), 0.06, 1500, 1) if len(free) else free,
                             getattr(self, "_carve", None), t=t)
        self.stats["ms"] = round((time.perf_counter() - t0) * 1e3, 2)

    def _insert(self, podom, pm, obs, free, carve=None, t=0.0):
        if self.kfs:
            a = self.kfs[-1]; self._path += math.hypot(podom[0] - a.odom[0], podom[1] - a.odom[1])
            z = compose(inverse(a.pose), pm); d = math.hypot(z[0], z[1])
            inf = np.array([1 / (0.01 + 0.03 * d) ** 2] * 2 + [1 / (0.005 + 0.02 * abs(z[2]) + 0.01 * d) ** 2])
            self.edges.append((len(self.kfs) - 1, len(self.kfs), z, inf))
        self.kfs.append(KF(podom, pm, obs, free, self._path, carve, t)); self.stats["kfs"] = len(self.kfs)
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

    # ------------------------------------------------------------------ lifelong mode (LIFE)
    def _odom_info(self, d, dth):
        return np.array([1 / (0.01 + 0.03 * d) ** 2] * 2 + [1 / (0.005 + 0.02 * abs(dth) + 0.01 * d) ** 2])

    def _life_trigger(self, t, podom, pm, P, free, carve):
        """Keyframe trigger (every KF_D / KF_A of odometry) in lifelong mode: re-anchor to a nearby node or add one."""
        self._recent = getattr(self, "_recent", None) or __import__("collections").deque(maxlen=8)
        self._recent.append((pm, P))
        if self.kfs:
            a, da, oa, pa = self._anchor
            dpath = math.hypot(podom[0] - oa[0], podom[1] - oa[1]); self._path += dpath
            # edge measurement = the scan-matched pose relative to the anchor node (as the default mode's chain edges);
            # its information comes from the odometry travelled since the anchor
            z = compose(inverse(self.kfs[a].pose), pm); self._anchor = (a, da, podom, self._path)
            self._anchor_d = getattr(self, "_anchor_d", 0.0) + dpath
            ac = getattr(self, "_anchor_acc", 0.0) + abs(wrap(podom[2] - oa[2])); self._anchor_acc = ac
            Xp = self._node_xy(); dd = np.hypot(Xp[:, 0] - pm[0], Xp[:, 1] - pm[1])
            ang = np.abs((self._node_th() - pm[2] + np.pi) % (2 * np.pi) - np.pi)
            ok = (dd < NODE_D) & (ang < NODE_A)
            if ok.any():
                j = int(np.argmin(np.where(ok, dd + 0.3 * ang, np.inf))); dj = compose(inverse(self.kfs[j].pose), pm)
                if j != a:
                    key = (min(a, j), max(a, j))
                    if key not in self._pairs:
                        zaj = compose(z, inverse(dj)); inf = self._odom_info(self._anchor_d, self._anchor_acc)
                        self._pairs.add(key); self.edges.append((a, j, zaj, inf)); self._robust.append(False)
                self._anchor = (j, dj, podom, self._path); self._anchor_d = 0.0; self._anchor_acc = 0.0
                self.stats["reanchor"] = self.stats.get("reanchor", 0) + 1
                k = self.kfs[j]
                if (math.hypot(dj[0], dj[1]) < REFRESH_D and abs(dj[2]) < REFRESH_A and t - k.t_obs > REFRESH_S
                        and len(P) > 60 and getattr(self, "_last_acc", None) is podom):
                    # node refresh: the newest view replaces the node's scan (only right after an accepted match)
                    self.field.add(k.pose, k.obs, -1); k.obs = transform(P, dj).astype(np.float32); k.t_obs = t
                    self.field.add(k.pose, k.obs); self.stats["refresh"] = self.stats.get("refresh", 0) + 1
                return
            inf = self._odom_info(self._anchor_d, self._anchor_acc)
            self.edges.append((a, len(self.kfs), z, inf)); self._robust.append(False); self._pairs.add((a, len(self.kfs)))
        else:
            self._pairs = set(); self._robust = []
        self.kfs.append(KF(podom, pm, P, free, self._path, carve, t)); n = len(self.kfs) - 1
        self._anchor = (n, (0.0, 0.0, 0.0), podom, self._path); self._anchor_d = 0.0; self._anchor_acc = 0.0
        self._xy = None; self.stats["kfs"] = len(self.kfs); self.field.add(pm, P)
        if n - self._last_loop_kf > 3: self._try_loop_life(n)

    def _node_xy(self):
        if getattr(self, "_xy", None) is None or len(self._xy) != len(self.kfs):
            self._xy = np.array([k.pose for k in self.kfs], float)
        return self._xy[:, :2]
    def _node_th(self): self._node_xy(); return self._xy[:, 2]

    def _check_consistency(self, i, j, z, inf):
        """A re-anchor edge between nodes far apart along the path is a loop: if it is inconsistent with the current
        graph (chi2 > 99 % 3-dof), re-optimise."""
        if abs(self.kfs[i].path - self.kfs[j].path) < 4.0: return
        e = np.array(compose(inverse(z), compose(inverse(self.kfs[i].pose), self.kfs[j].pose))); e[2] = wrap(e[2])
        if float((e * e * inf).sum()) > CHI2_99_3: self._optimize_life("reanchor")

    def _try_loop_life(self, n):
        if self._lc is not None: self.stats["lc_busy"] = self.stats.get("lc_busy", 0) + 1; return   # one search in flight
        snap = self._lc_snapshot(n)
        if snap is None: return
        if not self.async_lc: return self._lc_apply(self._lc_compute(snap))
        box = []
        def run():
            try:
                if self._lc_stream is not None:
                    with torch.cuda.stream(self._lc_stream): box.append(self._lc_compute(snap))
                else: box.append(self._lc_compute(snap))
            except Exception as e: print("slam2 loop worker error: %r" % e)
        th = threading.Thread(target=run, daemon=True, name="slam2-loop")
        self._lc = (th, box, self._nstep, self._lc_gen, time.perf_counter()); th.start()

    def _lc_poll(self):
        th, box, n0, gen, t0 = self._lc
        if self.lc_lag > 0:
            if self._nstep - n0 < self.lc_lag: return
            th.join()
        elif th.is_alive(): return
        self._lc = None
        if gen != self._lc_gen or not box: return          # graph reset / loaded meanwhile, or the worker failed
        r = box[0]; self.stats["lc_lag"] = self._nstep - n0
        self.lc_log.append((n0, self._nstep, r["ok"], round((time.perf_counter() - t0) * 1e3, 1)))
        self._lc_apply(r)

    def _lc_snapshot(self, n):
        """Loop candidate (main thread, cheap) + everything the search/solve needs, by reference (node scans are replaced,
        never mutated in place)."""
        cur = self.kfs[n]; Xp = self._node_xy(); paths = np.array([k.path for k in self.kfs])
        dd = np.hypot(Xp[:, 0] - cur.pose[0], Xp[:, 1] - cur.pose[1])
        c = np.nonzero((dd < 2.0) & (cur.path - paths > 4.0))[0]
        if len(c) == 0: return None
        j = int(c[np.argmin(dd[c])])
        near = np.nonzero((np.hypot(Xp[:, 0] - Xp[j, 0], Xp[:, 1] - Xp[j, 1]) < 2.0) & (cur.path - paths > 4.0))[0]
        return dict(n=n, j=j, cur=cur.pose, path_n=cur.path, path_j=self.kfs[j].path, pose_j=self.kfs[j].pose, t_n=cur.t, t_j=self.kfs[j].t,
                    sub=[(self.kfs[k].pose, self.kfs[k].obs) for k in near], recent=list(self._recent),
                    poses=[k.pose for k in self.kfs], edges=list(self.edges), robust=list(self._robust),
                    nodes=[(k.obs, k.free, k.carve) for k in self.kfs])

    def _lc_compute(self, s):
        """Loop search + pose-graph solve (+ planning-map re-render if the map moved) on a snapshot.  Pure: touches no
        live state, so it runs on the worker thread (or inline when synchronous)."""
        t0 = time.perf_counter(); cur = s["cur"]; sub = Field()
        for p, o in s["sub"]: sub.add(p, o)
        Q = np.concatenate([transform(Pk, compose(inverse(cur), pk)) for pk, Pk in s["recent"] if len(Pk)])
        Q = _voxel(Q.astype(np.float32), 0.04, 1500)
        est, sc, s_g = search(sub.fields(), sub.origin, Q, cur, 0.6, 0.06, math.radians(10), math.radians(1.5), prior=(1.0, math.radians(20)))
        r = dict(snap=s, ok=False, loop_ms=round((time.perf_counter() - t0) * 1e3, 1), score=round(sc, 3))
        if sc < LOOP_MIN: return r
        if math.hypot(est[0] - cur[0], est[1] - cur[1]) > 0.15 + 0.03 * (s["path_n"] - s["path_j"]): return r
        e = (s["j"], s["n"], compose(inverse(s["pose_j"]), est), np.array([1 / 0.03 ** 2, 1 / 0.03 ** 2, 1 / math.radians(1.0) ** 2]))
        t1 = time.perf_counter(); rb = s["robust"] + [True]
        new = optimize(s["poses"], s["edges"] + [e], robust=rb); scale = optimize.last_scale
        sh = np.hypot(*(np.array(new)[:, :2] - np.array(s["poses"])[:, :2]).T)
        r.update(ok=True, edge=e, new=new, scale=scale, rb=np.asarray(rb, bool), opt_ms=round((time.perf_counter() - t1) * 1e3, 1))
        if sh.max() > REMAP_D: r["grid"] = self._lc_build_grid(new, s["nodes"])
        return r

    def _lc_build_grid(self, poses, nodes):
        g = GridMap(res=self.grid.res, size_m=self.grid.n * self.grid.res, origin=tuple(self.grid.origin))
        with self.grid.lock: g.Hlo[:] = self.grid.Hlo; g.Hhi[:] = self.grid.Hhi   # KF scans are 2-D: keep live height evidence
        for p, (o, f, c) in zip(poses, nodes):
            for _ in range(3): g.integrate(p, o, f, None, c)                     # KFs ~1/4 of scans: weight them up
        return g

    def _lc_apply(self, r):
        """Main thread: add the loop edge, move the nodes, patch the field, swap in the re-rendered planning map."""
        s = r["snap"]; st = self.stats
        st["loop_ms"] = r["loop_ms"]; st["loop_score"] = r["score"]; st["loop_tries"] = st.get("loop_tries", 0) + 1
        if not r["ok"]: return
        n, j, Ns = s["n"], s["j"], len(s["poses"])
        self.edges.append(r["edge"]); self._robust.append(True); self._pairs.add((j, n)); self._last_loop_kf = n; st["loops"] += 1
        self.loop_log = getattr(self, "loop_log", []); self.loop_log.append((s["t_j"], s["t_n"], tuple(r["edge"][2]), len(self.edges) - 1))
        new = list(r["new"])
        if len(self.kfs) > Ns:      # nodes added while the solve ran: rigidly carried by the correction of node n (the anchor at submit)
            D = compose(new[n], inverse(s["poses"][n])); new += [compose(D, k.pose) for k in self.kfs[Ns:]]
        sc = r["scale"]; st["loops_downweighted"] = int((r["rb"] & (sc < 0.5)).sum())
        st["opt_ms"] = r["opt_ms"]; st["opts"] = st.get("opts", 0) + 1; self.loop_scale = sc
        old = [k.pose for k in self.kfs]
        sh = np.hypot(*(np.array(new)[:, :2] - np.array(old)[:, :2]).T); st["loop_shift_m"] = round(float(sh.max()), 3)
        for k, p, o, m in zip(self.kfs, new, old, sh):
            if m > 0.01 or abs(wrap(p[2] - o[2])) > math.radians(0.3):
                self.field.add(o, k.obs, -1); self.field.add(p, k.obs)       # field re-renders lazily on the next match
            k.pose = p
        self._xy = None
        a, da, oa, pa = self._anchor
        self.T = compose(compose(self.kfs[a].pose, da), inverse(oa))
        if sh.max() > REMAP_D:
            g = r.get("grid")
            if g is None: g = self._lc_build_grid([k.pose for k in self.kfs], [(k.obs, k.free, k.carve) for k in self.kfs])
            else:
                for k in self.kfs[Ns:]:
                    for _ in range(3): g.integrate(k.pose, k.obs, k.free, None, k.carve)
            self._lc_swap_grid(g)

    def _lc_swap_grid(self, g):
        with self.grid.lock:
            self.grid.L[:] = g.L; self.grid.seen[:] = g.seen; self.grid.version += 1
        self.stats["remaps"] = self.stats.get("remaps", 0) + 1

    def _optimize_life(self, why):
        t1 = time.perf_counter(); old = [k.pose for k in self.kfs]
        new = optimize(old, self.edges, robust=self._robust)
        sc = optimize.last_scale; rb = np.asarray(self._robust, bool)
        self.stats["loops_downweighted"] = int((rb & (sc < 0.5)).sum())
        self.stats["opt_ms"] = round((time.perf_counter() - t1) * 1e3, 1); self.stats["opts"] = self.stats.get("opts", 0) + 1
        self.loop_scale = sc
        sh = np.hypot(*(np.array(new)[:, :2] - np.array(old)[:, :2]).T)
        self.stats["loop_shift_m"] = round(float(sh.max()), 3)
        for k, p, o, m in zip(self.kfs, new, old, sh):
            if m > 0.01 or abs(wrap(p[2] - o[2])) > math.radians(0.3):
                self.field.add(o, k.obs, -1); self.field.add(p, k.obs)
            k.pose = p
        self._xy = None
        a, da, oa, pa = self._anchor
        self.T = compose(compose(self.kfs[a].pose, da), inverse(oa))
        if sh.max() > REMAP_D: self._rebuild_grid()

    def _rebuild_grid(self):
        g = GridMap(res=self.grid.res, size_m=self.grid.n * self.grid.res, origin=tuple(self.grid.origin))
        g.Hlo[:] = self.grid.Hlo; g.Hhi[:] = self.grid.Hhi
        for k in self.kfs:
            for _ in range(3): g.integrate(k.pose, k.obs, k.free, None, k.carve)
        with self.grid.lock:
            self.grid.L[:] = g.L; self.grid.seen[:] = g.seen; self.grid.version += 1
        self.stats["remaps"] = self.stats.get("remaps", 0) + 1

    def save(self, path):
        """Persist the graph (nodes: odom, pose, scan, path, t; edges; robust flags; T)."""
        import pickle
        with self._lock:
            d = dict(v=1, T=self.T, path=self._path, kfs=[(k.odom, k.pose, k.obs, k.free, k.path, k.t, k.t_obs) for k in self.kfs],
                     edges=[(i, j, tuple(z), np.asarray(f).tolist()) for i, j, z, f in self.edges], robust=list(getattr(self, "_robust", [])),
                     anchor=getattr(self, "_anchor", None))
        tmp = path + ".tmp"; pickle.dump(d, open(tmp, "wb"), protocol=4); os.replace(tmp, path)

    def load(self, path):
        import pickle
        d = pickle.load(open(path, "rb"))
        with self._lock:
            self.kfs = []; self.field = Field(); self._lc_gen += 1
            for od, po, ob, fr, pa, t, to in d["kfs"]:
                k = KF(od, po, ob, fr, pa, None, t); k.t_obs = to; self.kfs.append(k); self.field.add(po, ob)
            self.edges = [(i, j, z, np.array(f)) for i, j, z, f in d["edges"]]; self._robust = d["robust"]
            self._pairs = set((min(i, j), max(i, j)) for i, j, _, _ in self.edges); self._path = d["path"]; self._xy = None
        self._rebuild_grid()
