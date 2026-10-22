"""lineloc: 2D pose (x, y, yaw) of Kevin from image line segments matched to the casita line map, fused with
wheel odometry as an iterated-EKF update.  Per frame (fixed cost):
  1. LSD segments (capped at N_SEG longest), endpoints undistorted with the camera lens;
  2. gravity classification: the interpretation plane of a 3-D vertical line contains the up axis
     (|n_z| < sin 1.5 deg, n = r1 x r2 in the gravity-levelled base frame; camera tilt is known);
     collinear vertical pieces merged -> one bearing + elevation span each;
  3. association in a small window around the odometry prediction (Mahalanobis gate from the prior):
       vertical  -> map corner (bearing, plus height-span compatibility at the predicted range),
       other     -> map horizontal edge at its known height (image point-to-line distance, angle gate);
  4. optional RANSAC on the minimal 3-bearing resection (linear, closed form) when the prior is wide;
  5. robust Gauss-Newton (Huber IRLS) with the odometry prior, posterior covariance;
  6. accept only if >= MIN_V vertical inliers, >= MIN_ALL inliers and the update passes the chi2(3) 99 % test;
     otherwise keep the odometry prediction.
One noise parameter: SIGMA_PX (image measurement sd, px)."""
import math, numpy as np, cv2

SIGMA_PX = 1.5
VERT_TOL = math.sin(math.radians(1.5))
CHI2_1_99, CHI2_2_99, CHI2_3_99 = 6.635, 9.210, 11.345
HUBER_K = 1.345

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def _src():
    import os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    if p not in sys.path: sys.path.append(p)


def world_stl():
    """the house mesh: env LL_STL, else the resolved world mesh (src/world_params.py <- worlds/casita/world.json)."""
    import os
    if os.environ.get("LL_STL"): return os.environ["LL_STL"]
    _src(); import world_params
    return world_params.walls_stl()


def axle_height(tire=None):
    """base_link (axle) above the floor for wheels `tire` (None = the selected wheels; src/robot_params.py)."""
    _src(); import robot_params as RP
    return float(RP.V["axle_height_m"] if tire is None else RP.values(RP.resolve(wheels=tire))["axle_height_m"])


def mast_height(tire=None):
    """mast RGB camera centre above the floor for wheels `tire` (None = selected; stock 0.97615, soft_v3_2 0.99115)."""
    _src(); import robot_params as RP
    return float(RP.V["mast_rgb_height_m"] if tire is None else RP.values(RP.resolve(wheels=tire))["mast_rgb_height_m"])

class Camera:
    def __init__(self, fx, fy, cx, cy, dist=(0, 0, 0, 0), R_bc=None, t_bc=None):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy; self.k = np.array(list(dist) + [0] * (4 - len(dist)), float)[:4]
        self.R_bc = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]]) if R_bc is None else np.asarray(R_bc, float)  # optical -> base (level)
        if t_bc is None: t_bc = (0.0291, 0, mast_height())   # selected wheels (was a fixed stock 0.97615)
        self.t_bc = np.asarray(t_bc, float)    # camera centre in the base frame, z = height above the floor

    @staticmethod
    def tilt(pitch, roll):
        """base(level) <- camera-body rotation for a mast tilted by pitch (+ nose down, about base y) and roll (+ right side down)."""
        cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]); Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Ry @ Rx

    def undistort(self, uv, it=8):
        x = (uv[:, 0] - self.cx) / self.fx; y = (uv[:, 1] - self.cy) / self.fy; x0, y0 = x.copy(), y.copy(); k1, k2, p1, p2 = self.k
        if not self.k.any(): return np.c_[x, y]
        for _ in range(it):
            r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x); dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x = (x0 - dx) / rad; y = (y0 - dy) / rad
        return np.c_[x, y]

    def distort_norm(self, xy):
        x, y = xy[:, 0], xy[:, 1]; k1, k2, p1, p2 = self.k; r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.c_[xd * self.fx + self.cx, yd * self.fy + self.cy]


class LineMap:
    def __init__(self, path):
        M = np.load(path); self.V = M["V"]; self.HA, self.HB = M["HA"], M["HB"]; self.lo = M["lo"]; self.cell = float(M["cell"])
        self.nx, self.ny = M["shape"]; nV, nH = int(M["nV"]), int(M["nH"])
        self.visV = np.unpackbits(M["visV"], 1)[:, :nV].astype(bool).reshape(self.nx, self.ny, nV)
        self.top = float(max(self.V[:, 3].max(), self.HA[:, 2].max()))
        self.visH = np.unpackbits(M["visH"], 1)[:, :nH].astype(bool).reshape(self.nx, self.ny, nH)

    def visible(self, x, y):
        i = int(np.clip((x - self.lo[0]) / self.cell, 0, self.nx - 1)); j = int(np.clip((y - self.lo[1]) / self.cell, 0, self.ny - 1))
        return np.where(self.visV[i, j])[0], np.where(self.visH[i, j])[0]


def se2_cam(pose, cam):
    """world position (3,) of the camera centre and world<-base rotation (3x3) for pose (x, y, yaw)."""
    c, s = math.cos(pose[2]), math.sin(pose[2]); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([pose[0], pose[1], 0.0]) + R @ cam.t_bc, R


class LineLoc:
    def __init__(self, lmap, cam, sigma_px=SIGMA_PX, n_seg=200, min_len=20, lsd_scale=0.5, ransac=0, min_v=2, min_all=3,
                 use_h=True, use_v=True, n_lost=10 ** 9, reloc_draws=64, min_v_reloc=5, reloc_xy=0.25, reloc_yaw_deg=5.0,
                 uniq=0.0, h_junction=True, h_minlen=40, sigma_map=0.02):
        self.m, self.cam = lmap, cam; self.sig = sigma_px; self.n_seg, self.min_len = n_seg, min_len
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, lsd_scale); self.ransac = ransac
        self.min_v, self.min_all, self.use_h, self.use_v = min_v, min_all, use_h, use_v; self.rng = np.random.default_rng(0)
        self.uniq, self.h_junction, self.h_minlen, self.sig_map = uniq, h_junction, h_minlen, sigma_map
        self.n_lost, self.reloc_draws, self.min_v_reloc = n_lost, reloc_draws, min_v_reloc; self.n_fail = 0
        self.P_reloc = np.diag([reloc_xy ** 2, reloc_xy ** 2, math.radians(reloc_yaw_deg) ** 2])

    # ---------- image side ----------
    def segments(self, gray):
        s = self.lsd.detect(gray)[0]
        if s is None: return np.zeros((0, 4))
        s = s.reshape(-1, 4); L = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]); o = np.argsort(-L); o = o[L[o] > self.min_len][: self.n_seg]
        return s[o]

    def observe(self, segs):
        """-> verticals (beta, el_lo, el_hi) in the levelled base frame; others: undistorted normalised endpoints (k,2,2)."""
        cam = self.cam; n = len(segs)
        if n == 0: return np.zeros((0, 3)), np.zeros((0, 2, 2))
        xy = cam.undistort(segs.reshape(-1, 2)).reshape(n, 2, 2)
        rc = np.concatenate([xy, np.ones((n, 2, 1))], 2); rb = rc @ cam.R_bc.T          # rays in base frame
        nrm = np.cross(rb[:, 0], rb[:, 1]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        isv = np.abs(nrm[:, 2]) < VERT_TOL
        # vertical: bearing of the plane (contains z): horizontal direction of the mean ray
        rv = rb[isv]; m = rv.mean(1); beta = np.arctan2(m[:, 1], m[:, 0])
        el = np.arctan2(rv[..., 2], np.hypot(rv[..., 0], rv[..., 1])); V = np.c_[beta, el.min(1), el.max(1)]
        # merge pieces of one vertical (bearing within 0.3 deg)
        o = np.argsort(V[:, 0]); V = V[o]; xv = xy[isv][o]; out = []; vsg = []
        for v, sg in zip(V, xv):
            if out and abs(wrap(v[0] - out[-1][0])) < math.radians(0.3):
                q = out[-1]; q[1], q[2] = min(q[1], v[1]), max(q[2], v[2])
                if np.linalg.norm(sg[1] - sg[0]) > np.linalg.norm(vsg[-1][1] - vsg[-1][0]): vsg[-1] = sg
            else: out.append(list(v)); vsg.append(sg)
        self.vseg = np.array(vsg).reshape(-1, 2, 2)      # longest piece of each merged vertical (normalised), for comparisons
        H = xy[~isv]
        if self.h_minlen: H = H[np.hypot(*(segs[~isv, 2:] - segs[~isv, :2]).T) >= self.h_minlen]
        return np.array(out).reshape(-1, 3), H

    # ---------- map side ----------
    def pred_v(self, pose, ids):
        c, R = se2_cam(pose, self.cam); q = (self.m.V[ids, :2] - c[:2]) @ R[:2, :2]
        return np.arctan2(q[:, 1], q[:, 0]), np.hypot(q[:, 0], q[:, 1])

    def proj_h(self, pose, ids):
        """map horizontal edges -> camera-normalised endpoints (k,2,2) clipped to z_cam > 0.05; ok mask."""
        c, R = se2_cam(pose, self.cam); Rcw = (R @ self.cam.R_bc).T
        A = (self.m.HA[ids] - c) @ Rcw.T; B = (self.m.HB[ids] - c) @ Rcw.T; za, zb = A[:, 2], B[:, 2]
        ok = (za > 0.05) | (zb > 0.05); zc = 0.05
        t = np.clip((zc - za) / np.where(np.abs(zb - za) < 1e-9, 1e-9, zb - za), 0, 1)
        P = A + t[:, None] * (B - A); A = np.where((za < zc)[:, None], P, A); B = np.where((zb < zc)[:, None], P, B)
        a = A[:, :2] / A[:, 2:3]; b = B[:, :2] / B[:, 2:3]
        return np.stack([a, b], 1), ok

    @staticmethod
    def line_dist(seg, e):
        """signed distance of points e (k,m,2) to the infinite line through seg (k,2,2) (normalised coords)."""
        d = seg[:, 1] - seg[:, 0]; L = np.linalg.norm(d, axis=1) + 1e-12; nrm = np.c_[-d[:, 1], d[:, 0]] / L[:, None]
        return ((e - seg[:, None, 0]) * nrm[:, None]).sum(-1), L

    def sigmas(self, pose, vid, hid):
        """per-residual sd (px): image noise + map-model error sigma_map seen at the feature's range."""
        f = self.cam.fx; c, _ = se2_cam(pose, self.cam); out = []
        if len(vid): out.append(np.hypot(*(self.m.V[vid, :2] - c[:2]).T))
        if len(hid):
            A, B = self.m.HA[hid], self.m.HB[hid]; d = B - A; t = np.clip(((c - A) * d).sum(1) / (d * d).sum(1), 0, 1)
            out.append(np.repeat(np.linalg.norm(A + t[:, None] * d - c, axis=1), 2))
        rng = np.concatenate(out) if out else np.zeros(0)
        return np.sqrt(self.sig ** 2 + (f * self.sig_map / np.maximum(rng, 0.3)) ** 2)

    def residuals(self, pose, Vobs, vid, Hobs, hid):
        f = self.cam.fx
        rv = wrap(Vobs[:, 0] - self.pred_v(pose, vid)[0]) * f if len(vid) else np.zeros(0)
        if len(hid):
            seg, _ = self.proj_h(pose, hid); d, _ = self.line_dist(seg, Hobs); rh = d.ravel() * f
        else: rh = np.zeros(0)
        return np.r_[rv, rh]

    def jac(self, pose, *a, eps=(1e-4, 1e-4, 1e-5)):
        r0 = self.residuals(pose, *a); J = np.zeros((len(r0), 3))
        for k in range(3):
            p = pose.copy(); p[k] += eps[k]; J[:, k] = (self.residuals(p, *a) - r0) / eps[k]
        return r0, J

    # ---------- association ----------
    def associate(self, pose, P, Vobs, Hobs, gate_scale=1.0):
        vis_v, vis_h = self.m.visible(pose[0], pose[1]); f = self.cam.fx; s2 = self.sig ** 2
        vo, vi, cands = [], [], []
        if self.use_v and len(Vobs) and len(vis_v):
            beta, rng = self.pred_v(pose, vis_v)
            # bearing Jacobian w.r.t. pose (analytic-free: numerical, vectorised over all visible corners)
            Jb = np.zeros((len(vis_v), 3))
            for k, e in enumerate((1e-4, 1e-4, 1e-5)):
                p = pose.copy(); p[k] += e; Jb[:, k] = wrap(self.pred_v(p, vis_v)[0] - beta) / e
            var = (np.einsum("ij,jk,ik->i", Jb, P, Jb) + s2 / f ** 2) * gate_scale
            hc = self.cam.t_bc[2]; z0, z1 = self.m.V[vis_v, 2], self.m.V[vis_v, 3]
            for i, (b, el0, el1) in enumerate(Vobs):
                d2 = wrap(b - beta) ** 2 / var
                zl, zh = hc + rng * math.tan(el0), hc + rng * math.tan(el1); tol = 0.10 + 0.05 * rng
                ok = (d2 < CHI2_1_99) & (zl > z0 - tol) & (zh < z1 + tol)
                if ok.any():
                    j = np.where(ok)[0]; cands.append((i, vis_v[j], d2[j])); o = j[np.argsort(d2[j])]
                    # nearest neighbour, unless ambiguous (a second corner inside the gate that is not clearly worse)
                    if self.uniq and len(o) > 1 and d2[o[1]] < self.uniq * max(d2[o[0]], 1.0): continue
                    vo.append(i); vi.append(vis_v[o[0]])
        ho, hi = [], []
        if self.use_h and len(Hobs) and len(vis_h):
            if self.h_junction:     # only wall-floor and wall-ceiling junctions (known heights)
                zh = self.m.HA[vis_h, 2]; vis_h = vis_h[(zh < 0.02) | (zh > self.m.top - 0.02)]
            seg, ok = self.proj_h(pose, vis_h); seg, ids = seg[ok], vis_h[ok]
            if len(ids):
                # gate variance per map edge from the prior (numerical Jacobian of the endpoint distances at the obs midpoint is
                # obs dependent; use the map-edge line motion at its own midpoint as a conservative proxy)
                mid = seg.mean(1)[:, None]; Jl = np.zeros((len(ids), 3))
                for k, e in enumerate((1e-4, 1e-4, 1e-5)):
                    p = pose.copy(); p[k] += e; s2_, _ = self.proj_h(p, ids); Jl[:, k] = self.line_dist(s2_, mid)[0][:, 0] / e
                var = (np.einsum("ij,jk,ik->i", Jl, P, Jl) + s2 / f ** 2) * gate_scale
                dm = seg[:, 1] - seg[:, 0]; am = np.arctan2(dm[:, 1], dm[:, 0]); Lm = np.linalg.norm(dm, axis=1)
                for i, e in enumerate(Hobs):
                    do = e[1] - e[0]; ao = math.atan2(do[1], do[0]); dang = np.abs(wrap(2 * (ao - am)) / 2)
                    d, _ = self.line_dist(seg, np.repeat(e[None], len(ids), 0)); d2 = (d ** 2).sum(1) / var / 2
                    tm = (((e.mean(0) - seg[:, 0]) * dm).sum(1)) / (Lm ** 2 + 1e-12)     # obs midpoint along the map edge
                    okh = (d2 < CHI2_2_99 / 2) & (dang < math.radians(3)) & (tm > -0.05) & (tm < 1.05)
                    if okh.any():
                        j = np.where(okh)[0]; o = j[np.argsort(d2[j])]
                        if self.uniq and len(o) > 1 and d2[o[1]] < self.uniq * max(d2[o[0]], 1.0): continue
                        ho.append(i); hi.append(ids[o[0]])
        return np.array(vo, int), np.array(vi, int), cands, np.array(ho, int), np.array(hi, int)

    # ---------- minimal solver ----------
    @staticmethod
    def resect(P2, beta):
        """Bearings beta (robot frame, of the CAMERA centre) to known 2D points P2 (>= 3): linear closed form.
        returns list of (cx, cy, yaw) camera-centre poses."""
        vx, vy = np.cos(beta), np.sin(beta); px, py = P2[:, 0], P2[:, 1]
        A = np.c_[vy, vx, -px * vy + py * vx, -px * vx - py * vy]   # unknowns (a, b, c, s)
        _, _, Vt = np.linalg.svd(A); out = []
        for sgn in (1, -1):
            a, b, c, s = Vt[-1] * sgn; nrm = math.hypot(c, s)
            if nrm < 1e-9: continue
            a, b, c, s = a / nrm, b / nrm, c / nrm, s / nrm
            M = np.array([[c, s], [s, -c]]); t = np.linalg.solve(M, [a, b]); yaw = math.atan2(s, c)
            # cheirality: points in front along their bearings
            d = P2 - t; u = np.c_[np.cos(yaw + beta), np.sin(yaw + beta)]
            if ((d * u).sum(1) > 0).all(): out.append(np.r_[t, yaw])
        return out

    def cam_to_base(self, cp):
        c, s = math.cos(cp[2]), math.sin(cp[2]); t = self.cam.t_bc
        return np.array([cp[0] - (c * t[0] - s * t[1]), cp[1] - (s * t[0] + c * t[1]), cp[2]])

    def ransac_seed(self, pose, P, Vobs, cands, draws):
        """MSAC over minimal 3-bearing resections drawn from the gated candidate pairs; hypotheses outside the prior's
        chi2(3) 99 % region are discarded.  Fixed number of draws -> fixed cost."""
        if len(cands) < 3: return pose, False
        Pi = np.linalg.inv(P); f = self.cam.fx; s = self.sig / f; tau2 = CHI2_1_99
        obs = np.array([c[0] for c in cands]); ids_all = np.unique(np.concatenate([c[1] for c in cands]))
        col = {k: j for j, k in enumerate(ids_all)}; mask = np.zeros((len(cands), len(ids_all)), bool)
        for r, (_, ids, _) in enumerate(cands): mask[r, [col[k] for k in ids]] = True
        b_obs = Vobs[obs, 0]
        def cost(h):
            d2 = wrap(b_obs[:, None] - self.pred_v(h, ids_all)[0][None]) ** 2 / s ** 2; d2 = np.where(mask, d2, np.inf)
            return np.minimum(d2.min(1), tau2).sum()
        best, bc = pose, cost(pose)
        for _ in range(draws):
            tri = self.rng.choice(len(cands), 3, replace=False); ks = [cands[t][1][self.rng.integers(len(cands[t][1]))] for t in tri]
            if len(set(ks)) < 3: continue
            for cp in self.resect(self.m.V[ks, :2], Vobs[[cands[t][0] for t in tri], 0]):
                h = self.cam_to_base(cp); dp = h - pose; dp[2] = wrap(dp[2])
                if dp @ Pi @ dp > CHI2_3_99: continue
                c = cost(h)
                if c < bc: bc, best = c, h
        return best, True

    # ---------- update ----------
    def update(self, gray, pose, P):
        """pose, P: odometry-propagated prior.  -> (pose, P, info).
        After N_LOST consecutive rejected frames (odometry outlier: slip, bump) the window is widened to the relocalisation
        window and the MSAC minimal-solver seed is used; acceptance is then stricter (MIN_V_RELOC verticals)."""
        lost = self.n_fail >= self.n_lost
        if lost: P = P + self.P_reloc
        q, Pq, info = self._update(gray, pose, P, ransac=self.ransac or (self.reloc_draws if lost else 0), min_v=self.min_v_reloc if lost else self.min_v)
        info["lost"] = lost; self.n_fail = 0 if info.get("ok") else self.n_fail + 1
        return q, Pq, info

    def _update(self, gray, pose, P, ransac, min_v):
        Vobs, Hobs = self.observe(self.segments(gray)); info = dict(nV_obs=len(Vobs), nH_obs=len(Hobs))
        x = pose.copy()
        if ransac:
            _, _, cands, _, _ = self.associate(x, P, Vobs, Hobs)
            x, seeded = self.ransac_seed(x, P, Vobs, cands, ransac)
        Pi = np.linalg.inv(P); s = self.sig; Pa = P
        if ransac and seeded: Pa = np.diag([0.01 ** 2, 0.01 ** 2, math.radians(0.2) ** 2])   # window around the MSAC seed
        for outer in range(2):            # stage 0: verticals in the prior window; stage 1: verticals + horizontals in the posterior window
            vo, vi, _, ho, hi = self.associate(x, Pa, Vobs, Hobs if (outer == 1 or not self.use_v) else Hobs[:0])
            a = (Vobs[vo], vi, Hobs[ho], hi)
            if len(vi) + len(hi) == 0: break
            sd = self.sigmas(pose, a[1], a[3])
            for it in range(5):
                r, J = self.jac(x, *a); w = np.minimum(1.0, HUBER_K / np.maximum(np.abs(r) / sd, 1e-9))
                dp = x - pose; dp[2] = wrap(dp[2])
                Hm = J.T @ (J * (w / sd ** 2)[:, None]) + Pi; g = -J.T @ (w * r / sd ** 2) - Pi @ dp
                step = np.linalg.solve(Hm, g); x = x + step; x[2] = wrap(x[2])
                if np.abs(step[:2]).max() < 1e-5 and abs(step[2]) < 1e-6: break
            Pa = np.linalg.inv(Hm)
        if len(vi) + len(hi) == 0: info.update(ok=False, why="noassoc"); return pose, P, info
        r, J = self.jac(x, *a); sd = self.sigmas(x, a[1], a[3]); inl = np.abs(r) < 2.58 * sd; nv = int(inl[: len(vi)].sum()); nh = int(inl[len(vi):].reshape(-1, 2).all(1).sum()) if len(hi) else 0
        Jw = J[inl] / sd[inl, None]; Ppost = np.linalg.inv(Jw.T @ Jw + Pi)
        dp = x - pose; dp[2] = wrap(dp[2]); S = P - Ppost; nis = float(dp @ np.linalg.pinv(S, rcond=1e-9) @ dp)
        info.update(nV=nv, nH=nh, nV_assoc=len(vi), nH_assoc=len(hi), rms=float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else 1e9, nis=nis)
        if nv < min_v or nv + nh < self.min_all: info.update(ok=False, why="few"); return pose, P, info
        if nis > CHI2_3_99: info.update(ok=False, why="gate"); return pose, P, info
        info["ok"] = True; return x, Ppost, info
