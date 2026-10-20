"""Model-based mast-camera localisation against a known structural map (casita wall STL + floor + ceiling planes).

Method (established model-based edge tracking, e.g. RAPiD / Drummond-Cipolla style, chamfer variant):
  1. render the map from the predicted pose by ray casting (Embree): per-pixel face normal + depth;
     map edge pixels = crease (normal change > CREASE_DEG) or occlusion (depth jump) -> back-projected to 3D points,
     so occlusion and silhouettes are exact for the map at the predicted pose;
  2. image edges: Canny with thresholds from the image's own gradient distribution (Otsu on the gradient magnitude),
     Euclidean distance transform;
  3. robust Gauss-Newton (IRLS, Huber on DT residuals) over SE(2) (x, y, yaw), with the odometry prediction as a
     Gaussian prior (covariance from the odometry noise model, accumulated since the last accepted fix);
  4. gating: Mahalanobis test of the correction against the prior covariance (chi2, 3 dof, 99%) and an inlier-ratio
     test; rejected frames keep odometry.  Unobservable directions (e.g. one wall in view) stay on the prior through
     the information form, no special casing.
z, roll, pitch come from the twin (mast extrinsics); only x, y, yaw are estimated."""
import math, numpy as np, cv2, trimesh

AXLE_H = 0.0856
R_MAST = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])     # optical -> base (twin)
CREASE_DEG = 30.0
CHI2_3_99 = 11.345


def se2(x, y, yaw, z=AXLE_H):
    B = np.eye(4); c, s = math.cos(yaw), math.sin(yaw); B[:2, :2] = [[c, -s], [s, c]]; B[:3, 3] = [x, y, z]; return B


class StructMap:
    """Wall STL (+ floor z=0 and ceiling planes) in sim world coordinates."""
    def __init__(self, stl, ceiling_z=2.29, pad=1.0, zmin=-1.0):
        self.zmin = zmin       # only map edges above this height (e.g. above furniture: wall/ceiling structure)
        m = trimesh.load(stl); ext = m.bounds; scale = 0.001 if (ext[1] - ext[0]).max() > 100 else 1.0
        off = -ext[0] * scale; V = m.vertices * scale + off; F = m.faces
        x0, y0 = V[:, 0].min() - pad, V[:, 1].min() - pad; x1, y1 = V[:, 0].max() + pad, V[:, 1].max() + pad
        n = len(V); quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        Vf = np.c_[quad, np.zeros(4)]; Vc = np.c_[quad, np.full(4, ceiling_z)]
        V = np.vstack([V, Vf, Vc]); F = np.vstack([F, [[n, n + 1, n + 2], [n, n + 2, n + 3]], [[n + 4, n + 6, n + 5], [n + 4, n + 7, n + 6]]])
        self.mesh = trimesh.Trimesh(V, F, process=False); self.N = self.mesh.face_normals
        self.n_wall = len(m.faces)

    def render(self, c2w, K, w, h):
        """-> depth (h,w) (inf = miss), face id (h,w) (-1 = miss)."""
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
        dw = d @ c2w[:3, :3].T; o = np.repeat(c2w[None, :3, 3], len(dw), 0)
        loc, ri, ti = self.mesh.ray.intersects_location(o, dw, multiple_hits=False)
        z = np.full(len(dw), np.inf); fid = np.full(len(dw), -1)
        zc = ((loc - c2w[:3, 3]) @ c2w[:3, :3])[:, 2]; z[ri] = zc; fid[ri] = ti
        return z.reshape(h, w), fid.reshape(h, w)

    def edge_points(self, c2w, K, w, h, max_pts=600, rng=None, scale=1.0):
        """Visible structural-edge pixels at this pose -> 3D world points (crease or depth discontinuity).
        scale: global map scale (metric = scale * model coordinates, about the model origin)."""
        if scale != 1.0:
            c2 = c2w.copy(); c2[:3, 3] /= scale
            Xm, E = self.edge_points(c2, K, w, h, max_pts, rng); return Xm * scale, E
        z, fid = self.render(c2w, K, w, h); Nm = np.zeros((h, w, 3)); ok = fid >= 0; Nm[ok] = self.N[fid[ok]]
        E = np.zeros((h, w), bool); cth = math.cos(math.radians(CREASE_DEG))
        for dy, dx in ((0, 1), (1, 0)):
            a = (slice(0, h - dy), slice(0, w - dx)); b = (slice(dy, h), slice(dx, w))
            both = ok[a] & ok[b]
            crease = both & ((Nm[a] * Nm[b]).sum(-1) < cth)
            jump = both & (np.abs(np.nan_to_num(z[a] - z[b], posinf=9, neginf=9)) > 0.05 * np.minimum(z[a], z[b]))
            e = crease | jump
            # put the edge on the nearer pixel (the occluding / visible side)
            near_a = z[a] <= z[b]
            E[a] |= e & near_a; E[b] |= e & ~near_a
        E &= ok & (z < 8.0)
        # local line orientation of the rendered edge map (structure tensor of the blurred mask), mod pi
        Eb = cv2.GaussianBlur(E.astype(np.float32), (7, 7), 1.5); gx, gy = cv2.Sobel(Eb, cv2.CV_32F, 1, 0), cv2.Sobel(Eb, cv2.CV_32F, 0, 1)
        Jxx, Jyy, Jxy = [cv2.GaussianBlur(a, (7, 7), 1.5) for a in (gx * gx, gy * gy, gx * gy)]
        ori = (0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2) % np.pi        # line direction = normal + 90 deg
        v, u = np.nonzero(E); zz = z[v, u]; Xc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * zz, (v + 0.5 - K[1, 2]) / K[1, 1] * zz, zz], -1)
        Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]; th = ori[v, u]
        keep = Xw[:, 2] > self.zmin
        Xw, th = Xw[keep], th[keep]
        if len(Xw) > max_pts:
            rng = rng or np.random.default_rng(0); k = rng.choice(len(Xw), max_pts, replace=False); Xw, th = Xw[k], th[k]
        self.last_theta = th
        return Xw, E


_LSD = None
def image_dt(img_gray, mode="lsd"):
    """mode 'lsd': LSD line segments (von Gioi et al. 2012, a-contrario, parameter-free) rasterised -> DT.
    mode 'canny': Canny with Otsu-derived thresholds."""
    global _LSD
    if mode == "lsd":
        if _LSD is None: _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = _LSD.detect(img_gray)[0]; e = np.zeros(img_gray.shape, np.uint8)
        if lines is not None:
            for x0, y0, x1, y1 in lines.reshape(-1, 4): cv2.line(e, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 255, 1)
        dt = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5); return dt.astype(np.float32), e
    return image_dt_canny(img_gray)

NBIN = 8
def image_dt_dir(img_gray):
    """Directional chamfer (Liu et al. 2010, simplified): LSD segments binned by orientation (NBIN over pi);
    DT per bin built from that bin and its two neighbours (+-1 bin tolerance).  -> (NBIN,h,w) DT stack, edge mask."""
    global _LSD
    if _LSD is None: _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = _LSD.detect(img_gray)[0]; h, w = img_gray.shape; M = np.zeros((NBIN, h, w), np.uint8)
    if lines is not None:
        for x0, y0, x1, y1 in lines.reshape(-1, 4):
            b = int(((math.atan2(y1 - y0, x1 - x0) % math.pi) / math.pi * NBIN + 0.5)) % NBIN
            for bb in (b - 1, b, b + 1): cv2.line(M[bb % NBIN], (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 255, 1)
    D = np.stack([cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 5) for m in M]).astype(np.float32)
    return D, M.max(0)

def image_dt_canny(img_gray):
    """Canny with Otsu-derived hysteresis thresholds (high = Otsu of gradient magnitude, low = high/2) -> DT, edges."""
    g = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1); mag = np.hypot(gx, gy)
    m8 = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(m8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); hi = t / 255 * mag.max()
    e = cv2.Canny(g, hi / 2, hi, L2gradient=True)
    dt = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return dt.astype(np.float32), e


def bilinear(img, u, v):
    h, w = img.shape; u = np.clip(u, 0, w - 1.001); v = np.clip(v, 0, h - 1.001); u0 = u.astype(int); v0 = v.astype(int); a = u - u0; b = v - v0
    return (img[v0, u0] * (1 - a) * (1 - b) + img[v0, u0 + 1] * a * (1 - b) + img[v0 + 1, u0] * (1 - a) * b + img[v0 + 1, u0 + 1] * a * b)


class Localizer:
    def __init__(self, smap, K, w, h, T_bc, huber_px=2.0, sigma_px=1.5, inlier_px=2.0, info_scale=1.0, exposure_s=0.015):
        """info_scale: calibration of the fix information (pixel residuals are correlated, so J'WJ is over-confident);
        set from held-out data so the fix NEES matches chi2(3) (see calibrate()).  exposure_s: no fix while the
        yaw-rate motion blur |w| * exposure * fx exceeds the Huber scale (edges smeared beyond the residual model)."""
        self.m, self.K, self.w, self.h, self.Tbc = smap, K, w, h, T_bc
        self.hub, self.spx, self.inl, self.cinfo, self.exp = huber_px, sigma_px, inlier_px, info_scale, exposure_s

    def project(self, X, p):
        c2w = se2(*p) @ self.Tbc; Xc = (X - c2w[:3, 3]) @ c2w[:3, :3]
        ok = Xc[:, 2] > 0.15; u = Xc[:, 0] / np.maximum(Xc[:, 2], 1e-3) * self.K[0, 0] + self.K[0, 2]; v = Xc[:, 1] / np.maximum(Xc[:, 2], 1e-3) * self.K[1, 1] + self.K[1, 2]
        ok &= (u >= 1) & (u < self.w - 2) & (v >= 1) & (v < self.h - 2); return u, v, ok

    def update(self, gray, p_pred, P_pred, iters=8, directional=True, yaw_rate=0.0):
        """-> (pose, cov, info dict).  p = (x, y, yaw) world; P_pred 3x3 prior covariance."""
        if abs(yaw_rate) * self.exp * self.K[0, 0] > self.hub: return p_pred, P_pred, dict(ok=False, why="blur")
        ns = len(p_pred)            # 3: (x, y, yaw); 4: (x, y, yaw, ln map-scale)
        s0 = math.exp(p_pred[3]) if ns == 4 else 1.0
        X, _ = self.m.edge_points(se2(*p_pred[:3]) @ self.Tbc, self.K, self.w, self.h, scale=s0)
        if len(X) < 30: return p_pred, P_pred, dict(ok=False, why="few_map_edges", n=len(X))
        if directional:
            D, _ = image_dt_dir(gray); b = ((self.m.last_theta / math.pi * NBIN + 0.5).astype(int)) % NBIN
        else:
            D, _ = image_dt(gray); D = D[None]; b = np.zeros(len(X), int)
        Ip = np.linalg.inv(P_pred); p = np.array(p_pred, float); eps = np.full(ns, 1e-3)
        def resid(q):
            Xq = X * math.exp(q[3] - p_pred[3]) if ns == 4 else X
            u, v, ok = self.project(Xq, q[:3]); r = np.full(len(X), np.nan)
            for bb in np.unique(b[ok]):
                m = ok & (b == bb); r[m] = bilinear(D[bb], u[m], v[m])
            return r
        for it in range(iters):
            r0 = resid(p); J = np.zeros((len(X), ns))
            for k in range(ns):
                q = p.copy(); q[k] += eps[k]; J[:, k] = (resid(q) - r0) / eps[k]
            ok = np.isfinite(r0) & np.isfinite(J).all(1)
            if ok.sum() < 30: return p_pred, P_pred, dict(ok=False, why="few_visible", n=int(ok.sum()))
            r, Jo = r0[ok], J[ok]; wgt = np.where(r <= self.hub, 1.0, self.hub / r) / self.spx ** 2     # Huber IRLS
            Hm = Jo.T @ (Jo * wgt[:, None]); H = self.cinfo * Hm + Ip; g = self.cinfo * (Jo.T @ (wgt * r)) + Ip @ (p - p_pred)
            dp = -np.linalg.solve(H, g)
            if ns == 4: dp[3] = float(np.clip(dp[3], -0.02, 0.02))       # trust region on ln(scale) per iteration
            p += dp
            if np.abs(dp).max() < 1e-4: break
        r = resid(p); vis = np.isfinite(r); inl = float((r[vis] < self.inl).mean()) if vis.any() else 0.0
        inl0 = resid(np.array(p_pred, float)); inl0 = float((inl0[np.isfinite(inl0)] < self.inl).mean()) if np.isfinite(inl0).any() else 0.0
        P = np.linalg.inv(H); d = p - p_pred; maha = float(d @ Ip @ d)
        info = dict(n=int(vis.sum()), inl=inl, inl_pred=inl0, maha=maha, it=it + 1, Hm=Hm, p_raw=p.copy())
        if maha > (CHI2_3_99 if ns == 3 else 13.277) or inl < inl0:      # gate: consistent with the prior (chi2 99%) and does not worsen the alignment
            info.update(ok=False, why="gate"); return p_pred, P_pred, info
        info["ok"] = True; return p, P, info
