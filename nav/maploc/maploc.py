"""Model-based mast-camera localisation against a known structural map (casita wall STL + floor + ceiling planes).

Default (v4, LocalizerSeg; established model-based edge tracking, RAPiD / Drummond-Cipolla style, point-to-line):
  1. map edges = the wall mesh's own crease (> CREASE_DEG) and open-boundary edges, sampled every 2 cm (exact 3D, so
     sub-pixel at any resolution); per frame kept if visible (Embree ray reaches the point) and seen as an image
     discontinuity (probe rays either side: normal change > CREASE_DEG or depth jump > 5 %);
  2. image edges = LSD segments (von Gioi et al. 2012) with sub-pixel endpoints, full resolution, undistorted with the
     twin lens; residual = signed distance to the nearest orientation-compatible segment (+-22.5 deg), re-associated
     every iteration;
  3. robust Gauss-Newton (IRLS, Huber) over SE(2) (+ ln global map scale), odometry prediction as a Gaussian prior;
     when the prior yaw sd exceeds the association window, a robust-cost yaw grid within +-3 sigma seeds the solve;
  4. gating: Mahalanobis (chi2 99 %) + inlier fraction must not drop; no fix under yaw-rate motion blur.
Localizer (v3: pixel render + directional chamfer DT) is kept for comparison.
z, roll, pitch come from the twin (mast extrinsics); only x, y, yaw (+ map scale) are estimated."""
import math, os, json, numpy as np, cv2, trimesh

# base_link (= wheel axle) height above the floor = the tyre profile's loaded wheel radius (src/tire_profiles.json,
# single source of truth; read directly, not via src/tire_profile.py, whose module name clashes with the twin's).
# stock 0.08565 (twin rgb height_above_floor 0.97615 - xyz z 0.8905); soft_v3_2 0.10065 (+15 mm, commit 7fcec63).
_TPJ = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "tire_profiles.json")))
AXLE_H_STOCK = float(_TPJ["profiles"]["stock"]["wheel_r_loaded_m"])


def tire_name(tire=None):
    """profile name: tire (name or alias) if given, else the ACTIVE profile (KEVIN_TIRE env, else the json default)."""
    n = (tire if tire is not None else os.environ.get(_TPJ["env"], "")).strip() or _TPJ["default"]
    n = _TPJ["aliases"].get(n, n)
    if n not in _TPJ["profiles"]: raise ValueError("unknown tire profile %r; use one of %s" % (n, list(_TPJ["profiles"])))
    return n


def axle_height(tire=None):
    """base_link above the floor (m) for tyre profile `tire`; None = the active profile (use log_axle_height for logs)."""
    return float(_TPJ["profiles"][tire_name(tire)]["wheel_r_loaded_m"])


def log_axle_height(info):
    """per log: the tyre its info.json records (info["tire"], world.py / realmastlog.py); a log without it predates the
    soft tyres (7fcec63) -> stock."""
    t = (info or {}).get("tire")
    return axle_height(t) if t else AXLE_H_STOCK


AXLE_H = axle_height()   # active profile (live / new recordings); offline harness picks per log (harness.log_axle)
R_MAST = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])     # optical -> base (twin)
CREASE_DEG = 30.0
CHI2_3_99 = 11.345


def se2(x, y, yaw, z=None):
    if z is None: z = AXLE_H
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
    def __init__(self, smap, K, w, h, T_bc, huber_px=2.0, sigma_px=1.5, inlier_px=2.0, info_scale=1.0, exposure_s=0.015, axle_h=None):
        """info_scale: calibration of the fix information (pixel residuals are correlated, so J'WJ is over-confident);
        set from held-out data so the fix NEES matches chi2(3) (see calibrate()).  exposure_s: no fix while the
        yaw-rate motion blur |w| * exposure * fx exceeds the Huber scale (edges smeared beyond the residual model)."""
        self.m, self.K, self.w, self.h, self.Tbc = smap, K, w, h, T_bc
        self.zb = AXLE_H if axle_h is None else float(axle_h)   # base_link height (T_bc is relative to it)
        self.hub, self.spx, self.inl, self.cinfo, self.exp = huber_px, sigma_px, inlier_px, info_scale, exposure_s

    def project(self, X, p):
        c2w = se2(*p, z=self.zb) @ self.Tbc; Xc = (X - c2w[:3, 3]) @ c2w[:3, :3]
        ok = Xc[:, 2] > 0.15; u = Xc[:, 0] / np.maximum(Xc[:, 2], 1e-3) * self.K[0, 0] + self.K[0, 2]; v = Xc[:, 1] / np.maximum(Xc[:, 2], 1e-3) * self.K[1, 1] + self.K[1, 2]
        ok &= (u >= 1) & (u < self.w - 2) & (v >= 1) & (v < self.h - 2); return u, v, ok

    def update(self, gray, p_pred, P_pred, iters=8, directional=True, yaw_rate=0.0):
        """-> (pose, cov, info dict).  p = (x, y, yaw) world; P_pred 3x3 prior covariance."""
        if abs(yaw_rate) * self.exp * self.K[0, 0] > self.hub: return p_pred, P_pred, dict(ok=False, why="blur")
        ns = len(p_pred)            # 3: (x, y, yaw); 4: (x, y, yaw, ln map-scale)
        s0 = math.exp(p_pred[3]) if ns == 4 else 1.0
        X, _ = self.m.edge_points(se2(*p_pred[:3], z=self.zb) @ self.Tbc, self.K, self.w, self.h, scale=s0)
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


# ---------------------------------------------------------------------------------------------------------------------
# v4 (2026-10-20): sub-pixel, resolution-independent edge model.  The v3 pixel render put map edges on integer pixel
# centres (nearer pixel, with a +0.5 px centre-convention offset) and the image edges on a rasterised LSD mask ->
# ~0.5-1 px systematic offsets, which bias the range of floor/ceiling lines and so the global map scale (~-2 %).
# v4: map edges are the mesh's own crease/boundary edges (exact 3D polylines, sampled every 2 cm), tested per frame for
#     (a) visibility: ray from the camera reaches the point (Embree) and
#     (b) being an image discontinuity: probe rays +-PROBE_PX either side of the projected edge see a normal change
#         > CREASE_DEG or a depth jump > 5 % (same criterion as the v3 render, evaluated at exact geometry);
# image edges are LSD segments with their sub-pixel endpoints; residual = signed point-to-line distance to the
# nearest orientation-compatible segment (point-to-line ICP / RAPiD-style), re-associated every Gauss-Newton iteration.
PROBE_PX = 2.0
ANG_TOL = math.pi / 8           # orientation compatibility (same as 8 bins over pi with +-1 bin)


def _crease_edges(mesh, deg=CREASE_DEG):
    m = mesh.copy(); m.merge_vertices()
    ang = m.face_adjacency_angles; E = m.face_adjacency_edges[ang > math.radians(deg)]
    import trimesh.grouping as G
    es = m.edges_sorted; b = es[G.group_rows(es, require_count=1)]         # open boundaries (mesh is a soup)
    E = np.vstack([E, b]) if len(b) else E
    return m.vertices[E[:, 0]], m.vertices[E[:, 1]]


class GeoEdges:
    """Exact structural edges of a StructMap (wall mesh creases/boundaries; floor/ceiling junctions are the wall
    bottom/top edges)."""
    def __init__(self, smap, step=0.02):
        self.m = smap
        wall = trimesh.Trimesh(smap.mesh.vertices, smap.mesh.faces[:smap.n_wall], process=False)
        A, B = _crease_edges(wall); L = np.linalg.norm(B - A, axis=1); k = L > 1e-4; A, B, L = A[k], B[k], L[k]
        n = np.maximum(1, np.ceil(L / step).astype(int)); idx = np.repeat(np.arange(len(A)), n)
        t = (np.concatenate([np.arange(c) for c in n]) + 0.5) / np.repeat(n, n)
        self.P = A[idx] + (B - A)[idx] * t[:, None]; self.T = ((B - A) / L[:, None])[idx]
        self.total_m = float(L.sum())

    def _cast(self, o, d):
        """first hit along rays -> (dist (inf = miss), face id)."""
        loc, ri, ti = self.m.mesh.ray.intersects_location(o, d, multiple_hits=False)
        dist = np.full(len(d), np.inf); fid = np.full(len(d), -1)
        dist[ri] = np.einsum("ij,ij->i", loc - o[ri], d[ri]); fid[ri] = ti; return dist, fid

    def visible(self, c2w, K, w, h, scale=1.0, zmax=8.0, max_pts=800, rng=None, margin=3):
        """-> (Xw metric world points (n,3), image orientation theta (n,)) of edge samples visible as discontinuities."""
        R, c = c2w[:3, :3], c2w[:3, 3]; cm = c / scale                    # camera centre in model coordinates
        Xc = (self.P - cm) @ R * scale; z = Xc[:, 2]
        ok = (z > 0.2) & (z < zmax)
        u = Xc[:, 0] / np.maximum(z, 1e-3) * K[0, 0] + K[0, 2]; v = Xc[:, 1] / np.maximum(z, 1e-3) * K[1, 1] + K[1, 2]
        ok &= (u >= margin) & (u < w - 1 - margin) & (v >= margin) & (v < h - 1 - margin)
        i = np.nonzero(ok)[0]
        if len(i) == 0: return np.zeros((0, 3)), np.zeros(0)
        P, T, u, v = self.P[i], self.T[i], u[i], v[i]
        # (a) visibility
        d = P - cm; L = np.linalg.norm(d, axis=1); d /= L[:, None]
        hit, _ = self._cast(np.repeat(cm[None], len(d), 0), d)
        vis = hit >= L - (0.01 + 0.005 * L)
        P, T, u, v = P[vis], T[vis], u[vis], v[vis]
        if len(P) == 0: return np.zeros((0, 3)), np.zeros(0)
        # image direction of the edge
        Xc2 = ((P + 0.02 * T) - cm) @ R; u2 = Xc2[:, 0] / Xc2[:, 2] * K[0, 0] + K[0, 2]; v2 = Xc2[:, 1] / Xc2[:, 2] * K[1, 1] + K[1, 2]
        du, dv = u2 - u, v2 - v; nn = np.hypot(du, dv) + 1e-9; du, dv = du / nn, dv / nn
        # (b) discontinuity probes either side (perpendicular in the image)
        res = []
        for sgn in (-1, 1):
            pu, pv = u + sgn * PROBE_PX * (-dv), v + sgn * PROBE_PX * du
            rc = np.stack([(pu - K[0, 2]) / K[0, 0], (pv - K[1, 2]) / K[1, 1], np.ones_like(pu)], -1) @ R.T
            dist, fid = self._cast(np.repeat(cm[None], len(rc), 0), rc / np.linalg.norm(rc, axis=1)[:, None])
            zc = dist / np.linalg.norm(rc, axis=1); res.append((zc, fid))
        (z1, f1), (z2, f2) = res; miss = (f1 < 0) | (f2 < 0)
        N = self.m.N; cth = math.cos(math.radians(CREASE_DEG))
        crease = ~miss & ((N[np.maximum(f1, 0)] * N[np.maximum(f2, 0)]).sum(1) < cth)
        jump = ~miss & (np.abs(z1 - z2) > 0.05 * np.minimum(z1, z2))
        keep = (crease | jump | (miss & ~((f1 < 0) & (f2 < 0)))) & (P[:, 2] > self.m.zmin)
        P, th = P[keep], (np.arctan2(dv, du) % math.pi)[keep]
        if len(P) > max_pts:
            rng = rng or np.random.default_rng(0); k = rng.choice(len(P), max_pts, replace=False); P, th = P[k], th[k]
        return P * scale, th


_LSD4 = None
def lsd_segments(gray):
    """LSD (von Gioi et al.) segments with sub-pixel endpoints -> (n,4) float, orientation mod pi."""
    global _LSD4
    if _LSD4 is None: _LSD4 = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    s = _LSD4.detect(gray)[0]
    if s is None: return np.zeros((0, 4), np.float32), np.zeros(0)
    s = s.reshape(-1, 4); return s, np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0]) % math.pi


def associate(u, v, th, S, Sth, max_px):
    """nearest orientation-compatible segment per point -> (seg index or -1, point-to-segment distance)."""
    if len(S) == 0: return np.full(len(u), -1), np.full(len(u), np.inf)
    a = S[None, :, :2]; b = S[None, :, 2:]; p = np.stack([u, v], -1)[:, None, :]; ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    dd = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)
    dth = np.abs(th[:, None] - Sth[None]); dth = np.minimum(dth, math.pi - dth); dd[dth > ANG_TOL] = np.inf
    j = dd.argmin(1); dmin = dd[np.arange(len(u)), j]; j[dmin > max_px] = -1; return j, dmin


CHI2_99 = {3: 11.345, 4: 13.277, 5: 15.086}


def Ry(a):
    c, s = math.cos(a), math.sin(a); R = np.eye(4); R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c; return R


class LocalizerSeg(Localizer):
    """v4: GeoEdges + LSD point-to-line residuals; same prior / IRLS / gating / scale state as Localizer."""
    def __init__(self, smap, K, w, h, T_bc, huber_px=3.0, sigma_px=1.5, inlier_px=3.0, info_scale=1.0, exposure_s=0.015, assoc_px=15.0, axle_h=None):
        super().__init__(smap, K, w, h, T_bc, huber_px, sigma_px, inlier_px, info_scale, exposure_s, axle_h)
        self.ge = smap.geo if hasattr(smap, "geo") else GeoEdges(smap); smap.geo = self.ge; self.assoc = assoc_px

    def update(self, gray, p_pred, P_pred, iters=8, yaw_rate=0.0, **kw):
        if abs(yaw_rate) * self.exp * self.K[0, 0] > self.hub: return p_pred, P_pred, dict(ok=False, why="blur")
        # state: x, y, yaw [, ln map-scale [, static mast pitch (rad, + = nose down, about base y)]]
        ns = len(p_pred); s0 = math.exp(p_pred[3]) if ns >= 4 else 1.0; pt0 = p_pred[4] if ns >= 5 else 0.0
        X, th = self.ge.visible(se2(*p_pred[:3], z=self.zb) @ Ry(pt0) @ self.Tbc, self.K, self.w, self.h, scale=s0)
        if len(X) < 30: return p_pred, P_pred, dict(ok=False, why="few_map_edges", n=len(X))
        S, Sth = lsd_segments(gray)
        # capture range: when the prior yaw sd is large (e.g. after a blur-skipped spin with a turn-scale error) the
        # local association (assoc_px) cannot pull in a yaw error of tens of px -> robust-cost grid over yaw within
        # +-3 sigma (step = half the association window), Gauss-Newton starts from the best (prior term included).
        p_start = np.array(p_pred, float); syaw = math.sqrt(P_pred[2, 2]); step = 0.5 * self.assoc / self.K[0, 0]
        if 3 * syaw > step:
            def cost(q):
                u, v, ok = self.project(X, q[:3], q[4] if ns >= 5 else 0.0); j, d = associate(u[ok], v[ok], th[ok], S, Sth, self.assoc)
                d = np.where(j >= 0, d, self.assoc); hb = self.hub
                return float(np.where(d <= hb, 0.5 * d * d, hb * (d - 0.5 * hb)).sum() / self.spx ** 2 * self.cinfo / max(1, ok.sum()) * len(X)) + 0.5 * (q[2] - p_pred[2]) ** 2 / P_pred[2, 2]
            best = min(np.arange(-3 * syaw, 3 * syaw + 1e-9, step), key=lambda dy: cost(np.r_[p_pred[:2], p_pred[2] + dy, p_pred[3:]]))
            if best != 0.0:
                p_start[2] += best
                X, th = self.ge.visible(se2(*p_start[:3], z=self.zb) @ Ry(pt0) @ self.Tbc, self.K, self.w, self.h, scale=s0)
                if len(X) < 30: return p_pred, P_pred, dict(ok=False, why="few_map_edges", n=len(X))
        nrm = np.stack([-(S[:, 3] - S[:, 1]), S[:, 2] - S[:, 0]], -1); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        Ip = np.linalg.inv(P_pred); p = p_start.copy(); eps = np.full(ns, 1e-4)
        def proj(q):
            Xq = X * math.exp(q[3] - p_pred[3]) if ns >= 4 else X
            return self.project(Xq, q[:3], q[4] if ns >= 5 else 0.0)
        def assoc_frac(q):
            u, v, ok = proj(q); j, dmin = associate(u[ok], v[ok], th[ok], S, Sth, self.assoc)
            return float((dmin < self.inl).mean()) if ok.any() else 0.0
        for it in range(iters):
            u, v, ok = proj(p); j, _ = associate(u, v, th, S, Sth, self.assoc); m = ok & (j >= 0)
            if m.sum() < 30: return p_pred, P_pred, dict(ok=False, why="few_visible", n=int(m.sum()))
            jj = j[m]; a = S[jj, :2]; nj = nrm[jj]
            def r_of(q):
                uq, vq, _ = proj(q); return ((np.stack([uq[m], vq[m]], -1) - a) * nj).sum(1)
            r0 = r_of(p); J = np.zeros((len(r0), ns))
            for k in range(ns):
                q = p.copy(); q[k] += eps[k]; J[:, k] = (r_of(q) - r0) / eps[k]
            ar = np.abs(r0); wgt = np.where(ar <= self.hub, 1.0, self.hub / np.maximum(ar, 1e-9)) / self.spx ** 2
            Hm = J.T @ (J * wgt[:, None]); H = self.cinfo * Hm + Ip; g = self.cinfo * (J.T @ (wgt * r0)) + Ip @ (p - p_pred)
            dp = -np.linalg.solve(H, g)
            if ns >= 4: dp[3] = float(np.clip(dp[3], -0.02, 0.02))
            if ns >= 5: dp[4] = float(np.clip(dp[4], -0.005, 0.005))
            p += dp
            if np.abs(dp).max() < 1e-5: break
        inl, inl0 = assoc_frac(p), assoc_frac(np.array(p_pred, float))
        P = np.linalg.inv(H); d = p - p_pred; maha = float(d @ Ip @ d)
        info = dict(n=int(m.sum()), inl=inl, inl_pred=inl0, maha=maha, it=it + 1, Hm=Hm, p_raw=p.copy())
        if maha > CHI2_99[ns] or inl < inl0:
            info.update(ok=False, why="gate"); return p_pred, P_pred, info
        info["ok"] = True; return p, P, info

    def project(self, X, p, pitch=0.0):
        c2w = se2(*p, z=self.zb) @ Ry(pitch) @ self.Tbc; Xc = (X - c2w[:3, 3]) @ c2w[:3, :3]
        ok = Xc[:, 2] > 0.15; zz = np.maximum(Xc[:, 2], 1e-3)
        u = Xc[:, 0] / zz * self.K[0, 0] + self.K[0, 2]; v = Xc[:, 1] / zz * self.K[1, 1] + self.K[1, 2]
        ok &= (u >= 1) & (u < self.w - 2) & (v >= 1) & (v < self.h - 2); return u, v, ok
