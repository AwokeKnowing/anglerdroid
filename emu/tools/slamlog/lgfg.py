"""Candidate 'lgfg': learned-feature RGB-D odometry (DISK + LightGlue, 2023) fused with wheel+gyro odometry in a GTSAM
iSAM2 pose graph (SE(2) base poses), offline on a slamlog.  python -m emu.tools.slamlog.lgfg <logdir> [cam=rs2] [out.json]
Measurement models (no tuned weights):
  * VO / loop factor: base-frame xy of matched depth points, P_i = T_ij P_j; 2-point RANSAC with a 3-sigma gate from the
    point noise table SIG(z) (measured against GT on log L1 by calib(); z^2 extrapolation beyond), LS refit on inliers, covariance (J^T W J)^-1
    scaled by the inlier residual chi2/dof.  Ground robot: z, roll, pitch fixed by the twin -> planar SE(2) solve.
  * odometry factor: nav's wheel + gyro EKF increments, sigmas from the emu 'realistic' noise model (wheel radius 0.5 %,
    gyro scale 1 %, gyro white 3 mrad/s/sample at 200 Hz, encoder jitter 1.3 mm/s), Huber kernel (k = 1.345, the
    standard 95 %-efficiency constant) for unmodelled slip / pushes.
  * loop factor: keyframes every KF_D m or KF_A rad; candidates within 3 m (current estimate) and older than 10 s;
    accepted if >= MIN_INL RANSAC inliers and chi2(3) < 11.34 (99 %) against the current estimate (innovation
    covariance = measurement + both marginals, Adjoint-transported)."""
import json, math, sys, time
import numpy as np, torch, gtsam
import kornia.feature as KF
from .log import Log, wrap
from .metrics import evaluate

DEV = "cuda"; NKP = 2048
import os; ODOM = os.environ.get("LGFG_ODOM", "est")   # "0" VO only | "fixed" old factor (1 % scale prior in sigma) | "est"
# "est": odom factor with the gyro scale s and the odom clock offset tau as graph STATES (one each for the run):
#   z_k = odom increment between t_{k-1}+tau and t_k+tau (interpolated), heading / s;  priors s ~ N(1, 0.01) (BMI055 /
#   emu 'realistic' g_scale spec), tau ~ N(0, 0.1 s) (broad).  Per-step noise: the odom pose samples carry the nav tick's
#   timing jitter, so the increment error grows with the rate: sigma_th = |w| SIG_T, sigma_d = |v| SIG_T (+ the encoder /
#   gyro white terms).  SIG_T = 15 ms measured on calibration log L2 (robust std of the per-0.1 s heading residual after
#   fitting s, tau: 13-16 % of the step for 0.05-0.3 rad steps), not fitted on the test log L3.
SIG_T = 0.015
VOCOV = os.environ.get("LGFG_VOCOV", "resid")
CALIB = dict(z=[0.48, 0.69, 0.78, 0.87, 0.97, 1.11, 1.31, 1.65], sigma=[0.0014, 0.0022, 0.0037, 0.0064, 0.0088, 0.0103, 0.012, 0.0445])
# ^ calib() on log L1 (62k matched RS2 points under GT motion; includes GT timing jitter, so conservative)
def SIG(z): return np.where(np.asarray(z) > CALIB["z"][-1], CALIB["sigma"][-1] * (np.asarray(z) / CALIB["z"][-1]) ** 2, np.interp(z, CALIB["z"], CALIB["sigma"]))
KF_D, KF_A, MIN_INL, N_LOOP = 0.3, math.radians(20), 20, 3     # N_LOOP: compute bound (nearest keyframes tried)
R_ERR, G_SCALE, G_WHITE, G_HZ, ENC_J = 0.005, 0.01, 0.003, 200.0, 0.0013

class Front:
    def __init__(self, log, cam):
        self.disk = KF.DISK.from_pretrained("depth").to(DEV).eval(); self.lg = KF.LightGlueMatcher("disk").to(DEV).eval()
        self.log, self.cam = log, cam; self.K, (self.w, self.h) = log.K(cam, "color"); self.R, self.T = log.ext[cam]
    @torch.inference_mode()
    def depth_reg(self, f):
        """depth -> colour camera on the GPU (same as Log.depth_in_color, z-buffered with scatter amin)."""
        if not hasattr(self, "_ray"):
            Kd, (wd, hd) = self.log.K(self.cam, "depth"); v, u = torch.meshgrid(torch.arange(hd, device=DEV), torch.arange(wd, device=DEV), indexing="ij")
            self._ray = torch.stack([(u - Kd[0, 2]) / Kd[0, 0], (v - Kd[1, 2]) / Kd[1, 1]], -1).float().reshape(-1, 2)
            self._tc = torch.tensor(self.log.streams["color"]["pose"][1], device=DEV).float()
        z = torch.from_numpy(self.log.depth(f)).to(DEV).reshape(-1); m = z > 0; z = z[m]; r = self._ray[m]
        X = torch.stack([r[:, 0] * z, r[:, 1] * z, z], 1) - self._tc
        uc = torch.round(X[:, 0] / X[:, 2] * self.K[0, 0] + self.K[0, 2]).long(); vc = torch.round(X[:, 1] / X[:, 2] * self.K[1, 1] + self.K[1, 2]).long()
        ok = (uc >= 0) & (uc < self.w) & (vc >= 0) & (vc < self.h); out = torch.full((self.h * self.w,), float("inf"), device=DEV)
        out.scatter_reduce_(0, vc[ok] * self.w + uc[ok], X[ok, 2], reduce="amin"); out[torch.isinf(out)] = 0
        return out.reshape(self.h, self.w).cpu().numpy()

    @torch.inference_mode()
    def frame(self, f):
        rgb = self.log.rgb(f); z = self.depth_reg(f)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255)[None].to(DEV)
        o = self.disk(x, NKP, pad_if_not_divisible=True)[0]; kp = o.keypoints.cpu().numpy(); u = np.round(kp).astype(int)
        u[:, 0] = u[:, 0].clip(0, self.w - 1); u[:, 1] = u[:, 1].clip(0, self.h - 1); zz = z[u[:, 1], u[:, 0]]
        ok = (zz > 0.25) & (zz < 6.0)
        Pc = np.stack([(kp[:, 0] - self.K[0, 2]) / self.K[0, 0] * zz, (kp[:, 1] - self.K[1, 2]) / self.K[1, 1] * zz, zz], 1)
        Pb = Pc @ self.R.T + self.T
        return dict(kp=o.keypoints[ok], desc=o.descriptors[ok], P=Pb[ok, :2], z=zz[ok])
    @torch.inference_mode()
    def match(self, a, b):
        if len(a["kp"]) < 8 or len(b["kp"]) < 8: return np.zeros((0, 2), int)
        la = KF.laf_from_center_scale_ori(a["kp"][None], torch.ones(1, len(a["kp"]), 1, 1, device=DEV))
        lb = KF.laf_from_center_scale_ori(b["kp"][None], torch.ones(1, len(b["kp"]), 1, 1, device=DEV))
        _, idx = self.lg(a["desc"], b["desc"], la, lb, hw1=(self.h, self.w), hw2=(self.h, self.w)); return idx.cpu().numpy()

def se2_fit(A, B, w=None):
    """least squares x: A ~ R B + t (2-D)."""
    w = np.ones(len(A)) if w is None else w; ma = (w[:, None] * A).sum(0) / w.sum(); mb = (w[:, None] * B).sum(0) / w.sum()
    H = ((B - mb) * w[:, None]).T @ (A - ma); th = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = math.cos(th), math.sin(th); t = ma - np.array([[c, -s], [s, c]]) @ mb; return t[0], t[1], th

def rel(fa, fb, idx, rng):
    """T_ab (pose of frame b in frame a) from matches; returns (x, y, th, cov3, n_inl) or None."""
    if len(idx) < MIN_INL: return None
    A, B = fa["P"][idx[:, 0]], fb["P"][idx[:, 1]]; sg = SIG(np.maximum(fa["z"][idx[:, 0]], fb["z"][idx[:, 1]]))
    best = None
    for _ in range(200):
        i, j = rng.choice(len(A), 2, replace=False)
        if np.linalg.norm(B[i] - B[j]) < 0.05: continue
        x, y, th = se2_fit(A[[i, j]], B[[i, j]]); c, s = math.cos(th), math.sin(th)
        r = A - (B @ np.array([[c, s], [-s, c]]) + [x, y]); inl = (r ** 2).sum(1) < (3 * sg) ** 2 * 2.3   # chi2(2) 99 % ~ 9.2 = 3^2 * ~1
        if best is None or inl.sum() > best.sum(): best = inl
    if best is None or best.sum() < MIN_INL: return None
    for _ in range(2):
        W = 1 / sg[best] ** 2; x, y, th = se2_fit(A[best], B[best], W); c, s = math.cos(th), math.sin(th)
        r = A - (B @ np.array([[c, s], [-s, c]]) + [x, y]); best = (r ** 2).sum(1) < (3 * sg) ** 2 * 2.3
        if best.sum() < MIN_INL: return None
    W = 1 / sg[best] ** 2; Bb = B[best]; Rb = Bb @ np.array([[c, s], [-s, c]])
    J = np.zeros((2 * len(Bb), 3)); J[0::2, 0] = 1; J[1::2, 1] = 1; J[0::2, 2] = -Rb[:, 1]; J[1::2, 2] = Rb[:, 0]
    Wv = np.repeat(W, 2); chi = float((W[:, None] * r[best] ** 2).sum()) / max(1, 2 * len(Bb) - 3)
    cov = np.linalg.inv(J.T @ (Wv[:, None] * J)) * (chi if VOCOV == "resid" else max(chi, 1.0))   # resid: standard LS sigma^2 estimate from this pair's own inlier residuals
    # gtsam Pose2 between noise is in the body frame of a: (x, y) already in a's frame, theta last
    return x, y, th, cov, int(best.sum())

KS, KT = 10 ** 7, 10 ** 7 + 1; ODO = None

def odom_err(odo, ta, tb):
    t, X, Y, A = odo
    def z(tau, s):
        pa = gtsam.Pose2(np.interp(ta + tau, t, X), np.interp(ta + tau, t, Y), np.interp(ta + tau, t, A))
        pb = gtsam.Pose2(np.interp(tb + tau, t, X), np.interp(tb + tau, t, Y), np.interp(tb + tau, t, A)); d = pa.between(pb)
        return gtsam.Pose2(d.x(), d.y(), d.theta() / s)
    def e(xa, xb, s, tau): return z(tau, s).localCoordinates(xa.between(xb))
    def f(this, vals, H):
        k = this.keys(); xa, xb = vals.atPose2(k[0]), vals.atPose2(k[1]); s = vals.atVector(k[2])[0]; tau = vals.atVector(k[3])[0]
        r = e(xa, xb, s, tau)
        if H is not None:
            h = 1e-6
            for i, key in enumerate((0, 1)):
                J = np.zeros((3, 3))
                for j in range(3):
                    dv = np.zeros(3); dv[j] = h; xp = (xa if key == 0 else xb).retract(dv)
                    J[:, j] = ((e(xp, xb, s, tau) if key == 0 else e(xa, xp, s, tau)) - r) / h
                H[i] = J
            H[2] = ((e(xa, xb, s + h, tau) - r) / h)[:, None]; H[3] = ((e(xa, xb, s, tau + 1e-4) - r) / 1e-4)[:, None]
        return r
    return f

def odom_sigma(dx, dy, dth, dt):
    d = math.hypot(dx, dy); st = G_SCALE * abs(dth) + G_WHITE / math.sqrt(G_HZ) * math.sqrt(dt * G_HZ) + 1e-4
    sd = R_ERR * d + ENC_J * dt + 1e-4; return np.array([sd, sd + d * st, st])

def run(d, cam="rs2", loops=True, every=1, win=None):
    L = Log(d); N = L.nav(); fr = L.frames[cam][::every]; fr = [x for x in fr if N["t"][0] <= x[0] <= N["t"][-1] and L.gt_t[0] <= x[0] <= L.gt_t[-1]]
    if win: fr = [x for x in fr if fr[0][0] + win[0] <= x[0] <= fr[0][0] + win[1]]
    od = np.column_stack([np.interp([t for t, _ in fr], N["t_od"], N["odom"][:, k]) for k in range(2)] + [np.interp([t for t, _ in fr], N["t_od"], np.unwrap(N["odom"][:, 2]))])
    F = Front(L, cam); rng = np.random.default_rng(0)
    global ODO; ODO = (N["t_od"], N["odom"][:, 0], N["odom"][:, 1], np.unwrap(N["odom"][:, 2]))
    isam = gtsam.ISAM2(); out = []; kfs = []; feats = {}; prev = None; stats = dict(vo=0, vo_fail=0, loops=0, ms=[])
    huber = lambda nm: gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), nm)
    for k, (t, f) in enumerate(fr):
        t0 = time.perf_counter(); g = gtsam.NonlinearFactorGraph(); v = gtsam.Values(); fk = F.frame(f); feats[k] = fk
        if k == 0:
            g.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(*od[0]), gtsam.noiseModel.Diagonal.Sigmas([1e-4, 1e-4, 1e-5]))); v.insert(0, gtsam.Pose2(*od[0]))
            if ODOM == "est":
                v.insert(KS, np.array([1.0])); v.insert(KT, np.array([0.0]))
                g.add(gtsam.PriorFactorVector(KS, np.array([1.0]), gtsam.noiseModel.Isotropic.Sigma(1, 0.01)))
                g.add(gtsam.PriorFactorVector(KT, np.array([0.0]), gtsam.noiseModel.Isotropic.Sigma(1, 0.1)))
        else:
            a, b = gtsam.Pose2(*od[k - 1]), gtsam.Pose2(*od[k]); dz = a.between(b); dt = t - fr[k - 1][0]
            if ODOM == "fixed": g.add(gtsam.BetweenFactorPose2(k - 1, k, dz, huber(gtsam.noiseModel.Diagonal.Sigmas(odom_sigma(dz.x(), dz.y(), dz.theta(), dt)))))
            elif ODOM == "est":
                sg = odom_sigma(dz.x(), dz.y(), 0.0, dt) + np.array([1, 1, 0]) * math.hypot(dz.x(), dz.y()) / dt * SIG_T + np.array([0, 0, abs(dz.theta()) / dt * SIG_T])
                g.add(gtsam.CustomFactor(huber(gtsam.noiseModel.Diagonal.Sigmas(sg)), [k - 1, k, KS, KT], odom_err(ODO, fr[k - 1][0], t)))
            v.insert(k, isam.calculateEstimatePose2(k - 1).compose(dz))
            n_vo = 0
            for j in (k - 1, k - 2):
                if j < 0: continue
                r = rel(feats[j], fk, F.match(feats[j], fk), rng)
                if r is None: stats["vo_fail"] += 1; continue
                g.add(gtsam.BetweenFactorPose2(j, k, gtsam.Pose2(r[0], r[1], r[2]), huber(gtsam.noiseModel.Gaussian.Covariance(r[3])))); stats["vo"] += 1; n_vo += 1
            if n_vo == 0 and ODOM == "0":     # VO-only ablation: a frame with no VO link would leave the graph indeterminant
                # (gtsam IndeterminantLinearSystemException = the earlier 'stall'): bridge it with the odom increment
                g.add(gtsam.BetweenFactorPose2(k - 1, k, dz, gtsam.noiseModel.Diagonal.Sigmas(odom_sigma(dz.x(), dz.y(), dz.theta(), dt)))); stats["bridged"] = stats.get("bridged", 0) + 1
        isam.update(g, v); est = isam.calculateEstimatePose2(k)
        if loops and (not kfs or math.hypot(est.x() - kfs[-1][1].x(), est.y() - kfs[-1][1].y()) > KF_D or abs(wrap(est.theta() - kfs[-1][1].theta())) > KF_A):
            cur = isam.calculateEstimate(); added = 0; cand = []
            for (j, pj, tj) in kfs:
                pj = cur.atPose2(j); dd = math.hypot(est.x() - pj.x(), est.y() - pj.y())
                if t - tj >= 10 and dd <= 3.0: cand.append((dd + abs(wrap(est.theta() - pj.theta())), j, pj))
            for _, j, pj in sorted(cand, key=lambda c: c[0])[:N_LOOP]:
                r = rel(feats[j], fk, F.match(feats[j], fk), rng)
                if r is None: continue
                z = gtsam.Pose2(r[0], r[1], r[2]); rl = pj.between(est); dlt = rl.localCoordinates(z)
                Ad = rl.inverse().AdjointMap(); S = r[3] + Ad @ isam.marginalCovariance(j) @ Ad.T + isam.marginalCovariance(k)
                if dlt @ np.linalg.solve(S, dlt) > 11.34: stats["loop_rej"] = stats.get("loop_rej", 0) + 1; continue   # chi2(3) 99 %
                g2 = gtsam.NonlinearFactorGraph(); g2.add(gtsam.BetweenFactorPose2(j, k, z, huber(gtsam.noiseModel.Gaussian.Covariance(r[3])))); isam.update(g2, gtsam.Values()); added += 1
            stats["loops"] += added; est = isam.calculateEstimatePose2(k); kfs.append((k, est, t))
        for j in list(feats):
            if j < k - 2 and j not in {q[0] for q in kfs}: del feats[j]
        stats["ms"].append((time.perf_counter() - t0) * 1e3); out.append([t, est.x(), est.y(), est.theta()])
        if k % 100 == 0: print("frame %d/%d ms %.0f vo %d fail %d loops %d" % (k, len(fr), np.median(stats["ms"][-100:]), stats["vo"], stats["vo_fail"], stats["loops"]), flush=True)
    final = isam.calculateEstimate(); sm = np.array([[t, final.atPose2(k).x(), final.atPose2(k).y(), final.atPose2(k).theta()] for k, (t, _) in enumerate(fr)])
    out = np.array(out); out[:, 3] = np.unwrap(out[:, 3]); sm[:, 3] = np.unwrap(sm[:, 3])
    ms = np.array(stats.pop("ms"))
    if ODOM == "est": stats.update(gyro_scale=float(final.atVector(KS)[0]), odom_tau_s=float(final.atVector(KT)[0]))
    return dict(causal=evaluate(out, L), smoothed=evaluate(sm, L), frames=len(fr), ms_p50=float(np.median(ms)), ms_p90=float(np.percentile(ms, 90)),
                gpu_mb=float(torch.cuda.max_memory_allocated() / 1e6), **stats), out, sm

if __name__ == "__main__":
    a = sys.argv[1:]; cam = a[1] if len(a) > 1 else "rs2"
    w = [x for x in a if x.startswith("win=")]; w = tuple(map(float, w[0][4:].split(","))) if w else None
    r, out, sm = run(a[0], cam, loops="noloop" not in a, win=w); print(json.dumps(r, indent=1))
    o = [x for x in a if x.endswith(".json")]
    if o: json.dump(r, open(o[0], "w"), indent=1); np.save(o[0][:-5] + "_traj.npy", out); np.save(o[0][:-5] + "_sm.npy", sm)

def calib(d, cam="rs2", n=150):
    """Depth-point xy noise vs range from GT relative poses (log L1): returns (SIG0, SIG2) of sigma = SIG0 + SIG2 z^2,
    fitted to per-range-bin robust std (1.4826 MAD per axis) of matched-point residuals under the GT motion."""
    L = Log(d); F = Front(L, cam); fr = L.frames[cam]; rng = np.random.default_rng(1); R, Z = [], []
    for k in rng.choice(len(fr) - 1, n, replace=False):
        (ta, fa), (tb, fb) = fr[k], fr[k + 1]; a, b = F.frame(fa), F.frame(fb); idx = F.match(a, b)
        if len(idx) < MIN_INL: continue
        G = L.gt([ta, tb]); c, s = math.cos(G[0, 2]), math.sin(G[0, 2]); dxy = np.array([[c, s], [-s, c]]) @ (G[1, :2] - G[0, :2]); th = G[1, 2] - G[0, 2]
        A, B = a["P"][idx[:, 0]], b["P"][idx[:, 1]]; c, s = math.cos(th), math.sin(th)
        r = A - (B @ np.array([[c, s], [-s, c]]) + dxy); R.append(r); Z.append(np.maximum(a["z"][idx[:, 0]], b["z"][idx[:, 1]]))
    R = np.concatenate(R); Z = np.concatenate(Z); bins = np.quantile(Z, np.linspace(0, 1, 9)); zc, sc = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (Z >= lo) & (Z < hi); rr = R[m]; sc.append(1.4826 * np.median(np.abs(rr - np.median(rr, 0)))); zc.append(np.median(Z[m]))
    zc, sc = np.array(zc), np.array(sc); A = np.stack([np.ones_like(zc), zc ** 2], 1); p = np.linalg.lstsq(A, sc, rcond=None)[0]
    return dict(SIG0=float(p[0]), SIG2=float(p[1]), z=zc.round(2).tolist(), sigma=sc.round(4).tolist(), n_pts=int(len(Z)))
