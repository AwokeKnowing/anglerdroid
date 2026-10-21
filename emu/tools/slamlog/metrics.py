"""Trajectory + map metrics of an SE(2) estimate against sim ground truth.
Estimate = array [t, x, y, yaw] in any frame; it is anchored like the viewer HUD ('SLAM vs truth'): the estimate's first
pose is mapped onto GT at that time, nothing fitted.  Also reported: the SE(2) least-squares alignment (Umeyama, no scale)
ATE, which separates a consistent-but-rotated map from real drift.  Map metric: RS2 obstacle points (5..90 cm above the
floor) of sampled frames placed with the estimate vs with GT; 'wall_p50/p90' = distance (cm) from each estimated point to
the nearest GT-placed point (a doubled or smeared wall shows up in p90)."""
import numpy as np
from scipy.spatial import cKDTree
from .log import wrap, AXLE_H

def anchor(est, g0):
    t, x, y, a = est.T; dy = g0[2] - a[0]; c, s = np.cos(dy), np.sin(dy)
    X = x - x[0]; Y = y - y[0]
    return np.stack([g0[0] + c * X - s * Y, g0[1] + s * X + c * Y, a + dy], 1)

def umeyama2(P, Q):
    mp, mq = P.mean(0), Q.mean(0); H = (P - mp).T @ (Q - mq); U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T
    return R, mq - R @ mp

def traj(est, log):
    est = np.asarray(est, float); m = (est[:, 0] >= log.gt_t[0]) & (est[:, 0] <= log.gt_t[-1]); est = est[m]
    G = log.gt(est[:, 0]); A = anchor(est, G[0])
    e = np.hypot(*(A[:, :2] - G[:, :2]).T); eh = np.degrees(np.abs(wrap(A[:, 2] - G[:, 2])))
    R, T = umeyama2(est[:, 1:3], G[:, :2]); e2 = np.hypot(*((est[:, 1:3] @ R.T + T) - G[:, :2]).T)
    return dict(n=int(len(est)), dur_s=float(est[-1, 0] - est[0, 0]), ate_rmse_m=float(np.sqrt((e ** 2).mean())), ate_max_m=float(e.max()),
                final_m=float(e[-1]), head_max_deg=float(eh.max()), head_final_deg=float(eh[-1]),
                ate_aligned_rmse_m=float(np.sqrt((e2 ** 2).mean()))), A, G

def wallmap(log, poses_fn, stride=8, every=10):
    fr = log.frames["rs2"][::every]; R, T = log.ext["rs2"]; P = []
    for ts, f in fr:
        p = poses_fn(ts)
        if p is None: continue
        z = log.depth(f)[::stride, ::stride]; h, w = z.shape; K, _ = log.K("rs2", "depth")
        v, u = np.mgrid[0:h, 0:w] * stride; m = (z > 0.2) & (z < 4.0)
        X = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z[m], (v[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1) @ R.T + T
        hg = X[:, 2] + AXLE_H; X = X[(hg > 0.05) & (hg < 0.9)]
        c, s = np.cos(p[2]), np.sin(p[2]); P.append(np.stack([p[0] + c * X[:, 0] - s * X[:, 1], p[1] + s * X[:, 0] + c * X[:, 1]], 1))
    return np.concatenate(P) if P else np.zeros((0, 2))

def mapq(log, A_fn):
    Pg = wallmap(log, lambda t: log.gt(t)); Pe = wallmap(log, A_fn)
    d, _ = cKDTree(Pg).query(Pe); return dict(wall_p50_cm=float(np.percentile(d, 50) * 100), wall_p90_cm=float(np.percentile(d, 90) * 100), wall_p99_cm=float(np.percentile(d, 99) * 100))

def evaluate(est, log, with_map=True):
    r, A, G = traj(est, log); est = np.asarray(est, float); est = est[(est[:, 0] >= log.gt_t[0]) & (est[:, 0] <= log.gt_t[-1])]
    if with_map:
        def fn(t):
            if t < est[0, 0] or t > est[-1, 0]: return None
            return np.array([np.interp(t, est[:, 0], A[:, 0]), np.interp(t, est[:, 0], A[:, 1]), np.interp(t, est[:, 0], np.unwrap(A[:, 2]))])
        r.update(mapq(log, fn))
    return r
