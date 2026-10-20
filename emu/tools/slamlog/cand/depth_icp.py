"""Depth-only planar scan matching (both RealSense depths), seeded and gated by wheel+gyro odometry.
  python -m emu.tools.slamlog.cand.depth_icp <logdir> <out.json> [cams=rs1,rs2] [stride_px=6]
Per frameset (RS2 frame stamps; RS1 = nearest frame): depth -> points in base_link (twin extrinsics) -> keep 0.05..1.6 m
above the axle plane's floor band (walls/furniture; floor removed), <= 4 m range -> 2-D scan (x, y).  Scan-to-keyframe
point-to-point ICP in SE(2) (KD-tree, Huber-weighted, 15 iterations), initial guess = odometry increment since the
keyframe.  A result is accepted only if the correction to the odometry increment is inside a motion-dependent bound
(ASSUMED odometry error model: 2 cm + 5 % of the distance, 1 deg + 5 % of the rotation, since the keyframe -- about 3x
the sim's realistic wheel/track error) and >= 50 % of points match within 5 cm; otherwise odometry is used.
New keyframe after 0.25 m or 20 deg.  Planar by construction.  CPU only.  Scored with metrics.evaluate (same as baseline)."""
import sys, json, time, math, numpy as np, cv2
from scipy.spatial import cKDTree
from ..log import Log
from ..metrics import evaluate
d, out = sys.argv[1:3]; cams = (sys.argv[3] if len(sys.argv) > 3 else "rs1,rs2").split(","); S = int(sys.argv[4]) if len(sys.argv) > 4 else 6
L = Log(d); N = L.nav(); Kd, _ = L.K("rs2", "depth")
t_od, od = N["t_od"], N["odom"]; th_od = np.unwrap(od[:, 2])
DT = float(__import__("os").environ.get("DICP_DT", "0"))   # depth stamp -> odom time offset probe (s)
def odom(t): t = t + DT; return np.array([np.interp(t, t_od, od[:, 0]), np.interp(t, t_od, od[:, 1]), np.interp(t, t_od, th_od)])
def se2(p): c, s = math.cos(p[2]), math.sin(p[2]); return np.array([[c, -s, p[0]], [s, c, p[1]], [0, 0, 1.0]])
def inv(T): R = T[:2, :2]; Ti = np.eye(3); Ti[:2, :2] = R.T; Ti[:2, 2] = -R.T @ T[:2, 2]; return Ti
def par(T): return np.array([T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])])
H, W = 480, 848; vv, uu = np.mgrid[0:H:S, 0:W:S]; rx = (uu - Kd[0, 2]) / Kd[0, 0]; ry = (vv - Kd[1, 2]) / Kd[1, 1]
from twin_geometry import AXLE_H_M
def scan(cam, f):
    z = L.depth(f)[0:H:S, 0:W:S]; m = (z > 0.15) & (z < 4.0)
    X = np.stack([rx[m] * z[m], ry[m] * z[m], z[m]], 1); R, T = L.ext[cam]; B = X @ R.T + T     # base_link (axle height origin)
    h = B[:, 2] + AXLE_H_M; B = B[(h > 0.05) & (h < 1.6)]
    return B[:, :2]
def icp(src, dst_tree, dst, T0, it=15):
    T = T0.copy(); ok = 0.0
    for _ in range(it):
        P = src @ T[:2, :2].T + T[:2, 2]; dd, j = dst_tree.query(P, distance_upper_bound=0.3); m = np.isfinite(dd)
        if m.sum() < 30: return T, 0.0
        p, q = P[m], dst[j[m]]; w = np.minimum(1.0, 0.05 / np.maximum(dd[m], 1e-6))          # Huber
        mp, mq = (w[:, None] * p).sum(0) / w.sum(), (w[:, None] * q).sum(0) / w.sum()
        Hm = ((p - mp) * w[:, None]).T @ (q - mq); a = math.atan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
        c, s = math.cos(a), math.sin(a); dT = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]); dT[:2, 2] = mq - dT[:2, :2] @ mp
        T = dT @ T
        if abs(a) < 1e-4 and np.linalg.norm(dT[:2, 2]) < 1e-4: break
    P = src @ T[:2, :2].T + T[:2, 2]; dd, _ = dst_tree.query(P, distance_upper_bound=0.05); ok = float(np.isfinite(dd).mean())
    return T, ok
fr2 = L.frames["rs2"]; fr1 = L.frames.get("rs1", []); t1 = np.array([t for t, _ in fr1]) if fr1 else None
t0w = time.time(); est = []; kf = None; n_acc = n_rej = 0; diag = []
Tw = se2(odom(fr2[0][0]))
for t, f in fr2:
    pts = []
    if "rs2" in cams: pts.append(scan("rs2", f))
    if "rs1" in cams and t1 is not None: pts.append(scan("rs1", fr1[int(np.argmin(np.abs(t1 - t)))][1]))
    sc = np.concatenate(pts) if pts else np.zeros((0, 2)); o = odom(t)
    if kf is None: kf = dict(t=t, T=Tw.copy(), o=o, sc=sc, tree=cKDTree(sc) if len(sc) else None); est.append([t, *par(Tw)]); continue
    dO = inv(se2(kf["o"])) @ se2(o); dist = np.linalg.norm(dO[:2, 2]); rot = abs(par(dO)[2])
    dT = dO
    if kf["tree"] is not None and len(sc) > 50:
        Ti, ok = icp(sc, kf["tree"], kf["sc"], dO); c = par(inv(dO) @ Ti)
        if ok >= 0.5 and np.linalg.norm(c[:2]) <= 0.02 + 0.05 * dist and abs(c[2]) <= math.radians(1) + 0.05 * rot: dT = Ti; n_acc += 1
        else: n_rej += 1
        g0, g1 = L.gt(kf["t"]), L.gt(t); cg = par(inv(dO) @ inv(se2(g0)) @ se2(g1)); diag.append([t, dist, rot, ok, *c, *cg])
    Tw = kf["T"] @ dT; est.append([t, *par(Tw)])
    if dist > 0.25 or rot > math.radians(20): kf = dict(t=t, T=Tw.copy(), o=o, sc=sc, tree=cKDTree(sc) if len(sc) else None)
wall = time.time() - t0w; est = np.array(est); est[:, 3] = np.unwrap(est[:, 3])
r = evaluate(est, L); dur = est[-1, 0] - est[0, 0]
r.update(method="depth_icp", cams=cams, stride_px=S, wall_s=round(wall, 1), ms_per_frame=round(1000 * wall / len(est), 1), rtf=round(dur / wall, 2), icp_accepted=n_acc, icp_rejected=n_rej)
np.save(out[:-5] + '_diag.npy', np.array(diag)); json.dump(r, open(out, "w"), indent=1); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, dict))})
