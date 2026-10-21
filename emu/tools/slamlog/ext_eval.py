"""python -m emu.tools.slamlog.ext_eval <logdir> <cam> <exportdir> <posefile> <fmt> [out.json] [--w2c]
Score a third-party SLAM / reconstruction trajectory (camera poses) on a sensor log with the same metrics as baseline/lgfg.
fmt: 'ace0' (poses_final.txt: name qw qx qy qz tx ty tz focal conf; name = export frame id) or 'tum' (t|idx tx ty tz qx qy qz qw;
first column = export frame index if < 1e5, else a stamp; 'tum@F' = index/F, e.g. MASt3R-SLAM image folders).  Poses are camera-to-world unless --w2c (ace0: always world-to-camera).  Camera optical poses are
mapped to base_link with the twin extrinsics (T_b0_bk = T_bc T_c0_ck T_bc^-1, floor-aligned because base z is up), projected
to SE(2) (x, y, yaw), stamped with the export's list.txt stamps, then evaluated (anchored like the HUD + SE(2)-aligned).
Also reports a scale-fitted aligned ATE (for monocular/up-to-scale methods) and the out-of-plane tilt of the estimate."""
import json, os, re, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from .log import Log
from .metrics import traj, mapq, umeyama2
ACE0_CONF = float(os.environ.get("ACE0_CONF", 500))
GAP_S = float(os.environ.get("EXT_GAP_S", 0.5))     # wall map: only frames within GAP_S of an estimated pose (sparse/partial trajectories)

def load(posefile, fmt, stamps):
    T, ts = [], []
    for l in open(posefile):
        p = l.split()
        if not p or p[0].startswith("#"): continue
        if fmt == "ace0":
            if float(p[9]) < ACE0_CONF: continue          # unregistered frame (ACE0's own registration threshold)
            n = int(re.findall(r"(\d+)", os.path.basename(p[0]))[-1]); q = list(map(float, p[1:5])); t = list(map(float, p[5:8]))
            R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        else:
            n = float(p[0]); t = list(map(float, p[1:4])); R = Rot.from_quat(list(map(float, p[4:8]))).as_matrix()
            if fmt.startswith("tum@"): n *= float(fmt[4:])          # 'tum@30': first column = frame index / 30 (MASt3R-SLAM RGBFiles)
            n = int(round(n)) if n < 1e5 else n
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; T.append(M); ts.append(stamps[n] if isinstance(n, int) else n)
    o = np.argsort(ts); return np.array(ts)[o], np.array(T)[o]

def to_base(ts, Tc, ext, w2c=False):
    if w2c: Tc = np.linalg.inv(Tc)
    R, t = ext; Tbc = np.eye(4); Tbc[:3, :3] = R; Tbc[:3, 3] = t
    T0 = np.linalg.inv(Tc[0]); Tb = Tbc @ (T0 @ Tc) @ np.linalg.inv(Tbc)       # base_k in base_0
    yaw = np.unwrap(np.arctan2(Tb[:, 1, 0], Tb[:, 0, 0])); tilt = np.degrees(np.arccos(np.clip(Tb[:, 2, 2], -1, 1)))
    return np.column_stack([ts, Tb[:, 0, 3], Tb[:, 1, 3], yaw]), tilt, Tb[:, 2, 3]

def scaled_aligned(est, L):
    G = L.gt(est[:, 0])[:, :2]; P = est[:, 1:3]; mp, mg = P.mean(0), G.mean(0); R, _ = umeyama2(P, G)
    Pc = (P - mp) @ R.T; s = float((Pc * (G - mg)).sum() / (Pc ** 2).sum()); e = np.hypot(*(s * Pc + mg - G).T)
    return dict(scale=s, ate_aligned_scaled_rmse_m=float(np.sqrt((e ** 2).mean())))

def score(d, cam, exp, posefile, fmt, w2c=False):
    L = Log(d); st = {int(n): float(t) for t, n in (l.split() for l in open(os.path.join(exp, "list.txt")))}
    ts, Tc = load(posefile, fmt, st); est, tilt, z = to_base(ts, Tc, L.ext[cam], w2c)
    r, A, G = traj(est, L); m = (est[:, 0] >= L.gt_t[0]) & (est[:, 0] <= L.gt_t[-1]); e = est[m]; ya = np.unwrap(A[:, 2])
    def fn(t):
        i = np.searchsorted(e[:, 0], t); j = [k for k in (i - 1, i) if 0 <= k < len(e)]
        if not j or min(abs(e[k, 0] - t) for k in j) > GAP_S: return None
        return np.array([np.interp(t, e[:, 0], A[:, 0]), np.interp(t, e[:, 0], A[:, 1]), np.interp(t, e[:, 0], ya)])
    fr = L.frames[cam][::10]; r["coverage"] = float(np.mean([fn(t) is not None for t, _ in fr]))
    r.update(mapq(L, fn)); r.update(scaled_aligned(est, L)); r.update(n_poses=int(len(est)), tilt_p90_deg=float(np.percentile(tilt, 90)), z_p90_m=float(np.percentile(np.abs(z), 90)))
    return r, est

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    r, est = score(a[0], a[1], a[2], a[3], a[4], "--w2c" in sys.argv or a[4] == "ace0"); print(json.dumps(r, indent=1))   # ACE0 writes world-to-camera (COLMAP convention; checked on L3)
    if len(a) > 5: json.dump(r, open(a[5], "w"), indent=1); np.save(a[5][:-5] + "_traj.npy", est)
