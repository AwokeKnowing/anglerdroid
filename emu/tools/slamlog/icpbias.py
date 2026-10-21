"""Depth-ICP bias test on a log: ICP from the GT increment on logged depth vs noise-free renders at GT poses (L8 hard-coded).
  PYTHONPATH=. python emu/tools/slamlog/icpbias.py"""
import os, sys, math; os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco as mj
from scipy.spatial import cKDTree
m = mj.MjModel.from_binary_path("/dev/shm/kevin_emu_model.mjb"); d = mj.MjData(m)
sys.path.append("src"); import twin_geometry as G
from emu.tools.slamlog.log import Log
L = Log(os.path.expanduser("~/slamlogs/L8")); Kd, _ = L.K("rs2", "depth"); H, W = 480, 848; S = 10
vv, uu = np.mgrid[0:H:S, 0:W:S]; rx = (uu - Kd[0, 2]) / Kd[0, 0]; ry = (vv - Kd[1, 2]) / Kd[1, 1]
R2 = mj.Renderer(m, H, W); R2.enable_depth_rendering()
def se2(p): c, s = math.cos(p[2]), math.sin(p[2]); return np.array([[c, -s, p[0]], [s, c, p[1]], [0, 0, 1.0]])
def inv(T): R = T[:2, :2]; Ti = np.eye(3); Ti[:2, :2] = R.T; Ti[:2, 2] = -R.T @ T[:2, 2]; return Ti
def par(T): return np.array([T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])])
def toscan(z, cam):
    z = z[0:H:S, 0:W:S]; mk = (z > 0.15) & (z < 4.0); X = np.stack([rx[mk] * z[mk], ry[mk] * z[mk], z[mk]], 1)
    R, T = L.ext[cam]; B = X @ R.T + T; h = B[:, 2] + G.AXLE_H_M; return B[(h > 0.05) & (h < 1.6)][:, :2]
def render(p, cam="rs2_ir1"):
    d.qpos[:] = m.qpos0; d.qpos[0], d.qpos[1], d.qpos[2] = p[0], p[1], G.AXLE_H_M; d.qpos[3:7] = (math.cos(p[2] / 2), 0, 0, math.sin(p[2] / 2))
    mj.mj_forward(m, d); R2.update_scene(d, cam); return R2.render()
def icp(src, dst, T0, it=30):
    tr = cKDTree(dst); T = T0.copy()
    for _ in range(it):
        P = src @ T[:2, :2].T + T[:2, 2]; dd, j = tr.query(P, distance_upper_bound=0.3); mk = np.isfinite(dd)
        p, q = P[mk], dst[j[mk]]; w = np.minimum(1.0, 0.05 / np.maximum(dd[mk], 1e-6))
        mp, mq = (w[:, None] * p).sum(0) / w.sum(), (w[:, None] * q).sum(0) / w.sum()
        Hm = ((p - mp) * w[:, None]).T @ (q - mq); a = math.atan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
        c, s = math.cos(a), math.sin(a); dT = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]); dT[:2, 2] = mq - dT[:2, :2] @ mp; T = dT @ T
    return T
fr = L.frames["rs2"]; out = []
for i in range(0, len(fr) - 10, 20):
    (ta, fa), (tb, fb) = fr[i], fr[i + 10]; ga, gb = L.gt(ta), L.gt(tb); dG = inv(se2(ga)) @ se2(gb)
    Ta = icp(toscan(L.depth(fb), "rs2"), toscan(L.depth(fa), "rs2"), dG)      # logged (noisy) depth, GT init
    Tr = icp(toscan(render(gb), "rs2"), toscan(render(ga), "rs2"), dG)        # noise-free render at GT poses
    out.append([par(inv(dG) @ Ta)[2], par(inv(dG) @ Tr)[2], par(dG)[2], np.linalg.norm(par(inv(dG) @ Ta)[:2]), np.linalg.norm(par(inv(dG) @ Tr)[:2])])
O = np.degrees(np.array(out)[:, :3]); X = np.array(out)[:, 3:] * 1e3
print("pairs", len(O)); print("logged depth : yaw err mean %.3f med %.3f p90|.| %.3f deg; xy err med %.1f mm" % (O[:, 0].mean(), np.median(O[:, 0]), np.percentile(abs(O[:, 0]), 90), np.median(X[:, 0])))
print("render (clean): yaw err mean %.3f med %.3f p90|.| %.3f deg; xy err med %.1f mm" % (O[:, 1].mean(), np.median(O[:, 1]), np.percentile(abs(O[:, 1]), 90), np.median(X[:, 1])))
for nm, k in (("logged", 0), ("render", 1)): print(nm, "slope err vs gt yaw step %.4f" % np.polyfit(O[:, 2], O[:, k], 1)[0])
# also: logged vs render depth at the same GT pose (does the logged frame match the GT pose?)
r = []
for i in range(0, len(fr), 100):
    t, f = fr[i]; zl = L.depth(f); zr = render(L.gt(t)); mk = (zl > 0.2) & (zr > 0.2) & (zr < 4)
    r.append([np.median(zl[mk] - zr[mk]) * 1e3, np.median((zl[mk] - zr[mk]) / zr[mk]), np.mean(np.abs(zl[mk] - zr[mk]) < 0.02)])
print("logged - render depth at GT pose: median mm per frame", np.round([x[0] for x in r], 1).tolist()); print(" rel", np.round([x[1] for x in r], 4).tolist(), " frac<2cm", np.round([x[2] for x in r], 2).tolist())
