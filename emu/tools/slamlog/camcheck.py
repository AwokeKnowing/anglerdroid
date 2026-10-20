"""Sim camera vs twin check: MuJoCo camera poses/fovy vs twin_geometry, rendered depth vs mj_ray at pixel centres.
  PYTHONPATH=. python emu/tools/slamlog/camcheck.py   (reads the live model /dev/shm/kevin_emu_model.mjb; never builds a World)"""
import os, sys, math, time; os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco as mj
from emu import world as W
m = mj.MjModel.from_binary_path("/dev/shm/kevin_emu_model.mjb"); d = mj.MjData(m)
sys.path.append("src"); import twin_geometry as G
mj.mj_forward(m, d)
for _ in range(1500): mj.mj_step(m, d)
bid = [i for i in range(m.nbody) if "base" in (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or "")]
print("bodies with base:", [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) for i in bid])
cid1 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, "rs1_depth"); pb = m.cam_bodyid[cid1]; print("cam parent body", mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, pb))
Rb = d.xmat[pb].reshape(3, 3); pbw = d.xpos[pb].copy()
print("base world pos", pbw.round(5), "base roll/pitch deg", np.degrees([math.atan2(Rb[2,1], Rb[2,2]), -math.asin(Rb[2,0])]).round(4))
print("base z", round(pbw[2], 5), "twin axle h", G.AXLE_H_M)
Rd = mj.Renderer(m, W.DH, W.DW); Rd.enable_depth_rendering()
for cam, (T, R) in (("rs1_depth", (G.RS1_T, G.RS1_R)), ("rs2_ir1", (G.RS2_T, G.RS2_R))):
    c = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, cam)
    Rc = d.cam_xmat[c].reshape(3, 3) @ np.diag([1, -1, -1])           # mujoco cam -> optical
    Rt = Rb @ np.asarray(R); Tt = pbw + Rb @ np.asarray(T)
    dR = Rc.T @ Rt; ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    print(cam, "cam pos err mm", ((d.cam_xpos[c] - Tt) * 1e3).round(3), "rot err deg", round(ang, 4), "fovy", m.cam_fovy[c])
    Rd.update_scene(d, cam); z = Rd.render()
    K = W._intr(W.DW, W.DH, m.cam_fovy[c]); E = []
    for v in range(10, W.DH, 47):
        for u in range(10, W.DW, 53):
            ray = np.array([(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0]); dw = Rt @ ray; n = np.linalg.norm(dw)
            gid = np.zeros(1, np.int32); dist = mj.mj_ray(m, d, Tt, dw / n, None, 1, pb, gid)
            if dist > 0 and z[v, u] > 0: E.append([u, v, z[v, u], dist / n])
    E = np.array(E); r = E[:, 2] - E[:, 3]
    print("  render z - ray z: median %.2f mm, p90 |.| %.2f mm, rel %.5f, n %d" % (np.median(r) * 1e3, np.percentile(abs(r), 90) * 1e3, np.median(r / E[:, 3]), len(E)))
    # fit: horizontal pixel offset? compare left vs right half residual
    L = E[:, 0] < W.DW / 2; print("  left half med %.2f mm, right half med %.2f mm" % (np.median(r[L]) * 1e3, np.median(r[~L]) * 1e3))
