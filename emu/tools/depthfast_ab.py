"""A/B of emu.depthcam depth models on IDENTICAL frames (same renderer, pose, noise seed and bank sequence):
old depth() (mujoco Renderer.render float64 linearisation) vs _depth_fast (float32, preallocated).
python emu/tools/depthfast_ab.py [n_frames]  (needs the running sim's /dev/shm model, pose block and depthcam cfg)."""
import os, sys, json, time
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, mujoco as mj
from emu import shm, depthcam
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
p = json.load(open("/dev/shm/kevin_depthcam.json")); m = mj.MjModel.from_binary_path(p["model"]); d = mj.MjData(m)
_, _, po = shm.Block("kevin_campose", depthcam.pose_layout(m)).read()
Rd = mj.Renderer(m, p["DH"], p["DW"]); Rd.enable_depth_rendering()
os.environ["KEVIN_EMU_DEPTH_FAST"] = "0"; old = depthcam.depth_model(mj, m, Rd, p, 7)
os.environ["KEVIN_EMU_DEPTH_FAST"] = "1"; new = depthcam.depth_model(mj, m, Rd, p, 7)
def setpose(yaw):
    d.qpos[:] = po["qpos"]; d.mocap_pos[:] = po["mpos"][:m.nmocap]; d.mocap_quat[:] = po["mquat"][:m.nmocap]
    q = np.array(d.qpos[3:7]); r = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]); out = np.empty(4); mj.mju_mulQuat(out, r, q); d.qpos[3:7] = out
    mj.mj_kinematics(m, d); mj.mj_camlight(m, d)
D, inv_o, inv_n, edge_o, edge_n, mism, To, Tn = [], [], [], [], [], [], [], []
for f in range(N):
    setpose(2 * np.pi * f / N); cam = ("rs1_depth", "rs2_ir1")[f % 2]
    t0 = time.perf_counter(); a = old(d, cam); t1 = time.perf_counter(); b = new(d, cam); t2 = time.perf_counter(); To.append(t1 - t0); Tn.append(t2 - t1)
    va, vb = a > 0, b > 0; inv_o.append(1 - va.mean()); inv_n.append(1 - vb.mean()); mism.append((va != vb).mean())
    both = va & vb; D.append((b[both].astype(np.int32) - a[both].astype(np.int32)))
    # edge behaviour: invalid pixels next to a valid depth discontinuity (> 5 cm) -- occlusion shadow + band
    za = a.astype(np.float32); jump = np.zeros_like(va); jump[:, 1:] = np.abs(np.diff(za, axis=1)) > 50
    edge_o.append((~va & np.roll(jump, -1, 1)).sum()); edge_n.append((~vb & np.roll(jump, -1, 1)).sum())
D = np.concatenate(D)
print("frames %d (RS1/RS2 alternating, 360 deg of robot yaw)" % N)
print("depth diff new-old on pixels valid in both (mm): exact %.5f%%  |diff|<=1 mm %.5f%%  max %d  mean %+.6f" % (100 * np.mean(D == 0), 100 * np.mean(np.abs(D) <= 1), np.abs(D).max(), D.mean()))
print("invalid fraction old %.4f%% new %.4f%%  validity mismatch %.5f%% of pixels" % (100 * np.mean(inv_o), 100 * np.mean(inv_n), 100 * np.mean(mism)))
print("invalid next to depth edges: old %d new %d" % (sum(edge_o), sum(edge_n)))
print("time per frame (incl. render): old p50 %.1f ms  new p50 %.1f ms" % (1e3 * np.median(To[2:]), 1e3 * np.median(Tn[2:])))
