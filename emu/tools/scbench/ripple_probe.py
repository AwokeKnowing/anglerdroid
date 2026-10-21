"""Drive the emulator world straight (no robot code) and measure the ride ripple the soft tyres put on the cameras.

  KEVIN_EMU_RIPPLE=<m> python -m emu.tools.scbench.ripple_probe [v=0.3] [secs=6] [env=apartment]
Prints base z / pitch / roll ripple (sd, p-p), RS1 floor depth range seen by its 87x58 deg view corners, and whether the
floor would cross TD_FLOOR_CLIP (false obstacle) or a 4 cm block top would read as floor (missed obstacle), both before
and after the per-frame floor-plane levelling of nav/perceive.Ego._cliff (which removes tilt, not height)."""
import json, math, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")]
import numpy as np
import twin_geometry as G
import tire_profile as TP
from perception.ego_rs1 import TD_FLOOR_CLIP
sys.modules.pop("tire_profile")       # the twin's kevin_drive (imported by emu.world) has its own tire_profile module
sys.path.remove(sys.path[0]) if sys.path[0].endswith("src") else None
from emu import world as W

v = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3; T = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
class A: env = sys.argv[3] if len(sys.argv) > 3 else "apartment"; seed = 0; tire = TP.NAME; ir = 0; rt = 1.0
W.World._setup_io = lambda self: None     # no shm blocks / CAN sockets: never touch a live emulator's /dev/shm
w = W.World(A()); m, d = w.m, w.d; b = w.base
for _ in range(500): w.step(time.monotonic())
for n in w.nodes: n.state = n.CLOSED; n.ax.input_vel = n.dir * v / (2 * math.pi * TP.WHEEL_R_LOADED_M)
t1, R1 = G.frame("rs1_depth"); HF, VF = math.tan(math.radians(43.5)), math.tan(math.radians(29))
rows = []; dt = m.opt.timestep; k_frame = int(round((1 / 30) / dt))
for i in range(int(T / dt)):
    w.step(time.monotonic())
    if i * dt > 1.5 and i % k_frame == 0:          # 30 Hz camera frames after the speed settles
        R = d.xmat[b].reshape(3, 3); p = d.xpos[b]; o = p + R @ t1; Rc = R @ R1
        # floor depth along the 4 view corners + centre (optical z of the floor hit)
        z = []
        for a, c in ((0, 0), (HF, VF), (-HF, VF), (HF, -VF), (-HF, -VF)):
            ray = Rc @ np.array([a, c, 1.0]); s = -o[2] / ray[2]; z.append(s)   # depth (optical z) where the ray meets z=0
        rows.append([p[2], math.degrees(math.asin(-R[2, 0])), math.degrees(math.atan2(R[2, 1], R[2, 2])), *z])
A_ = np.array(rows); zc = A_[:, 3]; zall = A_[:, 3:]
sd = lambda x: float(np.std(x)); pp = lambda x: float(np.ptp(x))
clip = float(TD_FLOOR_CLIP); h1 = G.height("rs1_depth")
out = dict(tire=TP.NAME, env=A.env, noise=W.NOISE, ripple_m=w.rip_a if w.rip else 0.0, v=v, frames=len(A_), collisions=w.collisions,
           base_z_mm=dict(mean=round(1e3 * A_[:, 0].mean(), 3), sd=round(1e3 * sd(A_[:, 0]), 3), pp=round(1e3 * pp(A_[:, 0]), 3)),
           pitch_deg=dict(sd=round(sd(A_[:, 1]), 4), pp=round(pp(A_[:, 1]), 4)), roll_deg=dict(sd=round(sd(A_[:, 2]), 4), pp=round(pp(A_[:, 2]), 4)),
           rs1_floor_depth_centre_mm=dict(min=round(1e3 * zc.min(), 2), max=round(1e3 * zc.max(), 2)),
           rs1_floor_depth_corners_mm=dict(min=round(1e3 * zall.min(), 2), max=round(1e3 * zall.max(), 2)),
           clip_mm=round(1e3 * clip, 1), floor_margin_to_clip_mm_min_raw=round(1e3 * (zall.min() - clip), 2),
           floor_margin_to_clip_mm_min_levelled=round(1e3 * (zc.min() - clip), 2),
           block_4cm_top_depth_max_mm=round(1e3 * (zc.max() - 0.04), 2),
           false_obstacle_frames_raw=int((zall.min(1) < clip).sum()), missed_4cm_block_frames=int((zc - 0.04 > clip).sum()))
print(json.dumps(out))
