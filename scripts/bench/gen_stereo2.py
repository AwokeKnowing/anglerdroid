"""Offline stereo/RGB-D log generator v2 (fair benchmark): re-render the sim RS2 IR pair + depth along a recorded
GT track at reduced size (424x240, 10 Hz, compressed npz) with the emulator's sensor noise per level.
python gen_stereo2.py ENV RUN_JSON OUT.npz LEVEL[realistic|pessimistic] [sun]
 IR: gaussian read noise (2 DN realistic / 4 DN pessimistic), per-frame gain flicker (0 / +-8 %), 1-px motion-ish blur pessimistic.
 depth: D4xx model as world.py: Z^2*subpix/(f*B) (subpix 0.08 / 0.16), min-Z, left invalid band, edge shadows,
   random holes (0.4 % / 2 %); sun: subpix x2, smooth blob dropouts, 3 % random holes, IR washed (+contrast loss)."""
import sys, os, math, json, numpy as np
env, rj, out, lvl = sys.argv[1:5]; SUN = len(sys.argv) > 5 and sys.argv[5] == "sun"
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["KEVIN_EMU_NOISE"] = lvl
sys.path[:0] = [__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..'))]
import mujoco, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import diskguard
if not diskguard.check(): sys.exit("diskguard: refusing to write " + out)
from emu import world, envs
P = dict(realistic=dict(irs=2.0, gain=0.0, sub=0.08, holes=0.004, blur=0), pessimistic=dict(irs=4.0, gain=0.08, sub=0.16, holes=0.02, blur=1))[lvl]
hz = 10.0; DW, DH = (848, 480) if os.environ.get("GS_FULL") else (424, 240)
e = envs.build(env, 0); m = mujoco.MjModel.from_xml_string(world.compose(e, "stock")); d = mujoco.MjData(m)
R = mujoco.Renderer(m, DH, DW); Rd = mujoco.Renderer(m, DH, DW); Rd.enable_depth_rendering()
I = world._intr(DW, DH, world.FOVY_D); fpx = I["fx"]; I["baseline"] = world.BASELINE
T = json.load(open(rj))["track"]; t = np.array([r["t"] for r in T]); G = np.array([r["gt"] for r in T]); G[:, 2] = np.unwrap(G[:, 2])
ts = np.arange(t[0], t[-1], 1.0 / hz)
sx, sy, sth = e.start; rng = np.random.default_rng(1); W3 = np.array([0.299, 0.587, 0.114], np.float32) * 0.9
mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d); z0 = d.qpos[2]
cols = np.arange(DW, dtype=np.float32)[None, :]; sub = P["sub"] * (2 if SUN else 1)
IR1, IR2, DEP, GT = [], [], [], []
for tt in ts:
    g = [np.interp(tt, t, G[:, i]) for i in range(3)]
    wx = sx + math.cos(sth) * g[0] - math.sin(sth) * g[1]; wy = sy + math.sin(sth) * g[0] + math.cos(sth) * g[1]; th = sth + g[2]
    d.qpos[0], d.qpos[1], d.qpos[2] = wx, wy, z0; d.qpos[3:7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]; mujoco.mj_forward(m, d)
    gain = 1 + P["gain"] * rng.uniform(-1, 1); ims = []
    for cam in ("rs2_ir1", "rs2_ir2"):
        R.update_scene(d, cam); gi = R.render().astype(np.float32) @ W3
        if SUN: gi = 90 + 0.5 * gi                                          # ambient wash: contrast loss
        if P["blur"]: gi = cv2.blur(gi, (3, 1))
        ims.append(np.clip(gi * gain + rng.normal(0, P["irs"], (DH, DW)), 0, 255).astype(np.uint8))
    Rd.update_scene(d, "rs2_ir1"); z = Rd.render(); zz = z + z * z * (sub / (fpx * world.BASELINE)) * rng.standard_normal(z.shape)
    bad = (z < 0.175) | (z > 8) | (rng.random(z.shape) < (0.03 if SUN else P["holes"])) | (cols < fpx * world.BASELINE / np.maximum(z, 1e-3))
    bad[:, 1:] |= np.abs(z[:, 1:] - z[:, :-1]) > 0.08 * z[:, 1:]
    if SUN: bad |= envs._smooth(rng, (DH, DW), 20) > rng.uniform(0.25, 0.45)
    zz[bad] = 0
    IR1.append(ims[0]); IR2.append(ims[1]); DEP.append((zz * 1000).astype(np.uint16)); GT.append(g)
np.savez_compressed(out, t=ts, ir1=np.array(IR1), ir2=np.array(IR2), depth=np.array(DEP), gt=np.array(GT), intr=json.dumps(I), env=env, lvl=lvl, sun=SUN)
print("wrote", out, len(ts), "frames", os.path.getsize(out) // 2**20, "MB")
