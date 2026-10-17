"""Jetson emulator world: MuJoCo Kevin (twin lite model) in an emu/envs environment, real-time, exposing the
robot's *hardware* to unmodified robot code via the HAL shims (hal/sim):

  RS1  top-down D435   depth 848x480 z16 + color 640x480 @30 Hz      -> shm kevin_rs1
  RS2  forward D435i   depth + color + IR1/IR2 y8 @30 Hz, IMU 200 Hz  -> shm kevin_rs2, kevin_imu
  ODrive S1 x2         CANSimple over UDP (node 0 left, node 1 right): set_input_vel, axis state, SDO
                       read/write (flat_endpoints.json ids), watchdog, cyclic 0x09 encoder estimates @100 Hz
  ground truth         shm kevin_gt (pose, twist, collisions) + /dev/shm/kevin_gt_map.npz  (eval tools only)
  control              UDP JSON on :47801  {"cmd": "teleport"|"obstacle"|"clear_obstacles"|"pause"|"resume"}

  python -m emu.world --env apartment [--seed 0] [--tire stock|soft_v3_2] [--ir 1] [--rt 1.0]
"""
from __future__ import annotations
import argparse, json, math, os, socket, struct, sys, threading, time
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from emu import shm, envs  # noqa: E402

TWIN = os.environ.get("KEVIN_TWIN", os.path.expanduser("~/kevin-urdf-twin-serve/kevin-urdf-twin/simready"))
CAN_PORT = int(os.environ.get("KEVIN_CAN_PORT", "47800")); CTL_PORT = CAN_PORT + 1
TWO_PI = 2 * math.pi
DW, DH, CW, CH = 848, 480, 640, 480
FOVY_D, FOVY_C = 58.0, 42.0
VIEW = os.environ.get("KEVIN_EMU_VIEW") == "1"; VIEW_DIR = os.environ.get("KEVIN_EMU_VIEW_DIR"); VIEW_W, VIEW_H = 960, 540
if VIEW_DIR: os.makedirs(VIEW_DIR, exist_ok=True)
#  image slice depth(m), tex w,h, fovy, aspect, frustum line colour.  Viewer-only: the live image fills the frustum
#  cross-section at the given depth (8-10 cm from the lens), plus thin translucent lines for the full FOV out to the near/far clip range (VIEW_CLIP).
VIEW_CAMS = {"rs1_depth": (0.10, 212, 120, FOVY_D, DW / DH, (0.1, 0.9, 1.0)),   # mast D435 top-down: colorized depth
             "rs1_color": (0.09, 160, 120, FOVY_C, CW / CH, (1.0, 0.85, 0.1)),  # mast D435 RGB
             "rs2_ir1":   (0.10, 212, 120, FOVY_D, DW / DH, (0.1, 0.9, 1.0)),   # forward D435i: colorized depth
             "rs2_color": (0.09, 160, 120, FOVY_C, CW / CH, (1.0, 0.85, 0.1))}  # forward D435i RGB
VIEW_CLIP = (0.2, 1.2)   # drawn FOV range (m): D435 minimum depth ~0.2 m .. 1.2 m to keep the scene readable
FPS = 30
# Mounts = twin URDF optical frames (urdf/kevin.urdf, base_link = axle centre, x fwd, y left, z up):
#   realsense_d435_depth_optical  (-0.007, 0, 0.8619)  looking straight down, x_opt = robot-backward
#   realsense_d435_color_optical  (+0.008, 0, 0.8619)
#   realsense_d435i_depth_optical (-0.1301, 0, 0.3822) pitched 27 deg down; infra2 at y -0.050; color y +0.015
#   realsense_d435i_imu_optical   (-0.1382, -0.0055, 0.3921)
# NB: the stack's vision.TD_X_OFFSET (-75 px) implies the RS1 centre 4 cm ahead of the axle; the twin says 0.7 cm behind.
RS1_POS = (-0.007, 0.0, 0.8619); RS1C_POS = (0.008, 0.0, 0.8619)
# mast RGB (USB16MP01 board camera at the tip of beam_mast_top; twin frames.rgb, optical z = base +x, pitch 0).  Rendered
# at MAST_HZ into shm kevin_mast for the VLN (LightNav-0).  The twin has no lens FOV: KEVIN_MAST_HFOV (deg) until measured.
MAST_POS = (0.0291, 0.0, 0.8905); MW, MH = 640, 480; MAST_HFOV = float(os.environ.get("KEVIN_MAST_HFOV", "100"))
MAST_FOVY = math.degrees(2 * math.atan(math.tan(math.radians(MAST_HFOV) / 2) * MH / MW)); MAST_HZ = float(os.environ.get("KEVIN_MAST_HZ", "10"))
RS2_POS = (-0.1301, 0.0, 0.3822); RS2C_DY = 0.015; RS2_IMU = (-0.1382, -0.0055, 0.3921)
RS2_PITCH = math.radians(27.0); BASELINE = 0.050
# true (as-mounted) forward D435i pose: nominal unless KEVIN_EMU_RS2_RAND=1 (per seed: pitch U(24,30) deg, base roll
# U(-1,1) deg, height U(-1,1) cm) or KEVIN_EMU_RS2_PITCH/ROLL/DZ given.  Robot code keeps believing the twin's 27 deg.
RS2_TRUE = dict(pitch_deg=27.0, roll_deg=0.0, dz=0.0)
def set_rs2_mount(seed):
    if os.environ.get("KEVIN_EMU_RS2_RAND", "0") == "1":
        r = np.random.default_rng(seed + 101)
        RS2_TRUE.update(pitch_deg=float(r.uniform(24, 30)), roll_deg=float(r.uniform(-1, 1)), dz=float(r.uniform(-0.01, 0.01)))
    for k, e in (("pitch_deg", "KEVIN_EMU_RS2_PITCH"), ("roll_deg", "KEVIN_EMU_RS2_ROLL"), ("dz", "KEVIN_EMU_RS2_DZ")):
        if e in os.environ: RS2_TRUE[k] = float(os.environ[e])
def _rs2_axes(vx, vy):
    """camera/site xyaxes under the true mount: nominal axes built at the true pitch, then rolled about base x."""
    a = math.radians(RS2_TRUE["roll_deg"]); c, s_ = math.cos(a), math.sin(a)
    rx = lambda v: (v[0], c * v[1] - s_ * v[2], s_ * v[1] + c * v[2])
    return "%g %g %g %g %g %g" % (rx(vx) + rx(vy))
N_OBS = 6


def _intr(w, h, fovy):
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    return dict(w=w, h=h, fx=f, fy=f, ppx=(w - 1) / 2, ppy=(h - 1) / 2)


# ------------------------------------------------------------------ model composition
# ------------------------------------------------------------------ sensor / odometry error model
# KEVIN_EMU_NOISE = ideal | realistic | pessimistic   (per-episode randomised from --seed)
# realistic: BMI055 (D435i IMU) datasheet - gyro noise density 0.014 deg/s/sqrtHz (0.003 rad/s per 200 Hz sample),
#   zero-rate offset up to +-1 deg/s (we assume the ZUPT already removes most: residual 0.2 deg/s), sensitivity error
#   ~1 %, bias drift as a random walk; accel noise 150 ug/sqrtHz (~0.03 m/s^2 @200 Hz).  Diff-drive odometry:
#   systematic wheel-radius error ~0.5 %, effective-track error 1-2 % (tire compliance, pivot skid; Borenstein UMBmark).
# pessimistic: 3x the systematic terms, 1 deg/s residual bias, faster bias walk + thermal ramp, 10 ms IMU-camera
#   offset, slick floor (tire friction x0.6 -> real skid in pivots / on contact).
NOISE_LEVELS = {
    "ideal":       dict(floor=0, r_err=0.0,   tr_err=0.0,  quant=0,     g_white=0.003, g_bias=0.002,  g_scale=0.0,  g_rw=0.0,    g_drift=0.0,  a_white=0.03, imu_dt=0.0,   cam_jit=0.0,    tire_mu=1.0, can_jit=0.0),
    "realistic":   dict(floor=1, r_err=0.005, tr_err=0.015, quant=16384, g_white=0.003, g_bias=0.0035, g_scale=0.01, g_rw=3e-5,  g_drift=0.0,  a_white=0.03, imu_dt=0.003, cam_jit=0.001,  tire_mu=1.0, can_jit=0.002),
    "pessimistic": dict(floor=2, r_err=0.015, tr_err=0.04,  quant=16384, g_white=0.006, g_bias=0.017,  g_scale=0.03, g_rw=1.5e-4, g_drift=2e-6, a_white=0.06, imu_dt=0.010, cam_jit=0.003,  tire_mu=0.8, can_jit=0.005),
}
NOISE = os.environ.get("KEVIN_EMU_NOISE", "ideal")
NP = NOISE_LEVELS[NOISE]


def _floor_model(root, asset, wb, env, level):
    """Ground model.  Indoor (level 1 realistic, 2 pessimistic): heightfield with lumpy carpet (+-1.5/3 mm,
    10-30 cm) + sagged/bowed boards (+-3/5 mm over ~1 m) and rounded threshold strips (6-12 mm) in every doorway.
    Outdoor envs (env.terrain) always: their own terrain function on a 1 cm grid (curbs, ramps, joints, potholes,
    gravel, lawn bumps), env.ground = (friction, rolling resistance, condim, material), visual-only grass blades.
    The heightfield is shifted so the ground under the start pose is z=0 (spawn height unchanged).
    Not 'env*' geoms -> driving over them is not a collision (drops are counted via env.hazard)."""
    rng = np.random.default_rng(int(sum(map(ord, env.name))) + 17 * level)
    x0, y0, x1, y1 = env.extent
    outdoor = hasattr(env, "terrain"); res = float(os.environ.get("KEVIN_EMU_HFRES", 0.02 if outdoor else (0.05 if getattr(env, "hard_zones", None) else 0.025)))   # 1 cm: 630k-vertex hfield -> RS render 1.4 Hz
    nc, nr = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    xx, yy = np.meshgrid(np.linspace(x0, x1, nc), np.linspace(y0, y1, nr))
    if outdoor:
        h = env.terrain(xx, yy, rng).astype(np.float64)
        mu, roll, condim, mat = env.ground
    elif hasattr(env, "hard_zones"):
        # furnished house: old lumpy carpet (12 mm pile, +-3 mm lumps over 10-20 cm, +-4 mm sag over ~1 m; high rolling
        # resistance) everywhere except hard-floor zones (tile plates added below -> ~1 cm threshold at each edge)
        hard = np.zeros(xx.shape, bool)
        for a0, b0, a1, b1 in env.hard_zones: hard |= (xx >= a0) & (xx <= a1) & (yy >= b0) & (yy <= b1)
        h = 0.004 * envs._smooth(rng, xx.shape, 1.0 / res) + np.where(hard, 0.0, 0.012 + 0.003 * envs._smooth(rng, xx.shape, 0.15 / res))
        mu, roll, condim, mat = 0.9, 0.025, 6, "carpet"
    else:
        lump, sag = (0.0015, 0.003) if level == 1 else (0.003, 0.005)
        h = lump * envs._smooth(rng, xx.shape, 0.2 / res) + sag * envs._smooth(rng, xx.shape, 1.0 / res)
        # hard floor (sealed wood / tile): mu 0.7 realistic, 0.5 pessimistic (polished tile, dusty); rugs below
        mu, roll, condim, mat = (0.7, 0.002, 3, None) if level == 1 else (0.5, 0.002, 3, None)
    sr, sc = int(round((env.start[1] - y0) / res)), int(round((env.start[0] - x0) / res))
    # spawn on top of the terrain: reference z=0 to the HIGHEST point under the robot footprint (r 0.35 m), so no
    # wheel/caster starts inside a lawn bump (stock tire on lawn was ejected 1.4 m at spawn and lodged: 0 m driven)
    k = int(0.35 / res); sub = h[max(sr - k, 0):sr + k + 1, max(sc - k, 0):sc + k + 1]
    h -= float(sub.max()) if sub.size else h[np.clip(sr, 0, nr - 1), np.clip(sc, 0, nc - 1)]
    lo = float(h.min()); span = max(float(h.max()) - lo, 1e-4)
    hf = ET.SubElement(asset, "hfield", name="floor_hf", nrow=str(nr), ncol=str(nc),
                       size="%g %g %g %g" % ((x1 - x0) / 2, (y1 - y0) / 2, span, 0.05))
    hf.set("elevation", " ".join("%.4f" % v for v in ((h - lo) / span)[::-1].ravel()))   # MuJoCo hfield row 0 = +y edge
    ET.SubElement(wb, "geom", name="floor_uneven", type="hfield", hfield="floor_hf",
                  material=("m_" + mat) if mat else "floor", pos="%g %g %g" % ((x0 + x1) / 2, (y0 + y1) / 2, lo),
                  contype="1", conaffinity="3", condim=str(condim), friction="%g 0.005 %g" % (mu, roll))
    for i, (a0, b0, a1, b1) in enumerate(getattr(env, "hard_zones", [])):     # hard floor plates (tile), flat
        r0, r1 = int((b0 - y0) / res), int((b1 - y0) / res) + 1; c0, c1 = int((a0 - x0) / res), int((a1 - x0) / res) + 1
        top = float(h[max(r0, 0):r1, max(c0, 0):c1].max()) + 0.0005
        ET.SubElement(wb, "geom", name="floor_hard%d" % i, type="box", size="%g %g 0.01" % ((a1 - a0) / 2, (b1 - b0) / 2),
                      pos="%g %g %g" % ((a0 + a1) / 2, (b0 + b1) / 2, top - 0.01), material="m_tile",
                      contype="1", conaffinity="3", condim="3", friction="0.7 0.005 0.002")
    fl = root.find(".//geom[@name='floor']")
    if fl is not None: fl.set("pos", "0 0 %g" % (lo - 0.06))           # keep the plane below the hfield
    for i, (x, y, yaw, L) in enumerate(getattr(env, "thresholds", []) if level else []):
        hh = rng.uniform(0.006, 0.012); r = 0.015                            # rounded strip, 6-12 mm proud
        dx, dy = math.cos(yaw) * L / 2, math.sin(yaw) * L / 2
        zc = float(h[np.clip(int((y - y0) / res), 0, nr - 1), np.clip(int((x - x0) / res), 0, nc - 1)]) + hh - r
        ET.SubElement(wb, "geom", name="floor_thr%d" % i, type="capsule", size="%g" % r,
                      fromto="%g %g %g %g %g %g" % (x - dx, y - dy, zc, x + dx, y + dy, zc),
                      material="m_dark", contype="1", conaffinity="3", friction="0.8 0.005 0.0001")
    nrug = 0
    if not outdoor and level:                                                # carpet / rugs on some places: higher grip,
        for i, (lbl, (px, py)) in enumerate(sorted(env.places.items())):     # high rolling resistance, 6-10 mm edge
            if rng.random() > 0.45: continue
            hx, hy, t = rng.uniform(0.6, 1.0), rng.uniform(0.45, 0.8), rng.uniform(0.003, 0.005)
            zc = float(h[np.clip(int((py - y0) / res), 0, nr - 1), np.clip(int((px - x0) / res), 0, nc - 1)]) + t
            ET.SubElement(wb, "geom", name="floor_rug%d" % i, type="box", size="%g %g %g" % (hx, hy, t), pos="%g %g %g" % (px, py, zc),
                          euler="0 0 %g" % rng.uniform(0, 3.14), rgba="%g %g %g 1" % tuple(rng.uniform(0.25, 0.6, 3)),
                          contype="1", conaffinity="3", condim="6", friction="1.0 0.005 %g" % rng.uniform(0.01, 0.03))
            nrug += 1
    for i, (x, y, bh) in enumerate(getattr(env, "blades", [])):                # grass tufts: seen by depth, not by wheels
        zc = float(h[np.clip(int((y - y0) / res), 0, nr - 1), np.clip(int((x - x0) / res), 0, nc - 1)])
        ET.SubElement(wb, "geom", name="floor_blade%d" % i, type="box", size="0.006 0.02 %g" % (bh / 2),
                      pos="%g %g %g" % (x, y, zc + bh / 2), euler="%g %g %g" % (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, 3.14)),
                      material="m_grass", contype="0", conaffinity="0")
    print("emu.world: ground %s level %d: hfield %dx%d span %.1f mm mu %.2f roll %.3f, %d thresholds, %d rugs, %d blades" % (
        "outdoor" if outdoor else "indoor", level, nc, nr, span * 1e3, mu, roll, len(getattr(env, "thresholds", [])) if level else 0, nrug, len(getattr(env, "blades", []))), flush=True)


def compose(env: envs.Env, tire: str) -> str:
    kind = os.environ.get("KEVIN_EMU_MODEL", "lite")              # 'full' = full-detail twin (video / viewer)
    fn = "kevin_%s.xml" % kind if tire == "stock" else "kevin_%s_%s.xml" % (kind, tire)
    root = ET.parse(os.path.join(TWIN, fn)).getroot()
    comp = root.find("compiler")
    comp.set("meshdir", os.path.join(TWIN, comp.get("meshdir", "assets"))); comp.set("texturedir", os.path.join(TWIN, comp.get("texturedir", "assets")))
    asset = root.find("asset"); wb = root.find("worldbody")
    tdir = envs.ensure_textures()
    for m in envs.MATS:
        ET.SubElement(asset, "texture", name="t_" + m, type="2d", file=os.path.join(tdir, m + ".png"))
        ET.SubElement(asset, "material", name="m_" + m, texture="t_" + m, texrepeat="2 2", texuniform="true", specular="0.1", shininess="0.1")
    for i, g in enumerate(env.geoms):
        q = "%g 0 0 %g" % (math.cos(g.yaw / 2), math.sin(g.yaw / 2))
        a = dict(name="env%d" % i, pos="%g %g %g" % g.pos, quat=q, material="m_" + g.mat, contype="1", conaffinity="2", group="0", friction="0.6 0.005 0.0001")
        if g.kind == "box": ET.SubElement(wb, "geom", type="box", size="%g %g %g" % g.size, **a)
        elif g.kind == "cyl": ET.SubElement(wb, "geom", type="cylinder", size="%g %g" % g.size, **a)
        elif g.kind == "mesh":
            ET.SubElement(asset, "mesh", name="envmesh%d" % i, file=g.mesh, scale="%g %g %g" % g.size)
            if hasattr(env, "wall_occ"): a.update(contype="0", conaffinity="0")   # MuJoCo meshes collide as convex hulls
            ET.SubElement(wb, "geom", type="mesh", mesh="envmesh%d" % i, **a)
    if hasattr(env, "wall_occ"):                     # wall collision = run-length boxes of the wall section raster
        (ox, oy), r = env.gt_origin, envs.GT_RES; nb = 0
        zs = [z for z, _ in env.wall_levels]; edges = [0.0] + [(a + b) / 2 for a, b in zip(zs[:-1], zs[1:])] + [2.3]
        for li, (zc, occ) in enumerate(env.wall_levels):
            za, zb = edges[li], edges[li + 1]; open_ = {}; boxes = []
            for j in range(occ.shape[0] + 1):
                row = occ[j] if j < occ.shape[0] else np.zeros(occ.shape[1], bool)
                d = np.diff(np.r_[0, row.astype(np.int8), 0]); runs = set(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
                for k in list(open_):
                    if k not in runs: boxes.append((k, open_.pop(k), j))
                for k in runs: open_.setdefault(k, j)
            for (c0, c1), j0, j1 in boxes:
                ET.SubElement(wb, "geom", name="envwall%d" % nb, type="box", size="%g %g %g" % ((c1 - c0) * r / 2, (j1 - j0) * r / 2, (zb - za) / 2),
                              pos="%g %g %g" % (ox + (c0 + c1) * r / 2, oy + (j0 + j1) * r / 2, (za + zb) / 2), rgba="0 0 0 0", group="3",
                              contype="1", conaffinity="2", friction="0.6 0.005 0.0001"); nb += 1
        print("emu.world: %d wall collision boxes (%d height slabs)" % (nb, len(zs)))
    if NP["floor"] or os.environ.get("KEVIN_EMU_FLOOR") or hasattr(env, "terrain") or hasattr(env, "hard_zones"):
        _floor_model(root, asset, wb, env, int(os.environ.get("KEVIN_EMU_FLOOR", NP["floor"])))
    if getattr(env, "dog", False):  # yorkie ~25 cm long, ~20 cm tall (mocap, soft contact); geoms named dog_*
        db = ET.SubElement(wb, "body", name="dog", mocap="true", pos="%g %g 0" % (env.places.get("hallway", (1, 1))))
        sa = dict(contype="1", conaffinity="2", solref="0.03 1.5", solimp="0.6 0.9 0.01")
        ET.SubElement(db, "geom", name="dog_body", type="capsule", size="0.055", fromto="-0.07 0 0.13 0.06 0 0.135", material="m_fur2", **sa)
        ET.SubElement(db, "geom", name="dog_head", type="sphere", size="0.048", pos="0.11 0 0.175", material="m_fur", **sa)
        ET.SubElement(db, "geom", name="dog_snout", type="capsule", size="0.018", fromto="0.14 0 0.165 0.17 0 0.16", material="m_fur", **sa)
        for ex in (-1, 1): ET.SubElement(db, "geom", name="dog_ear%d" % (ex > 0), type="box", size="0.012 0.008 0.022", pos="0.105 %g 0.225" % (0.028 * ex), material="m_fur2", **sa)
        for lx in (-0.07, 0.07):
            for ly in (-0.03, 0.03): ET.SubElement(db, "geom", name="dog_leg%d%d" % (lx > 0, ly > 0), type="capsule", size="0.013", fromto="%g %g 0.11 %g %g 0.012" % (lx, ly, lx, ly), material="m_fur", **sa)
        ET.SubElement(db, "geom", name="dog_tail", type="capsule", size="0.01", fromto="-0.11 0 0.15 -0.15 0 0.2", material="m_fur2", **sa)
    for k in range(N_OBS):          # mocap obstacles for adversarial tests (parked below the floor)
        b = ET.SubElement(wb, "body", name="obs%d" % k, mocap="true", pos="0 0 -5")
        ET.SubElement(b, "geom", type="box", size="0.15 0.15 0.35", pos="0 0 0.35", material="m_dark", contype="1", conaffinity="2")
    for lx, ly in ((0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)):
        x0, y0, x1, y1 = env.extent
        ET.SubElement(wb, "light", pos="%g %g 2.3" % (x0 + lx * (x1 - x0), y0 + ly * (y1 - y0)), dir="0 0 -1", diffuse="0.18 0.18 0.17", castshadow="false")
    base = wb.find("body[@name='base_link']")
    # cameras: MuJoCo looks along -z with +y up; RS optical is z fwd, y down -> xyaxes = (x_opt, -y_opt)
    ET.SubElement(base, "camera", name="rs1_depth", pos="%g %g %g" % RS1_POS, xyaxes="-1 0 0 0 -1 0", fovy=str(FOVY_D))
    # top-down D435 IR pair for the cuVSLAM dual rig: IR1 = depth origin, IR2 = +50 mm along optical x (= base -x)
    ET.SubElement(base, "camera", name="rs1_ir1", pos="%g %g %g" % RS1_POS, xyaxes="-1 0 0 0 -1 0", fovy=str(FOVY_D))
    ET.SubElement(base, "camera", name="rs1_ir2", pos="%g %g %g" % (RS1_POS[0] - BASELINE, RS1_POS[1], RS1_POS[2]), xyaxes="-1 0 0 0 -1 0", fovy=str(FOVY_D))
    ET.SubElement(base, "camera", name="rs1_color", pos="%g %g %g" % RS1C_POS, xyaxes="-1 0 0 0 -1 0", fovy=str(FOVY_C))
    ET.SubElement(base, "camera", name="mast_rgb", pos="%g %g %g" % MAST_POS, xyaxes="0 -1 0 0 0 1", fovy="%.3f" % MAST_FOVY)
    pr = math.radians(RS2_TRUE["pitch_deg"]); s, c = math.sin(pr), math.cos(pr); dz = RS2_TRUE["dz"]
    for nm, dy in (("rs2_ir1", 0.0), ("rs2_ir2", -BASELINE), ("rs2_color", RS2C_DY)):
        ET.SubElement(base, "camera", name=nm, pos="%g %g %g" % (RS2_POS[0], RS2_POS[1] + dy, RS2_POS[2] + dz), xyaxes=_rs2_axes((0, -1, 0), (s, 0, c)), fovy=str(FOVY_D if "ir" in nm else FOVY_C))
    # IMU site in the D435i optical frame (x right, y down, z fwd): zaxis = (c,0,-s), xaxis = (0,-1,0)
    ET.SubElement(base, "site", name="rs2_imu", pos="%g %g %g" % (RS2_IMU[0], RS2_IMU[1], RS2_IMU[2] + dz), xyaxes=_rs2_axes((0, -1, 0), (-s, 0, -c)))
    if VIEW:   # viewer-only: live image = the frustum cross-section d metres in front of the lens (camera-oriented, no body) (group 4: never drawn by the sensor cameras)
        for nm, (d, tw, th, fov, asp, col) in VIEW_CAMS.items():
            ce = base.find("camera[@name='%s']" % nm); P = np.array([float(v) for v in ce.get("pos").split()])
            ax = np.array([float(v) for v in ce.get("xyaxes").split()]); X, Y = ax[:3], ax[3:]; Z = np.cross(X, Y)
            ty = math.tan(math.radians(fov) / 2) * d; tx = ty * asp
            ET.SubElement(asset, "texture", name="vt_" + nm, type="2d", builtin="flat", width=str(tw), height=str(th), rgb1="0.2 0.2 0.2")
            ET.SubElement(asset, "material", name="vm_" + nm, texture="vt_" + nm, emission="1", rgba="1 1 1 1")
            ET.SubElement(base, "geom", name="vplane_" + nm, type="box", size="%g %g 0.0005" % (tx, ty), group="4",
                          pos="%g %g %g" % tuple(P - d * Z), xyaxes="%g %g %g %g %g %g" % (*X, *Y), material="vm_" + nm,
                          contype="0", conaffinity="0", mass="0")
    sens = root.find("sensor")
    if sens is None: sens = ET.SubElement(root, "sensor")
    ET.SubElement(sens, "gyro", name="emu_gyro", site="rs2_imu"); ET.SubElement(sens, "accelerometer", name="emu_acc", site="rs2_imu")
    if NP["tire_mu"] != 1.0:                      # slick floor / carpet: real skid, not injected numbers
        for d in root.iter("default"):
            if d.get("class") == "tire":
                g0 = d.find("geom"); f = [float(v) for v in g0.get("friction").split()]; f[0] *= NP["tire_mu"]
                g0.set("friction", " ".join("%g" % v for v in f))
    vis = root.find("visual"); g = vis.find("global") if vis is not None else None
    if g is not None: g.set("offwidth", "1280"); g.set("offheight", "960")
    # near clip: MuJoCo's default znear is 0.01*extent (0.20 m in the office) -> desk tops 0.18-0.20 m below the
    # RS1 lens were clipped away and the camera saw *through* them to the floor.  A real D435 returns 0 (invalid)
    # inside min-Z instead; keep the clip at 1 cm and let --min-z produce the invalid band.
    mp = vis.find("map") if vis is not None else None
    if vis is None: vis = ET.SubElement(root, "visual")
    if mp is None: mp = ET.SubElement(vis, "map")
    mp.set("znear", "0.0005")                    # x extent (~20 m) = 1 cm
    return ET.tostring(root, encoding="unicode")


# ------------------------------------------------------------------ ODrive S1 emulation
CURRENT_SOFT_MAX = 10.0      # A: odrv.axis0.config.motor.current_soft_max (Kt 0.9506 Nm/A in the twin model)
class ODriveNode:
    IDLE, CLOSED = 1, 8

    def __init__(self, node_id, axis_dir, AxisCls, eps):
        self.id, self.dir, self.eps = node_id, axis_dir, eps
        self.ax = AxisCls(3.0, 4.0, i_lim=CURRENT_SOFT_MAX)   # src/s1-odrive-config.py current_soft_max
        self.state = self.IDLE; self.wd_en = False; self.wd_to = 2.5; self.t_feed = time.monotonic()
        self.pos = 0.0; self.vel = 0.0; self.disarm = 0

    def handle(self, cmd, data, reply, now):
        if cmd == 0x07:
            st = struct.unpack_from("<I", data)[0]
            if st == self.CLOSED and self.state != self.CLOSED:
                self.ax.reset(); self.ax.vel_setpoint = self.vel; self.ax.input_vel = 0.0
            self.state = st; self.t_feed = now
        elif cmd == 0x0D:
            v, _ff = struct.unpack_from("<ff", data); self.ax.input_vel = v; self.t_feed = now
        elif cmd == 0x18:
            self.disarm = 0
        elif cmd == 0x04:
            op, eid, _ = struct.unpack_from("<BHB", data)
            name, typ = self.eps.get(eid, ("?", "uint32"))
            fmt = {"bool": "?", "uint8": "B", "int8": "b", "uint16": "H", "int16": "h", "uint32": "I", "int32": "i", "float": "f", "uint64": "Q", "int64": "q"}.get(typ, "I")
            if op == 0:            # read
                val = {"vbus_voltage": 36.0 - 0.3 * abs(self.ax.current), "axis0.vel_estimate": self.vel, "axis0.pos_estimate": self.pos,
                       "axis0.config.enable_watchdog": self.wd_en, "axis0.current_state": self.state}.get(name, 0)
                if fmt in "?BbHhIiQq": val = int(val)
                try:
                    reply(self.id << 5 | 0x05, struct.pack("<BHB" + fmt, 0, eid, 0, val)[:8])
                except struct.error:
                    pass
            else:                  # write / call
                if name == "axis0.watchdog_feed": self.t_feed = now
                elif typ != "function":
                    try: val = struct.unpack_from("<" + fmt, data, 4)[0]
                    except struct.error: val = 0
                    if name == "axis0.config.enable_watchdog": self.wd_en = bool(val); self.t_feed = now
                    elif name == "axis0.config.watchdog_timeout": self.wd_to = float(val)
                    elif name == "axis0.controller.config.vel_ramp_rate": self.ax.ramp = float(val)
                    elif name == "axis0.controller.config.vel_limit": self.ax.vlim = float(val)

    def torque(self, fwd_turns, dt, now):
        self.vel = self.dir * fwd_turns
        self.pos += self.vel * dt
        if self.state == self.CLOSED and self.wd_en and now - self.t_feed > self.wd_to:
            self.state = self.IDLE; self.disarm = 0x800        # WATCHDOG_TIMER_EXPIRED
        if self.state != self.CLOSED:      # IDLE: phases open, but a hub motor still has bearing/seal drag + cogging
            # (~0.1 Nm detent on a 6.5" hoverboard motor).  With 0 torque the free wheels let Kevin roll ~7 cm and
            # yaw ~19 deg on the lumpy-carpet hfield before nav even started (2026-10-14).
            self.ax.current = 0.0; return 0.0          # drag/cogging: wheel-joint frictionloss (World.__init__)
        return self.dir * self.ax.update(self.vel, dt)       # forward-frame torque


# ------------------------------------------------------------------ world
class Dog:
    """Yorkie wandering like a dog: pauses 0.5-8 s (sniff/sit), walks 0.3-0.8 m/s, short runs 1.2-1.8 m/s, slow sniffing
    0.15 m/s; picks random targets 0.8-4 m away in free space (GT clearance, straight-line check); turns at <=4 rad/s;
    sidesteps away when Kevin is within 0.35 m (startle) - otherwise ignores the robot (may walk across its path)."""
    KEEP = 0.6
    def __init__(self, env, rng):
        self.e, self.r = env, rng; self.clr = env.clearance
        x, y = env.places.get("hallway", env.start[:2]); self.x, self.y, self.th = float(x), float(y), 0.0
        self.mode, self.tl, self.tg, self.v, self.t = "pause", 2.0, None, 0.0, 0.0
    def c(self, x, y):
        r, k = self.e.gt_world_to_rc(x, y)
        return float(self.clr[r, k]) if 0 <= r < self.clr.shape[0] and 0 <= k < self.clr.shape[1] else 0.0
    def pick(self, kx, ky, away=False):
        for _ in range(60):
            if away: a = math.atan2(self.y - ky, self.x - kx) + self.r.uniform(-0.8, 0.8); d = self.r.uniform(0.6, 1.5)
            else: a = self.r.uniform(-math.pi, math.pi); d = self.r.uniform(0.8, 4.0)
            tx, ty = self.x + d * math.cos(a), self.y + d * math.sin(a)
            if self.c(tx, ty) < 0.2 or math.hypot(tx - kx, ty - ky) < 0.6: continue
            if all(self.c(self.x + (tx - self.x) * f, self.y + (ty - self.y) * f) >= 0.11 and
                   math.hypot(self.x + (tx - self.x) * f - kx, self.y + (ty - self.y) * f - ky) >= min(self.KEEP, math.hypot(self.x - kx, self.y - ky))
                   for f in np.linspace(0.05, 1, 25)):
                u = self.r.random(); self.v = 1.5 if away else (self.r.uniform(1.2, 1.8) if u < 0.15 else self.r.uniform(0.3, 0.8) if u < 0.85 else 0.15)
                self.tg, self.mode = (tx, ty), "walk"; return
        self.mode, self.tl = "pause", self.r.uniform(0.5, 2.0)
    def step(self, dt, kx, ky):
        self.t += dt; dk = math.hypot(self.x - kx, self.y - ky)
        # Kevin is an obstacle to the dog like any other: keep >= KEEP from its centre (body half-length 0.33 m + dog
        # half-length 0.13 m + a little); inside that, move away.  (Old: startle only < 0.35 m and never while running,
        # so the dog could trot straight into a stopped Kevin.)
        if dk < self.KEEP - 0.05 and not (self.mode == "walk" and self.tg and math.hypot(self.tg[0] - kx, self.tg[1] - ky) > dk):
            self.pick(kx, ky, away=True)
        if self.mode == "pause":
            self.tl -= dt
            if self.tl <= 0: self.pick(kx, ky)
            return
        tx, ty = self.tg; dx, dy = tx - self.x, ty - self.y; dist = math.hypot(dx, dy)
        if dist < 0.08: self.mode, self.tl = "pause", self.r.uniform(0.5, 8.0); return
        err = (math.atan2(dy, dx) - self.th + math.pi) % (2 * math.pi) - math.pi
        self.th += max(-4 * dt, min(4 * dt, err))
        if abs(err) < 1.0:
            nx, ny = self.x + self.v * dt * math.cos(self.th), self.y + self.v * dt * math.sin(self.th)
            if self.c(nx, ny) < 0.09: self.mode, self.tl = "pause", self.r.uniform(0.3, 1.5); return
            dn = math.hypot(nx - kx, ny - ky)
            if dn < self.KEEP and dn < dk: self.pick(kx, ky, away=True); return      # would close in on Kevin
            self.x, self.y = nx, ny
    def pose(self):   # (pos, quat) with a small trot bob
        bob = 0.006 * abs(math.sin(self.t * 18)) if self.mode == "walk" and self.v > 0.2 else 0.0
        return (self.x, self.y, bob), (math.cos(self.th / 2), 0, 0, math.sin(self.th / 2))


class World:
    def __init__(self, a):
        import mujoco
        self.mj = mujoco; self.a = a
        self.env = envs.build(a.env, a.seed)
        set_rs2_mount(a.seed); print("emu.world: rs2 true mount %s" % RS2_TRUE, flush=True)
        xml = compose(self.env, a.tire)
        self.m = mujoco.MjModel.from_xml_string(xml); self.d = mujoco.MjData(self.m)
        # full-detail twin: the RealSense housings/lens meshes enclose the sensor cameras' origins; a camera never
        # images its own housing (1 cm hits -> stereo shadow over whole rows -> no_depth).  Hide them from the sensor
        # renders (group 5), keep them in the third-person view.
        for i in range(self.m.ngeom):
            if self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mn = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_MESH, int(self.m.geom_dataid[i])) or ""
                if mn.endswith(("rs_aluminum", "rs_ir_lens")): self.m.geom_group[i] = 5
        if self.m.nkey: mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.dt = self.m.opt.timestep
        sys.path.insert(0, TWIN); os.environ.setdefault("KEVIN_TIRE", a.tire)
        from kevin_drive import ODriveAxis      # twin's ODrive S1 model (ramp, PI, current + torque-speed limits)
        eps = {v["id"]: (k, v["type"]) for k, v in json.load(open(os.path.join(REPO, "src", "flat_endpoints.json")))["endpoints"].items()}
        self.nodes = [ODriveNode(0, -1, ODriveAxis, eps), ODriveNode(1, +1, ODriveAxis, eps)]   # left mounted mirrored (wheelbase invert_left)
        self.jl = self.m.joint("wheel_left_joint"); self.jr = self.m.joint("wheel_right_joint")
        # hub-motor bearing/seal drag + cogging detent as DRY friction (true stiction, MuJoCo frictionloss) in every axis
        # state; a viscous-like drag still let Kevin creep ~0.005 rad/s on the lumpy carpet while the drives were IDLE
        # MuJoCo soft contacts creep under sustained tangential load (Kevin parked on a 0.3 % carpet slope yawed
        # 0.003 rad/s with locked wheels); the standard remedy is the noslip post-solver (40 s parked: 0.11 -> 0.005 rad)
        self.m.opt.noslip_iterations = int(os.environ.get("KEVIN_EMU_NOSLIP", 10))
        for j in (self.jl, self.jr): self.m.dof_frictionloss[j.dofadr[0]] = max(float(self.m.dof_frictionloss[j.dofadr[0]]), float(os.environ.get("KEVIN_EMU_WHEELDRAG", 0.1)))
        self.al = self.m.actuator("wheel_left_motor").id; self.ar = self.m.actuator("wheel_right_motor").id
        mujoco.mj_forward(self.m, self.d)
        R = self.d.xmat[self.m.body("base_link").id].reshape(3, 3)
        self.sg = [float(np.sign((self.d.xmat[self.m.jnt_bodyid[j.id]].reshape(3, 3) @ self.m.jnt_axis[j.id]) @ R[:, 1])) for j in (self.jl, self.jr)]
        self.teleport(*self.env.start)
        self.base = self.m.body("base_link").id
        self.kevin_geoms = set(i for i in range(self.m.ngeom) if self._under(self.m.geom_bodyid[i], self.base))
        self.env_geoms = set(i for i in range(self.m.ngeom) if (self.m.geom(i).name or "").startswith("env") or self.m.body(self.m.geom_bodyid[i]).name.startswith("obs"))
        self.gyro = self.m.sensor("emu_gyro").adr[0]; self.acc = self.m.sensor("emu_acc").adr[0]
        self.rng = np.random.default_rng(a.seed + 1)
        self.gbias = self.rng.normal(0, NP["g_bias"], 3).astype(np.float32)
        self.gscale = (1.0 + self.rng.normal(0, NP["g_scale"], 3)).astype(np.float32); self.grw = np.zeros(3, np.float32)
        self.rerr = 1.0 + self.rng.normal(0, NP["r_err"], 2); self.trerr = 1.0 + self.rng.normal(0, NP["tr_err"])
        self.enc_pos = [0.0, 0.0]; self._rng_cam = np.random.default_rng(a.seed + 7)
        print("emu.world: noise=%s r_err=%s tr_err=%.4f g_bias=%s g_scale=%s" % (NOISE, np.round(self.rerr - 1, 4), self.trerr - 1, np.round(self.gbias, 4), np.round(self.gscale - 1, 4)), flush=True)
        self.collisions = 0; self._in_contact = False; self.paused = False; self.blind = set()
        self.dog = Dog(self.env, np.random.default_rng(a.seed + 11)) if getattr(self.env, "dog", False) and os.environ.get("KEVIN_EMU_DOG", "1") == "1" else None
        self.dog_mid = self.m.body("dog").mocapid[0] if self.dog else -1
        self.dog_geoms = set(i for i in range(self.m.ngeom) if (self.m.geom(i).name or "").startswith("dog_"))
        self.dog_contacts = 0; self._dog_in = False
        self._setup_io()

    def _under(self, b, root):
        while b > 0:
            if b == root: return True
            b = self.m.body_parentid[b]
        return False

    def teleport(self, x, y, yaw):
        q = self.d.qpos; q[0], q[1] = x, y; q[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.d.qvel[:] = 0; self.mj.mj_forward(self.m, self.d)

    # ---------------- io
    def _setup_io(self):
        L1 = [("depth", (DH, DW), "u2"), ("color", (CH, CW, 3), "u1")]
        IRL = [("ir1", (DH, DW), "u1"), ("ir2", (DH, DW), "u1")] if self.a.ir else []
        L2 = L1 + IRL; L1 = L1 + IRL                     # both RealSenses stream IR1/IR2 (cuVSLAM dual rig)
        self.b1 = shm.Block("kevin_rs1", L1, create=True); self.b2 = shm.Block("kevin_rs2", L2, create=True)
        self.imu = shm.ImuRing("kevin_imu", create=True)
        self.gt = shm.Block("kevin_gt", [("pose", (12,), "f8")], create=True)
        self.bm = shm.Block("kevin_mast", [("color", (MH, MW, 3), "u1")], create=True)
        di, ci = _intr(DW, DH, FOVY_D), _intr(CW, CH, FOVY_C)
        I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        def st(i, t): return dict(i, pose=[I3, t])
        ser1, ser2 = "SIM-RS1-TOP-000001", "SIM-RS2-FWD-000002"
        info = {"devices": {
            ser1: dict(name="Intel RealSense D435", fps=FPS, shm="kevin_rs1", layout=[[n, list(s), d] for n, s, d in L1], imu=None,
                       streams=dict(depth=st(di, [0, 0, 0]), color=st(ci, [-0.015, 0, 0]), ir1=st(di, [0, 0, 0]), ir2=st(di, [BASELINE, 0, 0]))),
            ser2: dict(name="Intel RealSense D435I", fps=FPS, shm="kevin_rs2", layout=[[n, list(s), d] for n, s, d in L2], imu="kevin_imu",
                       streams=dict(depth=st(di, [0, 0, 0]), color=st(ci, [-0.015, 0, 0]), ir1=st(di, [0, 0, 0]), ir2=st(di, [BASELINE, 0, 0])))},
            "mast": dict(shm="kevin_mast", layout=[["color", [MH, MW, 3], "u1"]], intr=_intr(MW, MH, MAST_FOVY), hfov_deg=MAST_HFOV, fovy_deg=MAST_FOVY,
                         pos=MAST_POS, pitch_deg=0.0, hz=MAST_HZ),
            "env": self.env.name, "seed": self.a.seed, "places": self.env.places, "start": self.env.start, "tire": self.a.tire,
            "mounts": dict(rs1=RS1_POS, rs2=RS2_POS, rs2_imu=RS2_IMU, rs2_pitch_deg=math.degrees(RS2_PITCH), rs2_true=RS2_TRUE), "pid": os.getpid(), "t0": time.monotonic()}
        np.savez_compressed("/dev/shm/kevin_gt_map.npz", occ=self.env.gt, origin=np.array(self.env.gt_origin), res=envs.GT_RES)
        shm.write_info(info)
        self.can = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.can.bind(("127.0.0.1", CAN_PORT)); self.can.setblocking(False)
        self.ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.ctl.bind(("127.0.0.1", CTL_PORT)); self.ctl.setblocking(False)
        self.clients = set(); self.FMT = struct.Struct("<dIB??8s")

    def _send(self, aid, data):
        pkt = self.FMT.pack(time.time(), aid, len(data), False, False, bytes(data).ljust(8, b"\0"))
        for c in list(self.clients):
            try: self.can.sendto(pkt, c)
            except OSError: self.clients.discard(c)

    def _poll_can(self, now):
        while True:
            try: pkt, addr = self.can.recvfrom(64)
            except BlockingIOError: break
            self.clients.add(addr)
            t, aid, dlc, rtr, ext, data = self.FMT.unpack(pkt)
            if aid == 0x7FF: continue
            node, cmd = aid >> 5, aid & 0x1F
            if node < len(self.nodes):
                self.nodes[node].handle(cmd, data[:dlc], self._send, now)

    def _poll_ctl(self):
        while True:
            try: pkt, addr = self.ctl.recvfrom(65536)
            except BlockingIOError: break
            try:
                c = json.loads(pkt); k = c.get("cmd"); r = {"ok": True}
                if k == "teleport": self.teleport(c["x"], c["y"], c.get("yaw", 0.0))
                elif k == "obstacle":
                    b = self.m.body("obs%d" % int(c.get("i", 0))).mocapid[0]; self.d.mocap_pos[b] = (c["x"], c["y"], 0.0)
                elif k == "clear_obstacles":
                    for i in range(N_OBS): self.d.mocap_pos[self.m.body("obs%d" % i).mocapid[0]] = (0, 0, -5)
                elif k == "blind": self.blind = set(c.get("cams", []))          # sensor-failure injection: zero depth
                elif k == "pause": self.paused = True
                elif k == "resume": self.paused = False
                elif k == "state": r.update(self.state())
                self.ctl.sendto(json.dumps(r).encode(), addr)
            except Exception as e:
                self.ctl.sendto(json.dumps({"ok": False, "err": str(e)}).encode(), addr)

    def state(self):
        q = self.d.qpos; yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        R = self.d.xmat[self.base].reshape(3, 3); vb = R.T @ self.d.qvel[:3]
        return dict(t=self.d.time, x=float(q[0]), y=float(q[1]), yaw=yaw, v=float(vb[0]), w=float(self.d.qvel[5]), collisions=self.collisions, last_contact=getattr(self, "last_contact", None), drives=[[n.state, round(n.ax.input_vel, 4), round(n.ax.vel_setpoint, 4), round(n.vel, 4), round(float(n.ax.integrator), 4)] for n in self.nodes[:2]],
                    dog=None if not self.dog else dict(x=self.dog.x, y=self.dog.y, th=self.dog.th, mode=self.dog.mode, contacts=self.dog_contacts, last=getattr(self, "last_dog_contact", None)))

    # ---------------- physics
    def step(self, now):
        d = self.d
        for n, j, a, sg in ((self.nodes[0], self.jl, self.al, self.sg[0]), (self.nodes[1], self.jr, self.ar, self.sg[1])):
            d.ctrl[a] = sg * n.torque(sg * d.qvel[j.dofadr[0]] / TWO_PI, self.dt, now)
        self.mj.mj_step(self.m, d)
        self._acc_sum = getattr(self, "_acc_sum", 0.0) + d.sensordata[self.acc:self.acc + 3]; self._acc_n = getattr(self, "_acc_n", 0) + 1

    def contact(self):
        hit = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.kevin_geoms and g2 in self.env_geoms) or (g2 in self.kevin_geoms and g1 in self.env_geoms):
                hit = True
                if not self._in_contact:              # remember which robot/env geoms touched (diagnostics)
                    nm = lambda g: self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_GEOM, g) or str(g)
                    self.last_contact = (nm(g1), nm(g2), [round(float(v), 3) for v in c.pos])
                break
        if self.dog_geoms:
            dh = any((c.geom1 in self.kevin_geoms and c.geom2 in self.dog_geoms) or (c.geom2 in self.kevin_geoms and c.geom1 in self.dog_geoms)
                     for c in (self.d.contact[i] for i in range(self.d.ncon)))
            if dh and not self._dog_in:
                self.dog_contacts += 1; vb = float(np.hypot(*self.d.qvel[:2]))
                self.last_dog_contact = dict(t=round(float(self.d.time), 2), kevin_speed=round(vb, 3), kevin_w=round(float(self.d.qvel[5]), 3),
                                             dog_mode=self.dog.mode, dog_v=self.dog.v, kevin=[round(float(v), 2) for v in self.d.qpos[:2]], dog=[round(self.dog.x, 2), round(self.dog.y, 2)])
                print("emu.world: DOG CONTACT %s" % self.last_dog_contact, flush=True)
            self._dog_in = dh
        hz = getattr(self.env, "hazard", None)
        if hz and hz(float(self.d.qpos[0]), float(self.d.qpos[1])):        # wheel over a curb / drop-off
            hit = True
            if not self._in_contact: self.last_contact = ("base", "drop-off", [round(float(self.d.qpos[0]), 3), round(float(self.d.qpos[1]), 3), 0.0])
        # one event per continuous contact: resting/scraping contact flickers on and off at the solver level, so an
        # event only starts after >= 0.5 s (sim) without any contact
        if hit:
            new = (self.d.time - getattr(self, "_last_hit_t", -1e9)) > 0.5; self._last_hit_t = self.d.time
        else: new = False
        if hit and new:
            self.collisions += 1; print("emu.world: CONTACT t=%.1f %s v=%.2f w=%.2f" % (self.d.time, getattr(self, "last_contact", None), float(np.hypot(*self.d.qvel[:2])), float(self.d.qvel[5])), flush=True)
        self._in_contact = hit

    def run(self):
        a = self.a; mj = self.mj
        rq = {"job": None}; cv = threading.Condition()
        rt = threading.Thread(target=self._render_loop, args=(rq, cv), daemon=True); rt.start()
        t0 = time.monotonic(); sim0 = self.d.time; k = 0; n_cam = 0; n_imu = 0; n_enc = 0
        last_report = t0; steps = 0
        print("emu.world: %s seed=%d tire=%s start=%s  CAN udp:%d  ctl udp:%d" % (self.env.name, a.seed, a.tire, self.env.start, CAN_PORT, CTL_PORT), flush=True)
        while True:
            now = time.monotonic()
            self._poll_can(now); self._poll_ctl()
            if self.paused:
                t0 = now - (self.d.time - sim0) / a.rt; time.sleep(0.005); continue
            target = sim0 + (now - t0) * a.rt
            if target - self.d.time > 0.25:           # fell far behind (debugger / overload): drop time, never fast-forward
                t0 = now - (self.d.time - sim0) / a.rt; target = self.d.time
            while self.d.time < target:
                self.step(now); steps += 1
                st_ns = int((t0 + (self.d.time - sim0) / a.rt) * 1e9)
                if self.d.time >= n_imu * 0.005:
                    n_imu += 1
                    if NP["g_rw"]: self.grw += self.rng.normal(0, NP["g_rw"] * math.sqrt(0.005), 3).astype(np.float32)
                    g = (self.d.sensordata[self.gyro:self.gyro + 3].astype(np.float32) * self.gscale + self.gbias + self.grw
                         + np.float32(NP["g_drift"] * self.d.time) + self.rng.normal(0, NP["g_white"], 3).astype(np.float32))
                    # BMI055 output = its digital low-pass of the force, not an instantaneous physics sample: average the
                    # 1 kHz steps of this 5 ms output period (anti-alias; raw samples alias contact chatter to +-40 m/s^2)
                    am = (self._acc_sum / max(self._acc_n, 1)) if getattr(self, "_acc_n", 0) else self.d.sensordata[self.acc:self.acc + 3]
                    self._acc_sum = 0.0; self._acc_n = 0
                    ac = np.asarray(am, np.float32) + self.rng.normal(0, NP["a_white"], 3).astype(np.float32)
                    self.imu.push(st_ns + int(NP["imu_dt"] * 1e9), g, ac)      # IMU clock offset vs cameras
                if self.d.time >= n_enc * 0.01:
                    n_enc += 1; self.contact()
                    if self.dog:
                        self.dog.step(0.01, float(self.d.qpos[0]), float(self.d.qpos[1])); p_, q_ = self.dog.pose()
                        self.d.mocap_pos[self.dog_mid] = p_; self.d.mocap_quat[self.dog_mid] = q_
                    # reported wheel speeds: per-wheel radius error, then the differential part scaled by the
                    # effective-track error (pivot skid / tire compliance), encoder quantisation on position
                    f = [nd.dir * nd.vel * self.rerr[k] for k, nd in enumerate(self.nodes[:2])]
                    if len(f) == 2:
                        mm, dd = 0.5 * (f[0] + f[1]), 0.5 * (f[1] - f[0]) / self.trerr; f = [mm - dd, mm + dd]
                    for k, nd in enumerate(self.nodes):      # S1 cyclic traffic as configured in src/s1-odrive-config.py (all 10 ms, heartbeat 100 ms)
                        b = nd.id << 5; iq = nd.ax.current
                        vrep = nd.dir * f[k] if k < len(f) else nd.vel
                        self.enc_pos[k % 2] += vrep * 0.01; prep = self.enc_pos[k % 2]
                        if NP["quant"]: prep = round(prep * NP["quant"]) / NP["quant"]
                        if NOISE == "ideal": vrep, prep = nd.vel, nd.pos
                        self._send(b | 0x09, struct.pack("<ff", prep, vrep + self.rng.normal(0, 0.002)))
                        self._send(b | 0x14, struct.pack("<ff", iq, iq))
                        self._send(b | 0x1C, struct.pack("<ff", iq * 0.95, iq * 0.95))
                        self._send(b | 0x03, struct.pack("<II", 0, nd.disarm))
                        self._send(b | 0x15, struct.pack("<ff", 35.0, 40.0))
                        self._send(b | 0x17, struct.pack("<ff", 36.0, iq * 0.1))
                        if n_enc % 10 == 0:
                            self._send(b | 0x01, struct.pack("<IBBBB", 0, nd.state, 0, 0, 0))
                    s = self.state()
                    self.gt.write(st_ns, pose=np.array([s["t"], s["x"], s["y"], s["yaw"], s["v"], s["w"], s["collisions"], st_ns * 1e-9,
                                                        self.nodes[0].vel, self.nodes[1].vel, self.nodes[0].ax.input_vel, self.nodes[1].ax.input_vel]))
                if self.d.time >= n_cam / FPS:
                    n_cam += 1
                    with cv:
                        rq["job"] = (st_ns, self.d.qpos.copy(), self.d.mocap_pos.copy(), self.d.mocap_quat.copy()); cv.notify()
            if now - last_report > 10:
                s = self.state()
                print("emu.world: t=%.1f x=%.2f y=%.2f yaw=%.2f v=%.2f collisions=%d  physics %.0f Hz cam %.1f Hz" % (
                    s["t"], s["x"], s["y"], s["yaw"], s["v"], s["collisions"], steps / (now - last_report), self._ncam / (now - last_report)), flush=True)
                steps = 0; self._ncam = 0; last_report = now
            time.sleep(0.0008)

    _ncam = 0

    def _view(self, dr, imgs):
        """Third-person chase view (KEVIN_EMU_VIEW=1, ~10 Hz): own model copy with a brighter headlight + extra lights,
        camera frustums drawn as scene lines (never in the sensor renders).  -> /dev/shm/kevin_view.jpg (+ numbered
        frames in KEVIN_EMU_VIEW_DIR for videos)."""
        mj = self.mj
        if not hasattr(self, "_vr"):
            import copy
            self._vm = copy.copy(self.m); vm = self._vm
            vm.vis.headlight.ambient[:] = 0.35; vm.vis.headlight.diffuse[:] = 0.55; vm.vis.headlight.specular[:] = 0.1
            vm.light_diffuse[:] = np.minimum(vm.light_diffuse * 2.2, 0.6)
            self._vr = mj.Renderer(vm, VIEW_H, VIEW_W); self._vc = mj.MjvCamera(); self._vn = 0; self._vaz = None
        vm, R, cam = self._vm, self._vr, self._vc
        import cv2
        for nm, img in imgs.items():                          # live textures (~10 Hz)
            tid = vm.texture("vt_" + nm).id; tw, th = int(vm.tex_width[tid]), int(vm.tex_height[tid]); adr = int(vm.tex_adr[tid])
            if img.dtype == np.uint16:                           # depth mm -> turbo colormap 0.2..4 m, holes black
                z = img.astype(np.float32) * 0.001; v = np.clip((z - 0.2) / 3.8, 0, 1)
                rgbi = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy(); rgbi[z <= 0] = 0
            else: rgbi = img
            t = cv2.resize(rgbi, (tw, th), interpolation=cv2.INTER_AREA)[::-1]   # GL texture origin = bottom-left
            vm.tex_data[adr:adr + tw * th * 3] = t.reshape(-1)
            R._gl_context.make_current(); mj.mjr_uploadTexture(vm, R._mjr_context, tid)
        try: vc = json.load(open("/dev/shm/kevin_view_cam.json"))
        except Exception: vc = {}
        camvis = bool(vc.get("camvis", True))  # default ON
        bx, by = float(dr.qpos[0]), float(dr.qpos[1]); q = dr.qpos[3:7]; yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        az = math.degrees(yaw) + 200.0                                      # behind-left of Kevin, smoothed
        self._vaz = az if self._vaz is None else self._vaz + 0.08 * (((az - self._vaz) + 180) % 360 - 180)
        cam.type = mj.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = (bx, by, 0.35); cam.distance = float(vc.get("dist", 3.2))
        cam.elevation = float(vc.get("elev", -62.0)); cam.azimuth = self._vaz + float(vc.get("az", 0.0))
        # keep the camera on Kevin's side of any wall: ray from the look-at point back toward the eye against the house
        # (wall collision boxes, group 3) and furniture (group 0); pull the eye in to 0.15 m before the first hit
        ea, ee = math.radians(cam.azimuth), math.radians(cam.elevation)
        back = -np.array([math.cos(ee) * math.cos(ea), math.cos(ee) * math.sin(ea), math.sin(ee)])   # look-at -> eye
        if not hasattr(self, "_rgrp"): self._rgrp = np.array([1, 0, 0, 1, 0, 0], np.uint8); self._rid = np.zeros(1, np.int32)
        hit = mj.mj_ray(vm, dr, np.array(cam.lookat, float), back, self._rgrp, 1, -1, self._rid)
        if 0 <= hit < cam.distance + 0.15:
            cam.distance = max(0.5, hit - 0.15); cam.elevation = min(cam.elevation, -70.0) if hit < 1.2 else cam.elevation
        if not hasattr(self, "_vopt"): self._vopt = mj.MjvOption(); self._vopt.geomgroup[5] = 1
        self._vopt.geomgroup[4] = 1 if camvis else 0  # group 4 = image-slice geometry
        R.update_scene(dr, cam, self._vopt); sc = R.scene
        near, far = VIEW_CLIP
        if camvis:  # skip frustum lines when camera vis off
            for nm, (_d, _tw, _th, _fov, _asp, col) in VIEW_CAMS.items():   # FOV: thin translucent lines, no filled planes
                cid = self.m.camera(nm).id; P = dr.cam_xpos[cid].copy(); Rm = dr.cam_xmat[cid].reshape(3, 3)
                ty = math.tan(math.radians(self.m.cam_fovy[cid]) / 2); tx = ty * _asp
                K = [Rm @ np.array([sx * tx, sy * ty, -1.0]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                N = [P + near * k for k in K]; F = [P + far * k for k in K]
                segs = [(P, f, 0.30) for f in F] + [(N[i], N[(i + 1) % 4], 0.22) for i in range(4)] + [(F[i], F[(i + 1) % 4], 0.22) for i in range(4)]
                for a_, b_, al in segs:
                    if sc.ngeom >= sc.maxgeom: break
                    g = sc.geoms[sc.ngeom]; mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), np.array((*col, al), np.float32))
                    mj.mjv_connector(g, mj.mjtGeom.mjGEOM_LINE, 1.0, np.asarray(a_, float), np.asarray(b_, float)); sc.ngeom += 1
        img = R.render()
        import cv2
        ok, jp = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            tmp = "/dev/shm/kevin_view.tmp.jpg"; open(tmp, "wb").write(jp.tobytes()); os.replace(tmp, "/dev/shm/kevin_view.jpg")
            if VIEW_DIR: open(os.path.join(VIEW_DIR, "f%06d.jpg" % self._vn), "wb").write(jp.tobytes())
        dg = getattr(self, "dog", None)
        if dg is not None:
            dog_x, dog_y = float(dr.mocap_pos[self.dog_mid][0]), float(dr.mocap_pos[self.dog_mid][1])
            qw, qx, qy, qz = [float(v) for v in dr.mocap_quat[self.dog_mid]]
            dog_yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            dog_pose = [dog_x, dog_y, dog_yaw]
        else:
            dog_pose = None
        js = dict(t=float(dr.time), env=self.env.name, start=list(self.env.start), places={k: list(v) for k, v in self.env.places.items()}, kevin=[bx, by, yaw], collisions=self.collisions, dog_contacts=getattr(self, "dog_contacts", 0), dog=dog_pose)
        open("/dev/shm/kevin_sim.tmp.json", "w").write(json.dumps(js)); os.replace("/dev/shm/kevin_sim.tmp.json", "/dev/shm/kevin_sim.json")
        self._vn += 1

    def _render_loop(self, rq, cv):
        mj = self.mj; m = self.m; dr = mj.MjData(m)
        Rd = mj.Renderer(m, DH, DW); Rd.enable_depth_rendering()
        Rc8 = mj.Renderer(m, DH, DW) if self.a.ir else None
        Rc = mj.Renderer(m, CH, CW)
        Rm = mj.Renderer(m, MH, MW); t_mast = [0]
        rng = np.random.default_rng(7)
        NB = 8   # pre-generated noise banks (generating 400k normals per frame costs more than the render)
        zn = [rng.standard_normal((DH, DW), dtype=np.float32) for _ in range(NB)]
        drop = [rng.random((DH, DW), dtype=np.float32) < self.a.holes for _ in range(NB)]
        irn = [rng.normal(0, 2.0, (DH, DW)).astype(np.float32) for _ in range(NB)]
        k = [0]
        # D4xx stereo depth model (Intel D400 datasheet 337029; "Tuning depth cameras for best performance" whitepaper):
        #   depth origin = LEFT IR imager; RMS error = Z^2 * subpixel / (f_px * B), subpixel ~0.08, B = 50 mm;
        #   invalid band on the LEFT edge, width = f_px * B / Z px (left imager sees what the right one cannot);
        #   min-Z ~0.175 m @848x480 (max disparity), holes at occlusion edges, random dropouts (dark/shiny).
        fpx = _intr(DW, DH, FOVY_D)["fx"]; sub = self.a.subpix; cols = np.arange(DW, dtype=np.float32)[None, :]
        # Sunlight (KEVIN_EMU_SUN=1, outdoor): the 850 nm projector is washed out -> passive stereo only.  D4xx
        # still ranges on textured ground, but subpixel error roughly doubles and bright low-texture surfaces
        # (sunlit concrete, white paint, sky glare) drop out in blobs; whole frames occasionally lose more.
        SUN = os.environ.get("KEVIN_EMU_SUN", "0") == "1"
        if SUN:
            sub = sub * 2.0
            sunm = [envs._smooth(rng, (DH, DW), 40) > rng.uniform(0.25, 0.45) for _ in range(NB)]
        def depth(cam):
            Rd.update_scene(dr, cam); z = Rd.render(); i = k[0] = (k[0] + 1) % NB
            zz = z + (z * z) * (sub / (fpx * BASELINE)) * zn[i]
            bad = (z < self.a.min_z) | (z > 8.0) | (drop[i] if self.a.holes > 0 else False)
            if SUN: bad |= sunm[i] | (rng.random(z.shape) < 0.03)
            if self.a.band: bad |= cols < fpx * BASELINE / np.maximum(z, 1e-3)       # left invalid band
            # stereo occlusion shadow (left-imager origin): pixel c is unseen by the right imager if a nearer pixel
            # c+s to its right has disparity >= d(c)+s.  Geometric, one-sided, width f*B*(1/zn-1/zf) px.  (Old rule
            # |dz|>8 % on both sides of every edge blanked thin grass blades twice -> >50 % invalid -> no_depth stop.)
            xr = cols - (fpx * BASELINE) / np.maximum(z, 1e-3)                      # right-image column
            smin = np.minimum.accumulate(xr[:, ::-1], axis=1)[:, ::-1]                 # min over c' >= c
            bad[:, :-1] |= smin[:, 1:] <= xr[:, :-1] - 0.5                             # something to the right lands left of us
            zz *= 1000.0; zz[bad] = 0
            return zz.astype(np.uint16)
        def rgb(R, cam):
            R.update_scene(dr, cam); return R.render()
        W3 = np.array([0.299 * 0.9, 0.587 * 0.9, 0.114 * 0.9], np.float32)
        while True:
            with cv:
                while rq["job"] is None: cv.wait()
                st_ns, qpos, mocap, mquat = rq["job"]; rq["job"] = None
            dr.qpos[:] = qpos; dr.mocap_pos[:] = mocap; dr.mocap_quat[:] = mquat; mj.mj_kinematics(m, dr); mj.mj_camlight(m, dr)
            d1 = depth("rs1_depth")
            if "rs1" in self.blind: d1[:] = 0
            out1 = dict(depth=d1, color=rgb(Rc, "rs1_color"))
            out = dict(depth=depth("rs2_ir1"), color=rgb(Rc, "rs2_color"))
            if "rs2" in self.blind: out["depth"][:] = 0
            if Rc8 is not None:
                for dev, o in (("rs1", out1), ("rs2", out)):
                    for nm in ("ir1", "ir2"):
                        g = rgb(Rc8, dev + "_" + nm).astype(np.float32) @ W3
                        # read noise is temporally independent on a real imager: a fixed 8-frame bank added at the same
                        # pixel positions acts like body-fixed texture and pulls visual odometry toward zero motion.
                        # Random 2-D shift of a bank pattern per frame -> no static pattern, ~1 ms.
                        g += np.roll(irn[int(rng.integers(NB))], (int(rng.integers(DH)), int(rng.integers(DW))), axis=(0, 1))
                        o[nm] = np.clip(g, 0, 255).astype(np.uint8)
            jt = (lambda: int(self._rng_cam.normal(0, NP["cam_jit"]) * 1e9)) if NP["cam_jit"] else (lambda: 0)
            self.b1.write(st_ns + jt(), **out1)
            self.b2.write(st_ns + jt(), **out); self._ncam += 1
            if st_ns - t_mast[0] >= 1e9 / MAST_HZ:
                t_mast[0] = st_ns; self.bm.write(st_ns, color=rgb(Rm, "mast_rgb"))
            if VIEW and self._ncam % 3 == 0:
                self._view(dr, {"rs1_depth": out1["depth"], "rs1_color": out1["color"], "rs2_ir1": out["depth"], "rs2_color": out["color"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="apartment", choices=list(envs.ENVS)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subpix", type=float, default=0.08, help="stereo subpixel RMS (Intel: ~0.08; domain-rand 0.05-0.2)")
    ap.add_argument("--holes", type=float, default=0.004, help="random depth dropout fraction")
    ap.add_argument("--min-z", type=float, default=0.175, help="D435 min-Z at 848x480")
    ap.add_argument("--no-band", dest="band", action="store_false", help="disable the left invalid band")
    ap.add_argument("--tire", default=os.environ.get("KEVIN_TIRE", "stock")); ap.add_argument("--ir", type=int, default=1)
    ap.add_argument("--rt", type=float, default=1.0, help="real-time factor (1.0 = wall clock; robot code assumes 1.0)")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    World(a).run()


if __name__ == "__main__":
    main()
