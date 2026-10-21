"""Before/after sensor renders of the emulator (KEVIN_EMU_REAL=0 vs 1) at fixed casita poses -> one PNG per mode.
python bench/slam/render_compare.py OUT.png   (mode from KEVIN_EMU_REAL)"""
import math, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("KEVIN_EMU_MODEL", "full")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path[:0] = [REPO]
import numpy as np, cv2, mujoco
from emu import world as W, envs, realism
POSES = [(3.6, 2.2, math.radians(90)), (3.0, 2.6, math.radians(-60)), (4.3, 5.9, math.radians(20)), (2.0, 7.0, math.radians(95)), (1.9, 11.2, math.radians(150)), (5.0, 11.0, math.radians(-30))]
e = envs.build("casita", 0); xml = W.compose(e, "stock"); m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
if m.nkey: mujoco.mj_resetDataKeyframe(m, d, 0)
Rc = mujoco.Renderer(m, W.CH, W.CW); Rm = mujoco.Renderer(m, W.MH, W.MW); Rd = mujoco.Renderer(m, W.DH, W.DW); Rd.enable_depth_rendering()
fx = {}
if realism.REAL:
    ci, mt = W._intr(W.CW, W.CH, W.FOVY_C), W.MAST_TRUE
    fx["c"] = realism.CamFX(W.CW, W.CH, [[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]], seed=2)
    fx["m"] = realism.CamFX(W.MW, W.MH, [[mt["fx"], 0, mt["cx"]], [0, mt["fy"], mt["cy"]], [0, 0, 1]], dist=(mt["k1"], mt["k2"]), render_K=W.MAST_RK, exposure_s=0.015, vignette=0.35, seed=3)
rows = []; tm = []
for (x, y, th) in POSES:
    d.qpos[0], d.qpos[1] = x, y; d.qpos[3:7] = (math.cos(th / 2), 0, 0, math.sin(th / 2)); mujoco.mj_forward(m, d)
    t0 = time.perf_counter()
    Rm.update_scene(d, "mast_rgb"); mi = Rm.render(); Rc.update_scene(d, "rs2_color"); ci_ = Rc.render()
    Rd.update_scene(d, "rs2_ir1"); z = Rd.render()
    for _ in range(12 if fx else 1):           # let auto-exposure settle on this view
        mo = fx["m"](mi, 0.0) if fx else mi; co = fx["c"](ci_, 0.0) if fx else ci_
    tm.append((time.perf_counter() - t0) * 1e3)
    dc = cv2.applyColorMap(np.clip(255 * (1 - (z - 0.2) / 4.0), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO); dc[z <= 0] = 0
    rows.append(np.hstack([cv2.cvtColor(mo, cv2.COLOR_RGB2BGR), cv2.cvtColor(co, cv2.COLOR_RGB2BGR), cv2.resize(dc, (640, 480))]))
img = np.vstack(rows); img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
lab = "REAL=1 (CC0 PBR textures, decor, ceiling, sun+lamps, CamFX, mast DFOV118 k1-0.05)" if realism.REAL else "REAL=0 (previous emulator rendering)"
cv2.putText(img, lab + "   cols: mast | RS2 color | RS2 depth(0.2-4.2 m)", (8, 18), 0, 0.5, (0, 255, 255), 1)
cv2.imwrite(sys.argv[1], img); print("wrote", sys.argv[1], "ms/pose (incl. AE settle loop)", np.round(tm, 1))
