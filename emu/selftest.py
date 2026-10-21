"""Offline checks of the emulator world (no robot code): physics stability, ODrive drive response, render cost.
  python -m emu.selftest [env]"""
import sys, time, math, os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from emu import world as W

from emu import world as W
TP = W._robot_tire_profile()
class A: env = sys.argv[1] if len(sys.argv) > 1 else "apartment"; seed = 0; tire = TP.NAME; ir = 1; rt = 1.0
R_W = TP.WHEEL_R_LOADED_M
w = W.World(A()); m, d = w.m, w.d
s0 = w.state(); print("start", {k: round(v, 3) for k, v in s0.items()})
for _ in range(1000): w.step(time.monotonic())
s1 = w.state(); print("after 1 s idle", {k: round(v, 3) for k, v in s1.items()})
for n in w.nodes: n.state = n.CLOSED; n.ax.input_vel = n.dir * 0.3 / (2 * math.pi * R_W)   # 0.3 m/s forward
t = time.perf_counter()
for _ in range(2000): w.step(time.monotonic())
dt = time.perf_counter() - t
s2 = w.state(); print("after 2 s @0.3 m/s cmd", {k: round(v, 3) for k, v in s2.items()}, "physics %.1f us/step" % (dt / 2000 * 1e6))
print("dist %.3f m (expect ~0.55: ramp 3 turn/s^2)" % math.hypot(s2['x'] - s1['x'], s2['y'] - s1['y']))
for n in w.nodes: n.ax.input_vel = n.dir * 0.0
for _ in range(1000): w.step(time.monotonic())
for n in w.nodes: n.ax.input_vel = -n.dir * 0.3 / (2 * math.pi * R_W) * (1 if n.id == 0 else -1) * -1
y0 = w.state()['yaw']
for _ in range(1000): w.step(time.monotonic())
print("spin: yaw %.2f -> %.2f, w=%.2f rad/s" % (y0, w.state()['yaw'], w.state()['w']))
# render cost
mj = w.mj; Rd = mj.Renderer(m, W.DH, W.DW); Rd.enable_depth_rendering(); Rc = mj.Renderer(m, W.CH, W.CW); Rc8 = mj.Renderer(m, W.DH, W.DW)
for name, R, cam in (("depth848", Rd, "rs1_depth"), ("rgb640", Rc, "rs1_color"), ("rgb848", Rc8, "rs2_ir1")):
    R.update_scene(d, cam); R.render(); t = time.perf_counter()
    for _ in range(20): R.update_scene(d, cam); x = R.render()
    print("render %s %.1f ms" % (name, (time.perf_counter() - t) / 20 * 1e3))
import cv2
Rd.update_scene(d, "rs1_depth"); z = Rd.render(); print("rs1 depth centre %.3f m (floor expected ~%.3f)" % (z[240, 424], W.RS1_POS[2] + 0.0857))
Rd.update_scene(d, "rs2_ir1"); z2 = Rd.render()
for nm, cam in (("rs1_color", "rs1_color"), ("rs2_color", "rs2_color"), ("rs2_ir1", "rs2_ir1")):
    R = Rc if "color" in nm else Rc8; R.update_scene(d, cam); cv2.imwrite(os.path.expanduser("~/.kevin/emu/emu_") + "%s.png" % nm, R.render()[..., ::-1])
cv2.imwrite(os.path.expanduser("~/.kevin/emu/emu_") + "rs1_depth.png", np.clip(z / 2.0 * 255, 0, 255).astype(np.uint8)); cv2.imwrite(os.path.expanduser("~/.kevin/emu/emu_") + "rs2_depth.png", np.clip(z2 / 5.0 * 255, 0, 255).astype(np.uint8))
os._exit(0)
