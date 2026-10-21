"""Offline floor-drive test (no live emu.world): compose casita + twin, drive Kevin straight with the twin ODrive
model (ramp 3 turns/s^2, PI, current limits) and report peak body speed / yaw rate / roll-pitch.
python -m emu.tools.floor_drive_test V X0 Y0 YAW DIST"""
import os, sys, math, numpy as np
os.environ.setdefault("KEVIN_EMU_MODEL", "full"); os.environ.setdefault("KEVIN_EMU_NOISE", "realistic")
import mujoco
from emu import envs, world
sys.path.insert(0, world.TWIN); from kevin_drive import ODriveAxis
RP = world._robot_params(); WHEELS = RP.V["wheels"]   # resolved robot config (KEVIN_WHEELS to change)
def run(m, V, x0, y0, yaw, dist):
    d = mujoco.MjData(m); m.opt.noslip_iterations = 10
    jl, jr = m.joint("wheel_left_joint"), m.joint("wheel_right_joint")
    for j in (jl, jr): m.dof_frictionloss[j.dofadr[0]] = max(float(m.dof_frictionloss[j.dofadr[0]]), 0.1)
    al, ar = m.actuator("wheel_left_motor").id, m.actuator("wheel_right_motor").id
    fj = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE); qa, va = m.jnt_qposadr[fj], m.jnt_dofadr[fj]
    if m.nkey: mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[qa:qa + 2] = [x0, y0]; d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]; mujoco.mj_forward(m, d)
    R = d.xmat[m.body("base_link").id].reshape(3, 3)
    sg = [float(np.sign((d.xmat[m.jnt_bodyid[j.id]].reshape(3, 3) @ m.jnt_axis[j.id]) @ R[:, 1])) for j in (jl, jr)]
    ax = [ODriveAxis(), ODriveAxis()]; wr = RP.V["wheel_r_m"]
    for _ in range(800): mujoco.mj_step(m, d)                                   # settle
    for a in ax: a.input_vel = V / (2 * math.pi * wr)
    p0 = d.qpos[qa:qa + 2].copy(); pk = dict(v=0, w=0, tilt=0)
    for k in range(int((dist / max(V, 0.05) + 2.0) / m.opt.timestep)):
        for a, j, act, s in ((ax[0], jl, al, sg[0]), (ax[1], jr, ar, sg[1])): d.ctrl[act] = s * a.update(s * d.qvel[j.dofadr[0]] / (2 * math.pi), m.opt.timestep)
        mujoco.mj_step(m, d)
        if np.hypot(*(d.qpos[qa:qa + 2] - p0)) > dist:
            for a in ax: a.input_vel = 0.0
        q = d.qpos[qa + 3:qa + 7]; tilt = math.degrees(math.acos(min(1, abs(1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
        pk['v'] = max(pk['v'], float(np.hypot(*d.qvel[va:va + 2]))); pk['w'] = max(pk['w'], abs(float(d.qvel[va + 5]))); pk['tilt'] = max(pk['tilt'], tilt)
    return pk, d.qpos[qa:qa + 2].copy()
if __name__ == "__main__":
    V, x0, y0, yaw, dist = (float(a) for a in sys.argv[1:6])
    e = envs.build("casita", 0); m = mujoco.MjModel.from_xml_string(world.compose(e, WHEELS))
    pk, end = run(m, V, x0, y0, yaw, dist)
    print("V %.2f from (%.2f,%.2f,%.2f) %.1f m: peak |v| %.2f m/s, |w| %.2f rad/s, tilt %.1f deg, end (%.2f,%.2f)" % (V, x0, y0, yaw, dist, pk['v'], pk['w'], pk['tilt'], *end))
