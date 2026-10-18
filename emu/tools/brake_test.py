"""Offline: max braking of Kevin (twin ODrive model, casita carpet at (2.5,2.4) heading +y).  Drive at V, then
command 0 with vel_ramp_rate R (turns/s^2): measured stop distance / time, mean + peak decel, peak pitch, wheel slip.
python -m emu.tools.brake_test"""
import os, sys, math, numpy as np
os.environ.setdefault("KEVIN_EMU_MODEL", "full"); os.environ.setdefault("KEVIN_EMU_NOISE", "realistic")
import mujoco
from emu import envs, world
sys.path.insert(0, world.TWIN); from kevin_drive import ODriveAxis
WR = 0.08565
def brake(m, V, R, x0=2.5, y0=2.0, yaw=math.pi / 2):
    d = mujoco.MjData(m); m.opt.noslip_iterations = 10
    jl, jr = m.joint("wheel_left_joint"), m.joint("wheel_right_joint")
    for j in (jl, jr): m.dof_frictionloss[j.dofadr[0]] = max(float(m.dof_frictionloss[j.dofadr[0]]), 0.1)
    al, ar = m.actuator("wheel_left_motor").id, m.actuator("wheel_right_motor").id
    fj = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE); qa, va = m.jnt_qposadr[fj], m.jnt_dofadr[fj]
    if m.nkey: mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[qa:qa + 2] = [x0, y0]; d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]; mujoco.mj_forward(m, d)
    Rm = d.xmat[m.body("base_link").id].reshape(3, 3)
    sg = [float(np.sign((d.xmat[m.jnt_bodyid[j.id]].reshape(3, 3) @ m.jnt_axis[j.id]) @ Rm[:, 1])) for j in (jl, jr)]
    ax = [ODriveAxis(), ODriveAxis()]
    for _ in range(800): mujoco.mj_step(m, d)
    for a in ax: a.input_vel = V / (2 * math.pi * WR)
    dt = m.opt.timestep; t = 0; out = None; T = []
    while t < 6.0:
        if out is None and t > 2.5:                            # cruise 2.5 s, then brake
            for a in ax: a.ramp = R; a.input_vel = 0.0
            out = dict(t0=t, p0=d.qpos[qa:qa + 2].copy(), v0=None)
        for a, j, act, s in ((ax[0], jl, al, sg[0]), (ax[1], jr, ar, sg[1])): d.ctrl[act] = s * a.update(s * d.qvel[j.dofadr[0]] / (2 * math.pi), dt)
        mujoco.mj_step(m, d); t += dt
        R_ = d.xmat[m.body("base_link").id].reshape(3, 3); vb = float((R_.T @ d.qvel[va:va + 3])[0])
        q = d.qpos[qa + 3:qa + 7]; pitch = math.degrees(math.asin(max(-1, min(1, 2 * (q[0] * q[2] - q[3] * q[1])))))
        wv = 0.5 * (sg[0] * d.qvel[jl.dofadr[0]] + sg[1] * d.qvel[jr.dofadr[0]]) * WR
        T.append((t, vb, wv, pitch, R_[2, 2]))
    T = np.array(T); b = T[:, 0] >= out['t0']; v0 = T[b][0, 1]
    st = T[b & (np.abs(T[:, 1]) < 0.01)]; ts = (st[0, 0] - out['t0']) if len(st) else float('nan')
    Tb = T[b]; i1 = np.argmax(np.abs(Tb[:, 1]) < 0.01) if (np.abs(Tb[:, 1]) < 0.01).any() else len(Tb) - 1
    dist = float(np.trapezoid(Tb[:i1, 1], Tb[:i1, 0])); acc = np.gradient(Tb[:i1 + 1, 1], Tb[:i1 + 1, 0]) if i1 > 2 else np.zeros(1)
    acc_s = np.convolve(acc, np.ones(20) / 20, mode='same')
    slip = float(np.max(np.abs(Tb[:i1 + 1, 1] - Tb[:i1 + 1, 2]))) if i1 > 0 else 0
    return dict(V=V, R=R, a_cmd=R * 2 * math.pi * WR, v0=v0, t_stop=ts, d_stop=dist, a_mean=v0 * v0 / (2 * dist) if dist > 0 else float('nan'),
                a_peak=float(-acc_s.min()), pitch_max=float(np.abs(Tb[:, 3]).max()), slip=slip, tipped=bool(Tb[:, 4].min() < 0.7))
if __name__ == "__main__":
    e = envs.build("casita", 0); m = mujoco.MjModel.from_xml_string(world.compose(e, "stock"))
    for V in (0.3, 0.45, 0.6):
        for R in (3.0, 5.0, 8.0, 12.0, 20.0):
            r = brake(m, V, R)
            print("V %.2f ramp %4.1f t/s2 (cmd %.2f m/s2): v0 %.3f stop %.3f s %.3f m  mean decel %.2f peak %.2f m/s2  pitch max %.1f deg  wheel-body slip max %.3f m/s  tipped %s" % (
                r['V'], r['R'], r['a_cmd'], r['v0'], r['t_stop'], r['d_stop'], r['a_mean'], r['a_peak'], r['pitch_max'], r['slip'], r['tipped']), flush=True)
