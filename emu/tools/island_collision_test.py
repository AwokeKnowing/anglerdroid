"""Offline (no live emu.world): compose casita with the twin, put Kevin at a pose and report wall contacts and
which Kevin geom touches.  python -m emu.tools.island_collision_test x y yaw [x y yaw ...]"""
import os, sys, time, math, numpy as np
os.environ.setdefault("KEVIN_EMU_MODEL", "full"); os.environ.setdefault("KEVIN_EMU_NOISE", "realistic")
import mujoco
from emu import envs, world
t0 = time.time(); e = envs.build("casita", 0); xml = world.compose(e, "stock"); m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
nw = sum(1 for i in range(m.ngeom) if (m.geom(i).name or "").startswith("envwall"))
print("build %.1f s, %d envwall boxes, %d slabs, top %.3f m" % (time.time() - t0, nw, len(e.wall_levels), e.wall_levels[-1][0] + envs.SLICE_DZ / 2))
jid = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE); qa = m.jnt_qposadr[jid]
a = [float(v) for v in sys.argv[1:]]
for x, y, yaw in zip(a[0::3], a[1::3], a[2::3]):
    mujoco.mj_resetData(m, d); d.qpos[qa:qa + 3] = [x, y, d.qpos[qa + 2]]; d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(m, d); hits = []
    for c in d.contact[:d.ncon]:
        n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
        if n1.startswith("envwall") or n2.startswith("envwall"):
            w, k = (n1, n2) if n1.startswith("envwall") else (n2, n1)
            gi = m.geom(w).id; hits.append((k or "geom%d" % (c.geom2 if w == n1 else c.geom1), w, round(float(c.dist), 4), np.round(c.pos, 3).tolist(),
                                           round(float(m.geom_pos[gi][2] - m.geom_size[gi][2]), 3), round(float(m.geom_pos[gi][2] + m.geom_size[gi][2]), 3)))
    print("pose (%.3f, %.3f, %.3f): %d wall contacts" % (x, y, yaw, len(hits)))
    for h in hits[:6]: print("   kevin %s  wall %s dist %+.4f at %s  box z [%.3f, %.3f]" % h)
