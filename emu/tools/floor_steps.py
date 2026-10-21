"""Scan the composed sim floor for steps Kevin drives into (offline, no live world).  python -m emu.tools.floor_steps [env]
1) hfield: adjacent samples differing > STEP (a one-cell ramp/cliff); 2) every floor box (tile plates, rugs): exposed
vertical face = box top minus the hfield 1 cm outside its edge (> 0 = a face Kevin's casters hit; < 0 = step down)."""
import os, sys, math, numpy as np
os.environ.setdefault("KEVIN_EMU_MODEL", "full"); os.environ.setdefault("KEVIN_EMU_NOISE", "realistic")
import mujoco
from emu import envs, world
STEP = 0.006
name = sys.argv[1] if len(sys.argv) > 1 else "casita"
e = envs.build(name, 0); m = mujoco.MjModel.from_xml_string(world.compose(e, "stock"))
g = m.geom("floor_uneven").id; h = m.geom_dataid[g]; nr, nc = m.hfield_nrow[h], m.hfield_ncol[h]; sx, sy, sz, _ = m.hfield_size[h]
Z = m.hfield_data[m.hfield_adr[h]:m.hfield_adr[h] + nr * nc].reshape(nr, nc) * sz + m.geom_pos[g][2]   # row 0 = -y (MuJoCo)
X0, Y0 = m.geom_pos[g][0] - sx, m.geom_pos[g][1] - sy; dx, dy = 2 * sx / (nc - 1), 2 * sy / (nr - 1)
def hz(x, y):
    c, r = (x - X0) / dx, (y - Y0) / dy; c0, r0 = int(np.clip(math.floor(c), 0, nc - 2)), int(np.clip(math.floor(r), 0, nr - 2)); fc, fr = c - c0, r - r0
    return float((1 - fr) * ((1 - fc) * Z[r0, c0] + fc * Z[r0, c0 + 1]) + fr * ((1 - fc) * Z[r0 + 1, c0] + fc * Z[r0 + 1, c0 + 1]))
dX, dY = np.abs(np.diff(Z, axis=1)), np.abs(np.diff(Z, axis=0))
print("hfield %dx%d, cell %.3f m: adjacent-sample steps > %.0f mm: %d along x, %d along y (max %.1f / %.1f mm)" % (nr, nc, dx, STEP * 1e3, (dX > STEP).sum(), (dY > STEP).sum(), dX.max() * 1e3, dY.max() * 1e3))
for i in range(m.ngeom):
    n = m.geom(i).name or ""
    if not n.startswith("floor_") or m.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX: continue
    p, s = m.geom_pos[i], m.geom_size[i]; q = m.geom_quat[i]; yaw = 2 * math.atan2(q[3], q[0]); c, si = math.cos(yaw), math.sin(yaw); top = p[2] + s[2]
    F = []
    for u in np.linspace(-1, 1, 80):
        for (lx, ly) in ((u * s[0], s[1] + 0.01), (u * s[0], -s[1] - 0.01), (s[0] + 0.01, u * s[1]), (-s[0] - 0.01, u * s[1])):
            x, y = p[0] + c * lx - si * ly, p[1] + si * lx + c * ly
            if X0 <= x <= X0 + 2 * sx and Y0 <= y <= Y0 + 2 * sy: F.append(top - hz(x, y))
    F = np.array(F); print("%-12s top %+.4f  exposed edge face: max %+.1f mm, median %+.1f mm, %d/%d edge samples > 3 mm" % (n, top, F.max() * 1e3, np.median(F) * 1e3, (F > 0.003).sum(), len(F)))
