"""Offline: drive Kevin (twin ODrive model) across every hard-zone (tile plate) edge and rug edge of casita at 0.34
and 0.6 m/s where a 1.4 m corridor is free; report peak |v|, |w|, tilt.  python -m emu.tools.floor_edges_test [label filter]"""
import os; os.environ.setdefault("KEVIN_EMU_MODEL", "full"); os.environ.setdefault("KEVIN_EMU_NOISE", "realistic")
import mujoco, numpy as np, math
from emu import envs, world
from emu.tools import floor_drive_test as T
e = envs.build("casita", 0); m = mujoco.MjModel.from_xml_string(world.compose(e, "stock"))
G = np.asarray(e.gt)
def free(x, y, r=0.32):
    for a in np.linspace(0, 2*math.pi, 16, endpoint=False):
        for rr in (0, r/2, r):
            rc = e.gt_world_to_rc(x + rr*math.cos(a), y + rr*math.sin(a))
            r_, c_ = int(rc[0]), int(rc[1])
            if not (0 <= r_ < G.shape[0] and 0 <= c_ < G.shape[1]) or G[r_, c_]: return False
    return True
def pick(axis, edge, lo, hi, yaw):   # crossing an edge line: axis 'y' => line y=edge, scan x in [lo,hi]
    for s in np.arange(lo, hi, 0.05):
        pts = [(s, edge + t) if axis == 'y' else (edge + t, s) for t in np.arange(-0.7, 0.71, 0.1)]
        if all(free(*p) for p in pts):
            x, y = (s, edge) if axis == 'y' else (edge, s); d = 0.6
            return (x - d*math.cos(yaw), y - d*math.sin(yaw))
specs = [("hard0 S edge y=3.0 N-bound", 'y', 3.0, 3.45, 6.95, 1.5708), ("hard0 S edge y=3.0 S-bound", 'y', 3.0, 3.45, 6.95, -1.5708),
         ("hard0 W edge x=3.4 E-bound", 'x', 3.4, 3.05, 7.25, 0.0), ("hard0 E edge x=7.0 W-bound", 'x', 7.0, 3.05, 7.25, 3.1416),
         ("hard1 W edge x=4.6 E-bound", 'x', 4.6, 7.35, 8.35, 0.0), ("hard1 N edge y=8.4 S-bound", 'y', 8.4, 4.65, 6.95, -1.5708),
         ("hard2 N edge y=4.9 S-bound", 'y', 4.9, 0.05, 1.25, -1.5708), ("hard2 S edge y=3.0 N-bound", 'y', 3.0, 0.05, 1.25, 1.5708),
         ("hard2 E edge x=1.3 W-bound", 'x', 1.3, 3.05, 4.85, 3.1416)]
for i in range(m.ngeom):
    n = m.geom(i).name or ""
    if n.startswith("floor_rug"):
        p, s = m.geom_pos[i], m.geom_size[i]; specs.append((n + " S edge N-bound", 'y', p[1] - s[1], p[0] - s[0] + 0.05, p[0] + s[0] - 0.05, 1.5708))
import sys
only = sys.argv[1] if len(sys.argv) > 1 else ""
for lab, axis, edge, lo, hi, yaw in [s for s in specs if only in s[0]]:
    st = pick(axis, edge, lo, hi, yaw)
    if st is None: print("%-28s no free 1.4 m corridor" % lab); continue
    for V in (0.34, 0.6):
        try: pk, end = T.run(m, V, st[0], st[1], yaw, 1.2)
        except Exception as ex: print(lab, "sim error", str(ex)[:60]); continue
        print("%-28s from (%.2f,%.2f) V %.2f: peak |v| %.2f m/s |w| %.2f rad/s tilt %4.1f deg  moved %.2f m" % (lab, st[0], st[1], V, pk['v'], pk['w'], pk['tilt'], math.hypot(end[0]-st[0], end[1]-st[1])), flush=True)
