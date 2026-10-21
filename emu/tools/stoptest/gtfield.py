"""Exact-ish GT obstacle distance field for casita (1 cm): mesh walls (WALL_RES slabs, undilated) + furniture boxes/cylinders.
Cached to gtfield_casita_v2.npz.  dist(x,y) = metres to the nearest obstacle cell centre (-0.005 for the cell half-size)."""
import os, sys, numpy as np
C = os.path.expanduser('~/m1_backup/dbg/gtfield_casita_v2.npz')
def load():
    if os.path.exists(C):
        z = np.load(C); return z['d'], tuple(z['org']), float(z['res'])
    sys.path.insert(0, os.path.expanduser('~/gits/awokeknowing/anglerdroid-slambench')); os.environ.setdefault('KEVIN_EMU_NOISE', 'realistic')
    from emu import envs
    from scipy import ndimage
    envs.GT_RES = 0.01
    e = envs.build('casita', 0); org = e.gt_origin; res = 0.01
    occ = np.zeros_like(e.wall_levels[0][1])
    for _, lv in e.wall_levels: occ |= lv
    # v2 (2026-10-17): emu.envs slices the mesh at 4 heights only (0.08/0.35/0.63/0.90 m), which misses the casita
    # island top (z 0.684-0.744 m).  Slice every 1 cm over the whole robot band (envs.BAND) for the true gap.
    import trimesh
    m, s, off = e._mesh; wr = e.wall_res
    for zc in np.arange(envs.BAND[0], envs.BAND[1] + 1e-9, 0.01):
        for seg in trimesh.intersections.mesh_plane(m, plane_normal=[0, 0, 1], plane_origin=[0, 0, (zc - off[2]) / s]):
            P = seg[:, :2] * s + off[:2]; a_, b_ = P[0], P[1]; n = int(np.hypot(*(b_ - a_)) / (res / 2)) + 2
            q = a_ + (b_ - a_) * np.linspace(0, 1, n)[:, None]; c = ((q - org) / res).astype(int)
            ok = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < occ.shape[1]) & (c[:, 1] < occ.shape[0]); occ[c[ok, 1], c[ok, 0]] = True
    for g in e.geoms:
        if g.kind == 'box': envs._raster_box(g, occ, org)
        elif g.kind == 'cyl': envs._raster_cyl(g, occ, org)
    d = ndimage.distance_transform_edt(~occ) * res - res / 2
    np.savez_compressed(C, d=d.astype(np.float32), org=np.array(org), res=res); return d, org, res
SIL12 = np.array([[-0.27371, -0.01022], [-0.1995, -0.15], [-0.08009, -0.20998], [0.08319, -0.20998], [0.12372, -0.15418], [0.12629, -0.13818], [0.12629, 0.13818], [0.12372, 0.15418], [0.08319, 0.20998], [-0.08009, 0.20998], [-0.1995, 0.15], [-0.27371, 0.01022]])
def hull_samples(sp=0.01):
    P = []
    for i in range(len(SIL12)):
        a, b = SIL12[i], SIL12[(i + 1) % len(SIL12)]; n = max(1, int(np.ceil(np.linalg.norm(b - a) / sp)))
        P += [a + (b - a) * t / n for t in range(n)]
    return np.array(P)
HS = hull_samples()
def gap(D, org, res, x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw); P = HS @ np.array([[c, s], [-s, c]]) + [x, y]
    ij = ((P - org) / res).astype(int); ij[:, 0] = ij[:, 0].clip(0, D.shape[1] - 1); ij[:, 1] = ij[:, 1].clip(0, D.shape[0] - 1)
    return float(D[ij[:, 1], ij[:, 0]].min())
if __name__ == '__main__':
    D, org, res = load(); print(D.shape, org, res)
    import json; S = json.load(open('/dev/shm/kevin_sim.json')); x, y, a = S['start']; print('gap at start', gap(D, org, res, x, y, a))
