"""True 3-D clearance of Kevin to the sim world (casita), for benches.  1 cm fields, cached.

Kevin by height band (twin kevin_full.xml, exact mesh cross-sections at 0.30/0.50/0.69/0.72/0.745/0.80/0.90 m,
2026-10-17; axle frame, x fwd): BANDS below.  Obstacles: mesh walls sliced every 1 cm + furniture geoms, one field
per band (obstacle parts within the band's heights).  gap = min over bands (m; -res/2 = touching).
gap2d() is the 2-D view the nav stack uses (SIL12 vs everything up to ROBOT_TOP_M)."""
import os, sys, math, numpy as np
def _box(x0, x1, y0, y1): return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
SIL12 = np.array([[-0.27371, -0.01022], [-0.16869, -0.15427], [-0.08009, -0.20998], [0.08319, -0.20998], [0.12372, -0.15418], [0.12629, -0.13818],
                  [0.12629, 0.13818], [0.12372, 0.15418], [0.08319, 0.20998], [-0.08009, 0.20998], [-0.16869, 0.15427], [-0.27371, 0.01022]])
UPPER = np.array([[0.03, -0.007], [0.008, -0.152], [-0.145, -0.154], [-0.20, -0.15], [-0.20, 0.15], [-0.145, 0.154], [0.008, 0.152], [0.03, 0.007]])
C = os.path.expanduser('~/.cache/kevin_emu/gtgap_casita_v4.npz')
def _samples(poly, sp=0.01):
    P = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]; n = max(1, int(np.ceil(np.linalg.norm(b - a) / sp)))
        P += [a + (b - a) * t / n for t in range(n)]
    return np.array(P)
BANDS = [(0.03, 0.20, SIL12),                         # chassis, wheels, casters, deck (0.177-0.19)
         (0.20, 0.30, UPPER),                         # mast base / electronics above the deck
         (0.30, 0.50, _box(-0.19, 0.03, -0.097, 0.097)),
         (0.50, 0.78, _box(-0.19, -0.15, -0.010, 0.010)), # mast column only (island top 0.684-0.744 is here)
         (0.78, 1.02, _box(-0.19, 0.036, -0.039, 0.039))] # mast head + RS1 (twin top 1.006)
HB = [_samples(b[2]) for b in BANDS]; HS = _samples(SIL12)
def load():
    if os.path.exists(C):
        z = np.load(C); return list(z['bands']), z['all'], tuple(z['org']), float(z['res'])
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); os.environ.setdefault('KEVIN_EMU_NOISE', 'realistic')
    from emu import envs
    from scipy import ndimage
    import trimesh
    envs.GT_RES = 0.01; res = 0.01
    e = envs.build('casita', 0); org = e.gt_origin; m, s, off = e._mesh
    shp = e.wall_levels[0][1].shape
    def field(z0, z1):
        occ = np.zeros(shp, bool)
        for zc in np.arange(z0, z1 + 1e-9, 0.01):
            for seg in trimesh.intersections.mesh_plane(m, plane_normal=[0, 0, 1], plane_origin=[0, 0, (zc - off[2]) / s]):
                P = seg[:, :2] * s + off[:2]; n = int(np.hypot(*(P[1] - P[0])) / (res / 2)) + 2
                q = P[0] + (P[1] - P[0]) * np.linspace(0, 1, n)[:, None]; c = ((q - org) / res).astype(int)
                ok = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < shp[1]) & (c[:, 1] < shp[0]); occ[c[ok, 1], c[ok, 0]] = True
        b0 = envs.BAND; envs.BAND = (z0, z1)
        for g in e.geoms:
            if g.kind == 'box': envs._raster_box(g, occ, org)
            elif g.kind == 'cyl': envs._raster_cyl(g, occ, org)
        envs.BAND = b0
        return (ndimage.distance_transform_edt(~occ) * res - res / 2).astype(np.float32)
    F = [field(z0, z1) for z0, z1, _ in BANDS]; al = field(envs.BAND[0], envs.ROBOT_TOP_M)
    os.makedirs(os.path.dirname(C), exist_ok=True); np.savez_compressed(C, bands=np.array(F), all=al, org=np.array(org), res=res)
    return F, al, org, res
def _g(D, org, res, H, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw); P = H @ np.array([[c, s], [-s, c]]) + [x, y]
    ij = ((P - org) / res).astype(int); ij[:, 0] = ij[:, 0].clip(0, D.shape[1] - 1); ij[:, 1] = ij[:, 1].clip(0, D.shape[0] - 1)
    return float(D[ij[:, 1], ij[:, 0]].min())
def gap(F, x, y, yaw):
    B, al, org, res = F; return min(_g(D, org, res, H, x, y, yaw) for D, H in zip(B, HB))
def gap2d(F, x, y, yaw):
    B, al, org, res = F; return _g(al, org, res, HS, x, y, yaw)
if __name__ == '__main__':
    F = load()
    for p in [(3.956, 5.483, -0.131), (4.20, 5.45, -0.131), (4.26, 5.44, -0.131), (4.8, 1.0, 1.5708)]: print(p, '3d %.3f  2d %.3f' % (gap(F, *p), gap2d(F, *p)))
