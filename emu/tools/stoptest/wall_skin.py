"""Max distance of the mesh-wall collision boxes (emu.world) from the rendered wall surface, casita slab 1.  Run from the repo root."""
import sys, math, numpy as np, trimesh
sys.path.insert(0, '.'); import os; os.environ.setdefault('KEVIN_EMU_NOISE', 'realistic')
from emu import envs
e = envs.build('casita', 0)
if hasattr(envs, '_finish'): pass
r = getattr(e, 'wall_res', envs.GT_RES); (ox, oy) = e.gt_origin; m, s, off = e._mesh
zc, occ = e.wall_levels[1]
ys, xs = np.nonzero(occ)
# cell corners of occupied cells that are on the free side (outer boundary of the collision skin)
C = []
for j, i in zip(ys, xs):
    for dj, di in ((0, 0), (0, 1), (1, 0), (1, 1)):
        C.append((ox + (i + di) * r, oy + (j + dj) * r))
C = np.unique(np.round(np.array(C), 4), axis=0)
rng = np.random.default_rng(0); C = C[rng.choice(len(C), min(20000, len(C)), replace=False)]
Q = np.c_[(C - off[:2]) / s, np.full(len(C), (zc - off[2]) / s)]
_, d, _ = trimesh.proximity.closest_point(m, Q); d = d * s
# inside-wall points have small d too; the protrusion is the max over corners
print('slab z=%.2f cells=%d  corner->visual-wall distance p50 %.3f p95 %.3f p99 %.3f max %.3f m' % (zc, occ.sum(), *np.percentile(d, [50, 95, 99]), d.max()))
nb = 0
for zc_, occ_ in e.wall_levels:
    open_ = {}
    for j in range(occ_.shape[0] + 1):
        row = occ_[j] if j < occ_.shape[0] else np.zeros(occ_.shape[1], bool)
        d_ = np.diff(np.r_[0, row.astype(np.int8), 0]); runs = set(zip(np.where(d_ == 1)[0], np.where(d_ == -1)[0]))
        for k in list(open_):
            if k not in runs: open_.pop(k); nb += 1
        for k in runs: open_.setdefault(k, j)
print('collision boxes', nb, 'res', r)
