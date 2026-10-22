import sys, math, numpy as np, cv2, warnings, collections; warnings.filterwarnings("ignore")
from bench import make, VL
n = sys.argv[1]; loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]
Pt = np.diag([0.003**2]*2 + [math.radians(.1)**2]); D = collections.defaultdict(list)
for i in range(1, len(g), 2):
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); Vobs, Hobs = loc.observe(loc.segments(im))
    vo, vi, _, _, _ = loc.associate(g[i], Pt, Vobs, Hobs); r = loc.residuals(g[i], Vobs[vo], vi, Hobs[:0], np.zeros(0, int))
    _, rng = loc.pred_v(g[i], vi)
    for k, rr, d in zip(vi, r, rng): D[k].append((rr, d))
print("corner  x      y     z0   z1    n   mean_px  sd_px  range_m")
for k in sorted(D, key=lambda k: -len(D[k]))[:30]:
    a = np.array(D[k]); x, y, z0, z1 = loc.m.V[k]; print(f"{k:4d} {x:6.3f} {y:6.3f} {z0:4.2f} {z1:4.2f} {len(a):4d} {a[:,0].mean():+6.2f} {a[:,0].std():5.2f} {a[:,1].mean():5.2f}")
