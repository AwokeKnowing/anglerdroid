import sys, math, numpy as np, cv2, warnings, collections; warnings.filterwarnings("ignore")
from bench import make, VL
n = sys.argv[1]; a, b = int(sys.argv[2]), int(sys.argv[3]); loc = make(); loc.h_junction = True; loc.h_minlen = 40; z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]
Pt = np.diag([0.001**2]*2 + [math.radians(.03)**2]); D = collections.defaultdict(list)
for i in range(a, b, 3):
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); Vobs, Hobs = loc.observe(loc.segments(im))
    _, _, _, ho, hi = loc.associate(g[i], Pt, Vobs, Hobs)
    r = loc.residuals(g[i], Vobs[:0], np.zeros(0, int), Hobs[ho], hi).reshape(-1, 2)
    c = loc.m.HA[hi]; 
    for k, rr in zip(hi, r): D[k].append(rr.mean())
for k in sorted(D, key=lambda k: -len(D[k])):
    v = np.array(D[k]); print(k, loc.m.HA[k].round(3), loc.m.HB[k].round(3), len(v), "%+.2f ± %.2f px" % (v.mean(), v.std()))
