import sys, math, numpy as np, cv2, warnings; warnings.filterwarnings("ignore")
from bench import make, VL
n = sys.argv[1]; loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]
Pt = np.diag([0.001**2]*2 + [math.radians(.03)**2]); rows = []
for i in range(1, len(g), 10):
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); Vobs, Hobs = loc.observe(loc.segments(im))
    vo, vi, _, ho, hi = loc.associate(g[i], Pt, Vobs, Hobs); r = loc.residuals(g[i], Vobs[:0], np.zeros(0,int), Hobs[ho], hi).reshape(-1, 2)
    zz = loc.m.HA[hi, 2]
    for rr, zh in zip(r, zz): rows.append([rr[0], rr[1], zh])
R = np.array(rows); a = np.abs(R[:, :2]).max(1)
print("n", len(R), "abs max-endpoint pct 50/90/99", np.percentile(a, [50, 90, 99]).round(2))
for lab, m in [("floor z<0.02", R[:,2] < 0.02), ("ceiling z>2.2", R[:,2] > 2.2), ("other", (R[:,2] >= 0.02) & (R[:,2] <= 2.2))]:
    print(lab, m.sum(), np.percentile(a[m], [50, 90]).round(2) if m.sum() else "")
