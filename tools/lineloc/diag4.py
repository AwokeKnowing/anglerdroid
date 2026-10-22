import sys, math, numpy as np, cv2, warnings; warnings.filterwarnings("ignore")
from bench import make, VL
from lineloc import wrap
n = sys.argv[1]; loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]; t = z["t"]
v = np.r_[0, np.hypot(*np.diff(g[:, :2], axis=0).T)]; rows = []
Pt = np.diag([0.005**2]*2 + [math.radians(.2)**2])
for i in range(1, len(g), 4):
    if v[i] > 1e-4 or abs(wrap(g[i,2]-g[i-1,2])) > 1e-4: continue
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); Vobs, Hobs = loc.observe(loc.segments(im))
    vo, vi, _, ho, hi = loc.associate(g[i], Pt, Vobs, Hobs); a = (Vobs[vo], vi, Hobs[ho], hi); x = g[i].copy()
    for it in range(10):
        r, J = loc.jac(x, *a); w = np.minimum(1, 1.345 / np.maximum(np.abs(r) / 1.5, 1e-9)); H = J.T @ (J * w[:, None]); x = x - np.linalg.solve(H + 1e-9*np.eye(3), J.T @ (w * r))
    d = x - g[i]; c, s = math.cos(g[i][2]), math.sin(g[i][2]); rows.append([c*d[0]+s*d[1], -s*d[0]+c*d[1], math.degrees(wrap(d[2])), len(vi), len(hi)])
R = np.array(rows); print(n, "still frames", len(R), "GN from GT, no prior: along %+.2f±%.2f lat %+.2f±%.2f cm yaw %+.3f±%.3f  nV %.0f nH %.0f" % (R[:,0].mean()*100, R[:,0].std()*100, R[:,1].mean()*100, R[:,1].std()*100, R[:,2].mean(), R[:,2].std(), R[:,3].mean(), R[:,4].mean()))
