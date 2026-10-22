import sys, math, numpy as np, cv2, warnings; warnings.filterwarnings("ignore")
from bench import make, VL
from lineloc import wrap
n = sys.argv[1]; loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]; t = z["t"]
v = np.r_[0, np.hypot(*np.diff(g[:, :2], axis=0).T) / np.diff(t)]; w = np.r_[0, np.abs(wrap(np.diff(g[:, 2]))) / np.diff(t)]
rng = np.random.default_rng(0); rows = []
for i in range(1, len(g), 3):
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); P = np.diag([0.03**2, 0.03**2, math.radians(1)**2])
    pr = g[i] + rng.multivariate_normal(np.zeros(3), P); x, Pp, info = loc.update(im, pr, P)
    d = x - g[i]; c, s = math.cos(g[i][2]), math.sin(g[i][2])
    rows.append([v[i], w[i], c*d[0]+s*d[1], -s*d[0]+c*d[1], math.degrees(wrap(d[2])), info.get("ok", False)])
R = np.array(rows, float); ok = R[:, 5] > 0
for name, m in [("still", (R[:,0] < 0.01) & (R[:,1] < 0.02)), ("translating", (R[:,0] >= 0.01) & (R[:,1] < 0.05)), ("turning", R[:,1] >= 0.05)]:
    m = m & ok
    if m.sum(): print(f"{n} {name:12s} n={m.sum():3d} along {np.mean(R[m,2])*100:+.2f}±{np.std(R[m,2])*100:.2f} lat {np.mean(R[m,3])*100:+.2f}±{np.std(R[m,3])*100:.2f} cm yaw {np.mean(R[m,4]):+.3f}±{np.std(R[m,4]):.3f} | v {np.median(R[m,0]):.3f} w {np.median(R[m,1]):.3f}")
