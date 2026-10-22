import sys, math, numpy as np, cv2, warnings; warnings.filterwarnings("ignore")
from bench import make, VL, rel, compose, Q
n = sys.argv[1]; a, b = int(sys.argv[2]), int(sys.argv[3]); loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g, od = z["gt"], z["odom"]
p = g[0].copy(); P = np.diag([0.02**2, 0.02**2, math.radians(1)**2])
for i in range(1, b):
    d = rel(od[i-1], od[i]); p = compose(p, d); c, s = math.cos(p[2]), math.sin(p[2]); R = np.array([[c,-s,0],[s,c,0],[0,0,1]]); P = P + R @ Q(d) @ R.T
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); eb = math.hypot(*(p[:2]-g[i][:2])); q, Pq, info = loc.update(im, p, P)
    if info.get("ok"): p, P = q, Pq
    if i >= a: print(i, "%.1f %.1f %.1f" % (eb*100, math.hypot(*(p[:2]-g[i][:2]))*100, math.sqrt(P[0,0]+P[1,1])*100), info.get("ok"), info.get("why"), info.get("lost"), "nVobs", info["nV_obs"], "nV", info.get("nV"), "nis", round(info.get("nis", -1), 1), "nH", info.get("nH"))
