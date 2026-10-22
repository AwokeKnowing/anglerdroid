import sys, math, numpy as np, cv2, os, warnings; warnings.filterwarnings("ignore")
import bench
from bench import make, VL, rel, compose, Q
from lineloc import wrap
n = sys.argv[1]; loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g, od = z["gt"], z["odom"]
p = g[0].copy(); P = np.diag([0.02**2, 0.02**2, math.radians(1)**2]); N = int(sys.argv[2]) if len(sys.argv) > 2 else len(g)
for i in range(1, N):
    d = rel(od[i-1], od[i]); p = compose(p, d); c, s = math.cos(p[2]), math.sin(p[2]); R = np.array([[c,-s,0],[s,c,0],[0,0,1]]); P = P + R @ Q(d) @ R.T
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); eb = math.hypot(*(p[:2]-g[i][:2]))
    q, Pq, info = loc.update(im, p, P)
    if info.get("ok"):
        ea = math.hypot(*(q[:2]-g[i][:2]))
        if info["lost"] or ea > eb + 0.03: print(i, "lost" if info["lost"] else "", "err %.1f -> %.1f cm" % (eb*100, ea*100), {k: (round(v,2) if isinstance(v,float) else v) for k, v in info.items()})
        p, P = q, Pq
