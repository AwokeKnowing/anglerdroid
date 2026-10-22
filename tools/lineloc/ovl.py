import sys, math, numpy as np, cv2
from bench import make, VL
from lineloc import se2_cam
n, i = sys.argv[1], int(sys.argv[2]); loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"][i]; cam = loc.cam
im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg"); segs = loc.segments(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
c, R = se2_cam(g, cam); Rcw = (R @ cam.R_bc).T; vv, hh = loc.m.visible(g[0], g[1])
def proj(X):
    Xc = (X - c) @ Rcw.T; ok = Xc[:, 2] > 0.05; return cam.distort_norm(Xc[:, :2] / np.maximum(Xc[:, 2:], 0.05)), ok
for k in vv:
    x, y, z0, z1 = loc.m.V[k]; X = np.c_[np.full(20, x), np.full(20, y), np.linspace(z0, z1, 20)]; uv, ok = proj(X)
    for a, b, o1, o2 in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
        if o1 and o2: cv2.line(im, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 255), 1)
for k in hh:
    X = loc.m.HA[k] + np.linspace(0, 1, 30)[:, None] * (loc.m.HB[k] - loc.m.HA[k]); uv, ok = proj(X)
    for a, b, o1, o2 in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
        if o1 and o2 and np.abs(a).max() < 2000: cv2.line(im, tuple(map(int, a)), tuple(map(int, b)), (0, 200, 255), 1)
for s in segs: cv2.line(im, tuple(map(int, s[:2])), tuple(map(int, s[2:])), (255, 100, 0), 1)
cv2.imwrite(sys.argv[3], cv2.resize(im, None, fx=1.5, fy=1.5))
