import sys, math, numpy as np, cv2, warnings; warnings.filterwarnings("ignore")
from bench import make, VL
from lineloc import wrap
n = sys.argv[1]; loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]
Pt = np.diag([0.003**2]*2 + [math.radians(.1)**2])
for i in range(1, len(g), 25):
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); Vobs, Hobs = loc.observe(loc.segments(im))
    vo, vi, _, _, _ = loc.associate(g[i], Pt, Vobs, Hobs); a = (Vobs[vo], vi, Hobs[:0], np.zeros(0, int)); x = g[i].copy()
    for it in range(10):
        r, J = loc.jac(x, *a); w = np.minimum(1, 1.345 / np.maximum(np.abs(r) / 1.5, 1e-9)); H = J.T @ (J * w[:, None]); x = x - np.linalg.solve(H, J.T @ (w * r))
    r, J = loc.jac(x, *a); C = np.linalg.inv(J.T @ J) * 1.0
    print(i, "nV", len(vi), "err %.2f cm %.3f deg" % (math.hypot(*(x[:2]-g[i][:2]))*100, math.degrees(wrap(x[2]-g[i][2]))), "pred sd(1px) %.2f %.2f cm %.3f deg" % (math.sqrt(C[0,0])*100, math.sqrt(C[1,1])*100, math.degrees(math.sqrt(C[2,2]))), "rms %.2f" % np.sqrt(np.mean(r**2)), "r0", np.round(loc.residuals(g[i], *a), 1))
