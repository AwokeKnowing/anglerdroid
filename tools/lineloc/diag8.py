import sys, math, numpy as np, cv2, warnings; warnings.filterwarnings("ignore")
from bench import make, VL
n = sys.argv[1]; loc = make(); loc.h_junction = True; z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]
Pt = np.diag([0.001**2]*2 + [math.radians(.03)**2]); rows = []
for i in range(1, len(g), 7):
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); Vobs, Hobs = loc.observe(loc.segments(im))
    _, _, _, ho, hi = loc.associate(g[i], Pt, Vobs, Hobs)
    if not len(hi): continue
    seg, _ = loc.proj_h(g[i], hi); d, _ = loc.line_dist(seg, Hobs[ho])
    # sign convention: positive = observed line BELOW the predicted one (image y down)
    dm = seg[:, 1] - seg[:, 0]; nrm = np.c_[-dm[:, 1], dm[:, 0]]; sgn = np.sign(nrm[:, 1])
    for k, dd, sg in zip(hi, d.mean(1) * loc.cam.fx, sgn): rows.append([loc.m.HA[k, 2], dd * sg])
R = np.array(rows)
for lab, m in [("floor", R[:, 0] < 0.02), ("ceiling", R[:, 0] > 2.2)]:
    print(n, lab, m.sum(), "mean below-pred px %+.2f  median %+.2f" % (R[m, 1].mean(), np.median(R[m, 1])))
