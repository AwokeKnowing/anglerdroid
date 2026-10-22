import sys, math, numpy as np, cv2, warnings; warnings.filterwarnings("ignore")
from bench import make, VL
from lineloc import wrap
n = sys.argv[1]; loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]
rng = np.random.default_rng(0)
for i in range(1, 200, 12):
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); P = np.diag([0.03**2, 0.03**2, math.radians(1)**2])
    pr = g[i] + rng.multivariate_normal(np.zeros(3), P); x, Pp, info = loc.update(im, pr, P)
    Vobs, Hobs = loc.observe(loc.segments(im))
    vo, vi, c, _, _ = loc.associate(pr, P, Vobs, Hobs); vo2, vi2, _, _, _ = loc.associate(g[i], np.diag([0.005**2]*2+[math.radians(.2)**2]), Vobs, Hobs)
    agree = len(set(zip(vo, vi)) & set(zip(vo2, vi2)))
    print(i, "prior err %.1f cm -> %.1f cm" % (math.hypot(*(pr[:2]-g[i][:2]))*100, math.hypot(*(x[:2]-g[i][:2]))*100), "Vobs", len(Vobs), "assoc prior", len(vi), "atGT", len(vi2), "agree", agree, "ncand", [len(q[1]) for q in c], info.get("ok"), info.get("why"), "sdpost %.1f cm" % (math.sqrt(Pp[0,0])*100))
