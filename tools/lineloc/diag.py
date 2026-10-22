import sys, math, numpy as np, cv2
from bench import make, VL
from lineloc import wrap
n = sys.argv[1] if len(sys.argv) > 1 else "L20"
loc = make(); z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]; rng = np.random.default_rng(0)
E = []; RV = []; RH = []
for i in range(0, len(g), 10):
    im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); P = np.diag([0.03**2, 0.03**2, math.radians(1)**2])
    pr = g[i] + rng.multivariate_normal(np.zeros(3), P); x, Pp, info = loc.update(im, pr, P)
    Vobs, Hobs = loc.observe(loc.segments(im)); vo, vi, _, ho, hi = loc.associate(g[i], np.diag([0.005**2, 0.005**2, math.radians(0.2)**2]), Vobs, Hobs)
    r = loc.residuals(g[i], Vobs[vo], vi, Hobs[ho], hi); RV += list(r[:len(vi)]); RH += list(r[len(vi):])
    E.append([math.hypot(*(x[:2]-g[i][:2])), math.degrees(wrap(x[2]-g[i][2])), info.get("ok", False)])
E = np.array(E, float); ok = E[:, 2] > 0
print("single-frame err cm (accepted): med %.2f p95 %.2f  yaw med %.3f | accepted %.0f%%" % (np.median(E[ok,0])*100, np.percentile(E[ok,0],95)*100, np.median(np.abs(E[ok,1])), ok.mean()*100))
RV, RH = np.array(RV), np.array(RH)
print("residuals at GT (px): V mean %.2f med %.2f sd %.2f n %d | H mean %.2f med %.2f sd %.2f n %d" % (RV.mean(), np.median(RV), RV.std(), len(RV), RH.mean(), np.median(RH), RH.std(), len(RH)))
