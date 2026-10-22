"""Read-only real check: lineloc on stationary mast frames (13:48 PDT) vs the casita line-fit pose (12:13 PDT, same spot)."""
import sys, math, numpy as np, cv2, json, warnings; warnings.filterwarnings("ignore")
from lineloc import Camera, LineMap, LineLoc, wrap
FIT = np.array([3.815, 1.406, math.radians(95.8)])           # base pose implied by casita fit A
def cam(h):
    c = Camera(196.78, 196.78, 319.5, 239.5, (-0.0262, 0, 0, 0), t_bc=(0.0291, 0, h))
    c.R_bc = Camera.tilt(math.radians(2.16), math.radians(-3.22)) @ c.R_bc; return c
M = LineMap("casita_lines.npz"); rng = np.random.default_rng(0); out = {}
for h in (0.99115,):
  for sp, cm_, deg_, dr in [("wide", 0.10, 3.0, 0), ("wide+msac", 0.10, 3.0, 256), ("tracked", 0.03, 1.0, 0)]:
    for name, kw in [("V", dict(use_h=False)), ("VH", dict(use_h=True, h_junction=True, h_minlen=40, sigma_map=0.02))]:
        name = f"{sp}_{name}"
        loc = LineLoc(M, cam(h), lsd_scale=0.8, ransac=dr, **kw); R = []
        for k in (0, 2, 4):
            im = cv2.imread(f"real/real_{k}.png", 0)
            for t in range(10):
                P = np.diag([cm_ ** 2, cm_ ** 2, math.radians(deg_) ** 2]); pr = FIT + rng.multivariate_normal(np.zeros(3), P)
                x, Pp, info = loc.update(im, pr, P)
                for _ in range(2):           # stationary: re-run with the posterior (as the EKF would on the next frames)
                    if info.get("ok"): loc.ransac = 0; x, Pp, info = loc.update(im, x, Pp); loc.ransac = dr
                if info.get("ok"): R.append(np.r_[x, math.sqrt(Pp[0, 0]), math.sqrt(Pp[1, 1]), info["nV"], info.get("nH", 0)])
        R = np.array(R)
        if len(R):
            med = np.median(R[:, :3], 0); d = med - FIT
            out[f"{name}_h{h}"] = dict(accepted=f"{len(R)}/30", pose=[round(med[0], 3), round(med[1], 3), round(math.degrees(med[2]), 2)],
                                      diff_cm=[round(d[0] * 100, 1), round(d[1] * 100, 1)], dyaw_deg=round(math.degrees(wrap(d[2])), 2),
                                      spread_cm=round(float(np.std(np.hypot(R[:, 0] - med[0], R[:, 1] - med[1]))) * 100, 2), sd_post_cm=[round(float(np.median(R[:, 3])) * 100, 2), round(float(np.median(R[:, 4])) * 100, 2)],
                                      nV=float(np.median(R[:, 5])), nH=float(np.median(R[:, 6])))
        else: out[f"{name}_h{h}"] = "no fix"
        print(f"{name}_h{h}", out[f"{name}_h{h}"])
json.dump(out, open("real/real_check.json", "w"), indent=1)
