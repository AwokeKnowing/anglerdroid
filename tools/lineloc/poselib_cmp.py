"""Per-frame solver comparison on sim frames: lineloc (3-DoF, gravity known) vs PoseLib PnPL (6-DoF, BSD-3, the line
solver LIMAP uses), same detections, same prior window associations.  python3 poselib_cmp.py L20 ..."""
import sys, math, time, numpy as np, cv2, poselib, warnings; warnings.filterwarnings("ignore")
from bench import make, VL
from lineloc import wrap, se2_cam
loc = make(); rng = np.random.default_rng(0); f = loc.cam.fx
cam = {"model": "PINHOLE", "width": 640, "height": 480, "params": [1.0, 1.0, 0.0, 0.0]}
res = {"lineloc": [], "poselib": [], "prior": []}; T = {"lineloc": [], "poselib": []}
for n in sys.argv[1:]:
    z = np.load(f"{VL}/mlx/{n}/poses.npz"); g = z["gt"]
    for i in range(1, len(g), 3):
        im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", 0); P = np.diag([0.03 ** 2, 0.03 ** 2, math.radians(1) ** 2])
        pr = g[i] + rng.multivariate_normal(np.zeros(3), P)
        t0 = time.perf_counter(); x, _, info = loc.update(im, pr, P); T["lineloc"].append((time.perf_counter() - t0) * 1e3)
        res["prior"].append([math.hypot(*(pr[:2] - g[i][:2])), abs(math.degrees(wrap(pr[2] - g[i][2]))), 1])
        res["lineloc"].append([math.hypot(*(x[:2] - g[i][:2])), abs(math.degrees(wrap(x[2] - g[i][2]))), info.get("ok", False)])
        # PoseLib on the same associations (NN in the prior window)
        t0 = time.perf_counter(); segs = loc.segments(im); Vobs, Hobs = loc.observe(segs); vseg = loc.vseg
        vo, vi, _, ho, hi = loc.associate(pr, P, Vobs, Hobs)
        l2a = np.r_[vseg[vo, 0], Hobs[ho, 0]]; l2b = np.r_[vseg[vo, 1], Hobs[ho, 1]]
        V = loc.m.V[vi]; l3a = np.r_[np.c_[V[:, :2], V[:, 2]], loc.m.HA[hi]]; l3b = np.r_[np.c_[V[:, :2], V[:, 3]], loc.m.HB[hi]]
        ok = False
        if len(l2a) >= 3:
            c, R = se2_cam(pr, loc.cam); Rcw = (R @ loc.cam.R_bc).T
            init = poselib.CameraPose(); init.R = Rcw; init.t = -Rcw @ c
            try:
                pose, st = poselib.estimate_absolute_pose_pnpl(np.zeros((0, 2)), np.zeros((0, 3)), l2a, l2b, l3a, l3b, cam,
                                                               {"max_reproj_error": 3.0 / f, "max_line_error": 3.0 / f}, {}, init)
                C = -pose.R.T @ pose.t; zc = pose.R.T @ np.array([0, 0, 1.0]); cp = np.array([C[0], C[1], math.atan2(zc[1], zc[0])])
                xb = loc.cam_to_base(cp); ok = True
            except Exception as e: print("poselib", e)
        T["poselib"].append((time.perf_counter() - t0) * 1e3)
        res["poselib"].append([math.hypot(*(xb[:2] - g[i][:2])), abs(math.degrees(wrap(xb[2] - g[i][2]))), ok] if ok else [np.nan, np.nan, False])
for k, v in res.items():
    v = np.array(v, float); m = v[:, 2] > 0
    print(f"{k:8s} accepted {m.mean()*100:5.1f}%  pos med {np.nanmedian(v[m,0])*100:.2f} p95 {np.nanpercentile(v[m,0],95)*100:.2f} cm  yaw med {np.nanmedian(v[m,1]):.3f} p95 {np.nanpercentile(v[m,1],95):.3f} deg" + (f"  {np.median(T[k]):.1f} ms" if k in T else ""))
