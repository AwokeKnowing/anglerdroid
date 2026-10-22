"""Replay sim recordings: odometry-primary SE(2) EKF with lineloc updates from the mast camera; errors vs GT.
python3 bench.py OUTDIR L18 L19 L20 ...   (config via env LL_*)"""
import sys, os, json, time, math, numpy as np, cv2
from lineloc import Camera, LineMap, LineLoc, wrap, axle_height
VL = os.environ.get("LL_VL", "/workspace/vl")    # sim recordings: mlx/<log>/poses.npz + <log>/mast/*.jpg
CT, CR = np.array([9.94e-04, 2.45e-04]), np.array([1.00e-09, 4.42e-04])     # maploc odometry noise model (fitted on L12-L14)
def Q(d): a = np.array([abs(d[0]) + abs(d[1]), abs(d[2])]); return np.diag([a @ CT + 1e-8, a @ CT + 1e-8, a @ CR + 1e-8])
def rel(a, b):
    c, s = math.cos(a[2]), math.sin(a[2]); dx, dy = b[0] - a[0], b[1] - a[1]; return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])
def compose(a, d):
    c, s = math.cos(a[2]), math.sin(a[2]); return np.array([a[0] + c * d[0] - s * d[1], a[1] + s * d[0] + c * d[1], a[2] + d[2]])
def anchor(tr, g0):
    out = [g0.copy()]
    for i in range(1, len(tr)): out.append(compose(out[-1], rel(tr[i - 1], tr[i])))
    return np.array(out)
def errs(est, gt):
    e = np.hypot(*(est[:, :2] - gt[:, :2]).T); y = np.degrees(np.abs(wrap(est[:, 2] - gt[:, 2])))
    return dict(pos_med=float(np.median(e)), pos_p95=float(np.percentile(e, 95)), pos_max=float(e.max()), yaw_med=float(np.median(y)), yaw_p95=float(np.percentile(y, 95)), yaw_max=float(y.max()))
def env(k, d, f=float): return f(os.environ.get(k, d))
mi = json.load(open(f"{VL}/mlx/mast.json"))
KF = env("LL_KF", 0, int) == 1; KF_D, KF_A, KF_T = env("LL_KFD", 0.02), math.radians(env("LL_KFA", 1.0)), env("LL_KFT", 1.0)
def log_tire(n):
    z = np.load(f"{VL}/mlx/{n}/poses.npz"); return str(z["tire"]) if "tire" in z.files and str(z["tire"]) else "stock"
def make(tire="stock"):
    # mast pos is relative to base_link = the axle, which sits at the log's tyre loaded radius (stock 0.08565, soft_v3_2
    # 0.10065; src/robot_params.py).  Was a fixed stock +0.08565: soft-tyre logs had the camera 1.5 cm too low.
    cam = Camera(mi["intr"]["fx"], mi["intr"]["fy"], mi["intr"]["ppx"], mi["intr"]["ppy"], mi["dist"], t_bc=(mi["pos"][0], mi["pos"][1], mi["pos"][2] + axle_height(tire)))
    return LineLoc(LineMap(os.environ.get("LL_MAP", "casita_lines.npz")), cam, sigma_px=env("LL_SIGMA", 1.5), n_seg=env("LL_NSEG", 200, int),
                   min_len=env("LL_MINLEN", 20), lsd_scale=env("LL_LSDS", 0.8), ransac=env("LL_RANSAC", 0, int),
                   use_h=env("LL_H", 1, int) == 1, use_v=env("LL_V", 1, int) == 1, min_v=env("LL_MINV", 2, int), min_all=env("LL_MINALL", 3, int),
                   n_lost=env("LL_NLOST", 10, int), reloc_draws=env("LL_RDRAWS", 64, int), reloc_xy=env("LL_RXY", 0.25), reloc_yaw_deg=env("LL_RYAW", 5.0),
                   uniq=env("LL_UNIQ", 0.0), h_junction=env("LL_HJ", 0, int) == 1, h_minlen=env("LL_HMIN", 0), sigma_map=env("LL_SMAP", 0.0))
def run(n, loc):
    z = np.load(f"{VL}/mlx/{n}/poses.npz"); g, od_raw = z["gt"], z["odom"]; od = anchor(od_raw, g[0])
    p = g[0].copy(); P = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(1) ** 2]); est = [p.copy()]; ms = []; acc = 0; why = {}; nv = []; nre = 0; last = p.copy(); t_last = z["t"][0]; nup = 0
    for i in range(1, len(g)):
        d = rel(od_raw[i - 1], od_raw[i]); p = compose(p, d); co, si = math.cos(p[2]), math.sin(p[2]); R = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]]); P = P + R @ Q(d) @ R.T
        # keyframe rule: a view that has not changed since the last update carries no new information about the
        # (static) map-model error; update after >= KF_D m / KF_A rad of motion or KF_T s.
        mv = rel(last, p)
        if KF and abs(mv[0]) + abs(mv[1]) < KF_D and abs(mv[2]) < KF_A and z["t"][i] - t_last < KF_T: est.append(p.copy()); continue
        im = cv2.imread(f"{VL}/{n}/{z['files'][i]}.jpg", cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter(); q, Pq, info = loc.update(im, p, P); ms.append((time.perf_counter() - t0) * 1e3); last = p.copy(); t_last = z["t"][i]; nup += 1
        if info.get("ok"): p, P = q, Pq; acc += 1; nv.append(info["nV"]); nre += info["lost"]
        else: why[info.get("why")] = why.get(info.get("why"), 0) + 1
        est.append(p.copy())
    est = np.array(est)
    return est, dict(frames=len(g), localized=acc / max(nup, 1), updates=nup, why=why, relocs=nre, nV_med=float(np.median(nv)) if nv else 0, odom=errs(od, g), lineloc=errs(est, g),
                     ms_med=float(np.median(ms)), ms_p95=float(np.percentile(ms, 95)), ms_max=float(np.max(ms)))
if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); res = {}
    for n in sys.argv[2:]:
        loc = make(log_tire(n)); loc.n_fail = 0; est, r = run(n, loc); r["tire"] = log_tire(n); r["cam_z"] = float(loc.cam.t_bc[2]); np.save(f"{out}/{n}_est.npy", est); res[n] = r
        L = r["lineloc"]; O = r["odom"]
        print(f"{n}: loc {r['localized']*100:.0f}%  med {L['pos_med']*100:.2f} p95 {L['pos_p95']*100:.2f} max {L['pos_max']*100:.2f} cm  yaw med {L['yaw_med']:.2f} p95 {L['yaw_p95']:.2f} | odom med {O['pos_med']*100:.2f} max {O['pos_max']*100:.2f} | {r['ms_med']:.1f}/{r['ms_p95']:.1f}/{r['ms_max']:.1f} ms  {r['why']}", flush=True)
    json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
