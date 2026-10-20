"""Offline replay: odometry-primary SE(2) tracking with map-edge fixes from the mast camera.  python run.py L14 [L12 ...]"""
from harness import *
CALF = os.environ.get("ML_CALF", "calib.json"); EVERY = int(os.environ.get("ML_EVERY", 1)); OUT = os.environ.get("ML_OUT", "res")
os.makedirs(OUT, exist_ok=True)
# odometry noise model (variance linear in motion, Thrun-style alphas), fitted once on all three logs vs GT (calibration)
ODOM_ERR = [float(v) for v in os.environ.get("ML_ODOM_ERR", "1,1").split(",")]   # (translation scale, yaw scale): synthetic miscalibration
def perturb(od):
    if ODOM_ERR == [1.0, 1.0]: return od
    out = [od[0].copy()]
    for i in range(1, len(od)):
        d = rel(od[i - 1], od[i]); d[:2] *= ODOM_ERR[0]; d[2] *= ODOM_ERR[1]; out.append(compose(out[-1], d))
    return np.array(out)
def fit_noise(names, win=10):
    """Variance proportional to motion (no constant term: a stationary robot accrues no odometry error), fitted by
    non-negative least squares on 1 s increments (longer than the GT timing jitter)."""
    from scipy.optimize import nnls
    A, et, er = [], [], []
    for n in names:
        z = dict(load(n)); z["odom"] = perturb(z["odom"])
        for i in range(win, len(z["t"]), win):
            do, dg = rel(z["odom"][i - win], z["odom"][i]), rel(z["gt"][i - win], z["gt"][i])
            mo = sum(abs(rel(z["odom"][j - 1], z["odom"][j])) for j in range(i - win + 1, i + 1))   # path, not net
            A.append([mo[0] + mo[1], mo[2]]); et.append(((do[:2] - dg[:2]) ** 2).sum() / 2); er.append(wrap(do[2] - dg[2]) ** 2)
    A = np.array(A); return nnls(A, np.array(et))[0] + 1e-9, nnls(A, np.array(er))[0] + 1e-9
CT, CR = fit_noise(["L12", "L13", "L14"])
def Q(d): a = np.array([abs(d[0]) + abs(d[1]), abs(d[2])]); return np.diag([a @ CT + 1e-8, a @ CT + 1e-8, a @ CR + 1e-8])
def errs(est, gt):
    e = np.hypot(*(est[:, :2] - gt[:, :2]).T); y = np.degrees(np.abs(wrap(est[:, 2] - gt[:, 2])))
    return dict(pos_med=float(np.median(e)), pos_p90=float(np.percentile(e, 90)), pos_max=float(e.max()), pos_final=float(e[-1]), yaw_med=float(np.median(y)), yaw_p90=float(np.percentile(y, 90)), yaw_final=float(y[-1]))
if __name__ == "__main__":
    print("odom noise model: var_xy = %.2e*|dpath| + %.2e*|dth| ; var_th = %.2e*|dpath| + %.2e*|dth|" % (*CT, *CR))
    allr = {}; names = sys.argv[1:]; CAL = json.load(open(CALF)) if os.path.exists(CALF) else {}
    for n in names:
        # leave-one-out: the fix-information scale comes from the OTHER logs
        cs = [CAL[m] for m in CAL if m != n]; cinf = float(np.exp(np.mean(np.log(cs)))) if cs and os.environ.get("ML_CAL", "1") == "1" else 1.0
        L = make_loc(info_scale=cinf)
        z = dict(load(n)); z["odom"] = perturb(z["odom"]); g = z["gt"]; yr = np.gradient(np.unwrap(z["odom"][:, 2]), z["t"]); od = anchor(z["odom"], g[0]); s2 = anchor(z["slam2"], g[0])
        SC = os.environ.get("ML_SCALE")       # estimate global map scale: initial value (e.g. 1.0); synthetic true scale via ML_MAP_TRUE
        p = g[0].copy(); P = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(1) ** 2])
        if SC: p = np.r_[p, math.log(float(SC))]; P = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(1) ** 2, 0.05 ** 2])   # 5 % prior on scale
        if SC and os.environ.get("ML_PITCH_EST") == "1":   # static mast pitch (mount/sag), 1 deg prior, no process noise
            p = np.r_[p, 0.0]; P = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(1) ** 2, 0.05 ** 2, math.radians(1) ** 2])
        est = [p.copy()]; acc = rej = 0; why = {}; ms = []; scl = []
        for i in range(1, len(z["t"])):
            d = rel(z["odom"][i - 1], z["odom"][i]); p[:3] = compose(p[:3], d); co, si = math.cos(p[2]), math.sin(p[2]); R = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
            P[:3, :3] = P[:3, :3] + R @ Q(d) @ R.T          # map scale: constant (no process noise) -> estimated jointly over the whole run
            if i % EVERY == 0:
                t0 = time.perf_counter(); q, Pq, info = L.update(gray(n, z["files"][i]), p, P, yaw_rate=yr[i]); ms.append((time.perf_counter() - t0) * 1e3)
                if info.get("ok"): p, P = q, Pq; acc += 1
                else: rej += 1; why[info.get("why")] = why.get(info.get("why"), 0) + 1
            est.append(p.copy()); scl.append(math.exp(p[3]) if len(p) >= 4 else 1.0)
        est = np.array(est); np.save(f"{OUT}/{n}_maploc.npy", est)
        # map-frame pose (what navigation in the map uses): metric / scale.  The map (= GT frame) is true; under a
        # wheel-scale error the scale state absorbs the odometry's metric error, so metric != map coordinates.
        est_map = est[:, :3].copy()
        if est.shape[1] >= 4: est_map[:, :2] /= np.exp(est[:, 3:4])
        r = dict(frames=len(est), dur_s=float(z["t"][-1] - z["t"][0]), path_m=float(np.hypot(*np.diff(g[:, :2], axis=0).T).sum()),
                 odom=errs(od, g), slam2=errs(s2, g), maploc=errs(est_map, g), maploc_metric=errs(est[:, :3], g), map_scale_final=scl[-1], map_scale_sd=float(math.sqrt(P[3, 3])) if len(p) >= 4 else 0.0, pitch_deg=float(math.degrees(p[4])) if len(p) >= 5 else None, pitch_sd_deg=float(math.degrees(math.sqrt(P[4, 4]))) if len(p) >= 5 else None, accepted=acc, rejected=rej, reject_rate=rej / max(1, acc + rej), why=why, info_scale=cinf,
                 ms_med=float(np.median(ms)), ms_p90=float(np.percentile(ms, 90)))
        allr[n] = r; print(n, json.dumps(r))
    json.dump(allr, open(f"{OUT}/summary.json", "w"), indent=1)
