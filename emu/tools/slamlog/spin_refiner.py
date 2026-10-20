"""Learned heading refiner for odometry (offline).  Strict split across drives: train on <train logs>, test on <test log>.
  PYTHONPATH=.:src python -m emu.tools.slamlog.spin_refiner <out.json> <test_log> <train_log> [<train_log> ...]
Per DT window: features = odom dtheta, gyro (raw, base z) dtheta, wheel dtheta, |w|, v, products with |w| (a linear
model with spin-rate interactions = 'small'); target = GT dtheta - odom dtheta.  Uncertainty: a second linear model of the
squared residual (heteroscedastic), clipped >= the train residual floor.  Ridge strength picked by leave-one-drive-out CV.
Matcher deltas are NOT inputs: the offline kslam replay does not reproduce live matching (7/123 accepted on L8), so they
would be train/test inconsistent; record KEVIN_SLAM_REC jobs to add them.
Compared on the test drive: odom, gyro-only during spins (|w| > 0.6 rad/s, the kslam no-match threshold), refined."""
import sys, json, math
import numpy as np
import twin_geometry as G
from .log import Log
from .metrics import evaluate

DT = 0.2; W_SPIN = 0.6

def series(lg):
    L = Log(lg); N = L.nav(); t_od, ui = np.unique(N["t_od"], return_index=True); od = N["odom"][ui]
    t = np.arange(max(t_od[0], L.gt_t[0]), min(t_od[-1], L.gt_t[-1]), DT)
    th = np.interp(t, t_od, np.unwrap(od[:, 2])); x = np.interp(t, t_od, od[:, 0]); y = np.interp(t, t_od, od[:, 1])
    ti = N["imu_t"]; gz = (N["gyro"] @ np.asarray(G.IMU_R).T)[:, 2]; gi = np.r_[0, np.cumsum(0.5 * (gz[1:] + gz[:-1]) * np.diff(ti))]
    gth = np.interp(t, ti, gi); ez = N["gz"]; ei = np.r_[0, np.cumsum(0.5 * (ez[1:] + ez[:-1]) * np.diff(N["t"]))]; eth = np.interp(t, N["t"], ei)
    tt = N["t"]; wh = (N["vr"] - N["vl"]) / G.TRACK_M; wi = np.r_[0, np.cumsum(wh[1:] * np.diff(tt))]; wth = np.interp(t, tt, wi)
    gt = L.gt(t); gyaw = np.unwrap(gt[:, 2])
    d = lambda a: np.diff(a)
    dth, dg, dw, dgt, de = d(th), d(gth), d(wth), d(gyaw), d(eth)
    c, s = np.cos(th[:-1]), np.sin(th[:-1]); dx, dy = d(x), d(y); fwd = c * dx + s * dy; lat = -s * dx + c * dy
    W = np.abs(dth) / DT; V = np.abs(fwd) / DT
    X = np.c_[dth, dg, dw, W * dth, W * dg, V * dth, np.sign(dth) * W, np.ones_like(dth)]
    return dict(L=L, t=t, X=X, y=dgt - dth, dth=dth, dg=dg, de=de, W=W, fwd=fwd, lat=lat, x0=(x[0], y[0], th[0]))

def ridge(X, y, lam):
    A = X.T @ X + lam * np.diag(np.r_[np.ones(X.shape[1] - 1), 0]); return np.linalg.solve(A, X.T @ y)

def traj(S, dth_new):
    th = S["x0"][2] + np.r_[0, np.cumsum(dth_new)]; x = [S["x0"][0]]; y = [S["x0"][1]]
    for k in range(len(dth_new)):
        hm = th[k] + 0.5 * dth_new[k]; x.append(x[-1] + math.cos(hm) * S["fwd"][k] - math.sin(hm) * S["lat"][k]); y.append(y[-1] + math.sin(hm) * S["fwd"][k] + math.cos(hm) * S["lat"][k])
    return np.c_[S["t"], x, y, th]

def score(S, dth_new):
    r = evaluate(traj(S, dth_new), S["L"], with_map=False); return {k: round(float(r[k]), 4) for k in ("ate_rmse_m", "ate_max_m", "head_max_deg", "head_final_deg", "ate_aligned_rmse_m")}

if __name__ == "__main__":
    out, test, train = sys.argv[1], sys.argv[2], sys.argv[3:]
    TR = [series(l) for l in train]; TE = series(test)
    lams = [1e-6, 1e-4, 1e-2, 1, 100]; cv = {}
    for lam in lams:                                    # leave-one-drive-out on the TRAIN drives only
        e = []
        for i in range(len(TR)):
            Xa = np.vstack([s["X"] for j, s in enumerate(TR) if j != i]); ya = np.concatenate([s["y"] for j, s in enumerate(TR) if j != i])
            b = ridge(Xa, ya, lam); e.append(np.mean((TR[i]["X"] @ b - TR[i]["y"]) ** 2))
        cv[lam] = float(np.mean(e))
    lam = min(cv, key=cv.get); Xa = np.vstack([s["X"] for s in TR]); ya = np.concatenate([s["y"] for s in TR]); b = ridge(Xa, ya, lam)
    r2 = (Xa @ b - ya) ** 2; Zv = np.c_[np.abs(Xa[:, 0]), np.abs(Xa[:, 1]), Xa[:, 6] ** 2 if False else np.abs(Xa[:, 6]), np.ones(len(ya))]; bv = ridge(Zv, r2, lam)
    floor = float(np.percentile(r2, 50))
    O = dict(test=test, train=train, dt=DT, lam=lam, cv_mse=cv, n_train_windows=int(len(ya)), n_test_windows=int(len(TE["y"])))
    for nm, S in [("test", TE)] + [("train_" + l.rstrip("/").split("/")[-1], s) for l, s in zip(train, TR)]:
        spin = S["W"] > W_SPIN; pred = S["X"] @ b
        Zt = np.c_[np.abs(S["X"][:, 0]), np.abs(S["X"][:, 1]), np.abs(S["X"][:, 6]), np.ones(len(pred))]; sd = np.sqrt(np.maximum(Zt @ bv, floor))
        res = dict(odom=score(S, S["dth"]), gyro_in_spins=score(S, np.where(spin, S["dg"], S["dth"])), ekfgz_in_spins=score(S, np.where(spin, S["de"], S["dth"])), refined=score(S, S["dth"] + pred))
        z = (S["y"] - pred) / sd; res["refined_unc"] = dict(frac_within_2sd=round(float(np.mean(np.abs(z) < 2)), 3), n=int(len(z)))
        res["spin_windows"] = int(spin.sum()); res["heading_err_from_spins_deg"] = round(float(np.degrees(np.sum(S["y"][spin]))), 2)
        res["heading_err_from_nonspin_deg"] = round(float(np.degrees(np.sum(S["y"][~spin]))), 2)
        O[nm] = res
    O["coef"] = b.tolist()
    json.dump(O, open(out, "w"), indent=1); print(json.dumps(O, indent=1))
