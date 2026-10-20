"""Offline replay of nav/lslam (lifelong kslam phase 1) on a sensor log, with fault injection.
  calib:  python -m emu.tools.slamlog.lslam_replay calib <train_log> <params.json>
          MEASURES the odometry error model (odom vs GT increments), the scan-match error R (CSM against a map built at
          GT poses, guess = GT + perturbation) and the RS2 floor-height sd (sensor gate).
  run:    python -m emu.tools.slamlog.lslam_replay run <log> <params.json> <out.json> [fault] [t0_frac]
          fault: none | corrupt (random depth 3 s) | rescale (depth x1.1, 5 s) | slip (translation x0.5, 2 s) | gyro (+0.5 deg/s from t0)
Scans: both RealSense depths -> base-frame points 0.05..1.6 m above the floor, < 3 m (as kslam), every 3rd RS2 frame (~10 Hz)."""
import sys, os, json, math, time
import numpy as np
from scipy.spatial import cKDTree
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import twin_geometry as G
from .log import Log
from .metrics import evaluate
from nav.lslam import model as M
from nav.lslam.core import LifelongSlam
from nav.slam import GridMap, Matcher

EVERY, S = 3, 8
FAULTS = dict(corrupt=3.0, rescale=5.0, slip=2.0, gyro=None)


class Scans:
    def __init__(self, L):
        self.L = L; K, _ = L.K("rs2", "depth"); H, W = 480, 848
        vv, uu = np.mgrid[0:H:S, 0:W:S]; self.rx = (uu - K[0, 2]) / K[0, 0]; self.ry = (vv - K[1, 2]) / K[1, 1]
        self.f2 = L.frames["rs2"][::EVERY]; f1 = L.frames["rs1"]; self.t1 = np.array([t for t, _ in f1]); self.f1 = f1

    def _cam(self, cam, f, fault, rng):
        z = self.L.depth(f)[::S, ::S]
        if fault == "corrupt": z = np.where(z > 0, rng.uniform(0.3, 4.0, z.shape), 0).astype(np.float32)
        elif fault == "rescale": z = z * 1.1
        m = (z > 0.15) & (z < 4.0); X = np.stack([self.rx[m] * z[m], self.ry[m] * z[m], z[m]], 1)
        R, T = self.L.ext[cam]; B = X @ R.T + T; h = B[:, 2] + G.AXLE_H_M; r = np.hypot(B[:, 0], B[:, 1])
        return B[(h > 0.05) & (h < 1.6) & (r < 3.0), :2], h[(np.abs(h) < 0.1) & (r < 2.5)]

    def get(self, i, fault=None, rng=None):
        t, f = self.f2[i]; p2, fh = self._cam("rs2", f, fault, rng)
        p1, _ = self._cam("rs1", self.f1[int(np.argmin(np.abs(self.t1 - t)))][1], fault, rng)
        return t, np.concatenate([p1, p2]).astype(np.float32), (float(np.median(fh)) if len(fh) > 50 else float("nan"))


def odom_series(L, fault=None, t0=None):
    N = L.nav(); t, od = N["t_od"], N["odom"].copy(); od[:, 2] = np.unwrap(od[:, 2])
    if fault in ("slip", "gyro"):
        out = [od[0].copy()]
        for i in range(1, len(od)):
            inc = M.between(od[i - 1], od[i])
            if fault == "slip" and t0 <= t[i] <= t0 + FAULTS["slip"]: inc[:2] *= 0.5
            if fault == "gyro" and t[i] >= t0: inc[2] += math.radians(0.5) * (t[i] - t[i - 1])
            out.append(M.compose(out[-1], inc))
        od = np.array(out); od[:, 2] = np.unwrap(od[:, 2])
    def at(tq):
        x = np.interp(tq, t, od[:, 0]); y = np.interp(tq, t, od[:, 1]); th = np.interp(tq, t, od[:, 2]); return np.array([x, y, M.wrap(th)])
    return at


def calib(log, out):
    L = Log(log); sc = Scans(L); od = odom_series(L); n = len(sc.f2); rng = np.random.default_rng(0)
    E, D, RT, FH = [], [], [], []; prev = None
    gmap = GridMap(40.0, 0.02, (-20.0, -20.0)); scans = []
    for i in range(n):
        t, pts, fh = sc.get(i); g = L.gt(t); g[2] = M.wrap(g[2]); o = od(t); FH.append(fh); scans.append((t, pts, g))
        if prev is not None:
            di = M.between(prev[1], o); dg = M.between(prev[0], g); e = M.innovation(di, dg)
            E.append(e); D.append(math.hypot(dg[0], dg[1])); RT.append(abs(dg[2]))
        prev = (g, o); gmap.integrate(tuple(g), pts, np.zeros((0, 2), np.float32))
    E, D, RT = np.array(E), np.array(D), np.array(RT); exy = np.hypot(E[:, 0], E[:, 1]) / math.sqrt(2)
    # sd_xy = a*dist + b, sd_th = c*rot + d*dist : least squares on |e| ~ half-normal (E|e| = sd*sqrt(2/pi))
    k = math.sqrt(math.pi / 2)
    a, b = np.linalg.lstsq(np.stack([D, np.ones_like(D)], 1), exy * k, rcond=None)[0]
    c, d = np.linalg.lstsq(np.stack([RT, D], 1), np.abs(E[:, 2]) * k, rcond=None)[0]
    # R under the ONLINE conditions: reference = the submap policy of nav/lslam/core.py, built at GT poses (an oracle
    # pose, so the error is the matcher's + the incomplete local map's, not drift); guess = GT + perturbation.
    from nav.lslam.submap import Submap
    from nav.lslam import core as C
    subs = [Submap(0, scans[0][2])]; ER = []; TAG = []; lastw = None
    for t, pts, g in scans:
        a_ = subs[-1]; ref = a_ if a_.n >= C.MIN_REF_SCANS else (subs[-2] if len(subs) > 1 else None)
        if ref is not None and len(pts) >= 50:
            guess = g + rng.normal(0, [0.03, 0.03, math.radians(1.0)])
            P = pts[rng.choice(len(pts), 1000, replace=False)] if len(pts) > 1000 else pts
            best, bs, med, _ = Matcher(ref.lf(), lin=0.12, lin_step=0.02, ang=math.radians(4), ang_step=math.radians(0.67), prior=0.0).match(P, tuple(M.between(ref.anchor, guess)))
            if bs > med: ER.append(M.innovation(M.compose(ref.anchor, np.array(best)), g)); TAG.append([ref is a_, ref.n, abs(M.between(g, M.compose(ref.anchor, np.zeros(3)))[2])])
        if lastw is not None:
            dw = M.between(lastw, g)
            if math.hypot(dw[0], dw[1]) < C.WRITE_DIST and abs(dw[2]) < C.WRITE_ROT: continue
        da = M.between(a_.anchor, g)
        if math.hypot(da[0], da[1]) > C.SUBMAP_DIST or abs(da[2]) > C.SUBMAP_ROT: a_ = Submap(len(subs), g); subs.append(a_)
        a_.integrate(M.between(a_.anchor, g), pts); lastw = g
    # robust covariance: iteratively drop errors outside the chi2(3) gate at ALPHA (those are what the gate rejects)
    ER = np.array(ER); keep = np.ones(len(ER), bool)
    for _ in range(10):
        Rk = np.cov(ER[keep].T); q = np.einsum("ij,jk,ik->i", ER, np.linalg.inv(Rk), ER); k2 = q < M.gate(3)
        if (k2 == keep).all(): break
        keep = k2
    out_frac = float(1 - keep.mean()); TAG = np.array(TAG, float)
    diag = dict(outlier_frac_active_ref=float(1 - keep[TAG[:, 0] == 1].mean()), outlier_frac_prev_ref=float(1 - keep[TAG[:, 0] == 0].mean()) if (TAG[:, 0] == 0).any() else None,
                frac_prev_ref=float((TAG[:, 0] == 0).mean()), outlier_err_xy_p50_m=float(np.median(np.hypot(ER[~keep, 0], ER[~keep, 1]))) if (~keep).any() else None,
                outlier_err_th_p50_deg=float(np.degrees(np.median(np.abs(ER[~keep, 2])))) if (~keep).any() else None,
                outlier_frac_ref_lt30=float(1 - keep[TAG[:, 1] < 30].mean()) if (TAG[:, 1] < 30).any() else None, outlier_frac_ref_ge30=float(1 - keep[TAG[:, 1] >= 30].mean()) if (TAG[:, 1] >= 30).any() else None)
    R_trim = np.cov(ER[keep].T)    # diagnostic only: trimming a heavy-tailed error collapses R (2026-10-19: 39 % 'outliers' with p50 2.8 cm)
    ER = np.array(ER); R = np.cov(ER.T); FH = np.array(FH); FH = FH[np.isfinite(FH)]
    p = dict(log=log, odom=dict(a=max(a, 1e-4), b=max(b, 1e-5), c=max(c, 1e-4), d=max(d, 1e-5)), R=R.tolist(),
             R_sd=dict(x_m=float(math.sqrt(R[0, 0])), y_m=float(math.sqrt(R[1, 1])), th_deg=float(math.degrees(math.sqrt(R[2, 2])))),
             R_bias=dict(x_m=float(ER[:, 0].mean()), y_m=float(ER[:, 1].mean()), th_deg=float(math.degrees(ER[:, 2].mean()))),
             floor=dict(mu=float(np.median(FH)), sd=float(1.4826 * np.median(np.abs(FH - np.median(FH))))), n_inc=len(E), n_match=len(ER), match_outlier_frac=out_frac, match_diag=diag, R_trim_sd=np.sqrt(np.diag(R_trim)).tolist(), n_submaps=len(subs))
    json.dump(p, open(out, "w"), indent=1); print(json.dumps(p, indent=1))


def run(log, params, out, fault="none", t0f=0.4):
    p = json.load(open(params)); L = Log(log); sc = Scans(L); n = int(os.environ.get("LSLAM_N", len(sc.f2)))
    tA, tB = sc.f2[0][0], sc.f2[n - 1][0]; t0 = tA + t0f * (tB - tA); dur = FAULTS.get(fault); t1 = t0 + dur if dur else tB
    od = odom_series(L, fault, t0); rng = np.random.default_rng(1)
    om = M.OdomModel(**p["odom"]); fl = p["floor"]; gf = M.gate(1)
    g0 = L.gt(tA); slam = LifelongSlam(om, p["R"], x0=(g0[0], g0[1], M.wrap(g0[2])))
    est, states = [], []; w0 = time.time()
    for i in range(n):
        t = sc.f2[i][0]; df = fault if fault in ("corrupt", "rescale") and t0 <= t <= t1 else None
        t, pts, fh = sc.get(i, df, rng)
        ok = math.isfinite(fh) and ((fh - fl["mu"]) / fl["sd"]) ** 2 < gf
        x, st = slam.step(t, od(t), pts, ok); est.append([t, *x]); states.append(st)
    wall = time.time() - w0; est = np.array(est); est[:, 3] = np.unwrap(est[:, 3])
    r = evaluate(est, L, with_map=False)
    ev = slam.events; first = [e for e in ev if e["t"] >= t0 and e["kind"] in ("reject", "state", "rollback")]
    pre = [e for e in ev if e["t"] < t0 and e["kind"] in ("reject", "rollback")]
    poisoned = sum(1 for e in slam.j.e if t0 <= e.t <= t1) if fault != "none" else 0
    r.update(log=log, fault=fault, t0=t0 - tA, fault_s=dur, detect_latency_s=(first[0]["t"] - t0) if first and fault != "none" else None,
             detect_kind=(first[0]["kind"] + ":" + first[0].get("why", first[0].get("to", ""))) if first and fault != "none" else None,
             writes_in_fault_window=poisoned, rollbacks=slam.stats["rollbacks"], stats=slam.stats, n_submaps=len(slam.subs),
             pre_fault_rejects=len(pre), state_frac={s: states.count(s) / len(states) for s in set(states)},
             nis_rej_pct=np.percentile(slam.nis_rej, [10, 50, 90]).tolist() if slam.nis_rej else None, nis_acc_mean=float(np.mean(slam.nis_acc)) if slam.nis_acc else None, ms_per_frame=round(1000 * wall / n, 1), rtf=round((tB - tA) / wall, 2),
             events=[e for e in ev if e["kind"] in ("state", "rollback", "submap")][:200])
    json.dump(r, open(out, "w"), indent=1, default=float)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("events",)})


if __name__ == "__main__":
    if sys.argv[1] == "calib": calib(*sys.argv[2:4])
    else: run(sys.argv[2], sys.argv[3], sys.argv[4], *(sys.argv[5:6] or ["none"]), *([float(sys.argv[6])] if len(sys.argv) > 6 else []))
