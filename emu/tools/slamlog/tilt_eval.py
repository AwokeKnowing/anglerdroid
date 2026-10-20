"""Evaluate nav.tilt (per-camera floor-tilt KF) on a log with GT body attitude (campose.npy, /tmp/cc/camposerec.py).
  OMP_NUM_THREADS=1 PYTHONPATH=.:src python -m emu.tools.slamlog.tilt_eval <log> <out.json> [stride=3]
Realism is emulated OFFLINE on the recorded clouds (exact for a rigid camera rotation, apart from FOV-edge coverage):
  miscal: each camera's real mount is rotated by a random (pitch, roll) offset, U(-MIS, MIS) per axis, seeded; nav still
          uses the nominal twin extrinsic.  MIS is an ASSUMED magnitude (James: the mast is not straight).
  sway:   RS1 (top-down, mast tip z 0.86 m; James 21:13: the twin is right) deflection s'' = -w0^2 s - 2 z w0 s' + kappa*accel
          (pitch from fore/aft, roll from lateral accel); w0, z, kappa ASSUMED (no measurement of the real mast yet).  The
          IMU is the RS2 (D435i) IMU on the low body mount: the gyro does NOT see the mast-tip sway.  RS2: no sway.
RS1 is filtered two ways: 'kf' = gyro-propagated (body rate) and 'kf_fit' = floor-fit-driven dynamics (no gyro; OU sigma/tau
measured from RS1's own fit series on the train half of that config, i.e. what the real bot can measure without GT).
Train/test split in time: filter parameters (sigma_n, OU sigma_d / tau, gyro noise) are measured on the first half
(unperturbed); errors are reported on the second half."""
import sys, os, math, time, json
import numpy as np, cv2
import twin_geometry as G
from .log import Log, depth_u16
from nav.tilt import fixed_pixels, fit, TiltKF, N_PIX
from nav.perceive import Rs2Cloud

MIS = math.radians(1.5); SEEDS = range(5)
W0, ZETA, KAPPA = 2 * math.pi * 1.5, 0.05, math.radians(0.5)       # sway: 1.5 Hz, 5 % damping, 0.5 deg per m/s^2 (assumed)

def Rpr(p, r):
    cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    return np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

def perturb(P, T, p, r):
    if p == 0 and r == 0: return P
    return (P - T) @ Rpr(p, r) + T            # = Rpr^T (P - T) + T per row

def quat_tilt(q):
    w, x, y, z = q.T; return np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)), np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

def main(lg, out, stride=3):
    L = Log(lg); N = L.nav(); C = np.load(os.path.join(lg, "campose.npy")); tc = C[:, 0] * 1e-9
    gp, gr = quat_tilt(C[:, 4:8]); AX = G.AXLE_H_M
    ti = N["imu_t"]; wb = N["gyro"] @ np.asarray(G.IMU_R).T
    t_od, ui = np.unique(N["t_od"], return_index=True); od = N["odom"][ui]
    tg = np.arange(t_od[0], t_od[-1], 0.02); x_ = np.interp(tg, t_od, od[:, 0]); y_ = np.interp(tg, t_od, od[:, 1]); th_ = np.unwrap(np.interp(tg, t_od, np.unwrap(od[:, 2])))
    vx = np.gradient(x_, tg) * np.cos(th_) + np.gradient(y_, tg) * np.sin(th_); ax = np.gradient(vx, tg); ay = vx * np.gradient(th_, tg)
    # sway (assumed model) integrated on the 50 Hz grid, per axis: s'' = -w0^2 s - 2 z w0 s' + kappa a
    sw = np.zeros((len(tg), 2)); sv = np.zeros(2); s = np.zeros(2)
    for k in range(1, len(tg)):
        dt = tg[k] - tg[k - 1]; acc = np.array([ax[k], ay[k]])
        sv += (-W0 * W0 * s - 2 * ZETA * W0 * sv + KAPPA * W0 * W0 * acc) * dt; s = s + sv * dt; sw[k] = s
    swr = np.gradient(sw, tg, axis=0)
    st = L.streams["depth"]; rc = Rs2Cloud(dict(width=st["w"], height=st["h"], fx=st["fx"], fy=st["fy"], ppx=st["ppx"], ppy=st["ppy"]))
    cams = {}
    for cam, rr in (("rs1", (0.15, 9.0)), ("rs2", (0.4, 2.5))):
        K, _ = L.K(cam, "depth"); R, T = L.ext[cam]; R = np.asarray(R); T = np.asarray(T)
        idx = fixed_pixels(K, R, T, AX, 480, 848, *rr); v, u = np.divmod(idx, 848)
        ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))], 1) @ R.T
        cams[cam] = dict(idx=idx, ray=ray, T=T)
    f1 = L.frames["rs1"]; t1 = np.array([t for t, _ in f1])
    frames = [(t, f) for t, f in L.frames["rs2"][::stride] if tc[0] < t < tc[-1]]
    tmid = frames[len(frames) // 2][0]
    # ---- configs: off, miscal (5 seeds), miscal+sway (5 seeds)
    cfgs = [("off", 0, (0, 0), (0, 0), False)]
    for sd in SEEDS:
        g = np.random.default_rng(100 + sd).uniform(-MIS, MIS, 4)
        cfgs += [("miscal", sd, tuple(g[:2]), tuple(g[2:]), False), ("miscal+sway", sd, tuple(g[:2]), tuple(g[2:]), True)]
    # ---- pass 1: per frame, load + fits for every config (store results; KFs need params measured on the train half)
    recs = []; tcpu = {"rs1": [], "rs2": []}
    for t, f in frames:
        i1 = int(np.argmin(np.abs(t1 - t))); d1 = depth_u16(L.d + "/" + f1[i1][1])
        d2 = depth_u16(L.d + "/" + f)
        g = (np.interp(t, tc, gp), np.interp(t, tc, gr)); swt = np.array([np.interp(t, tg, sw[:, 0]), np.interp(t, tg, sw[:, 1])])
        # RS2 obstacle cloud (as nav sees it) for phantom counting
        zz = d2[::rc.s, ::rc.s].ravel().astype(np.float32) * 1e-3; ok = (zz > 0.2) & (zz < rc.max_range); zz = zz[ok]
        P2 = np.stack([rc.rx[ok] * zz, rc.ry[ok] * zz, zz], 1) @ rc.R.T + rc.T; hmin = 0.03 + 0.012 * zz * zz
        r = dict(t=t, gt=g, sway=swt.tolist(), cfg=[])
        for name, sd, m1, m2, sway in cfgs:
            true = {"rs1": np.array(m1) + (swt if sway else 0), "rs2": np.array(m2)}
            ent = dict(true={c: (np.array(g) + true[c]).tolist() for c in true})
            for cam, dd in (("rs1", d1), ("rs2", d2)):
                cc = cams[cam]; z = dd.ravel()[cc["idx"]].astype(np.float32) * 1e-3; k = z > 0.1
                t0 = time.perf_counter()
                P = cc["ray"][k] * z[k, None] + cc["T"]; P = perturb(P, cc["T"], *true[cam]); fr = fit(P, AX, T=cc["T"])
                if name == "off": tcpu[cam].append(time.perf_counter() - t0)
                ent[cam] = None if fr is None else (fr[0], fr[1], fr[3], fr[4].tolist(), fr[5], fr[6], fr[7])
            # phantom candidates: RS2 points whose TRUE (attitude-levelled) height is floor but perceived height is an obstacle band
            Pp = perturb(P2, cams["rs2"]["T"], *true["rs2"]); hp = Pp[:, 2] + AX
            tp, trr = ent["true"]["rs2"]; htrue = hp - (math.tan(tp) * Pp[:, 0] - math.tan(trr) * Pp[:, 1])
            sel = (np.abs(htrue) < 0.015) & (hp < 1.0)
            ent["php"] = np.c_[Pp[sel, :2], hp[sel], hmin[sel]].astype(np.float32)
            r["cfg"].append(ent)
        recs.append(r)
    return recs, cfgs, tcpu, tmid, (ti, wb, tg, swr, sw)


def fit_dyn(recs, ci, cam, tmid):
    """OU sigma / tau of the camera's dynamic tilt from its own accepted-looking fits (train half): no GT needed."""
    Z = []; tt = []; Rm = []
    for r in recs:
        f = r["cfg"][ci][cam]
        if r["t"] >= tmid or f is None: continue
        Z.append([math.atan(f[0]), -math.atan(f[1])]); tt.append(r["t"]); Rm.append(np.diag(np.array(f[3])))
    Z = np.array(Z); Z = Z - np.median(Z, 0); dt = float(np.median(np.diff(tt))); sd = []; tau = []
    for k in range(2):
        x = Z[:, k]; x = x[np.abs(x) < 5 * 1.4826 * np.median(np.abs(x)) + 1e-9]     # outlier fits (clutter) out
        v = max(np.var(x) - np.mean(np.array(Rm)[:, k]), 1e-10); sd.append(math.sqrt(v))
        c = [np.mean(x[:-j] * x[j:]) / np.mean(x * x) for j in range(1, 60)]
        tau.append(next((j for j, q in enumerate(c, 1) if q < math.exp(-1)), 60) * dt)
    return sd, tau


def run_kf(recs, cfgs, tmid, gyro, ci, cam, prm, mode="gyro"):
    ti, wb, tg, swr, sw = gyro; name, sd, m1, m2, sway = cfgs[ci]
    if mode == "fit": s_d, tau = fit_dyn(recs, ci, cam, tmid); kf = TiltKF(prm[cam]["sn2"], s_d, tau, [0.0, 0.0])
    else: kf = TiltKF(prm[cam]["sn2"], prm["sig_d"], prm["tau"], prm["sg2"])
    outk = []; tl = None
    gy = gyro_only = np.zeros(2)
    for r in recs:
        t = r["t"]
        if tl is not None:
            m = (ti > tl) & (ti <= t); w = wb[m].mean(0) if m.any() else np.zeros(3)
            wp, wr = (w[1], w[0]) if mode == "gyro" else (0.0, 0.0); kf.predict(t - tl, wp, wr); gyro_only = gyro_only + np.array([wp, wr]) * (t - tl)
        tl = t; e = r["cfg"][ci]; f = e[cam]
        res = kf.update(None if f is None else (f[0], f[1], 0.0, f[2], np.array(f[3]), f[4], f[5], f[6]))
        outk.append(dict(t=t, est=kf.tilt().tolist(), res=res, true=e["true"][cam], gyro=gyro_only.tolist(), nb=None if f is None else f[5]))
    return outk


def stats(err):
    e = np.degrees(np.abs(np.asarray(err))); return dict(p50=round(float(np.median(e)), 3), p95=round(float(np.percentile(e, 95)), 3), max=round(float(e.max()), 3), n=int(len(e)))


if __name__ == "__main__":
    lg, out = sys.argv[1], sys.argv[2]; stride = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    recs, cfgs, tcpu, tmid, gyro = main(lg, out, stride); ti, wb, tg, swr, sw = gyro
    tr = [r for r in recs if r["t"] < tmid]; te_i = [i for i, r in enumerate(recs) if r["t"] >= tmid]
    # ---- parameters measured on the train half, off config
    gt = np.array([r["gt"] for r in tr]); tt = np.array([r["t"] for r in tr])
    sig_d = gt.std(0); ac = []
    for k in range(2):
        x = gt[:, k] - gt[:, k].mean(); dt = np.median(np.diff(tt)); c = [np.mean(x[:-j] * x[j:]) / np.mean(x * x) for j in range(1, 60)]
        j = next((j for j, v in enumerate(c, 1) if v < math.exp(-1)), 60); ac.append(j * dt)
    sg2 = np.array([np.var(wb[ti < ti[0] + 1.0, 1]), np.var(wb[ti < ti[0] + 1.0, 0])])   # first second: robot still (drive starts after 2 s)
    prm = dict(sig_d=sig_d.tolist(), tau=ac, sg2=sg2.tolist())
    for cam in ("rs1", "rs2"):
        s2 = [r["cfg"][0][cam][2] / r["cfg"][0][cam][6] for r in tr if r["cfg"][0][cam] is not None]; prm[cam] = dict(sn2=float(np.median(s2)))
    O = dict(log=lg, stride=stride, n_frames=len(recs), n_test=len(te_i), params_measured_train_half=dict(
        sig_d_deg=np.degrees(sig_d).tolist(), tau_s=ac, gyro_sd_dps=np.degrees(np.sqrt(sg2)).tolist(),
        sigma_n_mm_at_1m={c: 1e3 * math.sqrt(prm[c]["sn2"]) for c in ("rs1", "rs2")}),
        assumed=dict(miscal_deg=math.degrees(MIS), sway_f0_hz=W0 / 2 / math.pi, sway_zeta=ZETA, sway_deg_per_mps2=math.degrees(KAPPA)),
        cpu_ms_single_thread={c: dict(p50=round(1e3 * float(np.median(v)), 3), p95=round(1e3 * float(np.percentile(v, 95)), 3), points=N_PIX) for c, v in tcpu.items()})
    nb2 = np.array([recs[i]["cfg"][0]["rs2"][5] if recs[i]["cfg"][0]["rs2"] else 0 for i in te_i]); low = nb2 <= np.percentile(nb2, 10)
    O["little_floor_def"] = "test frames in the lowest decile of RS2 floor-band points among the fixed %d (<= %d)" % (N_PIX, int(np.percentile(nb2, 10)))
    res = {}
    for ci, (name, sd, m1, m2, sway) in enumerate(cfgs):
        ks = {cam: run_kf(recs, cfgs, tmid, gyro, ci, cam, prm) for cam in ("rs1", "rs2")}
        kfit = [run_kf(recs, cfgs, tmid, gyro, ci, "rs1", prm, "fit")[i] for i in te_i]
        R = {}
        for cam in ("rs1", "rs2"):
            k = [ks[cam][i] for i in te_i]; tru = np.array([x["true"] for x in k])
            R[cam] = dict(kf=stats(np.array([x["est"] for x in k])[:, 0] - tru[:, 0]), kf_roll=stats(np.array([x["est"] for x in k])[:, 1] - tru[:, 1]),
                          fixed_mount=stats(tru[:, 0]), accept=round(float(np.mean([x["res"] == "ok" for x in k])), 3),
                          reject={q: int(sum(x["res"] == q for x in k)) for q in ("nofit", "planarity", "innovation")})
            if cam == "rs2":
                R[cam]["little_floor"] = dict(accept=round(float(np.mean([k[i]["res"] == "ok" for i in np.flatnonzero(low)])), 3),
                                               kf=stats((np.array([x["est"] for x in k])[:, 0] - tru[:, 0])[low]), n=int(low.sum()))
        tru1 = np.array([x["true"] for x in kfit]); e1 = np.array([x["est"] for x in kfit])
        R["rs1"]["kf_fit"] = stats(e1[:, 0] - tru1[:, 0]); R["rs1"]["kf_fit_roll"] = stats(e1[:, 1] - tru1[:, 1])
        R["rs1"]["kf_fit_accept"] = round(float(np.mean([x["res"] == "ok" for x in kfit])), 3)
        R["rs1"]["sway_true_pitch"] = stats(np.array([recs[i]["sway"][0] for i in te_i])) if sway else None
        # RS1 estimate transferred to RS2 (one estimate for both cameras) and gyro-only from the starting point
        k1 = [ks["rs1"][i] for i in te_i]; k2 = [ks["rs2"][i] for i in te_i]; tru2 = np.array([x["true"] for x in k2])
        R["rs2"]["rs1_transfer"] = stats(np.array([x["est"] for x in k1])[:, 0] - tru2[:, 0])
        # phantom pixels (RS2 obstacle band, true floor): none / rs1-transfer / rs2 KF / oracle
        ph = dict(fixed_mount=0, rs1_transfer=0, rs2_kf=0, oracle=0)
        for j, i in enumerate(te_i):
            q = recs[i]["cfg"][ci]["php"]
            if not len(q): continue
            for nm, (p, r_) in (("fixed_mount", (0, 0)), ("rs1_transfer", k1[j]["est"]), ("rs2_kf", k2[j]["est"]), ("oracle", k2[j]["true"])):
                hl = q[:, 2] - (math.tan(p) * q[:, 0] - math.tan(r_) * q[:, 1]); ph[nm] += int(((hl > q[:, 3]) & (hl < 1.0)).sum())
        R["rs2_phantom_px_test"] = ph
        res.setdefault(name, []).append(dict(seed=sd, mount_deg=dict(rs1=np.degrees(m1).round(2).tolist(), rs2=np.degrees(m2).round(2).tolist()), **R))
    O["results"] = res
    json.dump(O, open(out, "w"), indent=1, default=float); print(json.dumps(O, indent=1, default=float)[:6000])
