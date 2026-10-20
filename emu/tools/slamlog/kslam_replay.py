"""Offline replay of kslam (nav.slam.Slam._process) on a sensor log, deterministic, at ~5 Hz (every 6th RS2 frame).
  calib: python -m emu.tools.slamlog.kslam_replay calib <train_log> <lslam_params.json> <out_headkf.json>
         MEASURES the matcher's heading error r_th (matcher correction minus the correction GT says was needed, accepted
         matches) and takes the gyro/odom heading model (q_th_rot, q_th_dist) from the lslam calib of the same log.
  run:   python -m emu.tools.slamlog.kslam_replay run <log> <out.json> [headkf.json]
Scans: emu.tools.slamlog.lslam_replay.Scans (both depths, floor-levelled); odom: nav's recorded odom (incl. Manhattan)."""
import sys, os, json, math, time
import numpy as np
from .log import Log
from .metrics import evaluate
from . import lslam_replay as LR
from nav.slam import Slam, GridMap, compose, wrap



def replay(log, headkf=None, record=False):
    """inputs as nav/run.py feeds kslam: RS2 Rs2Cloud obstacles (3 cm + 1.2 cm z^2 .. 95 cm, static mount) = map + match
    points, RS2 floor[::3] = free points, 5 Hz, matching skipped while |w| > 0.6 rad/s (run.py).  Not replayed: RS1 ego
    points/clear cells and ray carving (approximation, so 'before' = this replay, not the live numbers)."""
    if headkf: os.environ["KEVIN_SLAM_HEAD_KF"] = headkf
    else: os.environ.pop("KEVIN_SLAM_HEAD_KF", None)
    _rng = np.random.default_rng; np.random.default_rng = lambda *a, **k: _rng(0)    # kslam subsamples with an unseeded rng: repeatable
    import cv2
    from nav.perceive import Rs2Cloud
    L = Log(log); N = L.nav(); od = LR.odom_series(L); st = L.streams["depth"]
    rc = Rs2Cloud(dict(width=st["w"], height=st["h"], fx=st["fx"], fy=st["fy"], ppx=st["ppx"], ppy=st["ppy"]))
    th = np.unwrap(N["odom"][:, 2]); w_od = np.gradient(th, N["t_od"])
    s = Slam(GridMap(), match=True); s._run = False
    rec = []; orig = s.matcher.match
    def match(P, guess):
        r = orig(P, guess); rec.append((guess, r[0])); return r
    s.matcher.match = match
    fr = L.frames["rs2"]; est = []; w0 = time.time()
    for i, (t, f) in enumerate(fr):
        o = tuple(od(t))
        if i % 6 == 0:
            obs, flo = rc.points(cv2.imread(L.d + "/" + f + ".png", cv2.IMREAD_UNCHANGED))
            if len(obs) > 100:
                m0 = s.do_match; s.do_match = abs(np.interp(t, N["t_od"], w_od)) < 0.6
                k0 = len(rec); acc0 = s.stats["accepted"]
                s._process((t, o, obs.astype(np.float32), flo[::3].astype(np.float32), obs.astype(np.float32), None, None)); s.do_match = m0
                if record and len(rec) > k0: rec[-1] = rec[-1] + (t, s.stats["accepted"] > acc0)
        est.append([t, *compose(s.T, o)])
    est = np.array(est); est[:, 3] = np.unwrap(est[:, 3]); return L, est, rec, s, time.time() - w0


def calib(log, lparams, out):
    L, est, rec, s, wall = replay(log, None, True)
    o0 = LR.odom_series(L)(est[0, 0]); g0 = L.gt(est[0, 0]); off = g0[2] - o0[2]
    E = []
    for r in rec:
        if len(r) < 4 or not r[3]: continue
        guess, best, t, _ = r; need = wrap(L.gt(t)[2] - off - guess[2]); E.append(wrap(best[2] - guess[2]) - need)
    E = np.array(E); p = json.load(open(lparams))["odom"]
    h = dict(q_th_rot=p["qtr"], q_th_dist=p["qtd"], r_th=float(np.var(E)), r_th_sd_deg=float(math.degrees(np.std(E))),
             r_th_bias_deg=float(math.degrees(E.mean())), n=len(E), train=log)
    json.dump(h, open(out, "w"), indent=1); print(h)


def run(log, out, headkf=None):
    L, est, rec, s, wall = replay(log, headkf)
    r = evaluate(est, L, with_map=False); r.update(log=log, headkf=headkf, stats=s.stats, wall_s=round(wall, 1))
    json.dump(r, open(out, "w"), indent=1, default=float)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("ate_rmse_m", "ate_max_m", "head_max_deg", "ate_aligned_rmse_m", "log", "headkf", "wall_s")})


if __name__ == "__main__":
    if sys.argv[1] == "calib": calib(*sys.argv[2:5])
    else: run(*sys.argv[2:5])
