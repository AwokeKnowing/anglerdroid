"""Baselines on a recorded sim log: wheel-only odometry, wheel+gyro odometry (nav.slam.Odom: the robot's own dead
reckoning incl. ZUPT gyro-bias learning) and slam2 (nav/slam2.py, the current 2D SLAM, fed exactly like nav.run:
RS2 depth -> nav.perceive.Rs2Cloud obstacle/floor points at 5 Hz, skipped while |w| >= 0.6 rad/s), run synchronously.
-> RESULTS/<seq>/{odom_wheel,odom_wheel_gyro,slam2}.tum (+ .json)
python bench/slam/run_baselines.py DATA/<seq> RESULTS/<seq>   (robot venv)"""
import json, math, os, sys, time
import numpy as np, cv2
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path[:0] = [REPO]
import nav  # noqa  (sets src/ paths)
import twin_geometry as TG
from nav.slam import Odom, GridMap, compose
from scipy.spatial.transform import Rotation as Rot


def tum_line(t, x, y, th): q = Rot.from_euler("z", th).as_quat(); return "%.4f %.5f %.5f 0 %.6f %.6f %.6f %.6f\n" % (t, x, y, *q)


def main():
    seq, res = sys.argv[1], sys.argv[2]; os.makedirs(res, exist_ok=True)
    meta = json.load(open(os.path.join(seq, "meta.json")))
    W = np.genfromtxt(os.path.join(seq, "wheel.csv"), delimiter=",", names=True)
    I = np.genfromtxt(os.path.join(seq, "imu.csv"), delimiter=",", names=True)
    C = np.genfromtxt(os.path.join(seq, "cmd.csv"), delimiter=",", names=True)
    F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    gz_all = (TG.IMU_R @ ((np.stack([I["gx"], I["gy"], I["gz"]], 1) - TG.GYRO_BIAS) * TG.GYRO_SCALE).T)[2]
    ti = I["t"]
    def cmd_zero(t):
        k = max(0, np.searchsorted(C["t"], t) - 1); return C["v"][k] == 0 and C["w"][k] == 0
    from nav.slam2 import Slam2
    from nav.perceive import Rs2Cloud
    o_w, o_g = Odom(), Odom(); slam = Slam2(GridMap()); cloud = Rs2Cloud(dict(meta["intr"]["rs2_depth"], width=meta["intr"]["rs2_depth"]["w"], height=meta["intr"]["rs2_depth"]["h"]))
    out = {k: open(os.path.join(res, k + ".tum"), "w") for k in ("odom_wheel", "odom_wheel_gyro", "slam2")}
    fi = 0; nf = len(F["t"]); k0 = 0; ms = []; nsub = 0; t_prev = None
    for j in range(len(W["t"])):
        t = W["t"][j]; vl, vr = W["vl"][j], W["vr"][j]
        k1 = np.searchsorted(ti, t); gz = float(gz_all[k0:k1].mean()) if k1 > k0 else None; k0 = k1
        cz = cmd_zero(t)
        o_w.update(t, vl, vr, None, commanded_zero=cz); o_g.update(t, vl, vr, gz, commanded_zero=cz); slam.odom.update(t, vl, vr, gz, commanded_zero=cz)
        while fi < nf and F["t"][fi] <= t + 1e-6:
            tf = F["t"][fi]
            if fi % 3 == 0 and abs(slam.odom.w) < 0.6:          # 5 Hz map/match update, as nav.run
                d = cv2.imread(os.path.join(seq, "rs2_depth", "%06d.png" % fi), cv2.IMREAD_UNCHANGED)
                obs, flo = cloud.points(d); t0 = time.perf_counter()
                slam._step(tf, slam.odom.at(tf), obs.astype(np.float32), flo[::3].astype(np.float32)); ms.append((time.perf_counter() - t0) * 1e3); nsub += 1
            for k, od in (("odom_wheel", o_w.pose), ("odom_wheel_gyro", o_g.pose), ("slam2", slam.pose())): out[k].write(tum_line(tf, *od))
            fi += 1
    for f in out.values(): f.close()
    json.dump(dict(frame="base", note="wheel only (track %.3f m)" % TG.TRACK_M), open(os.path.join(res, "odom_wheel.json"), "w"))
    json.dump(dict(frame="base", note="wheel v + D435i gyro yaw, ZUPT bias (nav.slam.Odom)"), open(os.path.join(res, "odom_wheel_gyro.json"), "w"))
    st = {k: v for k, v in slam.stats.items() if isinstance(v, (int, float))}
    json.dump(dict(frame="base", note="nav/slam2.py RS2 depth scans @5Hz", runtime="%.1f ms/scan (%d scans)" % (np.mean(ms), nsub), stats=st), open(os.path.join(res, "slam2.json"), "w"))
    print("baselines done", seq, "frames", fi, "slam2", st)
    os._exit(0)


if __name__ == "__main__":
    main()
