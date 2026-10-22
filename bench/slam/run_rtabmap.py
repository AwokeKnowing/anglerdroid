"""RTAB-Map 0.22 (library, DEFAULT parameters; sim/rtabmap/rtabmap_kevin) on the RTX RS2 colour + registered depth:
  rtab_vo        RTAB-Map's own RGB-D visual odometry (F2M) + loop closure / graph optimisation
  rtab_wio       visual-inertial-WHEEL: the robot's own wheel+gyro odometry (nav.slam.Odom: wheel translation, gyro yaw
                 in a 3-state EKF [yaw, gyro scale, gyro bias] with ZUPT; IMU = imu_sync.csv, measured clock offset
                 removed) as RTAB-Map's odometry, RTAB-Map adds visual loop-closure / proximity links and optimises the graph
                 -- the standard rtabmap_ros wheel-odometry robot setup.  Odometry link covariance from the sensor model:
                 sigma_x = 0.005 |ds| (wheel radius tolerance) + 1e-4, sigma_y = 0.001 |ds| + 1e-4, z/roll/pitch 1e-4 (planar),
                 sigma_yaw = 2.12e-4 sqrt(dt) (gyro noise density) + 0.0035 dt (gyro bias bound) + 1e-5
python bench/slam/run_rtabmap.py DATA/<seq> RESULTS/<seq> [--modes rtab_vo,rtab_wio] [--tag r0]   (robot venv)"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path[:0] = [REPO]
BIN = os.path.join(REPO, "sim/rtabmap/rtabmap_kevin")
RT = os.path.expanduser(os.environ.get("RTABMAP_RT", "~/slambench/z/rtabmap_rt/lib"))   # vendored ldd closure of the micromamba env (disk)
ENV = dict(os.environ, LD_LIBRARY_PATH=RT + ":" + os.environ.get("LD_LIBRARY_PATH", "")) if os.path.isdir(RT) else None
from scipy.spatial.transform import Rotation as Rot


def wheel_gyro_odom(seq, meta):
    import nav  # noqa
    import twin_geometry as TG
    from nav.slam import Odom
    W = np.genfromtxt(os.path.join(seq, "wheel.csv"), delimiter=",", names=True); I = np.genfromtxt(os.path.join(seq, "imu_sync.csv"), delimiter=",", names=True)
    C = np.genfromtxt(os.path.join(seq, "cmd.csv"), delimiter=",", names=True); F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    gz_all = (TG.IMU_R @ ((np.stack([I["gx"], I["gy"], I["gz"]], 1) - TG.GYRO_BIAS) * TG.GYRO_SCALE).T)[2]; ti = I["t"]
    o = Odom(); fi = 0; k0 = 0; out = []
    for j in range(len(W["t"])):
        t = W["t"][j]; k1 = np.searchsorted(ti, t); gz = float(gz_all[k0:k1].mean()) if k1 > k0 else None; k0 = k1
        kc = max(0, np.searchsorted(C["t"], t) - 1); o.update(t, W["vl"][j], W["vr"][j], gz, commanded_zero=bool(C["v"][kc] == 0 and C["w"][kc] == 0))
        while fi < len(F["t"]) and F["t"][fi] <= t + 1e-6: out.append((F["t"][fi], *o.pose)); fi += 1
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res"); ap.add_argument("--modes", default="rtab_vo,rtab_wio"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); seq, res = os.path.abspath(a.seq), os.path.abspath(a.res); os.makedirs(res, exist_ok=True)
    meta = json.load(open(os.path.join(seq, "meta.json"))); rtx = json.load(open(os.path.join(seq, "rtx.json"))); c = rtx["rs2_640"]["intr"]; Tbo = np.array(rtx["rs2_640"]["T_base_optical"])
    camf = os.path.join(res, "_rtab_cam.txt"); open(camf, "w").write("%f %f %f %f %d %d\n" % (c["f"], c["f"], c["cx"], c["cy"], c["w"], c["h"]) + " ".join("%.8f" % v for v in Tbo[:3].ravel()) + "\n")
    lst = os.path.join(seq, "list_rgbd.txt"); nf = sum(1 for _ in open(lst))
    for mode in a.modes.split(","):
        name = mode + a.tag; wd = os.path.join(res, "_" + name); os.makedirs(wd, exist_ok=True); args = [BIN, "vo" if mode == "rtab_vo" else "odom", seq, lst, camf, os.path.join(wd, "out")]
        if mode == "rtab_wio":
            O = wheel_gyro_odom(seq, meta); rows = []; prev = None
            for t, x, y, th in O:
                q = Rot.from_euler("z", th).as_quat()
                if prev is None: ds, dt = 0.0, 0.0
                else: ds, dt = math.hypot(x - prev[1], y - prev[2]), t - prev[0]
                sd = [0.005 * ds + 1e-4, 0.001 * ds + 1e-4, 1e-4, 1e-4, 1e-4, 2.12e-4 * math.sqrt(max(dt, 0)) + 0.0035 * dt + 1e-5]
                rows.append([t, x, y, 0.0, *q, *sd]); prev = (t, x, y)
            of = os.path.join(wd, "odom.txt"); np.savetxt(of, np.array(rows), fmt="%.8f"); args.append(of)
        t0 = time.time(); p = subprocess.run(args, cwd=wd, stdout=open(os.path.join(wd, "log.txt"), "w"), stderr=subprocess.STDOUT, env=ENV); wall = time.time() - t0
        log = open(os.path.join(wd, "log.txt"), errors="replace").read(); line = [l for l in log.split("\n") if l.startswith("RTABMAP_KEVIN")]
        on = os.path.join(wd, "out_online.txt"); D = np.loadtxt(on, ndmin=2) if os.path.exists(on) and os.path.getsize(on) else np.zeros((0, 8))
        np.savetxt(os.path.join(res, name + ".tum"), D, fmt="%.6f")
        json.dump(dict(frame="base", per_frame=True, rc=p.returncode, note="RTAB-Map 0.22 defaults, %s, online map-corrected odometry per frame" % mode,
                       runtime="%s; wall %.1f s for %d frames (%.1f fps)" % (line[0] if line else "crashed rc=%d" % p.returncode, wall, nf, nf / wall)),
                  open(os.path.join(res, name + ".json"), "w"), indent=1)
        print(name, "rc", p.returncode, line[0] if line else log[-500:], "poses", len(D), flush=True)


if __name__ == "__main__":
    main()
