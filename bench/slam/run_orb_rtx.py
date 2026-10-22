"""ORB-SLAM3 (stock, default ORB/IMU-init parameters, viewer off) on the photoreal RTX cameras of a bench log:
  orb_rgbd       RGB-D           RTX RS2 colour + registered RS2 depth (prep_rtx.py)
  orb_rgbd_imu   RGB-D-Inertial  + D435i IMU (imu_sync.csv: measured clock offset removed)
  orb_mono_imu   Mono-Inertial   RTX mast camera + D435i IMU (scale from the IMU)
  orb_mono       Monocular       RTX mast (scale-free: Sim(3)-aligned to GT for scoring, flagged)
Camera->IMU extrinsics from the twin (IMU = rs2_imu optical frame); IMU noise from the sensor model / datasheet:
NoiseGyro = 0.003 rad/s white per 200 Hz sample -> 2.12e-4 rad/s/sqrt(Hz) (BMI055 datasheet 0.014 deg/s/sqrt(Hz) = 2.4e-4),
NoiseAcc = 0.03 m/s^2 per sample -> 2.12e-3 m/s^2/sqrt(Hz) (BMI055 150 ug/sqrt(Hz) = 1.5e-3), GyroWalk 3e-5 rad/s^2/sqrt(Hz)
(emulator 'realistic' g_rw), AccWalk 1e-4 (ORB-SLAM3's own D435i file; the emulator has no accel bias walk).
Per-frame ONLINE poses (what the robot would get; tracking OK or IMU-predicted) are scored.
python bench/slam/run_orb_rtx.py DATA/<seq> RESULTS/<seq> [--modes orb_rgbd,orb_rgbd_imu,orb_mono_imu,orb_mono] [--tag r0]"""
import argparse, json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORB = os.environ.get("ORB_SLAM3_DIR", os.path.expanduser("~/gits/awokeknowing/ORB_SLAM3")); PREFIX = os.path.expanduser("~/.local/share/mamba/envs/orbslam3")
BIN = os.path.join(REPO, "sim/orbslam3/kevin/orb_kevin")
IMU = dict(NoiseGyro=0.003 / math.sqrt(200), NoiseAcc=0.03 / math.sqrt(200), GyroWalk=3e-5, AccWalk=1e-4, Frequency=200.0)


def Tm(e): T = np.eye(4); T[:3, :3] = e["R"]; T[:3, 3] = e["t"]; return T


def yaml(cam, T_b_c, fps, rgbd):
    L = ['%YAML:1.0', 'File.version: "1.0"', 'Camera.type: "PinHole"']
    L += ["Camera1.%s: %.6f" % (k, v) for k, v in (("fx", cam["f"]), ("fy", cam["f"]), ("cx", cam["cx"]), ("cy", cam["cy"]), ("k1", cam.get("k1", 0.0)), ("k2", cam.get("k2", 0.0)), ("p1", 0.0), ("p2", 0.0))]
    L += ["Camera.width: %d" % cam["w"], "Camera.height: %d" % cam["h"], "Camera.fps: %d" % fps, "Camera.RGB: 1"]
    if rgbd: L += ["Stereo.ThDepth: 40.0", "Stereo.b: 0.0745", "RGBD.DepthMapFactor: 1000.0"]        # = ORB-SLAM3 RealSense_D435i.yaml
    L += ["IMU.T_b_c1: !!opencv-matrix", "   rows: 4", "   cols: 4", "   dt: f", "   data: [" + ", ".join("%.7f" % v for v in T_b_c.ravel()) + "]"]
    L += ["IMU.InsertKFsWhenLost: 0"] + ["IMU.%s: %.6e" % kv for kv in IMU.items()]   # always a real literal (cv::FileStorage rejects "200")
    L += ["ORBextractor.nFeatures: %d" % (1250 if rgbd else 1000), "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8", "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7"]
    L += ["Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9", "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08",
          "Viewer.CameraLineWidth: 3.0", "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -3.5", "Viewer.ViewpointF: 500.0"]
    return "\n".join(L) + "\n"


def run(mode, seq, res, tag):
    meta = json.load(open(os.path.join(seq, "meta.json"))); rtx = json.load(open(os.path.join(seq, "rtx.json")))
    Tbi = Tm(meta["extr_twin_base_from_optical"]["rs2_imu"])
    cname = "rs2_640" if mode.startswith("orb_rgbd") else "mast"; cam = rtx[cname]["intr"]; Tbc = np.array(rtx[cname]["T_base_optical"])
    wd = os.path.join(res, "_%s%s" % (mode, tag)); os.makedirs(wd, exist_ok=True)
    yp = os.path.join(wd, "cam.yaml"); open(yp, "w").write(yaml(cam, np.linalg.inv(Tbi) @ Tbc, int(round(meta["cam_hz"])), mode.startswith("orb_rgbd")))
    omode = mode[4:]; lst = os.path.join(seq, "list_rgbd.txt" if omode.startswith("rgbd") else "list_mast.txt")
    env = dict(os.environ, LD_LIBRARY_PATH="%s/lib:%s/lib:%s" % (PREFIX, ORB, os.environ.get("LD_LIBRARY_PATH", "")), OMP_NUM_THREADS="2")
    args = [BIN, omode, ORB + "/Vocabulary/ORBvoc.txt", yp, seq, lst, os.path.join(wd, "out")] + ([os.path.join(seq, "imu_sync.csv")] if "imu" in omode else [])
    t0 = time.time(); p = subprocess.run(args, cwd=wd, env=env, stdout=open(os.path.join(wd, "log.txt"), "w"), stderr=subprocess.STDOUT); wall = time.time() - t0
    log = open(os.path.join(wd, "log.txt"), errors="replace").read(); line = [l for l in log.split("\n") if l.startswith("ORB_KEVIN")]
    on = os.path.join(wd, "out_online.txt"); D = np.loadtxt(on, ndmin=2) if os.path.exists(on) and os.path.getsize(on) else np.zeros((0, 8))
    name = mode + tag; np.savetxt(os.path.join(res, name + ".tum"), D, fmt="%.6f")
    sf = omode == "mono"
    if not sf and len(D):                                    # base-frame copy for the shared occupancy mapper
        from scipy.spatial.transform import Rotation as Rot
        Tcb = np.linalg.inv(Tbc); B = []
        for r in D:
            T = np.eye(4); T[:3, :3] = Rot.from_quat(r[4:8]).as_matrix(); T[:3, 3] = r[1:4]; T = T @ Tcb
            B.append(np.r_[r[0], T[:3, 3], Rot.from_matrix(T[:3, :3]).as_quat()])
        np.savetxt(os.path.join(res, name + ".base.txt"), np.array(B), fmt="%.6f")
    nf = sum(1 for _ in open(lst)); n_maps = log.count("Creation of new map")
    json.dump(dict(frame="cam:" + cname, T_base_optical=Tbc.tolist(), scale_free=sf, per_frame=True, rc=p.returncode,
                   note="ORB-SLAM3 %s on RTX %s, stock params, online per-frame poses; maps created %d" % (omode, cname, n_maps),
                   runtime="%s; wall %.1f s for %d frames (%.1f fps)" % (line[0] if line else "crashed rc=%d" % p.returncode, wall, nf, nf / wall)),
              open(os.path.join(res, name + ".json"), "w"), indent=1)
    print(mode, tag, "rc", p.returncode, line[0] if line else "", "poses", len(D), "wall %.0f" % wall, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res"); ap.add_argument("--modes", default="orb_rgbd,orb_rgbd_imu,orb_mono_imu,orb_mono")
    ap.add_argument("--tag", default=""); ap.add_argument("-j", type=int, default=2)
    a = ap.parse_args(); seq, res = os.path.abspath(a.seq), os.path.abspath(a.res); os.makedirs(res, exist_ok=True)
    with ThreadPoolExecutor(a.j) as ex: list(ex.map(lambda m: run(m, seq, res, a.tag), a.modes.split(",")))
