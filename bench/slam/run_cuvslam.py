"""cuVSLAM (NVIDIA PyCuVSLAM, the Isaac ROS visual SLAM core that runs on Orin) on a recorded sim log.
Modes: stereo (RS2 IR pair), stereo_imu (+ D435i IMU, Inertial odometry), rgbd (IR1 + aligned depth).
Twin calibration (intrinsics, IR baseline, extrinsics) is given: cuVSLAM does not self-calibrate.
-> RESULTS/<seq>/cuvslam_<mode>.tum  (base frame)     python bench/slam/run_cuvslam.py DATA/<seq> RESULTS/<seq> MODE"""
import json, math, os, sys, time
import numpy as np, cv2
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path[:0] = [REPO, os.path.join(REPO, "src")]
import cuvslam
from cuvslam_tracker import rig_from_twin, _quat, _R_RIG_BASE, IMU_GYRO_NOISE_DENSITY, IMU_GYRO_RANDOM_WALK, IMU_ACCEL_NOISE_DENSITY, IMU_ACCEL_RANDOM_WALK
import twin_geometry as TG
from scipy.spatial.transform import Rotation as Rot

seq, res, mode = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(res, exist_ok=True)
meta = json.load(open(os.path.join(seq, "meta.json"))); I = meta["intr"]["rs2_ir"]; W_, H_ = I["w"], I["h"]
cams = []
for k in (1, 2):
    c = cuvslam.Camera(size=[W_, H_], principal=[I["ppx"], I["ppy"]], focal=[I["fx"], I["fy"]])
    R, t = rig_from_twin("rs2_ir%d" % k); c.rig_from_camera = cuvslam.Pose(rotation=_quat(R), translation=t.tolist()); cams.append(c)
if mode == "rgbd": cams = cams[:1]
rig = cuvslam.Rig(cams)
if mode == "stereo_imu":
    ic = cuvslam.ImuCalibration(gyroscope_noise_density=IMU_GYRO_NOISE_DENSITY, gyroscope_random_walk=IMU_GYRO_RANDOM_WALK,
                                accelerometer_noise_density=IMU_ACCEL_NOISE_DENSITY, accelerometer_random_walk=IMU_ACCEL_RANDOM_WALK, frequency=200.0)
    Ri, ti = rig_from_twin("rs2_imu"); ic.rig_from_imu = cuvslam.Pose(rotation=_quat(Ri), translation=ti.tolist()); rig.imus = [ic]
    oc = cuvslam.Tracker.OdometryConfig(odometry_mode=cuvslam.Tracker.OdometryMode.Inertial, rectified_stereo_camera=True)
elif mode == "rgbd":
    oc = cuvslam.Tracker.OdometryConfig(odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
                                        rgbd_settings=cuvslam.Tracker.OdometryRGBDSettings(depth_scale_factor=1000.0, depth_camera_id=0, enable_depth_stereo_tracking=False))
else:
    oc = cuvslam.Tracker.OdometryConfig(rectified_stereo_camera=True)
tr = cuvslam.Tracker(rig, oc, cuvslam.Tracker.SlamConfig())
F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
IMU = np.genfromtxt(os.path.join(seq, "imu.csv"), delimiter=",", names=True); ki = 0
T_rig_base = np.eye(4); T_rig_base[:3, :3] = _R_RIG_BASE; T_rig_base[:3, 3] = _R_RIG_BASE @ np.array([0, 0, TG.AXLE_H_M])
out = open(os.path.join(res, "cuvslam_%s.tum" % mode), "w"); ms = []; lost = 0
for i, tf in enumerate(F["t"]):
    ts = int(tf * 1e9)
    if mode == "stereo_imu":
        while ki < len(IMU["t"]) and IMU["t"][ki] <= tf:
            m = cuvslam.ImuMeasurement(); m.timestamp_ns = int(IMU["t"][ki] * 1e9)
            m.linear_accelerations = [float(IMU["ax"][ki]), float(IMU["ay"][ki]), float(IMU["az"][ki])]
            m.angular_velocities = [float(IMU["gx"][ki]), float(IMU["gy"][ki]), float(IMU["gz"][ki])]
            tr.register_imu_measurement(0, m); ki += 1
    l = cv2.imread(os.path.join(seq, "rs2_ir1", "%06d.png" % i), cv2.IMREAD_GRAYSCALE)
    t0 = time.perf_counter()
    if mode == "rgbd":
        d = cv2.imread(os.path.join(seq, "rs2_depth", "%06d.png" % i), cv2.IMREAD_UNCHANGED); pe, sp = tr.track(ts, [l], depths=[d])
    else:
        r = cv2.imread(os.path.join(seq, "rs2_ir2", "%06d.png" % i), cv2.IMREAD_GRAYSCALE); pe, sp = tr.track(ts, [l, r])
    ms.append((time.perf_counter() - t0) * 1e3)
    p = sp if (sp is not None and getattr(sp, "translation", None) is not None) else (pe.world_from_rig.pose if pe.world_from_rig is not None else None)
    if p is None: lost += 1; continue
    Twr = np.eye(4); Twr[:3, :3] = Rot.from_quat(list(p.rotation)).as_matrix(); Twr[:3, 3] = list(p.translation)
    Twb = Twr @ T_rig_base; q = Rot.from_matrix(Twb[:3, :3]).as_quat()
    out.write("%.4f %.5f %.5f %.5f %.6f %.6f %.6f %.6f\n" % (tf, *Twb[:3, 3], *q))
out.close()
json.dump(dict(frame="base", note="PyCuVSLAM %s, twin calibration, SLAM on" % mode, runtime="%.1f ms/frame (p99 %.1f), lost %d" % (np.mean(ms), np.percentile(ms, 99), lost)),
          open(os.path.join(res, "cuvslam_%s.json" % mode), "w"))
print("cuvslam", mode, "frames", len(F["t"]), "lost", lost, "ms", round(float(np.mean(ms)), 1))
