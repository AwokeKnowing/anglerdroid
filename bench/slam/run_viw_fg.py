"""viw_fg: visual-inertial-wheel factor graph (GTSAM iSAM2), the textbook construction, no tuned weights:
  states       keyframes at 5 Hz (every 3rd camera frame): base pose X (Pose3), velocity V, IMU bias B
  IMU          CombinedImuFactor (Forster et al. on-manifold preintegration), body_P_sensor = twin base<-rs2_imu,
               IMU stamps from imu_sync.csv (clock offset MEASURED by prep_rtx.py), noise = sensor model / datasheet:
               gyro 2.12e-4 rad/s/sqrt(Hz), accel 2.12e-3 m/s^2/sqrt(Hz), gyro bias walk 3e-5, accel bias walk 1e-4
  wheel        BetweenFactor<Pose3> of the planar unicycle increment integrated from the 100 Hz encoders.  Sigmas from the
               wheel spec: x 0.005|ds| (radius tolerance) + encoder noise, y 0.001|ds| + encoder noise (no-slip constraint),
               yaw 0.015|dth| (track tolerance) + encoder noise, z 5 mm (1 cm carpet ridges), roll/pitch 0.05 rad (free-ish:
               the base tilts on the carpet; gravity from the IMU owns them).  Turning: the gyro's noise is far below the wheel
               yaw sigma, so rotation comes from the IMU; straight: the wheel's translation sigma is below the IMU's
               double-integrated drift, so translation comes from the wheels -- by the covariances, not by a switch.
  visual       Open3D RGB-D odometry (Steinbrucker/Park hybrid photometric+geometric term, defaults) between consecutive
               keyframes on the RTX RS2 colour + registered depth, initialised with the wheel+IMU prediction; noise = the
               information matrix Open3D returns (as in Open3D's own pose-graph reconstruction), Huber robust kernel.
Outputs the ONLINE estimate of each keyframe right after its iSAM2 update (causal).
python bench/slam/run_viw_fg.py DATA/<seq> RESULTS/<seq> [--no-visual] [--tag r0]   (~/slambench/z/kissvenv: gtsam, open3d)"""
import argparse, json, math, os, time
import numpy as np, cv2, gtsam, open3d as o3d
from gtsam.symbol_shorthand import X, V, B
from scipy.spatial.transform import Rotation as Rot


def Tm(e): T = np.eye(4); T[:3, :3] = e["R"]; T[:3, 3] = e["t"]; return T


def adjoint(T):
    R, p = T[:3, :3], T[:3, 3]; P = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    A = np.zeros((6, 6)); A[:3, :3] = R; A[3:, 3:] = R; A[3:, :3] = P @ R; return A      # [w; v] ordering (GTSAM/Open3D)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res"); ap.add_argument("--no-visual", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); seq, res = os.path.abspath(a.seq), os.path.abspath(a.res); os.makedirs(res, exist_ok=True)
    meta = json.load(open(os.path.join(seq, "meta.json")))
    rtx = json.load(open(os.path.join(seq, "rtx.json"))) if not a.no_visual else None
    F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True); W = np.genfromtxt(os.path.join(seq, "wheel.csv"), delimiter=",", names=True)
    I = np.genfromtxt(os.path.join(seq, "imu_sync.csv"), delimiter=",", names=True)
    track = meta["twin"]["track_m"]; r_w = meta["twin"]["wheel_r_m"]
    Tbi = Tm(meta["extr_twin_base_from_optical"]["rs2_imu"])
    if rtx:
        Tbc = np.array(rtx["rs2_640"]["T_base_optical"]); Tcb = np.linalg.inv(Tbc); Ad_bc = adjoint(Tbc)
        c = rtx["rs2_640"]["intr"]; K = o3d.camera.PinholeCameraIntrinsic(c["w"], c["h"], c["f"], c["f"], c["cx"], c["cy"])
    p = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    p.setGyroscopeCovariance((0.003 / math.sqrt(200)) ** 2 * np.eye(3)); p.setAccelerometerCovariance((0.03 / math.sqrt(200)) ** 2 * np.eye(3))
    p.setIntegrationCovariance(1e-8 * np.eye(3)); p.setBiasOmegaCovariance((3e-5) ** 2 * np.eye(3)); p.setBiasAccCovariance((1e-4) ** 2 * np.eye(3))
    p.setBodyPSensor(gtsam.Pose3(Tbi))
    kf = list(range(0, len(F["t"]), 3)); tk = F["t"][kf]
    # gravity-aligned initial attitude from the first 0.5 s of accelerometer (robot at rest)
    m0 = I["t"] < tk[0] + 0.5; g_s = np.array([I["ax"][m0].mean(), I["ay"][m0].mean(), I["az"][m0].mean()]); g_b = Tbi[:3, :3] @ g_s
    roll0, pitch0 = math.atan2(g_b[1], g_b[2]), math.atan2(-g_b[0], math.hypot(g_b[1], g_b[2]))
    x0 = gtsam.Pose3(gtsam.Rot3.Ypr(0.0, pitch0, roll0), np.zeros(3))
    isam = gtsam.ISAM2(gtsam.ISAM2Params()); G = gtsam.NonlinearFactorGraph(); Vals = gtsam.Values()
    G.add(gtsam.PriorFactorPose3(X(0), x0, gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02, 1e-3, 1e-3, 1e-3, 1e-3]))))
    G.add(gtsam.PriorFactorVector(V(0), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, 0.01)))
    G.add(gtsam.PriorFactorConstantBias(B(0), gtsam.imuBias.ConstantBias(), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.0035 * 3, 0.0035 * 3, 0.0035 * 3]))))
    Vals.insert(X(0), x0); Vals.insert(V(0), np.zeros(3)); Vals.insert(B(0), gtsam.imuBias.ConstantBias())
    isam.update(G, Vals); G = gtsam.NonlinearFactorGraph(); Vals = gtsam.Values()
    est = isam.calculateEstimate(); out = [np.r_[tk[0], x0.translation(), Rot.from_matrix(x0.rotation().matrix()).as_quat()]]
    enc_sd = 0.002 * 2 * math.pi * r_w                                         # encoder speed noise per wheel (m/s, per 100 Hz sample)
    huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
    def rgbd(k):
        fi = int(F["i"][kf[k]])
        col = o3d.io.read_image(os.path.join(seq, "rtx_rs2", "%06d.jpg" % fi)); dep = o3d.io.read_image(os.path.join(seq, "rtx_rs2_depth", "%06d.png" % fi))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(col, dep, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=True)
    prev_img = None if a.no_visual else rgbd(0); nvis = nvis_ok = 0; tvis = []; t_start = time.time(); ki = np.searchsorted(I["t"], tk[0]); wi = np.searchsorted(W["t"], tk[0])
    for k in range(1, len(kf)):
        t0, t1 = tk[k - 1], tk[k]
        pim = gtsam.PreintegratedCombinedMeasurements(p, est.atConstantBias(B(k - 1))); tl = t0
        while ki < len(I["t"]) and I["t"][ki] <= t1:
            ts = I["t"][ki]; dt = ts - tl
            if dt > 0: pim.integrateMeasurement(np.array([I["ax"][ki], I["ay"][ki], I["az"][ki]]), np.array([I["gx"][ki], I["gy"][ki], I["gz"][ki]]), dt)
            tl = ts; ki += 1
        if t1 - tl > 1e-5: pim.integrateMeasurement(np.array([I["ax"][ki - 1], I["ay"][ki - 1], I["az"][ki - 1]]), np.array([I["gx"][ki - 1], I["gy"][ki - 1], I["gz"][ki - 1]]), t1 - tl)
        G.add(gtsam.CombinedImuFactor(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim))
        # wheel increment (unicycle, midpoint)
        dx = dy = dth = 0.0; n = 0; tl = t0
        while wi < len(W["t"]) and W["t"][wi] <= t1:
            dt = W["t"][wi] - tl; vl, vr = W["vl"][wi], W["vr"][wi]; v, w = 0.5 * (vl + vr), (vr - vl) / track
            th = dth + 0.5 * w * dt; dx += v * dt * math.cos(th); dy += v * dt * math.sin(th); dth += w * dt; tl = W["t"][wi]; wi += 1; n += 1
        T_dt = t1 - t0; ds = math.hypot(dx, dy); e_v = enc_sd * T_dt / math.sqrt(max(n, 1)) / math.sqrt(2); e_w = enc_sd * math.sqrt(2) / track * T_dt / math.sqrt(max(n, 1))
        sw = np.array([0.05, 0.05, 0.015 * abs(dth) + e_w + 1e-5, 0.005 * ds + e_v + 1e-5, 0.001 * ds + e_v + 1e-5, 0.005])
        Twh = gtsam.Pose3(gtsam.Rot3.Rz(dth), np.array([dx, dy, 0.0])); G.add(gtsam.BetweenFactorPose3(X(k - 1), X(k), Twh, gtsam.noiseModel.Diagonal.Sigmas(sw)))
        # prediction
        nav = pim.predict(gtsam.NavState(est.atPose3(X(k - 1)), est.atVector(V(k - 1))), est.atConstantBias(B(k - 1)))
        xp = gtsam.Pose3(nav.pose().rotation(), est.atPose3(X(k - 1)).compose(Twh).translation())   # IMU rotation, wheel translation as the initial guess
        Vals.insert(X(k), xp); Vals.insert(V(k), nav.velocity()); Vals.insert(B(k), est.atConstantBias(B(k - 1)))
        if not a.no_visual:
            img = rgbd(k); Tpk = est.atPose3(X(k - 1)).between(xp).matrix(); init = Tcb @ Tpk @ Tbc          # T_c(k-1)_c(k)
            q0 = time.perf_counter()
            ok, T_c, info = o3d.pipelines.odometry.compute_rgbd_odometry(img, prev_img, K, init, o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), o3d.pipelines.odometry.OdometryOption())
            tvis.append(time.perf_counter() - q0); nvis += 1
            if ok and np.all(np.isfinite(info)) and np.linalg.matrix_rank(info) == 6:
                Tb = Tbc @ T_c @ Tcb; Ai = np.linalg.inv(Ad_bc); info_b = Ai.T @ info @ Ai     # information transforms with the inverse adjoint
                G.add(gtsam.BetweenFactorPose3(X(k - 1), X(k), gtsam.Pose3(Tb), gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Gaussian.Information(0.5 * (info_b + info_b.T)))))
                nvis_ok += 1
            prev_img = img
        isam.update(G, Vals); G = gtsam.NonlinearFactorGraph(); Vals = gtsam.Values(); est = isam.calculateEstimate()
        x = est.atPose3(X(k)); out.append(np.r_[t1, x.translation(), Rot.from_matrix(x.rotation().matrix()).as_quat()])
        if k % 200 == 0: print("viw k %d/%d  vis ok %d/%d  %.1f s" % (k, len(kf), nvis_ok, nvis, time.time() - t_start), flush=True)
    wall = time.time() - t_start; name = ("wi_fg" if a.no_visual else "viw_fg") + a.tag
    np.savetxt(os.path.join(res, name + ".tum"), np.array(out), fmt="%.6f")
    bfin = est.atConstantBias(B(len(kf) - 1))
    json.dump(dict(frame="base", per_frame=False, note="GTSAM iSAM2: IMU preintegration + wheel%s factors, keyframes 5 Hz, online estimates" % ("" if a.no_visual else " + Open3D RGB-D odometry"),
                   visual_edges_ok=nvis_ok, visual_edges=nvis, gyro_bias_final=list(map(float, bfin.gyroscope())), drawn_gyro_bias=meta.get("drawn_errors", {}).get("gyro_bias"),
                   runtime="%.1f ms/keyframe total (%.1f ms Open3D), %d keyframes, %.1f s wall (%.1f input fps)" % (1e3 * wall / len(kf), 1e3 * np.mean(tvis) if tvis else 0, len(kf), wall, len(F["t"]) / wall)),
              open(os.path.join(res, name + ".json"), "w"), indent=1)
    print(name, "done", len(out), "kf, vis ok", nvis_ok, "/", nvis, "wall %.0f s" % wall, flush=True)


if __name__ == "__main__":
    main()
