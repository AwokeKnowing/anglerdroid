"""kfuse: put translation edges from a visual odometry (e.g. cuVSLAM stereo) under kslam's heading.
kslam provides yaw (wheel + gyro + Manhattan). For each frame interval, two body-frame displacements are fused by
inverse covariance:
  * wheel: sigma 5 mm + 3% of the step.  The wheel is flagged as slipping when the wheel yaw rate disagrees with the
    gyro yaw rate by more than 3 sigma; its sigma is then inflated 20x.  The gyro is the slip detector, not a
    hand-picked threshold on the vision side.
  * VO: sigma 4 mm + 2% of the step (its own rotation is NOT used, only its body-frame translation).  A step is
    rejected when its implied speed exceeds 1.5 m/s or it is missing (VO lost).
python bench/slam/kfuse.py DATA/<seq> RESULTS/<seq> [--yaw kslam] [--vo cuvslam_stereo] [--name kslam_vo]"""
import argparse, json, math, os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

def yaw_of(q): return Rot.from_quat(q).as_euler("zyx")[:, 0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res")
    ap.add_argument("--yaw", default="kslam"); ap.add_argument("--vo", default="cuvslam_stereo"); ap.add_argument("--name", default="kslam_vo")
    a = ap.parse_args(); seq, res = os.path.abspath(a.seq), os.path.abspath(a.res)
    M = json.load(open(os.path.join(seq, "meta.json"))); TR = M["twin"]["track_m"]
    K = np.loadtxt(os.path.join(res, a.yaw + ".tum")); V = np.loadtxt(os.path.join(res, a.vo + ".tum"))
    W = np.genfromtxt(os.path.join(seq, "wheel.csv"), delimiter=",", names=True)
    I = np.genfromtxt(os.path.join(seq, "imu.csv"), delimiter=",", names=True)
    R_imu = np.asarray(M["extr_twin_base_from_optical"]["rs2_imu"]["R"])
    gz = (R_imu @ np.stack([I["gx"], I["gy"], I["gz"]])).T[:, 2]
    kj = json.load(open(os.path.join(res, a.yaw + ".json"))); s = kj.get("selfcal", {}).get("gyro_scale_err_est", 0.0); b = kj.get("selfcal", {}).get("gyro_bias_est", 0.0)
    gz = gz * (1 + s) - b
    t = K[:, 0]; yk = yaw_of(K[:, 4:8])
    vt = V[:, 0]; VR = Rot.from_quat(V[:, 4:8]).as_matrix()
    vidx = {round(x, 3): j for j, x in enumerate(vt)}
    cw = np.concatenate([[0], np.cumsum(0.5 * (W["vl"] + W["vr"])[1:] * np.diff(W["t"]))])          # wheel distance
    cww = np.concatenate([[0], np.cumsum(((W["vr"] - W["vl"]) / TR)[1:] * np.diff(W["t"]))])        # wheel yaw
    cg = np.concatenate([[0], np.cumsum(gz[1:] * np.diff(I["t"]))])                                 # gyro yaw
    P = np.zeros((len(t), 2)); P[0] = K[0, 1:3]; nslip = 0; nvo = 0
    for k in range(1, len(t)):
        t0, t1 = t[k - 1], t[k]; dt = t1 - t0
        dw = np.interp(t1, W["t"], cw) - np.interp(t0, W["t"], cw)
        dyw = np.interp(t1, W["t"], cww) - np.interp(t0, W["t"], cww); dyg = np.interp(t1, I["t"], cg) - np.interp(t0, I["t"], cg)
        d_w = np.array([dw, 0.0]); sw = 0.005 + 0.03 * abs(dw)
        if abs(dyw - dyg) > 3 * (0.004 + 0.02 * abs(dyg)) + 0.003: sw *= 20; nslip += 1
        j1, j0 = vidx.get(round(t1, 3)), vidx.get(round(t0, 3)); d = d_w; 
        if j1 is not None and j0 is not None:
            d_v = (VR[j0].T @ (V[j1, 1:4] - V[j0, 1:4]))[:2]     # body-frame step (VO world frame need not be z-up)
            if np.linalg.norm(d_v) / max(dt, 1e-3) < 1.5:
                sv = 0.004 + 0.02 * np.linalg.norm(d_v); iw, iv = 1 / sw ** 2, 1 / sv ** 2
                d = (iw * d_w + iv * d_v) / (iw + iv); nvo += 1
        ym = yk[k - 1] + 0.5 * math.atan2(math.sin(yk[k] - yk[k - 1]), math.cos(yk[k] - yk[k - 1]))
        c, sn = math.cos(ym), math.sin(ym); P[k] = P[k - 1] + [c * d[0] - sn * d[1], sn * d[0] + c * d[1]]
    O = K.copy(); O[:, 1:3] = P
    np.savetxt(os.path.join(res, a.name + ".tum"), O, fmt="%.6f")
    json.dump(dict(frame="base", scale_free=False, per_frame=True, note="kfuse: %s heading + inverse-covariance wheel/%s translation, gyro-vs-wheel slip detection" % (a.yaw, a.vo),
                   runtime="post-process; slip-flagged steps %d, VO steps used %d of %d" % (nslip, nvo, len(t) - 1)), open(os.path.join(res, a.name + ".json"), "w"), indent=1)
    print(a.name, "slip", nslip, "vo", nvo, "of", len(t) - 1)

if __name__ == "__main__":
    main()
