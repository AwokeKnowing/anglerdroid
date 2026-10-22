"""Inputs for the RTX-camera SLAM bench from a recorded log + render_rtx.py output (no ground truth used):
  rtx_rs2_depth/%06d.png  RS2 depth registered to the RTX RS2 colour camera (what rs2.align(color) does on the robot):
                          depth optical -> colour optical with the twin extrinsics, projected with the colour K, z-buffer
                          (nearest wins), u16 mm, 0 = no data
  imu_sync.csv            imu.csv with the IMU->camera clock offset REMOVED.  The offset is MEASURED, not assumed: the
                          lag that maximises the normalised cross-correlation between the gyro yaw rate (rotated into
                          base) and the wheel yaw rate (vr - vl) / track, on the wheel/camera clock, 0.5 ms grid
                          over +-60 ms (wheel slip / track error scale the rate but do not shift it in time).
  list_rgbd.txt, list_mast.txt   t  rgb  [depth]
  sync.json               measured offset + correlation peak
python bench/slam/prep_rtx.py DATA/<seq>   (robot or sim venv: numpy, cv2)"""
import json, math, os, sys
import numpy as np, cv2


def Tm(e): T = np.eye(4); T[:3, :3] = e["R"]; T[:3, 3] = e["t"]; return T


def imu_offset(seq, meta):
    I = np.genfromtxt(os.path.join(seq, "imu.csv"), delimiter=",", names=True); W = np.genfromtxt(os.path.join(seq, "wheel.csv"), delimiter=",", names=True)
    Rbi = np.array(meta["extr_twin_base_from_optical"]["rs2_imu"]["R"]); gz = (Rbi @ np.stack([I["gx"], I["gy"], I["gz"]])) [2]
    wz = (W["vr"] - W["vl"]) / meta["twin"]["track_m"]
    tt = np.arange(W["t"][0] + 1, W["t"][-1] - 1, 0.002); a = np.interp(tt, W["t"], wz); a = (a - a.mean()) / a.std()
    lags = np.arange(-0.060, 0.0605, 0.0005); cc = []
    for L in lags:
        b = np.interp(tt + L, I["t"], gz); b = (b - b.mean()) / b.std(); cc.append(float((a * b).mean()))
    k = int(np.argmax(cc))
    if 0 < k < len(cc) - 1:                       # parabolic sub-grid peak
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2); off = lags[k] + d * 0.0005
    else: off = lags[k]
    return float(off), float(cc[k]), I


def main():
    seq = os.path.abspath(sys.argv[1]); meta = json.load(open(os.path.join(seq, "meta.json"))); rtx = json.load(open(os.path.join(seq, "rtx.json")))
    F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    off, peak, I = imu_offset(seq, meta)
    with open(os.path.join(seq, "imu_sync.csv"), "w") as f:
        f.write("t,gx,gy,gz,ax,ay,az\n")
        for r in zip(I["t"] - off, I["gx"], I["gy"], I["gz"], I["ax"], I["ay"], I["az"]): f.write("%.5f,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f\n" % r)
    json.dump(dict(imu_minus_cam_offset_s=round(off, 5), xcorr_peak=round(peak, 4), method="gyro-z vs wheel yaw-rate cross-correlation"), open(os.path.join(seq, "sync.json"), "w"), indent=1)
    print("imu offset measured %.2f ms (peak ncc %.3f)" % (off * 1e3, peak), flush=True)
    # depth -> colour registration
    di = meta["intr"]["rs2_depth"]; ci = rtx["rs2_640"]["intr"]
    Tbd = Tm(meta["extr_twin_base_from_optical"]["rs2_depth"]); Tbc = np.array(rtx["rs2_640"]["T_base_optical"]); Tcd = np.linalg.inv(Tbc) @ Tbd
    v, u = np.mgrid[0:di["h"], 0:di["w"]]; u, v = u.ravel().astype(np.float64), v.ravel().astype(np.float64)
    W_, H_ = ci["w"], ci["h"]; od = os.path.join(seq, "rtx_rs2_depth"); os.makedirs(od, exist_ok=True)
    def proj(uu, vv, z):
        P = np.stack([(uu - di["ppx"]) / di["fx"] * z, (vv - di["ppy"]) / di["fy"] * z, z], 1) @ Tcd[:3, :3].T + Tcd[:3, 3]
        return ci["f"] * P[:, 0] / P[:, 2] + ci["cx"], ci["f"] * P[:, 1] / P[:, 2] + ci["cy"], P[:, 2]
    global _job
    def _job(fi):
        d = cv2.imread(os.path.join(seq, "rs2_depth", "%06d.png" % fi), cv2.IMREAD_UNCHANGED).ravel().astype(np.float64) / 1000.0
        ok = d > 0; z = d[ok]; uu, vv = u[ok], v[ok]
        # librealsense rs2::align style: project the depth pixel's footprint corners, fill every colour pixel inside
        a0, b0, zc = proj(uu - 0.5, vv - 0.5, z); a1, b1, _ = proj(uu + 0.5, vv + 0.5, z)
        out = np.full(H_ * W_, np.inf, np.float32); U0, V0 = np.ceil(a0 - 0.5), np.ceil(b0 - 0.5)
        o = np.argsort(-zc)                                                    # far first, nearest written last wins
        for dj in range(3):
            for di_ in range(3):
                jj, ii = U0 + dj, V0 + di_
                m = (jj < a1 + 0.5) & (ii < b1 + 0.5) & (jj >= 0) & (ii >= 0) & (jj < W_) & (ii < H_) & (zc > 0.1)
                k = o[m[o]]; q = (ii[k] * W_ + jj[k]).astype(np.int64); out[q] = np.minimum(out[q], zc[k])
        out[~np.isfinite(out)] = 0
        cv2.imwrite(os.path.join(od, "%06d.png" % fi), (out.reshape(H_, W_) * 1000).astype(np.uint16))
    from multiprocessing import Pool
    with Pool(int(os.environ.get("PREP_PROCS", "3"))) as pool: pool.map(_job, list(F["i"].astype(int)), chunksize=32)
    with open(os.path.join(seq, "list_rgbd.txt"), "w") as f:
        for fi, t in zip(F["i"].astype(int), F["t"]): f.write("%.4f rtx_rs2/%06d.jpg rtx_rs2_depth/%06d.png\n" % (t, fi, fi))
    with open(os.path.join(seq, "list_mast.txt"), "w") as f:
        for fi, t in zip(F["i"].astype(int), F["t"]): f.write("%.4f rtx_mast/%06d.jpg\n" % (t, fi))
    print("prep done", seq, len(F["t"]), "frames", flush=True)


if __name__ == "__main__":
    main()
