"""Convert a real nav.record log (KEVIN_RECORD: one npz per 30 Hz control tick) into the bench/slam log layout so the
depth/wheel/IMU methods run unchanged (no RGB in these logs; no ground truth on the real robot).
  frames.csv + rs2_depth/%06d.png  every --every-th tick, depth decimated by --dec (intrinsics scaled accordingly)
  wheel.csv (30 Hz, t_od), imu.csv (every IMU sample once; device clock mapped to the tick clock by the median of
  meta.t - meta.imu_t; prep_rtx.imu_offset then MEASURES the residual), cmd.csv (meta v, w), odom_robot.tum (robot's own)
  meta.json  intrinsics from the log, extrinsics/twin from a sim log's meta.json (--twin-meta, same twin)
python3 real_to_bench.py LOGDIR OUT --twin-meta meta.json [--every 3] [--dec 2]   (numpy + cv2; runs on the Orin)"""
import argparse, glob, json, os
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("out"); ap.add_argument("--twin-meta", required=True)
ap.add_argument("--every", type=int, default=3); ap.add_argument("--dec", type=int, default=2); a = ap.parse_args()
os.makedirs(os.path.join(a.out, "rs2_depth"), exist_ok=True)
files = sorted(glob.glob(os.path.join(a.log, "t*.npz"))); M = json.load(open(a.twin_meta))
fw, fwh, fim, fcm, fod = (open(os.path.join(a.out, n), "w") for n in ("frames.csv", "wheel.csv", "imu.csv", "cmd.csv", "odom_robot.tum"))
fw.write("i,t\n"); fwh.write("t,vl,vr\n"); fim.write("t,gx,gy,gz,ax,ay,az\n"); fcm.write("t,v,w\n")
t0 = None; off = []; last_imu = -1e18; fi = 0; intr = None; imu_rows = []
for k, f in enumerate(files):
    z = np.load(f, allow_pickle=True); m = json.loads(str(z["meta"]))
    if t0 is None: t0 = m["t"]
    if m.get("imu_t"): off.append(m["t"] - m["imu_t"])
    fwh.write("%.4f,%.5f,%.5f\n" % (m["t_od"] - t0, m["vl"], m["vr"])); fcm.write("%.4f,%.4f,%.4f\n" % (m["t"] - t0, m["v"], m["w"]))
    po = m["odom"]; fod.write("%.4f %.5f %.5f 0 0 0 %.6f %.6f\n" % (m["t"] - t0, po[0], po[1], np.sin(po[2] / 2), np.cos(po[2] / 2)))
    if "imu_ts" in z.files:
        for ts, g, ac in zip(z["imu_ts"], z["imu_g"], z["imu_a"]):
            if ts > last_imu: imu_rows.append((ts, *g, *ac)); last_imu = ts
    if k % a.every == 0 and "rs2" in z.files:
        intr = intr or m["intr"]; d = z["rs2"][::a.dec, ::a.dec]
        cv2.imwrite(os.path.join(a.out, "rs2_depth", "%06d.png" % fi), d); fw.write("%d,%.4f\n" % (fi, m["t"] - t0)); fi += 1
c = float(np.median(off))
for r in imu_rows: fim.write("%.5f,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f\n" % (r[0] + c - t0, *r[1:]))
for x in (fw, fwh, fim, fcm, fod): x.close()
D = a.dec; I = dict(w=intr["width"] // D, h=intr["height"] // D, fx=intr["fx"] / D, fy=intr["fy"] / D, ppx=(intr["ppx"] + 0.5) / D - 0.5, ppy=(intr["ppy"] + 0.5) / D - 0.5)
meta = dict(env="real", source_log=os.path.abspath(a.log), cam_hz=30.0 / a.every, intr=dict(rs2_depth=I), extr_twin_base_from_optical=M["extr_twin_base_from_optical"],
            twin=M["twin"], robot_belief=M.get("robot_belief"), imu_clock_median_offset_s=c, frames=fi, notes="real robot log, no GT; depth decimated %dx" % D)
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("real_to_bench", fi, "frames", len(imu_rows), "imu", len(files), "ticks")
