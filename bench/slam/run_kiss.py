"""KISS-ICP 1.3 (Vizzo et al. 2023; default config, voxel = max_range/100 as kiss_icp_pipeline does) on the forward RS2
depth as a point cloud (848x480, every 2nd pixel, 0.2..4 m = the sensor's usable range; deskew off: global shutter).
Points are expressed in base_link (twin extrinsic) so the estimate is the base trajectory.
python bench/slam/run_kiss.py DATA/<seq> RESULTS/<seq> [--tag r0]   (~/slambench/z/kissvenv)"""
import argparse, json, os, sys, time
import numpy as np, cv2
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res"); ap.add_argument("--tag", default=""); a = ap.parse_args()
seq, res = os.path.abspath(a.seq), os.path.abspath(a.res); os.makedirs(res, exist_ok=True)
meta = json.load(open(os.path.join(seq, "meta.json"))); I = meta["intr"]["rs2_depth"]; e = meta["extr_twin_base_from_optical"]["rs2_depth"]; R, t = np.array(e["R"]), np.array(e["t"])
v, u = np.mgrid[0:I["h"]:2, 0:I["w"]:2]; u, v = u.ravel(), v.ravel(); rx, ry = (u - I["ppx"]) / I["fx"], (v - I["ppy"]) / I["fy"]
c = KISSConfig(); c.data.max_range = 4.0; c.data.min_range = 0.2; c.data.deskew = False; c.mapping.voxel_size = c.data.max_range / 100.0
k = KissICP(c); F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True); out = []; ms = []
for fi, tf in zip(F["i"].astype(int), F["t"]):
    d = cv2.imread(os.path.join(seq, "rs2_depth", "%06d.png" % fi), cv2.IMREAD_UNCHANGED)[v, u].astype(np.float64) / 1000.0
    ok = (d > 0.2) & (d < 4.0); P = np.stack([rx[ok] * d[ok], ry[ok] * d[ok], d[ok]], 1) @ R.T + t
    t0 = time.perf_counter(); k.register_frame(P, np.zeros(len(P))); ms.append((time.perf_counter() - t0) * 1e3)
    T = k.last_pose; out.append(np.r_[tf, T[:3, 3], Rot.from_matrix(T[:3, :3]).as_quat()])
name = "kiss_icp" + a.tag; np.savetxt(os.path.join(res, name + ".tum"), np.array(out), fmt="%.6f")
json.dump(dict(frame="base", per_frame=True, note="KISS-ICP 1.3 defaults on RS2 depth cloud (base frame)", runtime="%.1f ms/frame (%.0f fps)" % (np.mean(ms), 1e3 / np.mean(ms))),
          open(os.path.join(res, name + ".json"), "w"), indent=1)
print("kiss done", len(out), "%.1f ms/frame" % np.mean(ms))
