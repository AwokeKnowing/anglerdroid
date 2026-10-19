"""Per-frame camera poses from nav's wheel+gyro odometry, for an exported RGB-D sequence (export_rgbd.py list.txt).
  python -m emu.tools.slamlog.odom_prior <log dir> <cam> <export dir> <out.npy>
out: (N, 4, 4) T_odom<-camera (OpenCV camera axes), N = lines of list.txt; frames outside the nav window are NaN.
Used as a motion prior (relative camera motion between consecutive frames) by external SLAM candidates."""
import sys, numpy as np
from .log import Log

d, cam, ex, out = sys.argv[1:5]
L = Log(d); N = L.nav(); R_bc, T_bc = L.ext[cam]          # X_body = X_cam @ R.T + T  (body frame at axle height)
ts = np.array([float(l.split()[0]) for l in open(ex + "/list.txt")])
t, od = N["t_od"], N["odom"]; th = np.unwrap(od[:, 2])
x, y, a = (np.interp(ts, t, v, left=np.nan, right=np.nan) for v in (od[:, 0], od[:, 1], th))
P = np.full((len(ts), 4, 4), np.nan)
for i in range(len(ts)):
    if np.isnan(x[i]): continue
    c, s = np.cos(a[i]), np.sin(a[i]); Twb = np.eye(4); Twb[:2, :2] = [[c, -s], [s, c]]; Twb[:2, 3] = x[i], y[i]
    Tbc = np.eye(4); Tbc[:3, :3] = R_bc; Tbc[:3, 3] = T_bc; P[i] = Twb @ Tbc
np.save(out, P); print("odom prior: %d/%d frames" % (int((~np.isnan(P[:, 0, 0])).sum()), len(P)))
