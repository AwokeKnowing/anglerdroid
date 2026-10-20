"""RS2 floor pixels that the safety core's Rs2Cloud (static extrinsic) labels OBSTACLE, judged by this frame's fitted floor plane."""
import sys, numpy as np, cv2
from emu.tools.slamlog.log import Log, depth_u16
from emu.tools.slamlog.lslam_replay import floor_plane
from nav.perceive import Rs2Cloud
import twin_geometry as G
for lg in sys.argv[1:]:
    L = Log(lg); s = L.streams["depth"]; intr = dict(width=s["w"], height=s["h"], fx=s["fx"], fy=s["fy"], ppx=s["ppx"], ppy=s["ppy"])
    rc = Rs2Cloud(intr); fr = L.frames["rs2"][::10]; out = []
    for t, f in fr:
        d = depth_u16(L.d + "/" + f)
        z = d[::rc.s, ::rc.s].ravel().astype(np.float32) * 1e-3; ok = (z > 0.2) & (z < rc.max_range); z = z[ok]
        P = np.stack([rc.rx[ok] * z, rc.ry[ok] * z, z], 1) @ rc.R.T + rc.T; h = P[:, 2] + G.AXLE_H_M; r = np.hypot(P[:, 0], P[:, 1])
        hmin = 0.03 + 0.012 * z * z; obs = (h > hmin) & (h < 1.0)
        pl = floor_plane(P[:, 0], P[:, 1], h, r)
        if pl is None: continue
        hf = h - (pl[0] * P[:, 0] + pl[1] * P[:, 1] + pl[2]); phantom = obs & (np.abs(hf) < 0.015)
        if abs(pl[0]) > 0.2 or abs(pl[1]) > 0.2: continue
        out.append([phantom.sum(), np.degrees(np.arctan(pl[0])), np.degrees(np.arctan(pl[1])), pl[2] * 1e3, r[phantom].min() if phantom.any() else np.nan])
    A = np.array(out); print(lg.split("/")[-1], "frames", len(A), "with phantom floor-obstacle px: %.1f %%" % (100 * (A[:, 0] > 0).mean()),
          "| px/frame p99 %.0f max %.0f | nearest phantom m p5 %.2f | plane tilt x p95 %.2f deg (max %.2f), y p95 %.2f, offset p95 %.1f mm" % (
          np.percentile(A[:, 0], 99), A[:, 0].max(), np.nanpercentile(A[:, 4], 5) if np.isfinite(A[:, 4]).any() else np.nan,
          np.percentile(abs(A[:, 1]), 95), abs(A[:, 1]).max(), np.percentile(abs(A[:, 2]), 95), np.percentile(abs(A[:, 3]), 95)))
