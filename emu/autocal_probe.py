"""(robot side, under scripts/kevin-hal) park/drive windows in a clear lane, run nav.autocal, measure RS2 map
quality with the nominal vs auto-calibrated pose.  Writes one JSON.  Used by emu.autocal_eval."""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import nav  # noqa
from nav.hw import Hardware
from nav.perceive import Rs2Cloud
from nav.autocal import AutoCal, STILL_V
from emu import gt


def quality(cloud, depth, T, R):
    cloud.set_pose(T, R)
    obs, _ = cloud.points(depth)
    z = depth[::4, ::4].ravel().astype(np.float32) * 1e-3; ok = (z > 0.2) & (z < 3.0)
    P = np.stack([cloud.rx[ok] * z[ok], cloud.ry[ok] * z[ok], z[ok]], 1) @ np.asarray(R).T + np.asarray(T)
    lane = (np.abs(P[:, 1]) < 0.3) & (P[:, 0] > 0.3) & (P[:, 0] < 3.0)
    from twin_geometry import AXLE_H_M
    h = P[lane, 2] + AXLE_H_M
    o = obs[(np.abs(obs[:, 1]) < 0.3) & (obs[:, 0] > 0.3) & (obs[:, 0] < 3.0)] if len(obs) else obs
    far = lane & (P[:, 0] > 2.0)
    return dict(phantom_pts=int(len(o)), floor_bias_cm=round(100 * float(np.median(h)), 2) if len(h) else None,
                floor_bias_far_cm=round(100 * float(np.median(P[far, 2] + AXLE_H_M)), 2) if far.sum() else None)


def main():
    out = sys.argv[1]; pose = json.loads(sys.argv[2]); windows = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    hw = Hardware(stereo=False); cal = AutoCal(hw.rs2_intr, log=lambda s: print(s, flush=True)); cloud = Rs2Cloud(hw.rs2_intr)
    T0, R0 = cloud.T.copy(), cloud.R.copy()
    gt.ctl("clear_obstacles"); gt.ctl("teleport", x=pose[0], y=pose[1], yaw=pose[2]); time.sleep(1.5)
    hist = []; plan = []
    for k in range(windows): plan += [(0.0, 3.0), (0.25 if k % 2 == 0 else -0.25, 1.2)]
    plan += [(0.0, 3.0)]
    last = None; depth = None
    for v, d in plan:
        t_end = time.monotonic() + d
        while time.monotonic() < t_end:
            t, _ = hw.wait_rs1(200); hw.drive(v, 0.0)
            vl, vr = hw.wheel_vels()
            seq, st, depth, *_ = hw.rs2_snapshot()
            hw.imu.grab() if hasattr(hw.imu, "grab") else None
            still = v == 0.0 and abs(vl) < STILL_V and abs(vr) < STILL_V
            if seq != last:
                last = seq; nw = cal.n_win
                cal.update(still, np.array(hw.imu.accel, float), depth if still else None)
                if cal.n_win != nw: hist.append(dict(cal.est))
    cal.update(False)            # close the last window
    if cal.n_win and (not hist or hist[-1] != cal.est): hist.append(dict(cal.est))
    hw.drive(0, 0); time.sleep(0.5); seq, st, depth, *_ = hw.rs2_snapshot()
    T1, R1 = cal.pose()
    R = dict(est=cal.est, hist=hist, nominal=quality(cloud, depth, T0, R0), autocal=quality(cloud, depth, T1, R1))
    json.dump(R, open(out, "w"), indent=1); print(json.dumps(R["est"]), flush=True); hw.close()


if __name__ == "__main__":
    main()
