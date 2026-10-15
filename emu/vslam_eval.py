"""cuVSLAM (stereo IR, RS2) vs wheel+gyro odometry vs ground truth, in the emulator.
Drives a fixed loop (open-loop twists) and integrates both estimates from the same frames.
python -m emu.vslam_eval [--env apartment] [--imu]"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from emu import gt  # noqa
from emu.shm import read_info  # noqa


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--env", default="apartment"); ap.add_argument("--imu", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/.kevin/emu/vslam")); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import nav
    from nav.hw import Hardware
    from nav.slam import Odom, wrap
    from cuvslam_tracker import CuVSLAMTracker
    hw = Hardware(stereo=True)
    trk = CuVSLAMTracker(rs2_profile=hw.rs2.profile, imu_pipeline=hw.imu if a.imu else None)
    od = Odom(); st = read_info()["start"]
    # loop: forward / turn segments (v, w, secs)
    plan = [(0.0, 0.0, 2.0)] + [(0.35, 0.0, 3.0), (0.0, 0.9, 1.75), (0.3, 0.3, 3.0), (0.0, -0.9, 1.75), (0.35, 0.0, 2.0), (0.0, 1.2, 2.6)] * 2
    rows = []; last_seq = -1; vo0 = None; t_end = 0
    seg_t = time.monotonic(); si = 0
    while si < len(plan):
        v, w, d = plan[si]
        if time.monotonic() - seg_t > d: si += 1; seg_t = time.monotonic(); continue
        t, verts = hw.wait_rs1(200)
        if t is None: continue
        hw.drive(v, w)
        vl, vr = hw.wheel_vels(); od.update(t, vl, vr, hw.yaw_rate(), commanded_zero=(v == 0 and w == 0))
        seq, stamp, depth, color, il, ir = hw.rs2_snapshot()
        if seq != last_seq and il is not None:
            last_seq = seq; t0 = time.perf_counter(); r = trk.track(il, ir, int(stamp * 1e9)); ms = (time.perf_counter() - t0) * 1e3
            s = gt.state(); g = gt.to_map((s["x"], s["y"], s["yaw"]), st)
            rows.append(dict(t=t, gt=g, odom=od.pose, vo=(trk.x, trk.y, trk.theta), ok=r is not None, ms=ms))
    hw.drive(0, 0); hw.close()
    G = np.array([r["gt"] for r in rows]); O = np.array([r["odom"] for r in rows]); V = np.array([r["vo"] for r in rows])
    V = V - V[0]                                   # VO frame: x fwd, y left from its start (= robot start up to the camera offset)
    dist = float(np.sum(np.hypot(*np.diff(G[:, :2], axis=0).T)))
    def err(E): return dict(final=round(float(np.hypot(*(E[-1, :2] - G[-1, :2]))), 3), max=round(float(np.hypot(*(E[:, :2] - G[:, :2]).T).max()), 3),
                            yaw_final_deg=round(math.degrees(abs(wrap(E[-1, 2] - G[-1, 2]))), 2))
    R = dict(env=a.env, imu=a.imu, frames=len(rows), tracked=int(sum(r["ok"] for r in rows)), dist_m=round(dist, 2),
             vo_ms_mean=round(float(np.mean([r["ms"] for r in rows])), 2), odom=err(O), cuvslam=err(V))
    print(json.dumps(R)); json.dump(dict(summary=R, rows=rows), open(os.path.join(a.out, "vslam_%s%s.json" % (a.env, "_imu" if a.imu else "")), "w"), default=float)


if __name__ == "__main__":
    main()
