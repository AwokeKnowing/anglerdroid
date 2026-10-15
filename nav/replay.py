"""Replay a KEVIN_RECORD log through odometry + slam2 (RS2 path) offline: python -m nav.replay <logdir>
Prints recorded vs replayed final pose; same code path as nav.run (wheel+gyro odom, Rs2Cloud, Slam2.submit)."""
import glob, json, math, sys, time
import numpy as np
import nav  # noqa
from nav.slam import GridMap
from nav.slam2 import Slam2
from nav.perceive import Rs2Cloud


def main():
    fs = sorted(glob.glob(sys.argv[1].rstrip("/") + "/t*.npz")); assert fs, "no ticks"
    m0 = json.loads(str(np.load(fs[0])["meta"])); d0 = np.load(fs[0])
    intr = m0["intr"]
    slam = Slam2(GridMap()); cloud = Rs2Cloud(intr); n = 0; t_last = None
    for i, f in enumerate(fs):
        z = np.load(f); m = json.loads(str(z["meta"]))
        slam.odom.update(m["t"], m["vl"], m["vr"], m.get("gz"), commanded_zero=(m["v"] == 0 and m["w"] == 0))
        if i % 6 == 0 and "rs2" in z.files:
            obs, flo = cloud.points(z["rs2"]); slam.submit(m["t"], obs.astype(np.float32), flo.astype(np.float32)); n += 1; time.sleep(0.03)
        last = m
    time.sleep(0.5)
    rp, pp = last.get("pose"), slam.pose()
    print(json.dumps(dict(ticks=len(fs), scans=n, recorded_final=[round(v, 3) for v in rp] if rp else None, replay_final=[round(v, 3) for v in pp],
                          diff_xy_m=round(math.hypot(pp[0] - rp[0], pp[1] - rp[1]), 3) if rp else None, slam=slam.stats)))


if __name__ == "__main__":
    main()
