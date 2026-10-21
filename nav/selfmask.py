"""Self-view mask for the RS1 ego labels: ego pixels that read OBSTACLE because the camera sees the robot itself
(wheels, bumper corners).  Calibrate once per robot build:

  sim : python -m nav.selfmask --sim     (uses emulator ground truth; robot spins in place in free space)
  real: python -m nav.selfmask           (robot in >=0.6 m free space; pixels that are OBSTACLE in >=50% of
                                          frames while spinning are self — the world moves, the robot doesn't)
Writes nav/data/self_mask[_sim].npy (bool, FRAME_H x FRAME_W); Ego() ORs it into its body mask."""
from __future__ import annotations
import argparse, os, time
import numpy as np
import cv2
import nav  # noqa
from nav.hw import Hardware
from nav.perceive import Ego
from perception.labels import OBSTACLE

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sim", action="store_true"); ap.add_argument("--frames", type=int, default=150)
    a = ap.parse_args()
    hw = Hardware(); ego = Ego(); ego.self_mask[:] = False; ego.body[:] = False
    cnt = np.zeros(ego.lab.shape, np.int32); n = 0
    try:
        for i in range(a.frames + 30):
            t, v = hw.wait_rs1()
            if v is None: continue
            hw.drive(0.0, 0.8)
            if i < 30: continue
            ego.update(v); cnt += ego.lab == OBSTACLE; n += 1
    finally:
        hw.drive(0.0, 0.0); hw.close()
    m = cnt >= 0.5 * n                                  # rotating: real obstacles sweep across pixels, self stays put
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, "self_mask_sim.npy" if a.sim else "self_mask.npy"); np.save(p, m)
    ys, xs = np.nonzero(m)
    print("self mask: %d px -> %s ; robot-frame extent x[%.2f,%.2f] y[%.2f,%.2f]" % (m.sum(), p, ego.X[m].min() if m.any() else 0,
          ego.X[m].max() if m.any() else 0, ego.Y[m].min() if m.any() else 0, ego.Y[m].max() if m.any() else 0))


if __name__ == "__main__":
    main()
