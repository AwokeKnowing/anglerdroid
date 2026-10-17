"""Current-observation layer for the planners (the Nav2 costmap 'obstacle layer' idea: marking + clearing).

The SLAM map is memory.  Inside the area the RS1 top-down camera sees RIGHT NOW, the current frame is authoritative:
  observed obstacle -> occupied     (always blocks, even if the map says free)
  observed clear    -> free         (a drift-induced ghost wall in the map cannot block floor we can see)
  robot's own body  -> free         (the robot is standing there)
  not observed      -> map memory   (unchanged)
It also yields the 'known traversable' mask: observed clear | own body | the footprint swept over the last 4 m of
driving (and clears map memory there, unless observed occupied now) (Nav2 obstacle layer footprint_clearing_enabled: the robot's own deck hides a band right behind it from the
top-down camera, but it has just driven through that band).  Reversing is only planned into that space: the rear
is the least-observed side, so motion backwards must stay inside what the sensors currently see as free.

The layer is updated from nav's 30 Hz loop (pose = SLAM pose at the frame) and applied when the planning grids are
rebuilt, at the grid's resolution (obstacles dilated / clear eroded to the cell size: conservative both ways)."""
from __future__ import annotations
import math, threading
import numpy as np
import collections
import cv2
import nav  # noqa
from robot_config import RCX, RCY, EGO_PX_SIZE
from perception.labels import CLEAR
from twin_geometry import FOOTPRINT_SAFE

SWEPT_M = 4.0          # trail length whose footprint counts as known traversable (= nav.run trail, 80 x 5 cm: the
                       # retreat to the last rotation-feasible pose may reverse along all of it)

NONE, FREE, OBST, BODY = 0, 1, 2, 3


class ObsLayer:
    def __init__(self):
        self.lock = threading.Lock(); self.pose = None; self.src = None; self.t = None
        self.trail = collections.deque()                     # recent poses, >= 2 cm / 5 deg apart, total length <= SWEPT_M
        self.fp = np.asarray(FOOTPRINT_SAFE, float)

    def update(self, pose, lab, obs_mask, body, t=None):
        """pose: SLAM (x, y, th) at the frame; lab: ego labels; obs_mask: obstacle pixels (Ego.obs_mask); body: own hull."""
        s = np.zeros(lab.shape, np.uint8)
        s[lab == CLEAR] = FREE; s[body] = BODY; s[obs_mask] = OBST
        p = tuple(float(v) for v in pose)
        with self.lock:
            self.pose = p; self.src = s; self.t = t
            tr = self.trail
            if not tr or math.hypot(p[0] - tr[-1][0], p[1] - tr[-1][1]) > 0.02 or abs((p[2] - tr[-1][2] + math.pi) % (2 * math.pi) - math.pi) > 0.09:
                tr.append(p)
                L = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(list(tr)[:-1], list(tr)[1:]))
                while len(tr) > 2 and L > SWEPT_M:
                    a = tr.popleft(); L -= math.hypot(tr[0][0] - a[0], tr[0][1] - a[1])

    def _swept(self, org, res, shape, now=None):
        """footprint polygons at the recent trail poses (+ the current pose) -> bool grid."""
        out = np.zeros(shape, np.uint8)
        with self.lock: tr = list(self.trail) + ([tuple(now)] if now is not None else [])
        for x, y, th in tr:
            c, s = math.cos(th), math.sin(th)
            P = np.c_[x + c * self.fp[:, 0] - s * self.fp[:, 1], y + s * self.fp[:, 0] + c * self.fp[:, 1]]
            q = np.round((P - np.asarray(org)) / res - 0.5).astype(np.int32)
            cv2.fillPoly(out, [q], 1)
        return cv2.erode(out, np.ones((2, 2), np.uint8)).astype(bool)      # whole cells only

    def _warp(self, mask, org, res, shape):
        """ego-frame bool mask -> bool grid (shape) whose cell (r, c) centre is org + ((c, r) + 0.5) * res."""
        x, y, th = self.pose; c, s = math.cos(th), math.sin(th); k = res / EGO_PX_SIZE
        bx, by = org[0] + 0.5 * res - x, org[1] + 0.5 * res - y          # map offset of dst cell (0, 0) centre
        # ego col = RCX + u/px, row = RCY - v/px with u = c dx + s dy, v = -s dx + c dy, dx = bx + col*res, dy = by + row*res
        M = np.array([[c * k, s * k, RCX + (c * bx + s * by) / EGO_PX_SIZE],
                      [s * k, -c * k, RCY - (-s * bx + c * by) / EGO_PX_SIZE]], np.float64)
        out = cv2.warpAffine(mask.astype(np.uint8), M, (shape[1], shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out.astype(bool)

    def apply(self, occ, org, res):
        """occ: bool grid (rows = +y) with corner org (map m), cell res (m).  Returns (occ', traversable_observed)."""
        with self.lock: src, pose = self.src, self.pose
        if src is None: return occ, np.zeros_like(occ)
        k = max(1, int(math.ceil(res / EGO_PX_SIZE)))
        ker = np.ones((k + 1, k + 1), np.uint8)
        obst = cv2.dilate((src == OBST).astype(np.uint8), ker).astype(bool)         # any obstacle in the cell wins
        free = cv2.erode(((src == FREE) | (src == BODY)).astype(np.uint8), ker).astype(bool) & ~obst   # cell fully clear
        body = (src == BODY)
        Wo, Wf, Wb = (self._warp(m, org, res, occ.shape) for m in (obst, free, body))
        Ws = self._swept(org, res, occ.shape, now=pose)
        # Nav2 footprint clearing: map memory under the footprint swept over the trail (incl. the current pose) is
        # cleared -- Kevin cannot have driven through a wall; such cells are SLAM pose-error ghosts (seen: a counter
        # edge 10 cm from the axle after a fast approach -> every MPPI rollout 'dead' at t=0, "local planner found no
        # path", stuck for good).  Same rule as the safety envelope's map memory.  What the RS1 sees now still wins.
        return (occ & ~Wf & ~Wb & ~Ws) | Wo, (Wf | Wb | Ws) & ~Wo
