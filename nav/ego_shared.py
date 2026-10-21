"""Brain-side view of the safety core's RS1 ego labels (sc_ego, nav.bus): one Ego per RS1 frame, computed by the core
for its own decision, instead of a second identical Ego.update in the brain (Option A, 2026-10-20).

EgoShared IS an nav.perceive.Ego (same grids, body/self masks, clear_xy, rays, valid_fraction); only update() and
obstacle_xy() change: the labels, height, map labels, obstacle mask and occlusion flag are the core's.  The core's
obstacle_xy() already applied the stateful filters (grass, near-ring persistence, overhang) once; re-running them here
would advance their state a second time per frame, so obstacle_xy() here only reads the published mask.

Staleness: labels older than the core's own AGE_MAX_S at read time (capture -> now) are stale and are not returned;
wait() keeps waiting for a fresh frame until its timeout and then returns None (the brain's no-frame path: stop)."""
from __future__ import annotations
import time
import numpy as np
from nav import bus
from nav.perceive import Ego
from perception.labels import OBSTACLE
from nav.safety_core import AGE_MAX_S


class EgoShared(Ego):
    def __init__(self, blk):
        super().__init__()
        assert self.lab.shape == (bus.EGO_H, bus.EGO_W), "sc_ego layout != Ego label grid"
        self.blk = blk; self._seq = blk.seq; self._out = {}; self.frame = 0
        self.n_stale = self.n_gap = self.n_torn = 0; self.age = None; self.lab_map = None
        self.obs_mask = np.zeros(self.lab.shape, bool)

    def wait(self, timeout):
        """next fresh core frame -> (stamp_s, frame) or (None, None) after timeout (none, all stale, or writer stuck)."""
        t_end = time.monotonic() + timeout
        while True:
            s = self.blk.seq
            if s != self._seq and not (s & 1):
                sq, r = self.blk.try_read(out=self._out, budget_s=0.002)
                if sq is None: self.n_torn += 1; self._out = {}            # core paused mid-write: not a frame
                else:
                    self._seq = sq; age = time.monotonic() - float(r["t_cap"][0])
                    if age <= AGE_MAX_S: return self._take(r, age)
                    self.n_stale += 1                                      # stale: never used, wait for the next
            if time.monotonic() > t_end: return None, None
            time.sleep(0.0005)

    def _take(self, r, age):
        fr = int(r["frame"][0])
        if self.frame and fr > self.frame + 1: self.n_gap += fr - self.frame - 1
        self.frame = fr; self.age = age
        np.copyto(self.lab, r["lab"]); np.copyto(self.hgt, r["hgt"])
        self.lab_map = r["lab_map"].copy() if int(r["has_map"][0]) else None
        self.obs_mask = r["obs_mask"].astype(bool); self.occluded = bool(r["occluded"][0])
        return float(r["stamp"][0]), fr

    def update(self, verts=None):
        return self.lab                                                   # already the core's labels for this frame

    def obstacle_xy(self, max_n=None):
        m = self.obs_mask
        mm = m if self.lab_map is None else (m & (self.lab_map == OBSTACLE))
        self.map_obs = np.stack([self.X[mm], self.Y[mm]], 1)
        self.map_obs_h = self.hgt[mm].astype(np.float32) * 0.01
        P = np.stack([self.X[m], self.Y[m]], 1)
        if max_n and len(P) > max_n:
            P = P[np.random.default_rng(0).choice(len(P), max_n, replace=False)]
        return P

