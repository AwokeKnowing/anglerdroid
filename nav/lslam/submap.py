"""A rigid local occupancy submap (nav.slam GridMap, 2 cm) in its own frame, anchored at a pose-chain node."""
import numpy as np
from nav.slam import GridMap

SIZE_M = 16.0


class Submap:
    def __init__(self, sid, anchor):
        self.id = sid; self.anchor = np.asarray(anchor, float); self.n = 0; self.dirty = True
        self.g = GridMap(SIZE_M, 0.02, (-SIZE_M / 2, -SIZE_M / 2))

    def integrate(self, pose_sub, pts):
        self.g.integrate(tuple(pose_sub), pts, np.zeros((0, 2), np.float32)); self.n += 1; self.dirty = True

    def rebuild(self, entries):
        """Exact rebuild from journal entries (integration is order-preserving and deterministic)."""
        self.g = GridMap(SIZE_M, 0.02, (-SIZE_M / 2, -SIZE_M / 2)); self.n = 0
        for e in entries: self.integrate(e.pose_sub, e.pts)

    def lf(self):
        if self.dirty: self.g.refresh_likelihood(); self.dirty = False
        return self.g
