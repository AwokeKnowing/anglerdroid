"""Append-only scan journal.  The submap grids are a cache: any submap can be rebuilt exactly from its entries."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Entry:
    seq: int; t: float; submap: int; pose_sub: np.ndarray; pts: np.ndarray      # pose in the submap frame, robot-frame points


class Journal:
    def __init__(self): self.e = []

    def append(self, t, submap, pose_sub, pts):
        self.e.append(Entry(len(self.e), t, submap, np.asarray(pose_sub, float), np.asarray(pts, np.float32))); return self.e[-1].seq

    def __len__(self): return len(self.e)
    def of(self, submap): return [x for x in self.e if x.submap == submap]
    def truncate(self, n):
        """Drop every entry with seq >= n (rollback); returns the dropped entries."""
        dropped = self.e[n:]; del self.e[n:]; return dropped
