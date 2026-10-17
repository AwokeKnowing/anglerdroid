"""Global planner geometry: the coarse grid must line up with the SLAM grid in metres, anywhere in the map.
Regression: a 2 cm SLAM grid gave f=round(0.05/0.02)=2 (4 cm coarse cells) while metres<->cells used 5 cm, so the
coarse map was stretched 1.25x about the grid origin (-20,-20): ~5 m off in the house, phantom walls round the robot,
'unreachable: no path to <place>' for every goal.   Run: python tests/test_globalplan.py  (or pytest)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nav import globalplan as GP


class Grid:
    def __init__(self, res=0.02, size_m=40.0, origin=(-20.0, -20.0)):
        self.res = res; self.n = int(round(size_m / res)); self.origin = np.array(origin, float)
        self.L = np.zeros((self.n, self.n), np.float32); self.seen = np.zeros((self.n, self.n), bool)

    def wall(self, x0, y0, x1, y1):            # axis-aligned occupied box in metres
        c = lambda v, o: int((v - o) / self.res)
        self.L[c(y0, self.origin[1]):c(y1, self.origin[1]) + 1, c(x0, self.origin[0]):c(x1, self.origin[0]) + 1] = 4.0


def corridor(res):
    """1.2 m wide corridor along x at y in [-0.6, 0.6], from x=3 to x=10 (far from the grid origin)."""
    g = Grid(res)
    g.wall(3, 0.6, 10, 0.7); g.wall(3, -0.7, 10, -0.6)
    return g


def test_coarse_res_is_whole_cells():
    for res in (0.01, 0.02, 0.025, 0.04, 0.05, 0.1):
        cr = GP.coarse_res(Grid(res, size_m=4.0))
        assert abs(cr / res - round(cr / res)) < 1e-9 and 0.02 <= cr <= 0.1


def test_corridor_far_from_origin():
    for res in (0.01, 0.02, 0.05):
        g = corridor(res)
        r = GP.plan(g, (4.0, 0.0), (9.0, 0.0))
        assert r is not None, res
        assert r["snap_m"] < 0.05, (res, r["snap_m"])
        P = r["xy"]
        assert np.all(np.abs(P[:, 1]) < 0.6 - 0.2), (res, P[:, 1].min(), P[:, 1].max())   # stays inside, clear of walls
        ij = np.floor((P - g.origin) / g.res).astype(int)
        assert g.L[ij[:, 1], ij[:, 0]].max() <= 0, res


def test_wall_blocks_where_it_really_is():
    g = Grid(0.02); g.wall(5.0, -3, 5.1, 3)                    # wall across x=5..5.1, y in [-3,3]
    r = GP.plan(g, (4.0, 0.0), (6.0, 0.0))
    assert r is not None
    assert np.any(np.abs(r["xy"][:, 1]) > 3.0)                 # goes round the end of the wall, not through it


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
