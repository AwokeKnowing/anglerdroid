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



def test_leave_inflated_band_never_closer():
    """Robot parked 8 cm from a chair leg (deep inside the 24/20 cm inflation): it must get a path out, and the path must
    never come closer to the leg than the robot already is."""
    g = Grid(0.02); g.wall(5.0, 5.0, 5.04, 5.04)                 # 4 cm leg
    start = (5.02, 5.0 - 0.08)
    r = GP.plan(g, start, (5.02, 3.0))
    assert r is not None
    d = np.hypot(r["xy"][:, 0] - 5.02, r["xy"][:, 1] - 5.02)
    assert d.min() >= np.hypot(start[0] - 5.02, start[1] - 5.02) - 0.05, d.min()      # within one coarse cell


def _narrow_vs_open(close_detour):
    """Barrier x 4.5..7.0, y -1.0..1.0 with a 0.60 m slot along y=0 (axle clearance 0.30 < R_ROTATE: no room to spin,
    but drivable).  Around the barrier's top end (y > 1.0) is open.  Start (3.5, 0), goal (8, 0).
    slot: 4.5 m long, 2.5 m of it not rotatable;  detour: ~6.3 m, all rotatable."""
    g = Grid(0.02)
    g.wall(4.5, 0.30, 7.0, 1.0); g.wall(4.5, -1.0, 7.0, -0.30)
    if close_detour: g.wall(4.5, 1.0, 7.0, 4.0); g.wall(4.5, -4.0, 7.0, -1.0)
    return GP.plan(g, (3.5, 0.0), (8.0, 0.0))


def test_prefers_room_to_turn():
    """Turn-aware edge cost (length x (1 + not_rotatable)): the 2.5 m slot costs ~5 m (it may have to be driven out
    again), so the 6.3 m open detour wins; plain shortest path would take the slot (4.5 m)."""
    r = _narrow_vs_open(False); assert r is not None
    assert r["xy"][:, 1].max() > 1.0, "took the narrow slot although the open detour is cheaper in worst-case length"


def test_narrow_still_allowed_when_only_way():
    r = _narrow_vs_open(True); assert r is not None and r["snap_m"] < 0.05
    assert np.abs(r["xy"][:, 1]).max() < 0.3, "the slot is the only way; it must still be used"


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
