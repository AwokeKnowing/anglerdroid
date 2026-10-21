"""Free-space carving (nav.slam.carve_rays + GridMap.integrate carve): a removed obstacle clears once seen through,
a wall that ends the rays persists, cells out of view are kept, a table top is kept when rays pass only below it."""
import sys, os, math
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "src")]
import numpy as np
from nav.slam import GridMap, carve_rays

O = np.array([0.0, 0.0, 0.47], np.float32)          # RS2-like lens 47 cm above floor, robot at origin


def fan(x_end, h_end, ys=np.linspace(-0.6, 0.6, 61), floor=None):
    E = np.array([[x_end, y, h_end] for y in ys], np.float32)
    return (O, E, np.full(len(E), h_end < 0.02 if floor is None else floor))


def occ(g, x, y): r, c = g.rc(np.array([[x, y]])); return float(g.L[r[0], c[0]])


def box_points(x0, x1, y0, y1, h0, h1, d=0.02):
    P = [(x, y, h) for x in np.arange(x0, x1, d) for y in np.arange(y0, y1, d) for h in np.arange(h0, h1, 0.05)]
    P = np.array(P, np.float32); return P[:, :2], P[:, 2]


def test_removed_box_clears_wall_and_unseen_persist():
    g = GridMap(size_m=10, origin=(-5, -5)); pose = (0, 0, 0)
    bx, bh = box_points(1.5, 1.7, -0.1, 0.1, 0.03, 0.30)          # 30 cm box at 1.5 m
    wx, wh = box_points(3.0, 3.04, -1.0, 1.0, 0.03, 0.95)          # wall at 3 m
    ux, uh = box_points(1.5, 1.7, 2.0, 2.2, 0.03, 0.30)            # box out of view (y = 2 m)
    for _ in range(6): g.integrate(pose, np.r_[bx, wx, ux], np.zeros((0, 2)), np.r_[bh, wh, uh])
    assert occ(g, 1.6, 0) > 4 and occ(g, 3.02, 0) > 4 and occ(g, 1.6, 2.1) > 4
    # box removed: rays now reach the floor up to the wall and the wall itself (rows of a depth image)
    rays = [fan(x, 0.0) for x in np.arange(0.8, 3.0, 0.03)] + [fan(3.0, h, floor=False) for h in np.arange(0.05, 0.9, 0.04)]
    C = carve_rays(rays)
    for i in range(4):
        g.integrate(pose, wx, np.zeros((0, 2)), wh, C)              # current frame still sees the wall
    assert occ(g, 1.6, 0) < 0.7, occ(g, 1.6, 0)                     # removed box cleared
    assert occ(g, 3.02, 0) > 4, occ(g, 3.02, 0)                     # wall kept
    assert occ(g, 1.6, 2.1) > 4                                     # out of view kept


def test_table_top_kept_when_rays_pass_below():
    g = GridMap(size_m=10, origin=(-5, -5)); pose = (0, 0, 0)
    tx, th = box_points(1.5, 1.9, -0.3, 0.3, 0.70, 0.75)            # table top 70-75 cm (seen earlier)
    for _ in range(6): g.integrate(pose, tx, np.zeros((0, 2)), th)
    C = carve_rays([fan(x, 0.0) for x in np.arange(0.8, 3.0, 0.03)])    # floor rays pass under it (< 47 cm)
    for _ in range(4): g.integrate(pose, np.zeros((0, 2)), np.zeros((0, 2)), None, C)
    assert occ(g, 1.7, 0) > 4, occ(g, 1.7, 0)


def test_depth_noise_margin_keeps_wall_cells():
    g = GridMap(size_m=10, origin=(-5, -5)); pose = (0, 0, 0)
    wx, wh = box_points(3.0, 3.02, -1.0, 1.0, 0.03, 0.95)
    for _ in range(6): g.integrate(pose, wx, np.zeros((0, 2)), wh)
    rng = np.random.default_rng(0)            # returns scattered behind the wall by D435 noise (sigma 0.004 z^2 ~ 4 cm)
    for _ in range(10):
        rays = [fan(3.0 + rng.normal(0, 0.036), h, floor=False) for h in np.arange(0.05, 0.9, 0.04)]
        g.integrate(pose, np.zeros((0, 2)), np.zeros((0, 2)), None, carve_rays(rays))
    assert occ(g, 3.01, 0) > 0.7, occ(g, 3.01, 0)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
