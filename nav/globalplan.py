"""Global coarse path on the full SLAM map (~5 cm grid, Dijkstra over free cells, C via scipy.csgraph).
The 1 cm local window + MPPI tracks a subgoal taken from this path, so goals anywhere in the map are reachable.
Unknown space counts as free (optimistic exploration); walls are inflated by the robot half-width."""
from __future__ import annotations
import math, time
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

RES = 0.05                     # target coarse cell size; the real one is an integer multiple of the SLAM grid res


def coarse_res(grid):
    """Actual coarse cell size: f whole SLAM cells per side. Everything that maps metres <-> coarse cells must use this,
    never RES itself (2 cm grid -> f=2 -> 4 cm cells; using 5 cm there stretched the map 1.25x about the grid origin)."""
    return max(1, int(round(RES / grid.res))) * grid.res


def coarse_blocked(grid, inflate=0.20):
    cres = coarse_res(grid); f = int(round(cres / grid.res)); n = grid.n // f
    L = grid.L[:n * f, :n * f].reshape(n, f, n, f)
    occ = ((L > 1.5).sum((1, 3)) >= 1).astype(np.uint8)        # confident hits only (single noisy hits don't close doors)
    r = max(1, int(round(inflate / cres)))                       # nearest whole cell (0.24 -> 6x4 cm, 0.21 -> 5x4 cm)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(occ, k).astype(bool), occ.astype(bool)


def plan(grid, start, goal):
    """Try the full robot half-width first, then a tighter inflation (doors/narrow passes; MPPI does the exact footprint)."""
    for infl in (0.24, 0.21):                     # robot half-width is 0.207 (wheels); 0.12 routed through 0.3 m gaps (MPPI then stuck)
        r = _plan(grid, start, goal, infl)
        if r is not None: r["inflate"] = infl; return r
    return None


def _plan(grid, start, goal, inflate):
    """-> dict(xy=(N,2) map-frame path, goal=reachable goal, snap_m=distance requested->reachable, ms) or None."""
    t = time.perf_counter()
    blk, occ = coarse_blocked(grid, inflate); n = blk.shape[0]; o = grid.origin; cres = coarse_res(grid)
    def cell(p): return (min(max(int((p[1] - o[1]) / cres), 0), n - 1), min(max(int((p[0] - o[0]) / cres), 0), n - 1))
    s, g = cell(start), cell(goal)
    free = ~blk; free[s] = True                       # robot may sit in an inflated band; let it leave
    if not free[g]:                                   # nearest free cell to the requested goal
        dist = cv2.distanceTransform(blk.astype(np.uint8), cv2.DIST_L2, 3)
        ys, xs = np.nonzero(free)
        if not len(ys): return None
        j = np.argmin((ys - g[0]) ** 2 + (xs - g[1]) ** 2); g = (int(ys[j]), int(xs[j]))
    # sub-window bounding box (start/goal + 3 m margin) keeps the graph small
    m = int(4.0 / cres)
    r0, r1 = max(min(s[0], g[0]) - m, 0), min(max(s[0], g[0]) + m + 1, n)
    c0, c1 = max(min(s[1], g[1]) - m, 0), min(max(s[1], g[1]) + m + 1, n)
    F = free[r0:r1, c0:c1]; H, W = F.shape; idx = np.arange(H * W).reshape(H, W)
    rows, cols, wts = [], [], []
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4142), (1, -1, 1.4142)):
        if dx >= 0:
            A = idx[0:H - dy, 0:W - dx]; B = idx[dy:H, dx:W]; ok = F[0:H - dy, 0:W - dx] & F[dy:H, dx:W]
        else:
            A = idx[0:H - dy, 1:W]; B = idx[dy:H, 0:W - 1]; ok = F[0:H - dy, 1:W] & F[dy:H, 0:W - 1]
        rows.append(A[ok]); cols.append(B[ok]); wts.append(np.full(ok.sum(), w))
    rows, cols, wts = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    G = csr_matrix((wts, (rows, cols)), shape=(H * W, H * W))
    si = idx[s[0] - r0, s[1] - c0]; gi = idx[g[0] - r0, g[1] - c0]
    D, P = dijkstra(G, directed=False, indices=si, return_predecessors=True)
    if not np.isfinite(D[gi]):
        # goal (or its nearest free cell) is in a pocket not connected to us -- e.g. the unseen inside of a couch:
        # snap to the closest cell we can actually reach instead of failing (old: kept chasing the stale goal)
        ok = np.isfinite(D)
        if ok.sum() <= 1: return None
        yy, xx = np.divmod(np.nonzero(ok)[0], W); gc = cell(goal)
        j = np.argmin((yy + r0 - gc[0]) ** 2 + (xx + c0 - gc[1]) ** 2); gi = int(np.nonzero(ok)[0][j])
    path = [gi]
    while path[-1] != si: path.append(P[path[-1]])
    path = np.array(path[::-1]); yy, xx = np.divmod(path, W)
    xy = np.stack([o[0] + (xx + c0 + 0.5) * cres, o[1] + (yy + r0 + 0.5) * cres], 1)
    gq = xy[-1]
    if blk[cell(goal)] == 0 and math.hypot(gq[0] - goal[0], gq[1] - goal[1]) < 2 * cres: xy[-1] = goal; gq = np.asarray(goal, float)
    return dict(xy=xy, goal=gq, snap_m=float(math.hypot(gq[0] - goal[0], gq[1] - goal[1])), ms=(time.perf_counter() - t) * 1e3)


def subgoal(path_xy, robot, ahead=4.0):
    """Point on the global path ~ahead m (arclength) past the closest point to the robot."""
    d = np.hypot(path_xy[:, 0] - robot[0], path_xy[:, 1] - robot[1]); i = int(np.argmin(d))
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(path_xy[i:], axis=0).T))]
    j = i + int(np.searchsorted(seg, ahead)); j = min(j, len(path_xy) - 1)
    return path_xy[j], j == len(path_xy) - 1
