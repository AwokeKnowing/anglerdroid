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


def coarse_blocked(grid, inflate=0.20, obs=None):
    """obs: nav.obslayer.ObsLayer -- the current observation overrides the map where the RS1 sees right now."""
    cres = coarse_res(grid); f = int(round(cres / grid.res)); n = grid.n // f
    fine = grid.L[:n * f, :n * f] > 1.5                          # confident hits only (single noisy hits don't close doors)
    if obs is not None and obs.pose is not None:                 # layer only the few-metre patch the frame covers
        x, y = obs.pose[:2]; r = 3.0; o = grid.origin
        a0, a1 = max(int((x - r - o[0]) / grid.res), 0), min(int((x + r - o[0]) / grid.res), n * f)
        b0, b1 = max(int((y - r - o[1]) / grid.res), 0), min(int((y + r - o[1]) / grid.res), n * f)
        if a1 > a0 and b1 > b0:
            fine = fine.copy(); fine[b0:b1, a0:a1] = obs.apply(fine[b0:b1, a0:a1], o + np.array([a0, b0]) * grid.res, grid.res)[0]
    occ = fine.reshape(n, f, n, f).any((1, 3)).astype(np.uint8)
    r = max(1, int(round(inflate / cres)))                       # nearest whole cell (0.24 -> 6x4 cm, 0.21 -> 5x4 cm)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(occ, k).astype(bool), occ.astype(bool)


def plan(grid, start, goal, obs=None):
    """Try the full robot half-width first, then a tighter inflation (doors/narrow passes; MPPI does the exact footprint)."""
    from twin_geometry import R_PASS, HALF_W
    for infl in (R_PASS, HALF_W):                 # half-width + envelope margin, then the bare half-width (0.12 routed through 0.3 m gaps)
        r = _plan(grid, start, goal, infl, obs)
        if r is not None: r["inflate"] = infl; return r
    return None


def _plan(grid, start, goal, inflate, obs=None):
    """-> dict(xy=(N,2) map-frame path, goal=reachable goal, snap_m=distance requested->reachable, ms) or None."""
    t = time.perf_counter()
    blk, occ = coarse_blocked(grid, inflate, obs); n = blk.shape[0]; o = grid.origin; cres = coarse_res(grid)
    def cell(p): return (min(max(int((p[1] - o[1]) / cres), 0), n - 1), min(max(int((p[0] - o[0]) / cres), 0), n - 1))
    s, g = cell(start), cell(goal)
    free = ~blk; free[s] = True
    if blk[s] and not occ[s]:
        # the robot sits inside an inflated band (e.g. 16 cm from a chair leg with 24 cm inflation): let it leave, but only
        # through band cells at least as far from every obstacle as it is now -- it may never be routed closer
        clear = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5)
        ok = (blk & (clear >= clear[s] - 1e-3)).astype(np.uint8)
        _, lab = cv2.connectedComponents(ok, connectivity=8); free |= lab == lab[s]
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
    # turn-aware edge cost.  A cell where the robot cannot spin in place (axle clearance < R_ROTATE) is a stretch it can
    # only leave forwards or by backing out the way it came; if it turns out to be blocked, it is driven twice (in and
    # back out).  So its worst-case length is 2x: edges cost length * (1 + not_rotatable), the mean of their two ends.
    # Open space is preferred whenever the detour is shorter than the narrow stretch it avoids, and a narrow path is
    # still taken when it is the only one (its trail is then the known reversible exit, see nav.run._retreat).
    from twin_geometry import R_ROTATE
    clr = cv2.distanceTransform((~occ[r0:r1, c0:c1]).astype(np.uint8), cv2.DIST_L2, 5) * cres
    mult = 1.0 + (clr < R_ROTATE).astype(float)
    rows, cols, wts = [], [], []
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4142), (1, -1, 1.4142)):
        if dx >= 0:
            A = idx[0:H - dy, 0:W - dx]; B = idx[dy:H, dx:W]; ok = F[0:H - dy, 0:W - dx] & F[dy:H, dx:W]
        else:
            A = idx[0:H - dy, 1:W]; B = idx[dy:H, 0:W - 1]; ok = F[0:H - dy, 1:W] & F[dy:H, 0:W - 1]
        mA = mult[0:H - dy, 0:W - dx] if dx >= 0 else mult[0:H - dy, 1:W]
        mB = mult[dy:H, dx:W] if dx >= 0 else mult[dy:H, 0:W - 1]
        rows.append(A[ok]); cols.append(B[ok]); wts.append(w * 0.5 * (mA[ok] + mB[ok]))
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
