"""Global path + MPPI tracking on the live map, using kplan (nav/kplan: fastpath A*+pull+band, CUDA MPPI,
CUDA-graph replay).  The planning world is a 1 cm window (default 12 m) of the SLAM map, refreshed in place
(so the captured CUDA graph stays valid) by a background thread; the 30 Hz loop only replays the graph."""
from __future__ import annotations
import math, os, sys, threading, time
import numpy as np
import cv2

_K = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kplan")
if _K not in sys.path: sys.path.insert(0, _K)
os.environ.setdefault("KPLAN_GLOBAL", "fast")
import torch  # noqa: E402
import planner as KP  # noqa: E402
import fastpath as FP  # noqa: E402
from core import Limits  # noqa: E402


class Planner:
    def __init__(self, grid, win_m=12.0, dev="cuda"):
        self.g = grid; self.N = int(win_m * 100); self.win = win_m
        self.org = np.array([-win_m / 2, -win_m / 2])                      # map coords of window corner
        occ = np.zeros((self.N, self.N), bool); occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        self.W = KP.World(occ, dev=dev)
        self.m = FP.MapCPU(self.W)
        self.NG = FP.FastNewGoal(self.W, 1, m=self.m)
        self.M = self.NG.M; self.G = self.NG.G
        self.lock = threading.Lock(); self.goal = None; self.path = None; self.status = "idle"
        self.timing = dict(refresh_ms=0.0, newgoal_ms=0.0, plan_ms=0.0); self._map_ver = -1
        self.need_replan = False

    # ------------------------------------------------------------ map -> planning window
    def _occ_window(self):
        g = self.g; s = int(round(g.res / 0.01))
        c0 = np.floor((self.org - g.origin) / g.res).astype(int); n5 = self.N // s
        occ5 = np.zeros((n5, n5), np.uint8)
        r0, r1 = max(c0[1], 0), min(c0[1] + n5, g.n); q0, q1 = max(c0[0], 0), min(c0[0] + n5, g.n)
        if r1 > r0 and q1 > q0:
            occ5[r0 - c0[1]:r1 - c0[1], q0 - c0[0]:q1 - c0[0]] = g.occupied()[r0:r1, q0:q1]
        occ = cv2.resize(occ5, (self.N, self.N), interpolation=cv2.INTER_NEAREST).astype(bool)
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        return occ

    def refresh(self, robot_xy):
        """Recentre if needed, rebuild EDT/coarse grids in place + CPU A* grids. ~10-40 ms, off the control loop."""
        t = time.perf_counter()
        rel = np.asarray(robot_xy) - self.org
        if (rel < 3.0).any() or (rel > self.win - 3.0).any():
            self.org = np.round(np.asarray(robot_xy) - self.win / 2, 2); self.need_replan = True
        occ = self._occ_window()
        Wn = KP.World(occ, dev=self.W.dev)
        with self.lock:
            for k in ("edt", "cmax", "cmean", "cblocked", "ccost"):
                getattr(self.W, k).copy_(getattr(Wn, k))
            m = FP.MapCPU(self.W)
            self.m = m; self.NG.m = m
            if self.M.ccost_np is not None:
                self.M.ccost_np = torch.where(self.W.cblocked, torch.full_like(self.W.ccost.float(), 100.0), self.W.ccost.float()).cpu().numpy()
        self.timing["refresh_ms"] = (time.perf_counter() - t) * 1e3
        return self.timing["refresh_ms"]

    def to_win(self, xy): return np.asarray(xy, float) - self.org

    def clearance(self, xy):
        p = self.to_win(xy); ix, iy = int(p[0] * 100), int(p[1] * 100)
        if not (0 <= ix < self.N and 0 <= iy < self.N): return 0.0
        return float(self.m.edt[iy, ix])

    def snap_goal(self, xy, need=0.26, search=1.2):
        """Nearest point to xy (within search m) with planning clearance (James asks for 'the couch' -> stand next to it)."""
        if self.clearance(xy) >= need: return np.asarray(xy, float)
        best = None
        for r in np.arange(0.05, search + 1e-9, 0.05):
            for a in np.linspace(0, 2 * math.pi, max(8, int(2 * math.pi * r / 0.05)), endpoint=False):
                q = np.asarray(xy) + r * np.array([math.cos(a), math.sin(a)])
                if self.clearance(q) >= need: return q
        return best

    # ------------------------------------------------------------ goals
    def set_goal(self, state, goal_xy, heading=None):
        """state = (x, y, th, v, w) map frame. Returns (ok, message). New-goal latency measured into timing."""
        t = time.perf_counter()
        gq = self.snap_goal(goal_xy)
        if gq is None:
            self.status = "unreachable"; return False, "goal %.2f,%.2f is inside an obstacle" % tuple(goal_xy)
        with self.lock:
            st = np.array([*self.to_win(state[:2]), state[2], state[3], state[4]], np.float32)
            g = np.array([*self.to_win(gq), heading if heading is not None else 0.0])
            p = FP.plan_path(self.m, st[:2].astype(float), g[:2], eps=2.0)
            if p is None:
                self.status = "no path"; self.goal = None; return False, "no path to %.2f,%.2f" % tuple(gq)
            self.M.set_paths(g[None], [p], keep_progress=True); self.M.sprog.zero_()
            self.goal = np.asarray(gq); self.heading = heading; self.path = p["xy"] + self.org; self.status = "driving"
        self.timing["newgoal_ms"] = (time.perf_counter() - t) * 1e3
        return True, "ok"

    def replan(self, state):
        if self.goal is not None: return self.set_goal(state, self.goal, self.heading)

    def cancel(self):
        self.goal = None; self.path = None; self.status = "idle"

    def step(self, state):
        """One MPPI plan (CUDA graph). Returns (a, alpha) or None if no goal."""
        if self.goal is None: return None
        t = time.perf_counter()
        with self.lock:
            st = torch.as_tensor(np.array([[*self.to_win(state[:2]), state[2], state[3], state[4]]], np.float32), device=self.M.dev)
            u, info = self.G.plan(st)
            a, al = u[0].tolist()
        self.timing["plan_ms"] = (time.perf_counter() - t) * 1e3
        return a, al


LIMITS = Limits
