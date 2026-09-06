"""MPPI policy wrapper for sim — same NumPy planner as live Orin.

Safety scales are applied as an *action mask* before Robot.step also clips
(belt-and-suspenders; matches insect contract).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Live planner lives under src/
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mppi_costmap import MppiCostmapPlanner  # noqa: E402
from sim.policy import Policy


class MppiSimPolicy(Policy):
    """Async-goal MPPI using ego obs; never asks v>0 when fwd_scale==0."""

    def __init__(self, goal_xy=None, wander=True):
        self.planner = MppiCostmapPlanner()
        self.goal_xy = goal_xy
        self.wander = wander
        self.last_decision = "mppi_init"
        self.reset()

    def reset(self):
        self.planner.cancel()
        if self.goal_xy is not None:
            self.planner.set_goal(float(self.goal_xy[0]), float(self.goal_xy[1]))
            self.wander = False
        elif self.wander:
            self.planner.set_wander_mode(True)
        self.last_decision = "mppi_reset"

    def act(self, obs, height, safety_scales, pose):
        fwd = float(safety_scales.get("fwd", 1.0))
        bwd = float(safety_scales.get("bwd", 1.0))
        ang = float(safety_scales.get("ang", 1.0))
        px = float(pose.get("x", 0.0))
        py = float(pose.get("y", 0.0))
        th = float(pose.get("theta", 0.0))

        cmd = self.planner.tick(obs, (px, py, th), 0.033)
        if cmd is None:
            self.last_decision = "mppi_idle"
            return 0.0, 0.0

        v = float(cmd.get("fwd_mps", 0.0))
        w = float(cmd.get("ang_rads", 0.0))

        # Action mask from SafetyGuard (planner must respect remaining mobility)
        if v > 0:
            v *= fwd
            if fwd <= 0:
                v = 0.0
        elif v < 0:
            v *= bwd
            if bwd <= 0:
                v = 0.0
        w *= ang
        if ang <= 0:
            w = 0.0

        # If nose pinned but yaw free, bias spin toward open side via soft scores
        if fwd < 0.08 and ang > 0.2 and abs(w) < 0.05:
            w = 0.5 * ang  # gentle escape yaw; sign refined by planner next ticks

        self.last_decision = "mppi_v%.2f_w%.2f" % (v, w)
        return v, w
