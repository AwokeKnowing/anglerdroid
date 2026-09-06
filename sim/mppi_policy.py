"""MPPI policy wrapper for sim — same NumPy planner as live Orin.

Safety scales are applied as an *action mask* before Robot.step also clips
(belt-and-suspenders; matches insect contract).

When hard-pinned, do a short insect escape (back / spin) instead of dithering
near-zero MPPI samples — that was what made enjoy GIFs look stuck.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mppi_costmap import MppiCostmapPlanner  # noqa: E402
from sim.robot import soft_inflate, score_heading_with_lookahead


class MppiSimPolicy:
    """Async-goal MPPI using ego obs; never asks v>0 when fwd_scale==0."""

    PIN_STREAK = 4          # ~130ms hard nose pin → escape
    BACK_MPS = -0.14
    SPIN_W = 0.75
    ESCAPE_SPIN_STEPS = 48  # ~1.6s
    ESCAPE_BACK_STEPS = 22  # ~0.7s

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
        self.pin_streak = 0
        self.escape_phase = None  # None | "back" | "spin"
        self.escape_left = 0
        self.escape_sign = 1.0
        self.last_decision = "mppi_reset"

    def _open_side_sign(self, obs) -> float:
        """+1 = CCW (left), -1 = CW (right) toward clearer soft heading."""
        if obs is None:
            return 1.0
        soft = soft_inflate(obs)
        left = score_heading_with_lookahead(soft, math.radians(70))
        right = score_heading_with_lookahead(soft, math.radians(-70))
        return 1.0 if left >= right else -1.0

    def _mask(self, v, w, fwd, bwd, ang):
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
        return v, w

    def act(self, obs, height, safety_scales, pose):
        fwd = float(safety_scales.get("fwd", 1.0))
        bwd = float(safety_scales.get("bwd", 1.0))
        ang = float(safety_scales.get("ang", 1.0))
        px = float(pose.get("x", 0.0))
        py = float(pose.get("y", 0.0))
        th = float(pose.get("theta", 0.0))

        # --- Insect escape when hard-pinned (overrides dithering MPPI) ---
        if self.escape_phase is not None:
            self.escape_left -= 1
            if self.escape_phase == "back":
                v, w = self.BACK_MPS, 0.0
                self.last_decision = "mppi_escape_back"
                if self.escape_left <= 0 or bwd < 0.15:
                    self.escape_phase = "spin"
                    self.escape_left = self.ESCAPE_SPIN_STEPS
                    self.escape_sign = self._open_side_sign(obs)
            else:  # spin
                v, w = 0.0, self.SPIN_W * self.escape_sign
                self.last_decision = "mppi_escape_spin"
                if self.escape_left <= 0 or (fwd > 0.35 and ang > 0.4):
                    self.escape_phase = None
                    self.pin_streak = 0
                    # Re-seed wander so planner picks a new direction
                    if self.wander:
                        self.planner.set_wander_mode(True)
            return self._mask(v, w, fwd, bwd, ang)

        if fwd < 0.08:
            self.pin_streak += 1
        else:
            self.pin_streak = 0

        if self.pin_streak >= self.PIN_STREAK and ang > 0.15:
            self.escape_sign = self._open_side_sign(obs)
            if bwd > 0.2:
                self.escape_phase = "back"
                self.escape_left = self.ESCAPE_BACK_STEPS
            else:
                self.escape_phase = "spin"
                self.escape_left = self.ESCAPE_SPIN_STEPS
            self.last_decision = "mppi_escape_start"
            v = self.BACK_MPS if self.escape_phase == "back" else 0.0
            w = 0.0 if self.escape_phase == "back" else self.SPIN_W * self.escape_sign
            return self._mask(v, w, fwd, bwd, ang)

        cmd = self.planner.tick(obs, (px, py, th), 0.033)
        if cmd is None:
            # Idle: still spin toward open if partially constrained
            if fwd < 0.4 and ang > 0.2:
                s = self._open_side_sign(obs)
                self.last_decision = "mppi_idle_spin"
                return self._mask(0.0, 0.45 * s, fwd, bwd, ang)
            self.last_decision = "mppi_idle"
            return 0.0, 0.0

        v = float(cmd.get("fwd_mps", 0.0))
        w = float(cmd.get("ang_rads", 0.0))
        v, w = self._mask(v, w, fwd, bwd, ang)

        # Tiny-command dither after mask → move (not freeze).
        # Never flip a reverse into forward — that fought escape.
        # Wide |w| gate: shy MPPI often pairs tiny v with leftover w noise.
        if abs(v) < 0.05:
            if v < -0.005 and bwd > 0.15:
                self.last_decision = "mppi_keep_back"
                return self._mask(min(v, -0.10), 0.0, fwd, bwd, ang)
            if fwd >= 0.30 and v >= 0.0:
                self.last_decision = "mppi_nudge_fwd"
                return self._mask(0.12, w if abs(w) < 0.4 else 0.0, fwd, bwd, ang)
            if ang > 0.15:
                s = self._open_side_sign(obs)
                self.last_decision = "mppi_nudge_spin"
                return self._mask(0.0, 0.65 * s, fwd, bwd, ang)
            if bwd > 0.2:
                # Yaw also clamped — reverse to reopen (insect)
                self.last_decision = "mppi_nudge_reopen"
                return self._mask(-0.10, 0.0, fwd, bwd, ang)

        self.last_decision = "mppi_v%.2f_w%.2f" % (v, w)
        return v, w
