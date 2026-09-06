"""Pluggable policy interface + HouseBotLite reference implementation.

HouseBotLite ports real house_bot phased recover:
  early divert → late spin → BACK → ~180 SPIN → COMMIT other way
BUT refuses to override hard stops from SafetyGuard.
  - Never command v>0 when fwd_scale==0
  - Reverse only when bwd_scale > BWD_OK
  - Commit aborts / re-backs if nose hard-pinned again
"""

from __future__ import annotations

import math
import numpy as np


# Mirrors src/house_bot.py (sim tick ≈ look; dt handled by runner)
EARLY_FWD_SCALE = 0.55
LATE_FWD_SCALE = 0.18
LATE_STUCK_LOOKS = 3
BWD_OK = 0.20

BACK_MPS = -0.18
BACK_STEPS = 35          # ~1.15s at 30Hz
SPIN_RAD = 0.90
SPIN_STEPS = 96          # ~3.2s → ~180° ambition before ang_scale
COMMIT_STEPS = 165       # ~5.5s
COMMIT_V = 0.12
EARLY_W = 0.55
LATE_W = 0.85

OBS_THRESH = 100
RCX, RCY = 81, 119


class Policy:
    """Base policy interface."""

    def reset(self):
        pass

    def act(self, obs, height, safety_scales, pose):
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, v_max=0.1, w_max=0.5):
        self.v_max = v_max
        self.w_max = w_max

    def act(self, obs, height, safety_scales, pose):
        v = np.random.uniform(-self.v_max, self.v_max)
        w = np.random.uniform(-self.w_max, self.w_max)
        return float(v), float(w)


class StopPolicy(Policy):
    def act(self, obs, height, safety_scales, pose):
        return 0.0, 0.0


def _sector_free(obs, x0, x1, y0, y1):
    """Fraction of cells below obstacle threshold in a box."""
    h, w = obs.shape
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = obs[y0:y1, x0:x1]
    return float(np.mean(patch < OBS_THRESH))


def score_sectors(obs):
    """Cheap left/right/fwd free scores from ego obstacle map."""
    if obs is None:
        return {"left": 0.5, "right": 0.5, "fwd_near": 1.0, "fwd_mid": 1.0}
    left = _sector_free(obs, RCX - 50, RCX - 5, RCY - 55, RCY + 55)
    right = _sector_free(obs, RCX + 5, RCX + 50, RCY - 55, RCY + 55)
    fwd_near = _sector_free(obs, RCX + 5, RCX + 35, RCY - 25, RCY + 25)
    fwd_mid = _sector_free(obs, RCX + 35, RCX + 80, RCY - 35, RCY + 35)
    return {
        "left": left,
        "right": right,
        "fwd_near": fwd_near,
        "fwd_mid": fwd_mid,
    }


class HouseBotLite(Policy):
    """Look-before-leap + phased recover; never overrides hard stops."""

    def __init__(self, v_cruise=0.15, w_cruise=0.4):
        self.v_cruise = v_cruise
        self.w_cruise = w_cruise
        self.reset()

    def reset(self):
        self.phase = None          # None | back | spin | commit
        self.phase_steps = 0
        self.phase_turn = "left"   # left | right
        self.late_streak = 0
        self.commit_theta = None
        self.last_decision = "init"
        self.recover_starts = 0
        self.phase_log = []

    def _pick_turn(self, scores, curious=False):
        left = scores.get("left", 0.5)
        right = scores.get("right", 0.5)
        turn = "left" if left >= right else "right"
        if curious and abs(left - right) < 0.08:
            # slight bias flip for variety without RNG dependency in tests
            turn = "right" if turn == "left" else "left"
        return turn

    def _safe_cmd(self, v, w, fwd_scale, bwd_scale, ang_scale):
        """Policy-side clamp: never ask for motion SafetyGuard will hard-stop
        in a way that hides intent — especially never ask v>0 at fwd=0."""
        if v > 0 and fwd_scale <= 0:
            v = 0.0
        if v < 0 and bwd_scale <= BWD_OK:
            v = 0.0
        if abs(w) > 0 and ang_scale <= 0:
            w = 0.0
        return float(v), float(w)

    def _start_back(self, turn):
        self.phase = "back"
        self.phase_steps = 0
        self.phase_turn = turn
        self.late_streak = 0
        self.recover_starts += 1
        self.phase_log.append("back_" + turn)

    def _start_spin(self):
        self.phase = "spin"
        self.phase_steps = 0
        self.phase_log.append("spin_" + self.phase_turn)

    def _start_commit(self, pose):
        self.phase = "commit"
        self.phase_steps = 0
        self.commit_theta = float(pose.get("theta", 0.0))
        self.phase_log.append("commit")

    def _advance_phase(self, pose):
        if self.phase is None:
            return
        self.phase_steps += 1
        if self.phase == "back" and self.phase_steps >= BACK_STEPS:
            self._start_spin()
        elif self.phase == "spin" and self.phase_steps >= SPIN_STEPS:
            self._start_commit(pose)
        elif self.phase == "commit" and self.phase_steps >= COMMIT_STEPS:
            self.phase = None
            self.phase_turn = "left"
            self.commit_theta = None
            self.phase_log.append("done")

    def act(self, obs, height, safety_scales, pose):
        fwd_scale = float(safety_scales.get("fwd", 1.0))
        bwd_scale = float(safety_scales.get("bwd", 1.0))
        ang_scale = float(safety_scales.get("ang", 1.0))
        scores = score_sectors(obs)
        free_near = scores["fwd_near"]
        free_mid = scores["fwd_mid"]

        late = (fwd_scale < LATE_FWD_SCALE) or (free_near < 0.40)
        early = (
            (not late)
            and ang_scale > 0
            and (
                (fwd_scale < EARLY_FWD_SCALE and fwd_scale >= LATE_FWD_SCALE)
                or (free_mid < 0.72)
                or (free_near < 0.80 and free_mid < 0.85)
            )
        )

        # ── Phased recover owns control ──
        if self.phase is not None:
            self._advance_phase(pose)
            turn = self.phase_turn
            sign = 1.0 if turn == "left" else -1.0

            if self.phase == "back":
                v = BACK_MPS if bwd_scale > BWD_OK else 0.0
                w = 0.25 * sign if ang_scale > 0 else 0.0
                self.last_decision = "recover_back_" + turn
                return self._safe_cmd(v, w, fwd_scale, bwd_scale, ang_scale)

            if self.phase == "spin":
                # Pure yaw; never sneak forward while spinning out of a pinch
                v = 0.0
                w = SPIN_RAD * sign if ang_scale > 0 else 0.0
                self.last_decision = "recover_spin_" + turn
                return self._safe_cmd(v, w, fwd_scale, bwd_scale, ang_scale)

            if self.phase == "commit":
                if late:
                    self.late_streak += 1
                else:
                    self.late_streak = 0
                if late and self.late_streak >= LATE_STUCK_LOOKS:
                    preferred = self._pick_turn(scores, curious=True)
                    self._start_back(preferred)
                    self.last_decision = "recover_reback_" + preferred
                    v = BACK_MPS if bwd_scale > BWD_OK else 0.0
                    w = 0.25 * (1.0 if preferred == "left" else -1.0)
                    return self._safe_cmd(v, w, fwd_scale, bwd_scale, ang_scale)

                # Drive NEW heading; clamp hard — commit must not override fwd=0
                if fwd_scale <= 0:
                    v = 0.0
                    w = LATE_W * sign if ang_scale > 0 else 0.0
                    self.last_decision = "commit_blocked"
                else:
                    v = COMMIT_V * min(1.0, max(0.0, fwd_scale))
                    # Hold commit heading lightly
                    th = self.commit_theta
                    if th is not None:
                        err = (th - float(pose.get("theta", 0.0)) + math.pi) % (2 * math.pi) - math.pi
                        w = 0.6 * err
                    else:
                        w = 0.0
                    self.last_decision = "commit_" + turn
                return self._safe_cmd(v, w, fwd_scale, bwd_scale, ang_scale)

            # phase cleared mid-tick
            if self.phase is None:
                pass  # fall through to normal
            else:
                self.last_decision = "recover_idle"
                return 0.0, 0.0

        # ── Normal look-before-leap ──
        if early or late:
            if early and not late:
                self.late_streak = 0
            else:
                self.late_streak += 1

            turn = self._pick_turn(scores, curious=False)
            sign = 1.0 if turn == "left" else -1.0

            if late and self.late_streak >= LATE_STUCK_LOOKS:
                turn = self._pick_turn(scores, curious=True)
                self._start_back(turn)
                self.last_decision = "recover_back_" + turn
                v = BACK_MPS if bwd_scale > BWD_OK else 0.0
                w = 0.25 * (1.0 if turn == "left" else -1.0)
                return self._safe_cmd(v, w, fwd_scale, bwd_scale, ang_scale)

            if early and not late:
                # Soft veer: slow forward + yaw while clearance remains
                v = self.v_cruise * 0.45 if fwd_scale > 0 else 0.0
                w = EARLY_W * sign
                self.last_decision = "early_" + turn
                return self._safe_cmd(v, w, fwd_scale, bwd_scale, ang_scale)

            # Late but not yet recover: in-place spin
            v = 0.0
            w = LATE_W * sign if ang_scale > 0 else 0.0
            self.last_decision = "late_" + turn
            return self._safe_cmd(v, w, fwd_scale, bwd_scale, ang_scale)

        self.late_streak = 0
        self.last_decision = "cruise"
        return self._safe_cmd(self.v_cruise, 0.0, fwd_scale, bwd_scale, ang_scale)


def create_policy(name: str):
    if name == "random":
        return RandomPolicy()
    elif name == "housebot":
        return HouseBotLite()
    elif name == "stop":
        return StopPolicy()
    elif name in ("unsafe", "unsafe_commit"):
        from sim.unsafe_policy import UnsafeCommitPolicy
        return UnsafeCommitPolicy()
    else:
        raise ValueError(f"Unknown policy: {name}")
