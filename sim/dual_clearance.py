"""Soft/hard dual-clearance scales (pure Python, no hardware).

Insect-agility spike: split planner soft prefer-buffer from SafetyGuard hard
envelope. Soft_* and hard_* are both scales in [0, 1] with 1 = fully clear /
preferred. Planner cost is ``1 - soft_*``. Hard scales multiply commanded
velocities last; hard zero at contact is absolute and never weakened.

Defaults (meters):
  soft prefer band: soft_prefer_near=0.15 .. soft_prefer_far=0.40
  hard throttle:    hard_zero=0.025 .. hard_start=0.15
  ang floor:        when hard_fwd < 0.15 and hard_bwd > 0.3, keep hard_ang
                    >= ang_floor (0.35) unless laterally crushed
                    (lat_clear_m <= hard_zero).

Not wired to live drive.
"""
from __future__ import annotations

from dataclasses import dataclass


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def clearance_scale(clear_m: float, near_m: float, far_m: float) -> float:
    """Map clearance meters → scale [0, 1].

    scale = 0 when clear_m <= near_m (blocked / contact band)
    scale = 1 when clear_m >= far_m (fully open / preferred)
    linear between. Monotonic non-decreasing in clear_m.
    """
    if far_m <= near_m:
        # Degenerate: treat as step at near_m.
        return 1.0 if clear_m > near_m else 0.0
    if clear_m <= near_m:
        return 0.0
    if clear_m >= far_m:
        return 1.0
    return (float(clear_m) - near_m) / (far_m - near_m)


@dataclass(frozen=True)
class DualScales:
    """Dual soft (planner) / hard (SafetyGuard) clearance scales.

    soft_*: 1 = fully preferred/open for planner; soft_cost = 1 - soft_*.
    hard_*: 1 = full authority; 0 = hard stop (SafetyGuard multiply).
    """

    soft_fwd: float
    hard_fwd: float
    soft_bwd: float
    hard_bwd: float
    soft_ang: float
    hard_ang: float
    soft_blocked: bool  # soft_fwd below fully-preferred (scale < 1)
    hard_free: bool  # hard_fwd > 0
    squeeze: bool  # soft_blocked and hard_free

    def soft_cost_fwd(self) -> float:
        """Planner cost in [0, 1]: 0 preferred, 1 blocked."""
        return 1.0 - self.soft_fwd

    def soft_cost_bwd(self) -> float:
        return 1.0 - self.soft_bwd

    def soft_cost_ang(self) -> float:
        return 1.0 - self.soft_ang


def evaluate(
    fwd_clear_m: float,
    bwd_clear_m: float,
    lat_clear_m: float,
    *,
    soft_prefer_far: float = 0.40,
    soft_prefer_near: float = 0.15,
    hard_start: float = 0.15,
    hard_zero: float = 0.025,
    ang_floor: float = 0.35,
    ang_floor_fwd_max: float = 0.15,
    ang_floor_bwd_min: float = 0.3,
) -> DualScales:
    """Compute soft prefer scales + hard SafetyGuard scales from clearances.

    Parameters
    ----------
    fwd_clear_m, bwd_clear_m, lat_clear_m
        Minimum forward / backward / lateral clearance in meters.
    soft_prefer_far, soft_prefer_near
        Soft band: cost 0 (scale 1) at/above far; cost 1 (scale 0) at/below near.
    hard_start, hard_zero
        Hard throttle: scale 1 at/above start; hard zero (scale 0) at/below zero.
        hard_zero is ~2.5 cm — not 1 cm noise.
    ang_floor
        Minimum hard_ang when nose-pinched but rear free (insect spin-out),
        unless laterally crushed (lat_clear_m <= hard_zero).
    """
    soft_fwd = clearance_scale(fwd_clear_m, soft_prefer_near, soft_prefer_far)
    soft_bwd = clearance_scale(bwd_clear_m, soft_prefer_near, soft_prefer_far)
    soft_ang = clearance_scale(lat_clear_m, soft_prefer_near, soft_prefer_far)

    hard_fwd = clearance_scale(fwd_clear_m, hard_zero, hard_start)
    hard_bwd = clearance_scale(bwd_clear_m, hard_zero, hard_start)
    hard_ang = clearance_scale(lat_clear_m, hard_zero, hard_start)

    # Kinematic ang floor: spin-out when pinched nose + free rear.
    # Never apply when laterally crushed — hard_ang stays at contact zero.
    laterally_crushed = lat_clear_m <= hard_zero
    if (
        not laterally_crushed
        and hard_fwd < ang_floor_fwd_max
        and hard_bwd > ang_floor_bwd_min
    ):
        hard_ang = max(hard_ang, float(ang_floor))

    soft_fwd = _clamp01(soft_fwd)
    soft_bwd = _clamp01(soft_bwd)
    soft_ang = _clamp01(soft_ang)
    hard_fwd = _clamp01(hard_fwd)
    hard_bwd = _clamp01(hard_bwd)
    hard_ang = _clamp01(hard_ang)

    soft_blocked = soft_fwd < 1.0
    hard_free = hard_fwd > 0.0
    squeeze = soft_blocked and hard_free

    return DualScales(
        soft_fwd=soft_fwd,
        hard_fwd=hard_fwd,
        soft_bwd=soft_bwd,
        hard_bwd=hard_bwd,
        soft_ang=soft_ang,
        hard_ang=hard_ang,
        soft_blocked=soft_blocked,
        hard_free=hard_free,
        squeeze=squeeze,
    )
