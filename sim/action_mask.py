"""Apply DualScales as an MPPI / planner action mask (offline spike).

Hard SafetyGuard scales multiply commanded (v, w) last. Hard zero at contact
is absolute and never overridden. Soft scales inform mode / cost only.

Modes (planner-facing):
  open     — soft preferred + hard free
  prefer   — soft_blocked but not squeeze (rare: soft mid while hard still open
             is usually squeeze; prefer reserved for soft_cost-only hints)
  squeeze  — soft_blocked && hard_free: slow forward, keep yaw authority
  recover  — hard_fwd == 0 (nose contact): never command v>0; reverse/spin OK
  halt     — hard_fwd, hard_bwd, and hard_ang all zero (laterally crushed stop)

Not wired to live drive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sim.dual_clearance import DualScales, evaluate

Mode = Literal["open", "prefer", "squeeze", "recover", "halt"]


@dataclass(frozen=True)
class MaskedAction:
    v: float
    w: float
    mode: Mode
    scales: DualScales


def classify_mode(scales: DualScales) -> Mode:
    """Map DualScales → planner mode (no command mutation)."""
    if scales.hard_fwd == 0.0 and scales.hard_bwd == 0.0 and scales.hard_ang == 0.0:
        return "halt"
    if scales.hard_fwd == 0.0:
        return "recover"
    if scales.squeeze:
        return "squeeze"
    if scales.soft_blocked:
        return "prefer"
    return "open"


def apply_action_mask(
    v_cmd: float,
    w_cmd: float,
    scales: DualScales,
    *,
    squeeze_v_scale: float = 0.35,
) -> MaskedAction:
    """Mask commanded twist by hard scales; tag squeeze/recover modes.

    Contract:
    - hard_* multiply last (absolute envelope).
    - Never emit v>0 when hard_fwd==0 (recover / contact).
    - Never emit v<0 when hard_bwd==0.
    - Squeeze further attenuates forward v (planner prefer-slow), not hard.
    - Hard zero at contact is never lifted by squeeze or ang floor here
      (ang floor lives inside evaluate()).
    """
    mode = classify_mode(scales)

    # Hard envelope as last-line multiply.
    if v_cmd >= 0.0:
        v = float(v_cmd) * float(scales.hard_fwd)
    else:
        v = float(v_cmd) * float(scales.hard_bwd)
    w = float(w_cmd) * float(scales.hard_ang)

    if mode == "halt":
        return MaskedAction(v=0.0, w=0.0, mode=mode, scales=scales)

    if mode == "recover":
        # Nose contact: forbid forward regardless of floating-point residue.
        if v > 0.0:
            v = 0.0
        return MaskedAction(v=v, w=w, mode=mode, scales=scales)

    if mode == "squeeze" and v > 0.0:
        v *= float(squeeze_v_scale)

    return MaskedAction(v=v, w=w, mode=mode, scales=scales)


def action_feasible(
    v_cmd: float,
    w_cmd: float,
    scales: DualScales,
    *,
    eps: float = 1e-9,
) -> bool:
    """True if a raw sample survives the hard mask with non-trivial motion intent.

    Used as an MPPI sample gate: reject samples that hard-zero would nullify
    when the sample asked for motion on a blocked axis.
    """
    if v_cmd > eps and scales.hard_fwd <= eps:
        return False
    if v_cmd < -eps and scales.hard_bwd <= eps:
        return False
    if abs(w_cmd) > eps and scales.hard_ang <= eps:
        return False
    # Pure zero command is always "feasible" (idle) but useless for MPPI;
    # callers that want motion samples should check |v|+|w| themselves.
    return True


def soft_cost_bonus(scales: DualScales, *, w_soft: float = 1.0) -> float:
    """Additive MPPI cost from soft prefer buffer (0 preferred → higher cost)."""
    # Prefer forward corridor; ang soft cost helps avoid side-swipe paths.
    return float(w_soft) * (
        0.6 * scales.soft_cost_fwd()
        + 0.2 * scales.soft_cost_bwd()
        + 0.2 * scales.soft_cost_ang()
    )


def mask_from_clearances(
    v_cmd: float,
    w_cmd: float,
    fwd_m: float,
    bwd_m: float,
    lat_m: float,
    **evaluate_kwargs,
) -> MaskedAction:
    """Convenience: evaluate clearances then apply_action_mask."""
    scales = evaluate(fwd_m, bwd_m, lat_m, **evaluate_kwargs)
    return apply_action_mask(v_cmd, w_cmd, scales)
