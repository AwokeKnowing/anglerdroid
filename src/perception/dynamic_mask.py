"""Dynamic masking for SLAM/VO from honest ego labels (CONTRACT step 4 wedge).

Builds a per-cell mask of ego cells that SLAM / visual odometry should treat
as outliers — not trusted for static geometry (descriptors, scan thumbs,
keyframe obs).

Rules (docs/perception/CONTRACT.md):
  - SELF is always masked (robot volume is not world structure).
  - CLEAR / UNKNOWN are never masked by this wedge (they are not movers).
  - Optional ephemeral OBSTACLE masking vs a prior obstacle layer: cells
    labeled OBSTACLE that are absent from ``prior_obstacle`` can be masked
    as likely movers. If no prior is available, mask SELF only.
  - Never invent CLEAR under chassis; SELF wins. This module only zeros /
    ignores geometry for SLAM — it does not rewrite ego labels.

Hot-path: preallocate ``out`` / ``obs_out``; in-place fill. Target ≪ few ms
for 320×240 on Orin CPU. No ROS.

Env gate (live wire, default off): ``KEVIN_SLAM_DYNAMIC_MASK=1`` — vision
feeds the mask into self-SLAM keyframe obs (zeros masked cells before
descriptor / thumb). Full RGB-D+wheel+IMU dynamic SLAM remains TODO.

Live wire: when ``KEVIN_EVIDENCE_MAP=1`` and the evidence grid has
updates, vision passes a pose-warped EvidenceMap obstacle prior with
``mask_ephemeral=True`` so movers without accumulated static evidence are
excluded from VO/SLAM while persistent furniture remains. Still gated by
``KEVIN_SLAM_DYNAMIC_MASK=1`` (default off).
Live metrics line ``slam_mask:`` reports SELF / ephemeral / ms.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .labels import SELF, OBSTACLE


def build_slam_outlier_mask(
    ego_labels: np.ndarray,
    *,
    prior_obstacle: Optional[np.ndarray] = None,
    mask_ephemeral: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build bool mask of cells SLAM/VO should not trust for static geometry.

    Parameters
    ----------
    ego_labels : (H,W) uint8 — UNKNOWN|SELF|CLEAR|OBSTACLE
    prior_obstacle : optional (H,W) bool or uint8 — nonzero/True marks cells
        previously observed as (likely static) obstacle. Same shape as labels.
        Used only when ``mask_ephemeral`` is True.
    mask_ephemeral : if True and ``prior_obstacle`` is provided, also mask
        OBSTACLE cells that are absent from the prior (ephemeral movers).
        If prior is None, behaves like SELF-only (API ready for decay prior).
    out : optional preallocated bool (H,W); filled in-place when given.

    Returns
    -------
    mask : (H,W) bool — True = outlier / do not trust for static SLAM/VO
    """
    labels = np.asarray(ego_labels)
    if out is None:
        mask = np.zeros(labels.shape, dtype=bool)
    else:
        mask = out
        if mask.shape != labels.shape:
            raise ValueError(
                "out shape %s != labels shape %s" % (mask.shape, labels.shape))
        if mask.dtype != np.bool_:
            raise TypeError("out must be bool dtype, got %s" % mask.dtype)
        mask.fill(False)

    # SELF always masked — robot is not world structure
    mask |= labels == SELF

    if mask_ephemeral and prior_obstacle is not None:
        prior = np.asarray(prior_obstacle)
        if prior.shape != labels.shape:
            raise ValueError(
                "prior_obstacle shape %s != labels shape %s"
                % (prior.shape, labels.shape))
        # Ephemeral: current OBSTACLE without static prior. Do not invent CLEAR.
        ephemeral = (labels == OBSTACLE) & (prior == 0)
        mask |= ephemeral

    return mask


def apply_mask_to_obs(
    obs: np.ndarray,
    mask: np.ndarray,
    *,
    obs_out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Zero obstacle heights in masked cells (SLAM/VO outlier treatment).

    Does not invent known-clear; only clears obs where mask is True.
    Prefer ``obs_out`` prealloc so the live map buffer is not mutated.
    """
    if obs_out is None:
        out = np.array(obs, copy=True, dtype=obs.dtype)
    else:
        out = obs_out
        if out.shape != obs.shape:
            raise ValueError(
                "obs_out shape %s != obs shape %s" % (out.shape, obs.shape))
        np.copyto(out, obs)
    out[mask] = 0
    return out


def mask_counts(
    ego_labels: np.ndarray,
    mask: np.ndarray,
    *,
    prior_obstacle: Optional[np.ndarray] = None,
    mask_ephemeral: bool = False,
):
    """Return ``(self_n, ephemeral_n, total_n)`` for live ``slam_mask:`` metrics.

    ``self_n`` = masked SELF cells; ``ephemeral_n`` = masked OBSTACLE cells
    absent from prior (0 when ephemeral masking is off / no prior);
    ``total_n`` = all True cells in ``mask``.
    """
    labels = np.asarray(ego_labels)
    m = np.asarray(mask, dtype=bool)
    if m.shape != labels.shape:
        raise ValueError(
            "mask shape %s != labels shape %s" % (m.shape, labels.shape))
    self_n = int(np.count_nonzero((labels == SELF) & m))
    total_n = int(np.count_nonzero(m))
    if mask_ephemeral and prior_obstacle is not None:
        prior = np.asarray(prior_obstacle)
        if prior.shape != labels.shape:
            raise ValueError(
                "prior_obstacle shape %s != labels shape %s"
                % (prior.shape, labels.shape))
        eph_n = int(np.count_nonzero(
            (labels == OBSTACLE) & (prior == 0) & m))
    else:
        eph_n = 0
    return self_n, eph_n, total_n


def mask_as_uint8(mask: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert bool outlier mask to uint8 (0/255) for logging / Rerun."""
    if out is None:
        return (np.asarray(mask, dtype=bool).astype(np.uint8) * np.uint8(255))
    out.fill(0)
    out[mask] = 255
    return out
