"""Gated planner/costmap feed from honest ego labels or EvidenceMap.

Default consumers stay on legacy (obs, known). When enabled via
KEVIN_EGO_PLAN=1 and/or KEVIN_EGO_LABELS=1, costmap/planner/safety see
ego-derived dual maps. If KEVIN_EVIDENCE_MAP=1 and the evidence grid has
been updated, prefer a pose-warped EvidenceMap.to_obs_known() in ego space.

SELF honesty (CONTRACT):
  - never known-clear under chassis / footprint
  - SELF must not become obstacles in the costmap feed
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from globalmap import GlobalMap
from robot_config import EGO_PX_SIZE, FRAME_H, FRAME_W, FOOTPRINT_BOXES, RCX, RCY

from .ego_rs1 import labels_to_obs_known
from .evidence_map import EvidenceMap

Box = Tuple[int, int, int, int]


def apply_self_honest(
    obs: np.ndarray,
    known: np.ndarray,
    footprint_boxes: Sequence[Box] = FOOTPRINT_BOXES,
) -> None:
    """Force footprint/SELF cells to obs=0, known=0 (in-place).

    Never invent known-clear under chassis; never treat SELF as obstacle.
    """
    for x0, y0, x1, y1 in footprint_boxes:
        obs[y0:y1, x0:x1] = 0
        known[y0:y1, x0:x1] = 0


def ego_labels_to_planner_feed(
    labels: np.ndarray,
    height_cm: np.ndarray,
    *,
    obs_out: Optional[np.ndarray] = None,
    known_out: Optional[np.ndarray] = None,
    footprint_boxes: Sequence[Box] = FOOTPRINT_BOXES,
):
    """Per-frame ego labels → honest (obs, known) for planner consumers."""
    obs, known = labels_to_obs_known(
        labels, height_cm, obs_out=obs_out, known_out=known_out)
    apply_self_honest(obs, known, footprint_boxes)
    return obs, known


def evidence_to_ego_obs_known(
    evidence_map: EvidenceMap,
    pose_xy_theta: Tuple[float, float, float],
    *,
    ego_h: int = FRAME_H,
    ego_w: int = FRAME_W,
    ego_cx: float = float(RCX),
    ego_cy: float = float(RCY),
    ego_px_size: float = float(EGO_PX_SIZE),
    obs_out: Optional[np.ndarray] = None,
    known_out: Optional[np.ndarray] = None,
    footprint_boxes: Sequence[Box] = FOOTPRINT_BOXES,
    world_obs: Optional[np.ndarray] = None,
    world_known: Optional[np.ndarray] = None,
):
    """Warp EvidenceMap.to_obs_known() from world → ego, then SELF-honest.

    Uses GlobalMap._inverse_affine (world→ego) as the cv2.warpAffine matrix
    (OpenCV inverts it unless WARP_INVERSE_MAP is set).
    """
    import cv2

    if world_obs is None or world_known is None:
        w_obs, w_known = evidence_map.to_obs_known()
    else:
        w_obs, w_known = world_obs, world_known

    x, y, theta = (
        float(pose_xy_theta[0]),
        float(pose_xy_theta[1]),
        float(pose_xy_theta[2]),
    )
    # cv2.warpAffine without WARP_INVERSE_MAP treats M as src→dst and
    # inverts it internally. Pass world→ego (inverse_affine).
    M = GlobalMap._inverse_affine(
        x, y, theta, float(ego_cx), float(ego_cy), float(ego_px_size))
    M = np.asarray(M, dtype=np.float32)

    shape = (int(ego_h), int(ego_w))
    obs = obs_out if obs_out is not None else np.zeros(shape, dtype=np.uint8)
    known = known_out if known_out is not None else np.zeros(shape, dtype=np.uint8)
    if obs.shape != shape or known.shape != shape:
        raise ValueError("obs_out/known_out must be (ego_h, ego_w)")

    cv2.warpAffine(
        w_obs, M, (int(ego_w), int(ego_h)),
        dst=obs, flags=cv2.INTER_NEAREST, borderValue=0)
    cv2.warpAffine(
        w_known, M, (int(ego_w), int(ego_h)),
        dst=known, flags=cv2.INTER_NEAREST, borderValue=0)
    apply_self_honest(obs, known, footprint_boxes)
    return obs, known


def select_planner_feed(
    *,
    gated: bool,
    ego_labels: Optional[np.ndarray] = None,
    ego_height: Optional[np.ndarray] = None,
    ego_obs_shim: Optional[np.ndarray] = None,
    ego_known_shim: Optional[np.ndarray] = None,
    evidence_map: Optional[EvidenceMap] = None,
    pose_xy_theta: Optional[Tuple[float, float, float]] = None,
    prefer_evidence: bool = False,
    footprint_boxes: Sequence[Box] = FOOTPRINT_BOXES,
    obs_out: Optional[np.ndarray] = None,
    known_out: Optional[np.ndarray] = None,
    legacy_obs: Optional[np.ndarray] = None,
    legacy_known: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Pick planner (obs, known) source.

    Returns
    -------
    (obs, known, source) where source is 'evidence' | 'ego' | 'legacy'.
    When source=='legacy' and outs are provided, copies legacy into outs if given;
    otherwise returns the legacy arrays as-is (may be None).
    """
    if not gated:
        if obs_out is not None and legacy_obs is not None:
            np.copyto(obs_out, legacy_obs)
            obs = obs_out
        else:
            obs = legacy_obs
        if known_out is not None and legacy_known is not None:
            np.copyto(known_out, legacy_known)
            known = known_out
        else:
            known = legacy_known
        return obs, known, "legacy"

    # Prefer accumulated evidence when requested and non-empty.
    if (
        prefer_evidence
        and evidence_map is not None
        and int(getattr(evidence_map, "frame_i", 0)) > 0
        and pose_xy_theta is not None
    ):
        obs, known = evidence_to_ego_obs_known(
            evidence_map,
            pose_xy_theta,
            obs_out=obs_out,
            known_out=known_out,
            footprint_boxes=footprint_boxes,
        )
        return obs, known, "evidence"

    # Fresh labels → convert; else reuse last ego shim buffers.
    if ego_labels is not None and ego_height is not None:
        obs, known = ego_labels_to_planner_feed(
            ego_labels,
            ego_height,
            obs_out=obs_out,
            known_out=known_out,
            footprint_boxes=footprint_boxes,
        )
        return obs, known, "ego"

    if ego_obs_shim is not None and ego_known_shim is not None:
        obs = obs_out if obs_out is not None else np.empty_like(ego_obs_shim)
        known = known_out if known_out is not None else np.empty_like(ego_known_shim)
        np.copyto(obs, ego_obs_shim)
        np.copyto(known, ego_known_shim)
        apply_self_honest(obs, known, footprint_boxes)
        return obs, known, "ego"

    if obs_out is not None and legacy_obs is not None:
        np.copyto(obs_out, legacy_obs)
        obs = obs_out
    else:
        obs = legacy_obs
    if known_out is not None and legacy_known is not None:
        np.copyto(known_out, legacy_known)
        known = known_out
    else:
        known = legacy_known
    return obs, known, "legacy"
