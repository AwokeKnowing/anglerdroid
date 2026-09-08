"""RS1 top-down verts → ego label map (honest CLEAR / OBSTACLE / SELF).

First cut: CPU scatter matching vision.process_rs1_topdown geometry, then
geometric self from robot_config axle boxes. GPU path lands next.
"""
from __future__ import annotations

import numpy as np

from robot_config import (
    FRAME_H, FRAME_W, EGO_PX_SIZE,
    UNDER_ROBOT_BOXES, SELF_IGNORE_BOXES, FOOTPRINT_BOXES,
)

from .labels import UNKNOWN, SELF, CLEAR, OBSTACLE

# Keep in sync with vision.TD_FLOOR_CLIP until shared config exists.
TD_FLOOR_CLIP = np.float32(0.91)


def _clip_border(verts, border=4, orig_w=848, orig_h=480):
    if verts is None or len(verts) == 0:
        return verts
    # Same idea as vision._clip_decimated_border: drop noisy outer ring.
    # Without camera intrinsics here, skip if no pixel index channel.
    return verts


def label_rs1_ego(
    verts,
    *,
    out_h: int = FRAME_H,
    out_w: int = FRAME_W,
    floor_clip_m: float = float(TD_FLOOR_CLIP),
    px_size: float = float(EGO_PX_SIZE),
    under_boxes=None,
    self_boxes=None,
    height_out: np.ndarray | None = None,
):
    """Project RS1 pointcloud into ego labels.

    Returns
    -------
    labels : (H,W) uint8 — UNKNOWN|SELF|CLEAR|OBSTACLE
    height_cm : (H,W) uint8 — obstacle height above floor (0 if none)
    """
    labels = np.zeros((out_h, out_w), dtype=np.uint8)
    height = height_out if height_out is not None else np.zeros((out_h, out_w), dtype=np.uint8)
    if height_out is None:
        height.fill(0)
    else:
        height.fill(0)

    if verts is None or len(verts) == 0:
        _paint_self(labels, under_boxes, self_boxes)
        return labels, height

    v = np.asarray(verts, dtype=np.float32)
    z = v[:, 2]
    valid = z > 0.01
    if not np.any(valid):
        _paint_self(labels, under_boxes, self_boxes)
        return labels, height

    scale = np.float32(1.0 / px_size)
    center = np.float32([out_w * 0.5, out_h * 0.5])
    vv = v[valid]
    p = vv[:, :2] * scale + center
    with np.errstate(invalid="ignore"):
        ja, ia = p.astype(np.uint32).T
    ma = (ia < np.uint32(out_h)) & (ja < np.uint32(out_w))
    ia_m, ja_m = ia[ma], ja[ma]
    zv = vv[ma, 2]

    # Floor → CLEAR; below floor_clip → OBSTACLE (tallest height wins)
    floor = zv >= floor_clip_m
    labels[ia_m[floor], ja_m[floor]] = CLEAR

    obs = ~floor
    if np.any(obs):
        ia_o, ja_o = ia_m[obs], ja_m[obs]
        h = np.clip(((floor_clip_m - zv[obs]) * 100.0).astype(np.int32), 1, 100).astype(np.uint8)
        # Obstacle beats clear if both somehow hit (shouldn't after split)
        labels[ia_o, ja_o] = OBSTACLE
        np.maximum.at(height, (ia_o, ja_o), h)

    # Rotate 180° to match vision ego (RS1 mounted inverted relative to drive)
    labels = labels[::-1, ::-1].copy()
    height = height[::-1, ::-1].copy()

    _paint_self(labels, under_boxes, self_boxes)
    return labels, height


def _paint_self(labels, under_boxes, self_boxes):
    """SELF wins: neither clear nor obstacle."""
    boxes = list(under_boxes or UNDER_ROBOT_BOXES) + list(self_boxes or SELF_IGNORE_BOXES)
    if not boxes:
        boxes = list(FOOTPRINT_BOXES)
    for x0, y0, x1, y1 in boxes:
        labels[y0:y1, x0:x1] = SELF


def labels_to_obs_known(labels: np.ndarray, height_cm: np.ndarray):
    """Compat shim toward old (obs, known) consumers.

    CLEAR → known=255, obs=0
    OBSTACLE → known=255, obs=height
    SELF → known=0, obs=0   (honest: not sensed clear, not obstacle)
    UNKNOWN → known=0, obs=0
    """
    known = np.zeros(labels.shape, dtype=np.uint8)
    obs = np.zeros(labels.shape, dtype=np.uint8)
    clear = labels == CLEAR
    obstacle = labels == OBSTACLE
    known[clear | obstacle] = 255
    obs[obstacle] = height_cm[obstacle]
    return obs, known
