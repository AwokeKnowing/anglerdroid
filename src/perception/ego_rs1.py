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


def _blit_x(dst: np.ndarray, src: np.ndarray, dx: int) -> None:
    """Horizontal blit of src into dst at column offset dx (no wrap)."""
    h, w = src.shape
    if dx == 0:
        np.copyto(dst, src)
        return
    if dx > 0:
        if dx >= w:
            return
        dst[:, dx:w] = src[:, : w - dx]
    else:
        if -dx >= w:
            return
        dst[:, : w + dx] = src[:, -dx:]


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
    labels_out: np.ndarray | None = None,
    work_labels: np.ndarray | None = None,
    work_height: np.ndarray | None = None,
    x_offset: int = 0,
):
    """Project RS1 pointcloud into ego labels.

    Scatter is camera-centered (same as vision._k1 before TD_X_OFFSET blit),
    then rotated 180°, then optionally shifted by ``x_offset`` (pass
    vision.TD_X_OFFSET so axle boxes land on RCX/RCY), then SELF is painted.

    Returns
    -------
    labels : (H,W) uint8 — UNKNOWN|SELF|CLEAR|OBSTACLE
    height_cm : (H,W) uint8 — obstacle height above floor (0 if none)
    """
    # Camera-centered work buffers (pre-shift)
    cam_l = work_labels if work_labels is not None else np.zeros((out_h, out_w), dtype=np.uint8)
    cam_h = work_height if work_height is not None else np.zeros((out_h, out_w), dtype=np.uint8)
    cam_l.fill(0)
    cam_h.fill(0)

    if verts is not None and len(verts) > 0:
        v = np.asarray(verts, dtype=np.float32)
        z = v[:, 2]
        valid = z > 0.01
        if np.any(valid):
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
            cam_l[ia_m[floor], ja_m[floor]] = CLEAR

            obs = ~floor
            if np.any(obs):
                ia_o, ja_o = ia_m[obs], ja_m[obs]
                h = np.clip(
                    ((floor_clip_m - zv[obs]) * 100.0).astype(np.int32), 1, 100
                ).astype(np.uint8)
                cam_l[ia_o, ja_o] = OBSTACLE
                np.maximum.at(cam_h, (ia_o, ja_o), h)

    # Rotate 180° to match vision ego (RS1 mounted inverted relative to drive)
    cam_l_f = cam_l[::-1, ::-1]
    cam_h_f = cam_h[::-1, ::-1]

    labels = labels_out if labels_out is not None else np.zeros((out_h, out_w), dtype=np.uint8)
    height = height_out if height_out is not None else np.zeros((out_h, out_w), dtype=np.uint8)
    labels.fill(0)
    height.fill(0)

    if x_offset == 0:
        np.copyto(labels, cam_l_f)
        np.copyto(height, cam_h_f)
    else:
        _blit_x(labels, np.asarray(cam_l_f), int(x_offset))
        _blit_x(height, np.asarray(cam_h_f), int(x_offset))

    _paint_self(labels, under_boxes, self_boxes)
    return labels, height


def _paint_self(labels, under_boxes, self_boxes):
    """SELF wins: neither clear nor obstacle.

    Pass under_boxes=() / self_boxes=() to skip that group.
    Omit both (None) to use robot_config defaults.
    """
    if under_boxes is None and self_boxes is None:
        boxes = list(FOOTPRINT_BOXES)
    else:
        boxes = list(under_boxes if under_boxes is not None else UNDER_ROBOT_BOXES)
        boxes += list(self_boxes if self_boxes is not None else SELF_IGNORE_BOXES)
    for x0, y0, x1, y1 in boxes:
        labels[y0:y1, x0:x1] = SELF


def labels_to_obs_known(
    labels: np.ndarray,
    height_cm: np.ndarray,
    obs_out: np.ndarray | None = None,
    known_out: np.ndarray | None = None,
):
    """Compat shim toward old (obs, known) consumers.

    CLEAR → known=255, obs=0
    OBSTACLE → known=255, obs=height
    SELF → known=0, obs=0   (honest: not sensed clear, not obstacle)
    UNKNOWN → known=0, obs=0
    """
    known = known_out if known_out is not None else np.zeros(labels.shape, dtype=np.uint8)
    obs = obs_out if obs_out is not None else np.zeros(labels.shape, dtype=np.uint8)
    known.fill(0)
    obs.fill(0)
    clear = labels == CLEAR
    obstacle = labels == OBSTACLE
    known[clear | obstacle] = 255
    obs[obstacle] = height_cm[obstacle]
    return obs, known
