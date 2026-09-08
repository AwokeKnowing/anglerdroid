"""Fuse RS2 forward maps into RS1 ego labels without inventing CLEAR.

RS2 may add OBSTACLE freely (outside SELF). RS2 CLEAR is accepted only
inside the forward cone and free-range mask, never inside the footprint,
and never over an existing OBSTACLE. SELF is painted last and always wins.

See docs/perception/CONTRACT.md delivery step 2.
"""
from __future__ import annotations

import numpy as np

from robot_config import FRAME_H, FRAME_W, FOOTPRINT_BOXES, UNDER_ROBOT_BOXES

from .labels import UNKNOWN, SELF, CLEAR, OBSTACLE


def _blit(dst: np.ndarray, src: np.ndarray, dx: int, dy: int = 0) -> None:
    """Copy src into dst with pixel offset (dx, dy). Clipped, no wrap."""
    h, w = dst.shape[:2]
    if dy >= 0:
        sr0, sr1, dr0, dr1 = 0, h - dy, dy, h
    else:
        sr0, sr1, dr0, dr1 = -dy, h, 0, h + dy
    if dx >= 0:
        sc0, sc1, dc0, dc1 = 0, w - dx, dx, w
    else:
        sc0, sc1, dc0, dc1 = -dx, w, 0, w + dx
    if sr0 >= sr1 or sc0 >= sc1:
        return
    dst[dr0:dr1, dc0:dc1] = src[sr0:sr1, sc0:sc1]


def _paint_self(labels: np.ndarray, boxes=None) -> None:
    for x0, y0, x1, y1 in (boxes if boxes is not None else FOOTPRINT_BOXES):
        labels[y0:y1, x0:x1] = SELF


def fuse_rs2_into_ego(
    labels_rs1: np.ndarray,
    height_rs1: np.ndarray,
    obs2: np.ndarray,
    known2: np.ndarray,
    fw_dx: int,
    fw_dy: int = 0,
    *,
    fw_cone: np.ndarray | None = None,
    free_range: np.ndarray | None = None,
    footprint_boxes=None,
    under_boxes=None,
    labels_out: np.ndarray | None = None,
    height_out: np.ndarray | None = None,
    work_obs: np.ndarray | None = None,
    work_known: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Merge RS2 (obs,known) after rot90 into RS1 ego labels.

    Parameters
    ----------
    obs2, known2 : (H,W) uint8
        RS2 maps after ``np.rot90(..., k=-1)``, before fw offset.
    fw_dx, fw_dy : int
        Same offsets as vision depth_combine (td + FW_TD_X_DELTA, FW_Y_OFFSET).
    fw_cone : optional uint8 mask in ego frame — RS2 CLEAR/OBSTACLE only inside.
    free_range : optional uint8 mask — RS2 CLEAR only inside (obstacles still full cone).

    Returns
    -------
    labels, height_cm, metrics
    """
    h, w = labels_rs1.shape
    labels = labels_out if labels_out is not None else np.empty((h, w), dtype=np.uint8)
    height = height_out if height_out is not None else np.empty((h, w), dtype=np.uint8)
    np.copyto(labels, labels_rs1)
    np.copyto(height, height_rs1)

    obs_e = work_obs if work_obs is not None else np.zeros((h, w), dtype=np.uint8)
    kn_e = work_known if work_known is not None else np.zeros((h, w), dtype=np.uint8)
    obs_e.fill(0)
    kn_e.fill(0)
    _blit(obs_e, obs2, int(fw_dx), int(fw_dy))
    _blit(kn_e, known2, int(fw_dx), int(fw_dy))

    if fw_cone is not None:
        np.bitwise_and(obs_e, fw_cone, out=obs_e)
        np.bitwise_and(kn_e, fw_cone, out=kn_e)

    # Count would-be false CLEAR under chassis from RS2 before SELF wins.
    under = under_boxes if under_boxes is not None else UNDER_ROBOT_BOXES
    rs2_clear_under = 0
    for x0, y0, x1, y1 in under:
        rs2_clear_under += int(
            np.count_nonzero((kn_e[y0:y1, x0:x1] == 255) & (obs_e[y0:y1, x0:x1] == 0))
        )

    # Obstacles: max height; never leave as CLEAR.
    obs_m = obs_e > 0
    if np.any(obs_m):
        labels[obs_m] = OBSTACLE
        np.maximum(height, obs_e, out=height)

    # CLEAR only from RS2 known∩¬obs, range-limited, never over OBSTACLE/SELF.
    clear_m = (kn_e == 255) & (obs_e == 0)
    if free_range is not None:
        clear_m &= free_range > 0
    # Do not invent CLEAR over existing obstacle evidence from RS1.
    clear_m &= labels != OBSTACLE
    clear_m &= labels != SELF
    labels[clear_m] = CLEAR
    # height stays 0 on clear

    boxes = footprint_boxes if footprint_boxes is not None else FOOTPRINT_BOXES
    _paint_self(labels, boxes)
    height[labels == SELF] = 0
    height[labels == CLEAR] = 0
    height[labels == UNKNOWN] = 0

    n_self = int(np.count_nonzero(labels == SELF))
    n_clear = int(np.count_nonzero(labels == CLEAR))
    n_obs = int(np.count_nonzero(labels == OBSTACLE))
    n_unk = int(labels.size - n_self - n_clear - n_obs)
    metrics = {
        "rs2_clear_under_pre": rs2_clear_under,
        "rs2_obs_px": int(np.count_nonzero(obs_m)),
        "rs2_clear_accepted": int(np.count_nonzero(clear_m)),
        "labels_U": n_unk,
        "labels_S": n_self,
        "labels_C": n_clear,
        "labels_O": n_obs,
    }
    return labels, height, metrics
