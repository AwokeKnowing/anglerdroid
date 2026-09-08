"""Optional GPU-resident ego label path (CuPy).

Provides GPU scatter for label_rs1_ego when CuPy is available and
KEVIN_GPU_SCATTER=1 flag is set. CPU path uses original ego_rs1.label_rs1_ego
(already well-optimized with NumPy fancy indexing and np.maximum.at).

GPU path keeps verts → projection → scatter → rotate → blit on GPU to avoid
host-device transfer overhead. Requires CuPy installed.

Also provides GPU-resident fuse_rs2_into_ego_gpu behind KEVIN_GPU_FUSE=1
(default off). CPU path (fuse.fuse_rs2_into_ego) unchanged.

See docs/perception/CONTRACT.md step 4 (CAPTURE Hz reclaim).
"""
from __future__ import annotations

import os
import numpy as np

# Optional CuPy import — do not require at module load
_CUPY_AVAILABLE = False
_cp = None
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
    _cp = cp
except ImportError:
    pass

KEVIN_GPU_SCATTER = os.environ.get("KEVIN_GPU_SCATTER", "0") == "1"
KEVIN_GPU_FUSE = os.environ.get("KEVIN_GPU_FUSE", "0") == "1"


def label_rs1_ego_gpu(
    verts,
    *,
    out_h: int,
    out_w: int,
    floor_clip_m: float,
    px_size: float,
    labels_out: np.ndarray,
    height_out: np.ndarray,
    x_offset: int = 0,
    self_boxes=None,
):
    """GPU-resident label_rs1_ego using CuPy.

    Keeps entire pipeline on GPU: verts → project → scatter → rotate → blit.
    Requires CuPy. Falls back to CPU (ego_rs1.label_rs1_ego) if CuPy unavailable.

    Parameters
    ----------
    Same as ego_rs1.label_rs1_ego.

    Returns
    -------
    labels, height : CPU numpy arrays (same as original).

    Notes
    -----
    GPU path avoids host-device transfer for intermediate arrays. Useful on Orin
    when CuPy is installed. CPU path (original ego_rs1.label_rs1_ego) is already
    well-optimized with NumPy fancy indexing — do not replace it with slower code.
    """
    from .labels import CLEAR, OBSTACLE, SELF

    if not _CUPY_AVAILABLE or _cp is None:
        # Fall back to CPU
        from .ego_rs1 import label_rs1_ego
        return label_rs1_ego(
            verts,
            out_h=out_h,
            out_w=out_w,
            floor_clip_m=floor_clip_m,
            px_size=px_size,
            labels_out=labels_out,
            height_out=height_out,
            x_offset=x_offset,
            self_boxes=self_boxes,
        )

    # GPU path: keep everything on device until final transfer
    cam_l_gpu = _cp.zeros((out_h, out_w), dtype=_cp.uint8)
    cam_h_gpu = _cp.zeros((out_h, out_w), dtype=_cp.uint8)

    if verts is not None and len(verts) > 0:
        v_gpu = _cp.asarray(verts, dtype=_cp.float32)
        z_gpu = v_gpu[:, 2]
        valid_gpu = z_gpu > 0.01

        if _cp.any(valid_gpu):
            scale = _cp.float32(1.0 / px_size)
            center_gpu = _cp.array([out_w * 0.5, out_h * 0.5], dtype=_cp.float32)
            vv_gpu = v_gpu[valid_gpu]
            p_gpu = vv_gpu[:, :2] * scale + center_gpu
            ja_gpu, ia_gpu = p_gpu.astype(_cp.uint32).T
            ma_gpu = (ia_gpu < _cp.uint32(out_h)) & (ja_gpu < _cp.uint32(out_w))
            ia_m_gpu = ia_gpu[ma_gpu]
            ja_m_gpu = ja_gpu[ma_gpu]
            zv_gpu = vv_gpu[ma_gpu, 2]

            # Scatter CLEAR
            floor_gpu = zv_gpu >= floor_clip_m
            cam_l_gpu[ia_m_gpu[floor_gpu], ja_m_gpu[floor_gpu]] = CLEAR

            # Scatter OBSTACLE
            obs_gpu = ~floor_gpu
            if _cp.any(obs_gpu):
                ia_o_gpu = ia_m_gpu[obs_gpu]
                ja_o_gpu = ja_m_gpu[obs_gpu]
                h_gpu = _cp.clip(
                    ((floor_clip_m - zv_gpu[obs_gpu]) * 100.0).astype(_cp.int32), 1, 100
                ).astype(_cp.uint8)
                cam_l_gpu[ia_o_gpu, ja_o_gpu] = OBSTACLE
                _cp.maximum.at(cam_h_gpu, (ia_o_gpu, ja_o_gpu), h_gpu)

    # Rotate 180° on GPU
    cam_l_f_gpu = cam_l_gpu[::-1, ::-1]
    cam_h_f_gpu = cam_h_gpu[::-1, ::-1]

    # Blit with x_offset on GPU
    labels_gpu = _cp.zeros((out_h, out_w), dtype=_cp.uint8)
    height_gpu = _cp.zeros((out_h, out_w), dtype=_cp.uint8)
    if x_offset == 0:
        _cp.copyto(labels_gpu, cam_l_f_gpu)
        _cp.copyto(height_gpu, cam_h_f_gpu)
    else:
        _blit_x_gpu(labels_gpu, cam_l_f_gpu, int(x_offset))
        _blit_x_gpu(height_gpu, cam_h_f_gpu, int(x_offset))

    # Transfer to CPU
    _cp.copyto(labels_out, labels_gpu)
    _cp.copyto(height_out, height_gpu)

    # Paint SELF on CPU (box ops are trivial, avoid GPU transfer)
    if self_boxes is not None:
        for x0, y0, x1, y1 in self_boxes:
            labels_out[y0:y1, x0:x1] = SELF

    return labels_out, height_out


def _blit_x_gpu(dst, src, dx: int) -> None:
    """Horizontal blit on GPU (CuPy arrays)."""
    h, w = src.shape
    if dx == 0:
        _cp.copyto(dst, src)
        return
    if dx > 0:
        if dx >= w:
            return
        dst[:, dx:w] = src[:, : w - dx]
    else:
        if -dx >= w:
            return
        dst[:, : w + dx] = src[:, -dx:]


def fuse_rs2_into_ego_gpu(
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
    """GPU-resident fuse RS2 into RS1 ego labels using CuPy.

    Keeps blit → mask → merge → self on GPU. Falls back to CPU 
    (fuse.fuse_rs2_into_ego) if CuPy unavailable.

    Parameters
    ----------
    Same as fuse.fuse_rs2_into_ego.

    Returns
    -------
    labels, height, metrics : CPU numpy arrays (same as original).

    Notes
    -----
    GPU path avoids host-device transfer for intermediate arrays. Useful on Orin
    when CuPy is installed. CPU path (original fuse.fuse_rs2_into_ego) is already
    efficient — do not replace it with slower code.
    """
    from .labels import CLEAR, OBSTACLE, SELF
    from .fuse import fuse_rs2_into_ego
    from robot_config import FOOTPRINT_BOXES, UNDER_ROBOT_BOXES

    if not _CUPY_AVAILABLE or _cp is None:
        # Fall back to CPU
        return fuse_rs2_into_ego(
            labels_rs1, height_rs1,
            obs2, known2,
            fw_dx, fw_dy,
            fw_cone=fw_cone,
            free_range=free_range,
            footprint_boxes=footprint_boxes,
            under_boxes=under_boxes,
            labels_out=labels_out,
            height_out=height_out,
            work_obs=work_obs,
            work_known=work_known,
        )

    # GPU path: keep everything on device until final transfer
    h, w = labels_rs1.shape
    labels_gpu = _cp.asarray(labels_rs1, dtype=_cp.uint8)
    height_gpu = _cp.asarray(height_rs1, dtype=_cp.uint8)

    obs_e_gpu = _cp.zeros((h, w), dtype=_cp.uint8)
    kn_e_gpu = _cp.zeros((h, w), dtype=_cp.uint8)

    # Blit RS2 with offsets on GPU
    obs2_gpu = _cp.asarray(obs2, dtype=_cp.uint8)
    known2_gpu = _cp.asarray(known2, dtype=_cp.uint8)
    _blit_gpu(obs_e_gpu, obs2_gpu, int(fw_dx), int(fw_dy))
    _blit_gpu(kn_e_gpu, known2_gpu, int(fw_dx), int(fw_dy))

    # Apply cone masks on GPU
    if fw_cone is not None:
        cone_gpu = _cp.asarray(fw_cone, dtype=_cp.uint8)
        obs_e_gpu &= cone_gpu
        kn_e_gpu &= cone_gpu

    # Count would-be false CLEAR under chassis from RS2 before SELF wins
    under = under_boxes if under_boxes is not None else UNDER_ROBOT_BOXES
    rs2_clear_under = 0
    for x0, y0, x1, y1 in under:
        rs2_clear_under += int(
            _cp.count_nonzero((kn_e_gpu[y0:y1, x0:x1] == 255) & (obs_e_gpu[y0:y1, x0:x1] == 0))
        )

    # Obstacles: max height; never leave as CLEAR
    obs_m_gpu = obs_e_gpu > 0
    rs2_obs_px = int(_cp.count_nonzero(obs_m_gpu))
    labels_gpu[obs_m_gpu] = OBSTACLE
    _cp.maximum(height_gpu, obs_e_gpu, out=height_gpu)

    # CLEAR only from RS2 known∩¬obs, range-limited, never over OBSTACLE/SELF
    clear_m_gpu = (kn_e_gpu == 255) & (obs_e_gpu == 0)
    if free_range is not None:
        free_range_gpu = _cp.asarray(free_range, dtype=_cp.uint8)
        clear_m_gpu &= free_range_gpu > 0
    # Do not invent CLEAR over existing obstacle evidence from RS1
    clear_m_gpu &= labels_gpu != OBSTACLE
    clear_m_gpu &= labels_gpu != SELF
    rs2_clear_accepted = int(_cp.count_nonzero(clear_m_gpu))
    labels_gpu[clear_m_gpu] = CLEAR

    # Paint SELF on GPU
    boxes = footprint_boxes if footprint_boxes is not None else FOOTPRINT_BOXES
    for x0, y0, x1, y1 in boxes:
        labels_gpu[y0:y1, x0:x1] = SELF

    # Zero height for non-obstacle labels on GPU
    height_gpu[labels_gpu == SELF] = 0
    height_gpu[labels_gpu == CLEAR] = 0
    height_gpu[labels_gpu == 0] = 0  # UNKNOWN

    # Transfer to CPU
    if labels_out is None:
        labels_out = np.empty((h, w), dtype=np.uint8)
    if height_out is None:
        height_out = np.empty((h, w), dtype=np.uint8)
    _cp.copyto(labels_out, labels_gpu)
    _cp.copyto(height_out, height_gpu)

    # Compute final metrics on GPU then transfer
    n_self = int(_cp.count_nonzero(labels_gpu == SELF))
    n_clear = int(_cp.count_nonzero(labels_gpu == CLEAR))
    n_obs = int(_cp.count_nonzero(labels_gpu == OBSTACLE))
    n_unk = int(labels_gpu.size - n_self - n_clear - n_obs)

    metrics = {
        "rs2_clear_under_pre": rs2_clear_under,
        "rs2_obs_px": rs2_obs_px,
        "rs2_clear_accepted": rs2_clear_accepted,
        "labels_U": n_unk,
        "labels_S": n_self,
        "labels_C": n_clear,
        "labels_O": n_obs,
    }
    return labels_out, height_out, metrics


def _blit_gpu(dst, src, dx: int, dy: int = 0) -> None:
    """Blit on GPU (CuPy arrays) with (dx, dy) offset."""
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
