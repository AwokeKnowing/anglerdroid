"""Optional GPU-resident ego label path (CuPy).

Provides GPU scatter for label_rs1_ego when CuPy is available and
KEVIN_GPU_SCATTER=1 flag is set. CPU path uses original ego_rs1.label_rs1_ego
(already well-optimized with NumPy fancy indexing and np.maximum.at).

GPU path keeps verts → projection → scatter → rotate → blit on GPU to avoid
host-device transfer overhead. Requires CuPy installed.

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
