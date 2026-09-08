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
descriptor / thumb) and optionally zeros VO gray pixels whose RS2 depth
samples project into masked ego cells (``apply_ignore_to_gray`` /
``build_forward_ignore_from_verts``). Full RGB-D+wheel+IMU dynamic SLAM
remains TODO.

Live wire: when ``KEVIN_EVIDENCE_MAP=1`` and the evidence grid has
updates, vision passes a pose-warped EvidenceMap obstacle prior with
``mask_ephemeral=True`` so movers without accumulated static evidence are
excluded from VO/SLAM while persistent furniture remains. Still gated by
``KEVIN_SLAM_DYNAMIC_MASK=1`` (default off).
Live metrics lines ``slam_mask:`` / ``vo_ignore:`` report counts / ms.
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


def apply_ignore_to_gray(
    gray: np.ndarray,
    ignore_mask: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Zero gray pixels where ``ignore_mask`` is True (VO SAD ignore).

    Does not invent labels — only darkens pixels the caller marked. Prefer
    ``out`` prealloc so the live color→gray buffer is not mutated when the
    caller still needs the unmasked gray.
    """
    g = np.asarray(gray)
    m = np.asarray(ignore_mask, dtype=bool)
    if m.shape != g.shape[:2]:
        raise ValueError(
            "ignore_mask shape %s != gray shape %s" % (m.shape, g.shape[:2]))
    if out is None:
        result = np.array(g, copy=True, dtype=g.dtype)
    else:
        result = out
        if result.shape != g.shape:
            raise ValueError(
                "out shape %s != gray shape %s" % (result.shape, g.shape))
        np.copyto(result, g)
    result[m] = 0
    return result


def _estimate_depth_grid(n: int, orig_w: int = 848, orig_h: int = 480):
    """Estimate (gh, gw) for a decimated RS depth vertex buffer."""
    if n < 100:
        return 0, 0
    aspect = float(orig_w) / float(orig_h)
    w_est = int(round((n * aspect) ** 0.5))
    for w_try in (w_est, w_est - 1, w_est + 1, w_est + 2, w_est - 2):
        if w_try > 0 and n % w_try == 0:
            return n // w_try, w_try
    return 0, 0


def build_forward_ignore_from_verts(
    verts: np.ndarray,
    ego_ignore_mask: np.ndarray,
    *,
    rotation: np.ndarray,
    pivot: np.ndarray,
    translation: np.ndarray,
    scale: float,
    offset: np.ndarray,
    scatter_h: int,
    scatter_w: int,
    fw_dx: int = 0,
    fw_dy: int = 0,
    y_offset: float = 0.0,
    gray_h: int,
    gray_w: int,
    stride: int = 8,
    z_min: float = 0.28,
    stamp: int = 1,
    out: Optional[np.ndarray] = None,
) -> tuple:
    """Sparse RS2 verts → ego ignore → gray bool mask for VO.

    Reuses the same rigid+scale math as GPU ``_VERT_SCATTER_OBS``, then the
    same ``rot90(k=-1)`` + ``(fw_dx, fw_dy)`` blit as vision's RS2 path.
    Only marks gray pixels tied to depth samples that land in
    ``ego_ignore_mask`` (SELF / ephemeral) — never invents CLEAR.

    **Gap (documented):** color UV is a nearest remap of the decimated depth
    grid index → ``(gray_h, gray_w)``, not ``rs.align`` / stereo map_to_color.
    Sparse ``stride`` + small ``stamp`` keep Orin cost low; metrics report
    sample / hit / gray counts.

    Returns ``(ignore_gray, n_samp, n_hit, n_gray)``.
    """
    mask = np.asarray(ego_ignore_mask, dtype=bool)
    ego_h, ego_w = mask.shape
    if out is None:
        ignore = np.zeros((gray_h, gray_w), dtype=bool)
    else:
        ignore = out
        if ignore.shape != (gray_h, gray_w):
            raise ValueError(
                "out shape %s != (%d, %d)" % (ignore.shape, gray_h, gray_w))
        if ignore.dtype != np.bool_:
            raise TypeError("out must be bool dtype, got %s" % ignore.dtype)
        ignore.fill(False)

    v_all = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    n = int(v_all.shape[0])
    if n == 0 or stride < 1:
        return ignore, 0, 0, 0

    idx = np.arange(0, n, int(stride), dtype=np.int32)
    v = v_all[idx]
    z = v[:, 2]
    valid = (z >= float(z_min)) & np.isfinite(z)
    if not np.any(valid):
        return ignore, 0, 0, 0
    idx = idx[valid]
    v = v[valid]
    n_samp = int(v.shape[0])

    rot = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    piv = np.asarray(pivot, dtype=np.float32).reshape(3)
    trans = np.asarray(translation, dtype=np.float32).reshape(3)
    off = np.asarray(offset, dtype=np.float32).reshape(2)
    sc = np.float32(scale)

    # Match GPU: p.y += y_off; r = R*(p-piv)+piv-trans
    p = np.empty_like(v)
    np.copyto(p, v)
    p[:, 1] = p[:, 1] + np.float32(y_offset)
    # Match GLSL u_rot*v with moderngl column-major upload of numpy R → R.T@v
    r = (p - piv) @ rot + piv - trans
    sx = np.floor(r[:, 0] * sc + off[0]).astype(np.int32)
    sy = np.floor(r[:, 1] * sc + off[1]).astype(np.int32)
    in_sc = (
        (sx >= 0) & (sy >= 0) & (sx < int(scatter_w)) & (sy < int(scatter_h)))
    if not np.any(in_sc):
        return ignore, n_samp, 0, 0
    idx = idx[in_sc]
    sx = sx[in_sc]
    sy = sy[in_sc]

    ei = sx + int(fw_dy)
    ej = (int(scatter_h) - 1 - sy) + int(fw_dx)
    in_ego = (ei >= 0) & (ej >= 0) & (ei < ego_h) & (ej < ego_w)
    if not np.any(in_ego):
        return ignore, n_samp, 0, 0
    idx = idx[in_ego]
    sx = sx[in_ego]
    sy = sy[in_ego]
    ei = ei[in_ego]
    ej = ej[in_ego]
    hit = mask[ei, ej]
    if not np.any(hit):
        return ignore, n_samp, 0, 0
    idx = idx[hit]
    sx = sx[hit]
    sy = sy[hit]
    n_hit = int(idx.shape[0])

    gh, gw = _estimate_depth_grid(n)
    if gh > 0 and gw > 0:
        gy = idx // gw
        gx = idx - gy * gw
        cu = (gx * int(gray_w)) // gw
        cv = (gy * int(gray_h)) // gh
    else:
        cu = (sx * int(gray_w)) // int(scatter_w)
        cv = (sy * int(gray_h)) // int(scatter_h)

    st = int(stamp)
    for k in range(n_hit):
        u = int(cu[k])
        vv = int(cv[k])
        r0 = 0 if vv < st else vv - st
        r1 = gray_h if vv + st + 1 > gray_h else vv + st + 1
        c0 = 0 if u < st else u - st
        c1 = gray_w if u + st + 1 > gray_w else u + st + 1
        ignore[r0:r1, c0:c1] = True

    n_gray = int(np.count_nonzero(ignore))
    return ignore, n_samp, n_hit, n_gray
