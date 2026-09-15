"""Copied 80×60 labeler for bag-matched sim (do not import Orin vision here).

Source of truth for the algorithm: src/perception/fast_ego80.py + labels.py.
Keep this copy in sim/rl_nav so transfer work does not pull the live stack.
If live geometry changes (TD_X_OFFSET, RCX, floor clip), update both.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

UNKNOWN = 0
SELF = 1
CLEAR = 2
OBSTACLE = 3

EGO80_H = 60
EGO80_W = 80
EGO80_PX = 0.04
EGO80_RCX = 81.0 / 4.0
EGO80_RCY = 119.0 / 4.0
EGO80_X0 = -EGO80_RCX * EGO80_PX
EGO80_X1 = EGO80_X0 + EGO80_W * EGO80_PX
EGO80_Y0 = -(EGO80_H - EGO80_RCY) * EGO80_PX
EGO80_Y1 = EGO80_RCY * EGO80_PX
TD_FLOOR_CLIP = np.float32(0.91)
# Live TD_X_OFFSET = -75 px @ 1 cm → −18.75 @ 4 cm. Not a center crop of 848.
TD_X_OFFSET80 = -19

# RS2 down-pitch from horizontal. phys_h used 25.6-90 (=64.4° down) and the
# floor became a ~20° slope → red cone wedge. Side-view bag fit: 27.0° / 0.475 m.
FW_PITCH_DOWN_DEG = 27.0
_FW_PITCH = math.radians(FW_PITCH_DOWN_DEG)
FW_SIN_P = math.sin(_FW_PITCH)
FW_COS_P = math.cos(_FW_PITCH)
FW_CAM_H = 0.475
FW_OBS_H0 = 0.05
FW_OBS_H1 = 1.30
FW_CAM_X = 0.15


def downsample_labels_4(lab: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    h, w = lab.shape
    if h != 240 or w != 320:
        raise ValueError("expected 240x320 labels, got %sx%s" % (h, w))
    b = lab.reshape(60, 4, 80, 4)
    slf = (b == SELF).any(axis=(1, 3))
    obs = (b == OBSTACLE).any(axis=(1, 3))
    clr = (b == CLEAR).any(axis=(1, 3))
    dst = out if out is not None else np.zeros((60, 80), dtype=np.uint8)
    dst.fill(UNKNOWN)
    dst[clr] = CLEAR
    dst[obs] = OBSTACLE
    dst[slf] = SELF
    return dst


def labels_to_ego_float(lab: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """3-tone policy raster: UNKNOWN/SELF=0, CLEAR=0.5, OBSTACLE=1."""
    x = out if out is not None else np.zeros(lab.shape, dtype=np.float32)
    x.fill(0.0)
    x[lab == CLEAR] = 0.5
    x[lab == OBSTACLE] = 1.0
    return x


def decimate_z16(depth: np.ndarray, mag: int = 4) -> np.ndarray:
    mag = int(max(1, mag))
    h, w = depth.shape
    hh, ww = h // mag, w // mag
    b = depth[: hh * mag, : ww * mag].reshape(hh, mag, ww, mag)
    valid = b > 0
    filled = np.where(valid, b.astype(np.uint32), np.uint32(0xFFFFFFFF))
    out = filled.min(axis=(1, 3)).astype(np.uint16)
    out[out == np.uint16(0xFFFF)] = 0
    return out


def deproject_z16(depth_u16: np.ndarray, fx, fy, ppx, ppy, depth_scale=0.001):
    z = depth_u16.astype(np.float32) * np.float32(depth_scale)
    h, w = depth_u16.shape
    us = np.arange(w, dtype=np.float32)
    vs = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    m = z > 0.05
    if not np.any(m):
        return np.zeros((0, 3), dtype=np.float32)
    zz = z[m]
    xx = (uu[m] - np.float32(ppx)) / np.float32(fx) * zz
    yy = (vv[m] - np.float32(ppy)) / np.float32(fy) * zz
    return np.stack((xx, yy, zz), axis=1)


def scatter_rs1_80(
    verts: np.ndarray,
    *,
    floor_clip_m: float = float(TD_FLOOR_CLIP),
    labels_out: np.ndarray | None = None,
    x_offset: int = TD_X_OFFSET80,
) -> np.ndarray:
    lab = labels_out if labels_out is not None else np.zeros((EGO80_H, EGO80_W), dtype=np.uint8)
    lab.fill(UNKNOWN)
    if verts is None or len(verts) == 0:
        return lab
    v = np.asarray(verts, dtype=np.float32)
    z = v[:, 2]
    valid = z > 0.01
    if not np.any(valid):
        return lab
    scale = np.float32(1.0 / EGO80_PX)
    center = np.float32([EGO80_W * 0.5, EGO80_H * 0.5])
    vv = v[valid]
    p = vv[:, :2] * scale + center
    with np.errstate(invalid="ignore"):
        ja, ia = p.astype(np.uint32).T
    ma = (ia < np.uint32(EGO80_H)) & (ja < np.uint32(EGO80_W))
    ia_m, ja_m = ia[ma], ja[ma]
    zv = vv[ma, 2]
    cam = np.zeros((EGO80_H, EGO80_W), dtype=np.uint8)
    floor = zv >= np.float32(floor_clip_m)
    cam[ia_m[floor], ja_m[floor]] = CLEAR
    obs = ~floor
    if np.any(obs):
        cam[ia_m[obs], ja_m[obs]] = OBSTACLE
    cam = cam[::-1, ::-1]
    if x_offset == 0:
        np.copyto(lab, cam)
    elif x_offset > 0:
        if x_offset < EGO80_W:
            lab[:, x_offset:EGO80_W] = cam[:, : EGO80_W - x_offset]
    else:
        dx = -x_offset
        if dx < EGO80_W:
            lab[:, : EGO80_W - dx] = cam[:, dx:]
    return lab


# One hull at 4 cm/px: covers body + 10 cm wheels + mast. Live 1 cm atlas
# still uses the four robot_config boxes; here we avoid CAD wheel ears.
# Same as kernels.in_self_body: lx in [-0.16, 0.20], |ly| <= 0.22.
SELF_BOXES_CM = (
    (2.0, 0.0, 36.0, 44.0),
)
SELF_LX0, SELF_LX1 = -0.16, 0.20
SELF_LY = 0.22
THROT_MARK = 0.20


def self_mask_80() -> np.ndarray:
    m = np.zeros((EGO80_H, EGO80_W), dtype=bool)
    for cx, cy, sx, sy in SELF_BOXES_CM:
        x0 = int(np.floor(EGO80_RCX + (cx - 0.5 * sx) / 4.0))
        x1 = int(np.ceil(EGO80_RCX + (cx + 0.5 * sx) / 4.0))
        y0 = int(np.floor(EGO80_RCY - (cy + 0.5 * sy) / 4.0))
        y1 = int(np.ceil(EGO80_RCY - (cy - 0.5 * sy) / 4.0))
        m[max(0, y0):min(EGO80_H, y1), max(0, x0):min(EGO80_W, x1)] = True
    return m


def paint_self_80(labels: np.ndarray) -> np.ndarray:
    labels[self_mask_80()] = SELF
    return labels


_FOV_STENCIL = None


def fov_stencil_80():
    global _FOV_STENCIL
    if _FOV_STENCIL is None:
        p = Path(__file__).with_name("fov80_rs1.npy")
        if p.is_file():
            m = np.load(p)
            _FOV_STENCIL = m.astype(np.uint8) if m.shape == (EGO80_H, EGO80_W) else False
        else:
            _FOV_STENCIL = False
    return None if _FOV_STENCIL is False else _FOV_STENCIL


# Same body-frame house ∪ cone as kernels.py (k_obs FOV).
RS1_X0, RS1_X1 = -0.71, 1.58
RS1_Y0, RS1_Y1 = -1.10, 1.09
RS2_RANGE = 2.50
RS2_HALF = math.radians(40.0)


def fov_mask_80():
    xs = (np.arange(EGO80_W, dtype=np.float32) + 0.5) * EGO80_PX + EGO80_X0
    ys = EGO80_Y1 - (np.arange(EGO80_H, dtype=np.float32) + 0.5) * EGO80_PX
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx * xx + yy * yy)
    ang = np.arctan2(yy, xx)
    cone = (dist > 0.05) & (dist <= np.float32(RS2_RANGE)) & (np.abs(ang) <= np.float32(RS2_HALF))
    st = fov_stencil_80()
    if st is not None:
        return (st != 0) | cone
    house = (xx >= RS1_X0) & (xx <= RS1_X1) & (yy >= RS1_Y0) & (yy <= RS1_Y1)
    return house | cone


def fill_fov_like_sim(labels: np.ndarray) -> np.ndarray:
    """Match k_obs: one filled house ∪ cone, OBS stays, black outside."""
    inside = fov_mask_80()
    keep_obs = labels == OBSTACLE
    labels[~inside] = UNKNOWN
    labels[inside] = CLEAR
    labels[inside & keep_obs] = OBSTACLE
    return labels


def stamp_throttle_float(
    ego: np.ndarray,
    throt_f: float = 0.0,
    throt_b: float = 0.0,
    throt_l: float = 0.0,
    throt_r: float = 0.0,
) -> np.ndarray:
    xs = (np.arange(EGO80_W, dtype=np.float32) + 0.5) * EGO80_PX + EGO80_X0
    ys = EGO80_Y1 - (np.arange(EGO80_H, dtype=np.float32) + 0.5) * EGO80_PX
    xx, yy = np.meshgrid(xs, ys)
    hull = (xx >= SELF_LX0) & (xx <= SELF_LX1) & (np.abs(yy) <= SELF_LY)
    if float(throt_f) > 0.08:
        ego[hull & (xx >= 0.02)] = THROT_MARK
    if float(throt_b) > 0.08:
        ego[hull & (xx <= 0.02)] = THROT_MARK
    if float(throt_l) > 0.08:
        ego[hull & (yy >= 0.08)] = THROT_MARK
    if float(throt_r) > 0.08:
        ego[hull & (yy <= -0.08)] = THROT_MARK
    return ego


def fuse_rs2_cone_80(
    labels: np.ndarray,
    rs2_verts: np.ndarray | None,
    *,
    range_m: float = 2.50,
    half_rad: float = 0.6981317008,
) -> np.ndarray:
    if rs2_verts is None or len(rs2_verts) == 0:
        return labels
    v = np.asarray(rs2_verts, dtype=np.float32)
    y = v[:, 1]
    z = v[:, 2]
    phys_h = np.float32(FW_CAM_H) - y * np.float32(FW_COS_P) - z * np.float32(FW_SIN_P)
    fwd = -np.float32(FW_SIN_P) * y + np.float32(FW_COS_P) * z
    left = -v[:, 0]
    m = (fwd > 0.08) & (fwd < range_m)
    if not np.any(m):
        return labels
    fwd, left, phys_h = fwd[m], left[m], phys_h[m]
    body_x = np.float32(FW_CAM_X) + fwd
    ang = np.arctan2(left, body_x)
    m2 = np.abs(ang) <= half_rad
    if not np.any(m2):
        return labels
    body_x, left, phys_h = body_x[m2], left[m2], phys_h[m2]
    is_obs = (phys_h >= np.float32(FW_OBS_H0)) & (phys_h < np.float32(FW_OBS_H1))
    col = np.rint(EGO80_RCX + body_x / EGO80_PX).astype(np.int32)
    row = np.rint(EGO80_RCY - left / EGO80_PX).astype(np.int32)
    ok = (col >= 0) & (col < EGO80_W) & (row >= 0) & (row < EGO80_H)
    if not np.any(ok):
        return labels
    col, row, body_x, is_obs = col[ok], row[ok], body_x[ok], is_obs[ok]
    hit = np.full(EGO80_W, np.float32(range_m), dtype=np.float32)
    if np.any(is_obs):
        np.minimum.at(hit, col[is_obs], body_x[is_obs])
        labels[row[is_obs], col[is_obs]] = np.where(
            labels[row[is_obs], col[is_obs]] == SELF,
            SELF,
            OBSTACLE,
        )
    xs = (np.arange(EGO80_W, dtype=np.float32) + 0.5) * EGO80_PX + EGO80_X0
    ys = EGO80_Y1 - (np.arange(EGO80_H, dtype=np.float32) + 0.5) * EGO80_PX
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx * xx + yy * yy)
    angm = np.arctan2(yy, xx)
    cone = (dist > 0.05) & (dist <= range_m) & (np.abs(angm) <= half_rad)
    col_i = np.clip(np.rint(EGO80_RCX + xx / EGO80_PX).astype(np.int32), 0, EGO80_W - 1)
    free = cone & (dist < hit[col_i] - 0.04)
    take = free & (labels == UNKNOWN)
    labels[take] = CLEAR
    return labels


def label_frame_80(rs1_verts, rs2_verts=None, *, labels_out=None) -> np.ndarray:
    lab = scatter_rs1_80(rs1_verts, labels_out=labels_out)
    fuse_rs2_cone_80(lab, rs2_verts)
    fill_fov_like_sim(lab)
    paint_self_80(lab)
    return lab


def _scaled_intr(intr: dict | None, mag: int):
    if not intr:
        fx, fy, ppx, ppy = 380.0, 380.0, 316.0, 233.0
    else:
        fx = float(intr.get("fx", 380.0))
        fy = float(intr.get("fy", 380.0))
        ppx = float(intr.get("ppx", 316.0))
        ppy = float(intr.get("ppy", 233.0))
    m = float(max(1, mag))
    return fx / m, fy / m, ppx / m, ppy / m


def label_from_z16(
    rs1_z16,
    rs2_z16=None,
    *,
    intr1=None,
    intr2=None,
    mag: int = 4,
    labels_out=None,
) -> np.ndarray:
    v1 = None
    if rs1_z16 is not None:
        d1 = decimate_z16(rs1_z16, mag)
        fx, fy, ppx, ppy = _scaled_intr(intr1, mag)
        v1 = deproject_z16(d1, fx, fy, ppx, ppy)
    v2 = None
    if rs2_z16 is not None:
        d2 = decimate_z16(rs2_z16, mag)
        fx, fy, ppx, ppy = _scaled_intr(intr2, mag)
        v2 = deproject_z16(d2, fx, fy, ppx, ppy)
    return label_frame_80(v1, v2, labels_out=labels_out)
