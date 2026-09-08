#!/usr/bin/env python3
"""Unit tests for SLAM/VO dynamic mask (CONTRACT step 4 wedge)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, BODY_BOX, RCX, RCY
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.dynamic_mask import (
    build_slam_outlier_mask, apply_mask_to_obs, mask_counts, mask_as_uint8,
    apply_ignore_to_gray, build_forward_ignore_from_verts,
)
from perception import build_slam_outlier_mask as exported


def _blank():
    return np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)


def test_self_always_masked():
    labels = _blank()
    x0, y0, x1, y1 = BODY_BOX
    labels[y0:y1, x0:x1] = SELF
    labels[10:20, 10:20] = CLEAR
    labels[30:40, 30:40] = OBSTACLE
    labels[50:55, 50:55] = UNKNOWN
    mask = build_slam_outlier_mask(labels)
    assert np.all(mask[y0:y1, x0:x1]), "SELF must be masked"
    assert not np.any(mask[10:20, 10:20]), "CLEAR must not be masked (static prior case)"
    assert not np.any(mask[30:40, 30:40]), "OBSTACLE not masked without ephemeral prior"
    assert not np.any(mask[50:55, 50:55]), "UNKNOWN must not be masked"
    assert exported is build_slam_outlier_mask


def test_clear_unknown_not_wrongly_masked():
    labels = _blank()
    labels[:, :] = CLEAR
    mask = build_slam_outlier_mask(labels)
    assert not np.any(mask), "all-CLEAR must yield empty mask"
    labels[:, :] = UNKNOWN
    mask2 = build_slam_outlier_mask(labels)
    assert not np.any(mask2), "all-UNKNOWN must yield empty mask"


def test_ephemeral_obstacle_vs_prior():
    labels = _blank()
    labels[20:30, 100:110] = OBSTACLE  # no prior → ephemeral
    labels[40:50, 120:130] = OBSTACLE  # has prior → keep for static SLAM
    prior = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    prior[40:50, 120:130] = 1
    # Without mask_ephemeral / prior → SELF-only semantics (no obs masked)
    mask0 = build_slam_outlier_mask(labels)
    assert not np.any(mask0[20:30, 100:110])
    assert not np.any(mask0[40:50, 120:130])
    # With ephemeral + prior
    mask = build_slam_outlier_mask(
        labels, prior_obstacle=prior, mask_ephemeral=True)
    assert np.all(mask[20:30, 100:110]), "ephemeral OBSTACLE must be masked"
    assert not np.any(mask[40:50, 120:130]), "prior OBSTACLE must remain trusted"
    # mask_ephemeral True but prior None → SELF-only (API contract)
    labels2 = _blank()
    labels2[5:10, 5:10] = SELF
    labels2[20:25, 20:25] = OBSTACLE
    mask_np = build_slam_outlier_mask(
        labels2, prior_obstacle=None, mask_ephemeral=True)
    assert np.all(mask_np[5:10, 5:10])
    assert not np.any(mask_np[20:25, 20:25])


def test_apply_mask_to_obs_inplace_out():
    obs = _blank()
    obs[10:20, 10:20] = 40
    obs[30:40, 30:40] = 25
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    mask[10:20, 10:20] = True
    out = np.zeros_like(obs)
    apply_mask_to_obs(obs, mask, obs_out=out)
    assert np.all(out[10:20, 10:20] == 0)
    assert np.all(out[30:40, 30:40] == 25)
    assert np.all(obs[10:20, 10:20] == 40), "source obs must be unchanged"
    # in-place style when obs_out is a copy buffer already filled
    apply_mask_to_obs(obs, mask, obs_out=obs)  # mutate allowed when caller passes same
    assert np.all(obs[10:20, 10:20] == 0)


def test_prealloc_out_and_uint8():
    from robot_config import RCX, RCY
    labels = _blank()
    labels[RCY, RCX] = SELF
    out = np.ones((FRAME_H, FRAME_W), dtype=bool)  # dirty
    mask = build_slam_outlier_mask(labels, out=out)
    assert mask is out
    assert mask[RCY, RCX]
    assert mask.sum() == 1
    u8 = mask_as_uint8(mask)
    assert u8.dtype == np.uint8 and u8[RCY, RCX] == 255
    assert u8.sum() == 255


def test_hotpath_budget_320x240():
    labels = _blank()
    x0, y0, x1, y1 = BODY_BOX
    labels[y0:y1, x0:x1] = SELF
    labels[0:80, 0:80] = CLEAR
    labels[80:160, 80:160] = OBSTACLE
    prior = (labels == OBSTACLE).astype(np.uint8)
    prior[80:100, 80:100] = 0  # some ephemeral
    out = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    # warmup
    for _ in range(3):
        build_slam_outlier_mask(
            labels, prior_obstacle=prior, mask_ephemeral=True, out=out)
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        build_slam_outlier_mask(
            labels, prior_obstacle=prior, mask_ephemeral=True, out=out)
    ms = (time.perf_counter() - t0) * 1000.0 / n
    assert ms < 5.0, "dynamic mask too slow: %.2f ms (budget ~few ms)" % ms


def test_shape_mismatch_raises():
    labels = _blank()
    bad = np.zeros((10, 10), dtype=bool)
    try:
        build_slam_outlier_mask(labels, out=bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mask_counts_self_and_ephemeral():
    labels = _blank()
    x0, y0, x1, y1 = BODY_BOX
    labels[y0:y1, x0:x1] = SELF
    labels[20:30, 100:110] = OBSTACLE  # ephemeral
    labels[40:50, 120:130] = OBSTACLE  # static prior
    prior = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    prior[40:50, 120:130] = 1
    mask = build_slam_outlier_mask(
        labels, prior_obstacle=prior, mask_ephemeral=True)
    self_n, eph_n, tot_n = mask_counts(
        labels, mask, prior_obstacle=prior, mask_ephemeral=True)
    expect_self = int((x1 - x0) * (y1 - y0))
    expect_eph = 10 * 10
    assert self_n == expect_self, (self_n, expect_self)
    assert eph_n == expect_eph, (eph_n, expect_eph)
    assert tot_n == self_n + eph_n, (tot_n, self_n, eph_n)
    # no ephemeral when gate off
    mask0 = build_slam_outlier_mask(labels)
    s0, e0, t0 = mask_counts(labels, mask0)
    assert e0 == 0 and t0 == s0 == expect_self




def test_apply_ignore_to_gray_zeros_only_masked():
    gray = np.arange(FRAME_H * FRAME_W, dtype=np.uint8).reshape(FRAME_H, FRAME_W)
    src = gray.copy()
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    mask[10:20, 30:40] = True
    out = np.zeros_like(gray)
    apply_ignore_to_gray(gray, mask, out=out)
    assert np.all(out[10:20, 30:40] == 0)
    assert np.all(out[~mask] == src[~mask])
    assert np.all(gray == src), "source gray must be unchanged when out separate"
    # shape mismatch
    try:
        apply_ignore_to_gray(gray, np.zeros((10, 10), dtype=bool))
        assert False, "expected ValueError"
    except ValueError:
        pass




def test_build_forward_ignore_marks_hit_pixel():
    """Synthetic vert that maps into masked ego cell must mark some gray px."""
    import math, cv2
    # Mirror vision FW constants (keep test self-contained / no vision import).
    pitch = math.radians(25.6 - 90.0)
    rot, _ = cv2.Rodrigues(np.float64([pitch, 0, 0]))
    rot = rot.astype(np.float32)
    piv = np.array([0.0, -1.0, 0.02], dtype=np.float32)
    trans = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    scatter_h, scatter_w = FRAME_W, FRAME_H
    scale = 100.0
    offset = np.float32([scatter_w / 2.0, scatter_h / 2.0 + scale])
    fw_dx = -75 + 132  # TD_X_OFFSET + FW_TD_X_DELTA as in vision defaults
    fw_dy = -1

    # Build a dense-enough fake grid (283x160 like GPU default) of zeros,
    # then place one valid forward point that should land in-bounds.
    gh, gw = 160, 283
    verts = np.zeros((gh * gw, 3), dtype=np.float32)
    # Pick a camera-frame point ~1m ahead, slightly down
    verts[gh // 2 * gw + gw // 2] = np.array([0.0, 0.05, 1.0], dtype=np.float32)

    # Compute expected ego cell with same math as helper (GPU-matched)
    p = verts[gh // 2 * gw + gw // 2].copy()
    r = (p - piv) @ rot + piv - trans
    sx = int(np.floor(r[0] * scale + offset[0]))
    sy = int(np.floor(r[1] * scale + offset[1]))
    if not (0 <= sx < scatter_w and 0 <= sy < scatter_h):
        print("skip_geom sx,sy", sx, sy, "r", r)
        return
    ei = sx + fw_dy
    ej = (scatter_h - 1 - sy) + fw_dx
    if not (0 <= ei < FRAME_H and 0 <= ej < FRAME_W):
        print("skip_ego", ei, ej)
        return
    ego = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    ego[ei, ej] = True
    ign, ns, nh, ng = build_forward_ignore_from_verts(
        verts, ego,
        rotation=rot, pivot=piv, translation=trans,
        scale=scale, offset=offset,
        scatter_h=scatter_h, scatter_w=scatter_w,
        fw_dx=fw_dx, fw_dy=fw_dy,
        gray_h=FRAME_H, gray_w=FRAME_W,
        stride=1, stamp=1)
    assert ns >= 1
    assert nh == 1, (nh, sx, sy, ei, ej)
    assert ng > 0
    # Unmasked ego → no hits
    ego2 = np.zeros_like(ego)
    ign2, ns2, nh2, ng2 = build_forward_ignore_from_verts(
        verts, ego2,
        rotation=rot, pivot=piv, translation=trans,
        scale=scale, offset=offset,
        scatter_h=scatter_h, scatter_w=scatter_w,
        fw_dx=fw_dx, fw_dy=fw_dy,
        gray_h=FRAME_H, gray_w=FRAME_W,
        stride=1, stamp=1)
    assert nh2 == 0 and ng2 == 0


def test_ephemeral_vo_ignore_hit_and_gray_zeros():
    """Inject ephemeral OBSTACLE vs prior → VO ignore hit>0 + gray zeros.

    Movers are unavailable on this fire; synthetic verts stand in for a
    person/dog blob so apply_ignore_to_gray / build_forward_ignore_from_verts
    exercise the ephemeral path without waiting for live movers.
    """
    import math, cv2
    pitch = math.radians(25.6 - 90.0)
    rot, _ = cv2.Rodrigues(np.float64([pitch, 0, 0]))
    rot = rot.astype(np.float32)
    piv = np.array([0.0, -1.0, 0.02], dtype=np.float32)
    trans = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    scatter_h, scatter_w = FRAME_W, FRAME_H
    scale = 100.0
    offset = np.float32([scatter_w / 2.0, scatter_h / 2.0 + scale])
    fw_dx = -75 + 132
    fw_dy = -1

    def _ego_of(p):
        p = np.asarray(p, dtype=np.float32)
        r = (p - piv) @ rot + piv - trans
        sx = int(np.floor(r[0] * scale + offset[0]))
        sy = int(np.floor(r[1] * scale + offset[1]))
        ei = sx + fw_dy
        ej = (scatter_h - 1 - sy) + fw_dx
        return sx, sy, ei, ej

    # Ephemeral mover ~1.1m ahead-left; static furniture ~1.1m ahead-right.
    p_eph = np.array([-0.35, -0.05, 1.1], dtype=np.float32)
    p_static = np.array([0.35, -0.05, 1.1], dtype=np.float32)
    _, _, ei_e, ej_e = _ego_of(p_eph)
    _, _, ei_s, ej_s = _ego_of(p_static)
    assert 0 <= ei_e < FRAME_H and 0 <= ej_e < FRAME_W, (ei_e, ej_e)
    assert 0 <= ei_s < FRAME_H and 0 <= ej_s < FRAME_W, (ei_s, ej_s)

    labels = _blank()
    # Small OBSTACLE patches around projected cells
    for ei, ej, val in (
        (ei_e, ej_e, OBSTACLE),
        (ei_s, ej_s, OBSTACLE),
    ):
        labels[max(0, ei - 2):ei + 3, max(0, ej - 2):ej + 3] = val
    prior = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    prior[max(0, ei_s - 2):ei_s + 3, max(0, ej_s - 2):ej_s + 3] = 1

    mask = build_slam_outlier_mask(
        labels, prior_obstacle=prior, mask_ephemeral=True)
    assert np.any(mask[max(0, ei_e - 2):ei_e + 3, max(0, ej_e - 2):ej_e + 3]), (
        "ephemeral patch must be masked")
    assert not np.any(
        mask[max(0, ei_s - 2):ei_s + 3, max(0, ej_s - 2):ej_s + 3]), (
        "prior/static OBSTACLE must stay trusted for SLAM/VO")

    gh, gw = 160, 283
    verts = np.zeros((gh * gw, 3), dtype=np.float32)
    # Place both points on the fake depth grid
    i_eph = (gh // 2) * gw + (gw // 3)
    i_static = (gh // 2) * gw + (2 * gw // 3)
    verts[i_eph] = p_eph
    verts[i_static] = p_static

    common = dict(
        rotation=rot, pivot=piv, translation=trans,
        scale=scale, offset=offset,
        scatter_h=scatter_h, scatter_w=scatter_w,
        fw_dx=fw_dx, fw_dy=fw_dy,
        gray_h=FRAME_H, gray_w=FRAME_W,
        stride=1, stamp=1,
    )
    ign, ns, nh, ng = build_forward_ignore_from_verts(verts, mask, **common)
    assert ns >= 2, ns
    assert nh >= 1, (nh, "ephemeral vert must hit ignore mask")
    assert ng > 0, ng

    gray = np.full((FRAME_H, FRAME_W), 180, dtype=np.uint8)
    out = apply_ignore_to_gray(gray, ign)
    assert int(np.count_nonzero(out[ign] == 0)) == int(ng)
    assert np.all(out[~ign] == 180), "unmasked gray must stay intact"
    # Static-only mask → ephemeral vert alone would still miss; prove prior path
    # does not zero the static vert's gray when only ephemeral is masked.
    # Rebuild mask SELF-only (no ephemeral) — hits must be 0 for these verts
    # if neither lands in SELF (BODY_BOX). Our patches are away from axle.
    mask_self = build_slam_outlier_mask(labels)  # no ephemeral
    # Clear any accidental SELF overlap in labels for this check
    ign2, ns2, nh2, ng2 = build_forward_ignore_from_verts(
        verts, mask_self, **common)
    assert nh2 == 0 and ng2 == 0, (
        "without ephemeral, synthetic verts must not hit SELF-only mask",
        nh2, ng2)

    # Honesty: apply_ignore never invents CLEAR — only zeros gray
    assert out.dtype == gray.dtype
    assert not np.any(out[ign] != 0)


def test_build_forward_ignore_shape_and_empty():
    # Empty verts → empty ignore
    ego = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    ego[RCY, RCX] = True
    rot = np.eye(3, dtype=np.float32)
    piv = np.zeros(3, dtype=np.float32)
    trans = np.zeros(3, dtype=np.float32)
    off = np.float32([FRAME_H / 2.0, FRAME_W / 2.0])
    out = np.ones((FRAME_H, FRAME_W), dtype=bool)
    ign, ns, nh, ng = build_forward_ignore_from_verts(
        np.zeros((0, 3), dtype=np.float32), ego,
        rotation=rot, pivot=piv, translation=trans,
        scale=100.0, offset=off,
        scatter_h=FRAME_W, scatter_w=FRAME_H,
        gray_h=FRAME_H, gray_w=FRAME_W,
        out=out)
    assert ign is out and not np.any(out)
    assert ns == nh == ng == 0

if __name__ == "__main__":
    test_self_always_masked()
    print("OK self_masked")
    test_clear_unknown_not_wrongly_masked()
    print("OK clear_unknown")
    test_ephemeral_obstacle_vs_prior()
    print("OK ephemeral")
    test_apply_mask_to_obs_inplace_out()
    print("OK apply_obs")
    test_prealloc_out_and_uint8()
    print("OK prealloc")
    test_hotpath_budget_320x240()
    print("OK budget")
    test_shape_mismatch_raises()
    print("OK shape")
    test_mask_counts_self_and_ephemeral()
    print("OK mask_counts")
    test_apply_ignore_to_gray_zeros_only_masked()
    print("OK apply_ignore_gray")
    test_build_forward_ignore_marks_hit_pixel()
    print("OK forward_ignore_hit")
    test_ephemeral_vo_ignore_hit_and_gray_zeros()
    print("OK ephemeral_vo_ignore")
    test_build_forward_ignore_shape_and_empty()
    print("OK forward_ignore_empty")
    print("ALL PASS")
