#!/usr/bin/env python3
"""Unit tests for SLAM/VO dynamic mask (CONTRACT step 4 wedge)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, BODY_BOX
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.dynamic_mask import (
    build_slam_outlier_mask, apply_mask_to_obs, mask_counts, mask_as_uint8,
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
    print("ALL PASS")
