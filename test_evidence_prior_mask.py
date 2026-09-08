#!/usr/bin/env python3
"""EvidenceMap obstacle prior → SLAM dynamic mask (CONTRACT step 4 wire)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from globalmap import ORIGIN_X, ORIGIN_Y
from robot_config import FRAME_H, FRAME_W, RCX, RCY
from perception.labels import SELF, OBSTACLE
from perception.evidence_map import EvidenceMap
from perception.planner_feed import evidence_obstacle_prior_ego
from perception.dynamic_mask import build_slam_outlier_mask


def test_prior_ego_warps_obstacle_blob():
    ev = EvidenceMap()  # default MAP geometry (ORIGIN at axle for pose 0)
    gy, gx = int(ORIGIN_Y), int(ORIGIN_X)
    # Static blob at world origin (= ego axle at pose 0,0,0)
    ev.obstacle_evidence[gy - 3:gy + 4, gx - 3:gx + 4] = 5.0
    ev._frame_i = 1
    prior = evidence_obstacle_prior_ego(ev, (0.0, 0.0, 0.0))
    assert prior.shape == (FRAME_H, FRAME_W)
    assert prior.dtype == np.uint8
    # Axle neighborhood in ego should see the prior
    assert int(np.count_nonzero(prior[RCY - 8:RCY + 8, RCX - 8:RCX + 8])) > 0, (
        "warped prior empty near ego axle")


def test_ephemeral_vs_evidence_prior():
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels[20:30, 100:110] = OBSTACLE  # ephemeral
    labels[40:50, 120:130] = OBSTACLE  # static
    labels[RCY - 5:RCY + 5, RCX - 5:RCX + 5] = SELF
    prior = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    prior[40:50, 120:130] = 1
    mask = build_slam_outlier_mask(
        labels, prior_obstacle=prior, mask_ephemeral=True)
    assert np.all(mask[20:30, 100:110]), "ephemeral must mask"
    assert not np.any(mask[40:50, 120:130]), "prior obstacle stays"
    assert np.all(mask[RCY - 5:RCY + 5, RCX - 5:RCX + 5]), "SELF always masked"


def test_helper_out_prealloc_and_empty_map():
    ev = EvidenceMap()
    out = np.ones((FRAME_H, FRAME_W), dtype=np.uint8)
    prior = evidence_obstacle_prior_ego(ev, (0.0, 0.0, 0.0), out=out)
    assert prior is out
    assert not np.any(prior), "empty evidence → empty prior"


if __name__ == "__main__":
    test_prior_ego_warps_obstacle_blob()
    print("OK prior_warp")
    test_ephemeral_vs_evidence_prior()
    print("OK ephemeral")
    test_helper_out_prealloc_and_empty_map()
    print("OK prealloc_empty")
    print("ALL PASS")
