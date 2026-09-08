#!/usr/bin/env python3
"""Unit tests for gated planner feed (ego / evidence → obs,known). No hardware."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import BODY_BOX, FRAME_H, FRAME_W, FOOTPRINT_BOXES, RCX, RCY
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.evidence_map import EvidenceMap
from perception.planner_feed import (
    apply_self_honest,
    ego_labels_to_planner_feed,
    evidence_to_ego_obs_known,
    select_planner_feed,
)


def test_self_not_clear_not_obs_in_planner_feed():
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    x0, y0, x1, y1 = BODY_BOX
    labels[y0:y1, x0:x1] = SELF
    labels[10:20, 200:210] = CLEAR
    labels[30:40, 220:230] = OBSTACLE
    height[30:40, 220:230] = 40

    obs, known = ego_labels_to_planner_feed(labels, height)
    assert np.all(obs[y0:y1, x0:x1] == 0), "SELF must not be obstacles"
    assert np.all(known[y0:y1, x0:x1] == 0), "SELF must not be known-clear"
    assert np.all(known[10:20, 200:210] == 255)
    assert np.all(obs[10:20, 200:210] == 0)
    assert np.all(obs[30:40, 220:230] == 40)
    assert np.all(known[30:40, 220:230] == 255)

    # Belt: even if shim lied, apply_self_honest fixes footprint
    obs2 = obs.copy()
    known2 = known.copy()
    obs2[y0:y1, x0:x1] = 99
    known2[y0:y1, x0:x1] = 255
    apply_self_honest(obs2, known2)
    assert np.all(obs2[y0:y1, x0:x1] == 0)
    assert np.all(known2[y0:y1, x0:x1] == 0)


def test_select_legacy_when_ungated():
    legacy_obs = np.ones((FRAME_H, FRAME_W), dtype=np.uint8) * 7
    legacy_known = np.ones((FRAME_H, FRAME_W), dtype=np.uint8) * 255
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels[RCY, RCX + 40] = OBSTACLE
    height[RCY, RCX + 40] = 20

    obs, known, source = select_planner_feed(
        gated=False,
        ego_labels=labels,
        ego_height=height,
        legacy_obs=legacy_obs,
        legacy_known=legacy_known,
    )
    assert source == "legacy"
    assert obs is legacy_obs and known is legacy_known


def test_select_ego_when_gated():
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    x0, y0, x1, y1 = BODY_BOX
    labels[y0:y1, x0:x1] = SELF
    labels[RCY, RCX + 50] = OBSTACLE
    height[RCY, RCX + 50] = 33
    out_o = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    out_k = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    obs, known, source = select_planner_feed(
        gated=True,
        ego_labels=labels,
        ego_height=height,
        prefer_evidence=False,
        obs_out=out_o,
        known_out=out_k,
    )
    assert source == "ego"
    assert obs is out_o and known is out_k
    assert out_o[RCY, RCX + 50] == 33
    assert out_k[RCY, RCX + 50] == 255
    assert np.all(out_o[y0:y1, x0:x1] == 0)
    assert np.all(out_k[y0:y1, x0:x1] == 0)


def test_select_evidence_prefers_warped_map():
    em = EvidenceMap()
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    r, c = RCY, RCX + 40
    labels[r, c] = OBSTACLE
    height[r, c] = 25
    em.update(labels, height, (0.0, 0.0, 0.0), frame_i=1)
    assert em.frame_i == 1

    # Ego labels empty this call — evidence should still win when preferred
    empty = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    out_o = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    out_k = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    obs, known, source = select_planner_feed(
        gated=True,
        ego_labels=None,
        ego_height=None,
        evidence_map=em,
        pose_xy_theta=(0.0, 0.0, 0.0),
        prefer_evidence=True,
        obs_out=out_o,
        known_out=out_k,
    )
    assert source == "evidence"
    # Obstacle near (r,c) should appear in ego after identity-ish warp
    assert np.any(out_o > 0), "evidence obstacle should project into ego obs"
    assert np.any(out_k == 255)
    # Footprint must stay honest after warp
    for x0, y0, x1, y1 in FOOTPRINT_BOXES:
        assert np.all(out_o[y0:y1, x0:x1] == 0)
        assert np.all(out_k[y0:y1, x0:x1] == 0)


def test_evidence_warp_self_honest_under_chassis():
    em = EvidenceMap()
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    # CLEAR far ahead only (not under body)
    labels[RCY - 5:RCY + 5, RCX + 60:RCX + 80] = CLEAR
    em.update(labels, height, (0.0, 0.0, 0.0), frame_i=1)

    obs, known = evidence_to_ego_obs_known(em, (0.0, 0.0, 0.0))
    x0, y0, x1, y1 = BODY_BOX
    assert np.all(obs[y0:y1, x0:x1] == 0)
    assert np.all(known[y0:y1, x0:x1] == 0), "never known-clear under chassis"
    assert np.any(known == 255), "clear evidence should appear somewhere in ego"


if __name__ == "__main__":
    test_self_not_clear_not_obs_in_planner_feed()
    print("OK self_honest_conversion")
    test_select_legacy_when_ungated()
    print("OK select_legacy")
    test_select_ego_when_gated()
    print("OK select_ego")
    test_select_evidence_prefers_warped_map()
    print("OK select_evidence")
    test_evidence_warp_self_honest_under_chassis()
    print("OK evidence_self_honest")
    print("ALL PASS")
