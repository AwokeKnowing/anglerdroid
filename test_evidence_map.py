#!/usr/bin/env python3
"""Unit tests for EvidenceMap (CONTRACT step 3 — dynamic decay)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import FRAME_H, FRAME_W, BODY_BOX, RCX, RCY, EGO_PX_SIZE
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.evidence_map import (
    EvidenceMap, OBS_EVIDENCE_THRESH, CLEAR_EVIDENCE_THRESH,
)
from globalmap import ORIGIN_X, ORIGIN_Y, PX_SIZE, GlobalMap


def _blank_ego():
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    return labels, height


def test_self_does_not_add_obstacle():
    em = EvidenceMap()
    labels, height = _blank_ego()
    x0, y0, x1, y1 = BODY_BOX
    labels[y0:y1, x0:x1] = SELF
    labels[RCY, RCX] = SELF
    before = em.obstacle_evidence.copy()
    em.update(labels, height, (0.0, 0.0, 0.0))
    assert np.allclose(em.obstacle_evidence, before), "SELF must not raise obstacle_evidence"
    assert not np.any(em.obstacle_mask()), "no obstacles from SELF-only frame"


def test_obstacle_decays_clear_reclaims():
    em = EvidenceMap(obs_decay=0.5)  # fast decay for test
    labels, height = _blank_ego()
    r, c = RCY, RCX + 40
    labels[r, c] = OBSTACLE
    height[r, c] = 30
    em.update(labels, height, (0.0, 0.0, 0.0), frame_i=1)
    assert np.any(em.obstacle_mask()), "obstacle should register"
    obs_peak = float(em.obstacle_evidence.max())
    assert obs_peak > OBS_EVIDENCE_THRESH

    empty, eh = _blank_ego()
    for i in range(2, 12):
        em.update(empty, eh, (0.0, 0.0, 0.0), frame_i=i)
    assert float(em.obstacle_evidence.max()) < obs_peak
    assert not np.any(em.obstacle_mask()), "obstacle should decay away without re-observe"

    labels2, height2 = _blank_ego()
    labels2[r, c] = OBSTACLE
    height2[r, c] = 40
    em.update(labels2, height2, (0.0, 0.0, 0.0), frame_i=20)
    assert np.any(em.obstacle_mask())

    labels3, height3 = _blank_ego()
    labels3[r, c] = CLEAR
    reclaimed = False
    for i in range(21, 40):
        em.update(labels3, height3, (0.0, 0.0, 0.0), frame_i=i)
        if not np.any(em.obstacle_mask()) and np.any(em.drivable_mask()):
            reclaimed = True
            break
    assert reclaimed, "clear should reclaim after obstacle decays"
    assert not np.any(em.obstacle_mask()), "clear should reclaim after obstacle decays"
    assert np.any(em.drivable_mask()), "clear evidence should mark drivable"


def test_clear_does_not_invent_under_self():
    em = EvidenceMap()
    labels, height = _blank_ego()
    x0, y0, x1, y1 = BODY_BOX
    labels[:, :] = UNKNOWN
    labels[y0:y1, x0:x1] = SELF
    em.update(labels, height, (0.0, 0.0, 0.0))
    assert float(em.clear_evidence.max()) == 0.0, "SELF must not add clear_evidence"
    assert not np.any(em.drivable_mask())

    labels2, height2 = _blank_ego()
    labels2[y0:y1, x0:x1] = SELF
    labels2[10:20, 200:210] = CLEAR
    em2 = EvidenceMap()
    em2.update(labels2, height2, (0.0, 0.0, 0.0))
    M = GlobalMap._forward_affine(0.0, 0.0, 0.0, float(RCX), float(RCY), float(EGO_PX_SIZE))
    for ey in range(y0, min(y0 + 3, y1)):
        for ex in range(x0, min(x0 + 3, x1)):
            g = M @ np.array([ex, ey, 1.0], dtype=np.float64)
            gx, gy = int(round(g[0])), int(round(g[1]))
            if 0 <= gx < em2.map_w and 0 <= gy < em2.map_h:
                assert em2.clear_evidence[gy, gx] == 0.0
                assert em2.obstacle_evidence[gy, gx] == 0.0


def test_pose_transform_two_poses():
    em = EvidenceMap()
    labels, height = _blank_ego()
    r, c = RCY, RCX + 50
    labels[r, c] = CLEAR

    em.update(labels, height, (0.0, 0.0, 0.0), frame_i=1)
    cells1 = set(zip(*np.nonzero(em.clear_evidence > 0)))
    assert len(cells1) >= 1

    em.reset()
    em.update(labels, height, (1.0, 0.0, 0.0), frame_i=1)
    cells2 = set(zip(*np.nonzero(em.clear_evidence > 0)))
    assert len(cells2) >= 1
    assert cells1 != cells2, "different poses must land in different world cells"

    y1, x1 = next(iter(cells1))
    y2, x2 = next(iter(cells2))
    assert abs((x2 - x1) - int(round(1.0 / PX_SIZE))) <= 2
    assert abs(y2 - y1) <= 2


def test_to_obs_known_honest_unknown():
    em = EvidenceMap()
    obs, known = em.to_obs_known()
    assert np.all(known == 0) and np.all(obs == 0)
    labels, height = _blank_ego()
    labels[RCY, RCX + 30] = OBSTACLE
    height[RCY, RCX + 30] = 25
    em.update(labels, height, (0.0, 0.0, 0.0))
    obs, known = em.to_obs_known()
    assert np.any(known == 255)
    assert np.any(obs > 0)
    assert np.count_nonzero(known == 0) > (em.map_w * em.map_h) // 2


if __name__ == "__main__":
    test_self_does_not_add_obstacle()
    print("OK self_no_obstacle")
    test_obstacle_decays_clear_reclaims()
    print("OK decay_reclaim")
    test_clear_does_not_invent_under_self()
    print("OK no_invent_under_self")
    test_pose_transform_two_poses()
    print("OK pose_transform")
    test_to_obs_known_honest_unknown()
    print("OK to_obs_known")
    print("ALL PASS")
