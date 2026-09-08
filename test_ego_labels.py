#!/usr/bin/env python3
"""Fast unit tests for honest RS1 ego labels (perception contract)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from robot_config import BODY_BOX, FRAME_H, FRAME_W, EGO_PX_SIZE, RCX, RCY
from perception.labels import UNKNOWN, SELF, CLEAR, OBSTACLE
from perception.ego_rs1 import label_rs1_ego, labels_to_obs_known


def _grid_floor(z=0.95, extent_m=1.2, n=40):
    xs = np.linspace(-extent_m / 2, extent_m / 2, n)
    ys = np.linspace(-extent_m / 2, extent_m / 2, n)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z, np.float32)])
    return pts.astype(np.float32)


def test_floor_clear_outside_boxes():
    verts = _grid_floor(z=0.95)
    # No self paint — pure floor should be CLEAR where hit
    labels, height = label_rs1_ego(
        verts, under_boxes=(), self_boxes=(), x_offset=0)
    assert np.any(labels == CLEAR), "expected CLEAR floor hits"
    assert not np.any(labels == OBSTACLE), "floor must not be OBSTACLE"
    assert np.all(height[labels == CLEAR] == 0)


def test_obstacle_height():
    # Point above floor band near center → OBSTACLE after 180° flip (still center)
    verts = np.array([[0.0, 0.0, 0.50]], dtype=np.float32)
    labels, height = label_rs1_ego(
        verts, under_boxes=(), self_boxes=(), x_offset=0)
    assert np.any(labels == OBSTACLE)
    assert height[labels == OBSTACLE].max() >= 40  # ~41 cm above floor_clip 0.91


def test_body_box_self_wins():
    # Floor everywhere; after paint, BODY_BOX must be SELF (x_offset=0 paints
    # boxes in camera-center frame — use empty scatter + paint only).
    labels, _ = label_rs1_ego(
        None, under_boxes=[BODY_BOX], self_boxes=(), x_offset=0)
    x0, y0, x1, y1 = BODY_BOX
    assert np.all(labels[y0:y1, x0:x1] == SELF)
    # Outside body remains UNKNOWN when no verts
    assert labels[0, 0] == UNKNOWN


def test_self_not_in_obs_known_shim():
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    x0, y0, x1, y1 = BODY_BOX
    labels[y0:y1, x0:x1] = SELF
    labels[10:20, 10:20] = CLEAR
    labels[30:40, 30:40] = OBSTACLE
    height[30:40, 30:40] = 25
    obs, known = labels_to_obs_known(labels, height)
    assert np.all(obs[y0:y1, x0:x1] == 0)
    assert np.all(known[y0:y1, x0:x1] == 0), "SELF must not invent known-clear"
    assert np.all(known[10:20, 10:20] == 255)
    assert np.all(obs[10:20, 10:20] == 0)
    assert np.all(obs[30:40, 30:40] == 25)


def test_x_offset_shifts_hits():
    # Single floor point at origin → lands near frame center before offset
    verts = np.array([[0.0, 0.0, 0.95]], dtype=np.float32)
    labels0, _ = label_rs1_ego(
        verts, under_boxes=(), self_boxes=(), x_offset=0)
    labels1, _ = label_rs1_ego(
        verts, under_boxes=(), self_boxes=(), x_offset=-75)
    ys0, xs0 = np.where(labels0 == CLEAR)
    ys1, xs1 = np.where(labels1 == CLEAR)
    assert len(xs0) >= 1 and len(xs1) >= 1
    assert int(xs1[0]) == int(xs0[0]) - 75

def test_fuse_rs2_obstacle_adds():
    from perception.fuse import fuse_rs2_into_ego
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    # RS1 clear patch away from axle
    labels[20:40, 200:220] = CLEAR
    obs2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    known2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    # Obstacle in RS2 frame that lands at (30, 210) with fw_dx=0
    obs2[30, 210] = 40
    known2[30, 210] = 255
    out, ht, m = fuse_rs2_into_ego(
        labels, height, obs2, known2, fw_dx=0, fw_dy=0,
        footprint_boxes=(), under_boxes=())
    assert out[30, 210] == OBSTACLE
    assert ht[30, 210] == 40
    assert out[25, 205] == CLEAR


def test_fuse_rejects_clear_under_chassis():
    from perception.fuse import fuse_rs2_into_ego
    from robot_config import BODY_BOX
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    obs2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    known2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    x0, y0, x1, y1 = BODY_BOX
    known2[y0:y1, x0:x1] = 255  # would-be CLEAR under body
    out, ht, m = fuse_rs2_into_ego(
        labels, height, obs2, known2, fw_dx=0, fw_dy=0)
    assert m["rs2_clear_under_pre"] > 0
    assert np.all(out[y0:y1, x0:x1] == SELF)
    assert not np.any((out == CLEAR) & (labels == UNKNOWN) & False)
    # No CLEAR inside body
    assert not np.any(out[y0:y1, x0:x1] == CLEAR)


def test_fuse_clear_respects_free_range():
    from perception.fuse import fuse_rs2_into_ego
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    obs2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    known2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    known2[50, 200] = 255
    free = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    # free_range excludes (50,200)
    out, _, m = fuse_rs2_into_ego(
        labels, height, obs2, known2, fw_dx=0, fw_dy=0,
        free_range=free, footprint_boxes=(), under_boxes=())
    assert out[50, 200] == UNKNOWN
    assert m["rs2_clear_accepted"] == 0
    free[50, 200] = 255
    out2, _, m2 = fuse_rs2_into_ego(
        labels, height, obs2, known2, fw_dx=0, fw_dy=0,
        free_range=free, footprint_boxes=(), under_boxes=())
    assert out2[50, 200] == CLEAR
    assert m2["rs2_clear_accepted"] == 1


def test_fuse_does_not_clear_over_rs1_obstacle():
    from perception.fuse import fuse_rs2_into_ego
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    labels[60, 180] = OBSTACLE
    height[60, 180] = 30
    obs2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    known2 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    known2[60, 180] = 255  # RS2 says clear
    free = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
    out, ht, _ = fuse_rs2_into_ego(
        labels, height, obs2, known2, fw_dx=0, fw_dy=0,
        free_range=free, footprint_boxes=(), under_boxes=())
    assert out[60, 180] == OBSTACLE
    assert ht[60, 180] == 30

if __name__ == "__main__":
    test_floor_clear_outside_boxes()
    print("OK floor_clear")
    test_obstacle_height()
    print("OK obstacle_height")
    test_body_box_self_wins()
    print("OK body_self")
    test_self_not_in_obs_known_shim()
    print("OK shim")
    test_x_offset_shifts_hits()
    print("OK x_offset")
    test_fuse_rs2_obstacle_adds()
    print("OK fuse_obs")
    test_fuse_rejects_clear_under_chassis()
    print("OK fuse_under")
    test_fuse_clear_respects_free_range()
    print("OK fuse_range")
    test_fuse_does_not_clear_over_rs1_obstacle()
    print("OK fuse_no_overwrite_obs")
    print("ALL PASS")
