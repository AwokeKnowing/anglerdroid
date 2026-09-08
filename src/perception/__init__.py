"""Honest ego perception: UNKNOWN | SELF | CLEAR | OBSTACLE.

See docs/perception/CONTRACT.md.
"""
from .labels import UNKNOWN, SELF, CLEAR, OBSTACLE, LABEL_NAMES
from .ego_rs1 import label_rs1_ego, labels_to_obs_known
from .fuse import fuse_rs2_into_ego
from .evidence_map import EvidenceMap
from .planner_feed import (
    apply_self_honest, ego_labels_to_planner_feed,
    evidence_to_ego_obs_known, select_planner_feed,
)
from .dynamic_mask import (
    build_slam_outlier_mask, apply_mask_to_obs, mask_as_uint8,
)

__all__ = [
    "UNKNOWN", "SELF", "CLEAR", "OBSTACLE", "LABEL_NAMES",
    "label_rs1_ego", "labels_to_obs_known", "fuse_rs2_into_ego",
    "EvidenceMap",
    "apply_self_honest", "ego_labels_to_planner_feed",
    "evidence_to_ego_obs_known", "select_planner_feed",
    "build_slam_outlier_mask", "apply_mask_to_obs", "mask_as_uint8",
]
