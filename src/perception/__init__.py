"""Honest ego perception: UNKNOWN | SELF | CLEAR | OBSTACLE.

See docs/perception/CONTRACT.md.
"""
from .labels import UNKNOWN, SELF, CLEAR, OBSTACLE, LABEL_NAMES
from .ego_rs1 import label_rs1_ego

__all__ = [
    "UNKNOWN", "SELF", "CLEAR", "OBSTACLE", "LABEL_NAMES",
    "label_rs1_ego",
]
