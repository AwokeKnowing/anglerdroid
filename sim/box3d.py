"""Lightweight Box3D-style world: furniture as AABBs.

Not full Erin-Catto Box3D physics (robox3d spike is separate). Closes gaps G1/P4:
mast/body 3D collision + height projection into 2D maps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.robot import EGO_PX_SIZE


@dataclass
class Box3:
    x0: float
    y0: float
    x1: float
    y1: float
    z0: float = 0.0
    z1: float = 0.5
    name: str = "box"


BODY_L = 0.30
BODY_W = 0.42
MAST_H = 0.90
MAST_Z0 = 0.35
MAST_R = 0.04
FOOT_Z1 = 0.25


def house_boxes():
    return [
        Box3(-0.2, -0.2, 3.2, 0.0, 0, 1.2, "wall_s"),
        Box3(-0.2, 2.4, 3.2, 2.6, 0, 1.2, "wall_n"),
        Box3(-0.2, -0.2, 0.0, 2.6, 0, 1.2, "wall_w"),
        Box3(3.0, -0.2, 3.2, 2.6, 0, 1.2, "wall_e"),
        Box3(1.4, 0.6, 2.4, 1.4, 0.0, 0.45, "couch_seat"),
        Box3(1.4, 1.25, 2.4, 1.4, 0.45, 0.85, "couch_back"),
        Box3(0.4, 1.6, 1.1, 2.2, 0.70, 0.78, "table_top"),
        Box3(0.4, 1.6, 1.1, 2.2, 0.0, 0.05, "table_base"),
        Box3(2.5, 1.7, 2.85, 2.1, 0.0, 0.55, "chair"),
    ]


def rasterize_boxes(boxes, width_m=3.4, height_m=2.8, px=EGO_PX_SIZE):
    w = int(round(width_m / px))
    h = int(round(height_m / px))
    obs = np.zeros((h, w), dtype=np.uint8)
    height = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x0 = max(0, int(b.x0 / px))
        x1 = min(w, int(math.ceil(b.x1 / px)))
        y0 = max(0, int(b.y0 / px))
        y1 = min(h, int(math.ceil(b.y1 / px)))
        if x1 <= x0 or y1 <= y0:
            continue
        h_cm = int(min(255, max(1, round(b.z1 * 100))))
        if b.z0 < 0.15:
            obs[y0:y1, x0:x1] = 255
        np.maximum(height[y0:y1, x0:x1], h_cm, out=height[y0:y1, x0:x1])
        if b.z0 >= 0.4:
            obs[y0:y1, x0:x1] = np.maximum(obs[y0:y1, x0:x1], 200)
    return obs, height


def mast_collision(boxes, x, y, theta):
    mx, my = x, y
    mz0, mz1 = MAST_Z0, MAST_H
    for b in boxes:
        if (mx < b.x0 - MAST_R or mx > b.x1 + MAST_R or
                my < b.y0 - MAST_R or my > b.y1 + MAST_R):
            continue
        if mz1 < b.z0 or mz0 > b.z1:
            continue
        return True, b.name
    return False, None


def body_collision(boxes, x, y, theta):
    r = 0.5 * math.hypot(BODY_L, BODY_W)
    for b in boxes:
        if b.z0 > FOOT_Z1:
            continue
        cx = min(max(x, b.x0), b.x1)
        cy = min(max(y, b.y0), b.y1)
        if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
            return True, b.name
    return False, None
