"""keepouts.py – Named no-go zones for Kevin (map-frame + ego paint).

Depth/occupancy remains ground truth for hard obstacles. Soft things
(dog bed, rugs) and known hazards (treadmill, checkered door mat) are
painted into the ego obs map as synthetic obstacles when marked.

Marks are session-local until we have a persistent map origin.
File: ~/.kevin/keepouts.json
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robot_config import RCX, RCY, EGO_PX_SIZE, FRAME_H, FRAME_W

KEEP_PATH = os.path.expanduser("~/.kevin/keepouts.json")
_lock = threading.Lock()
_cache: Optional[dict] = None


def _default() -> dict:
    return {
        "version": 1,
        "note": "polygons_map_m are [x,y] rings in meters, map frame at session origin.",
        "named": [
            {"id": "checkered_door", "label": "checkered area by door", "kind": "floor_mat"},
            {"id": "dog_bed", "label": "dog bed", "kind": "soft"},
            {"id": "treadmill", "label": "treadmill", "kind": "hard"},
        ],
        "polygons_map_m": [],
        "disks_map_m": [],  # [{id,x,y,r_m,kind}]
    }


def load(force: bool = False) -> dict:
    global _cache
    with _lock:
        if _cache is not None and not force:
            return _cache
        try:
            with open(KEEP_PATH) as f:
                _cache = json.load(f)
        except Exception:
            _cache = _default()
            save(_cache)
        return _cache


def save(data: dict) -> None:
    global _cache
    os.makedirs(os.path.dirname(KEEP_PATH), exist_ok=True)
    with _lock:
        _cache = data
        with open(KEEP_PATH, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")


def mark_disk_at_pose(
    name: str,
    pose_xy_yaw: Tuple[float, float, float],
    radius_m: float = 0.55,
    kind: str = "soft",
    ahead_m: float = 0.35,
) -> dict:
    """Mark a disk slightly ahead of current pose (where the hazard is)."""
    x, y, yaw = pose_xy_yaw
    cx = x + ahead_m * math.cos(yaw)
    cy = y + ahead_m * math.sin(yaw)
    data = load(force=True)
    disks = list(data.get("disks_map_m") or [])
    # replace same id
    disks = [d for d in disks if d.get("id") != name]
    disks.append({"id": name, "x": cx, "y": cy, "r_m": float(radius_m), "kind": kind})
    data["disks_map_m"] = disks
    save(data)
    print("keepouts: marked %s at (%.2f,%.2f) r=%.2fm kind=%s" % (name, cx, cy, radius_m, kind))
    return data


def paint_ego(obs: np.ndarray, pose_xy_yaw: Tuple[float, float, float], value: int = 200) -> np.ndarray:
    """OR keepout disks into a copy of ego obs (H,W). pose = robot in map frame."""
    if obs is None:
        return obs
    data = load()
    disks = data.get("disks_map_m") or []
    if not disks:
        return obs
    out = np.array(obs, copy=True)
    if out.ndim == 3:
        out2 = out[:, :, 0].copy()
    else:
        out2 = out
    h, w = out2.shape[:2]
    rx, ry, yaw = pose_xy_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    # map -> ego: p_ego = R^T (p_map - robot)
    for d in disks:
        mx, my, rm = float(d["x"]), float(d["y"]), float(d["r_m"])
        dx, dy = mx - rx, my - ry
        ex = c * dx + s * dy
        ey = -s * dx + c * dy
        # ego pixel: x forward = col+, y left = row-
        col = int(round(RCX + ex / EGO_PX_SIZE))
        row = int(round(RCY - ey / EGO_PX_SIZE))
        rad = max(1, int(round(rm / EGO_PX_SIZE)))
        rr0, rr1 = max(0, row - rad), min(h, row + rad + 1)
        cc0, cc1 = max(0, col - rad), min(w, col + rad + 1)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        yy, xx = np.ogrid[rr0:rr1, cc0:cc1]
        mask = (yy - row) ** 2 + (xx - col) ** 2 <= rad * rad
        out2[rr0:rr1, cc0:cc1][mask] = np.maximum(out2[rr0:rr1, cc0:cc1][mask], value)
    if out.ndim == 3:
        out[:, :, 0] = out2
        return out
    return out2


def list_marks() -> List[dict]:
    return list(load().get("disks_map_m") or [])
