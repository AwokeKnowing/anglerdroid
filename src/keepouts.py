"""keepouts.py – Named no-go zones for Kevin (map-frame + ego paint).

Depth/occupancy remains ground truth for hard obstacles. Soft things
(dog bed, rugs) and known hazards (treadmill, checkered door mat) are
painted into the ego obs map as synthetic obstacles when marked.

Kind-aware paint (matches mppi_costmap OBS_THRESH=100):
  soft / floor_mat → value 90  (soft prefer cost, not hard hit)
  hard             → value 200 (hard obstacle)

Marks are session-local until we have a persistent map origin.
File: ~/.kevin/keepouts.json
Pose for CLI mark: ~/.kevin/latest_pose.json (written by paint_ego).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robot_config import RCX, RCY, EGO_PX_SIZE, FRAME_H, FRAME_W

KEEP_PATH = os.path.expanduser("~/.kevin/keepouts.json")
POSE_PATH = os.path.expanduser("~/.kevin/latest_pose.json")
_lock = threading.Lock()
_cache: Optional[dict] = None
_last_pose: Optional[Tuple[float, float, float]] = None
_last_pose_write = 0.0
POSE_WRITE_PERIOD_S = 0.5

# Paint values: soft band is (50, OBS_THRESH], hard is > OBS_THRESH (100).
PAINT_SOFT = 90
PAINT_HARD = 200

# Defaults when James says mark <name> near the hazard.
NAMED_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dog_bed": {"radius_m": 0.70, "kind": "soft", "ahead_m": 0.40},
    "checkered_door": {"radius_m": 0.55, "kind": "floor_mat", "ahead_m": 0.30},
    "treadmill": {"radius_m": 0.85, "kind": "hard", "ahead_m": 0.50},
}

_SOFT_KINDS = frozenset({"soft", "floor_mat", "rug", "mat"})


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


def paint_value_for_kind(kind: str) -> int:
    k = (kind or "soft").lower()
    if k in _SOFT_KINDS:
        return PAINT_SOFT
    return PAINT_HARD


def _remember_pose(pose_xy_yaw: Tuple[float, float, float]) -> None:
    """Cache pose in-process and throttle-write for CLI mark."""
    global _last_pose, _last_pose_write
    _last_pose = (
        float(pose_xy_yaw[0]),
        float(pose_xy_yaw[1]),
        float(pose_xy_yaw[2]),
    )
    now = time.monotonic()
    if now - _last_pose_write < POSE_WRITE_PERIOD_S:
        return
    _last_pose_write = now
    try:
        os.makedirs(os.path.dirname(POSE_PATH), exist_ok=True)
        tmp = POSE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {"x": _last_pose[0], "y": _last_pose[1], "yaw": _last_pose[2], "t": time.time()},
                f,
            )
        os.replace(tmp, POSE_PATH)
    except Exception:
        pass


def read_latest_pose() -> Optional[Tuple[float, float, float]]:
    if _last_pose is not None:
        return _last_pose
    try:
        with open(POSE_PATH) as f:
            d = json.load(f)
        return (float(d["x"]), float(d["y"]), float(d["yaw"]))
    except Exception:
        return None


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
    disks.append(
        {
            "id": name,
            "x": cx,
            "y": cy,
            "r_m": float(radius_m),
            "kind": kind,
            "paint": paint_value_for_kind(kind),
        }
    )
    data["disks_map_m"] = disks
    save(data)
    print(
        "keepouts: marked %s at (%.2f,%.2f) r=%.2fm kind=%s paint=%d"
        % (name, cx, cy, radius_m, kind, paint_value_for_kind(kind))
    )
    return data


def mark_named(
    name: str,
    pose_xy_yaw: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Mark a known named hazard using NAMED_DEFAULTS + latest pose."""
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "dogbed": "dog_bed",
        "bed": "dog_bed",
        "checkered": "checkered_door",
        "checkered_mat": "checkered_door",
        "mat": "checkered_door",
        "door_mat": "checkered_door",
        "door": "checkered_door",
        "tread": "treadmill",
    }
    key = aliases.get(key, key)
    if key not in NAMED_DEFAULTS:
        raise ValueError(
            "unknown keepout %r — known: %s" % (name, ", ".join(sorted(NAMED_DEFAULTS)))
        )
    pose = pose_xy_yaw or read_latest_pose()
    if pose is None:
        raise RuntimeError("no pose yet — wait for live main / paint_ego, or pass pose")
    d = NAMED_DEFAULTS[key]
    return mark_disk_at_pose(
        key,
        pose,
        radius_m=float(d["radius_m"]),
        kind=str(d["kind"]),
        ahead_m=float(d["ahead_m"]),
    )


def paint_ego(obs: np.ndarray, pose_xy_yaw: Tuple[float, float, float], value: int = None) -> np.ndarray:
    """OR keepout disks into a copy of ego obs (H,W). pose = robot in map frame.

    Soft kinds use PAINT_SOFT (prefer); hard kinds use PAINT_HARD.
    Optional ``value`` overrides every disk (legacy).
    """
    if obs is None:
        return obs
    _remember_pose(pose_xy_yaw)
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
        kind = str(d.get("kind") or "soft")
        pv = int(value) if value is not None else int(d.get("paint") or paint_value_for_kind(kind))
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
        out2[rr0:rr1, cc0:cc1][mask] = np.maximum(out2[rr0:rr1, cc0:cc1][mask], pv)
    if out.ndim == 3:
        out[:, :, 0] = out2
        return out
    return out2


def list_marks() -> List[dict]:
    return list(load().get("disks_map_m") or [])


def _self_test() -> None:
    from robot_config import FRAME_H, FRAME_W, RCX, RCY

    assert paint_value_for_kind("soft") == PAINT_SOFT
    assert paint_value_for_kind("floor_mat") == PAINT_SOFT
    assert paint_value_for_kind("hard") == PAINT_HARD
    assert PAINT_SOFT <= 100  # soft prefer band for MPPI
    assert PAINT_HARD > 100

    obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    # Fake disk at robot nose in map frame with pose at origin facing +x
    data = load(force=True)
    data["disks_map_m"] = [
        {"id": "dog_bed", "x": 0.40, "y": 0.0, "r_m": 0.5, "kind": "soft"},
        {"id": "treadmill", "x": 0.0, "y": 0.80, "r_m": 0.5, "kind": "hard"},
    ]
    save(data)
    painted = paint_ego(obs, (0.0, 0.0, 0.0))
    soft_max = int(painted[RCY - 5 : RCY + 5, RCX + 5 : RCX + 40].max())
    hard_max = int(painted[RCY - 40 : RCY - 5, RCX - 10 : RCX + 10].max())
    assert soft_max == PAINT_SOFT, soft_max
    assert hard_max == PAINT_HARD, hard_max
    # restore empty disks for live session
    data = load(force=True)
    data["disks_map_m"] = []
    save(data)
    print("keepouts: self-test OK soft=%d hard=%d" % (soft_max, hard_max))


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args or args[0] in ("test", "self-test"):
        _self_test()
    elif args[0] == "list":
        print(json.dumps(list_marks(), indent=2))
    elif args[0] == "mark" and len(args) >= 2:
        mark_named(args[1])
        print(json.dumps(list_marks(), indent=2))
    else:
        print("usage: python -m keepouts [test|list|mark <name>]")
        sys.exit(2)
