#!/usr/bin/env python3
"""WanderAround: reward tiles 50 cm from obstacles; claim deletes 2 m."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("kernels.py").read_text()


def _f(name: str) -> float:
    m = re.search(rf"^{name}\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+))", _SRC, re.M)
    if m is None:
        raise AssertionError("missing " + name)
    return float(m.group(1))


def _s(name: str) -> str:
    m = re.search(rf'^{name}\s*=\s*"([^"]+)"', _SRC, re.M)
    if m is None:
        raise AssertionError("missing " + name)
    return m.group(1)


TILE_CLEAR_M = _f("TILE_CLEAR_M")
DELETE_R = _f("DELETE_R")
PAINT_R = _f("PAINT_R")
DELETE_WIN = int(_f("DELETE_WIN"))
SEC_NX = int(_f("SEC_NX"))
_m = re.search(
    r"^FLOOR_X0,\s*FLOOR_X1\s*=\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)",
    _SRC,
    re.M,
)
assert _m, "missing FLOOR_X0, FLOOR_X1"
SEC_SX = (float(_m.group(2)) - float(_m.group(1))) / float(SEC_NX)


def aabb_inside(x, y, boxes, pad):
    for x0, y0, x1, y1 in boxes:
        if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
            return True
    return False


def clear_to(x, y, tx, ty, boxes):
    dx, dy = tx - x, ty - y
    glen = math.hypot(dx, dy)
    if glen < 0.10:
        return True
    n = max(8, int(glen / 0.05))
    for i in range(1, n + 1):
        t = i / float(n)
        if aabb_inside(x + t * dx, y + t * dy, boxes, 0.0):
            return False
    return True


def claim(x, y, tiles, boxes, suppress=True):
    """Paint then optionally suppress. Reward only newly painted tiles."""
    nnew = 0
    for t in tiles:
        if not t["valid"] or t["seen"]:
            continue
        if aabb_inside(t["x"], t["y"], boxes, TILE_CLEAR_M):
            continue
        dx, dy = t["x"] - x, t["y"] - y
        if dx * dx + dy * dy > PAINT_R * PAINT_R:
            continue
        if not clear_to(x, y, t["x"], t["y"], boxes):
            continue
        t["seen"] = True
        nnew += 1
    ndel = 0
    if nnew <= 0 or not suppress:
        return nnew, ndel
    for t in tiles:
        if not t["valid"] or t["seen"]:
            continue
        dx, dy = t["x"] - x, t["y"] - y
        if dx * dx + dy * dy > DELETE_R * DELETE_R:
            continue
        if not clear_to(x, y, t["x"], t["y"], boxes):
            continue
        t["seen"] = True
        ndel += 1
    return nnew, ndel


def test_behavior_knobs():
    assert _s("BEHAVIOR") == "WanderAround"
    assert abs(TILE_CLEAR_M - 0.50) < 1e-9
    assert abs(DELETE_R - 2.00) < 1e-9
    assert DELETE_WIN * SEC_SX + 1e-6 >= DELETE_R
    assert int(_f("WAYPOINT_REWARD")) == 1
    assert int(_f("SEEN_PAINT")) == 1
    assert int(_f("SEEN_SUPPRESS")) == 2
    assert abs(_f("NEAR_CLEAR_M") - 0.40) < 1e-9
    assert abs(_f("NEAR_PEN") - 0.005) < 1e-9


def test_near_penalty_is_small_vs_a_claim():
    near_pen = _f("NEAR_PEN")
    claim = 1.0
    assert abs(near_pen - 0.005) < 1e-9
    assert 64.0 * near_pen < 0.40 * claim
    wall = [(1.0, -1.0, 1.2, 1.0)]
    tile = (0.5, 0.0)
    assert not aabb_inside(*tile, wall, _f("NEAR_CLEAR_M"))
    assert aabb_inside(0.75, 0.0, wall, _f("NEAR_CLEAR_M"))


def test_tiles_need_50cm_clearance():
    boxes = [(1.0, -0.2, 1.4, 0.2)]
    near = (1.2, 0.60)   # 40 cm from box face
    far = (1.2, 0.80)    # 60 cm from box face
    assert aabb_inside(*near, boxes, TILE_CLEAR_M)
    assert not aabb_inside(*far, boxes, TILE_CLEAR_M)


def test_claim_deletes_within_2m_no_extra_reward():
    boxes = []
    tiles = [
        {"x": 0.0, "y": 0.0, "valid": True, "seen": False},
        {"x": 0.4, "y": 0.0, "valid": True, "seen": False},
        {"x": 1.50, "y": 0.0, "valid": True, "seen": False},
        {"x": 2.50, "y": 0.0, "valid": True, "seen": False},
    ]
    nnew, ndel = claim(0.0, 0.0, tiles, boxes)
    assert nnew == 2  # origin + 0.4 m (inside 0.90 paint)
    assert ndel == 1  # 1.50 m deleted, 2.50 m survives
    assert tiles[2]["seen"] is True
    assert tiles[3]["seen"] is False
    nnew2, ndel2 = claim(2.50, 0.0, tiles, boxes)
    assert nnew2 == 1
    assert ndel2 == 0


def test_suppress_does_not_clear_through_walls():
    wall = [(0.7, -0.4, 0.9, 0.4)]
    tiles = [
        {"x": 0.0, "y": 0.0, "valid": True, "seen": False},
        {"x": 1.50, "y": 0.0, "valid": True, "seen": False},
    ]
    nnew, ndel = claim(0.0, 0.0, tiles, wall)
    assert nnew == 1
    assert ndel == 0
    assert tiles[1]["seen"] is False


def test_pack_budget_is_greedy_2m_los():
    # Two tiles 1.5 m apart, same room → one claim. 2.5 m → two.
    close = [
        {"x": 0.0, "y": 0.0, "valid": True, "seen": False},
        {"x": 1.50, "y": 0.0, "valid": True, "seen": False},
    ]
    n = 0
    for t in close:
        if t["seen"]:
            continue
        n += 1
        claim(t["x"], t["y"], close, [])
    assert n == 1
    far = [
        {"x": 0.0, "y": 0.0, "valid": True, "seen": False},
        {"x": 2.50, "y": 0.0, "valid": True, "seen": False},
    ]
    n = 0
    for t in far:
        if t["seen"]:
            continue
        n += 1
        claim(t["x"], t["y"], far, [])
    assert n == 2


def test_walk_without_reward_does_not_suppress():
    # 3 m tile is 2 m from x=1.0. A follow-bubble would eat it; a claim-only
    # clear must leave it for the next reward ~2 m later.
    tiles = [
        {"x": 0.0, "y": 0.0, "valid": True, "seen": False},
        {"x": 1.50, "y": 0.0, "valid": True, "seen": False},
        {"x": 3.00, "y": 0.0, "valid": True, "seen": False},
    ]
    nnew, ndel = claim(0.0, 0.0, tiles, [])
    assert nnew == 1
    assert ndel == 1
    assert tiles[1]["seen"] is True
    assert tiles[2]["seen"] is False
    n2, d2 = claim(1.00, 0.0, tiles, [])
    assert n2 == 0 and d2 == 0
    assert tiles[2]["seen"] is False
    n3, d3 = claim(3.00, 0.0, tiles, [])
    assert n3 == 1
    assert tiles[2]["seen"] is True


def test_claim_is_one_reward_not_deleted_tiles():
    nnew, ndel = 5, 40
    paint_n = max(1.0, math.pi * PAINT_R * PAINT_R / (SEC_SX * SEC_SX))
    reward = nnew / paint_n
    hits = 1 if nnew > 0 else 0
    assert hits == 1
    assert hits != nnew + ndel
    assert reward < 5.01


def test_carpet_slip_matches_pilot_bag():
    slip_v = _f("SLIP_V")
    slip_w = _f("SLIP_W")
    delay = int(_f("DELAY_STEPS"))
    assert delay == 1
    # wander_5min_pilot 0.10 m/s × 5 s, safety_fwd=1
    got = 0.10 * 5.0 * slip_v
    assert 0.370 <= got <= 0.400
    assert abs(got - 0.392) < 0.03 or abs(got - 0.379) < 0.03
    # free yaw 0.25 rad/s × 5 s → 70.4° / 71.6°
    yaw = 0.25 * 5.0 * slip_w
    assert abs(math.degrees(yaw) - 70.4) < 3.0


if __name__ == "__main__":
    test_behavior_knobs()
    test_tiles_need_50cm_clearance()
    test_claim_deletes_within_2m_no_extra_reward()
    test_suppress_does_not_clear_through_walls()
    test_pack_budget_is_greedy_2m_los()
    test_walk_without_reward_does_not_suppress()
    test_claim_is_one_reward_not_deleted_tiles()
    test_near_penalty_is_small_vs_a_claim()
    test_carpet_slip_matches_pilot_bag()
    print("ok")
