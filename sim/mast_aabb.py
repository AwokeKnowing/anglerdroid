"""Box3D-style AABB mast/body collision scenarios (pure Python, no hardware).

Mirrors sim/box3d.py intent (G1): chassis FOOT can pass under furniture whose
top is below MAST_CLEAR, but the mast AABB collides with taller slabs.

Not wired to live drive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# Kevin thin dims (match kevin_thin.urdf / robot_config FOOT @ 1cm)
CHASSIS_HALF_XYZ = (0.15, 0.21, 0.08)  # half-extents; center at z=0.08
CHASSIS_CENTER_Z = 0.08
MAST_HALF_XYZ = (0.025, 0.025, 0.35)  # cylinder as AABB
MAST_CENTER_Z = 0.16 + 0.35  # chassis top + half mast
MAST_CLEAR_M = 0.45
MAST_TOP_Z_M = 0.16 + 0.70


@dataclass(frozen=True)
class Aabb:
    """Axis-aligned box in world frame: center + half-extents (meters)."""

    cx: float
    cy: float
    cz: float
    hx: float
    hy: float
    hz: float

    @property
    def z_min(self) -> float:
        return self.cz - self.hz

    @property
    def z_max(self) -> float:
        return self.cz + self.hz

    def overlaps(self, other: "Aabb") -> bool:
        return (
            abs(self.cx - other.cx) <= (self.hx + other.hx)
            and abs(self.cy - other.cy) <= (self.hy + other.hy)
            and abs(self.cz - other.cz) <= (self.hz + other.hz)
        )


def chassis_aabb(x: float = 0.0, y: float = 0.0) -> Aabb:
    hx, hy, hz = CHASSIS_HALF_XYZ
    return Aabb(x, y, CHASSIS_CENTER_Z, hx, hy, hz)


def mast_aabb(x: float = 0.0, y: float = 0.0) -> Aabb:
    hx, hy, hz = MAST_HALF_XYZ
    return Aabb(x, y, MAST_CENTER_Z, hx, hy, hz)


def box_from_corners(
    x0: float, y0: float, z0: float, x1: float, y1: float, z1: float
) -> Aabb:
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)
    return Aabb(cx, cy, cz, abs(x1 - x0) / 2, abs(y1 - y0) / 2, abs(z1 - z0) / 2)


@dataclass(frozen=True)
class MastScenario:
    name: str
    furniture: Aabb
    expect_mast_hit: bool
    expect_chassis_hit: bool
    note: str = ""


def scenarios(robot_xy: Tuple[float, float] = (0.0, 0.0)) -> List[MastScenario]:
    """Canonical G1 furniture cases under the robot pose."""
    x, y = robot_xy
    # Table slab over robot: top ~0.52m (> MAST_CLEAR) — mast hits, chassis free.
    table = Aabb(x, y, 0.50, 0.40, 0.40, 0.02)
    # Low coffee shelf: top 0.35m (< MAST_CLEAR) — neither mast nor chassis if
    # shelf is above chassis roof (0.16) but below mast clear… actually chassis
    # top is 0.16, shelf at 0.35 spans z 0.33–0.37 → mast z 0.16–0.86 overlaps.
    # For a true "under-clearance free" path the obstacle must sit entirely below
    # mast z_min (0.16) OR entirely above mast z_max — use a bumper at z=0.10
    # that only the chassis can hit, plus a high lintel that only mast hits.
    low_bumper = Aabb(x + 0.20, y, 0.05, 0.08, 0.25, 0.05)  # z 0–0.10
    # Doorway lintel: high beam above MAST_CLEAR — mast hits if robot under it.
    lintel = Aabb(x, y, 0.70, 0.35, 0.08, 0.05)  # z 0.65–0.75
    # Open air: far furniture, no overlap.
    far = Aabb(x + 2.0, y + 2.0, 0.50, 0.20, 0.20, 0.02)
    # Thin table ABOVE mast top (shelf on wall) — free.
    high_shelf = Aabb(x, y, 1.20, 0.40, 0.40, 0.02)  # z 1.18–1.22
    return [
        MastScenario("table_under", table, True, False, "classic mast vs table"),
        MastScenario("low_bumper", low_bumper, False, True, "FOOT-only contact"),
        MastScenario("doorway_lintel", lintel, True, False, "mast vs lintel"),
        MastScenario("open_floor", far, False, False, "no contact"),
        MastScenario("high_shelf", high_shelf, False, False, "above mast top"),
    ]


def evaluate(
    scenario: MastScenario, robot_xy: Tuple[float, float] = (0.0, 0.0)
) -> dict:
    x, y = robot_xy
    c = chassis_aabb(x, y)
    m = mast_aabb(x, y)
    mast_hit = m.overlaps(scenario.furniture)
    chassis_hit = c.overlaps(scenario.furniture)
    ok = (mast_hit == scenario.expect_mast_hit) and (
        chassis_hit == scenario.expect_chassis_hit
    )
    return {
        "name": scenario.name,
        "mast_hit": mast_hit,
        "chassis_hit": chassis_hit,
        "ok": ok,
        "note": scenario.note,
        "furniture_z": (scenario.furniture.z_min, scenario.furniture.z_max),
        "mast_z": (m.z_min, m.z_max),
        "chassis_z": (c.z_min, c.z_max),
    }


def run_all(robot_xy: Tuple[float, float] = (0.0, 0.0)) -> List[dict]:
    return [evaluate(s, robot_xy) for s in scenarios(robot_xy)]


def assert_all_green(robot_xy: Tuple[float, float] = (0.0, 0.0)) -> int:
    results = run_all(robot_xy)
    bad = [r for r in results if not r["ok"]]
    if bad:
        raise AssertionError(f"AABB mast scenarios failed: {bad}")
    return len(results)


if __name__ == "__main__":
    n = assert_all_green()
    for r in run_all():
        print(
            f"{r['name']}: mast={r['mast_hit']} chassis={r['chassis_hit']} ok={r['ok']}"
        )
    print(f"ok {n}/{n} aabb mast scenarios")
