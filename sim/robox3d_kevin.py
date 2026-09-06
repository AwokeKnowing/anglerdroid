"""Offline robox3d spike: thin Kevin URDF + mast vs table contact.

Headless-first. Optional short VizServer via ``python -m sim.robox3d_kevin --viz``.
Does not touch hardware / live drive.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Table top matches sim.box3d.house_boxes() "table_top" AABB:
#   x 0.4–1.1, y 1.6–2.2, z 0.70–0.78
TABLE_CENTER = (0.75, 1.9, 0.74)
TABLE_HALF = (0.35, 0.30, 0.04)  # half-extents for Body.add_box

# Kevin poses (axle origin). Mast local x≈-0.135 so under-table mast is still in AABB.
POSE_UNDER_TABLE = (0.75, 1.9, 0.0)
POSE_CLEAR = (0.5, 0.5, 0.0)

REPO_ROOT = Path(__file__).resolve().parents[1]
THIN_URDF = REPO_ROOT / "robot" / "kevin_thin.urdf"


def thin_urdf_path() -> Path:
    return THIN_URDF


def _require_robox3d():
    try:
        from robox3d import ContactSensor, World, load_urdf  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "robox3d is required for sim.robox3d_kevin (pip install robox3d)"
        ) from e
    from robox3d import ContactSensor, World, load_urdf

    return World, load_urdf, ContactSensor


def make_world_with_kevin(
    kevin_xyz=POSE_CLEAR,
    *,
    with_table: bool = True,
    table_center=TABLE_CENTER,
    table_half=TABLE_HALF,
    fixed_base: bool = False,
    gravity=(0.0, 0.0, 0.0),
):
    """Build a zero-g (default) world with thin Kevin and optional static table.

    Returns (world, robot, table_or_None). Caller must close the world
    (``with`` / ``world.destroy()``).
    """
    World, load_urdf, _ = _require_robox3d()
    world = World(gravity=gravity)
    robot = load_urdf(
        world,
        thin_urdf_path(),
        position=tuple(kevin_xyz),
        fixed_base=fixed_base,
        name="kevin_thin",
    )
    table = None
    if with_table:
        table = world.create_body(
            kind="static",
            name="table_top",
            position=tuple(table_center),
        )
        table.add_box(tuple(table_half))
    return world, robot, table


def mast_table_contact(
    kevin_xyz,
    *,
    table_center=TABLE_CENTER,
    table_half=TABLE_HALF,
    max_steps: int = 8,
    dt: float = 1.0 / 240.0,
) -> bool:
    """Return True if Kevin's body reports contact with the table within max_steps.

    Place Kevin so the mast AABB overlaps the table top for a positive case.
    Overlapping bodies are separated by the solver; contact impulses appear on
    early steps — we OR across the first ``max_steps`` frames.
    """
    _, _, ContactSensor = _require_robox3d()
    world, robot, table = make_world_with_kevin(
        kevin_xyz,
        with_table=True,
        table_center=table_center,
        table_half=table_half,
        fixed_base=False,
        gravity=(0.0, 0.0, 0.0),
    )
    try:
        sensor = ContactSensor(robot.base_body)
        table_sensor = ContactSensor(table)
        for _ in range(max_steps):
            world.step(dt)
            if sensor.read().touching or table_sensor.read().touching:
                return True
        return False
    finally:
        world.destroy()


def verify_mast_vs_table() -> dict:
    """Headless CI check: under table → contact; clear → no contact."""
    under = mast_table_contact(POSE_UNDER_TABLE)
    clear = mast_table_contact(POSE_CLEAR)
    return {
        "under_table_contact": under,
        "clear_of_table_contact": clear,
        "ok": under and not clear,
    }


def load_smoke() -> str:
    """Ensure load_urdf succeeds; return robot name."""
    world, robot, _ = make_world_with_kevin(
        POSE_CLEAR, with_table=False, fixed_base=True
    )
    try:
        assert robot.base_body is not None
        assert "base_link" in robot.links
        return robot.name
    finally:
        world.destroy()


def serve_viz(duration: float = 2.0, port: int = 8765) -> None:
    """Optional short VizServer demo; always stops."""
    from robox3d.viz import VizServer

    world, robot, _ = make_world_with_kevin(
        POSE_UNDER_TABLE, with_table=True, fixed_base=False, gravity=(0, 0, -9.81)
    )
    # Floor so Kevin doesn't fall forever
    floor = world.create_body(kind="static", name="floor", position=(0.75, 1.9, -0.09))
    floor.add_box((1.5, 1.5, 0.02))
    server = VizServer(world, robot=robot, port=port)
    server.start()
    try:
        dt = 1.0 / 240.0
        steps = 0
        start = time.monotonic()
        while world.time < duration:
            world.step(dt)
            server.update()
            steps += 1
            sleep = steps * dt - (time.monotonic() - start)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        server.stop()
        world.destroy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Thin Kevin / robox3d offline spike")
    parser.add_argument(
        "--viz",
        action="store_true",
        help="serve VizServer briefly (default: headless verify only)",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="viz seconds")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    name = load_smoke()
    print(f"load_urdf ok: {name} ({thin_urdf_path()})")
    result = verify_mast_vs_table()
    print(
        f"mast vs table: under={result['under_table_contact']} "
        f"clear={result['clear_of_table_contact']} ok={result['ok']}"
    )
    if not result["ok"]:
        return 1
    if args.viz:
        print(f"serving VizServer ≤{args.duration}s on port {args.port}…")
        serve_viz(duration=args.duration, port=args.port)
        print("VizServer stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
