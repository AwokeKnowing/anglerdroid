#!/usr/bin/env python3
"""Kevin in Newton: box chassis + hub wheels + back caster + mast, house STL.

Origin: axle midpoint, +X forward, +Y left, +Z up.
House STL stays a visual site. Walls the depth camera can hit are thin
collision boxes clustered from the vertical faces, not a mesh collider.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import warp as wp

import newton

BODY_L = 0.33
BODY_W = 0.33
BODY_H = 0.15
AXLE_FROM_REAR = 0.18
CASTER_HANG = 0.08
WHEEL_R = 0.09
WHEEL_W = 0.06
WHEEL_GAP = 0.005
CASTER_R = 0.025
MAST_CAM_Z = 0.90
MAST_POST = 0.03
CAM_SIZE = (0.09, 0.025, 0.03)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STL = Path("/home/james/Desktop/casitahouse-walls.stl")


def _box_center_x() -> float:
    rear = -AXLE_FROM_REAR
    front = rear + BODY_L
    return 0.5 * (rear + front)


def build_kevin(builder: newton.ModelBuilder, start_xy=(1.0, 1.0), yaw=0.0):
    rear_x = -AXLE_FROM_REAR
    cx = _box_center_x()
    z_body = WHEEL_R + 0.02  # keep the belly off the floor if the rear settles
    q_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))
    base = wp.transform(p=wp.vec3(float(start_xy[0]), float(start_xy[1]), 0.0), q=q_yaw)

    chassis = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        label="kevin_base",
    )
    body_cfg = newton.ModelBuilder.ShapeConfig(density=180.0, mu=0.6, gap=0.002)
    builder.add_shape_box(
        chassis,
        xform=wp.transform(p=wp.vec3(cx, 0.0, z_body), q=wp.quat_identity()),
        hx=BODY_L * 0.5,
        hy=BODY_W * 0.5,
        hz=BODY_H * 0.5,
        cfg=body_cfg,
        color=(0.15, 0.45, 0.85),
        label="body",
    )

    mast_cfg = newton.ModelBuilder.ShapeConfig(density=40.0, mu=0.4, gap=0.001)
    z_top = z_body + BODY_H * 0.5
    post_hz = max(0.02, 0.5 * (MAST_CAM_Z - z_top))
    post_cz = 0.5 * (z_top + MAST_CAM_Z)
    builder.add_shape_box(
        chassis,
        xform=wp.transform(p=wp.vec3(rear_x, 0.0, post_cz), q=wp.quat_identity()),
        hx=MAST_POST * 0.5,
        hy=MAST_POST * 0.5,
        hz=post_hz,
        cfg=mast_cfg,
        color=(0.55, 0.58, 0.62),
        label="mast_post",
    )
    boom_hx = AXLE_FROM_REAR * 0.5
    builder.add_shape_box(
        chassis,
        xform=wp.transform(p=wp.vec3(rear_x + boom_hx, 0.0, MAST_CAM_Z), q=wp.quat_identity()),
        hx=boom_hx,
        hy=MAST_POST * 0.5,
        hz=MAST_POST * 0.5,
        cfg=mast_cfg,
        color=(0.55, 0.58, 0.62),
        label="mast_boom",
    )
    builder.add_shape_box(
        chassis,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, MAST_CAM_Z + 0.012), q=wp.quat_identity()),
        hx=CAM_SIZE[0] * 0.5,
        hy=CAM_SIZE[2] * 0.5,
        hz=CAM_SIZE[1] * 0.5,
        cfg=mast_cfg,
        color=(0.15, 0.35, 0.55),
        label="mast_cams",
    )

    j_base = builder.add_joint_free(
        chassis,
        parent=-1,
        parent_xform=base,
        label="kevin_free",
    )

    y_wheel = BODY_W * 0.5 + WHEEL_GAP + WHEEL_W * 0.5
    wheel_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, mu=1.1, gap=0.0)
    wheel_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
    wheel_joints = []
    for name, y in (("wheel_left", y_wheel), ("wheel_right", -y_wheel)):
        hub = builder.add_link(label=name)
        builder.add_shape_cylinder(
            hub,
            xform=wp.transform(q=wheel_q),
            radius=WHEEL_R,
            half_height=WHEEL_W * 0.5,
            cfg=wheel_cfg,
            color=(0.08, 0.08, 0.1),
            label=name,
        )
        wheel_joints.append(
            builder.add_joint_revolute(
                parent=chassis,
                child=hub,
                parent_xform=wp.transform(p=wp.vec3(0.0, y, WHEEL_R), q=wp.quat_identity()),
                child_xform=wp.transform(),
                axis=newton.Axis.Y,
                target_vel=0.0,
                target_kd=8.0,
                damping=0.02,
                armature=0.01,
                effort_limit=12.0,
                actuator_mode=newton.JointTargetMode.VELOCITY,
                label=name + "_drive",
            )
        )

    # Fixed skid, not a ball joint. A free caster folds, the belly drops,
    # and the chassis sits immobilized while the wheels spin.
    caster_cfg = newton.ModelBuilder.ShapeConfig(density=80.0, mu=0.08, gap=0.0)
    caster_x = rear_x - CASTER_HANG
    builder.add_shape_sphere(
        chassis,
        xform=wp.transform(p=wp.vec3(caster_x, 0.0, CASTER_R), q=wp.quat_identity()),
        radius=CASTER_R,
        cfg=caster_cfg,
        color=(0.25, 0.25, 0.28),
        label="caster",
    )

    builder.add_articulation([j_base, *wheel_joints], label="kevin")
    build_kevin.chassis = chassis
    return wheel_joints


def build_kevin_urdf(
    builder: newton.ModelBuilder,
    start_xy=(1.0, 1.0),
    yaw=0.0,
    urdf_path: Path | None = None,
    freeze_casters: bool = True,
):
    """Load MakerBeam XL twin via ModelBuilder.add_urdf (floating + collapse fixed).

    Spawns with axle at z=wheel_r so tires kiss the ground plane. Continuous
    caster joints default to stiff position holds for first bring-up (set
    freeze_casters=False to leave them free). Drive wheels get VELOCITY
    actuators. Mesh paths resolve via sim/kevin-urdf-twin/urdf/meshes -> ../meshes.
    """
    from urdf_twin_fk import (
        DEFAULT_URDF,
        camera_poses,
        wheel_geometry,
        body_extents,
        audit_collisions_inertials,
    )

    urdf = Path(urdf_path or DEFAULT_URDF).resolve()
    if not urdf.exists():
        raise FileNotFoundError("URDF twin missing: %s" % urdf)
    mesh_link = urdf.parent / "meshes"
    if not mesh_link.exists():
        target = urdf.parent.parent / "meshes"
        if target.exists():
            mesh_link.symlink_to(target, target_is_directory=True)
            print("urdf mesh symlink", mesh_link, "->", target, flush=True)
        else:
            raise FileNotFoundError("URDF meshes missing at %s" % target)

    wheels = wheel_geometry(urdf)
    body = body_extents(urdf)
    cams = camera_poses(urdf)
    audit = audit_collisions_inertials(urdf)
    wheel_r = float(wheels["wheel_r"])

    q_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))
    xf = wp.transform(
        p=wp.vec3(float(start_xy[0]), float(start_xy[1]), wheel_r),
        q=q_yaw,
    )
    builder.add_urdf(
        str(urdf),
        xform=xf,
        floating=True,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=False,
        parse_visuals_as_colliders=False,
    )

    labels = list(builder.joint_label)

    def _find(substr: str) -> int:
        for i, lab in enumerate(labels):
            if substr in str(lab):
                return i
        raise KeyError("joint containing %r not in %s" % (substr, labels))

    j_free = 0
    j_left = _find("wheel_left_joint")
    j_right = _find("wheel_right_joint")
    caster_js = []
    for s in (
        "front_caster_yaw_joint",
        "front_caster_wheel_joint",
        "gate_caster_yaw_joint",
        "gate_caster_wheel_joint",
    ):
        try:
            caster_js.append(_find(s))
        except KeyError:
            pass

    qd_starts = list(builder.joint_qd_start)
    VEL = newton.JointTargetMode.VELOCITY
    POS = newton.JointTargetMode.POSITION

    def _dof(j: int) -> int:
        return int(qd_starts[j])

    for j in (j_left, j_right):
        d = _dof(j)
        builder.joint_target_mode[d] = VEL
        builder.joint_target_kd[d] = 8.0
        builder.joint_target_ke[d] = 0.0
        builder.joint_armature[d] = 0.01
        builder.joint_effort_limit[d] = 12.0

    if freeze_casters:
        for j in caster_js:
            d = _dof(j)
            builder.joint_target_mode[d] = POS
            builder.joint_target_ke[d] = 400.0
            builder.joint_target_kd[d] = 20.0
            q0 = int(builder.joint_q_start[j])
            if q0 < len(builder.joint_target_q):
                builder.joint_target_q[q0] = 0.0

    body_labels = list(builder.body_label)
    chassis = next(i for i, lab in enumerate(body_labels) if "base_link" in str(lab))

    build_kevin_urdf.chassis = chassis
    build_kevin_urdf.free_joint_label = str(labels[j_free])
    build_kevin_urdf.wheel_joints = [j_left, j_right]
    build_kevin_urdf.caster_joints = caster_js
    build_kevin_urdf.wheel_r = wheel_r
    build_kevin_urdf.track = float(wheels["track"])
    build_kevin_urdf.front_x = float(body["front_x"])
    build_kevin_urdf.rear_x = float(body["rear_x"])
    build_kevin_urdf.half_w = float(body["half_w"])
    build_kevin_urdf.camera_poses = cams
    build_kevin_urdf.audit = audit
    build_kevin_urdf.urdf = urdf
    n_col = int(audit.get("n_collision_geoms", 0))
    missing = audit.get("missing_inertial", [])
    print(
        "URDF twin %s freeze_casters=%s wheel_r=%.4f track=%.3f "
        "RS1_z=%.3f RS2_z=%.3f RS2_pitch=%.1f collisions=%d missing_inertial=%d"
        % (
            urdf.name,
            freeze_casters,
            wheel_r,
            float(wheels["track"]),
            float(cams["rs1"]["z"]),
            float(cams["rs2"]["z"]),
            float(cams["rs2"]["pitch_down_deg"]),
            n_col,
            len(missing),
        ),
        flush=True,
    )
    return [j_left, j_right]



def add_house(builder: newton.ModelBuilder, stl_path: Path):
    """Visual-only house. Repaired STL, site (no collider). Sensors raycast the mesh."""
    from house_mesh import FIXED_STL, REPORT, load_indexed, repair_house_stl, wall_segments_xy, wall_collision_boxes

    src = Path(stl_path)
    if not src.exists() and FIXED_STL.exists():
        src = FIXED_STL
    fixed = repair_house_stl(src, FIXED_STL)
    verts, faces = load_indexed(fixed)
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    verts[:, 2] -= float(verts[:, 2].min())
    mins = verts.min(0)
    maxs = verts.max(0)
    print(
        "house STL m: min=%s max=%s size=%s tris=%d (repaired %s)"
        % (mins.tolist(), maxs.tolist(), (maxs - mins).tolist(), len(faces), fixed)
    )
    mesh = newton.Mesh(verts, faces, compute_inertia=False, is_solid=False)
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mark_as_site()
    builder.add_shape_mesh(
        -1,
        mesh=mesh,
        cfg=cfg,
        color=(0.82, 0.78, 0.70),
        label="casitahouse_walls",
    )
    add_house.verts = verts
    add_house.faces = faces
    add_house.wall_segments = wall_segments_xy(verts, faces)
    boxes = wall_collision_boxes(verts, faces, thickness=0.08)
    add_house.wall_boxes = boxes
    add_house.box_count = len(boxes)
    add_house.report = dict(REPORT)
    add_house.path = fixed
    # Collision primitives, not a mesh collider. RS1/RS2 and the Newton
    # camera hit these boxes. The repaired STL stays a visual site.
    box_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.4, gap=0.0)
    for i, b in enumerate(boxes):
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(b["yaw"]))
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                p=wp.vec3(float(b["cx"]), float(b["cy"]), float(b["cz"])),
                q=q,
            ),
            hx=float(b["hx"]),
            hy=float(b["hy"]),
            hz=float(b["hz"]),
            cfg=box_cfg,
            color=(0.86, 0.80, 0.70),
            label="wall_box_%d" % i,
        )
    print("wall collision boxes: %d (thickness 0.08 m, no mesh collider)" % len(boxes))
    return mins, maxs


def add_random_obstacles(
    builder: newton.ModelBuilder,
    mins,
    maxs,
    n=6,
    seed=3,
    static=False,
    clutter="random",
    spawn=None,
    people_xy=None,
):
    """Couches/desks/tables with vacuum-width aisles. `n` kept for old callers."""
    del n, mins, maxs
    from room_clutter import add_furniture_shapes, layout_furniture

    pieces = layout_furniture(
        seed=int(seed), spec=clutter, spawn=spawn, people_xy=people_xy,
    )
    add_furniture_shapes(builder, pieces, static=static)
    add_random_obstacles.placed = pieces
    add_random_obstacles.spec = getattr(layout_furniture, "spec", clutter)
    add_random_obstacles.rooms = getattr(layout_furniture, "rooms", "")
    print(
        "clutter spec=%s n=%d %s"
        % (add_random_obstacles.spec, len(pieces), add_random_obstacles.rooms),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", default="gl", choices=("gl", "rerun", "null"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--stl", default=str(DEFAULT_STL))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fwd", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--urdf", action="store_true",
                    help="Build Kevin from MakerBeam XL URDF twin instead of box chassis")
    ap.add_argument("--urdf-path", default="",
                    help="Optional override path to kevin.urdf")
    ap.add_argument("--urdf-free-casters", action="store_true",
                    help="Do not freeze caster continuous joints (default: freeze)")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--capture-fps", type=float, default=6.0)
    ap.add_argument("--gif", default="")
    ap.add_argument("--record-3d", default="",
                    help="Newton ViewerGL chase-cam MP4 path (3D primary; skips 2D HouseView)")
    ap.add_argument("--no-2d-view", action="store_true",
                    help="Skip Kevin LIVE birdseye windows")
    ap.add_argument("--head-on", action="store_true")
    ap.add_argument(
        "--clutter",
        default="random",
        help="empty|sparse|medium|dense|packed|random  (per-room mix if random)",
    )
    args = ap.parse_args()

    if args.drive:
        from newton_drive import run_perception_drive
        run_perception_drive(args)
        return

    wp.init()
    builder = newton.ModelBuilder()
    mins, maxs = add_house(builder, Path(args.stl))
    add_random_obstacles(builder, mins, maxs, seed=args.seed)
    spawn = (float(mins[0]) + 1.2, float(mins[1]) + 1.2)
    build_kevin(builder, start_xy=spawn)
    builder.add_ground_plane()
    model = builder.finalize(device=args.device)
    print("built", spawn, "shapes", model.shape_count)


if __name__ == "__main__":
    main()
