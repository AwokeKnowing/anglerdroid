"""URDF FK helpers for the Kevin MakerBeam XL twin (sim-local).

Reads joint origins from kevin.urdf and composes fixed-chain transforms to
base_link. Used to wire RS1/RS2/(optional RGB) poses into SimRsCameras without
hardcoding twin numbers in newton_drive.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# resolved robot config (src/robot_params.py): the selected wheels' twin URDF and radius.
import sys as _sys
if str(REPO / 'src') not in _sys.path: _sys.path.append(str(REPO / 'src'))
import robot_params as _RP
DEFAULT_URDF = REPO / "sim" / "kevin-urdf-twin" / _RP.V["twin_urdf"]

# Names in the twin (RS1 = mast D435 looking down; RS2 = mid mast D435i).
RS1_BODY = "realsense_d435"
RS1_DEPTH = "realsense_d435_depth"
RS1_OPTICAL = "realsense_d435_depth_optical"
RS2_BODY = "realsense_d435i"
RS2_DEPTH = "realsense_d435i_depth"
RS2_OPTICAL = "realsense_d435i_depth_optical"
RGB_BODY = "rgb_camera"
RGB_OPTICAL = "rgb_camera_optical"
WHEEL_LEFT = "wheel_left"
WHEEL_RIGHT = "wheel_right"


def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _xyz_rpy(el: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    origin = el.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float64), np.eye(3)
    xyz = np.array([float(x) for x in (origin.get("xyz") or "0 0 0").split()], dtype=np.float64)
    rpy = [float(x) for x in (origin.get("rpy") or "0 0 0").split()]
    return xyz, _rpy_to_R(*rpy)


def load_joint_tree(urdf_path: Path | None = None) -> Dict[str, tuple]:
    """child_link -> (parent_link, R_parent_child, t_parent_child, joint_name, joint_type)."""
    path = Path(urdf_path or DEFAULT_URDF)
    root = ET.parse(path).getroot()
    tree: Dict[str, tuple] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        t, R = _xyz_rpy(joint)
        tree[child] = (parent, R, t, joint.get("name"), joint.get("type"))
    return tree


def fk_link(link: str, tree: Dict[str, tuple] | None = None, base: str = "base_link"):
    """Pose of `link` in `base` frame. Returns (t, R) with p_base = R @ p_link + t."""
    if tree is None:
        tree = load_joint_tree()
    R = np.eye(3)
    t = np.zeros(3, dtype=np.float64)
    cur = link
    guard = 0
    while cur != base:
        if cur not in tree:
            raise KeyError("link %r not connected to %r" % (link, base))
        parent, Rj, tj, _jn, _jt = tree[cur]
        t = Rj @ t + tj
        R = Rj @ R
        cur = parent
        guard += 1
        if guard > 256:
            raise RuntimeError("FK cycle while resolving %s" % link)
    return t, R


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = body axes) -> xyzw quaternion."""
    m = np.asarray(R, dtype=np.float64)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else q


def optical_to_sim_cam_R(R_optical: np.ndarray) -> np.ndarray:
    """Map ROS optical (+Z look) to SimRsCameras convention (look = -Z_cam).

    SimRsCameras builds pinhole rays with cam_z = -1, so camera +Z points
    opposite the look direction. ROS optical +Z is the look axis.
    """
    # Z_cam = -Z_opt; keep X_opt; Y_cam = Z_cam x X_cam
    X = np.asarray(R_optical[:, 0], dtype=np.float64).copy()
    Z_opt = np.asarray(R_optical[:, 2], dtype=np.float64)
    Z = -Z_opt
    Y = np.cross(Z, X)
    yn = float(np.linalg.norm(Y))
    if yn < 1e-9:
        Y = np.asarray(R_optical[:, 1], dtype=np.float64)
        X = np.cross(Y, Z)
        xn = float(np.linalg.norm(X))
        X = X / xn if xn else X
        Y = np.cross(Z, X)
    else:
        Y = Y / yn
        X = np.cross(Y, Z)
        xn = float(np.linalg.norm(X))
        X = X / xn if xn else X
    return np.column_stack((X, Y, Z))


def camera_poses(urdf_path: Path | None = None) -> dict:
    """Body-frame camera poses for SimRsCameras (p + quat_xyzw, look=-Z)."""
    tree = load_joint_tree(urdf_path)
    out = {}
    for key, link in (
        ("rs1", RS1_OPTICAL),
        ("rs2", RS2_OPTICAL),
        ("rgb", RGB_OPTICAL),
        ("rs1_body", RS1_BODY),
        ("rs2_body", RS2_BODY),
        ("rs1_depth", RS1_DEPTH),
        ("rs2_depth", RS2_DEPTH),
    ):
        t, R = fk_link(link, tree)
        if key in ("rs1", "rs2", "rgb"):
            R_cam = optical_to_sim_cam_R(R)
            look = -R_cam[:, 2]
            out[key] = {
                "link": link,
                "p": t.astype(np.float64),
                "q": rot_to_quat_xyzw(R_cam),
                "R": R_cam,
                "look": look,
                "z": float(t[2]),
            }
        else:
            out[key] = {"link": link, "p": t.astype(np.float64), "R": R}
    # Pitch of RS2 look about +Y (positive = nose down).
    look2 = out["rs2"]["look"]
    out["rs2"]["pitch_down_deg"] = float(math.degrees(math.atan2(-look2[2], max(1e-9, look2[0]))))
    return out


def wheel_geometry(urdf_path: Path | None = None) -> dict:
    """Track (hub-to-hub), wheel radius from collision cylinder, hub y."""
    path = Path(urdf_path or DEFAULT_URDF)
    root = ET.parse(path).getroot()
    tree = load_joint_tree(path)
    t_l, _ = fk_link(WHEEL_LEFT, tree)
    t_r, _ = fk_link(WHEEL_RIGHT, tree)
    radius = None
    for link in root.findall("link"):
        if link.get("name") not in (WHEEL_LEFT, WHEEL_RIGHT):
            continue
        col = link.find("collision")
        if col is None:
            continue
        cyl = col.find("geometry/cylinder")
        if cyl is not None:
            radius = float(cyl.get("radius"))
            break
    track = float(abs(t_l[1] - t_r[1]))
    return {
        "track": track,
        "wheel_r": float(radius if radius is not None else _RP.V["wheel_r_m"]),
        "y_left": float(t_l[1]),
        "y_right": float(t_r[1]),
        "hub_xyz_left": t_l.astype(np.float64),
        "hub_xyz_right": t_r.astype(np.float64),
    }


def body_extents(urdf_path: Path | None = None) -> dict:
    """Approx outer AABB from beam_* box visuals + known front/rear faces."""
    path = Path(urdf_path or DEFAULT_URDF)
    root = ET.parse(path).getroot()
    tree = load_joint_tree(path)
    xs, ys, zs = [], [], []
    for link in root.findall("link"):
        name = link.get("name") or ""
        if not (name.startswith("beam_") or name in ("base_link",)):
            continue
        try:
            t, _ = fk_link(name, tree) if name != "base_link" else (np.zeros(3), np.eye(3))
        except KeyError:
            continue
        for vis in link.findall("visual"):
            box = vis.find("geometry/box")
            if box is None:
                continue
            sx, sy, sz = [float(x) for x in box.get("size").split()]
            vt, _ = _xyz_rpy(vis)
            c = t + vt
            xs += [c[0] - sx * 0.5, c[0] + sx * 0.5]
            ys += [c[1] - sy * 0.5, c[1] + sy * 0.5]
            zs += [c[2] - sz * 0.5, c[2] + sz * 0.5]
    if not xs:
        return {"front_x": 0.115, "rear_x": -0.165, "half_w": 0.15, "height": 0.15}
    return {
        "front_x": float(max(xs)),
        "rear_x": float(min(xs)),
        "half_w": float(max(abs(min(ys)), abs(max(ys)))),
        "height": float(max(zs) - min(zs)),
        "z_min": float(min(zs)),
        "z_max": float(max(zs)),
        "length": float(max(xs) - min(xs)),
        "width": float(max(ys) - min(ys)),
    }


def audit_collisions_inertials(urdf_path: Path | None = None) -> dict:
    path = Path(urdf_path or DEFAULT_URDF)
    root = ET.parse(path).getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    n_col = sum(1 for lk in links for _ in lk.findall("collision"))
    n_vis = sum(1 for lk in links for _ in lk.findall("visual"))
    missing_inertial = [lk.get("name") for lk in links if lk.find("inertial") is None]
    continuous = [j.get("name") for j in joints if j.get("type") == "continuous"]
    fixed = [j.get("name") for j in joints if j.get("type") == "fixed"]
    col_links = [lk.get("name") for lk in links if lk.find("collision") is not None]
    return {
        "n_links": len(links),
        "n_joints": len(joints),
        "n_fixed": len(fixed),
        "n_continuous": len(continuous),
        "continuous": continuous,
        "n_collision_geoms": n_col,
        "n_visual_geoms": n_vis,
        "collision_links": col_links,
        "missing_inertial": missing_inertial,
    }


if __name__ == "__main__":
    cams = camera_poses()
    wheels = wheel_geometry()
    body = body_extents()
    audit = audit_collisions_inertials()
    print("RS1 z=%.4f look=%s" % (cams["rs1"]["z"], cams["rs1"]["look"]))
    print("RS2 z=%.4f pitch_down=%.2f look=%s" % (
        cams["rs2"]["z"], cams["rs2"]["pitch_down_deg"], cams["rs2"]["look"]))
    print("wheels", wheels)
    print("body", body)
    print("audit col=%d missing_inertial=%d continuous=%s" % (
        audit["n_collision_geoms"], len(audit["missing_inertial"]), audit["continuous"]))
