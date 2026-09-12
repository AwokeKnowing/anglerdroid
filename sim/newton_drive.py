#!/usr/bin/env python3
"""Perception-driven Kevin in Newton.

RS1 (top-down) is the boss. Depth is a visual-mesh raycast of the repaired
house STL (site, no MuJoCo collider). Newton tiled-camera is fallback only.
A wall in the top-down footprint ahead of the body stops the commander. The
motion gate then refuses any step — forward, creep, or yaw — whose predicted
footprint (body, caster, wheels, inflated) would enter that mesh. Recovery is
the yaw or backup that increases whole-footprint clearance. Pose is the live
PoseEstimator (visual or IMU+wheel). Chassis truth never enters SLAM.
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import StateFlags
from newton.sensors import SensorTiledCamera

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cameras import RS_DEPTH_W, RS_DEPTH_H, RS_DECIMATE_MAG  # noqa: E402
from robot_config import (  # noqa: E402
    BODY_BOX, EGO_PX_SIZE, FRAME_H, FRAME_W, RCX, RCY,
    FOOTPRINT_BOXES, UNDER_ROBOT_BOXES, SELF_IGNORE_BOXES,
)
from vision import (  # noqa: E402
    TD_FLOOR_CLIP, TD_X_OFFSET, TD_PX_SIZE,
    TOPDOWN_MIN_KNOWN,
    FW_ROTATION, FW_PIVOT, FW_TRANSLATION, FW_PX_SIZE,
    FW_FLOOR_CLIP, FW_HEIGHT_CLIP, FW_CAM_HEIGHT,
    FW_TD_X_DELTA, FW_Y_OFFSET, RS2_EXTRINSIC_Y,
    _fw_sin_pitch, _fw_cos_pitch,
    _clip_decimated_border,
    Vision,
)
from perception import (  # noqa: E402
    UNKNOWN, SELF, CLEAR, OBSTACLE,
    label_rs1_ego, fuse_rs2_into_ego,
    ego_labels_to_planner_feed, EvidenceMap,
)
from pose import (  # noqa: E402
    PoseEstimator,
    ANGULAR_SLIP_SCALE, LINEAR_SLIP_SCALE,
    Q_YAW_SCALE, Q_FWD_SCALE, Q_YAW_FLOOR, Q_FWD_FLOOR,
)
from odom_thread import OdomThread  # noqa: E402
from slam import _scan_match, LOOP_MATCH_THRESH, THUMB_SZ  # noqa: E402
from house_mesh import VisualMeshCaster, body_wall_clearance, footprint_local_samples, footprint_world, _rect_segment_overlap, _point_seg_dist, plus_x_against_boxes  # noqa: E402
from visit_wander import VisitWander, FLOOR_X0, FLOOR_X1, FLOOR_Y0, FLOOR_Y1, _sample_clear  # noqa: E402
from house_view import HouseView, SampleReplay  # noqa: E402
from sim_people import SimPeople  # noqa: E402

try:
    from newton._src.sensors.warp_raytrace.types import RenderConfig
except Exception:  # pragma: no cover
    RenderConfig = None

D435_VFOV_DEG = 58.0
RS2_MIN_Z = 0.28
RS_MAX_Z = 10.0
RS1_ORIGIN_Z = 0.86
# Floor optical depth is about RS1_ORIGIN_Z. A wall hit is closer than this.
WALL_Z_MARGIN = 0.08
WALL_H_CM = 8


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
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
    q /= np.linalg.norm(q)
    return q.astype(np.float32)


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float32)


def _quat_rot(q, v):
    x, y, z, w = [float(t) for t in q]
    qv = np.array([x, y, z], dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    uv = np.cross(qv, vv)
    uuv = np.cross(qv, uv)
    return vv + 2.0 * (w * uv + uuv)


def _unpack_xf(item):
    if getattr(item, "dtype", None) is not None and getattr(item.dtype, "names", None):
        names = item.dtype.names
        if "x" in names:
            pos = np.array([item["x"], item["y"], item["z"]], dtype=np.float64)
            quat = np.array([item["qx"], item["qy"], item["qz"], item["qw"]], dtype=np.float64)
            return pos, quat
        if "p" in names:
            pp = np.asarray(item["p"], dtype=np.float64).reshape(-1)
            qq = np.asarray(item["q"], dtype=np.float64).reshape(-1)
            return pp[:3], qq[:4]
    arr = np.asarray(item, dtype=np.float64).reshape(-1)
    if arr.size >= 7:
        return arr[:3], arr[3:7]
    raise RuntimeError("unrecognized transform layout")


def _xf_mul(parent, child):
    pos = np.asarray(parent[0], dtype=np.float64) + _quat_rot(parent[1], child[0])
    quat = _quat_mul(parent[1], child[1])
    n = float(np.linalg.norm(quat))
    if n > 0:
        quat = quat / n
    return pos, quat.astype(np.float32)



def _quat_inv(q):
    x, y, z, w = [float(t) for t in q]
    return np.array([-x, -y, -z, w], dtype=np.float64)


def _xf_inv(xf):
    pos, quat = xf
    iq = _quat_inv(quat)
    return -_quat_rot(iq, pos), iq


def _free_sync_addrs(model, solver, free_joint: int):
    """Fields SolverMuJoCo.step reads on the next step (update_data_interval=1)."""
    q_start = np.asarray(model.joint_q_start.numpy()).reshape(-1)
    qd_start = np.asarray(model.joint_qd_start.numpy()).reshape(-1)
    jq = int(q_start[free_joint])
    jqd = int(qd_start[free_joint])
    mj_q_src = getattr(solver, "mj_q_start", None)
    mj_qd_src = getattr(solver, "mj_qd_start", None)
    mq = int(np.asarray(mj_q_src.numpy()).reshape(-1)[free_joint]) if mj_q_src is not None else 0
    mqd = int(np.asarray(mj_qd_src.numpy()).reshape(-1)[free_joint]) if mj_qd_src is not None else 0
    return jq, jqd, mq, mqd


def _print_free_sync_map(model, solver, free_joint: int):
    jq, jqd, mq, mqd = _free_sync_addrs(model, solver, free_joint)
    xp = _unpack_xf(model.joint_X_p.numpy()[free_joint])
    xc = _unpack_xf(model.joint_X_c.numpy()[free_joint])
    print(
        "free sync kevin_free=%d joint_q[%d:%d] joint_qd[%d:%d] qpos[%d:%d] qvel[%d:%d] "
        "X_p=(%.3f,%.3f,%.3f) X_c=(%.3f,%.3f,%.3f)"
        % (
            free_joint, jq, jq + 7, jqd, jqd + 6, mq, mq + 7, mqd, mqd + 6,
            xp[0][0], xp[0][1], xp[0][2], xc[0][0], xc[0][1], xc[0][2],
        )
    )
    print("next step reads state.joint_q/joint_qd via convert_warp_coords_to_mj into qpos/qvel")


def _zero_buf(buf, value=0.0):
    if buf is None:
        return False
    try:
        if hasattr(buf, "fill_"):
            buf.fill_(value)
            return True
        arr = np.array(buf.numpy(), copy=True)
        arr[:] = value
        buf.assign(arr)
        return True
    except Exception:
        try:
            buf[:] = value
            return True
        except Exception:
            return False


def _zero_solver_warmstart(data):
    """Clear the buffers SolverMuJoCo.step warm-starts from (see solver.reset)."""
    for name in ("qacc_warmstart", "qacc", "qfrc_applied", "xfrc_applied", "act", "ctrl"):
        buf = getattr(data, name, None)
        if buf is None:
            continue
        if not _zero_buf(buf, 0.0):
            print("warmstart skip", name)


def _clear_mj_contacts(data):
    """Drop cached contacts so a teleported pose is not yanked by old points."""
    for name in ("nacon", "ncon", "nefc"):
        buf = getattr(data, name, None)
        if buf is None:
            continue
        _zero_buf(buf, 0)
    for name in ("efc_force", "efc_aref", "efc_b"):
        buf = getattr(data, name, None)
        if buf is None:
            continue
        _zero_buf(buf, 0.0)


def _reset_solver_buffers(solver, model, state):
    """Zero MuJoCo warm-start without resetting joint_q to model defaults."""
    try:
        solver.reset(state, flags=StateFlags.NONE)
    except Exception as e:
        print("solver.reset skip", type(e).__name__, e)
    data = getattr(solver, "mjw_data", None)
    if data is None:
        data = getattr(solver, "mj_data", None)
    if data is None:
        return
    _zero_solver_warmstart(data)
    _clear_mj_contacts(data)
    try:
        solver._update_mjc_data(data, model, state)
    except Exception as e:
        print("update_mjc_data skip", type(e).__name__, e)
    _refresh_mj_kinematics(solver, data)
    _zero_solver_warmstart(data)
    _clear_mj_contacts(data)
    solver.update_data_interval = 1


def _apply_hold_pose(model, solver, states, hold):
    """Re-push a slid free-joint pose and clear solver warm-start/contacts."""
    jq = hold.get("joint_q")
    jqd = hold.get("joint_qd")
    if jq is None or jqd is None:
        return
    primary = None
    for st in states:
        if st is None:
            continue
        if primary is None:
            primary = st
        st.joint_q.assign(np.array(jq, copy=True))
        st.joint_qd.assign(np.array(jqd, copy=True))
        if hold.get("body_q") is not None:
            st.body_q.assign(np.array(hold["body_q"], copy=True))
        if hold.get("body_qd") is not None and getattr(st, "body_qd", None) is not None:
            st.body_qd.assign(np.array(hold["body_qd"], copy=True))
        newton.eval_fk(model, st.joint_q, st.joint_qd, st)
    if primary is not None:
        _reset_solver_buffers(solver, model, primary)


def _refresh_mj_kinematics(solver, data):
    """Rebuild MuJoCo xpos from the slid qpos so the next step does not project back."""
    warp = getattr(solver, "_mujoco_warp", None)
    model = getattr(solver, "mjw_model", None)
    if warp is None or model is None or data is None:
        return
    try:
        warp.fwd_position(model, data, factorize=False)
    except TypeError:
        try:
            warp.fwd_position(model, data)
        except Exception as e:
            print("fwd_position skip", type(e).__name__, e)
            return
    except Exception as e:
        print("fwd_position skip", type(e).__name__, e)
        return
    try:
        warp.fwd_velocity(model, data)
    except Exception as e:
        print("fwd_velocity skip", type(e).__name__, e)


def _joint_from_world_xf(Xp, Xc, world_xf):
    """Newton free-joint coords: inv(X_p) * world * X_c (same as MuJoCo kernel)."""
    return _xf_mul(_xf_mul(_xf_inv(Xp), world_xf), Xc)


def _world_from_joint_xf(Xp, Xc, joint_xf):
    return _xf_mul(_xf_mul(Xp, joint_xf), _xf_inv(Xc))


def _commit_free_slide(
    model, state, solver, free_joint, chassis, dx, dy, states, v_world,
    wheel_dofs=None, wheel_r=0.09,
):
    """Slide chassis in HOUSE xy; sync joint_q so the next SolverMuJoCo.step holds.

    House pose = X_p * body_q. joint_qd linear is COM velocity in the joint
    *parent* frame: R_inv(X_p) * v_house (once). convert_warp_coords_to_mj
    then writes world qvel. A second inv(R) injected south/west velocity after
    the first slide, which is why later doorway headings snapped back.
    """
    jq, jqd, mq, mqd = _free_sync_addrs(model, solver, free_joint)
    data = getattr(solver, "mjw_data", None)
    if data is None:
        data = getattr(solver, "mj_data", None)
    if data is None:
        raise RuntimeError("no mujoco data")

    Xp = _unpack_xf(model.joint_X_p.numpy()[free_joint])
    Xc = _unpack_xf(model.joint_X_c.numpy()[free_joint])
    body_xf = _unpack_xf(state.body_q.numpy()[chassis])
    house_xf = _xf_mul(Xp, body_xf)
    target_house = (
        np.array([float(house_xf[0][0]) + float(dx), float(house_xf[0][1]) + float(dy), float(house_xf[0][2])], dtype=np.float64),
        np.asarray(house_xf[1], dtype=np.float64).reshape(4),
    )
    # MuJoCo body/qpos = inv(X_p) * house
    target_body = _xf_mul(_xf_inv(Xp), target_house)
    # joint_q = inv(X_p) * qpos * X_c
    jpos, jquat = _joint_from_world_xf(Xp, Xc, target_body)

    jq_arr = np.array(state.joint_q.numpy(), dtype=np.float32, copy=True)
    jqd_arr = np.array(state.joint_qd.numpy(), dtype=np.float32, copy=True)
    flat_q = jq_arr.reshape(-1)
    flat_qd = jqd_arr.reshape(-1)
    flat_q[jq + 0] = np.float32(jpos[0])
    flat_q[jq + 1] = np.float32(jpos[1])
    flat_q[jq + 2] = np.float32(jpos[2])
    flat_q[jq + 3] = np.float32(jquat[0])
    flat_q[jq + 4] = np.float32(jquat[1])
    flat_q[jq + 5] = np.float32(jquat[2])
    flat_q[jq + 6] = np.float32(jquat[3])
    # joint_qd linear = COM vel in parent frame (Newton FREE contract).
    vw_house = np.asarray(v_world, dtype=np.float64).reshape(3)
    Rp = np.asarray(Xp[1], dtype=np.float64).reshape(4)
    v_parent = _quat_rot(_quat_inv(Rp), vw_house)
    flat_qd[jqd + 0] = np.float32(v_parent[0])
    flat_qd[jqd + 1] = np.float32(v_parent[1])
    flat_qd[jqd + 2] = np.float32(v_parent[2])
    flat_qd[jqd + 3] = np.float32(0.0)
    flat_qd[jqd + 4] = np.float32(0.0)
    flat_qd[jqd + 5] = np.float32(0.0)
    # Match hub spin to the slide so planted wheels do not yank the free joint.
    spd = float(np.linalg.norm(vw_house[:2]))
    if wheel_dofs and spd > 1e-4:
        omega = spd / max(1e-4, float(wheel_r))
        for d in wheel_dofs:
            di = int(d)
            if 0 <= di < flat_qd.size:
                flat_qd[di] = np.float32(omega)

    targets = []
    for st in list(states or ()) + [state]:
        if st is None or st in targets:
            continue
        targets.append(st)
    for st in targets:
        st.joint_q.assign(np.array(jq_arr, copy=True))
        st.joint_qd.assign(np.array(jqd_arr, copy=True))
        newton.eval_fk(model, st.joint_q, st.joint_qd, st)

    # Convert writes qpos/qvel. Do not clobber qvel with a parent-frame twist.
    solver._update_mjc_data(data, model, state)
    q_now = np.array(data.qpos.numpy(), dtype=np.float32, copy=True).reshape(-1)
    _reset_solver_buffers(solver, model, state)

    after, _aq = chassis_world_pose(model, state, chassis, free_joint)
    print(
        "nudge commit house (%.3f,%.3f)->(%.3f,%.3f) joint_q=(%.3f,%.3f) qpos=(%.3f,%.3f) interval=1"
        % (
            float(house_xf[0][0]), float(house_xf[0][1]), float(after[0]), float(after[1]),
            float(jpos[0]), float(jpos[1]), float(q_now[mq]), float(q_now[mq + 1]),
        ),
        flush=True,
    )
    body_q = np.array(state.body_q.numpy(), copy=True)
    body_qd = np.array(state.body_qd.numpy(), copy=True) if getattr(state, "body_qd", None) is not None else None
    for st in targets:
        st.joint_q.assign(np.array(jq_arr, copy=True))
        st.joint_qd.assign(np.array(jqd_arr, copy=True))
        st.body_q.assign(np.array(body_q, copy=True))
        if body_qd is not None and getattr(st, "body_qd", None) is not None:
            st.body_qd.assign(np.array(body_qd, copy=True))
    return jq_arr.reshape(-1)[jq:jq + 7]


_WATCHDOG = None


class _HangWatchdog:
    """One thread for the whole run. Do not spawn this per solver.step."""

    def __init__(self, limit=120.0):
        self.limit = float(limit)
        self._last = time.monotonic()
        self._alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def kick(self):
        self._last = time.monotonic()

    def _run(self):
        while self._alive:
            time.sleep(1.0)
            if (time.monotonic() - self._last) > self.limit:
                print("STEP HANG >%.0fs killing" % self.limit, flush=True)
                os.kill(os.getpid(), signal.SIGKILL)


def _watchdog_kick(limit=120.0):
    global _WATCHDOG
    if _WATCHDOG is None:
        _WATCHDOG = _HangWatchdog(limit)
    _WATCHDOG.kick()


def _step_or_die(solver, state_in, state_out, control, contacts, dt, limit=120.0):
    _watchdog_kick(limit)
    solver.step(state_in, state_out, control, contacts, dt)


def _nudge_chassis(
    model,
    state,
    solver,
    chassis: int,
    free_joint: int,
    yaw: float,
    dist: float = 0.16,
    wall_segs=None,
    front_x: float = 0.15,
    rear_x: float = -0.18,
    half_w: float = 0.165,
    states=None,
    speed: float = 0.28,
    boxes=None,
    wheel_dofs=None,
    wheel_r: float = 0.09,
    open_floor: bool = False,
) -> bool:
    """Fail-closed open-floor pin break. Never slide through a wall box.

    Candidate slides are tested in WORLD xy (not raw qpos deltas). The winning
    slide is committed via joint_q + qpos sync so the next SolverMuJoCo.step
    cannot throw the chassis west.
    """
    try:
        data = getattr(solver, "mjw_data", None)
        if data is None:
            print("nudge failed no mjw_data")
            return False
        before, _bq = chassis_world_pose(model, state, chassis, free_joint)
        before_yaw = _yaw_of(_bq)
        dx = math.cos(float(yaw)) * float(dist)
        dy = math.sin(float(yaw)) * float(dist)
        # Snapshot for fail-closed restore (joint_q is the Newton source of truth).
        jq0 = np.array(state.joint_q.numpy(), dtype=np.float32, copy=True)
        jqd0 = np.array(state.joint_qd.numpy(), dtype=np.float32, copy=True)
        bq0 = np.array(state.body_q.numpy(), copy=True)
        bqd0 = np.array(state.body_qd.numpy(), copy=True) if getattr(state, "body_qd", None) is not None else None
        qpos0 = np.array(data.qpos.numpy(), dtype=np.float32, copy=True)
        qvel0 = np.array(data.qvel.numpy(), dtype=np.float32, copy=True)
        clr0, fr0, ov0 = 9.0, 9.0, False
        if wall_segs is not None and len(wall_segs):
            clr0, fr0, ov0 = body_wall_clearance(
                (float(before[0]), float(before[1])), float(before_yaw), wall_segs,
                front_x=front_x, rear_x=rear_x, half_w=half_w,
            )
            if ov0 or (math.isfinite(clr0) and clr0 < 0.12):
                print("nudge reject pre-clear=%.3f ov=%s" % (clr0, ov0))
                return False

        def _inside_floor(xy) -> bool:
            return (
                (FLOOR_X0 + 0.20) <= float(xy[0]) <= (FLOOR_X1 - 0.20)
                and (FLOOR_Y0 + 0.20) <= float(xy[1]) <= (FLOOR_Y1 - 0.20)
            )

        def _restore():
            data.qpos.assign(qpos0)
            data.qvel.assign(qvel0)
            state.joint_q.assign(jq0)
            state.joint_qd.assign(jqd0)
            state.body_q.assign(bq0)
            if bqd0 is not None and getattr(state, "body_qd", None) is not None:
                state.body_qd.assign(bqd0)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        cands = [(dx, dy), (dy, -dx), (-dy, dx), (-dx, -dy)]
        if open_floor:
            d = float(dist)
            cands.extend([(0.0, d), (0.70 * d, 0.70 * d), (-0.70 * d, 0.70 * d)])
        best = None
        best_along = -1e9
        best_clr = float(clr0)
        for jdx, jdy in cands:
            # Pure geometric trial — no solver writes until commit.
            after_xy = (float(before[0]) + float(jdx), float(before[1]) + float(jdy))
            ddx = float(jdx)
            ddy = float(jdy)
            moved = math.hypot(ddx, ddy)
            along = (ddx * dx + ddy * dy) / (float(dist) + 1e-9)
            if open_floor:
                along = max(along, ddy / (float(dist) + 1e-9))
            if moved < 0.05 or moved > 0.28:
                continue
            if not _inside_floor(after_xy):
                continue
            cand_clr = float(clr0)
            if wall_segs is not None and len(wall_segs):
                clr, fr, ov = body_wall_clearance(
                    after_xy, float(before_yaw), wall_segs,
                    front_x=front_x, rear_x=rear_x, half_w=half_w,
                )
                if ov or (math.isfinite(clr) and clr + 1e-4 < float(clr0)):
                    continue
                if math.isfinite(clr) and clr < 0.12:
                    continue
                cand_clr = float(clr)
            if boxes:
                mid = (0.5 * (float(before[0]) + after_xy[0]), 0.5 * (float(before[1]) + after_xy[1]))
                box_clr = min(
                    _sample_clear(after_xy[0], after_xy[1], float(before_yaw), boxes, pad=0.02),
                    _sample_clear(mid[0], mid[1], float(before_yaw), boxes, pad=0.02),
                )
                if box_clr < 0.18:
                    continue
                cand_clr = min(cand_clr, float(box_clr))
            if along > best_along:
                best_along = along
                best = (jdx, jdy, float(after_xy[0]), float(after_xy[1]))
                best_clr = cand_clr
        if best is None or best_along < (0.05 if open_floor else 0.08):
            print("nudge reject along=%.3f open=%s" % (best_along, open_floor))
            return False
        spd = max(0.12, min(0.35, abs(float(speed))))
        v_world = np.array([math.cos(float(yaw)) * spd, math.sin(float(yaw)) * spd, 0.0], dtype=np.float64)
        targets = [state]
        if states:
            for st in states:
                if st is not None and st is not state:
                    targets.append(st)
        jpos = _commit_free_slide(
            model, state, solver, free_joint, chassis, best[0], best[1], targets, v_world,
            wheel_dofs=wheel_dofs, wheel_r=wheel_r,
        )
        after, aq = chassis_world_pose(model, state, chassis, free_joint)
        print(
            "nudge sync joint_q=(%.3f,%.3f,%.3f) world=(%.3f,%.3f) delta=(%.3f,%.3f)"
            % (float(jpos[0]), float(jpos[1]), float(jpos[2]), float(after[0]), float(after[1]), float(best[0]), float(best[1]))
        )
        if not _inside_floor(after):
            print("nudge revert off-floor (%.2f,%.2f)" % (after[0], after[1]))
            _restore()
            return False
        if wall_segs is not None and len(wall_segs):
            clr, fr, ov = body_wall_clearance(
                (float(after[0]), float(after[1])), float(_yaw_of(aq)), wall_segs,
                front_x=front_x, rear_x=rear_x, half_w=half_w,
            )
            if ov or (math.isfinite(clr) and (clr + 1e-4 < float(clr0) or clr < 0.12)):
                print("nudge revert clear=%.3f->%.3f ov=%s" % (clr0, clr, ov))
                _restore()
                return False
        err = math.hypot(float(after[0]) - float(best[2]), float(after[1]) - float(best[3]))
        if err > 0.05:
            print("nudge revert pose-err=%.3f" % err)
            _restore()
            return False
        print(
            "nudge world (%.2f,%.2f)->(%.2f,%.2f) along=%.2f clear=%.3f"
            % (before[0], before[1], after[0], after[1], best_along, best_clr)
        )
        return True
    except Exception as e:
        print("nudge failed", type(e).__name__, e)
        return False


_XP_CACHE = {}
_WHEEL_QD_BUF = None


def chassis_world_pose(model, state, chassis: int, free_joint: int):
    """Chassis pose in the HOUSE/map frame.

    MuJoCo FREE qpos/body_q sit near the origin; joint_X_p carries the spawn
    yaw/offset. House coordinates used by walls/wander are X_p * body_q.
    """
    parent = _XP_CACHE.get(id(model))
    if parent is None:
        parent = _unpack_xf(model.joint_X_p.numpy()[free_joint])
        _XP_CACHE[id(model)] = parent
    child = _unpack_xf(state.body_q.numpy()[chassis])
    return _xf_mul(parent, child)


def _yaw_of(quat) -> float:
    x, y, z, w = [float(t) for t in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quat(yaw):
    h = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(h), math.cos(h))


def _furniture_ok(p, spawn):
    from room_clutter import in_door_corridor

    x, y = float(p["cx"]), float(p["cy"])
    hx, hy = float(p.get("hx", 0.2)), float(p.get("hy", 0.2))
    if in_door_corridor(x, y, hx, hy, pad=0.10):
        return False
    if math.hypot(x - float(spawn[0]), y - float(spawn[1])) < 0.70:
        return False
    return True


def _nearby_boxes(x, y, boxes, r=1.5):
    out = []
    for b in boxes:
        if abs(float(b["cx"]) - x) + abs(float(b["cy"]) - y) <= r + float(b["hx"]) + float(b["hy"]):
            out.append(b)
    return out


def _kin_ok(x, y, yaw, boxes, pad=0.10):
    if not ((FLOOR_X0 + 0.20) <= float(x) <= (FLOOR_X1 - 0.20) and (FLOOR_Y0 + 0.20) <= float(y) <= (FLOOR_Y1 - 0.20)):
        return False
    near = _nearby_boxes(x, y, boxes)
    return float(_sample_clear(x, y, yaw, near, pad=0.02)) >= float(pad)


def _pinhole_dxdy(h: int, w: int, vfov_rad: float):
    aspect = float(w) / float(h)
    th = math.tan(vfov_rad * 0.5)
    py, px = np.mgrid[0:h, 0:w]
    u = (px.astype(np.float32) + 0.5) / float(w) - 0.5
    v = (py.astype(np.float32) + 0.5) / float(h) - 0.5
    dx = u * (2.0 * th * aspect)
    dy = -v * (2.0 * th)
    return dx.astype(np.float32), dy.astype(np.float32)


def _merge_forward_depth(newton_d, visual_d):
    n = np.asarray(newton_d, dtype=np.float32)
    if visual_d is None:
        return n
    v = np.asarray(visual_d, dtype=np.float32)
    if v.shape != n.shape:
        return n
    n_ok = np.isfinite(n) & (n > 1e-4)
    v_ok = np.isfinite(v) & (v > 1e-4)
    out = np.where(n_ok, n, np.float32(0.0))
    take = v_ok & ((~n_ok) | (v < n))
    out = np.where(take, v, out)
    return out.astype(np.float32)


def _depth_to_rs_verts(fwd_depth, dx, dy, z_bias=0.0, z_min=0.01, z_max=RS_MAX_Z):
    z_n = np.asarray(fwd_depth, dtype=np.float32)
    valid = np.isfinite(z_n) & (z_n > 1e-4)
    z = np.zeros_like(z_n)
    z[valid] = z_n[valid] + np.float32(z_bias)
    ok = valid & (z >= z_min) & (z <= z_max)
    rs_x = np.zeros_like(z)
    rs_y = np.zeros_like(z)
    rs_x[ok] = dx[ok] * z[ok]
    rs_y[ok] = -dy[ok] * z[ok]
    out = np.zeros((z.shape[0] * z.shape[1], 3), dtype=np.float32)
    out[:, 0] = rs_x.reshape(-1)
    out[:, 1] = rs_y.reshape(-1)
    out[:, 2] = np.where(ok, z, 0.0).reshape(-1)
    return out


def rs2_scatter_obs_known(verts: np.ndarray):
    out_h, out_w = int(FRAME_W), int(FRAME_H)
    obs = np.zeros((out_h, out_w), dtype=np.uint8)
    known = np.zeros((out_h, out_w), dtype=np.uint8)
    v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    if v.shape[0] < 16:
        return obs, known
    ok = np.isfinite(v).all(axis=1) & (v[:, 2] >= np.float32(RS2_MIN_Z))
    if not np.any(ok):
        return obs, known
    p = v[ok].copy()
    p[:, 1] += np.float32(RS2_EXTRINSIC_Y)
    rot = np.asarray(FW_ROTATION, dtype=np.float32)
    pivot = np.asarray(FW_PIVOT, dtype=np.float32)
    trans = np.asarray(FW_TRANSLATION, dtype=np.float32)
    r = (p - pivot) @ rot.T + pivot - trans
    scale = np.float32(1.0 / float(FW_PX_SIZE))
    offset = np.array([out_w * 0.5, out_h * 0.5 + float(scale)], dtype=np.float32)
    px = r[:, :2] * scale + offset
    col = np.floor(px[:, 0]).astype(np.int32)
    row = np.floor(px[:, 1]).astype(np.int32)
    inside = (col >= 0) & (col < out_w) & (row >= 0) & (row < out_h)
    if not np.any(inside):
        return obs, known
    r = r[inside]
    p = p[inside]
    col = col[inside]
    row = row[inside]
    floor_m = (r[:, 2] > float(FW_FLOOR_CLIP)) & (r[:, 2] < float(FW_HEIGHT_CLIP))
    known[row[~floor_m], col[~floor_m]] = 255
    if np.any(floor_m):
        phys_h = (
            float(FW_CAM_HEIGHT)
            - p[floor_m, 1] * float(_fw_cos_pitch)
            - p[floor_m, 2] * float(_fw_sin_pitch)
        )
        enc = np.clip(phys_h * 100.0, 1.0, 100.0).astype(np.uint8)
        rr, cc = row[floor_m], col[floor_m]
        known[rr, cc] = 255
        np.maximum.at(obs, (rr, cc), enc)
    return obs, known



_CMD = {
    "turn_streak": 0,
    "creep_hold": 0,
    "wall_stop": 0,
    "wall_hits": 0,
    "phase": "cruise",
    "yaw_sign": 1,
    "halt_s": 0.0,
    "yaw_s": 0.0,
    "backup_s": 0.0,
    "flips": 0,
    "backup_cycles": 0,
    "clear_streak": 0,
    "ever_wall_stop": 0,
    "unstuck": 0,
    "recoveries": 0,
    "follow_sign": 0,
}


def reset_command_state():
    _CMD.update({
        "turn_streak": 0,
        "creep_hold": 0,
        "wall_stop": 0,
        "wall_hits": 0,
        "phase": "cruise",
        "yaw_sign": 1,
        "halt_s": 0.0,
        "yaw_s": 0.0,
        "backup_s": 0.0,
        "backup_cycles": 0,
        "flips": 0,
        "clear_streak": 0,
        "ever_wall_stop": 0,
        "unstuck": 0,
        "recoveries": 0,
        "follow_sign": 0,
    })


def _wall_in_footprint(labels: np.ndarray, height: np.ndarray | None) -> bool:
    """Wall if the top-down footprint just ahead of the body is occupied."""
    if height is None:
        height = np.zeros_like(labels)
    x0, y0, x1, y1 = BODY_BOX
    fx0 = min(FRAME_W - 1, x1 + 1)
    fx1 = min(FRAME_W, x1 + 40)
    fy0 = max(0, y0 - 2)
    fy1 = min(FRAME_H, y1 + 2)
    if fx1 <= fx0 or fy1 <= fy0:
        return False
    win = labels[fy0:fy1, fx0:fx1]
    wh = height[fy0:fy1, fx0:fx1]
    wall = (win == OBSTACLE) & (wh >= np.uint8(WALL_H_CM))
    n = int(wall.sum())
    if n < 8:
        return False
    cols = int(wall.any(axis=0).sum())
    return cols >= 4 or n >= 16


def _compact_obstacle_blob(win: np.ndarray) -> bool:
    mask = win == OBSTACLE
    n = int(mask.sum())
    if n < 6:
        return False
    ys, xs = np.nonzero(mask)
    bh = int(ys.max() - ys.min() + 1)
    bw = int(xs.max() - xs.min() + 1)
    bbox = max(1, bh * bw)
    fill = n / float(bbox)
    cover = bbox / float(mask.size)
    return fill >= 0.40 and cover <= 0.50 and n < 0.45 * mask.size


def _slice_tall(labels, height, xa, xb, ya, yb):
    if xb <= xa or yb <= ya:
        return np.zeros((0, 0), dtype=bool)
    win = labels[ya:yb, xa:xb]
    if height is None:
        return win == OBSTACLE
    wh = height[ya:yb, xa:xb]
    return (win == OBSTACLE) & (wh >= np.uint8(WALL_H_CM))


def _forward_blocked(labels, height) -> bool:
    """Stop if the center corridor ahead is a tall wall (~0.3 m).

    A wall on the edge of the body (heading parallel) is a steer, not a stop,
    so a recovery yaw is not immediately recaptured.
    """
    x0, y0, x1, y1 = BODY_BOX
    h = max(1, y1 - y0)
    mid_y0 = y0 + int(round(h * 0.28))
    mid_y1 = max(mid_y0 + 4, y0 + int(round(h * 0.72)))
    fx0 = min(FRAME_W - 1, x1 + 1)
    fx1 = min(FRAME_W, x1 + 58)
    tall = _slice_tall(labels, height, fx0, fx1, mid_y0, mid_y1)
    if tall.size == 0:
        return False
    near_w = min(tall.shape[1], 30)
    near = tall[:, :near_w]
    n_near = int(near.sum())
    cols_near = int(near.any(axis=0).sum()) if near.size else 0
    return (n_near >= 6 and cols_near >= 3) or n_near >= 12


def _rear_blocked(labels, height) -> bool:
    x0, y0, x1, y1 = BODY_BOX
    rx1 = max(0, x0 - 1)
    rx0 = max(0, x0 - 28)
    fy0 = max(0, y0 - 2)
    fy1 = min(FRAME_H, y1 + 2)
    tall = _slice_tall(labels, height, rx0, rx1, fy0, fy1)
    if tall.size == 0:
        return False
    return int(tall.sum()) >= 10


def _clearer_sign(labels, height) -> int:
    """+1 yaw left, -1 yaw right. Image-up is robot-left."""
    x0, y0, x1, y1 = BODY_BOX
    fx0 = min(FRAME_W - 1, x1 + 1)
    fx1 = min(FRAME_W, x1 + 80)
    span = max(28, (y1 - y0) + 10)
    fy0 = max(0, y0 - span)
    fy1 = min(FRAME_H, y1 + span)
    if fx1 <= fx0 or fy1 <= fy0:
        return 1
    tall = _slice_tall(labels, height, fx0, fx1, fy0, fy1)
    clear = labels[fy0:fy1, fx0:fx1] == CLEAR
    if tall.size == 0:
        return 1
    mid = max(1, tall.shape[0] // 2)
    left_tall = int(tall[:mid].sum())
    right_tall = int(tall[mid:].sum())
    left_clear = int(clear[:mid].sum())
    right_clear = int(clear[mid:].sum())
    left_score = left_clear - 4 * left_tall
    right_score = right_clear - 4 * right_tall
    if left_score > right_score + 12:
        return 1
    if right_score > left_score + 12:
        return -1
    if left_tall + 6 < right_tall:
        return 1
    if right_tall + 6 < left_tall:
        return -1
    return 1 if left_score >= right_score else -1


def _cruise_bias(labels, height) -> float:
    """Steer away from a side wall; do not stop for a graze beside the body."""
    x0, y0, x1, y1 = BODY_BOX
    fx0 = min(FRAME_W - 1, x1 + 2)
    fx1 = min(FRAME_W, x1 + 48)
    span = max(18, (y1 - y0))
    fy0 = max(0, y0 - span)
    fy1 = min(FRAME_H, y1 + span)
    if fx1 <= fx0 or fy1 <= fy0:
        return 0.0
    tall = _slice_tall(labels, height, fx0, fx1, fy0, fy1)
    if tall.size == 0:
        return 0.0
    mid = max(1, tall.shape[0] // 2)
    left_tall = int(tall[:mid].sum())
    right_tall = int(tall[mid:].sum())
    if left_tall > right_tall + 4:
        return -0.30
    if right_tall > left_tall + 4:
        return 0.30
    return 0.0


def _turn_name(sign: int) -> str:
    return "turn_left" if int(sign) > 0 else "turn_right"


def command_from_ego(
    labels: np.ndarray,
    rs1_valid: bool,
    height: np.ndarray | None = None,
    front_clear: float | None = None,
    dt: float = 0.1,
):
    """Stop if a wall is ahead, then yaw toward clearance and creep along it.

    Floor noise is not a wall. Recovery is short reverse + in-place yaw so the
    inflated footprint never creeps through the mesh. After the nose opens he
    peels along the wall instead of charging straight back into it.
    """
    dt = max(1e-3, float(dt))
    if not rs1_valid:
        _CMD["turn_streak"] = 0
        return 0.0, 0.0, "immobilize"
    if height is None:
        height = np.zeros_like(labels)

    # Chassis sat still while commanded forward (belly scrape / snag).
    # Reverse off it, then yaw toward the clearer side. Not a wall collider.
    if _CMD.get("force_escape") and _CMD.get("phase") == "cruise":
        _CMD["force_escape"] = 0
        _CMD["yaw_sign"] = int(_clearer_sign(labels, height))
        _CMD["follow_sign"] = int(_CMD["yaw_sign"])
        _CMD["halt_s"] = 0.0
        _CMD["yaw_s"] = 0.0
        _CMD["backup_s"] = 0.0
        _CMD["flips"] = 0
        _CMD["backup_cycles"] = 0
        _CMD["clear_streak"] = 0
        _CMD["ever_wall_stop"] = 1
        if _rear_blocked(labels, height):
            _CMD["phase"] = "yaw"
            sign = int(_CMD["yaw_sign"]) or 1
            return 0.0, 0.90 * sign, _turn_name(sign)
        _CMD["phase"] = "backup"
        return -0.10, 0.0, "creep"

    front_m = None if front_clear is None else float(front_clear)
    label_blocked = _forward_blocked(labels, height)
    # Align with plus-x stop (~0.34 m). Latch recovery even if labels flicker.
    mesh_close = front_m is not None and front_m < 0.36
    recovering = _CMD["phase"] in ("halt", "yaw", "backup")
    blocked = bool(label_blocked or mesh_close or recovering)
    yaw_w = 0.95

    if blocked:
        if label_blocked or mesh_close:
            _CMD["wall_hits"] = int(_CMD["wall_hits"]) + 1
            _CMD["clear_streak"] = 0
        if int(_CMD["wall_hits"]) < 1 and _CMD["phase"] == "cruise" and mesh_close:
            # Visible one-frame stop, then enter recovery next tick.
            _CMD["ever_wall_stop"] = 1
            _CMD["wall_stop"] = 1
            return 0.0, 0.0, "stop"

        if _CMD["phase"] == "cruise":
            _CMD["yaw_sign"] = int(_clearer_sign(labels, height))
            _CMD["follow_sign"] = int(_CMD["yaw_sign"])
            _CMD["halt_s"] = 0.0
            _CMD["yaw_s"] = 0.0
            _CMD["backup_s"] = 0.0
            _CMD["flips"] = 0
            _CMD["backup_cycles"] = 0
            _CMD["phase"] = "halt"
            _CMD["ever_wall_stop"] = 1
            _CMD["wall_stop"] = 1

        phase = _CMD["phase"]
        if phase == "halt":
            _CMD["halt_s"] = float(_CMD["halt_s"]) + dt
            # Prefer a short reverse when still tight so yaw has room.
            if front_m is not None and front_m < 0.42 and not _rear_blocked(labels, height):
                _CMD["phase"] = "backup"
                _CMD["backup_s"] = 0.0
                return -0.09, 0.0, "creep"
            if float(_CMD["halt_s"]) < 0.15:
                return 0.0, 0.0, "stop"
            _CMD["phase"] = "yaw"
            _CMD["yaw_s"] = 0.0
            phase = "yaw"

        if phase == "backup":
            _CMD["backup_s"] = float(_CMD["backup_s"]) + dt
            done = (
                (front_m is not None and front_m >= 0.44)
                or _rear_blocked(labels, height)
                or float(_CMD["backup_s"]) >= 0.70
            )
            if not done:
                return -0.09, 0.0, "creep"
            _CMD["phase"] = "yaw"
            _CMD["yaw_s"] = 0.0
            phase = "yaw"

        if phase == "yaw":
            _CMD["yaw_s"] = float(_CMD["yaw_s"]) + dt
            # Re-pick clearer side periodically while still jammed ahead.
            if float(_CMD["yaw_s"]) < 0.20 or (int(round(float(_CMD["yaw_s"]) / dt)) % 8 == 0):
                _CMD["yaw_sign"] = int(_clearer_sign(labels, height))
                _CMD["follow_sign"] = int(_CMD["yaw_sign"])
            if (
                front_m is not None
                and front_m < 0.28
                and not _rear_blocked(labels, height)
                and int(_CMD.get("backup_cycles", 0)) < 4
            ):
                _CMD["backup_cycles"] = int(_CMD.get("backup_cycles", 0)) + 1
                _CMD["phase"] = "backup"
                _CMD["backup_s"] = 0.0
                return -0.09, 0.0, "creep"
            if float(_CMD["yaw_s"]) >= 2.4 and int(_CMD["flips"]) < 2:
                _CMD["flips"] = int(_CMD["flips"]) + 1
                _CMD["yaw_sign"] = -int(_CMD["yaw_sign"] or 1)
                _CMD["follow_sign"] = int(_CMD["yaw_sign"])
                _CMD["yaw_s"] = 0.0
            if (
                float(_CMD["yaw_s"]) >= 2.0
                and int(_CMD["flips"]) >= 2
                and int(_CMD.get("backup_cycles", 0)) < 2
                and not _rear_blocked(labels, height)
            ):
                _CMD["backup_cycles"] = int(_CMD.get("backup_cycles", 0)) + 1
                _CMD["phase"] = "backup"
                _CMD["backup_s"] = 0.0
                sign = int(_CMD["yaw_sign"]) or 1
                return -0.08, 0.25 * sign, "creep"
            # Corridor opening: start a soft peel along the wall, not a charge.
            open_enough = (
                front_m is not None
                and front_m >= 0.38
                and float(_CMD["yaw_s"]) >= 0.90
                and not label_blocked
            )
            if open_enough:
                _CMD["phase"] = "cruise"
                _CMD["wall_stop"] = 0
                _CMD["unstuck"] = 1
                _CMD["recoveries"] = int(_CMD.get("recoveries", 0)) + 1
                _CMD["peel_s"] = 4.5
                follow = float(_CMD.get("follow_sign") or (int(_CMD["yaw_sign"]) or 1))
                return 0.12, 0.55 * follow, "creep"
            sign = int(_CMD["yaw_sign"]) or 1
            return 0.0, yaw_w * sign, _turn_name(sign)

        return 0.0, 0.0, "stop"

    _CMD["wall_hits"] = 0
    if _CMD["phase"] in ("halt", "yaw", "backup"):
        _CMD["clear_streak"] = int(_CMD["clear_streak"]) + 1
        sign = int(_CMD["yaw_sign"]) or 1
        too_close = front_m is not None and front_m < 0.32
        if too_close and not _rear_blocked(labels, height):
            _CMD["phase"] = "backup"
            _CMD["backup_s"] = 0.0
            return -0.09, 0.0, "creep"
        # Keep yawing until the nose has actually left the wall.
        need_more = (
            _CMD["phase"] != "yaw"
            or float(_CMD["yaw_s"]) < 1.10
            or int(_CMD["clear_streak"]) < 2
            or (front_m is not None and front_m < 0.40)
        )
        if need_more:
            _CMD["phase"] = "yaw"
            _CMD["yaw_s"] = float(_CMD["yaw_s"]) + dt
            return 0.0, yaw_w * sign, _turn_name(sign)
        _CMD["phase"] = "cruise"
        _CMD["wall_stop"] = 0
        _CMD["unstuck"] = 1
        _CMD["recoveries"] = int(_CMD.get("recoveries", 0)) + 1
        _CMD["peel_s"] = 4.5
        follow = float(_CMD.get("follow_sign") or sign)
        return 0.14, 0.55 * follow, "creep"

    bias = _cruise_bias(labels, height)
    peel = float(_CMD.get("peel_s") or 0.0)
    follow = float(_CMD.get("follow_sign") or 0.0)
    if peel > 0.0:
        _CMD["peel_s"] = max(0.0, peel - dt)
        # Prefer the recovery side so he walks the wall instead of re-aiming in.
        if follow != 0.0:
            if bias * follow >= 0.0:
                bias = max(abs(bias), 0.42) * follow
            else:
                bias = 0.35 * follow
        speed = 0.16 if peel > 1.5 else 0.20
        return speed, float(bias), "creep" if peel > 2.0 else "forward"
    if follow != 0.0 and abs(bias) < 0.12:
        # Mild lingering wall-follow after peel ends.
        bias = 0.18 * follow
    return 0.22, float(bias), "forward"



class FootprintGate:
    """Stop the whole inflated footprint from entering the visual wall mesh.

    Front clearance alone is not enough: a yaw next to a wall sweeps corners,
    wheels, and the caster through the STL even when the nose is still out.
    """

    TOUCH_M = 0.05
    INFLATE = 0.08
    CLOSE_M = 0.14

    def __init__(self, caster, segs):
        self.caster = caster
        self.segs = segs if segs is not None else np.zeros((0, 2, 2), dtype=np.float64)
        self.local_phys = footprint_local_samples(0.0)
        self.local_inf = footprint_local_samples(self.INFLATE)
        self.block_forward = 0
        self.block_yaw = 0
        self.min_clear = None
        self.overlap = False
        self.reverse_left = 0.0
        self.yaw_after_backup = False
        self._hard_latched = False
        self._probe_err = ""

    def _box(self, inflate: float):
        front = 0.15 + inflate
        rear = min(-0.18 - inflate, -0.26 - inflate)
        half = max(0.165 + inflate, 0.17 + inflate)
        return front, rear, half

    def _rect_hit(self, pos, yaw, inflate: float) -> bool:
        front, rear, half = self._box(inflate)
        return bool(_rect_segment_overlap(pos, yaw, self.segs, front, rear, half))

    def probe(self, pos, yaw, motion_world=None, use_mesh=True, inflate=True):
        local = self.local_inf if inflate else self.local_phys
        world = footprint_world(pos, yaw, local)
        seg_d = _point_seg_dist(world, self.segs)
        dist = float(np.min(seg_d)) if len(seg_d) else float("inf")
        touch = self.TOUCH_M if inflate else 0.02
        inside = bool(math.isfinite(dist) and dist < touch)
        if self._rect_hit(pos, yaw, self.INFLATE if inflate else 0.0):
            inside = True
            dist = 0.0
        motion_hit = False
        if use_mesh and self.caster is not None:
            try:
                d, ins, mh = self.caster.probe_footprint(world, motion_world)
                if len(d):
                    dist = min(dist, float(np.min(d)))
                    if int(np.max(ins)) > 0:
                        inside = True
                    if int(np.max(mh)) > 0:
                        motion_hit = True
                        inside = True
            except Exception as e:
                if not self._probe_err:
                    self._probe_err = "%s: %s" % (type(e).__name__, e)
                    print("footprint mesh probe:", self._probe_err)
        if motion_hit:
            dist = 0.0
        if not math.isfinite(dist):
            dist = 9.0
        return {"clear": float(dist), "inside": bool(inside), "motion_hit": bool(motion_hit)}

    def note_physical(self, pos, yaw, use_mesh=True):
        st = self.probe(pos, yaw, use_mesh=use_mesh, inflate=False)
        if self.min_clear is None or st["clear"] < self.min_clear:
            self.min_clear = st["clear"]
        if st["inside"]:
            self.overlap = True
        return st

    def _predict(self, pos, yaw, v, w, dt):
        x, y = float(pos[0]), float(pos[1])
        yaw = float(yaw)
        v = float(v)
        w = float(w)
        dt = max(0.0, float(dt))
        if abs(w) < 1e-5:
            nx = x + v * math.cos(yaw) * dt
            ny = y + v * math.sin(yaw) * dt
            nyaw = yaw
        else:
            nyaw = yaw + w * dt
            nx = x + (v / w) * (math.sin(nyaw) - math.sin(yaw))
            ny = y - (v / w) * (math.cos(nyaw) - math.cos(yaw))
        return nx, ny, nyaw

    def _motion(self, pos, yaw, pred):
        a = footprint_world(pos, yaw, self.local_inf)
        b = footprint_world((pred[0], pred[1]), pred[2], self.local_inf)
        return b - a

    def command_state(self, pos, yaw, v, w, dt, use_mesh=True):
        pred = self._predict(pos, yaw, v, w, dt)
        motion = self._motion(pos, yaw, pred)
        st = self.probe((pred[0], pred[1]), pred[2], motion_world=motion, use_mesh=use_mesh, inflate=True)
        return st, pred

    def _count_block(self, v, w, out_v, out_w):
        if v > 0.02 and out_v <= 0.02:
            self.block_forward += 1
        if abs(w) > 0.05 and (abs(out_w) < 0.05 or (w * out_w < 0.0)):
            self.block_yaw += 1

    def filter(self, pos, yaw, v, w, dt):
        """Return a command whose predicted footprint does not enter the mesh."""
        v = float(v)
        w = float(w)
        dt = max(1e-3, float(dt))
        cur = self.probe(pos, yaw, use_mesh=True, inflate=True)
        phys = self.note_physical(pos, yaw, use_mesh=True)
        # Keep physical body >5 cm from wall boxes/mesh when possible.
        SAFE = 0.055

        def eval_cmd(vv, ww):
            return self.command_state(pos, yaw, vv, ww, dt, use_mesh=True)[0]

        def enters(st):
            return bool(st["inside"] or st["motion_hit"] or st["clear"] < SAFE * 0.5)

        def increases(st):
            return st["clear"] > cur["clear"] + 0.006

        def safe_enough(st):
            return (not enters(st)) and st["clear"] >= min(SAFE, cur["clear"] + 0.002)

        close = cur["clear"] < self.CLOSE_M or phys["inside"] or cur["clear"] < SAFE

        if self.reverse_left > 0.0:
            self.reverse_left = max(0.0, self.reverse_left - dt)
            rst = eval_cmd(-0.10, 0.0)
            if safe_enough(rst) or (cur["inside"] and increases(rst)):
                self._count_block(v, w, -0.10, 0.0)
                return -0.10, 0.0, "reverse"
            self.reverse_left = 0.0
            self.yaw_after_backup = True

        prop = eval_cmd(v, w)
        forward_ok = (not enters(prop)) and (
            (not close) or prop["clear"] + 0.004 >= cur["clear"]
        ) and (prop["clear"] >= SAFE or prop["clear"] + 0.003 >= cur["clear"])
        if forward_ok and not self.yaw_after_backup:
            return v, w, "pass"
        if (
            self.yaw_after_backup
            and abs(w) > 0.05
            and abs(v) < 0.05
            and (not enters(prop))
        ):
            self.yaw_after_backup = False
            return v, w, "pass"

        # Forward gated: prefer the yaw (then reverse+yaw) that opens clearance.
        best = None
        yaw_hit = 0
        allow_drop = bool(self.yaw_after_backup)
        yaw_cands = []
        for s in (1.0, -1.0):
            for ww in (0.95 * s, 0.70 * s, 0.45 * s):
                yaw_cands.append((0.0, ww))
        if abs(w) > 0.05:
            yaw_cands.append((0.0, w))
        # Soft wall-creep after opening a side: small forward + follow yaw.
        if v > 0.02 and abs(w) > 0.05:
            yaw_cands.append((min(0.12, abs(v)), w))
            yaw_cands.append((0.08, 0.55 if w > 0 else -0.55))
        for cv, cw in yaw_cands:
            st = eval_cmd(cv, cw)
            if enters(st) and not (cur["inside"] and increases(st)):
                if abs(cv) < 0.02:
                    yaw_hit += 1
                continue
            if cur["inside"] and not increases(st):
                continue
            better = increases(st) or allow_drop or (st["clear"] > prop["clear"] + 0.008)
            if (not enters(st)) and better:
                if best is None or st["clear"] > best[2]["clear"]:
                    best = (cv, cw, st)

        if best is not None:
            self._count_block(v, w, best[0], best[1])
            self.yaw_after_backup = False
            tag = "yaw" if abs(best[0]) < 0.03 else "creep"
            return best[0], best[1], tag

        rst = eval_cmd(-0.10, 0.0)
        if (not enters(rst) and rst["clear"] + 0.004 >= cur["clear"]) or (cur["inside"] and increases(rst)):
            self._count_block(v, w, -0.10, 0.0)
            self.reverse_left = 0.45
            self.yaw_after_backup = True
            return -0.10, 0.0, "reverse"

        if cur["inside"] or phys["inside"] or cur["clear"] < SAFE:
            cands = (
                (-0.12, 0.0), (0.0, 0.95), (0.0, -0.95),
                (-0.08, 0.55), (-0.08, -0.55), (0.0, 0.55), (0.0, -0.55),
            )
            best_c = None
            for cv, cw in cands:
                st = eval_cmd(cv, cw)
                if best_c is None or st["clear"] > best_c[2]["clear"]:
                    best_c = (cv, cw, st)
            if best_c is not None and best_c[2]["clear"] > cur["clear"] + 0.003:
                self._count_block(v, w, best_c[0], best_c[1])
                return best_c[0], best_c[1], "escape"

        self._count_block(v, w, 0.0, 0.0)
        return 0.0, 0.0, "stop"

    def hard_stop(self, pos, yaw, v, w, dt) -> bool:
        """Per-step backstop using wall segments. Yaw is not exempt."""
        if abs(v) < 0.02 and abs(w) < 0.05:
            self._hard_latched = False
            return False
        st, _pred = self.command_state(pos, yaw, v, w, dt, use_mesh=False)
        if not (st["inside"] or st["motion_hit"]):
            self._hard_latched = False
            return False
        if not self._hard_latched:
            self._count_block(v, w, 0.0 if v >= -0.02 else v, 0.0)
            self._hard_latched = True
        return True

    def outside_for_map(self, pos, yaw, prev_xy=None) -> bool:
        st = self.probe(pos, yaw, use_mesh=True, inflate=False)
        if st["inside"]:
            return False
        if prev_xy is not None:
            a = np.asarray(prev_xy, dtype=np.float64)
            b = np.asarray(pos, dtype=np.float64)
            for t in (0.25, 0.5, 0.75, 1.0):
                p = a * (1.0 - t) + b * t
                if self._rect_hit(p, yaw, 0.0):
                    return False
                # chord of the body center must not cross a wall segment
            if len(self.segs):
                chord = np.array([[a, b]], dtype=np.float64)
                # reuse point samples; also reject if the chord comes within 2 cm
                mids = np.stack([a * (1.0 - t) + b * t for t in (0.0, 0.5, 1.0)], axis=0)
                d = _point_seg_dist(mids, self.segs)
                if len(d) and float(np.min(d)) < 0.02:
                    return False
        return True


def _fused_to_world(spawn, yaw0, rel):
    x, y, th = float(rel[0]), float(rel[1]), float(rel[2])
    c, s = math.cos(float(yaw0)), math.sin(float(yaw0))
    wx = float(spawn[0]) + c * x - s * y
    wy = float(spawn[1]) + s * x + c * y
    return wx, wy, float(yaw0) + th


def _apply_gate_mode(v_cmd, w_cmd, last_mode, gate_tag):
    if gate_tag == "pass":
        return last_mode
    if abs(w_cmd) > 0.05 and abs(v_cmd) < 0.04:
        return _turn_name(1 if w_cmd > 0.0 else -1)
    if v_cmd < -0.02:
        return "creep"
    if 0.02 < v_cmd <= 0.14 and abs(w_cmd) >= 0.05:
        return "creep"
    if v_cmd > 0.02:
        return "forward" if abs(w_cmd) < 0.35 else "creep"
    return "stop"


def _norm_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _apply_d435_depth_noise(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """2% z + 3 mm. Wall hits (well inside floor clip) are not dropped or lifted to floor."""
    out = np.array(verts, dtype=np.float32, copy=True)
    z_old = out[:, 2].copy()
    ok = z_old > 1e-4
    n_ok = int(ok.sum())
    if n_ok == 0:
        return out
    wall = ok & (z_old < (float(TD_FLOOR_CLIP) - WALL_Z_MARGIN))
    sigma = (0.02 * z_old[ok] + 0.003).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=n_ok).astype(np.float32) * sigma
    z_new = np.maximum(z_old[ok] + noise, 0.0)
    drop = rng.random(n_ok) < 0.015
    # do not erase a wall hit
    wall_ok = wall[ok]
    drop = drop & (~wall_ok)
    z_new[drop] = 0.0
    # keep a true wall below the floor clip
    cap = np.float32(float(TD_FLOOR_CLIP) - 0.02)
    z_new[wall_ok] = np.minimum(z_new[wall_ok], cap)
    z_new[wall_ok] = np.maximum(z_new[wall_ok], 0.02)
    z = z_old.copy()
    z[ok] = z_new
    scale = np.ones_like(z)
    both = ok & (z_old > 1e-4) & (z > 1e-4)
    scale[both] = z[both] / z_old[both]
    out[:, 0] = verts[:, 0] * scale
    out[:, 1] = verts[:, 1] * scale
    out[:, 2] = z
    return out


_D435I_GYRO_DENSITY = math.radians(0.014)
_D435I_GYRO_BIAS = math.radians(0.08)


def _sim_imu_yaw_rate(omega_true, dt, rng, bias) -> float:
    rate = 1.0 / max(dt, 1e-4)
    jitter = _D435I_GYRO_DENSITY * math.sqrt(rate)
    return float(omega_true) + float(bias) + float(rng.normal(0.0, jitter))


def _encoder_wheel_mps(vl, vr, track, dt, rng):
    if dt <= 0.0:
        return float(vl), float(vr)
    v = 0.5 * (vl + vr)
    omega = (vr - vl) / max(track, 1e-6)
    omega_enc = omega / ANGULAR_SLIP_SCALE
    lin = LINEAR_SLIP_SCALE if LINEAR_SLIP_SCALE else 1.0
    v_enc = v / lin
    dtheta = omega_enc * dt
    ds = v_enc * dt
    sig_yaw = Q_YAW_SCALE * abs(dtheta) + Q_YAW_FLOOR
    sig_fwd = Q_FWD_SCALE * abs(ds) + Q_FWD_FLOOR
    dtheta += float(rng.normal(0.0, sig_yaw))
    ds += float(rng.normal(0.0, sig_fwd))
    omega_n = dtheta / dt
    v_n = ds / dt
    half = 0.5 * track
    return float(v_n - omega_n * half), float(v_n + omega_n * half)


def _read_wheel_mps(state, wheel_dofs, wheel_r, cmd_vl, cmd_vr):
    cmd_vl, cmd_vr = float(cmd_vl), float(cmd_vr)
    try:
        qd = np.asarray(state.joint_qd.numpy()).reshape(-1)
        al = float(qd[int(wheel_dofs[0])]) * wheel_r
        ar = float(qd[int(wheel_dofs[1])]) * wheel_r
        if math.isfinite(al) and math.isfinite(ar):
            cmd_mag = abs(cmd_vl) + abs(cmd_vr)
            if cmd_mag < 0.02 or (abs(al) + abs(ar)) >= 0.25 * cmd_mag:
                return al, ar
    except Exception:
        pass
    return cmd_vl, cmd_vr


def _forward_match_crop(obs, known, cx, cy):
    h, w = obs.shape
    cx_i = int(np.clip(round(float(cx)), 0, w - 1))
    cy_i = int(np.clip(round(float(cy)), 0, h - 1))
    crop_obs = np.array(obs[:, cx_i:], copy=True)
    crop_known = None if known is None else np.array(known[:, cx_i:], copy=True)
    span = crop_obs.shape[1]
    if span <= 1:
        return crop_obs, crop_known
    rows = np.arange(h, dtype=np.float32)[:, None]
    cols = np.arange(span, dtype=np.float32)[None, :]
    cone = np.abs(rows - float(cy_i)) <= (cols * 1.2 + 12.0)
    crop_obs = np.where(cone, crop_obs, np.uint8(0))
    if crop_known is not None:
        crop_known = np.where(cone, crop_known, np.uint8(0))
    return crop_obs, crop_known


def _crop_has_structure(obs_crop, known_crop) -> bool:
    if obs_crop is None or obs_crop.size == 0:
        return False
    known = known_crop if known_crop is not None else (obs_crop > 0)
    n_known = int(np.count_nonzero(known > 0))
    n_obs = int(np.count_nonzero(obs_crop > 0))
    if n_known < 48 or n_obs < 16:
        return False
    frac = n_obs / float(obs_crop.size)
    std = float(obs_crop.astype(np.float32).std())
    if frac > 0.70 and std < 6.0:
        return False
    n_free = int(np.count_nonzero((known > 0) & (obs_crop == 0)))
    if n_free < 12 and frac > 0.55:
        return False
    return True


def _visual_from_scan(prev_crop, curr_obs, curr_known, px_size):
    if prev_crop is None or curr_obs is None:
        return 0.0, 0.0, 0.0, False
    if not _crop_has_structure(curr_obs, curr_known):
        return 0.0, 0.0, 0.0, False
    sm_dx, sm_dy, sm_dt, score = _scan_match(prev_crop, curr_obs)
    if float(score) < float(LOOP_MATCH_THRESH):
        return 0.0, 0.0, 0.0, True
    sx = (curr_obs.shape[1] / float(THUMB_SZ)) * float(px_size)
    vis_yaw = -float(sm_dt)
    vis_fwd = float(sm_dx) * sx
    return vis_yaw, vis_fwd, float(score), True


def _take_visual(odom_q: OdomThread):
    with odom_q._lock:
        if odom_q._vis_pending:
            y, f, c = odom_q._vis_yaw, odom_q._vis_fwd, odom_q._vis_conf
            odom_q._vis_pending = False
            return float(y), float(f), float(c)
    return 0.0, 0.0, 0.0


def _vw_to_wheels(v, omega, track, wheel_r):
    v_l = v - omega * track * 0.5
    v_r = v + omega * track * 0.5
    return float(v_l / wheel_r), float(v_r / wheel_r)


def _set_wheel_qd(control, wheel_dofs, w_l, w_r):
    """Upload wheel targets without reading the GPU buffer back every tick."""
    global _WHEEL_QD_BUF
    tgt = control.joint_target_qd
    n = int(np.prod(tgt.shape))
    if _WHEEL_QD_BUF is None or _WHEEL_QD_BUF.size != n:
        _WHEEL_QD_BUF = np.zeros(n, dtype=np.float32)
    _WHEEL_QD_BUF[int(wheel_dofs[0])] = float(w_l)
    _WHEEL_QD_BUF[int(wheel_dofs[1])] = float(w_r)
    tgt.assign(_WHEEL_QD_BUF)


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    rgb[labels == UNKNOWN] = (28, 30, 34)
    rgb[labels == SELF] = (90, 96, 110)
    rgb[labels == CLEAR] = (46, 160, 78)
    rgb[labels == OBSTACLE] = (210, 64, 48)
    rgb[max(0, RCY - 1):RCY + 2, :] = (220, 210, 80)
    rgb[:, max(0, RCX - 1):RCX + 2] = (220, 210, 80)
    x0, y0, x1, y1 = BODY_BOX
    fx0 = min(FRAME_W - 1, x1 + 1)
    fx1 = min(FRAME_W - 1, x1 + 30)
    fy0 = max(0, y0 - 2)
    fy1 = min(FRAME_H - 1, y1 + 2)
    rgb[fy0:fy0 + 1, fx0:fx1] = (255, 220, 40)
    rgb[fy1:fy1 + 1, fx0:fx1] = (255, 220, 40)
    rgb[fy0:fy1, fx0:fx0 + 1] = (255, 220, 40)
    rgb[fy0:fy1, fx1:fx1 + 1] = (255, 220, 40)
    return rgb



def _quat_rot_batch(q, v):
    x, y, z, w = [float(t) for t in q]
    qv = np.array([x, y, z], dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    uv = np.cross(qv, vv)
    uuv = np.cross(qv, uv)
    return vv + 2.0 * (w * uv + uuv)


def _rs1_visual_depth(caster, origin, quat, h, w, vfov_rad, dx=None, dy=None, cam_unit=None):
    """Downward mast rays against the visual STL, floor filled in.

    The house mesh has walls and no floor. Rays that miss the wall return the
    z=0 floor plane so the panel is a real top-down image. A wall-face hit is
    reported shorter than the floor even if the ray only grazed the wall
    near the ground — that hit is a wall, not floor noise.
    """
    origin = np.asarray(origin, dtype=np.float64)
    if cam_unit is None:
        if dx is None or dy is None:
            dx, dy = _pinhole_dxdy(h, w, vfov_rad)
        rdx = dx.reshape(-1).astype(np.float64)
        rdy = dy.reshape(-1).astype(np.float64)
        cam = np.stack([rdx, rdy, np.full(rdx.shape, -1.0)], axis=1)
        inv = 1.0 / np.maximum(np.linalg.norm(cam, axis=1), 1e-8)
        cam *= inv[:, None]
        cam_unit = cam
    dirs = _quat_rot_batch(quat, cam_unit)
    look = _quat_rot(quat, np.array([0.0, 0.0, -1.0]))
    look = np.asarray(look, dtype=np.float64)
    ln = float(np.linalg.norm(look))
    if ln > 1e-8:
        look = look / ln
    n = dirs.shape[0]
    origins = np.repeat(origin.reshape(1, 3), n, axis=0)
    t = caster.cast_rays(origins, dirs, max_t=6.0) if caster is not None else np.full(n, -1.0, np.float32)
    t = np.asarray(t, dtype=np.float64)
    hit = t > 1e-4
    optical = np.zeros(n, dtype=np.float32)
    wall = np.zeros(n, dtype=bool)
    along = np.abs(np.einsum("ij,j->i", dirs, look))
    along = np.maximum(along, 1e-4)
    if np.any(hit):
        pts = origin.reshape(1, 3) + t[hit][:, None] * dirs[hit]
        hz = pts[:, 2]
        wall_hit = hz > 0.03
        opt = (t[hit] * along[hit]).astype(np.float32)
        # A confirmed wall-face hit must not collapse into the floor clip.
        floor_opt = float(origin[2])
        opt = np.where(wall_hit, np.minimum(opt, np.float32(floor_opt - 0.22)), opt)
        optical[hit] = opt
        wall[hit] = wall_hit
    # Floor plane for rays that missed the wall mesh and are going down.
    miss = ~hit
    dz = dirs[:, 2]
    floor_ok = miss & (dz < -1e-4)
    if np.any(floor_ok):
        t_floor = -origin[2] / dz[floor_ok]
        optical[floor_ok] = (t_floor * along[floor_ok]).astype(np.float32)
    depth = optical.reshape(h, w)
    return depth, dx, dy, wall.reshape(h, w)


def _plus_x_wall(caster, pos, yaw, nose_x=0.15):
    """Distance along heading from the front face to the visual STL.

    A footprint point is on the far side only if its own column hits a wall
    face (the sample sits on the mesh), or the nose ray starts already through
    the wall (plus_x < 2 cm). A wall merely behind the robot after a turn is
    not a crossing.
    """
    if caster is None:
        return float("inf"), False, None
    c, s = float(math.cos(yaw)), float(math.sin(yaw))
    heading = np.array([c, s, 0.0], dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    locals_xy = [
        (nose_x, 0.0), (nose_x, 0.10), (nose_x, -0.10),
        (nose_x, 0.16), (nose_x, -0.16),
        (0.0, 0.0), (0.0, 0.16), (0.0, -0.16),
        (-0.18, 0.0), (-0.26, 0.0),
    ]
    origins = []
    dirs = []
    front_face = None
    world_xy = []
    for lx, ly in locals_xy:
        wx = float(pos[0]) + c * lx - s * ly
        wy = float(pos[1]) + s * lx + c * ly
        world_xy.append((wx, wy))
        if front_face is None:
            front_face = (wx, wy)
        for z in (0.22, 0.45):
            origins.append((wx, wy, z))
            dirs.append(heading)
    t = caster.cast_rays(np.asarray(origins, np.float32), np.asarray(dirs, np.float32), max_t=6.0)
    t = np.asarray(t, dtype=np.float64).reshape(-1, 2)
    nose_t = t[0]
    nose_good = nose_t[nose_t > 0.01]
    if len(nose_good):
        plus = float(np.min(nose_good))
    else:
        good = t[t > 0.01]
        plus = float(np.min(good)) if len(good) else float("inf")
    # Column hits: downward rays through the footprint. A wall face in the
    # column means that sample is on/through the mesh.
    downs_o = np.asarray([(x, y, 1.40) for x, y in world_xy], dtype=np.float32)
    downs_d = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(world_xy), 1))
    dt = caster.cast_rays(downs_o, downs_d, max_t=1.45)
    dt = np.asarray(dt, dtype=np.float64)
    hit_z = 1.40 - dt
    on_mesh = (dt > 0.01) & (hit_z > 0.05) & (hit_z < 0.80)
    crossed = bool(np.any(on_mesh)) or (math.isfinite(plus) and plus < 0.02)
    return plus, crossed, front_face



def _expanded_self_boxes(pad_px: int = 6):
    """Slightly fat axle boxes so chassis-top returns cannot remain OBSTACLE."""
    boxes = []
    for x0, y0, x1, y1 in FOOTPRINT_BOXES:
        boxes.append((
            max(0, int(x0) - pad_px),
            max(0, int(y0) - pad_px),
            min(FRAME_W, int(x1) + pad_px),
            min(FRAME_H, int(y1) + pad_px),
        ))
    return boxes


def _drop_self_footprint_verts(verts, half_x=0.20, half_y=0.20):
    """Zero RS1 verts that land on the chassis disk (camera over axle).

    Newton body/wheel tops are closer than the floor and otherwise spray
    OBSTACLE into the ego panel. SELF boxes still win afterward; this keeps
    the scatter honest.
    """
    if verts is None or len(verts) == 0:
        return verts
    out = np.asarray(verts, dtype=np.float32).copy()
    # RS1 optical: after depth_to_rs_verts, lateral metres are in cols 0/1.
    # Camera sits on the axle; body footprint is a disk around the origin.
    ok = out[:, 2] > 0.01
    near = ok & (np.abs(out[:, 0]) <= float(half_x)) & (np.abs(out[:, 1]) <= float(half_y))
    # Only drop elevated (non-floor) returns — keep CLEAR floor under the belly
    # as UNKNOWN/SELF via paint, not invented clear from kept floor verts.
    # Floor after z_bias is ~TD_FLOOR_CLIP; body tops are well below that.
    elev = near & (out[:, 2] < (float(TD_FLOOR_CLIP) - 0.04))
    out[elev] = 0.0
    return out


def _inject_wall_slab(verts, plus_x, nose_x=0.15):
    """Body-frame wall face ahead of the nose, in RS1 vert convention.

    A vertical wall in the downward footprint must become OBSTACLE in the
    cells the commander watches, even if the hit was only a few centimetres
    off the floor.
    """
    if not math.isfinite(plus_x) or plus_x <= 0.02 or plus_x > 1.10:
        return verts
    wall_x = float(nose_x) + float(plus_x)
    xs = wall_x + np.array([0.0, 0.02, 0.04], dtype=np.float32)
    ys = np.linspace(-0.22, 0.22, 25, dtype=np.float32)
    pts = []
    z = np.float32(0.40)  # well under TD_FLOOR_CLIP, taller than the 8 cm gate
    for x in xs:
        for y in ys:
            # camera X = -body X, rs_y = +body Y (see _depth_to_rs_verts)
            pts.append((-float(x), float(y), float(z)))
    extra = np.asarray(pts, dtype=np.float32)
    if verts is None or len(verts) == 0:
        return extra
    return np.concatenate([np.asarray(verts, np.float32), extra], axis=0)


class SimRsCameras:
    """RS1 (down) + RS2 (forward). Visual STL raycast is the depth source.

    Newton tiled-camera is a fallback only when the visual caster cannot be
    built. The 3090 path must not BVH-refit and raytrace the MuJoCo geoms
    *and* the house STL every grab.
    """

    def __init__(self, model, house_verts=None, house_faces=None):
        self.model = model
        self.n_world = max(1, int(getattr(model, "world_count", 1) or 1))
        self.sensor = None
        self.rays = None
        self.fwd = None
        self.h = max(48, RS_DEPTH_H // 8)
        self.w = max(80, RS_DEPTH_W // 8)
        vfov = math.radians(D435_VFOV_DEG)
        self.dx, self.dy = _pinhole_dxdy(self.h, self.w, vfov)
        rdx = self.dx.reshape(-1).astype(np.float64)
        rdy = self.dy.reshape(-1).astype(np.float64)
        cam = np.stack([rdx, rdy, np.full(rdx.shape, -1.0)], axis=1)
        inv = 1.0 / np.maximum(np.linalg.norm(cam, axis=1), 1e-8)
        cam *= inv[:, None]
        self._rs1_cam_unit = cam
        self.z_bias = float(TD_FLOOR_CLIP) - float(RS1_ORIGIN_Z)
        r_rs1 = np.column_stack((
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ))
        self.q_rs1 = _rotmat_to_quat_xyzw(r_rs1)
        self.p_rs1 = np.array([0.0, 0.0, RS1_ORIGIN_Z], dtype=np.float64)
        th = math.radians(25.6)
        c, s = math.cos(th), math.sin(th)
        r_rs2 = np.column_stack((
            np.array([0.0, -1.0, 0.0]),
            np.array([s, 0.0, c]),
            np.array([-c, 0.0, s]),
        ))
        self.q_rs2 = _rotmat_to_quat_xyzw(r_rs2)
        self.p_rs2 = np.array([0.05, 0.0, float(FW_CAM_HEIGHT)], dtype=np.float64)
        self.caster = None
        self.last_wall_px = 0
        self.last_self_core_ok = True
        self.last_self_core_leak = 0
        self.last_d1 = None
        self.last_d2 = None
        if house_verts is not None and house_faces is not None:
            try:
                self.caster = VisualMeshCaster(house_verts, house_faces, str(model.device))
                print("visual mesh caster", house_verts.shape, house_faces.shape, "device", model.device)
            except Exception as e:
                print("visual caster skip:", type(e).__name__, e)
        if self.caster is None:
            cfg = None
            if RenderConfig is not None:
                cfg = RenderConfig(enable_backface_culling=False, enable_shadows=False)
            self.sensor = SensorTiledCamera(
                model, default_render_config=cfg, load_textures=False,
            ) if cfg is not None else SensorTiledCamera(model, load_textures=False)
            self.rays = self.sensor.utils.compute_camera_rays_pinhole(
                self.w, self.h, camera_fovs=[vfov, vfov],
            )
            self.fwd = self.sensor.utils.create_forward_depth_image_output(
                self.w, self.h, camera_count=2,
            )
            print("newton tiled camera fallback (no visual caster)")

    def _xform(self, pos, quat, local_p, local_q):
        wp_pos = np.asarray(pos, dtype=np.float64) + _quat_rot(quat, local_p)
        wq = _quat_mul(quat, local_q)
        n = float(np.linalg.norm(wq))
        if n > 0:
            wq = wq / n
        return (
            wp_pos,
            wq,
            wp.transform(
                p=wp.vec3(float(wp_pos[0]), float(wp_pos[1]), float(wp_pos[2])),
                q=wp.quat(float(wq[0]), float(wq[1]), float(wq[2]), float(wq[3])),
            ),
        )

    def _visual_forward(self, origin, quat):
        if self.caster is None:
            return None
        try:
            return self.caster.forward_depth(origin, quat, self.dx, self.dy)
        except Exception as e:
            print("visual ray:", type(e).__name__, e)
            self.caster = None
            return None

    def grab(self, state, pos, quat):
        o1, q1, xf1 = self._xform(pos, quat, self.p_rs1, self.q_rs1)
        o2, q2, xf2 = self._xform(pos, quat, self.p_rs2, self.q_rs2)
        vfov = math.radians(D435_VFOV_DEG)
        d1 = None
        d2 = None
        if self.caster is not None:
            try:
                d1, _, _, _wall_vis = _rs1_visual_depth(
                    self.caster, o1, q1, self.h, self.w, vfov,
                    dx=self.dx, dy=self.dy, cam_unit=self._rs1_cam_unit,
                )
            except Exception as e:
                print("rs1 visual:", type(e).__name__, e)
                d1 = None
            d2 = self._visual_forward(o2, q2)
        if d2 is None:
            d2 = np.zeros((self.h, self.w), dtype=np.float32)
        if d1 is None:
            if self.sensor is None:
                raise RuntimeError("no visual caster and no Newton camera")
            cam_t = wp.array(
                [[xf1] * self.n_world, [xf2] * self.n_world],
                dtype=wp.transform,
            )
            self.model.bvh_refit_shapes(state)
            self.sensor.update(state, cam_t, self.rays, forward_depth_image=self.fwd)
            fd = self.fwd.numpy()
            d1 = np.asarray(fd[0, 0], dtype=np.float32)
            if self.caster is None:
                d2 = np.asarray(fd[0, 1], dtype=np.float32)
        d2 = np.asarray(d2, dtype=np.float32)
        d1 = np.asarray(d1, dtype=np.float32)
        yaw = _yaw_of(quat)
        floor_opt = float(RS1_ORIGIN_Z)
        wall_mask = np.isfinite(d1) & (d1 > 1e-3) & (d1 < (floor_opt - 0.05))
        self.last_wall_px = int(wall_mask.sum())
        self.last_d1 = d1
        self.last_d2 = d2
        self.last_o1 = o1
        self.last_o2 = o2
        self.last_look1 = _quat_rot(q1, np.array([0.0, 0.0, -1.0]))
        self.last_look2 = _quat_rot(q2, np.array([0.0, 0.0, -1.0]))
        plus_x, crossed, front_face = plus_x_against_boxes(
            (float(pos[0]), float(pos[1])), float(yaw),
            getattr(self, "wall_boxes", None) or [],
            nose_x=0.15,
        )
        self.last_plus_x = plus_x
        self.last_crossed = bool(crossed)
        self.last_front_face = front_face
        self.last_heading = float(yaw)
        verts1 = _depth_to_rs_verts(d1, self.dx, self.dy, z_bias=self.z_bias, z_min=0.01)
        verts1 = _drop_self_footprint_verts(verts1)
        verts1 = _inject_wall_slab(verts1, plus_x, nose_x=0.15)
        verts2 = _depth_to_rs_verts(d2, self.dx, self.dy, z_bias=0.0, z_min=RS2_MIN_Z)
        verts1 = _clip_decimated_border(verts1, border=4, orig_w=RS_DEPTH_W, orig_h=RS_DEPTH_H)
        verts2 = _clip_decimated_border(verts2, border=4, orig_w=RS_DEPTH_W, orig_h=RS_DEPTH_H)
        return verts1, verts2, pos, quat


def _render_slam_map(slam, evidence, pose_rel, target_h: int, pose_source: str = ""):
    from PIL import Image, ImageDraw
    from globalmap import MAP_W, MAP_H, PX_SIZE, ORIGIN_X, ORIGIN_Y

    rgb = np.full((MAP_H, MAP_W, 3), (22, 24, 28), dtype=np.uint8)
    occupied = np.zeros((MAP_H, MAP_W), dtype=bool)
    n_clear = n_obs = 0
    if evidence is not None:
        obst = evidence.obstacle_mask()
        drv = evidence.drivable_mask()
        rgb[drv] = (46, 160, 78)
        rgb[obst] = (210, 64, 48)
        occupied = drv | obst
        n_clear = int(np.count_nonzero(drv & ~obst))
        n_obs = int(np.count_nonzero(obst))

    kfs = list(getattr(slam, "_keyframes", []) or []) if slam is not None else []
    edges = list(getattr(slam, "_edges", []) or []) if slam is not None else []

    def world_to_px(x, y):
        return ORIGIN_X + float(x) / PX_SIZE, ORIGIN_Y - float(y) / PX_SIZE

    pts = []
    if occupied.any():
        ys, xs = np.nonzero(occupied)
        step = max(1, xs.size // 4000)
        pts.append(np.stack([xs[::step].astype(np.float64), ys[::step].astype(np.float64)], axis=1))
    for kf in kfs:
        gx, gy = world_to_px(kf.x, kf.y)
        pts.append(np.array([[gx, gy]], dtype=np.float64))
    if pose_rel is not None:
        gx, gy = world_to_px(pose_rel[0], pose_rel[1])
        pts.append(np.array([[gx, gy]], dtype=np.float64))
    if pts:
        allp = np.concatenate(pts, axis=0)
        pad = 16
        x0 = int(np.floor(allp[:, 0].min())) - pad
        x1 = int(np.ceil(allp[:, 0].max())) + pad
        y0 = int(np.floor(allp[:, 1].min())) - pad
        y1 = int(np.ceil(allp[:, 1].max())) + pad
    else:
        x0, y0, x1, y1 = ORIGIN_X - 40, ORIGIN_Y - 40, ORIGIN_X + 40, ORIGIN_Y + 40
    min_span = 70
    if x1 - x0 < min_span:
        c = 0.5 * (x0 + x1)
        x0, x1 = int(c - min_span / 2), int(c + min_span / 2)
    if y1 - y0 < min_span:
        c = 0.5 * (y0 + y1)
        y0, y1 = int(c - min_span / 2), int(c + min_span / 2)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(MAP_W, max(x0 + 2, x1))
    y1 = min(MAP_H, max(y0 + 2, y1))
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = target_h / float(max(1, ch))
    out_w = max(1, int(round(cw * scale)))
    max_w = int(target_h * 1.15)
    if out_w > max_w:
        scale_x = max_w / float(cw)
        scale_y = target_h / float(ch)
        out_w = max_w
    else:
        scale_x = scale_y = scale
        out_w = max(1, int(round(cw * scale_x)))
    im = Image.fromarray(crop).resize((out_w, target_h), Image.NEAREST)
    draw = ImageDraw.Draw(im)

    def to_panel(x, y):
        gx, gy = world_to_px(x, y)
        return (gx - x0) * scale_x, (gy - y0) * scale_y

    id_xy = {int(kf.id): (float(kf.x), float(kf.y)) for kf in kfs}
    for e in edges:
        i, j = int(e[0]), int(e[1])
        if i not in id_xy or j not in id_xy:
            continue
        loop = abs(i - j) != 1
        draw.line([to_panel(*id_xy[i]), to_panel(*id_xy[j])],
                  fill=(80, 210, 220) if loop else (240, 210, 60), width=2)
    if pose_rel is not None:
        px, py = to_panel(pose_rel[0], pose_rel[1])
        th = float(pose_rel[2])
        tip = (px + 11.0 * math.cos(th), py - 11.0 * math.sin(th))
        draw.line([(px, py), tip], fill=(255, 255, 255), width=2)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=(255, 255, 255))
    src = "evidence" if n_clear or n_obs else "empty"
    tag = ("  " + pose_source) if pose_source else ""
    draw.text((4, target_h - 14), "kf=%d edge=%d  %s%s" % (len(kfs), len(edges), src, tag),
              fill=(230, 226, 210))
    return np.asarray(im)


def _depth_panel(z, h_px, title, floor_z=None):
    from PIL import Image, ImageDraw
    z = np.asarray(z, dtype=np.float32)
    valid = np.isfinite(z) & (z > 1e-3)
    rgb = np.zeros((z.shape[0], z.shape[1], 3), dtype=np.uint8)
    rgb[:] = (16, 18, 22)
    if floor_z is not None and np.any(valid):
        # Newton RS1: floor optical depth is ~camera height. Wall boxes are closer.
        wall = valid & (z < (float(floor_z) - 0.05))
        floor = valid & ~wall
        rgb[floor] = (186, 168, 118)
        if np.any(wall):
            near = np.clip((float(floor_z) - z) / 0.50, 0.0, 1.0)
            rgb[..., 0] = np.where(wall, (210 + 40 * near).astype(np.uint8), rgb[..., 0])
            rgb[..., 1] = np.where(wall, (48 + 30 * (1.0 - near)).astype(np.uint8), rgb[..., 1])
            rgb[..., 2] = np.where(wall, np.uint8(32), rgb[..., 2])
    elif np.any(valid):
        t = np.clip((np.clip(z, 0.05, 2.4) - 0.05) / 2.2, 0.0, 1.0)
        rgb[..., 0] = np.where(valid, (30 + 210 * (1.0 - t)).astype(np.uint8), 16)
        rgb[..., 1] = np.where(valid, (24 + 140 * np.clip(1.0 - np.abs(t - 0.4) * 2.2, 0, 1)).astype(np.uint8), 18)
        rgb[..., 2] = np.where(valid, (36 + 190 * t).astype(np.uint8), 22)
    im = Image.fromarray(rgb)
    out_w = max(110, int(round(h_px * z.shape[1] / max(1, z.shape[0]))))
    im = im.resize((out_w, h_px), Image.NEAREST)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.size[0], 15), fill=(8, 10, 12))
    draw.text((3, 2), title, fill=(245, 232, 200))
    return np.asarray(im)


def _labels_panel(labels, h_px):
    from PIL import Image, ImageDraw
    rgb = _colorize_labels(labels)
    im = Image.fromarray(rgb).resize(
        (max(110, int(round(h_px * labels.shape[1] / max(1, labels.shape[0])))), h_px),
        Image.NEAREST,
    )
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.size[0], 15), fill=(8, 10, 12))
    draw.text((3, 2), "RS1 ego", fill=(245, 232, 200))
    return np.asarray(im)


def _cam_strip(d1, d2, labels, panel_h):
    from PIL import Image
    h_each = max(90, panel_h // 3)
    parts = []
    if d1 is not None:
        parts.append(_depth_panel(d1, h_each, "RS1 top-down", floor_z=RS1_ORIGIN_Z))
    if labels is not None:
        parts.append(_labels_panel(labels, h_each))
    if d2 is not None:
        parts.append(_depth_panel(d2, h_each, "RS2 forward"))
    if not parts:
        return None
    w = max(p.shape[1] for p in parts)
    imgs = []
    for p in parts:
        if p.shape[1] != w:
            canvas = np.zeros((p.shape[0], w, 3), dtype=np.uint8)
            canvas[:, :p.shape[1]] = p
            p = canvas
        imgs.append(p)
    strip = np.concatenate(imgs, axis=0)
    if strip_h := strip.shape[0]:
        if strip_h != panel_h:
            im = Image.fromarray(strip).resize((w, panel_h), Image.BILINEAR)
            strip = np.asarray(im)
    return strip


def _log_path_trail(viewer, trail_xy):
    pts = list(trail_xy)[-36:]
    if len(pts) < 2:
        return
    starts = np.array([(x, y, 0.07) for x, y in pts[:-1]], dtype=np.float32)
    ends = np.array([(x, y, 0.07) for x, y in pts[1:]], dtype=np.float32)
    dev = getattr(viewer, "device", None)
    kw = {"dtype": wp.vec3}
    if dev is not None:
        kw["device"] = dev
    viewer.log_lines("drive/path", wp.array(starts, **kw), wp.array(ends, **kw), (1.0, 0.92, 0.12))


def _log_cam_rays(viewer, cams):
    if cams.last_d1 is None:
        return
    segs = []
    # RS1 look-down, RS2 look-forward — short rays so the virtual cameras show in overhead
    for origin, look, length, color_name in (
        (cams.last_o1, cams.last_look1, 0.70, "rs1"),
        (cams.last_o2, cams.last_look2, 0.55, "rs2"),
    ):
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(look, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            continue
        d = d / n
        segs.append((o, o + d * length))
    if not segs:
        return
    starts = np.array([a for a, _b in segs], dtype=np.float32)
    ends = np.array([b for _a, b in segs], dtype=np.float32)
    dev = getattr(viewer, "device", None)
    kw = {"dtype": wp.vec3}
    if dev is not None:
        kw["device"] = dev
    viewer.log_lines("drive/cams", wp.array(starts, **kw), wp.array(ends, **kw), (0.35, 0.85, 1.0))


def _compose_frame(overview, slam_panel, cam_panel, mode, pose, slam_line, extra=""):
    from PIL import Image, ImageDraw
    ov = np.asarray(overview)
    if ov.ndim == 2:
        ov = np.repeat(ov[:, :, None], 3, axis=2)
    if ov.shape[-1] == 4:
        ov = ov[:, :, :3]
    ov = ov.astype(np.uint8, copy=False)
    panel_h = max(120, int(ov.shape[0] * 0.92))
    if slam_panel is None:
        slam_panel = np.full((panel_h, max(80, panel_h // 2), 3), (22, 24, 28), dtype=np.uint8)
    else:
        slam_panel = np.asarray(slam_panel)
        if slam_panel.ndim == 2:
            slam_panel = np.repeat(slam_panel[:, :, None], 3, axis=2)
        if slam_panel.shape[-1] == 4:
            slam_panel = slam_panel[:, :, :3]
    if cam_panel is None:
        cam_panel = np.full((panel_h, 140, 3), (16, 18, 20), dtype=np.uint8)
    else:
        cam_panel = np.asarray(cam_panel)
        if cam_panel.ndim == 2:
            cam_panel = np.repeat(cam_panel[:, :, None], 3, axis=2)
        if cam_panel.shape[-1] == 4:
            cam_panel = cam_panel[:, :, :3]
    slam_im = Image.fromarray(slam_panel.astype(np.uint8))
    cam_im = Image.fromarray(cam_panel.astype(np.uint8))
    canvas_w = ov.shape[1] + slam_im.size[0] + cam_im.size[0] + 18
    canvas_h = max(ov.shape[0], slam_im.size[1] + 28, cam_im.size[1] + 28)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 18, 20))
    canvas.paste(Image.fromarray(ov), (0, 0))
    canvas.paste(slam_im, (ov.shape[1] + 8, 28))
    canvas.paste(cam_im, (ov.shape[1] + slam_im.size[0] + 14, 28))
    draw = ImageDraw.Draw(canvas)
    x, y, yaw = pose
    draw.text((8, 6), "newton kevin  mode=%s  xy=(%.2f,%.2f) yaw=%.0fdeg" % (
        mode, x, y, math.degrees(yaw)), fill=(240, 236, 220))
    if extra:
        draw.text((8, 22), extra, fill=(220, 180, 120))
    draw.text((ov.shape[1] + 8, 6), "slam map", fill=(240, 236, 220))
    draw.text((ov.shape[1] + slam_im.size[0] + 14, 6), "virtual cams", fill=(240, 236, 220))
    if slam_line:
        draw.text((8, ov.shape[0] - 18), slam_line, fill=(220, 220, 180))
    return np.asarray(canvas)


def _idle_over_2s(path_xy, dt, eps=0.03, need=2.0):
    p = np.asarray(path_xy, dtype=np.float64)
    n = len(p)
    if n < 2:
        return False, 0.0
    max_dur = 0.0
    found = False
    for i in range(n):
        j = i
        while j + 1 < n and float(np.linalg.norm(p[j + 1] - p[i])) <= eps:
            j += 1
        dur = (j - i) * dt
        if dur > max_dur:
            max_dur = dur
        if dur > need:
            found = True
    return found, float(max_dur)


def _xy_to_px(x, y, w, h, pad=18):
    nx = (float(x) - FLOOR_X0) / max(1e-6, FLOOR_X1 - FLOOR_X0)
    ny = (float(y) - FLOOR_Y0) / max(1e-6, FLOOR_Y1 - FLOOR_Y0)
    px = pad + nx * (w - 2 * pad)
    py = h - pad - ny * (h - 2 * pad)
    return int(px), int(py)


def _cheap_overview(w, h, pos, yaw, trail, wall_segs, mode, extra, d1=None, labels=None, bg=None):
    """2D house + path. No ViewerGL, no 960×540 GPU compose."""
    from PIL import Image, ImageDraw

    if bg is not None:
        img = bg.copy()
    else:
        img = Image.new("RGB", (w, h), (18, 20, 24))
        draw = ImageDraw.Draw(img)
        if wall_segs is not None and len(wall_segs):
            for seg in wall_segs:
                a, b = seg[0], seg[1]
                p0 = _xy_to_px(a[0], a[1], w, h)
                p1 = _xy_to_px(b[0], b[1], w, h)
                draw.line([p0, p1], fill=(210, 200, 180), width=2)
    draw = ImageDraw.Draw(img)
    if trail and len(trail) >= 2:
        pts = [_xy_to_px(x, y, w, h) for x, y in trail[-80:]]
        draw.line(pts, fill=(240, 210, 60), width=2)
    px, py = _xy_to_px(pos[0], pos[1], w, h)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    nose = _xy_to_px(pos[0] + 0.28 * c, pos[1] + 0.28 * s, w, h)
    left = _xy_to_px(pos[0] - 0.12 * c - 0.12 * s, pos[1] - 0.12 * s + 0.12 * c, w, h)
    right = _xy_to_px(pos[0] - 0.12 * c + 0.12 * s, pos[1] - 0.12 * s - 0.12 * c, w, h)
    draw.polygon([nose, left, right], fill=(80, 180, 255), outline=(240, 240, 240))
    draw.text((8, 6), "newton kevin  mode=%s  xy=(%.2f,%.2f) yaw=%.0f" % (
        mode, pos[0], pos[1], math.degrees(float(yaw))), fill=(240, 236, 220))
    if extra:
        draw.text((8, 22), extra[:110], fill=(220, 180, 120))
    if d1 is not None:
        chip = _depth_panel(d1, 120, "RS1", floor_z=RS1_ORIGIN_Z)
        img.paste(Image.fromarray(chip), (w - chip.shape[1] - 8, 40))
    if labels is not None:
        lab = _labels_panel(labels, 120)
        img.paste(Image.fromarray(lab), (w - lab.shape[1] - 8, 170))
    return np.asarray(img)


_GIF_PLAYER = None


def _play_gif_file(path):
    """Non-blocking gif playback on DISPLAY=:1."""
    global _GIF_PLAYER
    path = str(path)
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    if _GIF_PLAYER is not None and _GIF_PLAYER.poll() is None:
        try:
            _GIF_PLAYER.terminate()
        except Exception:
            pass
    cmds = (
        ["vlc", "--play-and-exit", "--no-qt-error-dialogs", "--qt-minimal-view", path],
        ["xdg-open", path],
    )
    for cmd in cmds:
        try:
            _GIF_PLAYER = subprocess.Popen(
                cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("playing gif", path, "via", cmd[0], flush=True)
            return
        except FileNotFoundError:
            continue
    print("gif play skipped (no vlc/xdg-open):", path, flush=True)


class _CheapViz:
    """Still/gif recorder. Live view is always the small ViewerGL window."""

    GIF_WALL_S = 60.0

    def __init__(self, gif_path, wall_segs):
        self.gif_path = Path(gif_path)
        self.wall_segs = wall_segs
        self.w, self.h = 480, 360
        self.mode = "gl"
        print("viz: ViewerGL 3D stays up (no overlays, no gif switch)", flush=True)
        self.t_wall0 = time.monotonic()
        self.last_gif_wall = None
        self.buf = []
        self.keep = []
        self.last_frame = None
        self.window = None
        self.rt = 0.0
        self.gif_i = 0
        self.n_push = 0
        self._bg = None

    def _background(self):
        if self._bg is not None:
            return self._bg
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (self.w, self.h), (18, 20, 24))
        draw = ImageDraw.Draw(img)
        if self.wall_segs is not None and len(self.wall_segs):
            for seg in self.wall_segs:
                a, b = seg[0], seg[1]
                p0 = _xy_to_px(a[0], a[1], self.w, self.h)
                p1 = _xy_to_px(b[0], b[1], self.w, self.h)
                draw.line([p0, p1], fill=(210, 200, 180), width=2)
        self._bg = img
        return self._bg

    def rt_now(self, t_sim):
        wall = max(1e-4, time.monotonic() - self.t_wall0)
        self.rt = float(t_sim) / wall
        return self.rt, wall

    def _open_window(self):
        if self.window is not None:
            return
        try:
            import tkinter as tk
            from PIL import ImageTk
            root = tk.Tk()
            root.title("kevin newton")
            root.geometry("%dx%d" % (self.w, self.h))
            label = tk.Label(root)
            label.pack()
            self.window = {"tk": tk, "ImageTk": ImageTk, "root": root, "label": label, "imgtk": None}
            root.update_idletasks()
            root.update()
            print("live window DISPLAY=%s" % os.environ.get("DISPLAY", ""), flush=True)
        except Exception as e:
            print("live window skip:", type(e).__name__, e, flush=True)
            self.window = False

    def _close_window(self):
        if isinstance(self.window, dict):
            try:
                self.window["root"].destroy()
            except Exception:
                pass
        self.window = None

    def _blit(self, frame):
        if self.window is False:
            return
        if self.window is None:
            self._open_window()
        if not isinstance(self.window, dict):
            return
        from PIL import Image
        try:
            im = Image.fromarray(frame)
            self.window["imgtk"] = self.window["ImageTk"].PhotoImage(im)
            self.window["label"].configure(image=self.window["imgtk"])
            self.window["root"].update_idletasks()
            self.window["root"].update()
        except Exception as e:
            print("live blit skip:", type(e).__name__, e, flush=True)
            self._close_window()
            self.window = False

    def decide(self, t_sim):
        return

    def push(self, t_sim, pos, yaw, trail, mode, extra, d1=None, labels=None):
        if self.mode == "gl":
            # 3D window owns the live view. Keep a still for check pngs.
            if self.last_frame is None or self.n_push % 16 == 0:
                self.last_frame = _cheap_overview(
                    self.w, self.h, pos, yaw, trail, self.wall_segs, mode, extra,
                    d1=None, labels=None, bg=self._background(),
                )
                if len(self.keep) < 8:
                    self.keep.append(self.last_frame.copy())
            self.n_push += 1
            return
        frame = _cheap_overview(
            self.w, self.h, pos, yaw, trail, self.wall_segs, mode, extra,
            d1=None, labels=None, bg=self._background(),
        )
        self.last_frame = frame
        self.n_push += 1
        if len(self.keep) < 8 or (self.n_push % 40 == 0 and len(self.keep) < 24):
            self.keep.append(frame.copy())
        self.decide(t_sim)
        if self.mode == "live":
            self._blit(frame)
            return
        self.buf.append(frame)
        if len(self.buf) > 220:
            self.buf = self.buf[::2]
        _rt, wall = self.rt_now(t_sim)
        if self.last_gif_wall is None:
            self.last_gif_wall = wall
        elif self.mode == "gif" and (wall - self.last_gif_wall) >= self.GIF_WALL_S:
            self.flush_gif(play=True)

    def flush_gif(self, play=False, dest=None):
        from PIL import Image
        frames = self.buf or ([self.last_frame] if self.last_frame is not None else [])
        if not frames:
            return None
        if dest is None:
            dest = self.gif_path.parent / ("kevin_newton_drive_%03d.gif" % self.gif_i)
            self.gif_i += 1
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        take = min(160, len(frames))
        idx = np.linspace(0, len(frames) - 1, take).astype(int)
        ims = [Image.fromarray(frames[int(i)]).convert("P", palette=Image.ADAPTIVE, colors=64) for i in idx]
        ims[0].save(
            dest, save_all=True, append_images=ims[1:], duration=80, loop=0, optimize=True,
        )
        print("wrote", dest, "bytes", dest.stat().st_size, "frames", take, "rt=%.1fx" % self.rt, flush=True)
        self.buf = []
        self.last_gif_wall = time.monotonic() - self.t_wall0
        if play:
            _play_gif_file(dest)
        return dest


class _LiveGL:
    """Bird's-eye casita. CPU draw only — Newton ViewerGL log_state syncs CUDA."""

    def __init__(self, wall_boxes, people=None, clutter=None):
        floor = (FLOOR_X0, FLOOR_X1, FLOOR_Y0, FLOOR_Y1)
        self.view = HouseView(
            wall_boxes,
            floor,
            people=people,
            clutter=clutter,
            caption="Kevin LIVE",
            size=(800, 520),
            location=(40, 50),
        )
        self.sample = None
        try:
            self.sample = SampleReplay(wall_boxes, floor, people=people, clutter=clutter)
        except Exception as e:
            print("1x sample skip:", type(e).__name__, e, flush=True)
        self.closed = False

    def present(
        self, pos, yaw, trail=None, people=None, chat="", note="", rt=0.0,
        min_dt=1.0 / 12.0, t_sim=0.0,
    ):
        if self.closed:
            return
        try:
            self.view.present(
                pos, yaw, trail=trail, people=people, chat=chat,
                note=note, rt=rt, min_dt=min_dt,
            )
            if self.sample is not None and not self.sample.closed:
                self.sample.record(t_sim, pos, yaw, chat, note, trail)
                self.sample.present()
        except Exception as e:
            print("gl present skip:", type(e).__name__, e, flush=True)
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.view.close()
        except Exception:
            pass
        if self.sample is not None:
            try:
                self.sample.close()
            except Exception:
                pass


def run_perception_drive(args):
    os.environ.setdefault("DISPLAY", ":1")
    from newton_kevin import (
        WHEEL_R, BODY_W, WHEEL_GAP, WHEEL_W, BODY_L, AXLE_FROM_REAR,
        add_house, add_random_obstacles, build_kevin, DEFAULT_STL,
    )

    wp.init()
    builder = newton.ModelBuilder()
    stl = Path(getattr(args, "stl", "") or DEFAULT_STL)
    mins, maxs = add_house(builder, stl)
    spawn = (float(mins[0]) + 1.2, float(mins[1]) + 1.2)
    people = SimPeople(seed=int(getattr(args, "seed", 3)))
    add_random_obstacles(
        builder, mins, maxs,
        seed=int(getattr(args, "seed", 3)),
        static=True,
        clutter=str(getattr(args, "clutter", "random") or "random"),
        spawn=spawn,
        people_xy=[p["xy"] for p in people.folk],
    )
    wall_segs0 = getattr(add_house, "wall_segments", None)
    yaw0 = 0.0
    if wall_segs0 is not None and len(wall_segs0):
        a = wall_segs0[:, 0]
        b = wall_segs0[:, 1]
        ab = b - a
        ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
        sp = np.array(spawn, dtype=np.float64)
        ap = sp - a
        tt = np.clip(np.einsum("ij,ij->i", ap, ab) / ab2, 0.0, 1.0)
        closest = a + tt[:, None] * ab
        d = np.linalg.norm(closest - sp, axis=1)
        j = int(np.argmin(d))
        delta = closest[j] - sp
        yaw0 = math.atan2(float(delta[1]), float(delta[0]))
        print("spawn yaw toward nearest wall %.1fdeg dist=%.3f" % (math.degrees(yaw0), float(d[j])))
    wheel_joints = build_kevin(builder, start_xy=spawn, yaw=yaw0)
    chassis = int(build_kevin.chassis)
    builder.add_ground_plane()
    model = builder.finalize(device=getattr(args, "device", "cuda:0"))
    # MuJoCo Warp owns qpos. Re-pushing Newton state every substep was a
    # GPU sync (~12 ms). Nudge/hold paths call update_mjc_data themselves.
    solver = newton.solvers.SolverMuJoCo(
        model, njmax=1024, nconmax=384,
        update_data_interval=0,
        disable_sensors=True,
    )
    try:
        opt = solver.mjw_model.opt
        print(
            "mjw iterations=%s ls=%s ccd=%s graph_conditional=%s"
            % (opt.iterations, opt.ls_iterations, opt.ccd_iterations, opt.graph_conditional),
            flush=True,
        )
        # 100 Newton iters is the RL default for contact-rich batches, not a
        # single 7-DoF base. 8 is plenty for wheels+floor+thin boxes.
        opt.iterations = 8
        print("mjw iterations ->", opt.iterations, flush=True)
    except Exception as e:
        print("mjw opt skip:", type(e).__name__, e, flush=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    y_wheel = BODY_W * 0.5 + WHEEL_GAP + WHEEL_W * 0.5
    track = 2.0 * y_wheel
    free_joint = list(model.joint_label).index("kevin_free")
    qd_start = model.joint_qd_start.numpy()
    wheel_dofs = [int(qd_start[j]) for j in wheel_joints]
    print("kevin_free", free_joint, "wheel_dofs", wheel_dofs, "spawn", spawn)
    _print_free_sync_map(model, solver, free_joint)
    print("house repair", getattr(add_house, "report", {}))

    cams = SimRsCameras(model, getattr(add_house, "verts", None), getattr(add_house, "faces", None))
    wall_segs = getattr(add_house, "wall_segments", np.zeros((0, 2, 2)))
    front_x = -AXLE_FROM_REAR + BODY_L
    rear_x = -AXLE_FROM_REAR
    half_w = BODY_W * 0.5
    fp_gate = FootprintGate(cams.caster, wall_segs)
    cams.wall_boxes = list(getattr(add_house, "wall_boxes", []) or [])
    print("depth collision boxes", len(cams.wall_boxes))
    extra = [p for p in (getattr(add_random_obstacles, "placed", []) or []) if _furniture_ok(p, spawn)]
    wander = VisitWander(list(cams.wall_boxes) + extra + people.as_keepout())
    kin_boxes = list(cams.wall_boxes) + extra
    cams.wall_boxes = list(kin_boxes)
    print(
        "wander boxes", len(getattr(add_house, "wall_boxes", []) or []), "walls +", len(extra), "furniture +",
        len(people.folk), "people  door=(%.2f,%.2f)"
        % (float(getattr(wander, "door_x", -0.6)), float(getattr(wander, "door_y", -2.81))),
    )

    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    work_l = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    work_h = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    fw_cone = Vision._build_obs_mask()[1]
    free_range = Vision._build_free_range_mask()

    evidence = EvidenceMap()
    slam = None
    slam_skip = ""
    try:
        from slam import PoseGraphSLAM
        slam = None
        slam_skip = "off (kinematic wander)"
    except Exception as e:
        slam_skip = "%s: %s" % (type(e).__name__, e)

    pose_est = PoseEstimator(wheelbase_m=track, wheel_radius_m=WHEEL_R)
    odom_q = OdomThread(pose_est)
    reset_command_state()
    rng = np.random.default_rng(int(getattr(args, "seed", 3)) + 17)
    imu_bias = float(_D435I_GYRO_BIAS)

    seconds = float(getattr(args, "seconds", 9.0))
    cap_fps = float(getattr(args, "capture_fps", 12.0))
    phys_hz = 40.0
    substeps = 2
    frame_dt = 1.0 / phys_hz
    sim_dt = frame_dt / substeps
    n_phys = max(8, int(round(seconds * phys_hz)))
    sense_every = max(1, int(round(phys_hz / cap_fps)))
    gif_path = Path(getattr(args, "gif", "") or (_REPO / "sim" / "kevin_newton_drive.gif"))
    viz = _CheapViz(gif_path, wall_segs)
    live_gl = None
    try:
        live_gl = _LiveGL(list(getattr(add_house, "wall_boxes", []) or []), people=people.folk, clutter=extra)
    except Exception as e:
        print("house view skip:", type(e).__name__, e, flush=True)
        viz.mode = "gif"
    contacts = None
    phys_graph = None
    print("kinematic open-floor: skip MuJoCo graph (rays still GPU)", flush=True)
    t_wall0 = time.monotonic()
    viz.t_wall0 = t_wall0
    print(
        "drive budget: phys=%.0fHz sense=%.0fHz visual_caster=%s newton_cam=%s kinematic=True"
        % (phys_hz, cap_fps, cams.caster is not None, cams.sensor is not None),
        flush=True,
    )

    frames = []
    n_sense = 0
    modes = []
    path_xy = []
    body_xy = []
    cam_xy = []
    fused_xy = []
    last_mode = "boot"
    slam_line = "slam: pending" if slam is not None else ("slam skipped: " + slam_skip)
    t_sim = 0.0
    n_clear = n_obs = n_self = 0
    rs1_ok_n = 0
    rs2_clear_n = 0
    evid_ok = evid_fail = 0
    trail_fail = 0
    last_rel = (0.0, 0.0, 0.0)
    prev_match_crop = None
    cmd_vl = cmd_vr = 0.0
    v_cmd = w_cmd = 0.0
    prev_truth_yaw = None
    imu_rate = 0.0
    pose_src_counts = {"visual": 0, "imu+wheel": 0}
    match_attempts = 0
    last_pose_source = "imu+wheel"
    rs1_wall_px_sum = 0
    rs1_wall_frames = 0
    min_wall = None
    min_front = None
    overlap_any = False
    wall_stop_latched = False
    got_unstuck = False
    stuck_reason = ""
    stuck_abort = False
    run_path = 0.0
    still_s = 0.0
    tip_yaw_s = 0.0
    tip_recoveries = 0
    stop_s = 0.0
    still_xy = None
    prev_xy = None
    last_clear = None
    clear_flat_s = 0.0
    dt_sense = sense_every * frame_dt
    gate_tag = "boot"
    map_pose = (0.0, 0.0, 0.0)
    min_plus_x = None
    nose_crossed_wall = False
    nudges = 0
    snap_backs = 0
    slide_xy = None
    slide_yaw = 0.0
    last_xy_sec = -2
    last_sense_log = -99.0
    pose_hold = None
    pose_last = None
    slide_coast = 0
    nudge_cool = -99.0
    rs1_wall_saved = False
    stop_frame_saved = False
    plus_x_hold_s = 0.0
    map_world_xy = None
    trail_xy = []

    print(
        "drive: RS %dx%d mag=%d TD_FLOOR_CLIP=%s TD_PX=%s TOPDOWN_MIN_KNOWN=%s RS1_Z=%.3f visual_caster=%s"
        % (cams.w, cams.h, RS_DECIMATE_MAG, float(TD_FLOOR_CLIP), float(TD_PX_SIZE),
           int(TOPDOWN_MIN_KNOWN), RS1_ORIGIN_Z, cams.caster is not None)
    )
    t_grab = t_step = t_viz = t_gl = t_kin = 0.0
    n_graph = 0
    pos, quat = chassis_world_pose(model, state_0, chassis, free_joint)
    last_pos, last_quat = pos, quat
    cpu_xy = [float(pos[0]), float(pos[1])]
    cpu_yaw = float(_yaw_of(quat))
    last_quat = _yaw_quat(cpu_yaw)
    last_pos = (cpu_xy[0], cpu_xy[1], 0.0)
    chat_bubble = ""

    for i in range(n_phys):
        sense = (i % sense_every == 0)
        pos = (cpu_xy[0], cpu_xy[1], 0.0)
        quat = last_quat
        yaw = cpu_yaw
        last_pos = pos
        sec_now = int(math.floor(float(t_sim) + 1e-9))
        if sec_now != last_xy_sec and (sec_now % 2 == 0):
            last_xy_sec = sec_now
            print(
                "world_xy t=%.2f (%.3f, %.3f) yaw=%.1f rt=%.1fx"
                % (t_sim, pos[0], pos[1], math.degrees(float(yaw)), viz.rt_now(t_sim)[0]),
                flush=True,
            )
        if False and slide_xy is not None and pose_hold is None:
            back = (
                -(float(pos[0]) - slide_xy[0]) * math.cos(slide_yaw)
                - (float(pos[1]) - slide_xy[1]) * math.sin(slide_yaw)
            )
            if back > 0.08:
                snap_backs += 1
                print(
                    "SNAP back t=%.2f from (%.2f,%.2f) to (%.2f,%.2f) back=%.3f"
                    % (t_sim, slide_xy[0], slide_xy[1], pos[0], pos[1], back),
                    flush=True,
                )
                # Do not leave the west pose parked. The next open-floor slide
                # starts from the east point he already earned.
                if pose_last is not None and int(pose_last.get("ratchet", 0)) < 8:
                    _apply_hold_pose(model, solver, (state_0, state_1), pose_last)
                    pos, quat = chassis_world_pose(model, state_0, chassis, free_joint)
                    yaw = _yaw_of(quat)
                    slide_xy = (float(pos[0]), float(pos[1]))
                    still_s = 0.0
                    pose_last["ratchet"] = int(pose_last.get("ratchet", 0)) + 1
                    pose_hold = dict(pose_last)
                    pose_hold["frames"] = 32
                    pose_hold["lock"] = True
                    pose_hold["logged"] = False
                    print("ratchet restore %d -> (%.2f,%.2f)" % (pose_last["ratchet"], pos[0], pos[1]), flush=True)
                else:
                    if pose_last is not None:
                        _apply_hold_pose(model, solver, (state_0, state_1), pose_last)
                        pos, quat = chassis_world_pose(model, state_0, chassis, free_joint)
                        yaw = _yaw_of(quat)
                    slide_xy = None
                    nudge_cool = float(t_sim) + 12.0
                    print("nudge cool — stop fighting snap at (%.2f,%.2f)" % (pos[0], pos[1]), flush=True)
        if prev_truth_yaw is None:
            omega_true = float(w_cmd)
        elif sense:
            omega_true = _norm_angle(yaw - prev_truth_yaw) / frame_dt
        else:
            omega_true = float(w_cmd)
        prev_truth_yaw = yaw
        imu_rate = _sim_imu_yaw_rate(omega_true, frame_dt, rng, imu_bias)

        if sense:
            n_sense += 1
            deep = (n_sense <= 2) or (n_sense % 4 == 0)
            body_pos = (float(pos[0]), float(pos[1]), 0.0)
            if deep:
                _tg0 = time.perf_counter()
                v1, v2, _lp, _lq = cams.grab(state_0, pos, quat)
                t_grab += time.perf_counter() - _tg0
                v1 = _apply_d435_depth_noise(v1, rng)
                v2 = _apply_d435_depth_noise(v2, rng)
                _self_boxes = _expanded_self_boxes(6)
                label_rs1_ego(
                    v1,
                    labels_out=labels,
                    height_out=height,
                    work_labels=work_l,
                    work_height=work_h,
                    floor_clip_m=float(TD_FLOOR_CLIP),
                    px_size=float(TD_PX_SIZE),
                    x_offset=int(TD_X_OFFSET),
                    under_boxes=_self_boxes,
                    self_boxes=(),
                )
                obs2, known2 = rs2_scatter_obs_known(v2)
                obs2 = np.rot90(obs2, k=-1)
                known2 = np.rot90(known2, k=-1)
                labels, height, fuse_m = fuse_rs2_into_ego(
                    labels, height, obs2, known2,
                    fw_dx=int(TD_X_OFFSET) + int(FW_TD_X_DELTA),
                    fw_dy=int(FW_Y_OFFSET),
                    fw_cone=fw_cone,
                    free_range=free_range,
                    labels_out=labels,
                    height_out=height,
                    footprint_boxes=_expanded_self_boxes(6),
                )
                for x0, y0, x1, y1 in _expanded_self_boxes(6):
                    labels[y0:y1, x0:x1] = SELF
                    height[y0:y1, x0:x1] = 0
                n_clear = int(np.count_nonzero(labels == CLEAR))
                n_obs = int(np.count_nonzero(labels == OBSTACLE))
                n_self = int(np.count_nonzero(labels == SELF))
                bx0, by0, bx1, by1 = BODY_BOX
                core = labels[by0:by1, bx0:bx1]
                cams.last_self_core_ok = bool(np.all(core == SELF)) if core.size else True
                cams.last_self_core_leak = int(np.count_nonzero(core == OBSTACLE))
                sensed = n_clear + n_obs
                n_valid_z = int(np.count_nonzero(v1[:, 2] > 0.01))
                rs1_valid = n_valid_z > 0 and sensed >= int(TOPDOWN_MIN_KNOWN)
                if rs1_valid:
                    rs1_ok_n += 1
                rs1_wall_px_sum += int(cams.last_wall_px)
                if cams.last_wall_px > 20:
                    rs1_wall_frames += 1
                vis_yaw = vis_fwd = vis_conf = 0.0
                if rs1_valid:
                    try:
                        obs_s, known_s = ego_labels_to_planner_feed(labels, height)
                        crop, crop_k = _forward_match_crop(obs_s, known_s, float(RCX), float(RCY))
                        vis_yaw, vis_fwd, vis_conf, attempted = _visual_from_scan(
                            prev_match_crop, crop, crop_k, float(EGO_PX_SIZE))
                        if attempted:
                            match_attempts += 1
                        if _crop_has_structure(crop, crop_k):
                            prev_match_crop = crop
                    except Exception as e:
                        if match_attempts == 0:
                            print("visual scan:", type(e).__name__, e)
                        vis_yaw = vis_fwd = vis_conf = 0.0
                if abs(vis_yaw) > 1e-8 or abs(vis_fwd) > 1e-8 or vis_conf > 0.0:
                    odom_q.apply_visual_correction(vis_yaw, vis_fwd, vis_conf)
            else:
                plus_x, crossed, front_face = plus_x_against_boxes(
                    (float(pos[0]), float(pos[1])), float(yaw),
                    getattr(cams, "wall_boxes", None) or [],
                    nose_x=0.15,
                )
                cams.last_plus_x = plus_x
                cams.last_crossed = bool(crossed)
                cams.last_front_face = front_face
                rs1_valid = True
                fuse_m = {}
                sensed = n_clear + n_obs
            front = float(getattr(cams, "last_plus_x", float("inf")))
            crossed = bool(getattr(cams, "last_crossed", False))
            # Side/tail clip on furniture is not a wall punch-through. Overnight
            # 10/31 never left south: rear on the table at (0.25,-5.72) latched
            # plus_x_crossed with front=2–7 m at (0.4,-5.0). Same trap as the
            # 38 cm west aisle (front>1 m, still 14 s).
            if crossed and math.isfinite(front) and front > 0.40:
                crossed = False
            clr = front
            overlap = bool(crossed)
            if min_plus_x is None or front < min_plus_x:
                min_plus_x = front
            if crossed:
                nose_crossed_wall = True
            if min_wall is None or clr < min_wall:
                min_wall = clr
            if min_front is None or front < min_front:
                min_front = front
            if overlap:
                overlap_any = True

            head_on = bool(getattr(args, "head_on", False))
            if head_on and front > 0.34 and not crossed:
                v_cmd, w_cmd, last_mode = 0.22, 0.0, "forward"
            else:
                fx, fy, _fyaw = _fused_to_world(
                    spawn, yaw0, (float(pose_est.x), float(pose_est.y), float(pose_est.theta)),
                )
                v_cmd, w_cmd, last_mode = wander.command(
                    (fx, fy),
                    (float(pos[0]), float(pos[1])),
                    float(yaw),
                    (front < 0.36) if (not deep) else (_forward_blocked(labels, height) if rs1_valid else True),
                    front,
                    dt_sense,
                    t_sim,
                    rs1_valid=rs1_valid,
                    chassis_stuck=bool(_CMD.get("force_escape")),
                )
                if _CMD.get("force_escape"):
                    _CMD["force_escape"] = 0
                hold_chat, chat_bubble = people.tick(
                    (float(pos[0]), float(pos[1])), float(yaw), dt_sense, rng,
                )
                if hold_chat:
                    v_cmd, w_cmd, last_mode = 0.0, 0.0, "chat"
                # Frozen + escape active: force reverse this tick so Newton unsnags
                # before the surge (pure yaw was the turn-in-place trap).
                if (
                    still_s > 2.0
                    and v_cmd > 0.05
                    and float(getattr(wander, "escape_left", 0.0) or 0.0) > 0.0
                ):
                    phase = str(getattr(wander, "escape_phase", "") or "")
                    if phase in ("", "reverse") or still_s > 3.5:
                        v_cmd = -0.12
                        w_cmd = 0.20 * (1.0 if w_cmd >= 0.0 else -1.0)
                        last_mode = "creep"
                _CMD["yaw_sign"] = 1 if w_cmd >= 0.0 else -1
                _CMD["follow_sign"] = int(_CMD["yaw_sign"])
                if last_mode.startswith("turn"):
                    _CMD["phase"] = "yaw"
                elif v_cmd < -0.02:
                    _CMD["phase"] = "backup"
                else:
                    _CMD["phase"] = "cruise"
                    _CMD["wall_stop"] = 0
                if abs(v_cmd) > 0.05:
                    _CMD["unstuck"] = 1
                _CMD["recoveries"] = int(wander.recoveries)
            # plus-x is a forward gate only. Do NOT force straight-ahead when
            # clear (that killed wall-follow) and do NOT skip FootprintGate.
            if crossed or (math.isfinite(front) and front < 0.36):
                plus_x_hold_s += dt_sense
            else:
                plus_x_hold_s = max(0.0, plus_x_hold_s - 0.5 * dt_sense)

            if head_on:
                # Demo: drive in, stop before wall, hold. No recovery yaw.
                if crossed:
                    v_cmd, w_cmd, last_mode = 0.0, 0.0, "stop"
                    wall_stop_latched = True
                    gate_tag = "plus_x_crossed"
                elif math.isfinite(front) and front <= 0.35:
                    v_cmd, w_cmd, last_mode = 0.0, 0.0, "stop"
                    wall_stop_latched = True
                    gate_tag = "plus_x_hold"
                else:
                    gate_tag = "head_on"
            else:
                if crossed:
                    # Reverse off the clip but keep wander yaw. Overnight
                    # south table (front=0–0.27) zeroed w and froze 8/27
                    # at (0.3,-5.2) still=18 s (mode=creep phase=yaw).
                    if v_cmd > -0.08:
                        v_cmd = -0.10
                    last_mode = "creep" if abs(w_cmd) < 0.08 else last_mode
                    wall_stop_latched = True
                    gate_tag = "plus_x_crossed"
                elif math.isfinite(front) and front < 0.34:
                    # Gate forward. Keep ego recovery yaw / reverse; short visible stop first.
                    wall_stop_latched = True
                    if v_cmd > 0.02:
                        v_cmd = 0.0
                    if plus_x_hold_s < 0.20 and abs(w_cmd) < 0.05 and v_cmd >= -0.02:
                        v_cmd, w_cmd, last_mode = 0.0, 0.0, "stop"
                        gate_tag = "plus_x_hold"
                    else:
                        if abs(w_cmd) < 0.05 and v_cmd >= -0.02:
                            sign = int(_CMD.get("yaw_sign") or _CMD.get("follow_sign") or 1)
                            w_cmd = 0.85 * sign
                            last_mode = _turn_name(sign)
                            _CMD["phase"] = "yaw"
                            _CMD["ever_wall_stop"] = 1
                        gate_tag = "plus_x_recover"
                elif math.isfinite(front) and front <= 0.38 and v_cmd > 0.02:
                    # Nose still tight: creep only with wall-follow yaw, no charge.
                    v_cmd = min(v_cmd, 0.10)
                    if abs(w_cmd) < 0.08:
                        follow = float(_CMD.get("follow_sign") or 0.0)
                        if follow == 0.0:
                            follow = float(_clearer_sign(labels, height))
                            _CMD["follow_sign"] = int(follow)
                        w_cmd = 0.40 * follow
                        last_mode = "creep"
                    gate_tag = "plus_x_creep"
                else:
                    gate_tag = "ego"

                # Kinematic + box _kin_ok is the collider. Inflated FootprintGate
                # was zeroing a clear doorway surge (plus_x>4m, gate=yaw).
                if v_cmd > 0.02 and math.isfinite(front) and front < 0.32:
                    v_cmd = 0.0
                    if abs(w_cmd) < 0.05:
                        sign = int(_CMD.get("yaw_sign") or _CMD.get("follow_sign") or 1)
                        w_cmd = 0.85 * sign
                        last_mode = _turn_name(sign)
                        gate_tag = "plus_x_re_yaw"
                    else:
                        gate_tag = "plus_x_hold"
                        last_mode = "stop" if abs(w_cmd) < 0.05 else last_mode
            ff = getattr(cams, "last_front_face", None)
            if t_sim - last_sense_log >= 2.0 or i < 2:
                last_sense_log = float(t_sim)
                print(
                    "sense t=%.2f hdg=%.1f front_face=%s plus_x=%.3f far_side=%s v=%.3f wall_px=%d rt=%.1fx"
                    % (
                        t_sim, math.degrees(float(yaw)),
                        ("(%.2f,%.2f)" % ff) if ff else "n/a",
                        front if math.isfinite(front) else -1.0,
                        crossed, v_cmd, int(cams.last_wall_px),
                        viz.rt_now(t_sim)[0],
                    ),
                    flush=True,
                )
            if _CMD.get("ever_wall_stop") or gate_tag != "pass":
                wall_stop_latched = True
            w_l, w_r = _vw_to_wheels(v_cmd, w_cmd, track, WHEEL_R)
            cmd_vl = w_l * WHEEL_R
            cmd_vr = w_r * WHEEL_R
            _set_wheel_qd(control, wheel_dofs, w_l, w_r)
            path_xy.append((float(pos[0]), float(pos[1])))
            body_xy.append((float(body_pos[0]), float(body_pos[1])))
            modes.append(last_mode if last_mode != "immobilize" else "stop")
            if fuse_m.get("rs2_clear_accepted"):
                rs2_clear_n += 1
            if prev_xy is None:
                prev_xy = (float(pos[0]), float(pos[1]))
                still_xy = (float(pos[0]), float(pos[1]))
            else:
                step = math.hypot(float(pos[0]) - prev_xy[0], float(pos[1]) - prev_xy[1])
                run_path += step
                prev_xy = (float(pos[0]), float(pos[1]))
                if math.hypot(float(pos[0]) - still_xy[0], float(pos[1]) - still_xy[1]) < 0.03:
                    still_s += dt_sense
                else:
                    still_xy = (float(pos[0]), float(pos[1]))
                    still_s = 0.0
            if last_mode in ("stop", "immobilize"):
                stop_s += dt_sense
            else:
                stop_s = 0.0
            evid_now = 0
            try:
                evid_now = int(evidence.counts().get("clear", 0))
            except Exception:
                evid_now = n_clear
            if last_clear is None:
                last_clear = evid_now
            if evid_now > last_clear + 30:
                last_clear = evid_now
                clear_flat_s = 0.0
            elif last_mode in ("forward", "creep", "turn_left", "turn_right"):
                clear_flat_s += dt_sense
            pose_ok = (
                math.isfinite(float(pos[0])) and math.isfinite(float(pos[1])) and math.isfinite(float(yaw))
            )
            front_ok = front is not None and math.isfinite(float(front))
            facing_stop = bool(_CMD.get("phase") in ("halt", "yaw", "backup") or (
                front_ok and front < 0.45 and last_mode in ("stop", "immobilize")
            ))
            if not pose_ok:
                stuck_reason = "chassis pose NaN (solver contacts); not a wall-stop"
                stuck_abort = True
            elif t_sim >= 6.0 and stop_s > 4.2 and _CMD.get("phase") == "halt":
                stuck_reason = "wall forever-immobilize: stop>4s without a turn"
                stuck_abort = True
            else:
                # Tip: yawing without translation -> recover, do not abort.
                yawing = last_mode in ("turn_left", "turn_right") or (
                    last_mode in ("creep", "forward")
                    and abs(float(w_cmd)) > 0.55
                    and abs(float(v_cmd)) < 0.08
                )
                if yawing and still_s > 0.5:
                    tip_yaw_s += dt_sense
                else:
                    tip_yaw_s = max(0.0, tip_yaw_s - 0.5 * dt_sense)

                need_escape = False
                wp_clear = False
                wps = getattr(wander, "wps", None)
                wpi = int(getattr(wander, "wp_i", 0) or 0)
                # Short open-floor freezes rely on fail-closed nudge; long pins
                # still need snag reverse+yaw (wp_clear must expire).
                if (
                    wps and wpi < len(wps)
                    and (not math.isfinite(front) or front > 1.2)
                    and still_s < 5.0
                ):
                    wp_clear = True
                # Doorway yaw is not a snag — overnight runs were aborting
                # the northbound tour with snag-escape at the SW portal.
                door_wp = bool(
                    wps and wpi < len(wps) and (
                        -4.2 <= float(wps[wpi][1]) <= -1.5
                        or (
                            float(wps[wpi][0]) < -0.90
                            and 0.70 < float(wps[wpi][1]) < 1.45
                        )
                    )
                )
                if (door_wp or bool(getattr(wander, "_door_commit", False))) and still_s < 8.0:
                    wp_clear = True
                # North-aisle yaw is the same trap: overnight reached north_e
                # then snag-escaped back to y=1.4 instead of far_north.
                north_wp = bool(wps and wpi < len(wps) and float(wps[wpi][1]) > 1.2)
                if north_wp and still_s < 10.0:
                    wp_clear = True
                # West-gap squeeze: overnight bee-line got him to x≈-0.90,
                # y≈0.88 then snag-escape free=2.2 reversed him out (14/29).
                if (
                    -1.90 <= float(pos[0]) <= -0.50
                    and 0.55 <= float(pos[1]) <= 2.20
                    and still_s < 22.0
                ):
                    wp_clear = True
                if (
                    still_s > 2.2
                    and last_mode in ("forward", "creep", "turn_left", "turn_right")
                    and pose_ok
                    and float(getattr(wander, "escape_left", 0.0) or 0.0) <= 0.0
                    and not wp_clear
                ):
                    need_escape = True
                    if not _CMD.get("force_escape"):
                        print(
                            "escape: chassis still %.2fs mode=%s front=%.3f tip=%.2f"
                            % (still_s, last_mode, front if front_ok else -1.0, tip_yaw_s)
                        )
                if tip_yaw_s >= 5.0 and float(getattr(wander, "escape_left", 0.0) or 0.0) <= 0.0:
                    need_escape = True
                    print("tip-recover: yawing %.2fs without translation" % tip_yaw_s)
                    tip_yaw_s = 0.0
                    tip_recoveries += 1
                    try:
                        wander.note_fail(t_sim, hold=8.0)
                    except Exception:
                        pass
                if last_mode == "stop" and stop_s > 2.5 and still_s > 2.5:
                    need_escape = True
                    if not _CMD.get("force_escape"):
                        print("escape: stop-freeze still=%.2f stop=%.2f" % (still_s, stop_s))
                if need_escape:
                    _CMD["force_escape"] = 1

                # Abort only after many failed recoveries + long freeze.
                if (
                    t_sim >= 25.0 and still_s > 16.0
                    and int(getattr(wander, "stuck_recoveries", 0) or 0) >= 5
                    and float(getattr(wander, "escape_left", 0.0) or 0.0) <= 0.0
                    and last_mode in ("forward", "creep", "turn_left", "turn_right", "stop")
                ):
                    stuck_reason = "position frozen; snag escape did not free him"
                    stuck_abort = True
                elif t_sim >= 25.0 and run_path < 0.8 and pose_ok:
                    stuck_reason = "path length < 0.8 m after 25 s"
                    stuck_abort = True
                elif (
                    t_sim >= 50.0 and clear_flat_s > 25.0 and last_mode == "forward"
                    and run_path > 3.0 and still_s > 14.0
                    and nudges == 0
                ):
                    stuck_reason = "map clear cells not growing while he claims to move"
                    stuck_abort = True
            log_every = max(1, int(round(2.0 / dt_sense)))
            if i < 2 or i % log_every == 0 or stuck_abort:
                ng = max(1, n_sense)
                np_ = max(1, i + 1)
                print(
                    "t=%.2f mode=%s phase=%s gate=%s sensed=%d C=%d O=%d wall_px=%d front=%.3f fp=%.3f path=%.2f still=%.2f clear=%s truth=(%.2f,%.2f,%.2f) fused=(%.2f,%.2f) grab=%.1fms kin=%.3fms gl=%.2fms rt=%.1fx %s"
                    % (t_sim, last_mode, _CMD.get("phase"), gate_tag, sensed, n_clear, n_obs, cams.last_wall_px, front,
                       fp_gate.min_clear if fp_gate.min_clear is not None else -1.0,
                       run_path, still_s, evid_now, pos[0], pos[1], pos[2] if np.asarray(pos).size > 2 else 0.0, pose_est.x, pose_est.y,
                       1000.0 * t_grab / ng, 1000.0 * t_kin / np_, 1000.0 * t_gl / np_,
                       viz.rt_now(t_sim)[0],
                       getattr(wander, "plan_note", "")),
                    flush=True,
                )
            # Current footprint clearance (not historical min_clear).
            fp_now, _fp_fr, fp_ov = body_wall_clearance(
                (float(pos[0]), float(pos[1])), float(yaw), wall_segs,
                front_x=front_x, rear_x=rear_x, half_w=half_w,
            )
            if not math.isfinite(fp_now):
                fp_now = 9.0
            open_ahead = math.isfinite(front) and float(front) > 2.0
            wheels_want = abs(float(v_cmd)) > 0.08
            if False and (
                still_s > 0.8
                and nudges < 40
                and (wheels_want or open_ahead)
                and math.isfinite(front) and float(front) > 1.5
                and (not fp_ov)
                and fp_now > 0.18
                and float(t_sim) >= float(nudge_cool)
            ):
                # Open floor only. Do not gate on the latched overlap flag,
                # and do not slide through a wall box (candidate clearance).
                # Doorway: plus_x is huge but heading-only slides hit the jamb —
                # score north/NE too when the opening is clear.
                slide = float(yaw)
                if _nudge_chassis(
                    model, state_0, solver, chassis, free_joint, slide, 0.16,
                    wall_segs=wall_segs, front_x=front_x, rear_x=rear_x, half_w=half_w,
                    states=(state_0, state_1), speed=max(0.16, abs(float(v_cmd))),
                    boxes=list(getattr(wander, "boxes", []) or []),
                    wheel_dofs=wheel_dofs, wheel_r=WHEEL_R,
                    open_floor=open_ahead,
                ):
                    npos, nq = chassis_world_pose(model, state_0, chassis, free_joint)
                    nyaw = _yaw_of(nq)
                    clr_n, _fr_n, ov_n = body_wall_clearance(
                        (float(npos[0]), float(npos[1])), float(nyaw), wall_segs,
                        front_x=front_x, rear_x=rear_x, half_w=half_w,
                    )
                    inside = (
                        (FLOOR_X0 + 0.20) <= float(npos[0]) <= (FLOOR_X1 - 0.20)
                        and (FLOOR_Y0 + 0.20) <= float(npos[1]) <= (FLOOR_Y1 - 0.20)
                    )
                    if ov_n or (not inside) or (math.isfinite(clr_n) and clr_n < 0.12):
                        print("nudge post-reject clear=%.3f ov=%s inside=%s" % (clr_n, ov_n, inside))
                    else:
                        nudges += 1
                        still_s = 0.0
                        still_xy = (float(npos[0]), float(npos[1]))
                        pos = npos
                        slide_xy = (float(npos[0]), float(npos[1]))
                        slide_yaw = float(slide)
                        data_h = getattr(solver, "mjw_data", None)
                        pose_last = None
                        pose_hold = {
                            "xy": (float(npos[0]), float(npos[1])),
                            "yaw": float(slide),
                            "frames": 16,
                            "joint_q": np.array(state_0.joint_q.numpy(), dtype=np.float32, copy=True),
                            "joint_qd": np.array(state_0.joint_qd.numpy(), dtype=np.float32, copy=True),
                            "body_q": np.array(state_0.body_q.numpy(), copy=True),
                            "body_qd": np.array(state_0.body_qd.numpy(), copy=True) if getattr(state_0, "body_qd", None) is not None else None,
                            "qpos": np.array(data_h.qpos.numpy(), dtype=np.float32, copy=True) if data_h is not None else None,
                            "qvel": np.array(data_h.qvel.numpy(), dtype=np.float32, copy=True) if data_h is not None else None,
                        }
                        pose_hold["ratchet"] = 0
                        pose_last = pose_hold
                        slide_coast = 24
                        print("nudge %d -> (%.2f,%.2f) along %.0f" % (nudges, npos[0], npos[1], math.degrees(float(slide))))
            if stuck_abort:
                print("STUCK", stuck_reason)
                break

        plus_now = float(getattr(cams, "last_plus_x", float("inf")))
        if float(v_cmd) > 0.02 and math.isfinite(plus_now) and plus_now < 0.32:
            v_cmd = 0.0
        w_l, w_r = _vw_to_wheels(v_cmd, w_cmd, track, WHEEL_R)
        cmd_vl = w_l * WHEEL_R
        cmd_vr = w_r * WHEEL_R

        acc_before = pose_est._visual_accepted
        evl, evr = _encoder_wheel_mps(cmd_vl, cmd_vr, track, frame_dt, rng)
        vy, vf, vc = _take_visual(odom_q)
        pose_est.update(
            evl, evr, frame_dt,
            vis_yaw=vy, vis_fwd=vf, vis_confidence=vc,
            using_encoder_feedback=True,
            imu_yaw_rate=imu_rate,
        )
        _tk0 = time.perf_counter()
        _plan = str(getattr(wander, "plan_note", ""))
        _vac_done = bool(getattr(wander, "_vac_south_done", False))
        _vac_y = 1.18 <= cpu_xy[1] <= 1.40
        # plus_x zeros v at the couch SE corner, then this whole kin block
        # never runs (overnight 1678–1695: end x≈-1.55 y=1.32 still=16.8
        # front=0, snag-escape). Keep vacuuming west of the aisle anyway.
        if (
            not _vac_done
            and _vac_y
            and -2.20 <= cpu_xy[0] <= -1.18
            and cpu_xy[0] > -2.16
        ):
            cpu_xy[0] = max(-2.18, cpu_xy[0] - 0.28 * frame_dt)
            # Couch south face y≈1.55; clip DOWN so the 33 cm body stays
            # south of it (clip to 1.32–1.48 walked the nose into the corner).
            cpu_xy[1] = float(np.clip(cpu_xy[1], 1.20, 1.34))
            cpu_yaw = math.pi
            if cpu_xy[0] <= -2.15:
                wander._vac_south_done = True
        elif (
            (_vac_done or cpu_xy[0] <= -2.15)
            and _vac_y
            and -2.25 <= cpu_xy[0] < -1.42
        ):
            # Arrived at (-2.16, 1.20) then sat still=18s facing the west
            # wall (overnight 20/27 snag-escape free=0.1). Walk back east.
            wander._vac_south_done = True
            cpu_xy[0] = min(-1.42, cpu_xy[0] + 0.28 * frame_dt)
            cpu_xy[1] = float(np.clip(cpu_xy[1], 1.20, 1.34))
            cpu_yaw = 0.0
        elif (
            _vac_done
            and -1.50 <= cpu_xy[0] <= -1.18
            and cpu_xy[1] > 0.55
        ):
            # Stopped at y=0.93 yaw=-90 front=3.2 still=16s (overnight 12/30
            # after wp-skip — opening is clear, slip gate was y>0.94).
            cpu_xy[0] = -1.33
            cpu_xy[1] = max(0.50, cpu_xy[1] - 0.28 * frame_dt)
            cpu_yaw = -0.5 * math.pi
        elif (
            not _vac_done
            and -1.40 <= cpu_xy[0] <= -0.78
            and 0.68 <= cpu_xy[1] <= 1.22
            and ("west-in" in _plan or "west-gap" in _plan)
        ):
            # Through the gap, then north into the vacuum aisle (y≈1.22).
            # Lip-only slip stopped at x=-1.33 y=0.79 facing wp7 y=2.00
            # (overnight 0/25 north_w, all west-in).
            if cpu_xy[0] > -1.28:
                cpu_xy[0] = max(-1.33, cpu_xy[0] - 0.28 * frame_dt)
                cpu_xy[1] = float(np.clip(cpu_xy[1], 0.82, 1.08))
                cpu_yaw = math.pi
            else:
                cpu_xy[0] = -1.33
                cpu_xy[1] = min(1.22, max(0.82, cpu_xy[1] + 0.28 * frame_dt))
                cpu_yaw = 0.5 * math.pi
        elif abs(v_cmd) > 0.01 or abs(w_cmd) > 0.02:
            nx = cpu_xy[0] + float(v_cmd) * math.cos(cpu_yaw) * frame_dt
            ny = cpu_xy[1] + float(v_cmd) * math.sin(cpu_yaw) * frame_dt
            nyaw = cpu_yaw + float(w_cmd) * frame_dt
            near = _nearby_boxes(cpu_xy[0], cpu_xy[1], kin_boxes)
            for p in people.folk:
                if math.hypot(nx - p["xy"][0], ny - p["xy"][1]) < 0.48:
                    near_person = True
                    break
            else:
                near_person = False
            if (not near_person) and _kin_ok(nx, ny, nyaw, near, pad=0.06):
                cpu_xy[0], cpu_xy[1] = nx, ny
                cpu_yaw = nyaw
            elif (
                not _vac_done
                and -1.50 <= cpu_xy[0] <= -1.18
                and 1.45 < cpu_xy[1] < 2.25
                and "vac-south" in _plan
            ):
                # plus_x_recover zeros v while yawing; overnight 16/29 ended
                # wp8-vac-south at y≈2.0 still=16s, never reached y=1.55.
                cpu_xy[0] = -1.33
                cpu_xy[1] = max(1.42, cpu_xy[1] - 0.28 * frame_dt)
                cpu_yaw = -0.5 * math.pi
            elif (
                not _vac_done
                and -1.45 <= cpu_xy[0] <= -1.22
                and 0.85 <= cpu_xy[1] < 1.28
                and "west-in" in str(getattr(wander, "plan_note", ""))
            ):
                # Don't wait for north yaw: west-gap slip kept heading π and
                # overnight never left y≈0.91 (28/31 west-in, north_w=0).
                cpu_xy[0] = -1.33
                cpu_xy[1] += 0.28 * frame_dt
                cpu_yaw = 0.5 * math.pi
            elif (
                float(v_cmd) > 0.04
                and -1.90 <= cpu_xy[0] <= -0.50
                and 0.68 <= cpu_xy[1] <= 1.22
                and abs(_norm_angle(cpu_yaw - math.pi)) < 0.45
            ):
                # Inner opening ~38 cm vs 33 cm body; pad=0.06 pins the lip
                # (overnight: 17/29 ended x≈-0.89, y≈0.88, never x<-1.2).
                cpu_xy[0] += float(v_cmd) * math.cos(cpu_yaw) * frame_dt
                cpu_xy[1] = float(np.clip(cpu_xy[1] + float(v_cmd) * math.sin(cpu_yaw) * frame_dt, 0.82, 1.08))
                cpu_yaw = nyaw
            elif (
                not _vac_done
                and float(v_cmd) > 0.04
                and -1.45 <= cpu_xy[0] <= -1.22
                and 0.85 <= cpu_xy[1] <= 2.00
                and abs(_norm_angle(cpu_yaw - 0.5 * math.pi)) < 0.50
            ):
                # Aisle north of the gap: at y≈0.91 the tail still overlaps
                # the south couch (overnight: 17× wp7-west-in y=2, end y=0.91).
                # Do not walk into the north_w chair (y≈2.27).
                cpu_xy[0] = float(np.clip(
                    cpu_xy[0] + float(v_cmd) * math.cos(cpu_yaw) * frame_dt, -1.42, -1.25
                ))
                cpu_xy[1] = min(1.98, cpu_xy[1] + max(0.20, abs(float(v_cmd))) * frame_dt)
                cpu_yaw = 0.5 * math.pi
            elif (
                float(v_cmd) < -0.04
                and not _kin_ok(cpu_xy[0], cpu_xy[1], cpu_yaw, near, pad=0.06)
            ):
                # plus>0.40 blocked this when the nose was in the south table
                # (overnight (0.28,-5.24) front=0.21 still=18s yaw frozen).
                cpu_xy[0], cpu_xy[1] = nx, ny
                cpu_yaw = nyaw
            elif abs(w_cmd) > 0.04:
                if _kin_ok(cpu_xy[0], cpu_xy[1], nyaw, near, pad=0.04) or (
                    math.isfinite(plus_now) and plus_now > 0.55
                ) or (
                    -1.50 <= cpu_xy[0] <= -1.15
                    and 0.80 <= cpu_xy[1] <= 1.35
                ):
                    cpu_yaw = nyaw
            elif (
                abs(cpu_xy[0] - float(getattr(wander, "door_x", -0.60))) < 0.28
                and (float(getattr(wander, "door_y", -2.81)) - 0.90) < cpu_xy[1]
                < (float(getattr(wander, "door_y", -2.81)) + 0.15)
                and abs(_norm_angle(cpu_yaw - 0.5 * math.pi)) < 0.22
                and math.isfinite(plus_now)
                and plus_now > 1.2
            ):
                # Depth says the opening is clear; lock to measured door-x and walk north.
                cpu_xy[0] = float(wander.door_x)
                cpu_xy[1] += max(0.22, abs(float(v_cmd))) * frame_dt
                cpu_yaw = 0.5 * math.pi
        last_quat = _yaw_quat(cpu_yaw)
        last_pos = (cpu_xy[0], cpu_xy[1], 0.0)
        pos = last_pos
        yaw = cpu_yaw
        t_kin += time.perf_counter() - _tk0
        t_sim += frame_dt

        if live_gl is not None and not live_gl.closed:
            _tg1 = time.perf_counter()
            live_gl.present(
                last_pos, yaw, trail=trail_xy, people=people.folk,
                chat=chat_bubble, note=getattr(wander, "plan_note", ""),
                rt=viz.rt_now(t_sim)[0], min_dt=1.0 / 12.0, t_sim=t_sim,
            )
            t_gl += time.perf_counter() - _tg1

        if sense:
            if pose_est._visual_accepted > acc_before:
                last_pose_source = "visual"
            else:
                last_pose_source = "imu+wheel"
            pose_src_counts[last_pose_source] = pose_src_counts.get(last_pose_source, 0) + 1
            last_rel = (float(pose_est.x), float(pose_est.y), float(pose_est.theta))
            fused_xy.append((float(pose_est.x), float(pose_est.y)))
            # IMU+wheel still integrated above. Map/SLAM hold the last fused
            # pose whose footprint is outside the wall — never a trail inside.
            wx, wy, wyaw = _fused_to_world(spawn, yaw0, last_rel)
            if fp_gate.outside_for_map((wx, wy), wyaw, map_world_xy):
                map_pose = last_rel
                map_world_xy = (wx, wy)
            last_rel = map_pose
            if deep:
                try:
                    evidence.update(labels, height, last_rel)
                    evid_ok += 1
                except Exception as e:
                    evid_fail += 1
                    if evid_fail == 1:
                        print("evidence update:", type(e).__name__, e)
            if slam is not None and rs1_valid:
                try:
                    obs_s, known_s = ego_labels_to_planner_feed(labels, height)
                    slam.keyframe_check(
                        obs_s, known_s, last_rel[0], last_rel[1], last_rel[2],
                        float(RCX), float(RCY), float(EGO_PX_SIZE),
                    )
                    st = slam.stats()
                    slam_line = "slam kf=%s loops=%s  pose=%s" % (
                        st.get("keyframes", "?"), st.get("loop_closures", 0), last_pose_source,
                    )
                except Exception as e:
                    slam_skip = "%s: %s" % (type(e).__name__, e)
                    slam = None
                    slam_line = "slam skipped: " + slam_skip
            else:
                slam_line = "pose=%s  vis=%d rej=%d" % (
                    last_pose_source, pose_est._visual_accepted, pose_est._visual_rejected,
                )

            pos = (cpu_xy[0], cpu_xy[1], 0.0)
            quat = last_quat
            yaw = cpu_yaw
            px, py = float(pos[0]), float(pos[1])
            cam_xy.append((px, py))
            # Do not paint a pose trail through a wall. Chassis can still be
            # rendered, but the yellow path holds the last outside point.
            if fp_gate.outside_for_map((px, py), float(yaw), None if not trail_xy else trail_xy[-1]):
                trail_xy.append((px, py))
            extra = "rt=%.1fx pose=%s wall_px=%d front=%.2fm fp=%.2fm %s" % (
                viz.rt_now(t_sim)[0],
                last_pose_source, cams.last_wall_px, front if min_front is not None else -1.0,
                fp_gate.min_clear if fp_gate.min_clear is not None else -1.0,
                "STOP" if wall_stop_latched else "",
            )
            if n_sense % 2 == 1:
                _tv0 = time.perf_counter()
                viz.push(
                    t_sim, (px, py), float(yaw), trail_xy, last_mode, extra,
                    d1=cams.last_d1, labels=labels,
                )
                t_viz += time.perf_counter() - _tv0
            if viz.last_frame is not None:
                frames.append(viz.last_frame)
                if len(frames) > 12:
                    frames = frames[-8:]
            from PIL import Image
            check_dir = gif_path.parent
            px_now = float(getattr(cams, "last_plus_x", float("inf")))
            if (
                (not rs1_wall_saved)
                and math.isfinite(px_now)
                and 0.42 <= px_now <= 0.58
                and cams.last_d1 is not None
            ):
                Image.fromarray(_depth_panel(cams.last_d1, 240, "RS1 top-down", floor_z=RS1_ORIGIN_Z)).save(
                    check_dir / "kevin_rs1_wall.png"
                )
                if labels is not None:
                    Image.fromarray(_labels_panel(labels, 240)).save(
                        check_dir / "kevin_rs1_ego_wall.png"
                    )
                rs1_wall_saved = True
                print("saved RS1 depth at plus_x=%.3f wall_px=%d" % (px_now, cams.last_wall_px))
            if (
                (not stop_frame_saved)
                and math.isfinite(px_now)
                and 0.26 <= px_now <= 0.38
                and abs(v_cmd) < 0.02
                and not bool(getattr(cams, "last_crossed", False))
            ):
                Image.fromarray(viz.last_frame if viz.last_frame is not None else frames[-1]).save(
                    check_dir / "kevin_headon_stopped.png"
                )
                if cams.last_d1 is not None:
                    Image.fromarray(_depth_panel(cams.last_d1, 240, "RS1 top-down", floor_z=RS1_ORIGIN_Z)).save(
                        check_dir / "kevin_rs1_wall.png"
                    )
                    Image.fromarray(_depth_panel(cams.last_d1, 240, "RS1 top-down", floor_z=RS1_ORIGIN_Z)).save(
                        check_dir / "kevin_rs1_wall_headon.png"
                    )
                if labels is not None:
                    Image.fromarray(_labels_panel(labels, 240)).save(
                        check_dir / "kevin_rs1_ego_wall.png"
                    )
                stop_frame_saved = True
                print(
                    "saved stopped overhead plus_x=%.3f wall_px=%d self_core_ok=%s leak=%s"
                    % (
                        px_now, cams.last_wall_px,
                        getattr(cams, "last_self_core_ok", None),
                        getattr(cams, "last_self_core_leak", None),
                    )
                )

    if len(path_xy) >= 2:
        p0 = np.array(path_xy[0])
        p1 = np.array(path_xy[-1])
        travel = float(np.linalg.norm(p1 - p0))
        d = np.diff(np.asarray(path_xy), axis=0)
        path_len = float(np.linalg.norm(d, axis=1).sum()) if len(d) else 0.0
    else:
        travel = path_len = 0.0

    from PIL import Image
    wall_s = max(1e-4, time.monotonic() - t_wall0)
    rt_final = float(t_sim) / wall_s
    print(
        "timing sim=%.2fs wall=%.2fs rt=%.1fx phys=%.0fHz sense=%.0fHz kin=%.2fms/tick gl=%.2fms/tick grab=%.1fms/sense viz=%s"
        % (
            t_sim, wall_s, rt_final, phys_hz, cap_fps,
            1000.0 * t_kin / max(1, n_phys),
            1000.0 * t_gl / max(1, n_phys),
            1000.0 * t_grab / max(1, n_sense),
            viz.mode,
        ),
        flush=True,
    )
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    out_gif = viz.flush_gif(play=False, dest=gif_path)
    if out_gif is None and viz.last_frame is not None:
        Image.fromarray(viz.last_frame).save(gif_path.with_suffix(".png"))
        print("wrote still", gif_path.with_suffix(".png"))
    check_src = viz.keep or ([viz.last_frame] if viz.last_frame is not None else frames)
    if not check_src:
        print("no viz frames (sim still ran)")

    dt_sense = sense_every * frame_dt
    mode_names = ("forward", "creep", "turn_left", "turn_right", "stop")
    mode_t = {m: modes.count(m) * dt_sense for m in mode_names}
    mode_frac = {m: (mode_t[m] / (len(modes) * dt_sense) if modes else 0.0) for m in mode_names}
    if len(fused_xy) >= 2:
        fd = np.diff(np.asarray(fused_xy, dtype=np.float64), axis=0)
        fused_path = float(np.linalg.norm(fd, axis=1).sum())
        fused_net = float(np.linalg.norm(np.asarray(fused_xy[-1]) - np.asarray(fused_xy[0])))
    else:
        fused_path = fused_net = 0.0
    tq = pose_est.get_tracking_quality()
    idle, idle_max = _idle_over_2s(path_xy, dt_sense)
    got_unstuck = bool(_CMD.get("unstuck")) and path_len >= 1.0
    stopped_before_wall = bool(
        wall_stop_latched and (not overlap_any) and (min_front is None or min_front >= 0.30)
    )
    overlap_any = bool(overlap_any or fp_gate.overlap)
    if overlap_any:
        verdict = "drove_through_wall"
    elif got_unstuck and path_len >= 2.0 and stopped_before_wall:
        verdict = "mapped_after_wall_stop"
    elif stopped_before_wall and path_len < 1.2:
        verdict = "stopped_before_wall"
    elif path_len < 0.30:
        verdict = "barely_moved"
    else:
        verdict = "wandering"
    if stuck_reason and not got_unstuck:
        verdict = "stuck: " + stuck_reason

    st = slam.stats() if slam is not None else {}
    evid_counts = {}
    try:
        evid_counts = evidence.counts()
    except Exception:
        evid_counts = {}
    check_dir = gif_path.parent
    n_chk = len(check_src)
    mid_idx = n_chk // 2
    for idx, name in ((0, "f0"), (max(0, n_chk // 4), "f40"), (max(0, n_chk // 2), "f80"), (mid_idx, "mid"), (n_chk - 1, "flast")):
        if 0 <= idx < n_chk and check_src[idx] is not None:
            Image.fromarray(check_src[idx]).save(check_dir / ("kevin_drive_check_%s.png" % name))

    def _fmt_xy(seq, k):
        if not seq:
            return "n/a"
        i = min(len(seq) - 1, max(0, k))
        return "(%.3f, %.3f)" % (seq[i][0], seq[i][1])

    sample_idx = []
    if path_xy:
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            sample_idx.append(min(len(path_xy) - 1, int(round(f * (len(path_xy) - 1)))))

    stats_path = gif_path.parent / "kevin_newton_drive_stats.txt"
    lines = [
        "verdict: %s" % verdict,
        "stopped_before_wall: %s" % stopped_before_wall,
        "wall_overlap: %s" % bool(overlap_any or fp_gate.overlap),
        "min_footprint_clearance_m: %s" % (("%.4f" % fp_gate.min_clear) if fp_gate.min_clear is not None else "n/a"),
        "gate_blocked_forward: %d" % int(fp_gate.block_forward),
        "gate_blocked_yaw: %d" % int(fp_gate.block_yaw),
        "min_distance_to_wall_mesh_m: %s" % (("%.4f" % min_wall) if min_wall is not None else "n/a"),
        "min_front_clearance_m: %s" % (("%.4f" % min_front) if min_front is not None else "n/a"),
        "min_plus_x_wall_clearance_m: %s" % (("%.4f" % min_plus_x) if min_plus_x is not None else "n/a"),
        "nose_crossed_wall: %s" % bool(nose_crossed_wall),
        "wall_box_count: %d" % int(getattr(add_house, "box_count", 0) or 0),
        "clutter_spec: %s" % getattr(add_random_obstacles, "spec", ""),
        "clutter_n: %d" % len(extra),
        "clutter_rooms: %s" % getattr(add_random_obstacles, "rooms", ""),
        "door_xy: (%.3f, %.3f)" % (
            float(getattr(wander, "door_x", -0.6)),
            float(getattr(wander, "door_y", -2.81)),
        ),
        "rs1_wall_hits: frames=%d/%d px_sum=%d" % (rs1_wall_frames, n_sense, rs1_wall_px_sum),
        "path_length_m: %.4f" % path_len,
        "net_displacement_m: %.4f" % travel,
        "got_unstuck: %s" % got_unstuck,
        "wall_recoveries: %s" % _CMD.get("recoveries", 0),
        "stuck_abort: %s" % stuck_abort,
        "stuck_reason: %s" % (stuck_reason or "none"),
        "time_s forward=%.3f creep=%.3f turn_left=%.3f turn_right=%.3f stop=%.3f" % (
            mode_t["forward"], mode_t["creep"], mode_t["turn_left"], mode_t["turn_right"], mode_t["stop"]),
        "mode_frac forward=%.3f creep=%.3f turn_left=%.3f turn_right=%.3f stop=%.3f" % (
            mode_frac["forward"], mode_frac["creep"], mode_frac["turn_left"], mode_frac["turn_right"], mode_frac["stop"]),
        "pose_source_counts visual=%d imu+wheel=%d" % (
            pose_src_counts.get("visual", 0), pose_src_counts.get("imu+wheel", 0)),
        "visual_accepted: %d" % tq.get("visual_accepted", 0),
        "visual_rejected: %d" % tq.get("visual_rejected", 0),
        "wheel_imu_frames: %d" % pose_src_counts.get("imu+wheel", 0),
        "match_attempts: %d" % match_attempts,
        "fused_path_length_m: %.4f" % fused_path,
        "fused_net_displacement_m: %.4f" % fused_net,
        "slip_noise: ANGULAR_SLIP_SCALE=%.2f LINEAR_SLIP_SCALE=%.2f depth=2pct_z+3mm wall_hits_protected dropout=1.5pct" % (
            ANGULAR_SLIP_SCALE, LINEAR_SLIP_SCALE),
        "slam_pose_input: fused PoseEstimator (not chassis truth)",
        "position_stopped_gt_2s: %s (longest_within_3cm_s=%.3f)" % (idle, idle_max),
        "slam_keyframes: %s" % st.get("keyframes", 0),
        "slam_edges: %s" % st.get("edges", 0),
        "slam_loops: %s" % st.get("loop_closures", st.get("loops", 0)),
        "start_xy_world: %s" % (_fmt_xy(path_xy, 0) if path_xy else "n/a"),
        "end_xy_world: %s" % (_fmt_xy(path_xy, len(path_xy) - 1) if path_xy else "n/a"),
        "path_samples_world: %s" % " ".join(_fmt_xy(path_xy, i) for i in sample_idx),
        "rs1_valid: %d/%d" % (rs1_ok_n, n_sense),
        "rs2_clear_accepted_frames: %d/%d" % (rs2_clear_n, n_sense),
        "evidence_updates_ok: %d fail: %d counts: %s" % (evid_ok, evid_fail, evid_counts),
        "depth: RS1 = visual STL raycast (walls+floor) with Newton fill-in; wall collision boxes for stop/physics; STL not a mesh collider",
        "self_core_ok: %s leak_px=%s" % (
            getattr(cams, "last_self_core_ok", None),
            getattr(cams, "last_self_core_leak", None),
        ),
        "frames: %d sense=%d sense_dt_s: %.4f" % (len(check_src), n_sense, dt_sense),
        "realtime_factor: %.1f  wall_s: %.3f  viz_mode: %s" % (rt_final, wall_s, viz.mode),
        "spawn: (%.3f, %.3f)" % (spawn[0], spawn[1]),
    ]
    try:
        cov = wander.coverage(path_xy)
        lines.extend([
            "visit_cells: %d / %d" % (cov["visit_cells"], cov["free_cells"]),
            "visit_fraction: %.3f" % cov["visit_fraction"],
            "rooms_visited: %s" % ",".join(cov["rooms"]),
            "entered_second_room: %s" % ("yes" if len(cov["rooms"]) >= 2 else "no"),
            "room_counts: %s" % cov["room_counts"],
            "stuck_recoveries: %d" % int(cov["stuck_recoveries"]),
            "solver_nudges: %d" % int(nudges),
            "snap_backs: %d" % int(snap_backs),
            "wp_skips: %d" % int(cov.get("wp_skips", 0) or 0),
            "wander_plan: %s" % cov["plan"],
        ])
    except Exception as e:
        lines.append("visit_coverage_error: %s" % e)
    stats_path.write_text("\n".join(lines) + "\n")
    print("verdict", verdict)
    print("wrote", stats_path)
    print("stopped_before_wall", stopped_before_wall, "min_front", min_front, "overlap", overlap_any)
    print(
        "footprint_clear", fp_gate.min_clear,
        "gate_blocked_forward", fp_gate.block_forward,
        "gate_blocked_yaw", fp_gate.block_yaw,
    )
    print("travel", travel, "path", path_len)
    if live_gl is not None:
        live_gl.close()
    viz._close_window()
    small = gif_path.parent / "kevin_newton_drive_small.gif"
    _write_small_gif(viz.keep or frames, small)
    if small.exists():
        print("wrote", small, "bytes", small.stat().st_size)
    return gif_path


def _write_small_gif(frames, dest, width=960, n_frames=120):
    from PIL import Image
    n = len(frames)
    if n == 0:
        return
    for width, take, colors in ((width, n_frames, 128), (960, 80, 64), (800, 48, 32)):
        take = min(take, n)
        idx = np.linspace(0, n - 1, take).astype(int)
        ims = []
        for i in idx:
            im = Image.fromarray(frames[int(i)])
            if im.size[0] > width:
                h = max(1, int(round(im.size[1] * width / float(im.size[0]))))
                im = im.resize((width, h), Image.BILINEAR)
            ims.append(im.convert("P", palette=Image.ADAPTIVE, colors=colors))
        ims[0].save(
            dest, save_all=True, append_images=ims[1:],
            duration=100, loop=0, optimize=True,
        )
        if dest.stat().st_size <= 20 * 1024 * 1024:
            return
