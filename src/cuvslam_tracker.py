"""cuvslam_tracker.py – cuVSLAM-based visual SLAM for AnglerDroid.

Uses the RS2 D435 forward camera's stereo IR pair (848x480) for
GPU-accelerated visual odometry + SLAM via NVIDIA cuVSLAM.

Coordinate transform (cuVSLAM rig → our world):
  cuVSLAM rig frame : X-right, Y-down, Z-forward  (robot body on floor)
  Our world frame   : theta=0 facing +x, CCW positive (pose.py convention)

  Mapping: our_x = rig_z, our_y = -rig_x, our_theta = -yaw_around_Y

Requires: Python 3.10+, CUDA 12+, pycuvslam wheel for your platform.
Install:  pip install -e bin/aarch64   (Jetson)
          pip install -e bin/x86_64    (desktop)
from https://github.com/NVlabs/pycuvslam
"""

import math
import time
import numpy as np

try:
    import cuvslam
    HAS_CUVSLAM = True
except ImportError:
    HAS_CUVSLAM = False

CAMERA_HEIGHT_M = 0.97
CAMERA_SETBACK_M = 0.13
CAMERA_PITCH_DEG = 64.4

DEFAULT_IR_SIZE = (848, 480)
DEFAULT_IR_FOCAL = (424.0, 424.0)
DEFAULT_IR_PP = (424.0, 240.0)
DEFAULT_BASELINE = 0.05

HISTORY_SIZE = 300


class CuVSLAMTracker:
    """cuVSLAM stereo visual SLAM — drop-in pose source for the vision pipeline.

    Has the same public interface as PoseEstimator (.x, .y, .theta,
    .get_world_history(), .reset()) so the rest of the pipeline works unchanged.
    """

    def __init__(self, rs2_profile=None,
                 camera_height=CAMERA_HEIGHT_M,
                 camera_setback=CAMERA_SETBACK_M,
                 camera_pitch_deg=CAMERA_PITCH_DEG):
        if not HAS_CUVSLAM:
            raise ImportError(
                "cuvslam not installed. Requires Python 3.10+, CUDA 12+. "
                "See https://github.com/NVlabs/pycuvslam")

        self._cam_h = camera_height
        self._cam_setback = camera_setback
        self._cam_pitch_deg = camera_pitch_deg

        t0 = time.monotonic()
        cuvslam.warm_up_gpu()
        cuvslam.set_verbosity(1)
        print("cuvslam: GPU warm-up %.0fms" % ((time.monotonic() - t0) * 1000))

        if rs2_profile is not None:
            self._setup_from_profile(rs2_profile)
        else:
            self._setup_defaults()

        odom_cfg = cuvslam.Tracker.OdometryConfig(
            horizontal_stereo_camera=True,
            async_sba=True,
            enable_observations_export=False,
            enable_landmarks_export=False,
            enable_final_landmarks_export=False,
        )
        slam_cfg = cuvslam.Tracker.SlamConfig(
            sync_mode=False,
            max_map_size=500,
        )

        self._tracker = cuvslam.Tracker(self._rig, odom_cfg, slam_cfg)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._prev_theta = 0.0

        self._hx = np.zeros(HISTORY_SIZE, dtype=np.float64)
        self._hy = np.zeros(HISTORY_SIZE, dtype=np.float64)
        self._hlen = 0
        self._hidx = 0
        self._tracking = False
        self._frame_count = 0
        self._lost_count = 0

        v_str, v_maj, v_min = cuvslam.get_version()
        print("cuvslam: ready (v%s)" % v_str)

    # ── rig setup ───────────────────────────────────────────────────

    def _setup_from_profile(self, profile):
        """Extract stereo intrinsics and baseline from RS2 pipeline profile."""
        import pyrealsense2 as rs

        ir1_stream = profile.get_stream(rs.stream.infrared, 1)
        ir2_stream = profile.get_stream(rs.stream.infrared, 2)
        ir1_intr = ir1_stream.as_video_stream_profile().get_intrinsics()
        ir2_intr = ir2_stream.as_video_stream_profile().get_intrinsics()

        extr = ir2_stream.get_extrinsics_to(ir1_stream)
        baseline = abs(extr.translation[0])

        self._build_rig(
            ir1_intr.width, ir1_intr.height,
            ir1_intr.fx, ir1_intr.fy, ir1_intr.ppx, ir1_intr.ppy,
            ir2_intr.fx, ir2_intr.fy, ir2_intr.ppx, ir2_intr.ppy,
            baseline)

        print("cuvslam: RS2 intrinsics %dx%d  "
              "left(fx=%.1f fy=%.1f ppx=%.1f ppy=%.1f)  baseline=%.4fm" % (
                  ir1_intr.width, ir1_intr.height,
                  ir1_intr.fx, ir1_intr.fy, ir1_intr.ppx, ir1_intr.ppy,
                  baseline))

    def _setup_defaults(self):
        """Approximate D435 intrinsics when SDK profile unavailable."""
        w, h = DEFAULT_IR_SIZE
        fx, fy = DEFAULT_IR_FOCAL
        ppx, ppy = DEFAULT_IR_PP
        self._build_rig(w, h, fx, fy, ppx, ppy, fx, fy, ppx, ppy,
                        DEFAULT_BASELINE)
        print("cuvslam: using default D435 intrinsics (no RS2 profile)")

    def _build_rig(self, w, h,
                   lfx, lfy, lppx, lppy,
                   rfx, rfy, rppx, rppy,
                   baseline):
        pitch_rad = math.radians(self._cam_pitch_deg)
        qx = math.sin(pitch_rad / 2)
        qw = math.cos(pitch_rad / 2)

        left_cam = cuvslam.Camera(
            size=[w, h],
            principal=[lppx, lppy],
            focal=[lfx, lfy],
        )
        left_cam.rig_from_camera = cuvslam.Pose(
            rotation=[qx, 0.0, 0.0, qw],
            translation=[0.0, -self._cam_h, -self._cam_setback],
        )

        right_cam = cuvslam.Camera(
            size=[w, h],
            principal=[rppx, rppy],
            focal=[rfx, rfy],
        )
        right_cam.rig_from_camera = cuvslam.Pose(
            rotation=[qx, 0.0, 0.0, qw],
            translation=[baseline, -self._cam_h, -self._cam_setback],
        )

        self._rig = cuvslam.Rig([left_cam, right_cam])

    # ── tracking ────────────────────────────────────────────────────

    def track(self, ir_left, ir_right, timestamp_ns):
        """Process a stereo IR frame pair and update pose.

        Args:
            ir_left:  (H, W) uint8 left IR image
            ir_right: (H, W) uint8 right IR image
            timestamp_ns: monotonic timestamp in nanoseconds

        Returns:
            (delta_yaw, delta_fwd) if tracking succeeded, None if lost.
        """
        self._frame_count += 1
        self._prev_x = self.x
        self._prev_y = self.y
        self._prev_theta = self.theta

        pose_est, slam_pose = self._tracker.track(
            timestamp_ns, [ir_left, ir_right])

        if pose_est.world_from_rig is None:
            self._tracking = False
            self._lost_count += 1
            if self._lost_count <= 5 or self._lost_count % 30 == 0:
                print("cuvslam: tracking lost (frame %d, lost %d)" % (
                    self._frame_count, self._lost_count))
            return None

        self._tracking = True
        self._lost_count = 0
        pose = pose_est.world_from_rig.pose

        tx, ty, tz = float(pose.translation[0]), float(pose.translation[1]), float(pose.translation[2])
        self.x = tz
        self.y = -tx

        qx, qy, qz, qw = (float(pose.rotation[0]), float(pose.rotation[1]),
                           float(pose.rotation[2]), float(pose.rotation[3]))
        r02 = 2.0 * (qx * qz + qw * qy)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        self.theta = -math.atan2(r02, r22)

        delta_theta = self.theta - self._prev_theta
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))
        dx = self.x - self._prev_x
        dy = self.y - self._prev_y
        ct = math.cos(self._prev_theta)
        st = math.sin(self._prev_theta)
        delta_fwd = dx * ct + dy * st

        self._hx[self._hidx] = self.x
        self._hy[self._hidx] = self.y
        self._hidx = (self._hidx + 1) % HISTORY_SIZE
        if self._hlen < HISTORY_SIZE:
            self._hlen += 1

        if self._frame_count <= 3 or self._frame_count % 90 == 0:
            print("cuvslam: frame %d  pose=(%.3f, %.3f, %.1f°)  "
                  "d_yaw=%.3f° d_fwd=%.3fcm" % (
                      self._frame_count,
                      self.x, self.y, math.degrees(self.theta),
                      math.degrees(delta_theta), delta_fwd * 100))

        return delta_theta, delta_fwd

    # ── public interface (matches PoseEstimator) ────────────────────

    @property
    def is_tracking(self):
        return self._tracking

    def get_world_history(self):
        """(N, 2) float64 array of world [x, y], oldest → newest."""
        if self._hlen == 0:
            return np.empty((0, 2), dtype=np.float64)
        if self._hlen < HISTORY_SIZE:
            return np.column_stack([
                self._hx[:self._hlen], self._hy[:self._hlen]])
        idx = (np.arange(HISTORY_SIZE) + self._hidx) % HISTORY_SIZE
        return np.column_stack([self._hx[idx], self._hy[idx]])

    def reset(self):
        self.x = self.y = self.theta = 0.0
        self._prev_x = self._prev_y = self._prev_theta = 0.0
        self._hlen = self._hidx = 0

    def get_slam_metrics(self):
        """Return cuVSLAM metrics (for diagnostics/UI)."""
        try:
            metrics = self._tracker.get_slam_metrics()
        except Exception:
            metrics = None
        lc_poses = None
        try:
            lc_poses = self._tracker.get_loop_closure_poses()
        except Exception:
            pass
        return {
            'tracking': self._tracking,
            'frame_count': self._frame_count,
            'lost_count': self._lost_count,
            'lc_count': len(lc_poses) if lc_poses else 0,
            'lc_status': getattr(metrics, 'lc_status', None) if metrics else None,
        }
