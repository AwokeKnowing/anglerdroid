"""pose.py – Kalman-filtered 2D pose estimation.

Fuses wheel odometry (prediction) with visual odometry (measurement)
using a per-frame incremental Kalman filter on (dtheta, ds).

World frame: x=right, y=up, theta=0 facing +x, CCW positive.
Maintains a circular buffer of 900 world-space positions for trajectory drawing.
"""

import math
import numpy as np

HISTORY_SIZE = 900

# Wheel odometry noise — scales with magnitude plus a floor.
# These are *variances* of the delta per frame.
Q_YAW_SCALE = 0.05      # 5% of |dtheta| as 1-sigma
Q_FWD_SCALE = 0.05      # 5% of |ds|
Q_YAW_FLOOR = 0.001     # rad  minimum 1-sigma
Q_FWD_FLOOR = 0.0005    # m    minimum 1-sigma

# Visual odometry measurement noise (fixed 1-sigma)
R_YAW = 0.003   # rad
R_FWD = 0.002   # m

# Mahalanobis gate — chi-squared, 2 DOF, 95%
GATE_CHI2 = 5.991


class PoseEstimator:
    """Fused 2D pose with world-space history."""

    def __init__(self, wheelbase_m: float, wheel_radius_m: float):
        self._wb = wheelbase_m
        self._wr = wheel_radius_m

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self._hx = np.zeros(HISTORY_SIZE, dtype=np.float64)
        self._hy = np.zeros(HISTORY_SIZE, dtype=np.float64)
        self._hlen = 0
        self._hidx = 0

    def reset(self):
        self.x = self.y = self.theta = 0.0
        self._hlen = self._hidx = 0

    # ── main entry point (call once per vision frame) ──────────────

    def update(self, v_left_mps: float, v_right_mps: float,
               dt: float,
               vis_yaw: float, vis_fwd: float):
        """Fuse wheel + visual odom, integrate pose, record history.

        Args:
            v_left_mps, v_right_mps: wheel linear velocities (m/s), positive=forward.
            dt: seconds since last frame.
            vis_yaw: visual odom yaw (rad, positive=CCW).
            vis_fwd: visual odom forward (m, positive=forward).

        Returns:
            (fused_yaw, fused_fwd) — the fused incremental motion this frame,
            suitable for warping the ego costmap.
        """
        if dt <= 0:
            return 0.0, 0.0

        # ── 1. Wheel odometry prediction ──
        v = (v_left_mps + v_right_mps) * 0.5
        omega = (v_right_mps - v_left_mps) / self._wb
        dtheta_w = omega * dt
        ds_w = v * dt
        have_wheel = abs(v_left_mps) > 1e-6 or abs(v_right_mps) > 1e-6

        # ── 2. Visual odometry measurement ──
        have_vis = abs(vis_yaw) > 1e-8 or abs(vis_fwd) > 1e-8

        # ── 3. Kalman fusion ──
        if have_wheel and have_vis:
            dtheta, ds = self._fuse(dtheta_w, ds_w, vis_yaw, vis_fwd)
        elif have_wheel:
            dtheta, ds = dtheta_w, ds_w
        elif have_vis:
            dtheta, ds = vis_yaw, vis_fwd
        else:
            return 0.0, 0.0

        # ── 4. Integrate into global pose ──
        self.theta += dtheta
        # Normalise theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        self.x += ds * math.cos(self.theta)
        self.y += ds * math.sin(self.theta)

        # ── 5. Record to history ──
        self._hx[self._hidx] = self.x
        self._hy[self._hidx] = self.y
        self._hidx = (self._hidx + 1) % HISTORY_SIZE
        if self._hlen < HISTORY_SIZE:
            self._hlen += 1

        return dtheta, ds

    # ── Kalman fusion of incremental deltas ────────────────────────

    @staticmethod
    def _fuse(dtheta_w, ds_w, dtheta_v, ds_v):
        """Per-frame Kalman: wheel odom = prior, visual odom = measurement."""
        # Process (wheel) covariance — velocity-dependent + floor
        q0 = (Q_YAW_SCALE * abs(dtheta_w) + Q_YAW_FLOOR) ** 2
        q1 = (Q_FWD_SCALE * abs(ds_w) + Q_FWD_FLOOR) ** 2

        # Measurement (visual) covariance
        r0 = R_YAW ** 2
        r1 = R_FWD ** 2

        # Innovation
        y0 = dtheta_v - dtheta_w
        y1 = ds_v - ds_w

        # Innovation covariance (diagonal — sensors are independent)
        s0 = q0 + r0
        s1 = q1 + r1

        # Mahalanobis distance (chi-squared with 2 DOF)
        maha = (y0 * y0) / s0 + (y1 * y1) / s1
        if maha > GATE_CHI2:
            return dtheta_w, ds_w

        # Kalman gain (diagonal)
        k0 = q0 / s0
        k1 = q1 / s1

        return dtheta_w + k0 * y0, ds_w + k1 * y1

    # ── Trajectory output ──────────────────────────────────────────

    def get_world_history(self):
        """Return (N, 2) float64 array of world [x, y] oldest→newest."""
        if self._hlen == 0:
            return np.empty((0, 2), dtype=np.float64)
        if self._hlen < HISTORY_SIZE:
            return np.column_stack([
                self._hx[:self._hlen], self._hy[:self._hlen]
            ])
        idx = (np.arange(HISTORY_SIZE) + self._hidx) % HISTORY_SIZE
        return np.column_stack([self._hx[idx], self._hy[idx]])

    def world_to_ego_pixels(self, px_size: float, cx: float, cy: float):
        """Project world history into ego costmap pixel coordinates.

        Robot sits at pixel (cx, cy) facing RIGHT on the costmap.
        px_size: metres per pixel.

        Returns (N, 2) int32 array of [col, row].
        """
        pts = self.get_world_history()
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.int32)

        dx = pts[:, 0] - self.x
        dy = pts[:, 1] - self.y

        ct = math.cos(self.theta)
        st = math.sin(self.theta)

        # Rotate world-delta into robot frame (forward, left)
        local_fwd  =  dx * ct + dy * st
        local_left = -dx * st + dy * ct

        col = cx + local_fwd / px_size
        row = cy - local_left / px_size

        return np.column_stack([col, row]).astype(np.int32)
