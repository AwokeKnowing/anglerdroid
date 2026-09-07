"""wheel_imu_prior.py – Wheel + IMU prediction prior for self-SLAM.

Provides a robust pose prediction when visual odometry is lost or unreliable.
This is a HYPOTHESIS experiment (self-slam-wheel-imu-prior-v0) to test if
fusing wheel + IMU outside of vision can improve SLAM lock reliability.

Design philosophy:
  - Pure prediction: does NOT depend on visual odometry
  - Simple EKF: track pose (x, y, theta) with covariance
  - Wheel velocities + IMU yaw rate as inputs
  - Boot from ~/.kevin/latest_pose.json when present
  - Save pose periodically for crash recovery

Usage:
    prior = WheelIMUPrior(wheelbase_m=0.30, wheel_radius_m=0.0762)
    prior.load_latest_pose()  # Boot from saved pose if present
    
    # In high-rate loop (~100 Hz):
    prior.predict(v_left, v_right, imu_yaw_rate, dt)
    x, y, theta = prior.x, prior.y, prior.theta
    
    # Optionally apply visual correction when healthy:
    if visual_healthy:
        prior.correct_visual(vis_yaw, vis_fwd, vis_confidence)
    
    # Save periodically:
    prior.save_latest_pose()  # Every ~10s or on shutdown
"""

import os
import json
import math
import time
import numpy as np


# ── Slip compensation (same as pose.py) ─────────────────────────────
ANGULAR_SLIP_SCALE = 0.92  # Tracked vehicles on carpet: reduce reported rotation
LINEAR_SLIP_SCALE = 1.0

# ── Process noise (wheel + IMU prediction) ──────────────────────────
# Q scales with velocity magnitude to account for slip uncertainty
Q_X_SCALE = 0.10        # 10% of |ds| in x direction
Q_Y_SCALE = 0.05        # 5% of |ds| in y direction (less lateral slip)
Q_THETA_SCALE = 0.15    # 15% of |dtheta| (higher for carpet slip)
Q_X_FLOOR = 0.001       # m
Q_Y_FLOOR = 0.0005      # m
Q_THETA_FLOOR = 0.003   # rad

# IMU yaw rate noise (when IMU available)
Q_IMU_YAW_RATE = 0.01   # rad/s (1-sigma)

# ── Visual correction noise ─────────────────────────────────────────
# R is scaled by 1/confidence, so base values assume confidence=1.0
R_YAW_BASE = 0.002      # rad (trust visual yaw more than wheels)
R_FWD_BASE = 0.002      # m

# Mahalanobis gate for visual correction (chi-squared, 2 DOF, 95%)
GATE_CHI2 = 5.991

# ── Pose persistence ────────────────────────────────────────────────
LATEST_POSE_PATH = os.path.expanduser("~/.kevin/latest_pose.json")
SAVE_INTERVAL_S = 10.0  # Save pose every 10 seconds


class WheelIMUPrior:
    """Wheel + IMU prediction prior with optional visual correction.
    
    Tracks 2D pose (x, y, theta) and covariance (3×3 matrix).
    Prediction step integrates wheel velocities + IMU yaw rate.
    Optional visual correction when feature tracking is healthy.
    """
    
    def __init__(self, wheelbase_m: float, wheel_radius_m: float):
        self.wb = wheelbase_m
        self.wr = wheel_radius_m
        
        # State: [x, y, theta]
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Covariance (3×3): uncertainty in [x, y, theta]
        # Start with moderate uncertainty (haven't moved yet)
        self.P = np.diag([0.01, 0.01, 0.01])  # m², m², rad²
        
        # Metrics
        self._predict_count = 0
        self._correct_count = 0
        self._last_save_time = 0.0
        self._boot_source = None  # 'json' or None
    
    # ── Prediction step ─────────────────────────────────────────────
    
    def predict(self, v_left_mps: float, v_right_mps: float, 
                imu_yaw_rate: float, dt: float):
        """Predict pose from wheel velocities + IMU yaw rate.
        
        Args:
            v_left_mps: Left wheel velocity (m/s)
            v_right_mps: Right wheel velocity (m/s)
            imu_yaw_rate: IMU gyro yaw rate (rad/s, body frame Z-axis)
            dt: Time delta (seconds)
        
        Returns:
            (dtheta, ds): Pose delta for this step
        """
        if dt <= 0:
            return 0.0, 0.0
        
        # 1. Wheel odometry with slip compensation
        v = (v_left_mps + v_right_mps) * 0.5
        omega_wheel = (v_right_mps - v_left_mps) / self.wb
        dtheta_wheel = omega_wheel * dt * ANGULAR_SLIP_SCALE
        ds = v * dt * LINEAR_SLIP_SCALE
        
        # 2. Fuse with IMU yaw rate (complementary blend)
        # IMU weight: higher when we have IMU, else fall back to wheel-only
        if abs(imu_yaw_rate) > 1e-6:
            dtheta_imu = imu_yaw_rate * dt
            # 50% IMU, 50% wheel (simple blend; could use Kalman here too)
            imu_weight = 0.5
            dtheta = (1.0 - imu_weight) * dtheta_wheel + imu_weight * dtheta_imu
        else:
            dtheta = dtheta_wheel
        
        # 3. Integrate into global pose
        self.theta += dtheta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        dx_world = ds * math.cos(self.theta)
        dy_world = ds * math.sin(self.theta)
        self.x += dx_world
        self.y += dy_world
        
        # 4. Update covariance (simplified EKF prediction)
        # Process noise Q scales with motion magnitude
        q_x = (Q_X_SCALE * abs(ds) + Q_X_FLOOR) ** 2
        q_y = (Q_Y_SCALE * abs(ds) + Q_Y_FLOOR) ** 2
        q_theta = (Q_THETA_SCALE * abs(dtheta) + Q_THETA_FLOOR) ** 2
        
        # Add IMU noise if used
        if abs(imu_yaw_rate) > 1e-6:
            q_theta += (Q_IMU_YAW_RATE * dt) ** 2
        
        Q = np.diag([q_x, q_y, q_theta])
        
        # Linearized state transition Jacobian (simplified: no rotation coupling)
        # F = I + dt * [0, 0, -ds*sin(theta);
        #               0, 0,  ds*cos(theta);
        #               0, 0,  0]
        # For small dt, approximate F ≈ I
        F = np.eye(3)
        F[0, 2] = -ds * math.sin(self.theta)
        F[1, 2] = ds * math.cos(self.theta)
        
        # P = F P F^T + Q
        self.P = F @ self.P @ F.T + Q
        
        # Bound covariance to prevent unbounded growth (simple clamp)
        self.P[0, 0] = min(self.P[0, 0], 1.0)   # max 1m² x uncertainty
        self.P[1, 1] = min(self.P[1, 1], 1.0)   # max 1m² y uncertainty
        self.P[2, 2] = min(self.P[2, 2], 0.25)  # max 0.5 rad theta uncertainty
        
        self._predict_count += 1
        return dtheta, ds
    
    # ── Visual correction (optional) ────────────────────────────────
    
    def correct_visual(self, vis_yaw: float, vis_fwd: float, 
                       vis_confidence: float = 1.0):
        """Apply visual odometry correction (when healthy).
        
        Args:
            vis_yaw: Visual delta yaw (radians)
            vis_fwd: Visual delta forward (meters)
            vis_confidence: Visual confidence [0-1]
        
        Returns:
            True if correction applied, False if gated (Mahalanobis too large)
        """
        # Measurement: [yaw, fwd] (we observe delta pose from visual)
        # This is a simplification; proper EKF would observe full pose
        z = np.array([vis_yaw, vis_fwd])
        
        # Predicted measurement (we predicted zero relative motion in this step)
        # Since we already integrated prediction, visual measures residual
        h = np.array([0.0, 0.0])
        
        # Innovation
        y = z - h
        
        # Measurement noise (scaled by confidence)
        r_scale = 1.0 / max(vis_confidence, 0.1)
        R = np.diag([
            (R_YAW_BASE * r_scale) ** 2,
            (R_FWD_BASE * r_scale) ** 2,
        ])
        
        # Innovation covariance (only use theta and forward motion)
        # H = [[0, 0, 1], [cos(theta), sin(theta), 0]] for full EKF
        # Simplified: S ≈ R (measurement noise dominates for small corrections)
        H = np.array([
            [0.0, 0.0, 1.0],      # dtheta measurement
            [math.cos(self.theta), math.sin(self.theta), 0.0],  # ds measurement
        ])
        S = H @ self.P @ H.T + R
        
        # Mahalanobis distance (gate outliers)
        try:
            S_inv = np.linalg.inv(S)
            maha = y.T @ S_inv @ y
            if maha > GATE_CHI2:
                # Visual correction too far from prediction → reject
                return False
        except np.linalg.LinAlgError:
            # Singular covariance → reject
            return False
        
        # Kalman gain
        K = self.P @ H.T @ S_inv
        
        # State update
        state = np.array([self.x, self.y, self.theta])
        state = state + K @ y
        self.x, self.y, self.theta = state[0], state[1], state[2]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Covariance update
        self.P = (np.eye(3) - K @ H) @ self.P
        
        self._correct_count += 1
        return True
    
    # ── Pose persistence ────────────────────────────────────────────
    
    def load_latest_pose(self) -> bool:
        """Load pose from ~/.kevin/latest_pose.json if present.
        
        Returns:
            True if loaded, False if file not found or invalid.
        """
        if not os.path.exists(LATEST_POSE_PATH):
            return False
        
        try:
            with open(LATEST_POSE_PATH, 'r') as f:
                data = json.load(f)
            
            self.x = float(data.get('x', 0.0))
            self.y = float(data.get('y', 0.0))
            self.theta = float(data.get('theta', 0.0))
            
            # Load covariance if present
            if 'P' in data:
                P_list = data['P']
                self.P = np.array(P_list, dtype=np.float64).reshape(3, 3)
            
            self._boot_source = 'json'
            print(f"wheel_imu_prior: loaded pose from {LATEST_POSE_PATH}")
            print(f"  x={self.x:.3f}m, y={self.y:.3f}m, theta={math.degrees(self.theta):.1f}°")
            return True
            
        except Exception as e:
            print(f"wheel_imu_prior: failed to load {LATEST_POSE_PATH}: {e}")
            return False
    
    def save_latest_pose(self, force=False):
        """Save pose to ~/.kevin/latest_pose.json.
        
        Args:
            force: If True, save regardless of time since last save.
                   If False, only save if SAVE_INTERVAL_S has elapsed.
        """
        now = time.monotonic()
        if not force and (now - self._last_save_time) < SAVE_INTERVAL_S:
            return
        
        self._last_save_time = now
        
        data = {
            'x': float(self.x),
            'y': float(self.y),
            'theta': float(self.theta),
            'P': self.P.tolist(),
            'timestamp': time.time(),
            'predict_count': self._predict_count,
            'correct_count': self._correct_count,
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(LATEST_POSE_PATH), exist_ok=True)
        
        try:
            with open(LATEST_POSE_PATH, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"wheel_imu_prior: failed to save {LATEST_POSE_PATH}: {e}")
    
    # ── Metrics ─────────────────────────────────────────────────────
    
    def get_metrics(self):
        """Return diagnostic metrics."""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'theta_deg': math.degrees(self.theta),
            'cov_x': self.P[0, 0],
            'cov_y': self.P[1, 1],
            'cov_theta': self.P[2, 2],
            'predict_count': self._predict_count,
            'correct_count': self._correct_count,
            'boot_source': self._boot_source,
        }
    
    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Reset pose to specified values (or origin)."""
        self.x = x
        self.y = y
        self.theta = theta
        self.P = np.diag([0.01, 0.01, 0.01])
        self._predict_count = 0
        self._correct_count = 0
