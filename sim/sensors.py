"""High-fidelity sensor emulation for host simulator (i777).

Simulates Kevin's perception pipeline:
- Dual RealSense depth (RS1 topdown, RS2 forward) with dropout + noise
- RGB webcam (synthetic or playback)
- Wheel odometry with slip + drift
- Visual odometry correction (simplified Kalman fusion)
"""

import math
import numpy as np
from collections import deque

# Import robot config for ego-space geometry
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from robot_config import (
    FRAME_W, FRAME_H, RCX, RCY, EGO_PX_SIZE,
    FOOT_X0, FOOT_Y0, FOOT_X1, FOOT_Y1,
    MAST_CLEAR_CM
)

# Fidelity levels
FIDELITY_MODES = {
    'none': {
        'depth_dropout_rate': 0.0,
        'depth_noise_sigma_base': 0.0,
        'pose_drift_sigma': 0.0,
        'latency_ms': 0.0,
        'sensor_jitter_ms': 0.0,
    },
    'low': {
        'depth_dropout_rate': 0.0,
        'depth_noise_sigma_base': 0.0,
        'pose_drift_sigma': 0.0,
        'latency_ms': 0.0,
        'sensor_jitter_ms': 0.0,
    },
    'medium': {
        'depth_dropout_rate': 0.008,  # 0.8% dropout
        'depth_noise_sigma_base': 0.003,  # 3mm @ 1m
        'pose_drift_sigma': 0.002,  # 2mm per step
        'latency_ms': 50.0,
        'sensor_jitter_ms': 2.0,
    },
    'high': {
        'depth_dropout_rate': 0.015,  # 1.5% dropout
        'depth_noise_sigma_base': 0.008,  # 8mm @ 1m
        'pose_drift_sigma': 0.004,  # 4mm per step
        'latency_ms': 150.0,
        'sensor_jitter_ms': 5.0,
    },
}


class RealsenseDepthSensor:
    """Simulated RealSense D435i depth camera with noise model.
    
    Emulates measured dropout + Gaussian noise from real RS cameras.
    """
    
    def __init__(self, serial_id, fidelity='medium', rng=None):
        """Initialize depth sensor.
        
        Args:
            serial_id: Camera serial (e.g., '815412070676' for RS1)
            fidelity: 'none', 'low', 'medium', 'high'
            rng: Optional numpy random generator
        """
        self.serial_id = serial_id
        self.fidelity_params = FIDELITY_MODES.get(fidelity, FIDELITY_MODES['medium'])
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def capture(self, world_obs, world_height, robot_pose):
        """Capture depth from world given robot pose.
        
        Args:
            world_obs: (H, W) uint8 obstacle map (world frame)
            world_height: (H, W) uint8 height map (world frame)
            robot_pose: dict with 'x', 'y', 'theta'
            
        Returns:
            (obs_ego, height_ego): (240, 320) uint8 ego-space maps
        """
        # Transform world to ego (same as sim/robot.py)
        obs_ego, height_ego = self._world_to_ego(
            world_obs, world_height, robot_pose
        )
        
        # Apply fidelity noise
        if self.fidelity_params['depth_dropout_rate'] > 0:
            obs_ego, height_ego = self._apply_dropout(obs_ego, height_ego)
        
        if self.fidelity_params['depth_noise_sigma_base'] > 0:
            obs_ego, height_ego = self._apply_noise(obs_ego, height_ego)
        
        return obs_ego, height_ego
    
    def _world_to_ego(self, world_obs, world_height, pose):
        """Vectorized world→ego transform (matches sim/robot.py)."""
        h, w = world_obs.shape
        cos_t = math.cos(pose['theta'])
        sin_t = math.sin(pose['theta'])
        
        # Precomputed ego grid
        ey, ex = np.mgrid[0:FRAME_H, 0:FRAME_W]
        dx_ego = (ex - RCX).astype(np.float64) * EGO_PX_SIZE
        dy_ego = (ey - RCY).astype(np.float64) * EGO_PX_SIZE
        
        # Rotate to world frame
        dx_world = dx_ego * cos_t - dy_ego * sin_t
        dy_world = dx_ego * sin_t + dy_ego * cos_t
        
        # World pixel coords
        wx_px = np.rint((pose['x'] + dx_world) / EGO_PX_SIZE).astype(np.int32)
        wy_px = np.rint((pose['y'] + dy_world) / EGO_PX_SIZE).astype(np.int32)
        valid = (wx_px >= 0) & (wx_px < w) & (wy_px >= 0) & (wy_px < h)
        
        # Sample world maps
        ego_obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        ego_height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        ego_obs[valid] = world_obs[wy_px[valid], wx_px[valid]]
        ego_height[valid] = world_height[wy_px[valid], wx_px[valid]]
        
        return ego_obs, ego_height
    
    def _apply_dropout(self, obs, height):
        """Random pixel dropout (IR glare, low texture areas)."""
        rate = self.fidelity_params['depth_dropout_rate']
        mask = self.rng.random(obs.shape) < rate
        
        # Don't dropout pixels inside robot footprint (always force clear)
        mask[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = False
        
        obs = obs.copy()
        height = height.copy()
        obs[mask] = 0  # Unknown
        height[mask] = 0
        
        return obs, height
    
    def _apply_noise(self, obs, height):
        """Distance-dependent Gaussian noise (σ grows with range)."""
        sigma_base = self.fidelity_params['depth_noise_sigma_base']
        
        # Compute distance from robot center
        ey, ex = np.mgrid[0:FRAME_H, 0:FRAME_W]
        dx = (ex - RCX) * EGO_PX_SIZE
        dy = (ey - RCY) * EGO_PX_SIZE
        dist = np.sqrt(dx**2 + dy**2)
        
        # σ grows linearly with distance (σ=3mm @ 1m → σ=15mm @ 5m)
        sigma_map = sigma_base * (1.0 + 4.0 * dist)
        
        # Add Gaussian noise to height
        height = height.astype(np.float32)
        noise = self.rng.normal(0, 1, height.shape) * sigma_map * 100  # cm
        height = np.clip(height + noise, 0, 255).astype(np.uint8)
        
        # Obs map: flip clear↔occupied sparsely (mimics noise boundary)
        flip_rate = sigma_base * 2.0  # ~0.6% for medium fidelity
        flip_mask = self.rng.random(obs.shape) < flip_rate
        flip_mask[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = False  # Never inside footprint
        
        obs = obs.copy()
        obs[flip_mask] = np.where(obs[flip_mask] >= 100, 0, 255).astype(np.uint8)
        
        return obs, height


class WheelOdometrySensor:
    """Simulated ODrive wheel encoders with slip + drift.
    
    Matches measured carpet slip (angular scale 0.92) and drift accumulation.
    """
    
    def __init__(self, fidelity='medium', rng=None):
        self.fidelity_params = FIDELITY_MODES.get(fidelity, FIDELITY_MODES['medium'])
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Calibrated slip (from src/pose.py)
        self.angular_slip_scale = 0.92  # Under-reports rotation on carpet
        
        # Accumulated drift
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.drift_theta = 0.0
    
    def measure(self, true_v, true_w, dt):
        """Measure velocity with slip + noise.
        
        Args:
            true_v: True linear velocity (m/s)
            true_w: True angular velocity (rad/s)
            dt: Time step (s)
            
        Returns:
            (v_meas, w_meas): Measured velocities with noise
        """
        sigma = self.fidelity_params['pose_drift_sigma']
        
        # Angular slip
        w_meas = true_w * self.angular_slip_scale
        
        # Add noise
        v_noise = self.rng.normal(0, sigma / dt)  # Convert σ to velocity
        w_noise = self.rng.normal(0, sigma / dt * 2.0)  # Angular noise higher
        
        v_meas = true_v + v_noise
        w_meas = w_meas + w_noise
        
        # Accumulate drift (random walk)
        self.drift_x += self.rng.normal(0, sigma)
        self.drift_y += self.rng.normal(0, sigma)
        self.drift_theta += self.rng.normal(0, sigma * 0.5)  # radians
        
        return v_meas, w_meas
    
    def get_drift_offset(self):
        """Return accumulated drift (x, y, theta)."""
        return self.drift_x, self.drift_y, self.drift_theta


class VisualOdometrySensor:
    """Simplified visual odometry correction (mimics Kalman fusion).
    
    Real VO uses dense optical flow + outlier rejection; this is a statistical stub.
    """
    
    def __init__(self, fidelity='medium', rng=None):
        self.fidelity_params = FIDELITY_MODES.get(fidelity, FIDELITY_MODES['medium'])
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def correct_pose(self, wheel_pose, true_pose):
        """Apply VO correction to wheel odometry.
        
        Args:
            wheel_pose: dict with 'x', 'y', 'theta' from wheel odom
            true_pose: dict with true pose (for computing correction)
            
        Returns:
            fused_pose: dict with Kalman-fused pose
        """
        # Fusion weights: 85% wheel, 15% VO correction (from src/pose.py)
        wheel_weight = 0.85
        vo_weight = 0.15
        
        # Compute VO correction (with noise)
        sigma = self.fidelity_params['pose_drift_sigma'] * 3.0  # VO noisier than wheel
        
        vo_x = true_pose['x'] + self.rng.normal(0, sigma)
        vo_y = true_pose['y'] + self.rng.normal(0, sigma)
        vo_theta = true_pose['theta'] + self.rng.normal(0, sigma * 0.5)
        
        # Fused pose
        fused_x = wheel_weight * wheel_pose['x'] + vo_weight * vo_x
        fused_y = wheel_weight * wheel_pose['y'] + vo_weight * vo_y
        
        # Angle fusion (handle wraparound)
        dtheta = (vo_theta - wheel_pose['theta'] + math.pi) % (2 * math.pi) - math.pi
        fused_theta = wheel_pose['theta'] + vo_weight * dtheta
        fused_theta = (fused_theta + math.pi) % (2 * math.pi) - math.pi
        
        return {
            'x': float(fused_x),
            'y': float(fused_y),
            'theta': float(fused_theta),
        }


class RgbWebcamSensor:
    """Simulated USB webcam (640×480, 30fps).
    
    For v0: Returns blank/synthetic frames. house_bot face/keepout triggers stubbed.
    """
    
    def __init__(self, resolution=(640, 480), fidelity='medium', rng=None):
        self.resolution = resolution
        self.fidelity_params = FIDELITY_MODES.get(fidelity, FIDELITY_MODES['medium'])
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def capture(self, robot_pose, world):
        """Capture RGB frame (synthetic stub for v0).
        
        Returns:
            (H, W, 3) uint8 RGB image (currently blank)
        """
        # Stub: return blank frame
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Optional: Add synthetic noise (compression artifacts, brightness flicker)
        if self.fidelity_params['sensor_jitter_ms'] > 0:
            brightness_delta = self.rng.integers(-10, 11)
            frame = np.clip(frame.astype(np.int16) + brightness_delta, 0, 255).astype(np.uint8)
        
        return frame


class SensorSuite:
    """Combined sensor package for host simulator."""
    
    def __init__(self, fidelity='medium', seed=None):
        rng = np.random.default_rng(seed)
        
        self.fidelity = fidelity
        self.rs1_topdown = RealsenseDepthSensor('815412070676', fidelity, rng)
        self.rs2_forward = RealsenseDepthSensor('944622074292', fidelity, rng)
        self.wheel_odom = WheelOdometrySensor(fidelity, rng)
        self.visual_odom = VisualOdometrySensor(fidelity, rng)
        self.rgb_webcam = RgbWebcamSensor(fidelity=fidelity, rng=rng)
        
    def capture_depth(self, robot_pose, world_obs, world_height):
        """Capture combined depth map (matches live vision.py pipeline).
        
        Returns:
            (obs_combined, height_combined): (240, 320) uint8 ego-space
        """
        # RS1 topdown + RS2 forward (simplified: just use one for now)
        # Real pipeline combines with alignment offsets; for v0, use single sensor
        obs_ego, height_ego = self.rs1_topdown.capture(
            world_obs, world_height, robot_pose
        )
        
        return obs_ego, height_ego
    
    def capture_rgb(self, robot_pose, world):
        """Capture RGB frame (stub for v0)."""
        return self.rgb_webcam.capture(robot_pose, world)
    
    def measure_velocity(self, true_v, true_w, dt):
        """Measure velocity from wheel encoders with noise."""
        return self.wheel_odom.measure(true_v, true_w, dt)
    
    def fuse_pose(self, wheel_pose, true_pose):
        """Fuse wheel + visual odometry."""
        return self.visual_odom.correct_pose(wheel_pose, true_pose)


class SimClock:
    """Wall-clock vs sim time for multi-rate testing."""
    
    def __init__(self, hz=30.0, real_time_factor=1.0):
        """Initialize clock.
        
        Args:
            hz: Simulation frequency (Hz)
            real_time_factor: 1.0 = real-time, >1.0 = faster, <1.0 = slower
        """
        self.hz = hz
        self.dt = 1.0 / hz
        self.real_time_factor = real_time_factor
        self.sim_time = 0.0
        self.wall_time_start = None
        
    def reset(self):
        """Reset clock to t=0."""
        self.sim_time = 0.0
        self.wall_time_start = None
        
    def tick(self):
        """Advance simulation time by one step."""
        self.sim_time += self.dt
        
    def sleep_until_next_frame(self):
        """Sleep to maintain real-time factor (no-op for >1x speed)."""
        if self.real_time_factor >= 100.0:
            # Fast mode: no sleep
            return
        
        import time
        if self.wall_time_start is None:
            self.wall_time_start = time.monotonic()
            return
        
        expected_wall_time = self.sim_time / self.real_time_factor
        elapsed_wall_time = time.monotonic() - self.wall_time_start
        sleep_time = expected_wall_time - elapsed_wall_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
