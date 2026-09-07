"""odom_thread.py – High-rate odometry thread for wheel+IMU integration.

Runs at ~100-200 Hz, much faster than the 30 Hz capture loop. This ensures
fresh odometry even when capture is slow or shedding budget.

Capture loop samples the latest pose snapshot without blocking the odom thread.
Visual odometry corrections can still be applied on the capture thread or fed
back to the odom thread (design TBD).

Thread-safe pose snapshot using double-buffering or simple mutex (GIL is
sufficient for read-only access to float fields in CPython).
"""

import time
import threading
import numpy as np


class OdomThread:
    """High-rate odometry integrator thread (~100-200 Hz).
    
    Integrates wheel velocities + IMU yaw rate into pose at high frequency,
    independent of the 30 Hz capture loop. Capture samples the latest pose
    without blocking odometry.
    
    Args:
        pose_estimator: PoseEstimator instance (shared with capture for now)
        wheelbase: Wheelbase instance for reading velocities
        imu_pipeline: IMUPipeline instance for yaw rate (optional)
        target_hz: Target loop rate (default 100 Hz)
    """
    
    def __init__(self, pose_estimator, wheelbase=None, imu_pipeline=None, 
                 target_hz=100.0):
        self._pose = pose_estimator
        self._wheelbase = wheelbase
        self._imu = imu_pipeline
        self._target_hz = target_hz
        self._running = False
        self._thread = None
        self._last_update_time = None
        
        # Pose snapshot for capture loop (thread-safe read)
        self._lock = threading.Lock()
        self._snapshot_x = 0.0
        self._snapshot_y = 0.0
        self._snapshot_theta = 0.0
        self._snapshot_time = 0.0
        self._snapshot_encoder_ok = False
        # Latest visual odometry correction from capture (consumed once per odom tick)
        self._vis_yaw = 0.0
        self._vis_fwd = 0.0
        self._vis_conf = 0.0
        self._vis_pending = False
        
    def start(self):
        """Start the high-rate odometry thread."""
        if self._running:
            return
        self._running = True
        self._last_update_time = time.monotonic()
        self._thread = threading.Thread(target=self._odom_loop, 
                                        name="odom-thread", daemon=True)
        self._thread.start()
        print("odom_thread: started (target %.0f Hz)" % self._target_hz)
    
    def stop(self):
        """Stop the odometry thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("odom_thread: stopped")
    
    def _odom_loop(self):
        """High-rate loop: read wheels+IMU, integrate, publish pose."""
        interval = 1.0 / self._target_hz
        _loop_count = 0
        _t_start = time.monotonic()
        
        while self._running:
            t0 = time.monotonic()
            
            # --- 1. Read wheel velocities (non-blocking) ---
            vl, vr = 0.0, 0.0
            using_encoder_feedback = False
            if self._wheelbase is not None:
                try:
                    vl, vr = self._wheelbase.get_wheel_velocities_mps()
                    health = self._wheelbase.get_encoder_health()
                    using_encoder_feedback = bool(
                        health.get('encoder_ok') and health.get('age_s', 99) < 1.0)
                except Exception:
                    pass
            
            # --- 2. Read IMU yaw rate (non-blocking) ---
            imu_yaw_rate = 0.0
            if self._imu is not None and self._imu.ok:
                try:
                    if self._imu.grab():
                        # Transform to body frame (assuming FW_PITCH_DEG from vision)
                        # For now, just take raw Z-axis gyro (this will be refined)
                        _, _, imu_yaw_rate = self._imu.get_angular_velocity_body(
                            camera_pitch_deg=-64.4)  # FW_PITCH_DEG + 90
                except Exception:
                    pass
            
            # --- 3. Compute dt ---
            now = time.monotonic()
            if self._last_update_time is None:
                dt = 0.0
            else:
                dt = now - self._last_update_time
            self._last_update_time = now
            
            # --- 4. Integrate (pose estimator handles fusion) ---
            # Consume one pending visual correction from capture (frame-tied VO).
            with self._lock:
                if self._vis_pending:
                    vis_yaw, vis_fwd, vis_conf = (
                        self._vis_yaw, self._vis_fwd, self._vis_conf)
                    self._vis_pending = False
                else:
                    vis_yaw = vis_fwd = vis_conf = 0.0
            if dt > 0:
                self._pose.update(
                    vl, vr, dt,
                    vis_yaw=vis_yaw, vis_fwd=vis_fwd, vis_confidence=vis_conf,
                    using_encoder_feedback=using_encoder_feedback,
                    imu_yaw_rate=imu_yaw_rate)
            
            # --- 5. Publish pose snapshot (thread-safe) ---
            with self._lock:
                self._snapshot_x = self._pose.x
                self._snapshot_y = self._pose.y
                self._snapshot_theta = self._pose.theta
                self._snapshot_time = time.time()
                self._snapshot_encoder_ok = using_encoder_feedback
            
            # --- 6. Rate limiting ---
            _loop_count += 1
            if _loop_count % 1000 == 0:
                elapsed = time.monotonic() - _t_start
                actual_hz = _loop_count / elapsed
                print("odom_thread: %.1f Hz (target %.1f Hz) after %d loops" 
                      % (actual_hz, self._target_hz, _loop_count))
            
            dt_loop = time.monotonic() - t0
            sleep_time = max(0.0, interval - dt_loop)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_pose_snapshot(self):
        """Return latest pose snapshot (x, y, theta, timestamp, encoder_ok).
        
        Thread-safe: can be called from capture loop without blocking odom.
        
        Returns:
            (x, y, theta, timestamp, encoder_ok) tuple
        """
        with self._lock:
            return (self._snapshot_x, self._snapshot_y,
                    self._snapshot_theta, self._snapshot_time,
                    self._snapshot_encoder_ok)
    
    def set_wheelbase(self, wheelbase):
        """Update wheelbase reference (safe if called before/after start)."""
        self._wheelbase = wheelbase

    def set_imu(self, imu_pipeline):
        """Update IMU pipeline reference."""
        self._imu = imu_pipeline

    def apply_visual_correction(self, vis_yaw, vis_fwd, vis_confidence):
        """Queue a frame-tied visual odometry correction for the next odom tick.

        Capture computes VO on synced RS2 color and calls this; the high-rate
        thread consumes it once on the next pose.update so wheels are not
        double-integrated.
        """
        with self._lock:
            self._vis_yaw = float(vis_yaw)
            self._vis_fwd = float(vis_fwd)
            self._vis_conf = float(vis_confidence)
            self._vis_pending = True
