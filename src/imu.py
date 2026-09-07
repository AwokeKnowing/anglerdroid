"""imu.py – D435i Motion Module IMU pipeline.

Separate pipeline for gyro + accel to avoid frame starvation when sharing
with depth/color. Designed for pose stabilization when visual lock is weak.

Hardware requirements:
  - D435i with Motion Module firmware
  - librealsense 2.58.4+ with FORCE_RSUSB_BACKEND=ON (Jetson Orin NX JetPack 6)
  - Working rates: accel @ 250 Hz, gyro @ 200 Hz

Usage:
    imu = IMUPipeline(serial="944622074292")
    if imu.ok:
        imu.grab()
        gyro = imu.gyro  # (x, y, z) rad/s, right-hand: X-right, Y-down, Z-forward
        accel = imu.accel  # (x, y, z) m/s², same frame
        ts = imu.timestamp  # seconds (monotonic)
"""

import time
import numpy as np

try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False

# IMU rates (Hz). Use these exact rates on D435i after librealsense 2.58.4.
# accel@200 fails on some units; accel@250 + gyro@200 is the working combo.
ACCEL_HZ = 250
GYRO_HZ = 200


class IMUPipeline:
    """Separate RealSense pipeline for D435i Motion Module (gyro + accel).
    
    CRITICAL: Do NOT share this pipeline with depth/color streams.
    Sharing can starve IMU frames or depth frames unpredictably.
    
    Opens gyro @ 200 Hz, accel @ 250 Hz.
    Exposes latest gyro (rad/s) and accel (m/s²) in camera frame.
    
    Camera frame (RealSense convention):
      X = right, Y = down, Z = forward (right-hand coordinates)
    
    Graceful degradation: If Motion Module unavailable, ok=False and grab()
    is a no-op. Check .ok before using .gyro / .accel.
    """
    
    def __init__(self, serial: str):
        """Open IMU pipeline for the specified D435i serial.
        
        Args:
            serial: D435i serial number (e.g. "944622074292")
        """
        self.serial = serial
        self.ok = False
        self.gyro = np.zeros(3, dtype=np.float32)   # (x, y, z) rad/s
        self.accel = np.zeros(3, dtype=np.float32)  # (x, y, z) m/s²
        self.timestamp = 0.0  # seconds (monotonic)
        
        self._pipe = None
        self._profile = None
        
        if not HAS_RS:
            print(f"imu: pyrealsense2 not available (serial {serial})")
            return
        
        try:
            cfg = rs.config()
            cfg.enable_device(serial)
            
            # Enable gyro and accel streams at working rates
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, GYRO_HZ)
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, ACCEL_HZ)
            
            self._pipe = rs.pipeline()
            self._profile = self._pipe.start(cfg)
            
            # Discard first few frames (IMU settling)
            for _ in range(10):
                try:
                    self._pipe.wait_for_frames(100)
                except Exception:
                    break
            
            self.ok = True
            print(f"imu: pipeline started for {serial} "
                  f"(gyro {GYRO_HZ} Hz, accel {ACCEL_HZ} Hz)")
            
        except Exception as e:
            self.ok = False
            if self._pipe:
                try:
                    self._pipe.stop()
                except Exception:
                    pass
                self._pipe = None
            print(f"imu: Motion Module not available on {serial}: {e}")
            print(f"     → graceful degradation: IMU fusion disabled")
    
    def grab(self) -> bool:
        """Poll for latest IMU frame and update gyro/accel.
        
        Non-blocking. Updates self.gyro, self.accel, self.timestamp.
        
        Returns:
            True if new data available, False if no new frame or error.
        """
        if not self.ok or not self._pipe:
            return False
        
        try:
            # Non-blocking poll
            frames = self._pipe.poll_for_frames()
            if not frames:
                return False
            
            # Extract gyro and accel
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            accel_frame = frames.first_or_default(rs.stream.accel)
            
            if gyro_frame:
                motion = gyro_frame.as_motion_frame()
                data = motion.get_motion_data()
                self.gyro[0] = data.x
                self.gyro[1] = data.y
                self.gyro[2] = data.z
            
            if accel_frame:
                motion = accel_frame.as_motion_frame()
                data = motion.get_motion_data()
                self.accel[0] = data.x
                self.accel[1] = data.y
                self.accel[2] = data.z
            
            # Use latest timestamp (gyro or accel, whichever is newer)
            if gyro_frame or accel_frame:
                self.timestamp = time.monotonic()
                return True
            
            return False
            
        except Exception as e:
            # Never crash the grab loop
            if not hasattr(self, '_grab_err_logged'):
                print(f"imu: grab error (continuing): {e}")
                self._grab_err_logged = True
            return False
    
    def stop(self):
        """Stop the IMU pipeline and release resources."""
        if self._pipe:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None
        self.ok = False
    
    def get_angular_velocity_body(self, camera_pitch_deg: float = 64.4) -> tuple:
        """Transform gyro from camera frame to robot body frame.
        
        Camera is pitched down (typically 64.4° for D435i forward mount).
        Robot body frame: X-forward, Y-left, Z-up (REP-103 / standard robotics).
        
        Args:
            camera_pitch_deg: Camera pitch angle (positive = nose down).
        
        Returns:
            (omega_x, omega_y, omega_z) in body frame (rad/s).
            omega_z = yaw rate (most useful for pose fusion).
        """
        import math
        
        # Camera frame: X-right, Y-down, Z-forward
        # Body frame: X-forward, Y-left, Z-up
        # Camera pitched by pitch_deg around X-axis (camera frame)
        
        pitch_rad = math.radians(camera_pitch_deg)
        c = math.cos(pitch_rad)
        s = math.sin(pitch_rad)
        
        gx, gy, gz = self.gyro
        
        # Rotate camera gyro to body frame:
        # Body-X (forward) ≈ Camera-Z * cos(pitch) + Camera-Y * sin(pitch)
        # Body-Y (left)    ≈ -Camera-X
        # Body-Z (up)      ≈ -Camera-Z * sin(pitch) + Camera-Y * cos(pitch)
        
        omega_body_x = gz * c + gy * s
        omega_body_y = -gx
        omega_body_z = -gz * s + gy * c
        
        return omega_body_x, omega_body_y, omega_body_z
