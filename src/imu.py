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

import threading
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
    
    def __init__(self, serial: str, device=None):
        """Open IMU pipeline for the specified D435i serial.
        
        Args:
            serial: D435i serial number (e.g. "944622074292")
            device: the rs.device of an already running video pipeline on this camera (RSCamera.profile.get_device()).
                Given: the motion sensor of THAT device is opened directly (sensor API, callback).  Needed on the
                Orin RSUSB build: a second pipeline on a device whose video pipeline is running fails with
                "No device connected" and leaves the IMU HID wedged (0 samples) until a hardware reset (2026-10-20).
        """
        self.serial = serial
        self.ok = False
        self.gyro = np.zeros(3, dtype=np.float32)   # (x, y, z) rad/s
        self.accel = np.zeros(3, dtype=np.float32)  # (x, y, z) m/s²
        self.timestamp = 0.0  # seconds (monotonic)
        self._ring_n = 512
        self._ring_g = np.zeros((self._ring_n, 3), dtype=np.float32)
        self._ring_a = np.zeros((self._ring_n, 3), dtype=np.float32)
        self._ring_t = np.zeros(self._ring_n, dtype=np.float64)
        self._ring_w = 0
        self._ring_lock = threading.Lock()
        
        self._pipe = None
        self._profile = None
        
        if not HAS_RS:
            print(f"imu: pyrealsense2 not available (serial {serial})")
            return
        self._sensor = None; self._grab_w = 0
        if device is not None:
            try:
                self._open_sensor(device); return
            except Exception as e:                       # e.g. the sim shim has no sensor API: separate pipeline
                self.ok = False; self._sensor = None
                print(f"imu: motion sensor of the running device not usable on {serial} ({e}); trying a separate pipeline")
        
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
    
    def _open_sensor(self, device):
        """Motion sensor of a running device: gyro GYRO_HZ + accel ACCEL_HZ into the same ring, from a callback."""
        m = [x for x in device.query_sensors() if x.is_motion_sensor()]
        if not m: raise RuntimeError("no motion sensor on this device")
        m = m[0]; want = {rs.stream.gyro: GYRO_HZ, rs.stream.accel: ACCEL_HZ}; sel = {}
        for pr in m.get_stream_profiles():
            if want.get(pr.stream_type()) == pr.fps(): sel[pr.stream_type()] = pr
        if len(sel) != 2: raise RuntimeError("gyro %d Hz / accel %d Hz profiles not offered" % (GYRO_HZ, ACCEL_HZ))
        m.open(list(sel.values())); m.start(self._on_motion); self._sensor = m; self.ok = True
        print(f"imu: motion sensor of the running device {self.serial} started "
              f"(gyro {GYRO_HZ} Hz, accel {ACCEL_HZ} Hz, sensor API)")

    def _on_motion(self, f):
        """librealsense callback thread: one ring sample per gyro or accel frame (latest of both, sensor stamp)."""
        try:
            d = f.as_motion_frame().get_motion_data(); now = time.monotonic()
            try: ts = float(f.get_timestamp()) * 1e-3
            except Exception: ts = now
            with self._ring_lock:
                if f.get_profile().stream_type() == rs.stream.gyro: self.gyro[0], self.gyro[1], self.gyro[2] = d.x, d.y, d.z
                else: self.accel[0], self.accel[1], self.accel[2] = d.x, d.y, d.z
                self.timestamp = ts if abs(ts - now) < 1.0 else now
                i = self._ring_w % self._ring_n
                self._ring_g[i] = self.gyro; self._ring_a[i] = self.accel; self._ring_t[i] = self.timestamp; self._ring_w += 1
        except Exception as e:
            if not hasattr(self, '_cb_err_logged'):
                print(f"imu: motion callback error (continuing): {e}"); self._cb_err_logged = True

    def grab(self) -> bool:
        """Poll for latest IMU frame and update gyro/accel.
        
        Non-blocking. Updates self.gyro, self.accel, self.timestamp.
        
        Returns:
            True if new data available, False if no new frame or error.
        """
        if self.ok and self._sensor is not None:          # sensor API: the callback fills the ring
            with self._ring_lock:
                got = self._ring_w != self._grab_w; self._grab_w = self._ring_w
            return got
        if not self.ok or not self._pipe:
            return False
        
        try:
            got = False
            for _ in range(256):                       # drain everything queued (a stalled loop = 100+ samples)
                frames = self._pipe.poll_for_frames()
                if not frames:
                    break
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
                if gyro_frame or accel_frame:
                    # sample time = the sensor's own stamp (same ms clock the depth frames use, cameras.py), not the
                    # host poll time: a poll drains up to 100+ queued samples, which all got one identical poll stamp
                    # (cuVSLAM dropped them as non-increasing; dt for gyro integration was wrong).  Fall back to the
                    # host clock if the device stamps are in another time domain.
                    now = time.monotonic(); f0 = gyro_frame or accel_frame
                    try: ts = float(f0.get_timestamp()) * 1e-3
                    except Exception: ts = now
                    self.timestamp = ts if abs(ts - now) < 1.0 else now
                    with self._ring_lock:
                        i = self._ring_w % self._ring_n
                        self._ring_g[i] = self.gyro
                        self._ring_a[i] = self.accel
                        self._ring_t[i] = self.timestamp
                        self._ring_w += 1
                    got = True
            return got
            
        except Exception as e:
            # Never crash the grab loop
            if not hasattr(self, '_grab_err_logged'):
                print(f"imu: grab error (continuing): {e}")
                self._grab_err_logged = True
            return False
    
    def stop(self):
        """Stop the IMU pipeline and release resources."""
        if getattr(self, "_sensor", None) is not None:
            try: self._sensor.stop(); self._sensor.close()
            except Exception: pass
            self._sensor = None
        if self._pipe:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None
        self.ok = False
    
    def get_angular_velocity_body(self, camera_pitch_deg: float = 63.0) -> tuple:
        """Transform gyro from camera frame to robot body frame.
        
        Camera is pitched down (twin: D435i forward mount 27° down = 63.0° from vertical; was 64.4 bag fit).
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

    def drain_ring(self, since: int, max_n: int = 16):
        """Copy IMU samples written after ``since`` (exclusive ring count).

        Returns (gyro[N,3], accel[N,3], t[N], new_since).
        """
        with self._ring_lock:
            w = int(self._ring_w)
            nring = self._ring_n
            avail = min(max_n, max(0, w - int(since)), nring)
            if avail <= 0:
                return (
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros(0, dtype=np.float64),
                    w,
                )
            start = w - avail
            out_g = np.empty((avail, 3), dtype=np.float32)
            out_a = np.empty((avail, 3), dtype=np.float32)
            out_t = np.empty(avail, dtype=np.float64)
            for k in range(avail):
                i = (start + k) % nring
                out_g[k] = self._ring_g[i]
                out_a[k] = self._ring_a[i]
                out_t[k] = self._ring_t[i]
            return out_g, out_a, out_t, w
