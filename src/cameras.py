"""cameras.py – Hardware: 2x RealSense D435 + 1x USB webcam.
Pre-allocated numpy buffers. Blocking grab(). No processing.
"""

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False

FRAME_W, FRAME_H = 320, 240
RS_DEPTH_W, RS_DEPTH_H = 848, 480
RGB_CAP_W, RGB_CAP_H = 640, 480
RS_DECIMATE_MAG = 4


def _set_sensor_opt(sensor, option, value):
    try:
        sensor.set_option(option, value)
    except Exception:
        pass


def _open_rgb_capture(device_id):
    """Open V4L2 camera by path or int index. Tries path, then numeric fallback."""
    import re
    for attempt in (device_id, None):
        if attempt is None:
            if not isinstance(device_id, str):
                break
            m = re.search(r"video(\d+)$", device_id)
            if not m:
                break
            attempt = int(m.group(1))
        cap = cv2.VideoCapture(attempt, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            continue
        return cap
    return None


class RSCamera:
    """RealSense D435: depth 848x480 -> decimated pointcloud + color 320x240.

    With decimate_mag=8 the pointcloud has only 106x60 = 6360 vertices,
    making downstream numpy processing trivial (<1 ms).
    Set compute_pointcloud=False for cameras that only provide color.
    """

    def __init__(self, serial, decimate_mag=RS_DECIMATE_MAG, compute_pointcloud=True):
        if not HAS_RS:
            raise ImportError("pyrealsense2 not available")

        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, RS_DEPTH_W, RS_DEPTH_H, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.rgb8, 30)

        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg)

        sensor = profile.get_device().first_depth_sensor()
        _set_sensor_opt(sensor, rs.option.visual_preset, 3)       # High Density
        _set_sensor_opt(sensor, rs.option.laser_power, 360)
        _set_sensor_opt(sensor, rs.option.enable_auto_exposure, 1)
        _set_sensor_opt(sensor, rs.option.emitter_enabled, 1)
        _set_sensor_opt(sensor, rs.option.depth_units, 0.001)
        _set_sensor_opt(sensor, rs.option.receiver_gain, 16)

        self._compute_pc = compute_pointcloud
        if compute_pointcloud:
            self._decimate = rs.decimation_filter()
            self._decimate.set_option(rs.option.filter_magnitude, decimate_mag)
            self._pc = rs.pointcloud()
            self.verts = None  # allocated on first grab (SDK decimation size varies)
        else:
            self._decimate = None
            self._pc = None
            self.verts = None

        self.color = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.ok = False

    def grab(self):
        """Block until next frameset. Fills self.color (and self.verts if enabled)."""
        frames = self._pipe.wait_for_frames()
        d = frames.get_depth_frame()
        c = frames.get_color_frame()
        if not d or not c:
            self.ok = False
            return False

        self.color[:] = np.asarray(c.get_data())

        if self._compute_pc:
            d = self._decimate.process(d)
            points = self._pc.calculate(d)
            v = points.get_vertices()
            raw = np.asanyarray(v).view(np.float32).reshape(-1, 3)
            if self.verts is None or self.verts.shape[0] != raw.shape[0]:
                self.verts = np.zeros_like(raw)
            np.copyto(self.verts, raw)

        self.ok = True
        return True

    def stop(self):
        try:
            self._pipe.stop()
        except Exception:
            pass


class WebCam:
    """USB webcam: 640x480 capture -> 320x240 RGB, pre-allocated buffer."""

    def __init__(self, device_id):
        self._cap = _open_rgb_capture(device_id)
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB_CAP_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_CAP_H)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            print("cameras: webcam opened (%dx%d MJPG -> %dx%d)" % (RGB_CAP_W, RGB_CAP_H, FRAME_W, FRAME_H))
        else:
            if self._cap:
                self._cap.release()
            self._cap = None
            print("cameras: webcam not opened (check device path)")

        self.color = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.ok = False

    def grab(self):
        """Block until next frame. Fills self.color."""
        if not self._cap or not self._cap.isOpened():
            self.ok = False
            return False
        ret, f = self._cap.read()
        if not ret or f is None:
            self.ok = False
            return False
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        if f.shape[1] != FRAME_W or f.shape[0] != FRAME_H:
            f = cv2.resize(f, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        f = f[::-1, :]  # vflip (camera mounted upside down)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        np.copyto(self.color, f)
        self.ok = True
        return True

    def stop(self):
        if self._cap:
            self._cap.release()
