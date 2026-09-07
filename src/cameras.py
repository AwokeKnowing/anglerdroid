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

from robot_config import FRAME_W, FRAME_H
RS_DEPTH_W, RS_DEPTH_H = 848, 480
RGB_CAP_W, RGB_CAP_H = 640, 480
RS_DECIMATE_MAG = 3  # detail-first default (~45k verts on JP6). Do NOT drop mag for speed alone — see AGENTS.md Depth detail metric. Env override OK for bake-offs.


def _set_sensor_opt(sensor, option, value):
    try:
        sensor.set_option(option, value)
    except Exception:
        pass



def find_rgb_device():
    """Pick the USB RGB webcam V4L node (not RealSense).

    RealSense also registers /dev/video* nodes; defaulting to video0 often
    hits a RealSense metadata/depth node that OpenCV cannot capture.
    Prefer names containing 'USB Camera' / 'webcam', skip 'RealSense' / 'Intel'.
    """
    import os
    base = '/sys/class/video4linux'
    if not os.path.isdir(base):
        return None
    candidates = []
    for ent in sorted(os.listdir(base), key=lambda s: int(s.replace('video', '') or -1)):
        if not ent.startswith('video'):
            continue
        name_path = os.path.join(base, ent, 'name')
        try:
            name = open(name_path, 'r').read().strip()
        except OSError:
            continue
        low = name.lower()
        if 'realsense' in low or 'intel(r)' in low:
            continue
        path = '/dev/' + ent
        if not os.path.exists(path):
            continue
        score = 0
        if 'usb camera' in low or 'webcam' in low or '16mp' in low:
            score += 10
        if 'camera' in low:
            score += 1
        candidates.append((score, int(ent.replace('video', '')), path, name))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-t[0], t[1]))
    path, name = candidates[0][2], candidates[0][3]
    print('cameras: auto-selected RGB %s (%s)' % (path, name))
    return path


def _open_rgb_capture(device_id):
    """Open V4L2 camera by path or int index. Tries path, then numeric fallback."""
    import re
    if device_id is None or device_id == "":
        device_id = find_rgb_device()
        if not device_id:
            return None
    # Normalize "dev/videoN" -> "/dev/videoN"
    if isinstance(device_id, str) and device_id.startswith('dev/'):
        device_id = '/' + device_id
    attempts = []
    if isinstance(device_id, str):
        attempts.append(device_id)
        m = re.search(r"video(\d+)$", device_id)
        if m:
            attempts.append(int(m.group(1)))
    else:
        attempts.append(device_id)
    for attempt in attempts:
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

    With decimate_mag=8 the pointcloud has ~106x60 = 6360 vertices;
    making downstream numpy processing trivial (<1 ms).
    Set compute_pointcloud=False for cameras that only provide color.
    Set capture_ir=True to also capture stereo IR frames (for cuVSLAM).
    """

    def __init__(self, serial, decimate_mag=RS_DECIMATE_MAG,
                 compute_pointcloud=True, capture_ir=False):
        if not HAS_RS:
            raise ImportError("pyrealsense2 not available")

        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, RS_DEPTH_W, RS_DEPTH_H, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.rgb8, 30)

        self._capture_ir = capture_ir
        if capture_ir:
            cfg.enable_stream(rs.stream.infrared, 1, RS_DEPTH_W, RS_DEPTH_H, rs.format.y8, 30)
            cfg.enable_stream(rs.stream.infrared, 2, RS_DEPTH_W, RS_DEPTH_H, rs.format.y8, 30)

        self._pipe = rs.pipeline()
        self.profile = self._pipe.start(cfg)
        # Discard a few frames while AE/laser settle — avoids first-grab timeouts.
        for _ in range(5):
            try:
                self._pipe.wait_for_frames(2000)
            except Exception:
                break

        # Keep only the newest frames — a deep RS queue is a classic multi-second lag source.
        try:
            for sens in self.profile.get_device().sensors:
                _set_sensor_opt(sens, rs.option.frames_queue_size, 2)
        except Exception:
            pass
        # Keep queues tiny so we always take the newest frame (no multi-frame lag).
        try:
            for sens in self.profile.get_device().sensors:
                _set_sensor_opt(sens, rs.option.frames_queue_size, 1)
        except Exception:
            pass

        sensor = self.profile.get_device().first_depth_sensor()
        _set_sensor_opt(sensor, rs.option.visual_preset, 3)       # High Density
        _set_sensor_opt(sensor, rs.option.laser_power, 360)
        _set_sensor_opt(sensor, rs.option.emitter_enabled, 1)
        _set_sensor_opt(sensor, rs.option.depth_units, 0.001)
        _set_sensor_opt(sensor, rs.option.receiver_gain, 16)
        # Cap exposure so low-light AE cannot stretch past ~30Hz budget.
        # Depth exposure is microseconds; 15000us = 15ms leaves headroom under 33ms.
        try:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
            if sensor.supports(rs.option.exposure):
                sensor.set_option(rs.option.exposure, 15000)  # 15ms
            if sensor.supports(rs.option.gain):
                # bump gain a bit to compensate for capped exposure in dark rooms
                r = sensor.get_option_range(rs.option.gain)
                sensor.set_option(rs.option.gain, min(r.max, max(r.min, 64)))
        except Exception as e:
            print("cameras: depth exposure cap failed: %s" % e)

        # Color sensor AE can also blow the frame period in low light.
        try:
            for sens in self.profile.get_device().sensors:
                name = ""
                try:
                    name = sens.get_info(rs.camera_info.name).lower()
                except Exception:
                    pass
                if "rgb" not in name and "color" not in name:
                    continue
                if sens.supports(rs.option.enable_auto_exposure):
                    sens.set_option(rs.option.enable_auto_exposure, 0)
                if sens.supports(rs.option.exposure):
                    # D400 RGB exposure units are not always us; keep near default (~166)
                    # but never unbounded AE. Prefer a mid value then raise gain.
                    r = sens.get_option_range(rs.option.exposure)
                    target = min(r.max, max(r.min, 200))
                    sens.set_option(rs.option.exposure, target)
                if sens.supports(rs.option.gain):
                    r = sens.get_option_range(rs.option.gain)
                    sens.set_option(rs.option.gain, min(r.max, max(r.min, 96)))
                _set_sensor_opt(sens, rs.option.frames_queue_size, 1)
        except Exception as e:
            print("cameras: color exposure cap failed: %s" % e)

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
        self.ir_left = None
        self.ir_right = None
        self.ok = False

    def grab(self):
        """Take the newest frameset without multi-second stalls.

        Prefer poll_for_frames (non-blocking). If empty, wait briefly (~1 frame).
        Never wait hundreds of ms — that destroys 30Hz freshness.
        Never raises — sets ok=False on miss.
        Set KEVIN_GRAB_PROF=1 to accumulate poll/color/pc phase ms.
        """
        import os, time as _time
        _prof = os.environ.get('KEVIN_GRAB_PROF') == '1'
        _t0 = _time.monotonic() if _prof else 0.0
        try:
            frames = self._pipe.poll_for_frames()
            waited = False
            if not frames:
                frames = self._pipe.wait_for_frames(40)
                waited = True
        except Exception:
            self.ok = False
            return False
        _t1 = _time.monotonic() if _prof else 0.0
        d = frames.get_depth_frame()
        c = frames.get_color_frame()
        if not d or not c:
            self.ok = False
            return False

        self.color[:] = np.asarray(c.get_data())
        _t2 = _time.monotonic() if _prof else 0.0

        if self._compute_pc:
            d = self._decimate.process(d)
            points = self._pc.calculate(d)
            v = points.get_vertices()
            raw = np.asanyarray(v).view(np.float32).reshape(-1, 3)
            if self.verts is None or self.verts.shape[0] != raw.shape[0]:
                self.verts = np.zeros_like(raw)
            np.copyto(self.verts, raw)
        _t3 = _time.monotonic() if _prof else 0.0
        if _prof:
            # Rolling means on instance for probe scripts
            def _acc(name, dt):
                n = getattr(self, '_prof_n', 0)
                prev = getattr(self, name, 0.0)
                setattr(self, name, (prev * n + dt * 1000.0) / (n + 1))
            if not hasattr(self, '_prof_n'):
                self._prof_n = 0
            _acc('_prof_poll_ms', _t1 - _t0)
            _acc('_prof_color_ms', _t2 - _t1)
            _acc('_prof_pc_ms', _t3 - _t2)
            self._prof_waited_frac = (
                (getattr(self, '_prof_waited_frac', 0.0) * self._prof_n + (1.0 if waited else 0.0))
                / (self._prof_n + 1))
            self._prof_n += 1

        if self._capture_ir:
            ir1 = frames.get_infrared_frame(1)
            ir2 = frames.get_infrared_frame(2)
            if ir1 and ir2:
                self.ir_left = np.asarray(ir1.get_data()).copy()
                self.ir_right = np.asarray(ir2.get_data()).copy()

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
            # Cap exposure for 30Hz freshness in low light (V4L units vary by driver).
            try:
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=manual on many UVC drivers
                self._cap.set(cv2.CAP_PROP_EXPOSURE, 50)      # short; raise if too dark
                self._cap.set(cv2.CAP_PROP_GAIN, 64)
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
        """Non-blocking when possible: never stall the 30Hz RS barrier.

        Uses grab()+retrieve(). If no new USB frame is ready, keeps the last
        color buffer and returns True so parallel RS grabs aren't gated on
        webcam period (was a multi-10ms bleed on Orin).
        """
        if not self._cap or not self._cap.isOpened():
            self.ok = False
            return False
        # Non-blocking poll of driver queue
        if not self._cap.grab():
            # No new frame — keep last self.color (sticky). Still "ok" if we ever had one.
            return bool(getattr(self, '_had_frame', False))
        ret, f = self._cap.retrieve()
        if not ret or f is None:
            return bool(getattr(self, '_had_frame', False))
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        if f.shape[1] != FRAME_W or f.shape[0] != FRAME_H:
            f = cv2.resize(f, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        f = f[::-1, ::-1]  # vflip + hflip (camera upside down and mirrored)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        np.copyto(self.color, f)
        self._had_frame = True
        self.ok = True
        return True

    def stop(self):
        if self._cap:
            self._cap.release()
