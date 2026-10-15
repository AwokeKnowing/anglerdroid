"""cameras.py – Hardware: 2x RealSense D435 + 1x USB webcam.
Pre-allocated numpy buffers. Each camera's clock is wait_for_frames / V4L read
on its own pump thread; capture/main wait for a new seq (≤1 frame late).
"""

import os
import threading
import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False

from robot_config import FRAME_W, FRAME_H
RS_DEPTH_W = int(os.environ.get("RS_DEPTH_W", "848"))
RS_DEPTH_H = int(os.environ.get("RS_DEPTH_H", "480"))
RS_FPS = int(os.environ.get("RS_FPS", "30"))
RGB_CAP_W, RGB_CAP_H = 640, 480
RS_DECIMATE_MAG = 3  # detail-first default (~45k verts on JP6). Do NOT drop mag for speed alone — see AGENTS.md Depth detail metric. Env override OK for bake-offs.
# Two frame periods: miss one hitch, never spin past the camera clock.
WAIT_FRAME_MS = int(os.environ.get("KEVIN_WAIT_FRAME_MS", "66"))


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
    Set capture_ir=True to also capture stereo IR frames (cuVSLAM / i777 ORB-SLAM3).
    """

    def __init__(self, serial, decimate_mag=RS_DECIMATE_MAG,
                 compute_pointcloud=True, capture_ir=False, emitter=True):
        if not HAS_RS:
            raise ImportError("pyrealsense2 not available")

        self._capture_ir = capture_ir
        self._pipe = rs.pipeline()
        self.profile = self._start_streams(serial, RS_DEPTH_W, RS_DEPTH_H, capture_ir)
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
        _set_sensor_opt(sensor, rs.option.laser_power, 360 if emitter else 0)
        _set_sensor_opt(sensor, rs.option.emitter_enabled, 1 if emitter else 0)
        _set_sensor_opt(sensor, rs.option.depth_units, 0.001)
        _set_sensor_opt(sensor, rs.option.receiver_gain, 16)
        if not emitter:
            print("cameras: %s IR emitter OFF (visual SLAM, not speckle)" % serial)
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

        dw, dh = self._depth_wh()
        self.color = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.depth = np.zeros((dh, dw), dtype=np.uint16)
        self.depth_stamp = 0.0
        # Pre-allocate IR buffers (if capturing IR, allocate on first grab)
        self.ir_left = None
        self.ir_right = None
        self.ok = False
        # Double buffer: ingest writes work; publish swaps so readers hold a
        # stable array for one more frame while the pump waits on the next.
        self._color_work = self.color
        self._color_pub = np.zeros_like(self.color)
        self._depth_work = self.depth
        self._depth_pub = np.zeros_like(self.depth)
        self._verts_work = None
        self._verts_pub = None
        self._ir_l_work = None
        self._ir_l_pub = None
        self._ir_r_work = None
        self._ir_r_pub = None
        self.pub_lock = threading.Lock()
        self.pub_seq = 0
        self.pub_t = 0.0
        print("cameras: %s depth %dx%d color %dx%d @ %d Hz" % (
            serial, dw, dh, RGB_CAP_W, RGB_CAP_H, RS_FPS), flush=True)

    def _cfg(self, serial, dw, dh, capture_ir):
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, RS_FPS)
        cfg.enable_stream(rs.stream.color, RGB_CAP_W, RGB_CAP_H, rs.format.rgb8, RS_FPS)
        if capture_ir:
            cfg.enable_stream(rs.stream.infrared, 1, dw, dh, rs.format.y8, RS_FPS)
            cfg.enable_stream(rs.stream.infrared, 2, dw, dh, rs.format.y8, RS_FPS)
        return cfg

    def _start_streams(self, serial, dw, dh, capture_ir):
        """Prefer requested depth size @ RS_FPS; USB2 D435 often cannot do 848p."""
        try:
            return self._pipe.start(self._cfg(serial, dw, dh, capture_ir))
        except Exception as e:
            if (dw, dh) == (640, 480):
                raise
            print("cameras: %s %dx%d@%d failed (%s); retry 640x480@%d"
                  % (serial, dw, dh, RS_FPS, e, RS_FPS), flush=True)
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = rs.pipeline()
            return self._pipe.start(self._cfg(serial, 640, 480, capture_ir))

    def _depth_wh(self):
        try:
            vs = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            i = vs.get_intrinsics()
            return int(i.width), int(i.height)
        except Exception:
            return RS_DEPTH_W, RS_DEPTH_H

    def depth_intrinsics(self):
        """fx, fy, ppx, ppy, width, height of the raw (undecimated) depth stream."""
        try:
            vs = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            i = vs.get_intrinsics()
            return {
                "fx": float(i.fx), "fy": float(i.fy),
                "ppx": float(i.ppx), "ppy": float(i.ppy),
                "width": int(i.width), "height": int(i.height),
            }
        except Exception:
            return {}

    def wait_frameset(self, timeout_ms=None):
        """Block on the camera clock. Do not poll — that desyncs the loop."""
        if timeout_ms is None:
            timeout_ms = WAIT_FRAME_MS
        try:
            return self._pipe.wait_for_frames(int(timeout_ms))
        except Exception:
            return None

    def _ensure_pair(self, raw, work, pub):
        if work is None or work.shape != raw.shape:
            work = np.zeros_like(raw)
            pub = np.zeros_like(raw)
        return work, pub

    def ingest(self, frames):
        """Copy + decimate + pc into the work slot. Does not publish."""
        import os
        _prof = os.environ.get("KEVIN_GRAB_PROF") == "1"
        _t1 = time.monotonic() if _prof else 0.0
        d = frames.get_depth_frame()
        c = frames.get_color_frame()
        if not d:
            self.ok = False
            return False

        raw_z = np.asarray(d.get_data())
        if raw_z.shape != self._depth_work.shape:
            self._depth_work = np.array(raw_z, dtype=np.uint16, copy=True)
            self._depth_pub = np.zeros_like(self._depth_work)
            self.depth = self._depth_pub
        else:
            np.copyto(self._depth_work, raw_z)
        try:
            self.depth_stamp = float(d.get_timestamp()) * 1e-3
        except Exception:
            self.depth_stamp = time.monotonic()

        if c:
            raw_c = np.asarray(c.get_data())
            if raw_c.shape[0] != FRAME_H or raw_c.shape[1] != FRAME_W:
                raw_c = cv2.resize(raw_c, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            np.copyto(self._color_work, raw_c)
        _t2 = time.monotonic() if _prof else 0.0

        if self._compute_pc:
            d = self._decimate.process(d)
            points = self._pc.calculate(d)
            v = points.get_vertices()
            raw = np.asanyarray(v).view(np.float32).reshape(-1, 3)
            self._verts_work, self._verts_pub = self._ensure_pair(
                raw, self._verts_work, self._verts_pub)
            np.copyto(self._verts_work, raw)
        _t3 = time.monotonic() if _prof else 0.0
        if _prof:
            def _acc(name, dt):
                n = getattr(self, "_prof_n", 0)
                prev = getattr(self, name, 0.0)
                setattr(self, name, (prev * n + dt * 1000.0) / (n + 1))
            if not hasattr(self, "_prof_n"):
                self._prof_n = 0
            _acc("_prof_poll_ms", 0.0)
            _acc("_prof_color_ms", _t2 - _t1)
            _acc("_prof_pc_ms", _t3 - _t2)
            self._prof_waited_frac = 1.0
            self._prof_n += 1

        if self._capture_ir:
            ir1 = frames.get_infrared_frame(1)
            ir2 = frames.get_infrared_frame(2)
            if ir1 and ir2:
                ir1_data = np.asarray(ir1.get_data())
                ir2_data = np.asarray(ir2.get_data())
                self._ir_l_work, self._ir_l_pub = self._ensure_pair(
                    ir1_data, self._ir_l_work, self._ir_l_pub)
                self._ir_r_work, self._ir_r_pub = self._ensure_pair(
                    ir2_data, self._ir_r_work, self._ir_r_pub)
                np.copyto(self._ir_l_work, ir1_data)
                np.copyto(self._ir_r_work, ir2_data)

        self.ok = True
        return True

    def publish(self):
        """Swap work → published. Readers keep .color/.verts/.depth as pub."""
        with self.pub_lock:
            self._color_work, self._color_pub = self._color_pub, self._color_work
            self._depth_work, self._depth_pub = self._depth_pub, self._depth_work
            self.color = self._color_pub
            self.depth = self._depth_pub
            if self._verts_work is not None:
                self._verts_work, self._verts_pub = self._verts_pub, self._verts_work
                self.verts = self._verts_pub
            if self._ir_l_work is not None:
                self._ir_l_work, self._ir_l_pub = self._ir_l_pub, self._ir_l_work
                self._ir_r_work, self._ir_r_pub = self._ir_r_pub, self._ir_r_work
                self.ir_left = self._ir_l_pub
                self.ir_right = self._ir_r_pub
            self.pub_seq += 1
            self.pub_t = time.monotonic()
        return True

    def grab_latest(self, timeout_ms=None):
        """Wait for the next hardware frameset, ingest, publish. Camera clock."""
        frames = self.wait_frameset(timeout_ms)
        if not frames:
            self.ok = False
            return False
        if not self.ingest(frames):
            return False
        return self.publish()

    def grab(self):
        """Compat for probes: blocking wait + ingest + publish."""
        return self.grab_latest()

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

        self._color_work = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self._color_pub = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.color = self._color_pub
        self.ok = False
        self.pub_lock = threading.Lock()
        self.pub_seq = 0
        self.pub_t = 0.0
        self._had_frame = False

    def _ingest_bgr(self, f):
        if f is None:
            return False
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        if f.shape[1] != FRAME_W or f.shape[0] != FRAME_H:
            f = cv2.resize(f, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        f = f[::-1, ::-1]  # vflip + hflip (camera upside down and mirrored)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        np.copyto(self._color_work, f)
        with self.pub_lock:
            self._color_work, self._color_pub = self._color_pub, self._color_work
            self.color = self._color_pub
            self.pub_seq += 1
            self.pub_t = time.monotonic()
        self._had_frame = True
        self.ok = True
        return True

    def grab_latest(self, timeout_ms=None):
        """Blocking V4L read — this thread is the RGB 30 Hz clock."""
        if not self._cap or not self._cap.isOpened():
            self.ok = False
            return False
        ret, f = self._cap.read()
        if not ret:
            return False
        return self._ingest_bgr(f)

    def grab(self):
        """Non-blocking poll of the driver queue (sticky last color)."""
        if not self._cap or not self._cap.isOpened():
            self.ok = False
            return False
        if not self._cap.grab():
            return bool(self._had_frame)
        ret, f = self._cap.retrieve()
        if not ret or f is None:
            return bool(self._had_frame)
        return self._ingest_bgr(f)

    def stop(self):
        if self._cap:
            self._cap.release()


def _grab_cpus():
    """Optional KEVIN_GRAB_CPU=rs1,rs2,rgb core ids. Off by default.

    Pinning the waiter can cut scheduler jitter on wait_for_frames. The
    published numpy still has to reach the process core, so a pin is not
    free — measure before leaving it on.
    """
    raw = os.environ.get("KEVIN_GRAB_CPU", "").strip()
    if not raw:
        return {}
    parts = [p.strip() for p in raw.split(",")]
    names = ("rs1", "rs2", "rgb")
    out = {}
    for i, name in enumerate(names):
        if i >= len(parts) or parts[i] == "":
            continue
        try:
            out[name] = int(parts[i])
        except ValueError:
            pass
    return out


class FramePump:
    """One thread per camera: block on the device, publish newest only."""

    def __init__(self, cam, name, cpu=None):
        self.cam = cam
        self.name = name
        self.cpu = cpu
        self.seq = 0
        self.pub_t = 0.0
        self.ok = False
        self._ev = threading.Event()
        self._stop = threading.Event()
        self._th = None

    def start(self):
        self._th = threading.Thread(
            target=self._loop, name="grab-%s" % self.name, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        if self.cpu is not None:
            try:
                os.sched_setaffinity(0, {int(self.cpu)})
                print("cameras: grab-%s pinned to cpu %s" % (self.name, self.cpu),
                      flush=True)
            except Exception as e:
                print("cameras: grab-%s affinity failed: %s" % (self.name, e),
                      flush=True)
        while not self._stop.is_set():
            try:
                ok = bool(self.cam.grab_latest())
            except Exception:
                ok = False
            if ok:
                self.seq = int(getattr(self.cam, "pub_seq", self.seq + 1))
                self.pub_t = float(getattr(self.cam, "pub_t", time.monotonic()))
                self.ok = True
                self._ev.set()
            elif self._stop.wait(0.001):
                break

    def wait_new(self, last_seq, timeout_s):
        """True when pub_seq > last_seq. Sleeps on the camera event."""
        deadline = time.monotonic() + float(timeout_s)
        while self.seq <= last_seq and not self._stop.is_set():
            rem = deadline - time.monotonic()
            if rem <= 0.0:
                return False
            self._ev.clear()
            if self.seq > last_seq:
                return True
            self._ev.wait(rem)
        return self.seq > last_seq

    def stop(self):
        self._stop.set()
        self._ev.set()
        if self._th is not None:
            self._th.join(timeout=1.5)
            self._th = None
