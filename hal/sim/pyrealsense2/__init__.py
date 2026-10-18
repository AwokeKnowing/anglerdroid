"""Emulated librealsense (pyrealsense2) for the Jetson emulator.  Same import name, same API subset the
anglerdroid stack uses (cameras.py, imu.py, cuvslam_tracker.py, vision.py, main.py).  Frames come from the
emulator world over shared memory (emu/shm.py) at the camera's real rate; wait_for_frames blocks on the
camera clock exactly like the SDK.  Active only when hal/sim is on PYTHONPATH (KEVIN_HAL=sim)."""
from __future__ import annotations
import enum, sys, time, math
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from emu import shm as _shm  # noqa: E402

__version__ = "2.55.1-kevin-emu"
EMULATED = True


class stream(enum.IntEnum):
    any = 0; depth = 1; color = 2; infrared = 3; fisheye = 4; gyro = 5; accel = 6; pose = 7


class format(enum.IntEnum):
    any = 0; z16 = 1; rgb8 = 5; bgr8 = 6; y8 = 9; motion_xyz32f = 17


class option(enum.IntEnum):
    backlight_compensation = 0; brightness = 1; contrast = 2; exposure = 3; gain = 4; laser_power = 13
    visual_preset = 15; frames_queue_size = 19; depth_units = 34; enable_auto_exposure = 10
    emitter_enabled = 18; filter_magnitude = 21; receiver_gain = 68; holes_fill = 23


class camera_info(enum.IntEnum):
    name = 0; serial_number = 1; firmware_version = 2; product_line = 9; usb_type_descriptor = 12


class distortion(enum.IntEnum):
    none = 0; brown_conrady = 4; inverse_brown_conrady = 2


class error(RuntimeError):
    pass


class option_range:
    def __init__(self, mn, mx, step=1.0, default=0.0):
        self.min, self.max, self.step, self.default = mn, mx, step, default


class intrinsics:
    def __init__(self, w=0, h=0, fx=0., fy=0., ppx=0., ppy=0.):
        self.width, self.height, self.fx, self.fy, self.ppx, self.ppy = int(w), int(h), fx, fy, ppx, ppy
        self.model = distortion.brown_conrady; self.coeffs = [0.0] * 5

    def __repr__(self):
        return "[ %dx%d  p[%.1f %.1f]  f[%.1f %.1f] ]" % (self.width, self.height, self.ppx, self.ppy, self.fx, self.fy)


class extrinsics:
    def __init__(self, rotation=None, translation=None):
        self.rotation = list(rotation or [1, 0, 0, 0, 1, 0, 0, 0, 1]); self.translation = list(translation or [0, 0, 0])


class vector:
    def __init__(self, x, y, z): self.x, self.y, self.z = float(x), float(y), float(z)


# ------------------------------------------------------------------ devices
def _info():
    return _shm.read_info()


class sensor:
    def __init__(self, name): self._name = name; self._opts = {}
    def supports(self, opt): return True
    def set_option(self, opt, val): self._opts[opt] = val
    def get_option(self, opt): return self._opts.get(opt, 0.0)
    def get_option_range(self, opt): return option_range(0.0, 10000.0 if opt == option.exposure else 248.0)
    def get_info(self, ci): return self._name
    def is_depth_sensor(self): return "Stereo" in self._name
    def as_depth_sensor(self): return self
    def get_depth_scale(self): return 0.001
    def first_depth_sensor(self): return self


class device:
    def __init__(self, serial, d):
        self._serial, self._d = serial, d
        self.sensors = [sensor("Stereo Module"), sensor("RGB Camera")] + ([sensor("Motion Module")] if d.get("imu") else [])
    def get_info(self, ci):
        return {camera_info.serial_number: self._serial, camera_info.name: self._d["name"],
                camera_info.firmware_version: "5.16.0.1", camera_info.product_line: "D400",
                camera_info.usb_type_descriptor: "3.2"}.get(ci, "")
    def supports(self, ci): return True
    def first_depth_sensor(self): return self.sensors[0]
    def query_sensors(self): return self.sensors
    def hardware_reset(self): pass


class context:
    @property
    def devices(self):
        try:
            return [device(s, d) for s, d in sorted(_info()["devices"].items())]
        except RuntimeError:
            return []
    def query_devices(self): return self.devices


# ------------------------------------------------------------------ profiles
class stream_profile:
    def __init__(self, st, idx, fmt, fps, intr=None, pose=None):
        self._st, self._idx, self._fmt, self._fps, self._intr, self._pose = st, idx, fmt, fps, intr, pose
    def stream_type(self): return self._st
    def stream_index(self): return self._idx
    def format(self): return self._fmt
    def fps(self): return self._fps
    def as_video_stream_profile(self): return self
    def as_motion_stream_profile(self): return self
    def get_intrinsics(self): return self._intr
    def width(self): return self._intr.width
    def height(self): return self._intr.height
    def get_extrinsics_to(self, other):
        # poses are camera->rig (R row-major 3x3, t) in the device frame; return this->other
        Ra, ta = np.array(self._pose[0]).reshape(3, 3), np.array(self._pose[1])
        Rb, tb = np.array(other._pose[0]).reshape(3, 3), np.array(other._pose[1])
        R = Rb.T @ Ra; t = Rb.T @ (ta - tb)
        # librealsense stores rotation column-major
        return extrinsics(R.T.reshape(-1).tolist(), t.tolist())


class pipeline_profile:
    def __init__(self, dev, profiles): self._dev, self._p = dev, profiles
    def get_device(self): return self._dev
    def get_stream(self, st, idx=-1):
        for p in self._p:
            if p._st == st and (idx in (-1, 0) or p._idx == idx):
                return p
        raise error("stream %s not enabled" % st)
    def get_streams(self): return list(self._p)


class config:
    def __init__(self): self._serial = None; self._req = []
    def enable_device(self, serial): self._serial = str(serial)
    def enable_stream(self, st, *a):
        idx = 0
        if st == stream.infrared and a and isinstance(a[0], int) and len(a) == 5:
            idx, a = a[0], a[1:]
        self._req.append((st, idx, a))
    def disable_all_streams(self): self._req = []
    def resolve(self, p): return None
    def can_resolve(self, p): return True


# ------------------------------------------------------------------ frames
class frame:
    def __init__(self, data, stamp_ms, number, prof=None, kind="video"):
        self._data, self._t, self._n, self._prof, self._kind = data, stamp_ms, number, prof, kind
    def __bool__(self): return self._data is not None
    def get_data(self): return self._data
    def get_timestamp(self): return self._t
    def get_frame_number(self): return self._n
    def get_profile(self): return self._prof
    def get_width(self): return self._data.shape[1]
    def get_height(self): return self._data.shape[0]
    def as_video_frame(self): return self
    def as_depth_frame(self): return self
    def as_frame(self): return self
    def as_motion_frame(self): return self
    def is_depth_frame(self): return self._kind == "depth"
    def get_motion_data(self): return vector(*self._data)
    def get_distance(self, x, y): return float(self._data[y, x]) * 0.001
    def keep(self): pass


class composite_frame(frame):
    def __init__(self, frames, stamp_ms, n):
        super().__init__(True, stamp_ms, n); self._f = frames
    def __bool__(self): return bool(self._f)
    def size(self): return len(self._f)
    def __len__(self): return len(self._f)
    def __iter__(self): return iter(self._f.values())
    def first_or_default(self, st, fmt=None):
        for (s, i), f in self._f.items():
            if s == st: return f
        return None
    def first(self, st, fmt=None):
        f = self.first_or_default(st)
        if f is None: raise error("no %s frame" % st)
        return f
    def get_depth_frame(self): return self._f.get((stream.depth, 0))
    def get_color_frame(self): return self._f.get((stream.color, 0))
    def get_infrared_frame(self, idx=0): return self._f.get((stream.infrared, idx or 1))
    def get_pose_frame(self): return None


class pipeline:
    def __init__(self, ctx=None):
        self._dev = None; self._cam = None; self._imu = None; self._seen = 0; self._imu_seen = 0
        self._want = set(); self._profile = None; self._n = 0
    def start(self, cfg=None, *a):
        info = _info(); devs = info["devices"]
        serial = (cfg._serial if cfg else None) or sorted(devs)[0]
        if serial not in devs:
            raise error("No device connected (emulator has %s)" % ",".join(sorted(devs)))
        d = devs[serial]; dev = device(serial, d)
        req = cfg._req if cfg and cfg._req else [(stream.depth, 0, ()), (stream.color, 0, ())]
        profs = []
        for st, idx, args in req:
            if st in (stream.gyro, stream.accel):
                if not d.get("imu"): raise error("Couldn't resolve requests (no motion module on %s)" % serial)
                self._imu = _shm.ImuRing(d["imu"]); self._imu_seen = int(self._imu.hdr[0])
                profs.append(stream_profile(st, 0, format.motion_xyz32f, 200 if st == stream.gyro else 250))
                self._want.add((st, 0)); continue
            key = {stream.depth: "depth", stream.color: "color", stream.infrared: "ir%d" % (idx or 1)}[st]
            s = d["streams"][key]
            if len(args) >= 2 and args[0] and (int(args[0]), int(args[1])) != (s["w"], s["h"]):
                raise error("Couldn't resolve requests: %s %sx%s (emulator streams %dx%d)" % (key, args[0], args[1], s["w"], s["h"]))
            intr = intrinsics(s["w"], s["h"], s["fx"], s["fy"], s["ppx"], s["ppy"])
            profs.append(stream_profile(st, idx or (1 if st == stream.infrared else 0),
                                        {"depth": format.z16, "color": format.rgb8}.get(key, format.y8), d["fps"], intr, s["pose"]))
            self._want.add((st, idx or (1 if st == stream.infrared else 0)))
        if any(p._st in (stream.depth, stream.color, stream.infrared) for p in profs):
            self._cam = _shm.Block(d["shm"], d["layout"])
            self._seen = self._cam.seq
        self._profile = pipeline_profile(dev, profs); self._dev = d
        return self._profile
    def stop(self):
        if self._cam: self._cam.close(); self._cam = None
    def get_active_profile(self): return self._profile

    def _video_frames(self):
        seq, st_ns, a = self._cam.read()
        self._seen = seq; self._n += 1; t_ms = st_ns * 1e-6
        f = {}
        for (st, idx) in self._want:
            key = {stream.depth: "depth", stream.color: "color"}.get(st, "ir%d" % idx)
            if key in a:
                f[(st, idx)] = frame(a[key], t_ms, self._n, self._profile.get_stream(st, idx), "depth" if st == stream.depth else "video")
        return composite_frame(f, t_ms, self._n)

    def _imu_frames(self, one=True):
        n, s = self._imu.since(self._imu_seen)
        if len(s) == 0: return None
        if one:
            s = s[:1]; self._imu_seen += 1
        else:
            self._imu_seen = n
        r = s[-1]; t_ms = r["t"] * 1e-6
        return composite_frame({(stream.gyro, 0): frame(r["g"].copy(), t_ms, self._imu_seen, None, "motion"),
                                (stream.accel, 0): frame(r["a"].copy(), t_ms, self._imu_seen, None, "motion")}, t_ms, self._imu_seen)

    def wait_for_frames(self, timeout_ms=5000):
        if self._cam is None:
            t_end = time.monotonic() + timeout_ms * 1e-3
            while time.monotonic() < t_end:
                f = self._imu_frames()
                if f: return f
                time.sleep(0.001)
            raise error("Frame didn't arrive within %d" % timeout_ms)
        if not self._cam.wait_new(self._seen, timeout_ms * 1e-3):
            raise error("Frame didn't arrive within %d" % timeout_ms)
        return self._video_frames()

    def poll_for_frames(self):
        if self._cam is None:
            return self._imu_frames() or composite_frame({}, 0, 0)
        s = self._cam.seq
        if s == self._seen or (s & 1):
            return composite_frame({}, 0, 0)
        return self._video_frames()

    def try_wait_for_frames(self, timeout_ms=5000):
        try: return True, self.wait_for_frames(timeout_ms)
        except error: return False, None


# ------------------------------------------------------------------ processing blocks
class _block:
    def __init__(self, *a): self._opts = {}
    def set_option(self, o, v): self._opts[o] = v
    def get_option(self, o): return self._opts.get(o, 0)
    def supports(self, o): return True


class decimation_filter(_block):
    """librealsense semantics: median of mag x mag for mag 2-3, mean of valid (non-zero) for >= 4."""
    def __init__(self, mag=2):
        super().__init__(); self._opts[option.filter_magnitude] = mag
    def process(self, f):
        m = int(self._opts.get(option.filter_magnitude, 2)); z = f.get_data()
        if m <= 1: return f
        H, W = (z.shape[0] // m) * m, (z.shape[1] // m) * m
        b = z[:H, :W].reshape(H // m, m, W // m, m).transpose(0, 2, 1, 3).reshape(H // m, W // m, m * m)
        if m <= 3:      # median of the valid (non-zero) samples; = np.nanmedian, ~10x faster (sort of <= 9 values)
            bb = np.sort(np.where(b == 0, np.float32(np.inf), b.astype(np.float32)), axis=2)
            cnt = (b > 0).sum(2); lo = np.maximum(cnt - 1, 0) // 2; hi = cnt // 2
            hi = np.minimum(hi, m * m - 1)
            med = 0.5 * (np.take_along_axis(bb, lo[..., None], 2)[..., 0] + np.take_along_axis(bb, hi[..., None], 2)[..., 0])
            out = np.where(cnt > 0, med, 0.0).astype(np.uint16)
        else:
            v = b > 0; cnt = v.sum(2); out = np.where(cnt > 0, b.sum(2, dtype=np.uint32) / np.maximum(cnt, 1), 0).astype(np.uint16)
        p = f._prof; i = p._intr
        ni = intrinsics(W // m, H // m, i.fx / m, i.fy / m, (i.ppx + 0.5) / m - 0.5, (i.ppy + 0.5) / m - 0.5)
        return frame(out, f._t, f._n, stream_profile(p._st, p._idx, p._fmt, p._fps, ni, p._pose), "depth")


class points(frame):
    def get_vertices(self, dims=1): return self._data
    def get_texture_coordinates(self, dims=1): return np.zeros((len(self._data), 2), np.float32)
    def size(self): return len(self._data)


class pointcloud(_block):
    def __init__(self): super().__init__(); self._cache = {}
    def map_to(self, f): pass
    def calculate(self, f):
        z = f.get_data(); i = f._prof._intr; H, W = z.shape
        key = (W, H, i.fx, i.fy, i.ppx, i.ppy)
        if key not in self._cache:
            u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            self._cache = {key: (((u - i.ppx) / i.fx).reshape(-1), ((v - i.ppy) / i.fy).reshape(-1))}
        rx, ry = self._cache[key]
        zz = z.reshape(-1).astype(np.float32) * 0.001
        return points(np.stack([rx * zz, ry * zz, zz], 1), f._t, f._n, f._prof)


class align(_block):
    def __init__(self, to): super().__init__(); self._to = to
    def process(self, fs): return fs


class spatial_filter(_block):
    def process(self, f): return f


class temporal_filter(spatial_filter): pass
class hole_filling_filter(spatial_filter): pass
class threshold_filter(spatial_filter): pass


def rs2_deproject_pixel_to_point(intr, px, depth):
    return [(px[0] - intr.ppx) / intr.fx * depth, (px[1] - intr.ppy) / intr.fy * depth, depth]


def rs2_project_point_to_pixel(intr, p):
    return [p[0] / p[2] * intr.fx + intr.ppx, p[1] / p[2] * intr.fy + intr.ppy]
