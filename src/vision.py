"""
vision.py - Single-thread vision: 2 RealSense + 1 RGB cam → 320x240 frames + 640x480 atlas.
Pre-allocated buffers; one writer thread; readers use .frames, .atlas, .timestamp.
Atlas: UL=rgb1, UR=rgbd1_rgb, LL=rgbd2_rgb, LR=top-down depth (merged pointcloud, 3ch same intensity).
"""

import threading
import time
import numpy as np
import cv2

# Optional: fail gracefully if not on robot
try:
    import pyrealsense2 as rs
    import open3d as o3d
    HAS_RS = True
except ImportError:
    HAS_RS = False

FRAME_W, FRAME_H = 320, 240
ATLAS_W, ATLAS_H = 640, 480
TARGET_FPS = 30
# rgb1 USB webcam: capture at 640x480 then scale down to 320x240 (no rotation; camera physically oriented)
RGB1_CAPTURE_W, RGB1_CAPTURE_H = 640, 480
# Center crosshair for alignment: 2 px at center (159-160, 119-120), white at 30% opacity
CROSSHAIR_CX, CROSSHAIR_CY = 159, 119  # 2 px wide: 159-160, 119-120
CROSSHAIR_OPACITY = 0.3


def _draw_center_crosshair(region, opacity=CROSSHAIR_OPACITY):
    """Draw white center crosshair on 320x240 region; blend at opacity (no copy of region)."""
    r, c = CROSSHAIR_CY, CROSSHAIR_CX
    blend = 1.0 - opacity
    white = 255.0 * opacity
    region[r, :] = (region[r, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[r + 1, :] = (region[r + 1, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c] = (region[:, c].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c + 1] = (region[:, c + 1].astype(np.float32) * blend + white).astype(np.uint8)


def _open_rgb_capture(device_id):
    """Open V4L2 camera by path (e.g. /dev/video12) or int index. Tries path first, then index. Verifies with a test read."""
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


def _set_sensor_opt(sensor, option, value):
    """Set a depth sensor option; skip silently if unsupported by this firmware/device."""
    try:
        sensor.set_option(option, value)
    except Exception:
        pass


def _rs_pipeline(serial):
    """Start RealSense pipeline with depth processing filters (High Density, 30fps)."""
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.rgb8, 30)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)

    sensor = profile.get_device().first_depth_sensor()
    _set_sensor_opt(sensor, rs.option.visual_preset, 3)         # High Density
    _set_sensor_opt(sensor, rs.option.laser_power, 360)
    _set_sensor_opt(sensor, rs.option.enable_auto_exposure, 1)
    _set_sensor_opt(sensor, rs.option.emitter_enabled, 1)
    _set_sensor_opt(sensor, rs.option.depth_units, 0.001)
    _set_sensor_opt(sensor, rs.option.receiver_gain, 16)

    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)

    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, 0.3)
    threshold.set_option(rs.option.max_distance, 3.0)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 1)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    temporal.set_option(rs.option.holes_fill, 3)

    hole_filling = rs.hole_filling_filter(1)

    filters = [decimation, threshold, spatial, temporal, hole_filling]
    depth_scale = sensor.get_depth_scale()
    return pipe, filters, depth_scale


def _rs_frame(pipe, filters, depth_scale):
    """Get color + post-processed depth (no alignment). Returns (color, depth_m, depth_intrinsics)."""
    frames = pipe.wait_for_frames()
    d = frames.get_depth_frame()
    c = frames.get_color_frame()
    if not d or not c:
        return None, None, None
    for filt in filters:
        d = filt.process(d)
    color = np.asarray(c.get_data())
    depth = np.asarray(d.get_data()).astype(np.float32) * depth_scale
    intr = d.profile.as_video_stream_profile().intrinsics
    return color, depth, intr


def _pcd_from_depth(depth, intr, depth_scale, depth_trunc=3.0):
    """Build Open3D pointcloud from depth only (no color needed for top-down view)."""
    w, h = intr.width, intr.height
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, intr.fx, intr.fy, intr.ppx, intr.ppy)
    depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray((depth_clean / depth_scale).astype(np.uint16)))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intrinsic,
        depth_scale=1.0 / depth_scale,
        depth_trunc=depth_trunc,
    )
    return pcd


def _topdown_from_pcd(pcd, out_h=FRAME_H, out_w=FRAME_W, xy_range=2.0):
    """Rasterize merged pointcloud to top-down 320x240 (z = height; value = intensity). 3ch same."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # Map x,y to pixel; value from z (normalize to 0-255)
    px = ((x / xy_range + 0.5) * (out_w - 1)).astype(np.int32)
    py = ((0.5 - y / xy_range) * (out_h - 1)).astype(np.int32)
    valid = (px >= 0) & (px < out_w) & (py >= 0) & (py < out_h)
    px, py, z = px[valid], py[valid], z[valid]
    z_norm = np.clip((z - z.min()) / (z.max() - z.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    img[py, px, 0] = z_norm
    img[py, px, 1] = z_norm
    img[py, px, 2] = z_norm
    return img


class Vision:
    """
    Pre-allocated vision state. One thread calls .update() in a loop; others read .frames, .atlas, .timestamp.
    """

    def __init__(
        self,
        rs1_serial: str,
        rs2_serial: str,
        rgb1_device_id,
        headless: bool = True,
    ):
        self.rs1_serial = rs1_serial
        self.rs2_serial = rs2_serial
        self.rgb1_device_id = rgb1_device_id
        self.headless = headless

        # Pre-allocated outputs (read-only for clients)
        self.frames = [
            np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),  # rgb1
            np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),  # rgbd1
            np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),  # rgbd2
        ]
        self.atlas = np.zeros((ATLAS_H, ATLAS_W, 3), dtype=np.uint8)
        self.timestamp = 0.0
        self._lock = threading.Lock()

        self._running = False
        self._thread = None
        self._pipe1 = None
        self._pipe2 = None
        self._filters1 = []
        self._filters2 = []
        self._depth_scale1 = 0.001
        self._depth_scale2 = 0.001
        self._rgb1_cap = None
        self._have_rs = HAS_RS

    def start(self):
        """Start capture thread (30 fps). RealSense first (by serial), then rgb1 - so we don't steal a RS device."""
        if self._running:
            return
        if not self._have_rs:
            print("vision: pyrealsense2/open3d not available; running stub")
            self._running = True
            self._thread = threading.Thread(target=self._stub_loop, daemon=True)
            self._thread.start()
            return
        try:
            if self.rs1_serial:
                self._pipe1, self._filters1, self._depth_scale1 = _rs_pipeline(self.rs1_serial)
            if self.rs2_serial:
                self._pipe2, self._filters2, self._depth_scale2 = _rs_pipeline(self.rs2_serial)
        except Exception as e:
            print(f"vision: RealSense init failed: {e}")
            self._have_rs = False
            self._running = True
            self._thread = threading.Thread(target=self._stub_loop, daemon=True)
            self._thread.start()
            return
        # Open rgb1 after RealSense so we only touch the non-RS camera (e.g. USB webcam at video0)
        self._rgb1_cap = _open_rgb_capture(self.rgb1_device_id)
        if self._rgb1_cap is not None and self._rgb1_cap.isOpened():
            # 640x480 capture, scale down to 320x240 (no rotation; camera physically oriented)
            self._rgb1_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._rgb1_cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB1_CAPTURE_W)
            self._rgb1_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB1_CAPTURE_H)
            self._rgb1_cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            try:
                self._rgb1_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            print("vision: rgb1 camera opened (640x480, MJPG → scale to 320x240)")
        else:
            if self._rgb1_cap is not None:
                self._rgb1_cap.release()
            self._rgb1_cap = None
            print("vision: rgb1 camera not opened (use debug_camera.py to find correct device; avoid RealSense nodes)")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("vision: capture thread started (30 fps)")

    def _stub_loop(self):
        """No cameras: just update timestamp and black frames."""
        interval = 1.0 / TARGET_FPS
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                self.timestamp = time.time()
            time.sleep(max(0, interval - (time.monotonic() - t0)))

    def _capture_loop(self):
        interval = 1.0 / TARGET_FPS
        while self._running:
            t0 = time.monotonic()
            # RGB1: 640x480 capture → scale to 320x240 (same aspect, no rotation/crop)
            rgb1 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            if self._rgb1_cap and self._rgb1_cap.isOpened():
                ret, f = self._rgb1_cap.read()
                if ret and f is not None:
                    if f.ndim == 2:
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    elif f.shape[2] == 4:
                        f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
                    if f.shape[1] != FRAME_W or f.shape[0] != FRAME_H:
                        f = cv2.resize(f, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
                    f = f[::-1, :]
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    rgb1 = f

            # RS1 (top-right in atlas): color 320x240; flip 180 via numpy view
            rgbd1 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            pcd1 = None
            if self._pipe1:
                c1, d1, i1 = _rs_frame(self._pipe1, self._filters1, self._depth_scale1)
                if c1 is not None:
                    rgbd1 = c1[::-1, ::-1]
                    if d1 is not None and i1 is not None:
                        pcd1 = _pcd_from_depth(d1, i1, self._depth_scale1)

            # RS2
            rgbd2 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            pcd2 = None
            if self._pipe2:
                c2, d2, i2 = _rs_frame(self._pipe2, self._filters2, self._depth_scale2)
                if c2 is not None:
                    rgbd2 = c2
                    if d2 is not None and i2 is not None:
                        pcd2 = _pcd_from_depth(d2, i2, self._depth_scale2)

            # Merge pointclouds (simple concat; no transform for minimal v2)
            merged = o3d.geometry.PointCloud()
            if pcd1 is not None:
                merged += pcd1
            if pcd2 is not None:
                merged += pcd2
            topdown = _topdown_from_pcd(merged) if len(merged.points) > 0 else np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

            # Update shared state in one go (rgbd1 already 180-flipped via view)
            with self._lock:
                self.frames[0][:] = rgb1
                self.frames[1][:] = rgbd1
                self.frames[2][:] = rgbd2
                # Atlas: UL=rgb1, UR=rgbd1 (180°), LL=rgbd2, LR=topdown
                self.atlas[0:FRAME_H, 0:FRAME_W] = rgb1
                self.atlas[0:FRAME_H, FRAME_W:ATLAS_W] = rgbd1
                self.atlas[FRAME_H:ATLAS_H, 0:FRAME_W] = rgbd2
                self.atlas[FRAME_H:ATLAS_H, FRAME_W:ATLAS_W] = topdown
                for yo, xo in [(0, 0), (0, FRAME_W), (FRAME_H, 0), (FRAME_H, FRAME_W)]:
                    _draw_center_crosshair(self.atlas[yo : yo + FRAME_H, xo : xo + FRAME_W])
                self.timestamp = time.time()

            elapsed = time.monotonic() - t0
            time.sleep(max(0, interval - elapsed))

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._pipe1:
            try:
                self._pipe1.stop()
            except Exception:
                pass
        if self._pipe2:
            try:
                self._pipe2.stop()
            except Exception:
                pass
        if self._rgb1_cap:
            self._rgb1_cap.release()
        print("vision: stopped")

    def read(self):
        """Return (frames, atlas, timestamp) under lock (safe copy)."""
        with self._lock:
            return (
                [f.copy() for f in self.frames],
                self.atlas.copy(),
                self.timestamp,
            )
