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


def _open_rgb_capture(device_id):
    """Open V4L2 camera by path (e.g. /dev/video12) or int index. Tries path first, then index if path looks like /dev/videoN."""
    import re
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap
    cap.release()
    if isinstance(device_id, str):
        m = re.search(r"video(\d+)$", device_id)
        if m:
            idx = int(m.group(1))
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()
    return None


def _rs_pipeline(serial: str):
    """Start RealSense pipeline: depth 848x480, color 320x240."""
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
    cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.rgb8, 60)
    pipe = rs.pipeline()
    pipe.start(cfg)
    return pipe


def _rs_align(profile):
    align_to = rs.stream.color
    return rs.align(align_to)


def _rs_frame(pipe, align, depth_scale, clip_m: float = 3.0):
    """Get one aligned color + depth; return color (320x240 RGB), depth (np), intrinsics for pointcloud."""
    frames = pipe.wait_for_frames()
    aligned = align.process(frames)
    d = aligned.get_depth_frame()
    c = aligned.get_color_frame()
    if not d or not c:
        return None, None, None
    color = np.asarray(c.get_data())
    depth = np.asarray(d.get_data()).astype(np.float32) * depth_scale
    depth[depth <= 0] = np.nan
    depth[depth > clip_m] = np.nan
    intr = c.profile.as_video_stream_profile().intrinsics
    return color, depth, intr


def _pcd_from_rgbd(color, depth, intr, depth_scale, depth_trunc):
    """Build Open3D pointcloud from 320x240 color+depth."""
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    intrinsic = o3d.camera.PinholeCameraIntrinsic(FRAME_W, FRAME_H, fx, fy, cx, cy)
    depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray((depth_clean / depth_scale).astype(np.uint16)))
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(color))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0 / depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic, np.eye(4)
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
        self._align1 = None
        self._align2 = None
        self._depth_scale1 = 0.001
        self._depth_scale2 = 0.001
        self._rgb1_cap = None
        self._have_rs = HAS_RS

    def start(self):
        """Start capture thread (30 fps)."""
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
                self._pipe1 = _rs_pipeline(self.rs1_serial)
                self._align1 = _rs_align(self._pipe1.get_active_profile())
                self._depth_scale1 = self._pipe1.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
            if self.rs2_serial:
                self._pipe2 = _rs_pipeline(self.rs2_serial)
                self._align2 = _rs_align(self._pipe2.get_active_profile())
                self._depth_scale2 = self._pipe2.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        except Exception as e:
            print(f"vision: RealSense init failed: {e}")
            self._have_rs = False
            self._running = True
            self._thread = threading.Thread(target=self._stub_loop, daemon=True)
            self._thread.start()
            return
        self._rgb1_cap = _open_rgb_capture(self.rgb1_device_id)
        if self._rgb1_cap is not None and self._rgb1_cap.isOpened():
            self._rgb1_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._rgb1_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            if self._rgb1_cap is not None:
                self._rgb1_cap.release()
            self._rgb1_cap = None
            print("vision: rgb1 camera not opened (try debug_camera.py to test device)")
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
        clip_m = 3.0
        while self._running:
            t0 = time.monotonic()
            # RGB1
            rgb1 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            if self._rgb1_cap and self._rgb1_cap.isOpened():
                ret, f = self._rgb1_cap.read()
                if ret:
                    f = cv2.resize(f, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
                    if f.ndim == 2:
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
                    elif f.shape[2] == 4:
                        f = cv2.cvtColor(f, cv2.COLOR_BGRA2RGB)
                    else:
                        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    rgb1 = f

            # RS1
            rgbd1 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            pcd1 = None
            if self._pipe1 and self._align1:
                c1, d1, i1 = _rs_frame(self._pipe1, self._align1, self._depth_scale1, clip_m)
                if c1 is not None:
                    rgbd1 = c1
                    if d1 is not None and i1 is not None:
                        pcd1 = _pcd_from_rgbd(c1, d1, i1, self._depth_scale1, clip_m)

            # RS2
            rgbd2 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            pcd2 = None
            if self._pipe2 and self._align2:
                c2, d2, i2 = _rs_frame(self._pipe2, self._align2, self._depth_scale2, clip_m)
                if c2 is not None:
                    rgbd2 = c2
                    if d2 is not None and i2 is not None:
                        pcd2 = _pcd_from_rgbd(c2, d2, i2, self._depth_scale2, clip_m)

            # Merge pointclouds (simple concat; no transform for minimal v2)
            merged = o3d.geometry.PointCloud()
            if pcd1 is not None:
                merged += pcd1
            if pcd2 is not None:
                merged += pcd2
            topdown = _topdown_from_pcd(merged) if len(merged.points) > 0 else np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

            # Update shared state in one go
            with self._lock:
                self.frames[0][:] = rgb1
                self.frames[1][:] = rgbd1
                self.frames[2][:] = rgbd2
                # Atlas: UL=rgb1, UR=rgbd1, LL=rgbd2, LR=topdown
                self.atlas[0:FRAME_H, 0:FRAME_W] = rgb1
                self.atlas[0:FRAME_H, FRAME_W:ATLAS_W] = rgbd1
                self.atlas[FRAME_H:ATLAS_H, 0:FRAME_W] = rgbd2
                self.atlas[FRAME_H:ATLAS_H, FRAME_W:ATLAS_W] = topdown
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
