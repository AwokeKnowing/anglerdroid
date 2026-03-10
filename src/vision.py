"""vision.py – Process camera frames into atlas + obstacle map.
Camera hardware lives in cameras.py. Depth processing is pure numpy
(no Open3D), replicating reference/workingvisionobs3ms.py pipeline.

Atlas layout (640x480): UL=rgb1, UR=rgbd1_rgb, LL=rgbd2_rgb, LR=topdown obstacle map.
"""

import threading
import time
import numpy as np
import cv2

from cameras import RSCamera, WebCam, HAS_RS, FRAME_W, FRAME_H

ATLAS_W, ATLAS_H = 640, 480
TARGET_FPS = 30

CROSSHAIR_CX, CROSSHAIR_CY = 159, 119
CROSSHAIR_OPACITY = 0.3

# --- Depth-to-topdown parameters (from reference/workingvisionobs3ms.py) ---
# View transform: z-shift puts virtual camera 1m behind origin
VIEW_Z_OFFSET = np.float32(1.60)
NEAR_CLIP = np.float32(0.03)

# Depth-to-intensity formula: (255 - ((z + OFFSET) * SCALE).uint8) - BIAS
# Intentional uint8 wrapping gives a non-linear distance mapping that
# Otsu can cleanly binarise into obstacles vs free space.
DEPTH_OFFSET = np.float32(0.25)
DEPTH_SCALE = np.float32(400.0)
DEPTH_BIAS = np.uint8(48)

# RS1 (forward camera) extrinsic offset.
# Camera is ~10cm higher than the original calibration baseline.
# In RealSense camera coords Y points down, so camera-higher = negative Y offset.
# >>> Change RS1_EXTRINSIC_Y to compensate (e.g. -0.10 for 10cm higher). <<<
RS1_EXTRINSIC_Y = 0.0  # meters


def _draw_center_crosshair(region, opacity=CROSSHAIR_OPACITY):
    r, c = CROSSHAIR_CY, CROSSHAIR_CX
    blend = 1.0 - opacity
    white = 255.0 * opacity
    region[r, :] = (region[r, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[r + 1, :] = (region[r + 1, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c] = (region[:, c].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c + 1] = (region[:, c + 1].astype(np.float32) * blend + white).astype(np.uint8)


def depth_topdown(verts, out_h=FRAME_H, out_w=FRAME_W, y_offset=0.0):
    """Project decimated RS pointcloud to obstacle map via pure numpy.

    Replicates the exact pipeline from reference/workingvisionobs3ms.py:
      1. Perspective projection (isotropic focal = out_h)
      2. Depth-to-intensity with intentional uint8 wrapping
      3. Otsu threshold -> binary obstacle image

    verts:    Nx3 float32 from rs.pointcloud (camera coords, meters).
    y_offset: vertical extrinsic compensation (meters, camera-Y-down).
    Returns:  (out_h, out_w, 3) uint8.
    """
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    if len(verts) == 0:
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    v = verts.copy()
    if y_offset != 0.0:
        v[:, 1] += np.float32(y_offset)
    v[:, 2] += VIEW_Z_OFFSET

    # Perspective projection (matches reference: focal length = out_h for both axes)
    aspect = np.float32(out_h) / np.float32(out_w)
    scale = np.float32([out_w * aspect, out_h])
    center = np.float32([out_w * 0.5, out_h * 0.5])
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :2] / v[:, 2:3] * scale + center

    proj[v[:, 2] < NEAR_CLIP] = np.nan

    # uint32 cast: NaN -> 0, negatives wrap to huge values (filtered by bounds)
    j, i = proj.astype(np.uint32).T
    m = (i < np.uint32(out_h)) & (j < np.uint32(out_w))

    # Depth-to-intensity from original z (uint8 wrapping is intentional)
    z = verts[m, 2]
    vals = np.uint8(255) - ((z + DEPTH_OFFSET) * DEPTH_SCALE).astype(np.uint8) - DEPTH_BIAS
    vals[vals == 255] = 0

    out[i[m], j[m]] = vals

    _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


class Vision:
    """Pre-allocated vision state. One capture thread; readers use .frames, .atlas, .timestamp."""

    def __init__(self, rs1_serial, rs2_serial, rgb1_device_id, headless=True):
        self.rs1_serial = rs1_serial
        self.rs2_serial = rs2_serial
        self.rgb1_device_id = rgb1_device_id

        self.frames = [
            np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),
            np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),
            np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),
        ]
        self.atlas = np.zeros((ATLAS_H, ATLAS_W, 3), dtype=np.uint8)
        self.timestamp = 0.0
        self._lock = threading.Lock()

        self._running = False
        self._thread = None
        self._rs1 = None
        self._rs2 = None
        self._webcam = None

    def start(self):
        if self._running:
            return

        if not HAS_RS:
            print("vision: pyrealsense2 not available; running stub")
            self._running = True
            self._thread = threading.Thread(target=self._stub_loop, daemon=True)
            self._thread.start()
            return

        try:
            if self.rs1_serial:
                self._rs1 = RSCamera(self.rs1_serial, compute_pointcloud=True)
            if self.rs2_serial:
                self._rs2 = RSCamera(self.rs2_serial, compute_pointcloud=False)
        except Exception as e:
            print("vision: RealSense init failed: %s" % e)
            self._running = True
            self._thread = threading.Thread(target=self._stub_loop, daemon=True)
            self._thread.start()
            return

        self._webcam = WebCam(self.rgb1_device_id)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("vision: capture thread started")

    def _stub_loop(self):
        interval = 1.0 / TARGET_FPS
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                self.timestamp = time.time()
            time.sleep(max(0, interval - (time.monotonic() - t0)))

    def _capture_loop(self):
        topdown = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        black = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        while self._running:
            # Grab all cameras (each blocks until frame ready -> natural 30fps pacing)
            if self._webcam:
                self._webcam.grab()
            if self._rs1:
                self._rs1.grab()
            if self._rs2:
                self._rs2.grab()

            # Depth -> obstacle map (every frame; <1ms on 6360 decimated points)
            if self._rs1 and self._rs1.ok and self._rs1.verts is not None:
                topdown = depth_topdown(self._rs1.verts, y_offset=RS1_EXTRINSIC_Y)

            rgb1 = self._webcam.color if (self._webcam and self._webcam.ok) else black
            rgbd1 = self._rs1.color[::-1, ::-1] if (self._rs1 and self._rs1.ok) else black
            rgbd2 = self._rs2.color if (self._rs2 and self._rs2.ok) else black

            with self._lock:
                self.frames[0][:] = rgb1
                self.frames[1][:] = rgbd1
                self.frames[2][:] = rgbd2
                self.atlas[0:FRAME_H, 0:FRAME_W] = rgb1
                self.atlas[0:FRAME_H, FRAME_W:ATLAS_W] = rgbd1
                self.atlas[FRAME_H:ATLAS_H, 0:FRAME_W] = rgbd2
                self.atlas[FRAME_H:ATLAS_H, FRAME_W:ATLAS_W] = topdown
                for yo, xo in [(0, 0), (0, FRAME_W), (FRAME_H, 0), (FRAME_H, FRAME_W)]:
                    _draw_center_crosshair(self.atlas[yo:yo + FRAME_H, xo:xo + FRAME_W])
                self.timestamp = time.time()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._rs1:
            self._rs1.stop()
        if self._rs2:
            self._rs2.stop()
        if self._webcam:
            self._webcam.stop()
        print("vision: stopped")

    def read_atlas(self):
        """Return (atlas_copy, timestamp) -- lightweight read for main loop."""
        with self._lock:
            return self.atlas.copy(), self.timestamp

    def read(self):
        """Return (frames, atlas, timestamp) under lock (safe copy)."""
        with self._lock:
            return (
                [f.copy() for f in self.frames],
                self.atlas.copy(),
                self.timestamp,
            )
