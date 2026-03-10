"""vision.py – Process camera frames into atlas + obstacle map.
Camera hardware lives in cameras.py. Depth processing is pure numpy.

RS2 (forward camera, 944622074292) pointcloud is rotated -64.4° around X
to produce a bird's-eye obstacle map via orthographic projection.
Replicates reference/firstmergedvision-working2cam.py pipeline.

Atlas layout (640x480): UL=rgb1, UR=rgbd1_rgb, LL=rgbd2_rgb, LR=topdown obstacle map.
"""

import math
import threading
import time
import numpy as np
import cv2

from cameras import RSCamera, WebCam, HAS_RS, FRAME_W, FRAME_H

ATLAS_W, ATLAS_H = 640, 480
TARGET_FPS = 30

CROSSHAIR_CX, CROSSHAIR_CY = 159, 119
CROSSHAIR_OPACITY = 0.3

# --- Forward camera (RS2) → bird's-eye obstacle map ---
# Rotation: pitch = 25.6° - 90° = -64.4° (camera mounting angle compensation)
# Converts forward-looking depth into top-down view.
FW_PITCH_DEG = 25.6 - 90.0  # -64.4°
_fw_pitch_rad = math.radians(FW_PITCH_DEG)
_fw_R, _ = cv2.Rodrigues(np.float64([_fw_pitch_rad, 0, 0]))
FW_ROTATION = _fw_R.astype(np.float32)

# View transform params (from reference/firstmergedvision-working2cam.py)
FW_PIVOT = np.array([0.0, -1.0, 0.02], dtype=np.float32)
FW_TRANSLATION = np.array([0.0, -1.0, 0.0], dtype=np.float32)

# Orthographic projection: 1px = FW_PX_SIZE meters (0.01 = 1cm)
FW_PX_SIZE = np.float32(0.01)

# Height clip in rotated frame (meters). Controls max forward distance visible.
FW_HEIGHT_CLIP = np.float32(1.30)

# Morphological close kernel (pre-built)
_fw_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

# RS2 (forward camera) extrinsic Y offset.
# Camera is ~10cm higher than original calibration.
# In RealSense coords Y points down, so camera-higher = negative Y offset.
# >>> Change RS2_EXTRINSIC_Y to compensate (e.g. -0.10 for 10cm higher). <<<
RS2_EXTRINSIC_Y = 0.0  # meters


def _draw_center_crosshair(region, opacity=CROSSHAIR_OPACITY):
    r, c = CROSSHAIR_CY, CROSSHAIR_CX
    blend = 1.0 - opacity
    white = 255.0 * opacity
    region[r, :] = (region[r, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[r + 1, :] = (region[r + 1, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c] = (region[:, c].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c + 1] = (region[:, c + 1].astype(np.float32) * blend + white).astype(np.uint8)


def depth_topdown_forward(verts, out_h=FRAME_H, out_w=FRAME_W, y_offset=0.0):
    """Project forward RS camera pointcloud to bird's-eye obstacle map.

    Pipeline from reference/firstmergedvision-working2cam.py:
      1. View transform: rotate forward camera by -64.4° pitch → top-down
      2. Height clip (z < 1.30 in rotated frame)
      3. Orthographic projection at 1px/cm
      4. All valid points → 255
      5. Morphological close to fill gaps

    verts:    Nx3 float32 from rs.pointcloud (forward camera, meters).
    y_offset: vertical extrinsic offset (meters, camera Y-down).
    Returns:  (out_h, out_w, 3) uint8.
    """
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    if len(verts) == 0:
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    v = verts.copy()
    if y_offset != 0.0:
        v[:, 1] += np.float32(y_offset)

    # View transform: rotate forward → top-down (matches reference exactly)
    v = np.dot(v - FW_PIVOT, FW_ROTATION) + FW_PIVOT - FW_TRANSLATION

    # Height clip (z in rotated frame corresponds to real-world height)
    v = v[v[:, 2] < FW_HEIGHT_CLIP]
    if len(v) == 0:
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    # Orthographic projection (1px = 1cm, camera origin near bottom of image)
    scale = np.float32(1.0 / FW_PX_SIZE)
    proj = v[:, :2] * scale + np.float32([out_w / 2.0, out_h / 2.0 + 1.0 * scale])

    j, i = proj.astype(np.uint32).T
    m = (i < np.uint32(out_h)) & (j < np.uint32(out_w))

    out[i[m], j[m]] = 255

    cv2.morphologyEx(out, cv2.MORPH_CLOSE, _fw_kernel, iterations=2, dst=out)
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
                self._rs1 = RSCamera(self.rs1_serial, compute_pointcloud=False)
            if self.rs2_serial:
                self._rs2 = RSCamera(self.rs2_serial, compute_pointcloud=True)
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
            if self._webcam:
                self._webcam.grab()
            if self._rs1:
                self._rs1.grab()
            if self._rs2:
                self._rs2.grab()

            # Forward camera depth → bird's-eye obstacle map
            if self._rs2 and self._rs2.ok and self._rs2.verts is not None:
                topdown = depth_topdown_forward(self._rs2.verts, y_offset=RS2_EXTRINSIC_Y)

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
