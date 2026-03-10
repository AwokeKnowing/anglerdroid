"""vision.py – Process camera frames into atlas + obstacle map.
Camera hardware lives in cameras.py. Depth processing is pure numpy.

RS1 (top-down camera) pointcloud → perspective depth map (already top-down).
RS2 (forward camera) pointcloud → rotated -64.4° around X → bird's-eye view.
Both are combined into the topdown obstacle map quadrant.

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

# --- RS1 (top-down camera) depth params (from reference/workingvisionobs3ms.py) ---
VIEW_Z_OFFSET = np.float32(1.80)
NEAR_CLIP = np.float32(0.03)
DEPTH_OFFSET = np.float32(0.25)
DEPTH_SCALE = np.float32(400.0)
DEPTH_BIAS = np.uint8(48)

# --- RS2 (forward camera) → bird's-eye rotation ---
# Pitch = 25.6° - 90° = -64.4° (camera mounting angle compensation)
FW_PITCH_DEG = 25.6 - 90.0
_fw_pitch_rad = math.radians(FW_PITCH_DEG)
_fw_R, _ = cv2.Rodrigues(np.float64([_fw_pitch_rad, 0, 0]))
FW_ROTATION = _fw_R.astype(np.float32)

# View transform params (from reference/firstmergedvision-working2cam.py)
FW_PIVOT = np.array([0.0, -1.0, 0.02], dtype=np.float32)
FW_TRANSLATION = np.array([0.0, -1.0, 0.0], dtype=np.float32)
FW_PX_SIZE = np.float32(0.01)       # 1px = 1cm
FW_HEIGHT_CLIP = np.float32(1.30)   # max height in rotated frame
_fw_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

# RS2 (forward camera) extrinsic Y offset (~10cm higher than calibration).
# Camera Y-down: camera-higher = negative Y offset.
# >>> Change to e.g. -0.10 when ready to compensate. <<<
RS2_EXTRINSIC_Y = 0.0

# Alignment offsets (pixels, + = shift right). Adjusted via sliders in main.py.
TD_X_OFFSET = -75
FW_X_OFFSET = 65


def _blit_x(dst, src, dx):
    """Copy src into 2D dst with horizontal pixel offset dx. Clipped, no wrap."""
    w = dst.shape[1]
    if abs(dx) >= w:
        return
    if dx > 0:
        dst[:, dx:] = src[:, :w - dx]
    elif dx < 0:
        dst[:, :w + dx] = src[:, -dx:]
    else:
        np.copyto(dst, src)


def _draw_center_crosshair(region, opacity=CROSSHAIR_OPACITY):
    r, c = CROSSHAIR_CY, CROSSHAIR_CX
    blend = 1.0 - opacity
    white = 255.0 * opacity
    region[r, :] = (region[r, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[r + 1, :] = (region[r + 1, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c] = (region[:, c].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c + 1] = (region[:, c + 1].astype(np.float32) * blend + white).astype(np.uint8)


def depth_topdown(verts, out_h=FRAME_H, out_w=FRAME_W):
    """RS1 (top-down camera) pointcloud → obstacle map via perspective projection.
    Replicates reference/workingvisionobs3ms.py. Returns single-channel uint8."""
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    if len(verts) == 0:
        return out

    v = verts.copy()
    v[:, 2] += VIEW_Z_OFFSET

    aspect = np.float32(out_h) / np.float32(out_w)
    scale = np.float32([out_w * aspect, out_h])
    center = np.float32([out_w * 0.5, out_h * 0.5])
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :2] / v[:, 2:3] * scale + center

    proj[v[:, 2] < NEAR_CLIP] = np.nan

    j, i = proj.astype(np.uint32).T
    m = (i < np.uint32(out_h)) & (j < np.uint32(out_w))

    z = verts[m, 2]
    vals = np.uint8(255) - ((z + DEPTH_OFFSET) * DEPTH_SCALE).astype(np.uint8) - DEPTH_BIAS
    vals[vals == 255] = 0

    out[i[m], j[m]] = vals
    _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return out


def depth_topdown_forward(verts, out_h=FRAME_H, out_w=FRAME_W, y_offset=0.0, debug=True):
    """RS2 (forward camera) pointcloud → rotated bird's-eye obstacle map.
    Replicates reference/firstmergedvision-working2cam.py. Returns single-channel uint8."""
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    if len(verts) == 0:
        if debug:
            print("fw: no verts")
        return out

    v = verts.copy()
    if y_offset != 0.0:
        v[:, 1] += np.float32(y_offset)

    # Debug: raw pointcloud before rotation (perspective view, same as RS1 style)
    if debug:
        dbg_raw = np.zeros((out_h, out_w), dtype=np.uint8)
        vr = v.copy()
        vr[:, 2] += np.float32(1.0)
        asp = np.float32(out_h) / np.float32(out_w)
        with np.errstate(divide='ignore', invalid='ignore'):
            rp = vr[:, :2] / vr[:, 2:3] * np.float32([out_w * asp, out_h]) + np.float32([out_w * 0.5, out_h * 0.5])
        rj, ri = rp.astype(np.uint32).T
        rm = (ri < np.uint32(out_h)) & (rj < np.uint32(out_w))
        dbg_raw[ri[rm], rj[rm]] = 255
        cv2.imshow("fw_raw_persp", dbg_raw)

    v = np.dot(v - FW_PIVOT, FW_ROTATION) + FW_PIVOT - FW_TRANSLATION

    # Debug: after rotation, before height clip
    if debug:
        n_total = len(v)
        n_pass = int(np.sum(v[:, 2] < FW_HEIGHT_CLIP))
        print("fw: %d verts, %d pass clip (z<%.2f), z range [%.3f, %.3f]" % (
            n_total, n_pass, float(FW_HEIGHT_CLIP),
            float(v[:, 2].min()) if n_total > 0 else 0,
            float(v[:, 2].max()) if n_total > 0 else 0))

        dbg_noclip = np.zeros((out_h, out_w), dtype=np.uint8)
        sc = np.float32(1.0 / FW_PX_SIZE)
        pr = v[:, :2] * sc + np.float32([out_w / 2.0, out_h / 2.0 + 1.0 * sc])
        dj, di = pr.astype(np.uint32).T
        dm = (di < np.uint32(out_h)) & (dj < np.uint32(out_w))
        dbg_noclip[di[dm], dj[dm]] = 255
        cv2.imshow("fw_rotated_noclip", dbg_noclip)

    v = v[v[:, 2] < FW_HEIGHT_CLIP]
    if len(v) == 0:
        if debug:
            print("fw: all clipped!")
        return out

    scale = np.float32(1.0 / FW_PX_SIZE)
    proj = v[:, :2] * scale + np.float32([out_w / 2.0, out_h / 2.0 + 1.0 * scale])

    j, i = proj.astype(np.uint32).T
    m = (i < np.uint32(out_h)) & (j < np.uint32(out_w))

    if debug:
        n_inbounds = int(np.sum(m))
        print("fw: %d after clip, %d in bounds" % (len(v), n_inbounds))

    out[i[m], j[m]] = 255

    if debug:
        cv2.imshow("fw_before_morph", out.copy())

    cv2.morphologyEx(out, cv2.MORPH_CLOSE, _fw_kernel, iterations=2, dst=out)

    if debug:
        cv2.imshow("fw_final", out.copy())

    return out


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
        black = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        while self._running:
            if self._webcam:
                self._webcam.grab()
            if self._rs1:
                self._rs1.grab()
            if self._rs2:
                self._rs2.grab()

            # RS1 top-down depth → rotate 180° (hflip+vflip, view — no copy)
            td1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            if self._rs1 and self._rs1.ok and self._rs1.verts is not None:
                td1 = depth_topdown(self._rs1.verts)
            td1 = td1[::-1, ::-1]

            # RS2 forward depth → generate at (W,H), then CW 90° → (H,W)
            # np.rot90(k=-1) = transpose + flip = view, no multiply
            td2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
            if self._rs2 and self._rs2.ok and self._rs2.verts is not None:
                td2 = depth_topdown_forward(self._rs2.verts,
                                            out_h=FRAME_W, out_w=FRAME_H,
                                            y_offset=RS2_EXTRINSIC_Y, debug=True)
            td2 = np.rot90(td2, k=-1)

            # Blit into RGB channels with x-offsets (direct slice, no intermediate copy)
            # Overlap → yellow, td1-only → green, td2-only → red
            topdown = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            _blit_x(topdown[:, :, 0], td2, int(FW_X_OFFSET))   # R = forward
            _blit_x(topdown[:, :, 1], td1, int(TD_X_OFFSET))   # G = topdown

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
