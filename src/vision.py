"""vision.py – Process camera frames into atlas + obstacle map.
Camera hardware lives in cameras.py. Depth processing is pure numpy.

RS1 (top-down camera) pointcloud → orthographic depth map (already top-down).
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

# Robot footprint on costmap (pixels). Robot faces RIGHT.
ROBOT_W = 30        # front-back (x direction)  — locked
ROBOT_H = 42        # side-to-side (y direction) — locked
ROBOT_CX_OFF = -78  # x offset from crosshair center — locked

# --- RS1 (top-down camera) depth params ---
TD_PX_SIZE = np.float32(0.010)   # 1px = 10mm (orthographic, same as FW)
TD_FLOOR_CLIP = np.float32(0.91) # reject floor (farther than this Z). Fixed.

# --- RS2 (forward camera) → bird's-eye rotation ---
# Pitch = 25.6° - 90° = -64.4° (camera mounting angle compensation)
FW_PITCH_DEG = 25.6 - 90.0
_fw_pitch_rad = math.radians(FW_PITCH_DEG)
_fw_R, _ = cv2.Rodrigues(np.float64([_fw_pitch_rad, 0, 0]))
FW_ROTATION = _fw_R.astype(np.float32)

# View transform params (from reference/firstmergedvision-working2cam.py)
FW_PIVOT = np.array([0.0, -1.0, 0.02], dtype=np.float32)
FW_TRANSLATION = np.array([0.0, -1.0, 0.0], dtype=np.float32)
FW_PX_SIZE = np.float32(0.010)      # 1px = 1cm (fixed)
FW_HEIGHT_CLIP = np.float32(1.30)   # max height in rotated frame (fixed)
_fw_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
_known_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# RS2 (forward camera) extrinsic Y offset (~10cm higher than calibration).
# Camera Y-down: camera-higher = negative Y offset.
# >>> Change to e.g. -0.10 when ready to compensate. <<<
RS2_EXTRINSIC_Y = 0.0

# Alignment offsets (pixels). TD_X_OFFSET adjustable via slider; FW_X locked to TD + delta.
TD_X_OFFSET = -75
FW_TD_X_DELTA = 132             # fw_x = td_x + this
FW_Y_OFFSET = -1                # fixed


def _blit(dst, src, dx, dy=0):
    """Copy src into 2D dst with pixel offset (dx, dy). +dy = down, -dy = up. Clipped, no wrap."""
    h, w = dst.shape[:2]
    if dy >= 0:
        sr0, sr1, dr0, dr1 = 0, h - dy, dy, h
    else:
        sr0, sr1, dr0, dr1 = -dy, h, 0, h + dy
    if dx >= 0:
        sc0, sc1, dc0, dc1 = 0, w - dx, dx, w
    else:
        sc0, sc1, dc0, dc1 = -dx, w, 0, w + dx
    if sr0 >= sr1 or sc0 >= sc1:
        return
    dst[dr0:dr1, dc0:dc1] = src[sr0:sr1, sc0:sc1]


def _draw_center_crosshair(region, opacity=CROSSHAIR_OPACITY):
    r, c = CROSSHAIR_CY, CROSSHAIR_CX
    blend = 1.0 - opacity
    white = 255.0 * opacity
    region[r, :] = (region[r, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[r + 1, :] = (region[r + 1, :].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c] = (region[:, c].astype(np.float32) * blend + white).astype(np.uint8)
    region[:, c + 1] = (region[:, c + 1].astype(np.float32) * blend + white).astype(np.uint8)


def depth_topdown(verts, out_h=FRAME_H, out_w=FRAME_W):
    """RS1 (top-down camera) pointcloud → (obstacles, known) via orthographic projection.
    Floor is clipped by Z before projection. Remaining points = obstacles.
    Returns two single-channel uint8 arrays of shape (out_h, out_w)."""
    obs = np.zeros((out_h, out_w), dtype=np.uint8)
    known = np.zeros((out_h, out_w), dtype=np.uint8)
    if len(verts) == 0:
        return obs, known

    z = verts[:, 2]
    valid = z > 0
    obstacle = valid & (z < TD_FLOOR_CLIP)

    scale = np.float32(1.0 / TD_PX_SIZE)
    center = np.float32([out_w * 0.5, out_h * 0.5])

    # Known mask: all valid points (including floor)
    v_valid = verts[valid]
    if len(v_valid) > 0:
        p_all = v_valid[:, :2] * scale + center
        with np.errstate(invalid='ignore'):
            ja, ia = p_all.astype(np.uint32).T
        ma = (ia < np.uint32(out_h)) & (ja < np.uint32(out_w))
        known[ia[ma], ja[ma]] = 255

    # Obstacle mask: only points closer than floor
    v_obs = verts[obstacle]
    if len(v_obs) > 0:
        p_obs = v_obs[:, :2] * scale + center
        with np.errstate(invalid='ignore'):
            jo, io = p_obs.astype(np.uint32).T
        mo = (io < np.uint32(out_h)) & (jo < np.uint32(out_w))
        obs[io[mo], jo[mo]] = 255

    return obs, known


def depth_topdown_forward(verts, out_h=FRAME_H, out_w=FRAME_W, y_offset=0.0):
    """RS2 (forward camera) pointcloud → (obstacles, known) via rotated bird's-eye.
    Returns two single-channel uint8 arrays of shape (out_h, out_w)."""
    obs = np.zeros((out_h, out_w), dtype=np.uint8)
    known = np.zeros((out_h, out_w), dtype=np.uint8)
    if len(verts) == 0:
        return obs, known

    v = verts.copy()
    if y_offset != 0.0:
        v[:, 1] += np.float32(y_offset)

    # Debug: raw pointcloud before rotation (perspective view)
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

    # Project ALL rotated points → known mask (camera footprint)
    scale = np.float32(1.0 / FW_PX_SIZE)
    proj = v[:, :2] * scale + np.float32([out_w / 2.0, out_h / 2.0 + 1.0 * scale])
    j, i = proj.astype(np.uint32).T
    m_all = (i < np.uint32(out_h)) & (j < np.uint32(out_w))
    known[i[m_all], j[m_all]] = 255

    # Debug: all rotated points (before height clip)
    cv2.imshow("fw_rotated_noclip", known.copy())

    # Height clip → obstacle points only
    m_obs = m_all & (v[:, 2] < FW_HEIGHT_CLIP)
    obs[i[m_obs], j[m_obs]] = 255

    cv2.imshow("fw_before_morph", obs.copy())

    cv2.morphologyEx(obs, cv2.MORPH_CLOSE, _fw_kernel, iterations=2, dst=obs)

    cv2.imshow("fw_final", obs.copy())

    return obs, known


_PILLOW_RADIUS = 30  # distance field radius from robot edge (pixels = cm)
_obs_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_robot_inv = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)


def _build_costmap(obs_combined, known_combined):
    """RGB costmap: white=free, grey=unknown, dark=obstacles.
    Distance field from robot colours obstacles red (close) → yellow (far).
    3px dilation smooths obstacle edges. Robot drawn in blue."""
    h, w = obs_combined.shape
    costmap = np.full((h, w, 3), 255, dtype=np.uint8)

    unknown = known_combined == 0
    obs_mask = obs_combined > 0

    rcx = CROSSHAIR_CX + ROBOT_CX_OFF
    rcy = CROSSHAIR_CY
    rx0 = max(0, rcx - ROBOT_W // 2)
    ry0 = max(0, rcy - ROBOT_H // 2)
    rx1 = min(w, rx0 + ROBOT_W)
    ry1 = min(h, ry0 + ROBOT_H)
    obs_mask[ry0:ry1, rx0:rx1] = False

    # 3px dilation on obstacles (smooth jagged edges, slight thickening)
    obs_u8 = obs_mask.astype(np.uint8) * 255
    cv2.dilate(obs_u8, _obs_dilate_kernel, dst=obs_u8)
    obs_mask = obs_u8 > 0
    obs_mask[ry0:ry1, rx0:rx1] = False

    # Distance from robot edge (pillow field)
    _robot_inv[:] = 255
    _robot_inv[ry0:ry1, rx0:rx1] = 0
    robot_dist = cv2.distanceTransform(_robot_inv, cv2.DIST_L2, 5)

    # Base colours
    costmap[unknown] = (128, 128, 128)
    costmap[obs_mask] = (50, 50, 50)

    # Obstacles within pillow radius: red (close) → yellow (far from robot)
    obs_in_pillow = obs_mask & (robot_dist <= _PILLOW_RADIUS)
    if np.any(obs_in_pillow):
        t = np.clip(robot_dist[obs_in_pillow] / float(_PILLOW_RADIUS), 0.0, 1.0)
        costmap[obs_in_pillow, 0] = 255
        costmap[obs_in_pillow, 1] = (t * 255).astype(np.uint8)
        costmap[obs_in_pillow, 2] = 0

    # Body: 30w × 30h, light blue (RGB)
    costmap[max(0, rcy - 15):rcy + 15, max(0, rcx - 15):rcx + 15] = (100, 160, 255)
    # Tracks: 17w × 6h centred at (rcx+5), dark blue (RGB)
    tx0 = max(0, rcx - 3)
    tx1 = rcx + 14
    costmap[max(0, rcy - 21):max(0, rcy - 15), tx0:tx1] = (30, 60, 180)
    costmap[rcy + 15:min(h, rcy + 21), tx0:tx1] = (30, 60, 180)

    return costmap


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

            # RS1 top-down depth → (obstacles, known), rotate 180°
            z1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            k1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            if self._rs1 and self._rs1.ok and self._rs1.verts is not None:
                z1, k1 = depth_topdown(self._rs1.verts)
            obs1 = z1[::-1, ::-1]
            known1 = k1[::-1, ::-1]

            # RS2 forward depth → (obstacles, known) at (W,H), then CW 90°
            z2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
            k2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
            if self._rs2 and self._rs2.ok and self._rs2.verts is not None:
                z2, k2 = depth_topdown_forward(self._rs2.verts,
                                               out_h=FRAME_W, out_w=FRAME_H,
                                               y_offset=RS2_EXTRINSIC_Y)
            obs2 = np.rot90(z2, k=-1)
            known2 = np.rot90(k2, k=-1)

            # Build known-area mask (union of both cameras, morph-closed to fill holes)
            fw_dx, fw_dy = int(TD_X_OFFSET) + FW_TD_X_DELTA, int(FW_Y_OFFSET)
            td_dx = int(TD_X_OFFSET)
            known_combined = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            _blit(known_combined, known1, td_dx)
            kc_tmp = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            _blit(kc_tmp, known2, fw_dx, fw_dy)
            np.maximum(known_combined, kc_tmp, out=known_combined)
            cv2.morphologyEx(known_combined, cv2.MORPH_CLOSE, _known_kernel,
                             iterations=2, dst=known_combined)

            # Combine obstacles (union of TD and FW cameras)
            obs_combined = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            _blit(obs_combined, obs1, td_dx)
            obs_tmp = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            _blit(obs_tmp, obs2, fw_dx, fw_dy)
            np.maximum(obs_combined, obs_tmp, out=obs_combined)

            topdown = _build_costmap(obs_combined, known_combined)

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
