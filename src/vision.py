"""vision.py – Process camera frames into atlas + obstacle map.
Camera hardware lives in cameras.py.  Depth processing is pure numpy/cv2.

Each depth camera produces a per-pixel classification (2.5D height map):
  FREE (0)       — floor detected (known, no obstacle)
  OBSTACLE (1-100) — height above floor in centimetres (capped at 100 cm)
  UNOBSERVED     — no valid depth data (blind spot, behind obstacle, outside FOV)

RS1 (top-down camera): orthographic projection, classification by Z threshold.
RS2 (forward camera):  pitch-rotated to bird's-eye, 2D raycasting for known mask.
Both are combined, masked to their respective FOVs, and fed to the global map.

Atlas layout (960×960):
  Row 0–239:   rgb1 (320×240) | rgbd1 (320×240) | rgbd2 (320×240)
  Row 240–959:  global map (960×720)
"""

import math
import threading
import time
import numpy as np
import cv2

from cameras import RSCamera, WebCam, HAS_RS, FRAME_W, FRAME_H
import odometry
from safety import SafetyGuard
from pose import PoseEstimator
from globalmap import GlobalMap, MAP_W, MAP_H
from slam import PoseGraphSLAM

CAM_ROW_H = FRAME_H                          # 240
ATLAS_W = FRAME_W * 3                        # 960
ATLAS_H = CAM_ROW_H + MAP_H                  # 960
TARGET_FPS = 30

CROSSHAIR_CX, CROSSHAIR_CY = 159, 119
CROSSHAIR_OPACITY = 0.3
DEBUG_CAMERAS = False

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
FW_HEIGHT_CLIP = np.float32(1.30)   # max obstacle height to accept (m)
FW_FLOOR_CLIP  = np.float32(0.05)   # min height: ignore below 5cm (carpet/floor)
FW_CAM_HEIGHT  = np.float32(0.97)   # RS2 forward camera height above floor (m)
_fw_sin_pitch  = np.float32(abs(math.sin(_fw_pitch_rad)))  # sin(64.4°)≈0.903
_fw_cos_pitch  = np.float32(abs(math.cos(_fw_pitch_rad)))  # cos(64.4°)≈0.431
_fw_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

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
    """RS1 (top-down camera) pointcloud → (obs, known) via orthographic projection.

    Classification (per pixel):
      known=255, obs=0      → floor detected (z >= TD_FLOOR_CLIP) — free
      known=255, obs=1..100 → obstacle height in cm above floor — obstacle
      known=0               → no valid depth at this pixel — unobserved

    Height = (TD_FLOOR_CLIP - z) * 100, clamped to [1, 100].
    Multiple points at the same pixel keep the tallest (np.maximum.at).
    Returns two uint8 arrays of shape (out_h, out_w).
    """
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

    # Obstacle height: (TD_FLOOR_CLIP - z) in cm, tallest wins per pixel
    v_obs = verts[obstacle]
    if len(v_obs) > 0:
        p_obs = v_obs[:, :2] * scale + center
        z_obs = v_obs[:, 2]
        height_cm = np.clip(
            ((TD_FLOOR_CLIP - z_obs) * 100).astype(np.int32),
            1, 100).astype(np.uint8)
        with np.errstate(invalid='ignore'):
            jo, io = p_obs.astype(np.uint32).T
        mo = (io < np.uint32(out_h)) & (jo < np.uint32(out_w))
        np.maximum.at(obs, (io[mo], jo[mo]), height_cm[mo])

    return obs, known


_raycast_dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_polar_dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
_N_RAYS = 500


def depth_topdown_forward(verts, out_h=FRAME_H, out_w=FRAME_W, y_offset=0.0):
    """RS2 (forward camera) pointcloud → (obs, known) via rotated bird's-eye.

    Height is computed from the original camera coordinates before rotation:
      h = FW_CAM_HEIGHT - y_cam * sin(pitch) + z_cam * cos(pitch)
    This gives true physical height above floor regardless of rotation accuracy.
    The rotation transform is used only for the (x,y) top-down projection.

    Returns two uint8 arrays of shape (out_h, out_w).
    """
    obs = np.zeros((out_h, out_w), dtype=np.uint8)
    known = np.zeros((out_h, out_w), dtype=np.uint8)
    if len(verts) == 0:
        return obs, known

    # Filter out invalid depth pixels (z=0 → vertex (0,0,0)).
    # Without this, they project to the camera centre after rotation and
    # create false obstacles that kill the raycast known-mask.
    valid = verts[:, 2] > np.float32(0)
    v = verts[valid].copy()
    if len(v) == 0:
        return obs, known
    if y_offset != 0.0:
        v[:, 1] += np.float32(y_offset)

    # Physical height above floor from camera coords (before rotation).
    # Camera: Y-down, Z-forward, mounted at FW_CAM_HEIGHT, pitch θ below horiz.
    # Y_cam_world = (0, -sinθ, -cosθ),  Z_cam_world = (0, cosθ, -sinθ)
    # ∴ height = H - y_cam·cos(θ) - z_cam·sin(θ)
    phys_h = (FW_CAM_HEIGHT
              - v[:, 1] * _fw_cos_pitch
              - v[:, 2] * _fw_sin_pitch)

    if DEBUG_CAMERAS:
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

    # Rotation for top-down (x,y) position only
    v = np.dot(v - FW_PIVOT, FW_ROTATION) + FW_PIVOT - FW_TRANSLATION

    scale = np.float32(1.0 / FW_PX_SIZE)
    offset = np.float32([out_w / 2.0, out_h / 2.0 + 1.0 * scale])
    proj = v[:, :2] * scale + offset
    with np.errstate(invalid='ignore'):
        j, i = proj.astype(np.uint32).T
    m_all = (i < np.uint32(out_h)) & (j < np.uint32(out_w))

    # Classification: rotated z' with calibrated thresholds (don't touch these)
    m_obs = m_all & (v[:, 2] > FW_FLOOR_CLIP) & (v[:, 2] < FW_HEIGHT_CLIP)
    # Height values: physical height from camera geometry (accurate per-point)
    height_cm = np.clip((phys_h[m_obs] * 100).astype(np.int32), 1, 100).astype(np.uint8)
    np.maximum.at(obs, (i[m_obs], j[m_obs]), height_cm)

    if DEBUG_CAMERAS:
        cv2.imshow("fw_before_morph", obs.copy())

    # Morphological close on binary mask only (don't spread height values)
    obs_bin = (obs > 0).astype(np.uint8)
    cv2.morphologyEx(obs_bin, cv2.MORPH_CLOSE, _fw_kernel, iterations=2,
                     dst=obs_bin)
    obs[obs_bin & (obs == 0)] = 1

    if DEBUG_CAMERAS:
        cv2.imshow("fw_final", obs.copy())

    # --- 2D raycasting: known mask via polar line-of-sight ---
    # Camera origin projected into the top-down pixel grid
    cam_world = (np.dot(np.float32([0, 0, 0]) - FW_PIVOT, FW_ROTATION)
                 + FW_PIVOT - FW_TRANSLATION)
    cam_px = cam_world[:2] * scale + offset
    cx, cy = float(cam_px[0]), float(cam_px[1])

    max_r = int(math.sqrt(float(out_h) ** 2 + float(out_w) ** 2)) + 1
    center = (cx, cy)

    # obs → polar (rows = angle bins, cols = distance from camera)
    polar_obs = cv2.warpPolar(
        obs, (max_r, _N_RAYS), center, float(max_r),
        cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST)
    # Dilate in polar space to close sampling gaps (prevents rays leaking
    # through thin obstacles).  Only affects ray termination, not output obs.
    cv2.dilate(polar_obs, _polar_dilate_kern, dst=polar_obs)

    # Valid-point mask (dilated to fill gaps) → polar
    valid_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    valid_mask[i[m_all], j[m_all]] = 255
    cv2.dilate(valid_mask, _raycast_dilate_kern, dst=valid_mask)
    polar_valid = cv2.warpPolar(
        valid_mask, (max_r, _N_RAYS), center, float(max_r),
        cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST)

    # Per-angle: nearest obstacle, farthest valid point
    has_obs_row = polar_obs > 0
    any_obs_row = has_obs_row.any(axis=1)
    first_obs_col = np.argmax(has_obs_row, axis=1)

    has_val_row = polar_valid > 0
    any_val_row = has_val_row.any(axis=1)
    last_val_col = max_r - 1 - np.argmax(has_val_row[:, ::-1], axis=1)

    # Endpoint: nearest obstacle if any, else farthest valid, else no ray
    endpoint = np.where(
        any_obs_row, first_obs_col,
        np.where(any_val_row, last_val_col, np.int32(-1)))

    # Build polar known: cols 0..endpoint = 255 per row
    cols = np.arange(max_r, dtype=np.int32)
    polar_known = np.where(
        (endpoint[:, None] >= 0) & (cols[None, :] <= endpoint[:, None]),
        np.uint8(255), np.uint8(0)).astype(np.uint8)

    # Warp back to Cartesian
    known = cv2.warpPolar(
        polar_known, (out_w, out_h), center, float(max_r),
        cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

    if not hasattr(depth_topdown_forward, '_diag_n'):
        depth_topdown_forward._diag_n = 0
    depth_topdown_forward._diag_n += 1
    if depth_topdown_forward._diag_n <= 3 or depth_topdown_forward._diag_n % 60 == 0:
        ep_valid = endpoint[endpoint >= 0]
        print("fw_raycast: pts=%d obs_px=%d known_px=%d "
              "valid_mask_px=%d polar_obs_nz=%d ep_valid=%d ep_range=[%d,%d] "
              "center=(%.0f,%.0f)" % (
                  len(v), int(np.count_nonzero(obs)),
                  int(np.count_nonzero(known)),
                  int(np.count_nonzero(valid_mask)),
                  int(np.count_nonzero(polar_obs)),
                  len(ep_valid),
                  int(ep_valid.min()) if len(ep_valid) else -1,
                  int(ep_valid.max()) if len(ep_valid) else -1,
                  cx, cy))

    return obs, known


class Vision:
    """Pre-allocated vision state. One capture thread; readers use .frames, .atlas, .timestamp."""

    def __init__(self, rs1_serial, rs2_serial, rgb1_device_id, headless=True):
        print("Vision: init start")
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
        self._persistent_obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        self._safety = SafetyGuard()
        self._pose = PoseEstimator(wheelbase_m=0.34, wheel_radius_m=0.08565)
        self._global_map = PoseGraphSLAM()
        self._obs_mask, self._fw_cone_mask = self._build_obs_mask()
        self._wheelbase = None
        self._last_capture_time = None

        self._running = False
        self._thread = None
        self._rs1 = None
        self._rs2 = None
        self._webcam = None

    @staticmethod
    def _build_obs_mask():
        """Pre-compute ego-space masks.

        Returns (combined_mask, fw_cone_mask):
          combined_mask — RS1 rectangle ∪ RS2 80° cone, robot excluded.
          fw_cone_mask — RS2 80° cone only (used to clip RS2 known before
                         combining with RS1 known, preventing the raycasted
                         known area from leaking into the RS1 rectangle).
        """
        rcx = CROSSHAIR_CX + ROBOT_CX_OFF          # 81 — robot center column
        rcy = CROSSHAIR_CY                          # 119 — robot center row

        mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

        # RS1 top-down rectangle (loose for now — tight clip pending debug).
        TD_EDGE = 10
        td_col_end = FRAME_W + int(TD_X_OFFSET)     # 245
        mask[TD_EDGE:FRAME_H - TD_EDGE, TD_EDGE:td_col_end - TD_EDGE] = 255

        # RS2 forward 80° cone (±40°), 2.5m range, from robot center
        yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
        dx = (xx - rcx).astype(np.float32)
        dy = (yy - rcy).astype(np.float32)
        dist = np.sqrt(dx * dx + dy * dy)
        angle = np.abs(np.degrees(np.arctan2(dy, dx)))
        cone = (angle <= 40.0) & (dist <= 250.0)

        fw_cone = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        fw_cone[cone] = 255

        mask[cone] = 255

        # Clear robot footprint (force-set to known+free in capture loop)
        rx0 = max(0, rcx - ROBOT_W // 2 - 2)
        ry0 = max(0, rcy - ROBOT_H // 2 - 2)
        rx1 = min(FRAME_W, rcx + ROBOT_W // 2 + 3)
        ry1 = min(FRAME_H, rcy + ROBOT_H // 2 + 2)
        mask[ry0:ry1, rx0:rx1] = 0
        return mask, fw_cone

    def set_wheelbase(self, wb):
        """Provide wheelbase reference for wheel odometry fusion."""
        self._wheelbase = wb

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
        _loop_times = []

        while self._running:
            _t0 = time.monotonic()

            if self._webcam:
                self._webcam.grab()
            if self._rs1:
                self._rs1.grab()
            if self._rs2:
                self._rs2.grab()
            _t_grab = time.monotonic()

            # Snapshot pose at capture time (before odometry advances it)
            cap_x, cap_y, cap_theta = self._pose.x, self._pose.y, self._pose.theta

            # RS1 top-down depth → (obstacles, known), rotate 180°
            z1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            k1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            if self._rs1 and self._rs1.ok and self._rs1.verts is not None:
                z1, k1 = depth_topdown(self._rs1.verts)
            obs1 = z1[::-1, ::-1]
            known1 = k1[::-1, ::-1]
            if not hasattr(self, '_k1_bbox_n'):
                self._k1_bbox_n = 0
            self._k1_bbox_n += 1
            if self._k1_bbox_n <= 5:
                nz = np.nonzero(known1)
                if len(nz[0]) > 0:
                    r0, r1 = int(nz[0].min()), int(nz[0].max())
                    c0, c1 = int(nz[1].min()), int(nz[1].max())
                    td_dx = int(TD_X_OFFSET)
                    print("rs1_known bbox (after 180° flip): "
                          "rows %d..%d  cols %d..%d  "
                          "(after td_dx=%d shift: cols %d..%d)  "
                          "total_px=%d" % (r0, r1, c0, c1,
                                           td_dx, c0 + td_dx, c1 + td_dx,
                                           len(nz[0])))
            _t_rs1 = time.monotonic()

            # RS2 forward depth → (obstacles, known) at (W,H), then CW 90°
            z2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
            k2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
            if self._rs2 and self._rs2.ok and self._rs2.verts is not None:
                z2, k2 = depth_topdown_forward(self._rs2.verts,
                                               out_h=FRAME_W, out_w=FRAME_H,
                                               y_offset=RS2_EXTRINSIC_Y)
            obs2 = np.rot90(z2, k=-1)
            known2 = np.rot90(k2, k=-1)
            _t_depth = time.monotonic()

            # --- Combine into ego-space (obs_combined, known_combined) ---
            # Principle: a pixel is "known" ONLY where a camera actually measured
            # depth.  No morphological expansion of known — blind spots (mast
            # shadow, sensor holes) stay unobserved.
            fw_dx, fw_dy = int(TD_X_OFFSET) + FW_TD_X_DELTA, int(FW_Y_OFFSET)
            td_dx = int(TD_X_OFFSET)

            # RS2 known: clip to 80° forward cone before combining
            kc_tmp = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            _blit(kc_tmp, known2, fw_dx, fw_dy)
            if not hasattr(self, '_kdiag_n'):
                self._kdiag_n = 0
            self._kdiag_n += 1
            if self._kdiag_n <= 3 or self._kdiag_n % 60 == 0:
                _k2nz = int(np.count_nonzero(known2))
                _kcnz_pre = int(np.count_nonzero(kc_tmp))
                np.bitwise_and(kc_tmp, self._fw_cone_mask, out=kc_tmp)
                _kcnz_post = int(np.count_nonzero(kc_tmp))
                print("fw_known: known2=%d blit=%d after_cone=%d "
                      "fw_dx=%d fw_dy=%d" % (_k2nz, _kcnz_pre, _kcnz_post,
                                              fw_dx, fw_dy))
            else:
                np.bitwise_and(kc_tmp, self._fw_cone_mask, out=kc_tmp)

            # Union of both cameras' known masks
            known_combined = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            _blit(known_combined, known1, td_dx)
            np.maximum(known_combined, kc_tmp, out=known_combined)

            # Union of both cameras' obstacle masks
            obs_combined = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            _blit(obs_combined, obs1, td_dx)
            obs_tmp = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            _blit(obs_tmp, obs2, fw_dx, fw_dy)
            np.maximum(obs_combined, obs_tmp, out=obs_combined)

            # Clip both to valid observation area:
            #   RS1 → edge-trimmed rectangle, RS2 → 80° cone (already clipped above)
            np.bitwise_and(obs_combined, self._obs_mask, out=obs_combined)
            np.bitwise_and(known_combined, self._obs_mask, out=known_combined)

            # Robot footprint is always known-free (robot physically occupies it)
            rcx = CROSSHAIR_CX + ROBOT_CX_OFF
            rcy = CROSSHAIR_CY
            rx0 = max(0, rcx - ROBOT_W // 2 - 2)
            ry0 = max(0, rcy - ROBOT_H // 2 - 2)
            rx1 = min(FRAME_W, rcx + ROBOT_W // 2 + 3)
            ry1 = min(FRAME_H, rcy + ROBOT_H // 2 + 2)
            obs_combined[ry0:ry1, rx0:rx1] = 0
            known_combined[ry0:ry1, rx0:rx1] = 255
            _t_obs = time.monotonic()

            # --- Odometry: visual + wheel → Kalman fused pose ---
            if self._rs2 and self._rs2.ok:
                fw_gray = cv2.cvtColor(self._rs2.color, cv2.COLOR_RGB2GRAY)
                vis_yaw, vis_fwd, vis_conf = odometry.update(fw_gray)
            else:
                vis_yaw, vis_fwd, vis_conf = 0.0, 0.0, 0.0

            now = time.monotonic()
            dt = (now - self._last_capture_time) if self._last_capture_time else 0.0
            self._last_capture_time = now

            if self._wheelbase is not None:
                vl, vr = self._wheelbase.get_wheel_velocities_mps()
            else:
                vl, vr = 0.0, 0.0

            fused_yaw, fused_fwd = self._pose.update(
                vl, vr, dt, vis_yaw, vis_fwd, vis_conf)
            _t_odom = time.monotonic()

            if not hasattr(self, '_odom_log_n'):
                self._odom_log_n = 0
            self._odom_log_n += 1
            if self._odom_log_n % 30 == 0:
                wb = self._wheelbase
                enc_info = 'no_wb'
                if wb:
                    enc_ok = getattr(wb, '_enc_ok', '?')
                    enc_age = (time.monotonic() -
                               getattr(wb, '_enc_last_good', 0))
                    enc_info = 'enc=%s age=%.1fs' % (enc_ok, enc_age)
                print("odom: vl=%.4f vr=%.4f dt=%.4f "
                      "pose=(%.3f,%.3f,%.1f°) %s"
                      % (vl, vr, dt,
                         self._pose.x, self._pose.y,
                         np.degrees(self._pose.theta),
                         enc_info))

            # --- Height diagnostic (every 30 frames) ---
            if not hasattr(self, '_hdiag_n'):
                self._hdiag_n = 0
            self._hdiag_n += 1
            if self._hdiag_n % 30 == 0:
                om = obs_combined[obs_combined > 0]
                if len(om) > 0:
                    bins = [0, 5, 10, 20, 30, 50, 70, 100, 256]
                    h, _ = np.histogram(om, bins)
                    print("heights: " + " ".join(
                        "%d-%d:%d" % (bins[i], bins[i+1]-1, h[i])
                        for i in range(len(h))))

            # --- Update global occupancy map ---
            rcx_f = float(CROSSHAIR_CX + ROBOT_CX_OFF)
            rcy_f = float(CROSSHAIR_CY)

            self._global_map.update(
                obs_combined, known_combined,
                cap_x, cap_y, cap_theta,
                rcx_f, rcy_f, float(TD_PX_SIZE))
            _t_gmap_up = time.monotonic()

            # Project global map to ego space for costmap/safety.
            ego_proj = self._global_map.project_to_ego(
                self._pose.x, self._pose.y, self._pose.theta,
                rcx_f, rcy_f, float(TD_PX_SIZE), FRAME_H, FRAME_W)

            self._persistent_obs[:] = 0
            self._persistent_obs[ego_proj < 90] = 255
            self._persistent_obs[obs_combined > 0] = 255

            self._safety.update(self._persistent_obs, fused_yaw, fused_fwd)
            _t_safety = time.monotonic()

            # --- Render global map with trajectory + safety overlay ---
            trail = self._pose.get_world_history()
            gmap_render = self._global_map.render(
                self._pose.x, self._pose.y, self._pose.theta,
                trail_xy=trail,
                fwd_scale=self._safety.fwd_scale,
                bwd_scale=self._safety.bwd_scale,
                ang_scale=self._safety.ang_scale)
            _t_render = time.monotonic()

            rgb1 = self._webcam.color if (self._webcam and self._webcam.ok) else black
            rgbd1 = self._rs1.color[::-1, ::-1] if (self._rs1 and self._rs1.ok) else black
            rgbd2 = self._rs2.color if (self._rs2 and self._rs2.ok) else black

            with self._lock:
                self.frames[0][:] = rgb1
                self.frames[1][:] = rgbd1
                self.frames[2][:] = rgbd2
                self.atlas[0:CAM_ROW_H, 0:FRAME_W] = rgb1
                self.atlas[0:CAM_ROW_H, FRAME_W:FRAME_W * 2] = rgbd1
                self.atlas[0:CAM_ROW_H, FRAME_W * 2:FRAME_W * 3] = rgbd2
                self.atlas[CAM_ROW_H:ATLAS_H, 0:ATLAS_W] = gmap_render
                self.timestamp = time.time()
            _t_end = time.monotonic()

            _loop_times.append((_t_grab - _t0, _t_rs1 - _t_grab,
                                _t_depth - _t_rs1, _t_obs - _t_depth,
                                _t_odom - _t_obs, _t_gmap_up - _t_odom,
                                _t_safety - _t_gmap_up,
                                _t_render - _t_safety, _t_end - _t_render))
            if len(_loop_times) % 100 == 0:
                avg = np.mean(_loop_times[-100:], axis=0) * 1000
                print("capture: grab=%.1f rs1=%.1f rs2=%.1f obs=%.1f odom=%.1f "
                      "gmap_up=%.1f safety=%.1f render=%.1f blit=%.1f "
                      "TOTAL=%.1fms" % (*avg, sum(avg)))

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

    @property
    def safety_fwd_scale(self):
        return self._safety.fwd_scale

    @property
    def safety_bwd_scale(self):
        return self._safety.bwd_scale

    @property
    def safety_ang_scale(self):
        return self._safety.ang_scale

    @property
    def safety_throttled(self):
        return self._safety.is_throttled
