"""vision.py – Process camera frames into atlas + obstacle map.
Camera hardware lives in cameras.py.  Depth processing is pure numpy/cv2.

Each depth camera produces a per-pixel classification (2.5D height map):
  FREE (0)       — floor detected (known, no obstacle)
  OBSTACLE (1-100) — height above floor in centimetres (capped at 100 cm)
  UNOBSERVED     — no valid depth data (blind spot, behind obstacle, outside FOV)

RS1 (top-down camera): orthographic projection, classification by Z threshold.
RS2 (forward camera):  pitch-rotated to bird's-eye, floor-as-free scatter + raycast known mask.
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

from robot_config import (FRAME_W, FRAME_H,
                          CROSSHAIR_CX, CROSSHAIR_CY, EGO_PX_SIZE,
                          WHEEL_RADIUS_M, WHEELBASE_M,
                          ROBOT_W, ROBOT_H, ROBOT_CX_OFF,
                          RCX, RCY, FOOT_X0, FOOT_Y0, FOOT_X1, FOOT_Y1)
from cameras import RSCamera, WebCam, HAS_RS
from safety import SafetyGuard
from pose import PoseEstimator
from globalmap import GlobalMap, MAP_W, MAP_H, ORIGIN_X, ORIGIN_Y, PX_SIZE as MAP_PX_SIZE
from slam import PoseGraphSLAM

CAM_ROW_H = FRAME_H                          # 240
ATLAS_W = FRAME_W * 3                        # 960
ATLAS_H = CAM_ROW_H + MAP_H                  # 960
TARGET_FPS = 30

CROSSHAIR_OPACITY = 0.3
DEBUG_CAMERAS = False

# --- RS1 (top-down camera) depth params ---
TD_PX_SIZE = np.float32(EGO_PX_SIZE)
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
        self.debug_depth = False
        self._pitch_cal_request = False
        self._pitch_cal_done = False
        self._lock = threading.Lock()
        self._persistent_obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        self._safety = SafetyGuard()
        self._pose = PoseEstimator(wheelbase_m=WHEELBASE_M, wheel_radius_m=WHEEL_RADIUS_M)
        self._global_map = PoseGraphSLAM()
        self._obs_mask, self._fw_cone_mask = self._build_obs_mask()
        self._free_range_mask = self._build_free_range_mask()
        self._wheelbase = None
        self._last_capture_time = None

        from gpu_render import GPURenderer
        self._gpu = GPURenderer(MAP_W, MAP_H, ATLAS_W, ATLAS_H)
        self._gpu.configure_depth_forward(
            rotation=FW_ROTATION, pivot=FW_PIVOT,
            translation=FW_TRANSLATION,
            px_size=float(FW_PX_SIZE),
            cam_height=float(FW_CAM_HEIGHT),
            sin_pitch=float(_fw_sin_pitch),
            cos_pitch=float(_fw_cos_pitch),
            floor_clip=float(FW_FLOOR_CLIP),
            height_clip=float(FW_HEIGHT_CLIP),
            out_h=FRAME_W, out_w=FRAME_H)
        self._gpu.configure_odom(fx=307.0, ds_factor=4, search=8)
        self._gpu.configure_gmap(MAP_W, MAP_H, FRAME_W, FRAME_H,
                                 ORIGIN_X, ORIGIN_Y, MAP_PX_SIZE)

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
                         combining with RS1 known, limiting the known area
                         to the camera's actual FOV).
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
        mask[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = 0
        return mask, fw_cone

    @staticmethod
    def _build_free_range_mask():
        """Range-limited mask for free evidence.

        At distance, small pose errors cause ego→global misalignment.
        Floor pixels adjacent to an obstacle leak into its global cells,
        generating spurious "free" evidence that erodes obstacles.
        We only trust "known + no obstacle = free" within this range.
        Obstacle *detection* still uses the full cone.
        """
        FREE_RANGE_PX = 200   # 2.0m at 1cm/px
        rcx = CROSSHAIR_CX + ROBOT_CX_OFF
        rcy = CROSSHAIR_CY
        yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
        dd = np.sqrt((xx - rcx).astype(np.float32)**2 +
                     (yy - rcy).astype(np.float32)**2)
        mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        mask[dd <= FREE_RANGE_PX] = 255
        return mask

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

    def _render_side_view(self):
        """Render a side-view cross-section showing height vs forward distance.

        Plots RS1 (cyan) and RS2 (yellow) depth points from the side so you
        can see if the floor planes align.  200x150 px, placed upper-right of 3D area.
        """
        W, H = 200, 150
        sv = np.zeros((H, W, 3), dtype=np.uint8)
        sv[:] = 20

        max_d = 2.5
        h_lo, h_hi = -0.15, 0.35

        def to_px(fwd, height):
            px = int(fwd / max_d * (W - 1))
            py = int((1.0 - (height - h_lo) / (h_hi - h_lo)) * (H - 1))
            return px, py

        # Floor line (y=0)
        _, y0 = to_px(0, 0.0)
        if 0 <= y0 < H:
            sv[y0, :] = [60, 60, 60]

        # 5cm obstacle threshold
        _, yt = to_px(0, 0.05)
        if 0 <= yt < H:
            sv[yt, :] = [0, 50, 80]

        sin_p = self._gpu._df_sin_p if hasattr(self._gpu, '_df_sin_p') else float(_fw_sin_pitch)
        cos_p = self._gpu._df_cos_p if hasattr(self._gpu, '_df_cos_p') else float(_fw_cos_pitch)
        cam_h = self._gpu._df_cam_h if hasattr(self._gpu, '_df_cam_h') else float(FW_CAM_HEIGHT)

        # RS1 top-down points (cyan)
        if self._rs1 and self._rs1.ok and self._rs1.verts is not None:
            pts = self._rs1.verts.reshape(-1, 3)
            valid = pts[:, 2] > 0
            pts = pts[valid]
            if len(pts) > 0:
                step = max(1, len(pts) // 2000)
                pts = pts[::step]
                td_height = float(TD_FLOOR_CLIP) - pts[:, 2]
                fwd = -pts[:, 1]
                for i in range(len(pts)):
                    px, py = to_px(float(fwd[i]), float(td_height[i]))
                    if 0 <= px < W and 0 <= py < H:
                        sv[py, px] = [200, 200, 0]

        # RS2 forward points (yellow-green)
        if self._rs2 and self._rs2.ok and self._rs2.verts is not None:
            pts = self._rs2.verts.reshape(-1, 3)
            valid = pts[:, 2] > 0.1
            pts = pts[valid]
            if len(pts) > 0:
                step = max(1, len(pts) // 2000)
                pts = pts[::step]
                phys_h = cam_h - pts[:, 1] * cos_p - pts[:, 2] * sin_p
                fwd_d = pts[:, 2] * float(_fw_cos_pitch) - pts[:, 1] * float(_fw_sin_pitch)
                for i in range(len(pts)):
                    px, py = to_px(float(fwd_d[i]), float(phys_h[i]))
                    if 0 <= px < W and 0 <= py < H:
                        sv[py, px] = [0, 200, 100]

        # Border
        sv[0, :] = sv[-1, :] = sv[:, 0] = sv[:, -1] = [80, 80, 80]

        return sv

    def request_calibration(self):
        """Trigger pitch calibration from the UI. Resets state for a fresh 20-frame run."""
        self._pitch_cal_deltas = []
        self._pitch_cal_hoffsets = []
        self._pitch_cal_done = False
        self._pitch_cal_request = True
        print("pitch_cal: calibration requested — collecting 20 frames")

    def _calibrate_rs2_pitch(self, verts):
        """Estimate RS2 pitch error by fitting a plane to floor points.

        Uses a centered region (|x| < 20cm, depth 0.5–1.5m) for best
        depth accuracy.  Regresses phys_h = δ·f + h_off to find pitch
        error δ and height offset h_off.
        """
        pts = verts.reshape(-1, 3)
        # Centered square: moderate depth, near optical axis
        valid = ((pts[:, 2] > 0.5) & (pts[:, 2] < 1.5) &
                 (np.abs(pts[:, 0]) < 0.20))
        pts = pts[valid]
        if len(pts) < 200:
            return

        sin_p, cos_p = float(_fw_sin_pitch), float(_fw_cos_pitch)
        cam_h = float(FW_CAM_HEIGHT)

        phys_h = cam_h - pts[:, 1] * cos_p - pts[:, 2] * sin_p
        floor = np.abs(phys_h) < 0.08
        fp = pts[floor]
        fh = phys_h[floor]

        if len(fp) < 50:
            return

        f = fp[:, 1] * sin_p - fp[:, 2] * cos_p
        A = np.column_stack([f, np.ones(len(f))])
        result = np.linalg.lstsq(A, fh, rcond=None)
        delta, h_off = result[0]

        if not hasattr(self, '_pitch_cal_deltas'):
            self._pitch_cal_deltas = []
            self._pitch_cal_hoffsets = []
        self._pitch_cal_deltas.append(delta)
        self._pitch_cal_hoffsets.append(h_off)

        if len(self._pitch_cal_deltas) < 20:
            return

        med_delta = float(np.median(self._pitch_cal_deltas))
        med_hoff = float(np.median(self._pitch_cal_hoffsets))
        err_deg = math.degrees(med_delta)

        print("pitch_cal: error=%.3f° h_offset=%.1fcm (%d samples, %d pts/frame)" % (
            err_deg, med_hoff * 100, len(self._pitch_cal_deltas), len(fp)))

        new_pitch = _fw_pitch_rad + med_delta
        new_sin = float(abs(math.sin(new_pitch)))
        new_cos = float(abs(math.cos(new_pitch)))
        new_cam_h = float(FW_CAM_HEIGHT) - med_hoff

        if abs(med_delta) > math.radians(0.05) or abs(med_hoff) > 0.005:
            print("pitch_cal: correcting pitch %.2f° → %.2f° "
                  "(sin %.4f→%.4f, cos %.4f→%.4f) cam_h %.3f→%.3f" % (
                  FW_PITCH_DEG, FW_PITCH_DEG + err_deg,
                  sin_p, new_sin, cos_p, new_cos,
                  float(FW_CAM_HEIGHT), new_cam_h))
            self._gpu.update_pitch_params(new_sin, new_cos, cam_h=new_cam_h)
        else:
            print("pitch_cal: within tolerance (%.3f° / %.1fcm)" % (err_deg, med_hoff * 100))

        self._pitch_cal_done = True
        self._pitch_cal_request = False

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
            if self._k1_bbox_n <= 2:
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

            # RS2 forward depth → (obstacles, known, raw_scatter) at (W,H), then CW 90°
            z2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
            k2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
            _raw_scatter = None
            _dbg = self.debug_depth
            if self._rs2 and self._rs2.ok and self._rs2.verts is not None:
                if getattr(self, '_pitch_cal_request', False):
                    self._calibrate_rs2_pitch(self._rs2.verts)
                _gpu_result = self._gpu.depth_forward_gpu(
                    self._rs2.verts, y_offset=RS2_EXTRINSIC_Y, debug=_dbg)
                if _gpu_result is not None:
                    z2, k2, _raw_scatter = _gpu_result
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
            if self._kdiag_n <= 2 or self._kdiag_n % 300 == 0:
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
            obs_combined[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = 0
            known_combined[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = 255
            _t_obs = time.monotonic()

            # --- Odometry: visual + wheel → Kalman fused pose ---
            vis_yaw, vis_fwd, vis_conf = 0.0, 0.0, 0.0
            if self._rs2 and self._rs2.ok:
                fw_gray = cv2.cvtColor(self._rs2.color, cv2.COLOR_RGB2GRAY)
                _odom_result = self._gpu.odom_gpu(fw_gray)
                if _odom_result is not None:
                    vis_yaw, vis_fwd, vis_conf = _odom_result

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
            if self._odom_log_n % 90 == 0:
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
            if self._hdiag_n % 300 == 0:
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

            self._gpu.gmap_update_gpu(
                obs_combined, known_combined,
                cap_x, cap_y, cap_theta,
                rcx_f, rcy_f, float(TD_PX_SIZE),
                free_range_mask=self._free_range_mask)
            self._global_map.keyframe_check(
                obs_combined, known_combined,
                cap_x, cap_y, cap_theta,
                rcx_f, rcy_f, float(TD_PX_SIZE))
            _t_gmap_up = time.monotonic()

            ego_proj = self._gpu.gmap_project_gpu(
                self._pose.x, self._pose.y, self._pose.theta,
                rcx_f, rcy_f, float(TD_PX_SIZE), FRAME_H, FRAME_W)

            self._persistent_obs[:] = 0
            if ego_proj is not None:
                self._persistent_obs[ego_proj < 90] = 255
            self._persistent_obs[obs_combined > 0] = 255
            self._persistent_obs[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = 0

            self._safety.update(self._persistent_obs, fused_yaw, fused_fwd)
            _t_safety = time.monotonic()

            # --- GPU renders full atlas (3D view + cameras + minimap + battery) ---
            trail = self._pose.get_world_history()
            rgb1 = self._webcam.color if (self._webcam and self._webcam.ok) else black
            rgbd1 = self._rs1.color[::-1, ::-1] if (self._rs1 and self._rs1.ok) else black
            rgbd2 = self._rs2.color if (self._rs2 and self._rs2.ok) else black

            bat_frac = 0.0
            if self._wheelbase:
                pct = self._wheelbase.battery_pct
                bat_frac = max(0.0, min(1.0, pct / 100.0)) if pct >= 0 else 0.0

            atlas = self._gpu.render(
                self._pose.x, self._pose.y, self._pose.theta,
                cameras=[rgb1, rgbd1, rgbd2],
                trail_xy=trail,
                fwd_scale=self._safety.fwd_scale,
                bwd_scale=self._safety.bwd_scale,
                ang_scale=self._safety.ang_scale,
                battery_frac=bat_frac)

            # Debug overlays — gated by debug_depth flag
            if _dbg and atlas is not None:
                if _raw_scatter is not None:
                    dbg = np.rot90(_raw_scatter, k=-1)
                    dbg_rgb = np.zeros((dbg.shape[0], dbg.shape[1], 3), dtype=np.uint8)
                    dbg_rgb[dbg == 1] = [0, 255, 0]
                    dbg_rgb[dbg >= 2] = [255, 0, 0]
                    dh, dw = dbg_rgb.shape[:2]
                    atlas[ATLAS_H - dh:ATLAS_H, ATLAS_W - dw:ATLAS_W] = dbg_rgb

                dbg2 = np.zeros((known_combined.shape[0], known_combined.shape[1], 3), dtype=np.uint8)
                dbg2[(known_combined > 0) & (obs_combined == 0)] = [0, 255, 0]
                dbg2[obs_combined > 0] = [255, 0, 0]
                d2h, d2w = dbg2.shape[:2]
                atlas[ATLAS_H - d2h:ATLAS_H, 0:d2w] = dbg2

                # Side-view cross-section (upper-right of 3D area)
                sv = self._render_side_view()
                svh, svw = sv.shape[:2]
                atlas[FRAME_H:FRAME_H + svh, ATLAS_W - svw:ATLAS_W] = sv

            with self._lock:
                self.frames[0][:] = rgb1
                self.frames[1][:] = rgbd1
                self.frames[2][:] = rgbd2
                if atlas is not None:
                    self.atlas[:] = atlas
                self.timestamp = time.time()
            _t_render = time.monotonic()
            _t_end = _t_render

            _loop_times.append((_t_grab - _t0, _t_rs1 - _t_grab,
                                _t_depth - _t_rs1, _t_obs - _t_depth,
                                _t_odom - _t_obs, _t_gmap_up - _t_odom,
                                _t_safety - _t_gmap_up,
                                _t_end - _t_safety))
            if len(_loop_times) % 300 == 0:
                avg = np.mean(_loop_times[-300:], axis=0) * 1000
                print("capture: grab=%.1f rs1=%.1f rs2=%.1f obs=%.1f odom=%.1f "
                      "gmap=%.1f safety=%.1f render=%.1f "
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
