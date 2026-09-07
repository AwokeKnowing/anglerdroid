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
import concurrent.futures
import time
import numpy as np
import cv2

from loop_timing import FrameBudget

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
from checkered_mat import TopdownHazardDetector
from imu import IMUPipeline
from odom_thread import OdomThread

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
FW_FLOOR_CLIP  = np.float32(0.15)   # ego-forward < 15cm from camera → free (not obstacle)
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


def _clip_decimated_border(verts, border=4, orig_w=848, orig_h=480, out=None):
    """Zero out border pixels of a decimated RS depth grid.

    The RS SDK decimation filter averages depth in NxN blocks.  At frame
    edges, blocks include invalid (zero) pixels, producing small non-zero
    depth values that deproject to wildly wrong 3-D positions.  Zeroing
    the border removes these systematic artifacts.

    If out is provided (Nx3 float32, N>=len(verts)), copy into it and mutate
    in place — no fresh alloc. Otherwise allocates a copy (legacy callers).
    """
    src = np.asarray(verts).reshape(-1, 3)
    n = int(src.shape[0])
    if out is not None and out.shape[0] >= n and out.shape[1] == 3:
        v = out[:n]
        np.copyto(v, src.astype(np.float32, copy=False))
    else:
        v = np.array(src, dtype=np.float32, copy=True)
    if n < 100:
        return v
    aspect = float(orig_w) / orig_h
    w_est = int(round(math.sqrt(n * aspect)))
    for w_try in (w_est, w_est - 1, w_est + 1, w_est + 2, w_est - 2):
        if w_try > 0 and n % w_try == 0:
            h = n // w_try
            g = v.reshape(h, w_try, 3)
            g[:border, :, :] = 0
            g[-border:, :, :] = 0
            g[:, :border, :] = 0
            g[:, -border:, :] = 0
            return g.reshape(-1, 3)
    return v


def check_topdown_all_in_one(verts, out_h=FRAME_H, out_w=FRAME_W,
                             near_threshold_m=0.30, near_min_pixels=50,
                             overhang_near_m=0.30, overhang_far_m=0.70,
                             overhang_row_min=10, overhang_row_max=50,
                             soft_near_m=0.35, soft_far_m=1.00,
                             soft_min_height_cm=5.0, soft_max_height_cm=30.0,
                             soft_row_min=10, soft_row_max=80,
                             lateral_col_margin=30,
                             soft_min_pixels=100, overhang_min_pixels=80,
                             floor_clip_m=TD_FLOOR_CLIP):
    """Combined topdown checks: near-field, overhang, soft-low in single pass.
    
    OPTIMIZED: Iterates over verts ONCE instead of 3+ separate passes.
    
    Returns: dict with all check results
    """
    result = {
        'near_field': False, 'near_close_count': 0, 'near_min_z': float('inf'),
        'overhang': False, 'overhang_count': 0, 'overhang_median_z': float('inf'),
        'soft_low': False, 'soft_low_count': 0, 'soft_low_median_height': float('inf'),
    }
    
    if len(verts) == 0:
        return result
    
    verts = _clip_decimated_border(verts)
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    
    # Valid depth filter (shared)
    valid = z > 0.01
    if not np.any(valid):
        return result
    
    z_valid = z[valid]
    result['near_min_z'] = float(np.min(z_valid))
    
    # === Near-field check (any Z < threshold) ===
    near_mask = valid & (z < near_threshold_m)
    result['near_close_count'] = int(np.sum(near_mask))
    result['near_field'] = result['near_close_count'] >= near_min_pixels
    
    # === Overhang and soft-low checks (require image projection) ===
    # Filter for medium-range valid depth
    mid_range = valid & (z >= overhang_near_m) & (z <= soft_far_m)
    
    if np.any(mid_range):
        v_mid = verts[mid_range]
        z_mid = v_mid[:, 2]
        
        # Project to image coordinates (before 180° rotation)
        scale = np.float32(1.0 / TD_PX_SIZE)
        center = np.float32([out_w * 0.5, out_h * 0.5])
        p = v_mid[:, :2] * scale + center
        cols, rows = p[:, 0], p[:, 1]
        
        # After 180° rotation: (row, col) → (out_h - 1 - row, out_w - 1 - col)
        rows_rot = out_h - 1 - rows
        cols_rot = out_w - 1 - cols
        
        # Overhang: 30-70cm range, forward strip rows 10-50
        ovh_range = (z_mid >= overhang_near_m) & (z_mid <= overhang_far_m)
        ovh_strip = (
            (rows_rot >= overhang_row_min) & (rows_rot <= overhang_row_max) &
            (cols_rot >= lateral_col_margin) & (cols_rot < out_w - lateral_col_margin)
        )
        ovh_mask = ovh_range & ovh_strip
        
        if np.any(ovh_mask):
            ovh_z = z_mid[ovh_mask]
            result['overhang_count'] = len(ovh_z)
            result['overhang_median_z'] = float(np.median(ovh_z))
            result['overhang'] = result['overhang_count'] >= overhang_min_pixels
        
        # Soft-low: 35cm-1m range, height 5-30cm, forward strip rows 10-80
        soft_range = (z_mid >= soft_near_m) & (z_mid <= soft_far_m)
        height_cm = (floor_clip_m - z_mid) * 100.0
        soft_height = (height_cm >= soft_min_height_cm) & (height_cm <= soft_max_height_cm)
        soft_strip = (
            (rows_rot >= soft_row_min) & (rows_rot <= soft_row_max) &
            (cols_rot >= lateral_col_margin) & (cols_rot < out_w - lateral_col_margin)
        )
        soft_mask = soft_range & soft_height & soft_strip
        
        if np.any(soft_mask):
            soft_heights = height_cm[soft_mask]
            result['soft_low_count'] = len(soft_heights)
            result['soft_low_median_height'] = float(np.median(soft_heights))
            result['soft_low'] = result['soft_low_count'] >= soft_min_pixels
    
    return result


def check_topdown_near_field(verts, threshold_m=0.30, min_pixels=50):
    """DEPRECATED: Use check_topdown_all_in_one for better performance.
    
    Check if top-down camera sees a close object (near-field hazard reflex).
    """
    if len(verts) == 0:
        return False, 0, float('inf')

    verts = _clip_decimated_border(verts)
    z = verts[:, 2]
    valid = z > 0.01
    z_valid = z[valid]

    if len(z_valid) == 0:
        return False, 0, float('inf')

    close_mask = z_valid < threshold_m
    close_count = int(np.sum(close_mask))
    min_z = float(np.min(z_valid))

    triggered = close_count >= min_pixels

    return triggered, close_count, min_z


def check_topdown_soft_low_obstacle(verts, 
                                     near_m=0.35, far_m=1.00,
                                     min_height_cm=5.0, max_height_cm=30.0,
                                     forward_strip_row_min=10, forward_strip_row_max=80,
                                     lateral_col_margin=30,
                                     min_pixels=100,
                                     out_h=FRAME_H, out_w=FRAME_W,
                                     floor_clip_m=TD_FLOOR_CLIP):
    """Check if top-down camera sees soft low obstacles (dog bed, cushions) in forward region.

    Detects LOW-HEIGHT positive obstacles (above floor but below table/mast height) that
    existing reflexes miss. Dog beds, cushions, soft furniture typically 5-30cm high.

    This fills the gap between:
    - Near-field reflex (objects <30cm from camera, any height)
    - Overhang approach (elevated structures 30-70cm ahead at table height)
    - Floor classification (objects <5cm treated as floor noise)

    Args:
        verts: Nx3 point cloud from RS1 (X, Y, Z in metres, Z toward camera).
        near_m: Near distance threshold in metres (default 0.35m, after near-field zone).
        far_m: Far distance threshold in metres (default 1.00m, local sensing range).
        min_height_cm: Minimum obstacle height in cm above floor (default 5cm, above noise).
        max_height_cm: Maximum obstacle height in cm above floor (default 30cm, below table).
        forward_strip_row_min: Top edge of forward strip in image (default 10, after 180° rotation).
        forward_strip_row_max: Bottom edge of forward strip in image (default 80, wider than overhang).
        lateral_col_margin: Margin from image edges in columns (default 30px).
        min_pixels: Minimum number of low obstacle pixels to trigger (filters noise).
        out_h: Image height (default FRAME_H = 240).
        out_w: Image width (default FRAME_W = 320).
        floor_clip_m: Floor detection threshold in metres (default TD_FLOOR_CLIP).

    Returns:
        (triggered: bool, low_obs_count: int, median_height_cm: float)
        triggered: True if enough soft low obstacle pixels detected in forward strip.
        low_obs_count: Number of pixels matching low obstacle criteria.
        median_height_cm: Median height in cm of low obstacle pixels, or inf if none.

    Note:
        RS1 image after 180° rotation: top of image (low row indices) = forward region.
        This detector runs AFTER near-field and overhang checks.
        Height computed as (floor_clip_m - z) * 100 cm.
    """
    if len(verts) == 0:
        return False, 0, float('inf')

    verts = _clip_decimated_border(verts)
    
    # Extract coordinates
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    
    # Filter for valid depth in detection range (after near-field, before far)
    valid_z = (z > 0.01) & (z >= near_m) & (z <= far_m)
    
    if not np.any(valid_z):
        return False, 0, float('inf')
    
    # Compute height above floor in cm
    height_cm = (floor_clip_m - z) * 100.0
    
    # Filter for low obstacle height band (above floor noise, below table/mast)
    low_obs_mask = valid_z & (height_cm >= min_height_cm) & (height_cm <= max_height_cm)
    
    if not np.any(low_obs_mask):
        return False, 0, float('inf')
    
    # Project low obstacle points to image coordinates
    v_low_obs = verts[low_obs_mask]
    scale = np.float32(1.0 / TD_PX_SIZE)
    center = np.float32([out_w * 0.5, out_h * 0.5])
    
    # p = [col, row] in image before 180° rotation
    p = v_low_obs[:, :2] * scale + center
    cols, rows = p[:, 0], p[:, 1]
    
    # After 180° rotation: (row, col) → (out_h - 1 - row, out_w - 1 - col)
    rows_rotated = out_h - 1 - rows
    cols_rotated = out_w - 1 - cols
    
    # Check if points fall in forward strip ROI (after rotation)
    in_strip = (
        (rows_rotated >= forward_strip_row_min) &
        (rows_rotated <= forward_strip_row_max) &
        (cols_rotated >= lateral_col_margin) &
        (cols_rotated < out_w - lateral_col_margin)
    )
    
    low_obs_heights = height_cm[low_obs_mask][in_strip]
    
    if len(low_obs_heights) == 0:
        return False, 0, float('inf')
    
    low_obs_count = len(low_obs_heights)
    median_height_cm = float(np.median(low_obs_heights))
    
    triggered = low_obs_count >= min_pixels
    
    return triggered, low_obs_count, median_height_cm


def check_topdown_overhang_approach(verts, near_m=0.30, far_m=0.70,
                                     forward_strip_row_min=10, forward_strip_row_max=50,
                                     lateral_col_margin=30,
                                     min_pixels=80,
                                     out_h=FRAME_H, out_w=FRAME_W):
    """Check if top-down camera sees an overhang (table underside) in forward image strip.

    Uses a STRIP-BASED ROI in the RS1 depth image frame, not a 3D cone filter.
    The forward strip naturally excludes mast/self-geometry, which appears in the
    center/rear of the topdown image.

    Detects elevated structures (table undersides, shelves) at medium distance (30-70cm)
    in the forward approach region. This provides EARLIER warning than near-field reflex,
    allowing the robot to stop BEFORE committing to drive under overhangs.

    Args:
        verts: Nx3 point cloud from RS1 (X, Y, Z in metres, Z toward camera).
        near_m: Near distance threshold in metres (default 0.30m = 30cm).
        far_m: Far distance threshold in metres (default 0.70m = 70cm).
        forward_strip_row_min: Top edge of forward strip in image (default 10, after 180° rotation).
        forward_strip_row_max: Bottom edge of forward strip in image (default 50, after 180° rotation).
        lateral_col_margin: Margin from image edges in columns (default 30px).
        min_pixels: Minimum number of overhang pixels to trigger (filters noise).
        out_h: Image height (default FRAME_H = 240).
        out_w: Image width (default FRAME_W = 320).

    Returns:
        (triggered: bool, overhang_count: int, median_z: float)
        triggered: True if enough overhang pixels detected in forward strip.
        overhang_count: Number of pixels in forward strip at overhang distance.
        median_z: Median Z value of overhang pixels, or inf if none.

    Note:
        RS1 image after 180° rotation: top of image (low row indices) = forward region.
        Forward strip rows 10-50 correspond to ~0.10-0.40m ahead of robot center.
        Mast/self-geometry appears in center/rear (rows >80), outside forward strip.
        This strip-based ROI naturally avoids mast false positives.
    """
    if len(verts) == 0:
        return False, 0, float('inf')

    verts = _clip_decimated_border(verts)
    
    # Extract coordinates
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    
    # Filter for valid depth and overhang Z range
    valid_z = (z > 0.01) & (z >= near_m) & (z <= far_m)
    
    if not np.any(valid_z):
        return False, 0, float('inf')
    
    # Project valid overhang points to image coordinates
    v_overhang = verts[valid_z]
    scale = np.float32(1.0 / TD_PX_SIZE)
    center = np.float32([out_w * 0.5, out_h * 0.5])
    
    # p = [col, row] in image before 180° rotation
    p = v_overhang[:, :2] * scale + center
    cols, rows = p[:, 0], p[:, 1]
    
    # After 180° rotation: (row, col) → (out_h - 1 - row, out_w - 1 - col)
    # So forward region (large Y, large row) → top of rotated image (small row)
    rows_rotated = out_h - 1 - rows
    cols_rotated = out_w - 1 - cols
    
    # Check if points fall in forward strip ROI (after rotation)
    in_strip = (
        (rows_rotated >= forward_strip_row_min) &
        (rows_rotated <= forward_strip_row_max) &
        (cols_rotated >= lateral_col_margin) &
        (cols_rotated < out_w - lateral_col_margin)
    )
    
    overhang_z = v_overhang[in_strip, 2]
    
    if len(overhang_z) == 0:
        return False, 0, float('inf')
    
    overhang_count = len(overhang_z)
    median_z = float(np.median(overhang_z))
    
    triggered = overhang_count >= min_pixels
    
    return triggered, overhang_count, median_z


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

    verts = _clip_decimated_border(verts)

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





def process_rs1_sparse_reflexes(verts, work_verts=None,
                                 out_h=FRAME_H, out_w=FRAME_W,
                                 near_threshold_m=0.30, near_min_pixels=50,
                                 overhang_near_m=0.30, overhang_far_m=0.70,
                                 overhang_row_min=10, overhang_row_max=50,
                                 lateral_col_margin=30,
                                 overhang_min_pixels=80,
                                 floor_clip_m=TD_FLOOR_CLIP,
                                 large_obj_stride=8):
    """RS1 sparse reflexes only (near-field + overhang).
    
    Extracted from process_rs1_topdown for use with GPU heightmap.
    Soft-low can be derived from GPU heightmap separately if needed.
    """
    result = {
        'near_field': False, 'near_close_count': 0, 'near_min_z': float('inf'),
        'overhang': False, 'overhang_count': 0, 'overhang_median_z': float('inf'),
    }
    if verts is None or len(verts) == 0:
        return result
    
    v = _clip_decimated_border(verts, out=work_verts)
    z = v[:, 2]
    valid = z > 0.01
    if not np.any(valid):
        return result
    
    # Sparse large-object reflexes (near + overhang) with stride
    s = max(1, int(large_obj_stride))
    z_s = z[::s]
    valid_s = z_s > 0.01
    near_min_s = max(3, int(math.ceil(near_min_pixels / float(s))))
    ovh_min_s = max(3, int(math.ceil(overhang_min_pixels / float(s))))
    
    if np.any(valid_s):
        result['near_min_z'] = float(np.min(z_s[valid_s]))
        near_mask_s = valid_s & (z_s < near_threshold_m)
        result['near_close_count'] = int(np.sum(near_mask_s)) * s
        result['near_field'] = int(np.sum(near_mask_s)) >= near_min_s
        
        # Overhang: sparse mid-range + forward strip
        mid_s = valid_s & (z_s >= overhang_near_m) & (z_s <= overhang_far_m)
        if np.any(mid_s):
            idx = np.arange(0, len(v), s, dtype=np.int32)
            idx = idx[mid_s]
            v_mid = v[idx]
            z_mid = v_mid[:, 2]
            scale = np.float32(1.0 / TD_PX_SIZE)
            center = np.float32([out_w * 0.5, out_h * 0.5])
            p = v_mid[:, :2] * scale + center
            cols, rows = p[:, 0], p[:, 1]
            rows_rot = out_h - 1 - rows
            cols_rot = out_w - 1 - cols
            ovh_strip = (
                (rows_rot >= overhang_row_min) & (rows_rot <= overhang_row_max) &
                (cols_rot >= lateral_col_margin) & (cols_rot < out_w - lateral_col_margin)
            )
            if np.any(ovh_strip):
                ovh_z = z_mid[ovh_strip]
                result['overhang_count'] = int(len(ovh_z)) * s
                result['overhang_median_z'] = float(np.median(ovh_z))
                result['overhang'] = int(len(ovh_z)) >= ovh_min_s
    
    return result


def process_rs1_topdown(verts, out_obs, out_known, work_verts=None,
                        out_h=FRAME_H, out_w=FRAME_W,
                        near_threshold_m=0.30, near_min_pixels=50,
                        overhang_near_m=0.30, overhang_far_m=0.70,
                        overhang_row_min=10, overhang_row_max=50,
                        soft_near_m=0.35, soft_far_m=1.00,
                        soft_min_height_cm=5.0, soft_max_height_cm=30.0,
                        soft_row_min=10, soft_row_max=80,
                        lateral_col_margin=30,
                        soft_min_pixels=100, overhang_min_pixels=80,
                        floor_clip_m=TD_FLOOR_CLIP,
                        large_obj_stride=8):
    """RS1 topdown: dense ego+soft-low, sparse near/overhang.

    Tables/hands/overhangs are big uniform — James: only a few samples needed.
    Soft-low (dog-bed edges) + ego fill keep full mag=3 density for GSD.

    large_obj_stride: subsample factor for near + overhang only (default 8).
    Pixel thresholds for those reflexes scale down with the stride.
    """
    result = {
        'near_field': False, 'near_close_count': 0, 'near_min_z': float('inf'),
        'overhang': False, 'overhang_count': 0, 'overhang_median_z': float('inf'),
        'soft_low': False, 'soft_low_count': 0, 'soft_low_median_height': float('inf'),
    }
    out_obs.fill(0)
    out_known.fill(0)
    if verts is None or len(verts) == 0:
        return result

    v = _clip_decimated_border(verts, out=work_verts)
    z = v[:, 2]
    valid = z > 0.01
    if not np.any(valid):
        return result

    # --- Sparse large-object reflexes (near + overhang) ---
    # Stride the cloud; scale min_pixels so trigger rate stays similar.
    s = max(1, int(large_obj_stride))
    z_s = z[::s]
    valid_s = z_s > 0.01
    near_min_s = max(3, int(math.ceil(near_min_pixels / float(s))))
    ovh_min_s = max(3, int(math.ceil(overhang_min_pixels / float(s))))
    if np.any(valid_s):
        result['near_min_z'] = float(np.min(z_s[valid_s]))
        near_mask_s = valid_s & (z_s < near_threshold_m)
        # Report counts scaled back to full-cloud equivalent for logs/thresholds feel
        result['near_close_count'] = int(np.sum(near_mask_s)) * s
        result['near_field'] = int(np.sum(near_mask_s)) >= near_min_s

        # Overhang: sparse mid-range + forward strip
        mid_s = valid_s & (z_s >= overhang_near_m) & (z_s <= overhang_far_m)
        if np.any(mid_s):
            # Need xy of strided verts — index into v
            idx = np.arange(0, len(v), s, dtype=np.int32)
            idx = idx[mid_s]
            v_mid = v[idx]
            z_mid = v_mid[:, 2]
            scale = np.float32(1.0 / TD_PX_SIZE)
            center = np.float32([out_w * 0.5, out_h * 0.5])
            p = v_mid[:, :2] * scale + center
            cols, rows = p[:, 0], p[:, 1]
            rows_rot = out_h - 1 - rows
            cols_rot = out_w - 1 - cols
            ovh_strip = (
                (rows_rot >= overhang_row_min) & (rows_rot <= overhang_row_max) &
                (cols_rot >= lateral_col_margin) & (cols_rot < out_w - lateral_col_margin)
            )
            if np.any(ovh_strip):
                ovh_z = z_mid[ovh_strip]
                result['overhang_count'] = int(len(ovh_z)) * s
                result['overhang_median_z'] = float(np.median(ovh_z))
                result['overhang'] = int(len(ovh_z)) >= ovh_min_s

    # --- Dense ego scatter (GSD / obstacle map) ---
    scale = np.float32(1.0 / TD_PX_SIZE)
    center = np.float32([out_w * 0.5, out_h * 0.5])
    v_valid = v[valid]
    p_all = v_valid[:, :2] * scale + center
    with np.errstate(invalid='ignore'):
        ja, ia = p_all.astype(np.uint32).T
    ma = (ia < np.uint32(out_h)) & (ja < np.uint32(out_w))
    out_known[ia[ma], ja[ma]] = 255

    obstacle = valid & (z < floor_clip_m)
    if np.any(obstacle):
        v_obs = v[obstacle]
        p_obs = v_obs[:, :2] * scale + center
        height_cm = np.clip(
            ((floor_clip_m - v_obs[:, 2]) * 100).astype(np.int32),
            1, 100).astype(np.uint8)
        with np.errstate(invalid='ignore'):
            jo, io = p_obs.astype(np.uint32).T
        mo = (io < np.uint32(out_h)) & (jo < np.uint32(out_w))
        np.maximum.at(out_obs, (io[mo], jo[mo]), height_cm[mo])

    # --- Dense soft-low (dog-bed edges need detail) ---
    soft_band = valid & (z >= soft_near_m) & (z <= soft_far_m)
    if np.any(soft_band):
        v_soft = v[soft_band]
        z_soft = v_soft[:, 2]
        height_cm = (floor_clip_m - z_soft) * 100.0
        soft_height = (height_cm >= soft_min_height_cm) & (height_cm <= soft_max_height_cm)
        if np.any(soft_height):
            v_sh = v_soft[soft_height]
            h_sh = height_cm[soft_height]
            p = v_sh[:, :2] * scale + center
            cols, rows = p[:, 0], p[:, 1]
            rows_rot = out_h - 1 - rows
            cols_rot = out_w - 1 - cols
            soft_strip = (
                (rows_rot >= soft_row_min) & (rows_rot <= soft_row_max) &
                (cols_rot >= lateral_col_margin) & (cols_rot < out_w - lateral_col_margin)
            )
            if np.any(soft_strip):
                soft_heights = h_sh[soft_strip]
                result['soft_low_count'] = int(len(soft_heights))
                result['soft_low_median_height'] = float(np.median(soft_heights))
                result['soft_low'] = result['soft_low_count'] >= soft_min_pixels

    return result



class Vision:
    """Pre-allocated vision state. One capture thread; readers use .frames, .atlas, .timestamp."""

    def __init__(self, rs1_serial, rs2_serial, rgb1_device_id, headless=True,
                 slam_backend='self'):
        print("Vision: init start")
        self.rs1_serial = rs1_serial
        self.rs2_serial = rs2_serial
        self.rgb1_device_id = rgb1_device_id
        self._slam_backend = slam_backend

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
        self._topdown_ok = False
        self._topdown_known_px = 0
        self._topdown_lost_n = 0
        self._persistent_height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        self._topdown_near_field = False
        self._near_field_close_count = 0
        self._near_field_min_z = float('inf')
        self._topdown_overhang_approach = False
        self._overhang_approach_count = 0
        self._overhang_approach_median_z = float('inf')
        self._topdown_soft_low_obstacle = False
        self._soft_low_obstacle_count = 0
        self._soft_low_obstacle_median_height = float('inf')
        self._topdown_hazard = False
        self._topdown_hazard_reason = None
        self._topdown_hazard_corner_count = 0
        self._topdown_hazard_edge_count = 0
        self._topdown_hazard_detector = TopdownHazardDetector()
        self._safety = SafetyGuard()
        self._pose = PoseEstimator(wheelbase_m=WHEELBASE_M, wheel_radius_m=WHEEL_RADIUS_M)
        self._cuvslam = None
        self._global_map = PoseGraphSLAM()
        self._obs_mask, self._fw_cone_mask = self._build_obs_mask()
        self._free_range_mask = self._build_free_range_mask()
        self._wheelbase = None
        self._last_capture_time = None
        
        # Pre-allocated depth processing buffers (cleared per-frame, not re-allocated)
        self._z1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        self._k1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        # Mag=3 → ~45k verts; size for mag=1 headroom so copyto never reallocs
        self._rs1_work_verts = np.zeros((848 * 480 // 1, 3), dtype=np.float32)
        self._z2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
        self._k2 = np.zeros((FRAME_W, FRAME_H), dtype=np.uint8)
        self._kc_tmp = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        self._known_combined = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        self._obs_combined = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        self._obs_tmp = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        
        # SLAM lock state (critical for operator awareness)
        self._slam_locked = False
        self._slam_lock_reason = "not_initialized"
        self._slam_lock_lost_at = 0.0
        self._slam_lock_warnings = 0

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
        self._gpu.configure_depth_topdown(
            px_size=float(TD_PX_SIZE),
            floor_clip=float(TD_FLOOR_CLIP),
            out_h=FRAME_H, out_w=FRAME_W)
        self._gpu.configure_odom(fx=307.0, ds_factor=4, search=8)
        self._gpu.configure_gmap(MAP_W, MAP_H, FRAME_W, FRAME_H,
                                 ORIGIN_X, ORIGIN_Y, MAP_PX_SIZE)

        # Capture loop budget for 30 Hz frame timing
        CAPTURE_BUDGET_MS = 1000.0 / TARGET_FPS  # 33.33 ms at 30 Hz
        CAPTURE_SHED_THRESHOLD = 0.85  # Shed droppable stages if >85% budget used
        self._capture_budget = FrameBudget(
            budget_ms=CAPTURE_BUDGET_MS,
            shed_threshold=CAPTURE_SHED_THRESHOLD,
            use_capture_priorities=True
        )

        self._running = False
        self._thread = None
        self._odom_thread = None  # High-rate wheel+IMU integration thread
        self._rs1 = None
        self._rs2 = None
        self._webcam = None
        self._imu = None  # IMU pipeline for D435i Motion Module

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
    
    def _update_slam_lock_status(self):
        """Update SLAM lock status based on encoder health and tracking quality.
        
        SLAM is considered "locked" when:
        1. Encoders are working (enc_ok=True, age <1s)
        2. Visual odometry accepting frames OR wheel-only is acceptable
        3. Top-down depth is working
        
        When NOT locked:
        - Map-frame features are unreliable (keepouts, navigation)
        - Autonomous motion should be disabled
        - Operator must be clearly warned
        """
        reasons = []
        
        # Check encoder health
        enc_ok = False
        if self._wheelbase:
            health = self._wheelbase.get_encoder_health()
            enc_ok = health['encoder_ok'] and health['age_s'] < 1.0
            if not enc_ok:
                if not health['encoder_ok']:
                    reasons.append("encoder_failed")
                else:
                    reasons.append(f"encoder_stale_{health['age_s']:.1f}s")
        else:
            reasons.append("no_wheelbase")
        
        # Check tracking quality
        quality = self._pose.get_tracking_quality()
        visual_ok = quality['visual_accept_rate'] > 0.2  # At least 20% visual
        time_since_visual = quality['time_since_visual']
        
        # Wheel-only is acceptable if visual is recently working
        if not visual_ok and time_since_visual > 5.0:
            reasons.append(f"no_visual_{time_since_visual:.0f}s")
        
        # Check top-down (already tracked via _topdown_ok)
        if not self._topdown_ok:
            reasons.append("topdown_lost")
        
        # Determine lock status
        was_locked = self._slam_locked
        self._slam_locked = enc_ok and (visual_ok or time_since_visual < 5.0) and self._topdown_ok
        self._slam_lock_reason = ", ".join(reasons) if reasons else "ok"
        
        # Log state changes
        if was_locked and not self._slam_locked:
            self._slam_lock_lost_at = time.monotonic()
            self._slam_lock_warnings += 1
            print(f"🔴 SLAM LOCK LOST: {self._slam_lock_reason}")
            print(f"   ⚠️  Map-frame navigation DISABLED until lock restored")
        elif not was_locked and self._slam_locked:
            downtime = time.monotonic() - self._slam_lock_lost_at if self._slam_lock_lost_at > 0 else 0
            print(f"🟢 SLAM LOCKED (was unlocked {downtime:.1f}s)")
            print(f"   ✓ Encoders working, tracking quality good")

    def set_wheelbase(self, wb):
        """Provide wheelbase reference for wheel odometry fusion."""
        self._wheelbase = wb
        if self._odom_thread is not None:
            self._odom_thread.set_wheelbase(wb)

    def start(self):
        if self._running:
            return

        if not HAS_RS:
            print("vision: pyrealsense2 not available; running stub")
            self._running = True
            self._thread = threading.Thread(target=self._stub_loop, daemon=True)
            self._thread.start()
            return

        use_ir = (self._slam_backend == 'cuvslam')
        try:
            if self.rs1_serial:
                self._rs1 = RSCamera(self.rs1_serial, compute_pointcloud=True)
            if self.rs2_serial:
                self._rs2 = RSCamera(self.rs2_serial, compute_pointcloud=True,
                                     capture_ir=use_ir)
        except Exception as e:
            print("vision: RealSense init failed: %s" % e)
            self._running = True
            self._thread = threading.Thread(target=self._stub_loop, daemon=True)
            self._thread.start()
            return

        # IMU pipeline (separate from depth/color to avoid frame starvation)
        # Initialize BEFORE cuVSLAM so we can pass it to the tracker
        if self.rs2_serial:
            try:
                self._imu = IMUPipeline(self.rs2_serial)
                if self._imu.ok:
                    print("vision: IMU pipeline active (gyro 200 Hz, accel 250 Hz)")
            except Exception as e:
                print("vision: IMU init failed (%s) — pose fusion will use wheel+visual only" % e)
                self._imu = None
        
        if self._slam_backend == 'cuvslam':
            try:
                from cuvslam_tracker import CuVSLAMTracker
                rs2_profile = self._rs2.profile if self._rs2 else None
                # Pass IMU pipeline to cuVSLAM for stereo-inertial mode
                self._cuvslam = CuVSLAMTracker(
                    rs2_profile=rs2_profile,
                    imu_pipeline=self._imu if self._imu and self._imu.ok else None)
                print("vision: cuVSLAM backend active")
            except Exception as e:
                print("vision: cuVSLAM init failed (%s) — falling back to self" % e)
                self._slam_backend = 'self'
                self._cuvslam = None

        self._webcam = WebCam(self.rgb1_device_id)
        self._grab_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="camgrab")

        # Start high-rate odometry thread BEFORE capture loop
        # This ensures pose is continuously updated even if capture is slow
        self._odom_thread = OdomThread(
            pose_estimator=self._pose,
            wheelbase=self._wheelbase,
            imu_pipeline=self._imu,
            target_hz=100.0)
        self._odom_thread.start()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._extras_running = True
        self._extras_thread = threading.Thread(
            target=self._vision_extras_loop, name="vision-extras", daemon=True)
        self._extras_thread.start()
        print("vision: odom thread started (target 100 Hz, wheel+IMU integration)")
        print("vision: extras thread started (~3 Hz RGB hazards/faces side loop)")
        print("vision: capture thread started (slam=%s)" % self._slam_backend)

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

        # RS2 forward points: green = floor (phys_h < 5cm), red = obstacle
        # X-axis = world forward distance, Y-axis = phys_h
        if self._rs2 and self._rs2.ok and self._rs2.verts is not None:
            pts = self._rs2.verts.reshape(-1, 3)
            valid = pts[:, 2] > 0.1
            pts = pts[valid]
            if len(pts) > 0:
                step = max(1, len(pts) // 2000)
                sampled = pts[::step]
                phys_h = cam_h - sampled[:, 1] * cos_p - sampled[:, 2] * sin_p
                fwd_d = -sin_p * sampled[:, 1] + cos_p * sampled[:, 2]
                for i in range(len(sampled)):
                    px, py = to_px(float(fwd_d[i]), float(phys_h[i]))
                    if 0 <= px < W and 0 <= py < H:
                        h = float(phys_h[i])
                        if h < 0.05:
                            sv[py, px] = [0, 200, 100]
                        else:
                            sv[py, px] = [200, 60, 60]

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

        Uses a percentile-based floor finder that works even when the
        default pitch parameters are far off.  Takes the lowest 20% of
        phys_h values in depth bins as floor, then regresses to find the
        pitch correction.
        """
        pts = verts.reshape(-1, 3)
        valid = ((pts[:, 2] > 0.3) & (pts[:, 2] < 2.0) &
                 (np.abs(pts[:, 0]) < 0.25))
        pts = pts[valid]
        if len(pts) < 500:
            print("pitch_cal: too few points (%d)" % len(pts))
            return

        sin_p, cos_p = float(_fw_sin_pitch), float(_fw_cos_pitch)
        cam_h = float(FW_CAM_HEIGHT)

        phys_h = cam_h - pts[:, 1] * cos_p - pts[:, 2] * sin_p

        # Find floor using lowest 20th-percentile in each depth bin.
        # Robust to furniture/obstacles which have HIGHER phys_h.
        n_bins = 8
        z_edges = np.linspace(0.3, 2.0, n_bins + 1)
        floor_mask = np.zeros(len(pts), dtype=bool)
        for b in range(n_bins):
            in_bin = (pts[:, 2] >= z_edges[b]) & (pts[:, 2] < z_edges[b + 1])
            if np.sum(in_bin) < 20:
                continue
            h_bin = phys_h[in_bin]
            thresh = np.percentile(h_bin, 20)
            floor_mask[in_bin] = h_bin <= thresh

        fp = pts[floor_mask]
        fh = phys_h[floor_mask]
        if len(fp) < 100:
            print("pitch_cal: too few floor pts (%d)" % len(fp))
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

        n_collected = len(self._pitch_cal_deltas)
        if n_collected <= 3 or n_collected % 5 == 0:
            print("pitch_cal: frame %d/%d  delta=%.3f° h_off=%.1fcm (%d floor pts)" % (
                n_collected, 20, math.degrees(delta), h_off * 100, len(fp)))

        if n_collected < 20:
            return

        med_delta = float(np.median(self._pitch_cal_deltas))
        med_hoff = float(np.median(self._pitch_cal_hoffsets))
        err_deg = math.degrees(med_delta)

        print("pitch_cal: RESULT error=%.3f° h_offset=%.1fcm (%d samples)" % (
            err_deg, med_hoff * 100, n_collected))

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


    def _vision_extras_loop(self):
        """~3 Hz side loop: RGB image detections off the 30Hz critical path.

        Chessboard / bump RGB, faces, gestures belong here (or on i777).
        Sticky results feed self._topdown_hazard* for safety. Named keepout
        `checkered_door` is the primary avoid for the door mat.
        """
        interval = 1.0 / 3.0
        while getattr(self, '_extras_running', False) and getattr(self, '_running', False):
            t0 = time.monotonic()
            try:
                color = None
                if self._rs1 and self._rs1.ok and self._rs1.color is not None:
                    color = self._rs1.color.copy()
                if color is not None and getattr(self, '_topdown_hazard_detector', None) is not None:
                    rs1_rgb_rotated = color[::-1, ::-1]
                    hazard_triggered, hazard_reason = self._topdown_hazard_detector.check(
                        rs1_rgb_rotated)
                    self._topdown_hazard = hazard_triggered
                    self._topdown_hazard_reason = hazard_reason
                    self._topdown_hazard_corner_count = self._topdown_hazard_detector.corner_count
                    self._topdown_hazard_edge_count = self._topdown_hazard_detector.edge_count
                    n = getattr(self, '_extras_hazard_log_n', 0)
                    if hazard_triggered and (n % 9 == 0):
                        print("vision-extras: hazard=%s corners=%d edges=%d (3Hz)"
                              % (hazard_reason, self._topdown_hazard_corner_count,
                                 self._topdown_hazard_edge_count))
                    self._extras_hazard_log_n = n + 1
            except Exception as e:
                if not getattr(self, '_extras_err_n', 0):
                    print("vision-extras error: %s" % e)
                self._extras_err_n = getattr(self, '_extras_err_n', 0) + 1
            dt = time.monotonic() - t0
            time.sleep(max(0.0, interval - dt))

    def _capture_loop(self):
        black = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        _loop_times = []
        _stage_times = []  # Track: grab, pose+hazard, rs1_checks, rs2, obs_comb, odom, gmap, safety, render
        _use_cuvslam = (self._cuvslam is not None)

        while self._running:
            _t0 = time.monotonic()
            _t_start = _t0
            self._capture_budget.reset_frame()

            try:
                # Critical path: RealSense only (parallel).
                _rs = [c for c in (self._rs1, self._rs2) if c]
                if len(_rs) <= 1:
                    for c in _rs:
                        c.grab()
                else:
                    list(self._grab_pool.map(lambda c: c.grab(), _rs))
            except Exception as e:
                # Never let a camera glitch kill the capture thread (blank atlas forever).
                if not getattr(self, '_grab_err_n', 0):
                    print("vision: grab error (continuing): %s" % e)
                self._grab_err_n = getattr(self, '_grab_err_n', 0) + 1
            _t_grab = time.monotonic()

            # --- Pose update (cuVSLAM or wheel+visual from odom thread) ---
            if _use_cuvslam:
                fused_yaw, fused_fwd = 0.0, 0.0
                if (self._rs2 and self._rs2.ok
                        and self._rs2.ir_left is not None):
                    ts_ns = int(time.monotonic() * 1e9)
                    result = self._cuvslam.track(
                        self._rs2.ir_left, self._rs2.ir_right, ts_ns)
                    if result is not None:
                        fused_yaw, fused_fwd = result
                cap_x = self._cuvslam.x
                cap_y = self._cuvslam.y
                cap_theta = self._cuvslam.theta
                pose_src = self._cuvslam
                self._last_capture_time = time.monotonic()
            else:
                # Defer pose snapshot to after odometry section (below)
                pose_src = self._pose

            # RGB image hazards (checkered door mat, bump edges, faces, gestures)
            # intentionally NOT in the 30Hz capture path — see _vision_extras_loop (~3Hz).
            # Capture only consumes the latest async result (sticky). Depth reflexes stay here.
            _t_hazard = time.monotonic()
            
            # RS1 top-down depth → (obstacles, known), rotate 180°
            # Use preallocated buffers (clear instead of allocate)
            self._z1[:] = 0
            self._k1[:] = 0
            if self._rs1 and self._rs1.ok and self._rs1.verts is not None:
                # GPU heightmap @30Hz for policy + sparse CPU reflexes
                v_clean = _clip_decimated_border(self._rs1.verts, out=self._rs1_work_verts)
                _gpu_result = self._gpu.depth_topdown_gpu(v_clean)
                
                if _gpu_result is not None:
                    # GPU path: heightmap from GPU, sparse reflexes from CPU
                    self._z1[:], self._k1[:] = _gpu_result
                    
                    # Sparse reflexes (near + overhang) on CPU stride=8
                    _rs1_sparse = process_rs1_sparse_reflexes(
                        self._rs1.verts, work_verts=self._rs1_work_verts)
                    
                    # Soft-low not needed on GPU path: dog bed shows as normal
                    # heightmap obstacle. Prior "soft-low" pain was empty-frame nav,
                    # not missing a special detector (James 2026-09-07).
                    _rs1 = {
                        'near_field': _rs1_sparse['near_field'],
                        'near_close_count': _rs1_sparse['near_close_count'],
                        'near_min_z': _rs1_sparse['near_min_z'],
                        'overhang': _rs1_sparse['overhang'],
                        'overhang_count': _rs1_sparse['overhang_count'],
                        'overhang_median_z': _rs1_sparse['overhang_median_z'],
                        'soft_low': False,
                        'soft_low_count': 0,
                        'soft_low_median_height': float('inf'),
                    }
                else:
                    # CPU fallback: full process_rs1_topdown (sparse + dense)
                    _rs1 = process_rs1_topdown(
                        self._rs1.verts, self._z1, self._k1,
                        work_verts=self._rs1_work_verts)
                
                self._topdown_near_field = _rs1['near_field']
                self._near_field_close_count = _rs1['near_close_count']
                self._near_field_min_z = _rs1['near_min_z']
                self._topdown_overhang_approach = _rs1['overhang']
                self._overhang_approach_count = _rs1['overhang_count']
                self._overhang_approach_median_z = _rs1['overhang_median_z']
                self._topdown_soft_low_obstacle = _rs1['soft_low']
                self._soft_low_obstacle_count = _rs1['soft_low_count']
                self._soft_low_obstacle_median_height = _rs1['soft_low_median_height']
                # Rate-limited reflex logs (every 90 frames while sticky)
                if _rs1['near_field']:
                    self._near_field_log_n = getattr(self, '_near_field_log_n', 0) + 1
                    if self._near_field_log_n == 1 or self._near_field_log_n % 90 == 0:
                        print("vision: NEAR-FIELD REFLEX triggered — "
                              "close_px=%d min_z=%.3fm"
                              % (_rs1['near_close_count'], _rs1['near_min_z']))
                elif getattr(self, '_near_field_log_n', 0):
                    self._near_field_log_n = 0
                if _rs1['overhang']:
                    self._overhang_approach_log_n = getattr(self, '_overhang_approach_log_n', 0) + 1
                    if self._overhang_approach_log_n == 1 or self._overhang_approach_log_n % 90 == 0:
                        print("vision: OVERHANG APPROACH detected — "
                              "ovh_px=%d median_z=%.3fm"
                              % (_rs1['overhang_count'], _rs1['overhang_median_z']))
                elif getattr(self, '_overhang_approach_log_n', 0):
                    self._overhang_approach_log_n = 0
                if _rs1['soft_low']:
                    self._soft_low_obstacle_log_n = getattr(self, '_soft_low_obstacle_log_n', 0) + 1
                    if self._soft_low_obstacle_log_n == 1 or self._soft_low_obstacle_log_n % 90 == 0:
                        print("vision: SOFT LOW OBSTACLE detected — "
                              "low_obs_px=%d median_h=%.1fcm"
                              % (_rs1['soft_low_count'], _rs1['soft_low_median_height']))
                elif getattr(self, '_soft_low_obstacle_log_n', 0):
                    self._soft_low_obstacle_log_n = 0
            else:
                self._topdown_near_field = False
                self._near_field_close_count = 0
                self._near_field_min_z = float('inf')
                self._topdown_overhang_approach = False
                self._overhang_approach_count = 0
                self._overhang_approach_median_z = float('inf')
                self._topdown_soft_low_obstacle = False
                self._soft_low_obstacle_count = 0
                self._soft_low_obstacle_median_height = float('inf')
                
            obs1 = self._z1[::-1, ::-1]
            known1 = self._k1[::-1, ::-1]
            # Top-down depth is ground truth for open-space. No valid known
            # coverage ⇒ immobilize (empty map must NOT look like free space).
            self._topdown_known_px = int(np.count_nonzero(known1))
            TOPDOWN_MIN_KNOWN = 800  # px; healthy runs are typically >>10k
            self._topdown_ok = bool(
                self._rs1 is not None
                and getattr(self._rs1, 'ok', False)
                and self._rs1.verts is not None
                and self._topdown_known_px >= TOPDOWN_MIN_KNOWN
            )
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
            # DROPPABLE: Can fallback to topdown-only if budget tight
            # Use preallocated buffers (clear instead of allocate)
            self._z2[:] = 0
            self._k2[:] = 0
            _raw_scatter = None
            _dbg = self.debug_depth
            if self._capture_budget.should_run("rs2_process"):
                with self._capture_budget.stage("rs2_process"):
                    if self._rs2 and self._rs2.ok and self._rs2.verts is not None:
                        if getattr(self, '_pitch_cal_request', False):
                            self._calibrate_rs2_pitch(self._rs2.verts)
                        rs2_clean = _clip_decimated_border(self._rs2.verts)
                        _gpu_result = self._gpu.depth_forward_gpu(
                            rs2_clean, y_offset=RS2_EXTRINSIC_Y, debug=_dbg)
                        if _gpu_result is not None:
                            self._z2[:], self._k2[:], _raw_scatter = _gpu_result
            obs2 = np.rot90(self._z2, k=-1)
            known2 = np.rot90(self._k2, k=-1)
            _t_rs2 = time.monotonic()
            _t_depth = _t_rs2  # Keep for backwards compat with old logging

            # --- Combine into ego-space (obs_combined, known_combined) ---
            fw_dx, fw_dy = int(TD_X_OFFSET) + FW_TD_X_DELTA, int(FW_Y_OFFSET)
            td_dx = int(TD_X_OFFSET)

            # Use preallocated buffer (clear instead of allocate)
            self._kc_tmp[:] = 0
            _blit(self._kc_tmp, known2, fw_dx, fw_dy)
            if not hasattr(self, '_kdiag_n'):
                self._kdiag_n = 0
            self._kdiag_n += 1
            if self._kdiag_n <= 2 or self._kdiag_n % 300 == 0:
                _k2nz = int(np.count_nonzero(known2))
                _kcnz_pre = int(np.count_nonzero(self._kc_tmp))
                np.bitwise_and(self._kc_tmp, self._fw_cone_mask, out=self._kc_tmp)
                _kcnz_post = int(np.count_nonzero(self._kc_tmp))
                print("fw_known: known2=%d blit=%d after_cone=%d "
                      "fw_dx=%d fw_dy=%d" % (_k2nz, _kcnz_pre, _kcnz_post,
                                              fw_dx, fw_dy))
            else:
                np.bitwise_and(self._kc_tmp, self._fw_cone_mask, out=self._kc_tmp)

            # Use preallocated buffers (clear instead of allocate)
            self._known_combined[:] = 0
            _blit(self._known_combined, known1, td_dx)
            np.maximum(self._known_combined, self._kc_tmp, out=self._known_combined)

            self._obs_combined[:] = 0
            _blit(self._obs_combined, obs1, td_dx)
            self._obs_tmp[:] = 0
            _blit(self._obs_tmp, obs2, fw_dx, fw_dy)
            np.maximum(self._obs_combined, self._obs_tmp, out=self._obs_combined)

            np.bitwise_and(self._obs_combined, self._obs_mask, out=self._obs_combined)
            np.bitwise_and(self._known_combined, self._obs_mask, out=self._known_combined)

            self._obs_combined[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = 0
            self._known_combined[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = 255

            _t_obs = time.monotonic()

            # --- Odometry (self-made stack only; cuVSLAM handled above) ---
            # NEW ARCHITECTURE: Wheel+IMU integration happens on dedicated odom thread (~100-200 Hz).
            # Capture loop samples latest pose snapshot without blocking odom thread.
            # Visual odometry (future work): can still run here and apply corrections
            # via a thread-safe mechanism, or stay at 30 Hz as "frame-tied" refinement.
            fused_yaw, fused_fwd = 0.0, 0.0
            if not _use_cuvslam:
                # Frame-tied visual odometry → queue into fast odom thread (no double wheel integrate).
                if self._rs2 and self._rs2.ok and self._odom_thread is not None:
                    fw_gray = cv2.cvtColor(self._rs2.color, cv2.COLOR_RGB2GRAY)
                    _odom_result = self._gpu.odom_gpu(fw_gray)
                    if _odom_result is not None:
                        vis_yaw, vis_fwd, vis_conf = _odom_result
                        fused_yaw, fused_fwd = vis_yaw, vis_fwd
                        self._odom_thread.apply_visual_correction(
                            vis_yaw, vis_fwd, vis_conf)

                # Sample latest pose from odom thread (wheel+IMU @~100Hz + pending VO)
                if self._odom_thread:
                    cap_x, cap_y, cap_theta, _, using_encoder_feedback = (
                        self._odom_thread.get_pose_snapshot())
                else:
                    cap_x, cap_y, cap_theta = self._pose.x, self._pose.y, self._pose.theta
                    using_encoder_feedback = False

                now = time.monotonic()
                self._last_capture_time = now

            _t_odom = time.monotonic()

            if not hasattr(self, '_odom_log_n'):
                self._odom_log_n = 0
            self._odom_log_n += 1
            if self._odom_log_n % 90 == 0:
                if _use_cuvslam:
                    metrics = self._cuvslam.get_slam_metrics()
                    print("cuvslam: pose=(%.3f,%.3f,%.1f°) "
                          "tracking=%s lc=%d frames=%d" % (
                              self._cuvslam.x, self._cuvslam.y,
                              np.degrees(self._cuvslam.theta),
                              metrics.get('tracking'),
                              metrics.get('lc_count', 0),
                              metrics.get('frame_count', 0)))
                else:
                    wb = self._wheelbase
                    enc_info = 'no_wb'
                    if wb:
                        enc_ok = getattr(wb, '_enc_ok', '?')
                        enc_age = (time.monotonic() -
                                   getattr(wb, '_enc_last_good', 0))
                        enc_info = 'enc=%s age=%.1fs' % (enc_ok, enc_age)
                    print("odom: pose=(%.3f,%.3f,%.1f°) enc_fb=%s %s"
                          % (self._pose.x, self._pose.y,
                             np.degrees(self._pose.theta),
                             using_encoder_feedback, enc_info))

            # --- Height diagnostic (every 30 frames) ---
            if not hasattr(self, '_hdiag_n'):
                self._hdiag_n = 0
            self._hdiag_n += 1
            if self._hdiag_n % 300 == 0:
                om = self._obs_combined[self._obs_combined > 0]
                if len(om) > 0:
                    bins = [0, 5, 10, 20, 30, 50, 70, 100, 256]
                    h, _ = np.histogram(om, bins)
                    print("heights: " + " ".join(
                        "%d-%d:%d" % (bins[i], bins[i+1]-1, h[i])
                        for i in range(len(h))))

            # --- Update global occupancy map ---
            rcx_f = float(CROSSHAIR_CX + ROBOT_CX_OFF)
            rcy_f = float(CROSSHAIR_CY)

            # CRITICAL SAFETY: Only update SLAM/gmap with reliable pose
            # When using commanded velocity fallback (encoders failed), pose is NOT ground truth.
            # Scenario: robot stuck, wheels spinning → commanded vel says moving but actually stationary.
            # Never feed bad pose to SLAM or it will corrupt the map.
            
            skip_slam_update = False
            skip_reason = None
            
            if not _use_cuvslam:  # Only applies to self-SLAM with wheel odom
                if not using_encoder_feedback:
                    skip_slam_update = True
                    skip_reason = "encoder_fallback"
                elif pose_src.is_stuck:
                    skip_slam_update = True
                    skip_reason = "stuck"
            
            # GMAP updates - DROPPABLE (expensive GPU ops, non-safety-critical)
            if not skip_slam_update and self._capture_budget.should_run("gmap"):
                with self._capture_budget.stage("gmap"):
                    self._gpu.gmap_update_gpu(
                        self._obs_combined, self._known_combined,
                        cap_x, cap_y, cap_theta,
                        rcx_f, rcy_f, float(TD_PX_SIZE),
                        free_range_mask=self._free_range_mask)
                    self._global_map.keyframe_check(
                        self._obs_combined, self._known_combined,
                        cap_x, cap_y, cap_theta,
                        rcx_f, rcy_f, float(TD_PX_SIZE))
                    
                    # Sync GPU map after loop closure rebuild
                    if self._global_map.needs_gpu_sync():
                        cpu_map, cpu_height = self._global_map.get_cpu_map()
                        self._gpu.gmap_reset(cpu_map, cpu_height)
                        self._global_map.clear_gpu_sync_flag()
                        print("vision: GPU gmap synchronized after loop closure")
            else:
                # Log skip first time and periodically
                if not hasattr(self, '_slam_skip_warned') or self._slam_skip_warned != skip_reason:
                    if skip_slam_update:
                        print(f"⚠️  SLAM update skipped: {skip_reason} (pose not ground truth)")
                    self._slam_skip_warned = skip_reason
            
            _t_gmap_up = time.monotonic()

            ego_proj = self._gpu.gmap_project_gpu(
                pose_src.x, pose_src.y, pose_src.theta,
                rcx_f, rcy_f, float(TD_PX_SIZE), FRAME_H, FRAME_W)

            self._persistent_obs[:] = 0
            self._persistent_height[:] = 0
            if ego_proj is not None:
                self._persistent_obs[ego_proj < 90] = 255
            # obs_combined is height-cm (1..100) where obstacles exist
            self._persistent_obs[self._obs_combined > 0] = 255
            self._persistent_height[:] = self._obs_combined.astype(np.uint8, copy=False)
            self._persistent_obs[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = 0
            self._persistent_height[FOOT_Y0:FOOT_Y1, FOOT_X0:FOOT_X1] = 0

            # ── Check SLAM lock status (encoder + tracking quality) ───
            self._update_slam_lock_status()
            
            self._safety.update(self._persistent_obs, fused_yaw, fused_fwd,
                                height_cm=self._persistent_height,
                                topdown_near_field=self._topdown_near_field,
                                topdown_overhang_approach=self._topdown_overhang_approach,
                                topdown_hazard=self._topdown_hazard,
                                topdown_soft_low_obstacle=self._topdown_soft_low_obstacle)
            
            # Hard immobilize conditions (CRITICAL SAFETY ONLY)
            # Philosophy: Immobilize only for immediate safety hazards.
            # When SLAM bad → block map-frame ops, but local navigation still works.
            immobilize = False
            immobilize_reason = None
            
            # CRITICAL: Check if stuck (wheels spinning but not moving)
            if pose_src.is_stuck:
                immobilize = True
                immobilize_reason = f"STUCK (wheels spinning, no motion)"
            
            if not self._topdown_ok:
                # No top-down depth reading - safety critical
                immobilize = True
                immobilize_reason = "TOPDOWN LOST"
                self._topdown_lost_n = getattr(self, '_topdown_lost_n', 0) + 1
                if self._topdown_lost_n == 1 or self._topdown_lost_n % 60 == 0:
                    print(
                        "vision: TOPDOWN LOST — immobilized "
                        "(rs1_ok=%s known_px=%d)"
                        % (getattr(self._rs1, 'ok', None), self._topdown_known_px)
                    )
            else:
                if getattr(self, '_topdown_lost_n', 0):
                    print(
                        "vision: TOPDOWN OK — drive re-enabled "
                        "(known_px=%d after %d lost frames)"
                        % (self._topdown_known_px, self._topdown_lost_n)
                    )
                self._topdown_lost_n = 0
            
            # Note: SLAM lock status tracked but does NOT immobilize.
            # When SLAM not locked:
            #   ✅ Local navigation works (obstacle avoid, near-field, MPPI ~1m goals)
            #   ❌ Map-frame operations disabled (global nav, map keepouts)
            # This allows graceful degradation: roam works even without full SLAM.
            if not self._slam_locked:
                if not hasattr(self, '_slam_unlock_logged') or not self._slam_unlock_logged:
                    print(f"⚠️  SLAM NOT LOCKED ({self._slam_lock_reason})")
                    print(f"   → Map-frame operations disabled (keepouts, global nav)")
                    print(f"   → Local navigation still works (obstacle avoid, MPPI)")
                    self._slam_unlock_logged = True
            else:
                if hasattr(self, '_slam_unlock_logged') and self._slam_unlock_logged:
                    print(f"✓ SLAM LOCKED — map-frame operations re-enabled")
                self._slam_unlock_logged = False
            
            if immobilize:
                # STUCK: kill forward only so HouseBot RECOVER can reverse/turn off a lip.
                # TOPDOWN LOST / other hard immobilize: freeze all axes.
                self._safety._fwd_scale = 0.0
                stuck_only = bool(immobilize_reason and immobilize_reason.startswith("STUCK"))
                if not stuck_only:
                    self._safety._bwd_scale = 0.0
                    self._safety._ang_scale = 0.0
                # Log first immobilization and periodically
                if not hasattr(self, '_immobilize_warned') or self._immobilize_warned != immobilize_reason:
                    if stuck_only:
                        print(f"⚠️  vision: {immobilize_reason} — fwd disabled; reverse/turn allowed for recover")
                    else:
                        print(f"⚠️  vision: {immobilize_reason} — autonomous motion disabled")
                    self._immobilize_warned = immobilize_reason
            else:
                self._immobilize_warned = None
            
            _t_safety = time.monotonic()

            # --- GPU atlas render (DROPPABLE: viz for humans/i777, not policy) ---
            # Policy needs heightmap/obs/safety (already published above) at 30Hz.
            # Atlas paint can run at lower rate when budget tight.
            # Expected: render ~9ms; if budget exceeded, skip to stay under 33.3ms target.
            atlas = None
            if self._capture_budget.should_run("render"):
                with self._capture_budget.stage("render"):
                    trail = pose_src.get_world_history()
                    rgb1 = self._webcam.color if (self._webcam and self._webcam.ok) else black
                    rgbd1 = self._rs1.color[::-1, ::-1] if (self._rs1 and self._rs1.ok) else black
                    rgbd2 = self._rs2.color if (self._rs2 and self._rs2.ok) else black

                    bat_frac = 0.0
                    if self._wheelbase:
                        pct = self._wheelbase.battery_pct
                        bat_frac = max(0.0, min(1.0, pct / 100.0)) if pct >= 0 else 0.0

                    atlas = self._gpu.render(
                        pose_src.x, pose_src.y, pose_src.theta,
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

                        dbg2 = np.zeros((self._known_combined.shape[0], self._known_combined.shape[1], 3), dtype=np.uint8)
                        dbg2[(self._known_combined > 0) & (self._obs_combined == 0)] = [0, 255, 0]
                        dbg2[self._obs_combined > 0] = [255, 0, 0]
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
            else:
                # Render skipped — only update timestamp (atlas/frames stay stale)
                with self._lock:
                    self.timestamp = time.time()
            _t_render = time.monotonic()
            _t_end = _t_render

            # === Stage timing collection ===
            # Webcam last: never on RS grab clock or pose bucket
            if self._webcam is not None:
                try:
                    self._webcam.grab()
                except Exception:
                    pass

            _stage_times.append((
                (_t_grab - _t_start) * 1000.0,      # grab
                (_t_hazard - _t_grab) * 1000.0,     # hazard_rgb + pose
                (_t_rs1 - _t_hazard) * 1000.0,      # rs1_checks (near/overhang/soft + depth_topdown)
                (_t_rs2 - _t_rs1) * 1000.0,         # rs2_process (GPU depth)
                (_t_obs - _t_rs2) * 1000.0,         # obs_combine
                (_t_odom - _t_obs) * 1000.0,        # odom
                (_t_gmap_up - _t_odom) * 1000.0,    # gmap
                (_t_safety - _t_gmap_up) * 1000.0,  # safety
                (_t_render - _t_safety) * 1000.0,   # render
            ))
            
            # Detailed report every 90 frames
            if len(_stage_times) % 90 == 0 and len(_stage_times) >= 90:
                recent = np.array(_stage_times[-90:])
                avg = np.mean(recent, axis=0)
                p95 = np.percentile(recent, 95, axis=0)
                total = np.sum(avg)
                labels = ["grab", "pose+hazard", "rs1_checks", "rs2_gpu", 
                         "obs_comb", "odom", "gmap", "safety", "render"]
                print("=" * 80)
                print(f"CAPTURE TIMING (last 90 frames): TOTAL={total:.1f}ms ({1000/total:.1f} Hz)")
                print("=" * 80)
                print(f"{'STAGE':15s} {'MEAN':>8s} {'P95':>8s} {'%TOTAL':>8s}")
                print("-" * 80)
                for i, label in enumerate(labels):
                    pct = (avg[i] / total) * 100.0
                    print(f"{label:15s} {avg[i]:7.1f}ms {p95[i]:7.1f}ms {pct:7.1f}%")
                print("=" * 80)
                print(f"Target: 33.3ms/frame (30 Hz). Current: {total:.1f}ms ({1000/total:.1f} Hz)")
                print("=" * 80)

            _loop_times.append((_t_grab - _t0, _t_rs1 - _t_grab,
                                _t_depth - _t_rs1, _t_obs - _t_depth,
                                _t_odom - _t_obs, _t_gmap_up - _t_odom,
                                _t_safety - _t_gmap_up,
                                _t_end - _t_safety))
            if len(_loop_times) % 300 == 0:
                avg = np.mean(_loop_times[-300:], axis=0) * 1000
                total_avg = sum(avg)
                
                # Get budget stats
                lifetime = self._capture_budget.get_lifetime_stats()
                shed_counts = lifetime.get("shed_counts", {})
                
                # Build shed info
                shed_info = ""
                if shed_counts:
                    shed_parts = ["%s=%d" % (k, v) for k, v in sorted(shed_counts.items())]
                    shed_info = "  shed:[%s]" % ",".join(shed_parts)
                
                print("capture: grab=%.1f rs1=%.1f rs2=%.1f obs=%.1f odom=%.1f "
                      "gmap=%.1f safety=%.1f render=%.1f "
                      "TOTAL=%.1fms (budget %.1fms @ 30Hz)%s" % (*avg, total_avg, 33.3, shed_info))

    def stop(self):
        self._extras_running = False
        pool = getattr(self, "_grab_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._grab_pool = None
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._odom_thread:
            self._odom_thread.stop()
        if self._rs1:
            self._rs1.stop()
        if self._rs2:
            self._rs2.stop()
        if self._imu:
            self._imu.stop()
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
    def topdown_depth_ok(self):
        """True when RS1 top-down depth produced enough known open/occ pixels."""
        return bool(getattr(self, '_topdown_ok', False))

    @property
    def topdown_known_px(self):
        return int(getattr(self, '_topdown_known_px', 0))

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
    def slam_locked(self):
        """True when SLAM is locked (encoders + tracking working).
        
        When False, map-frame navigation features should be disabled:
        - Do NOT mark keepouts
        - Do NOT trust global map for navigation
        - Do NOT use autonomous wander
        
        Use this before any map-frame operation.
        """
        return self._slam_locked
    
    @property
    def slam_lock_reason(self):
        """Human-readable reason for SLAM lock status."""
        return self._slam_lock_reason
    
    @property
    def is_stuck(self):
        """True when robot is stuck (wheels spinning but not moving).
        
        CRITICAL SAFETY: When stuck:
        - Forward scale is forced to 0 (no more digging into the lip)
        - Reverse/turn stay available for HouseBot RECOVER
        - SLAM/gmap updates are skipped while stuck
        """
        return self._pose.is_stuck if hasattr(self._pose, 'is_stuck') else False
    
    @property
    def stuck_count(self):
        """Number of times stuck has been detected."""
        return self._pose.stuck_count if hasattr(self._pose, 'stuck_count') else 0
    
    @property
    def safety_throttled(self):
        return self._safety.is_throttled

    @property
    def topdown_near_field(self):
        """True when top-down camera detected a close object (<30cm) overhead."""
        return bool(getattr(self, '_topdown_near_field', False))

    @property
    def near_field_close_count(self):
        """Number of close pixels when near-field reflex is active."""
        return int(getattr(self, '_near_field_close_count', 0))

    @property
    def near_field_min_z(self):
        """Minimum (closest) Z distance in metres when near-field detected."""
        return float(getattr(self, '_near_field_min_z', float('inf')))

    @property
    def topdown_overhang_approach(self):
        """True when top-down camera detected an overhang approach (30-70cm) ahead."""
        return bool(getattr(self, '_topdown_overhang_approach', False))

    @property
    def overhang_approach_count(self):
        """Number of overhang pixels when overhang approach is detected."""
        return int(getattr(self, '_overhang_approach_count', 0))

    @property
    def overhang_approach_median_z(self):
        """Median Z distance in metres of overhang approach points."""
        return float(getattr(self, '_overhang_approach_median_z', float('inf')))

    @property
    def topdown_soft_low_obstacle(self):
        """True when top-down camera detected soft low obstacle (dog bed, cushion) ahead."""
        return bool(getattr(self, '_topdown_soft_low_obstacle', False))

    @property
    def soft_low_obstacle_count(self):
        """Number of low obstacle pixels when soft low obstacle is detected."""
        return int(getattr(self, '_soft_low_obstacle_count', 0))

    @property
    def soft_low_obstacle_median_height(self):
        """Median height in cm of soft low obstacle points."""
        return float(getattr(self, '_soft_low_obstacle_median_height', float('inf')))
