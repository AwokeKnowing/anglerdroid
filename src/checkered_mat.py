"""checkered_mat.py – Ego-frame wood bump + checkered mat detector (no SLAM).

Vision-based hard-stop when wood floor bump/lip or checkered door mat is ahead.
Uses RS1 top-down RGB (NOT webcam); no map, no pose, no disks. Classical CV for Jetson.

Primary hazards detected from RS1 top-down view:
  (a) Wood floor bump/lip: texture transition, edge, color shift
  (b) Checkered door mat: alternating black-white grid pattern

Detection runs in near-field ROI (forward of robot in RS1 view after 180° rotation).
Returns (triggered: bool, score: float, meta: dict) where score ∈ [0,1].
"""

import numpy as np
import cv2

# ── Tunable constants ──
# ROI: forward portion of RS1 top-down view (floor ahead of robot).
# RS1 is rotated 180° in vision.py, so after rotation:
#   - Right side = forward (robot nose direction)
#   - Left side = backward (robot tail)
# ROI_FORWARD_FRAC covers the forward (right) portion after rotation.
ROI_FORWARD_START = 0.50  # Start at center (robot body)
ROI_FORWARD_END = 0.90    # End near right edge (forward near-field, ~40cm ahead)

# Detection threshold: score ≥ this → trigger hard-stop
MIN_SCORE = 0.20  # Lower threshold for wood bump (subtler than strong checkered)

# Hysteresis: once triggered, require score to drop below (MIN_SCORE - HYSTERESIS) to clear
HYSTERESIS = 0.10

# Corner density floor: checkered has many corners (grid intersections)
MIN_CORNER_DENSITY = 0.08  # corners per 100 pixels

# Line detection: checkered has straight edges (Hough)
HOUGH_THRESHOLD = 15
HOUGH_MIN_LINE_LEN = 10
HOUGH_MAX_LINE_GAP = 5


def detect_checkered_mat(rgb_bgr, roi_fwd_start=ROI_FORWARD_START, roi_fwd_end=ROI_FORWARD_END,
                         min_score=MIN_SCORE, prev_triggered=False):
    """Detect wood bump or checkered mat in RS1 top-down forward view (no SLAM).

    Detects TWO hazard types from RS1 RGB:
      (a) Wood floor bump/lip: strong edge, texture transition, color shift
      (b) Checkered door mat: alternating black-white grid pattern

    Args:
        rgb_bgr: HxWx3 uint8 RGB or BGR (RS1 top-down color, after 180° rotation).
        roi_fwd_start: Forward ROI start as fraction of W (0.0-1.0, default 0.5 = center).
        roi_fwd_end: Forward ROI end as fraction of W (0.0-1.0, default 0.9 = near edge).
        min_score: Threshold to trigger (0.0-1.0).
        prev_triggered: Previous detection state (for hysteresis).

    Returns:
        (triggered: bool, score: float, meta: dict)
        meta = {"wood_score": float, "checker_score": float, "edges": int, "corners": int}
    """
    if rgb_bgr is None or rgb_bgr.size == 0:
        return False, 0.0, {}

    h, w = rgb_bgr.shape[:2]
    if h < 20 or w < 20:
        return False, 0.0, {}

    # Extract forward ROI (right portion after RS1 180° rotation = ahead of robot)
    x0 = max(0, int(w * roi_fwd_start))
    x1 = min(w, int(w * roi_fwd_end))
    if x1 <= x0:
        return False, 0.0, {}
    roi = rgb_bgr[:, x0:x1]  # Full height, forward columns

    # Grayscale + CLAHE (adaptive contrast)
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if rgb_bgr.shape[2] == 3 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ══════════════════════════════════════════════════════════════════════
    # WOOD BUMP / LIP DETECTION: Strong horizontal edge, texture transition
    # ══════════════════════════════════════════════════════════════════════
    # Wood bump appears as:
    #   - Strong horizontal edge (floor level change)
    #   - Texture transition (smooth → rough or color shift)
    #   - High edge response in middle of ROI (not at frame border)
    
    edges = cv2.Canny(enhanced, 50, 150)
    # Horizontal edge detector: look for strong H edges in middle rows (not border)
    h_roi, w_roi = roi.shape[:2]
    mid_y0 = int(h_roi * 0.3)  # Skip top 30% (may have shadows)
    mid_y1 = int(h_roi * 0.7)  # Skip bottom 30% (robot body)
    mid_edges = edges[mid_y0:mid_y1, :]
    
    # Detect horizontal lines (wood bump = strong H line across ROI)
    h_lines = cv2.HoughLinesP(mid_edges, rho=1, theta=np.pi/180, threshold=20,
                               minLineLength=int(w_roi * 0.3), maxLineGap=10)
    h_line_count = 0
    if h_lines is not None:
        for line in h_lines:
            try:
                coords = line[0] if len(line) > 0 and isinstance(line[0], (list, tuple, np.ndarray)) else line
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[:4]
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if angle < 20 or angle > 160:  # Horizontal-ish
                        h_line_count += 1
            except (TypeError, IndexError, ValueError):
                continue
    
    # Wood bump score: strong H lines + high edge density in mid-region
    edge_density = float(np.sum(mid_edges > 0)) / max(1, mid_edges.size)
    wood_h_line_score = min(1.0, h_line_count / 3.0)  # 3+ H lines → strong
    wood_edge_score = min(1.0, edge_density * 10.0)  # Dense edges
    wood_score = 0.6 * wood_h_line_score + 0.4 * wood_edge_score

    # ══════════════════════════════════════════════════════════════════════
    # CHECKERED MAT DETECTION: Alternating grid pattern
    # ══════════════════════════════════════════════════════════════════════
    # ── Component 1: Corner density (checkered has grid intersections) ──
    corners = cv2.goodFeaturesToTrack(enhanced, maxCorners=500, qualityLevel=0.01,
                                      minDistance=5, blockSize=3)
    corner_count = len(corners) if corners is not None else 0
    roi_area = h_roi * w_roi
    corner_density = (corner_count / max(1, roi_area)) * 100.0
    density_score = min(1.0, corner_density / (MIN_CORNER_DENSITY * 100.0))

    # ── Component 2: Perpendicular lines (checkered has H+V grid) ──
    # Reuse edges from wood detection above
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP)
    checker_h_lines, v_lines = 0, 0
    if lines is not None:
        for line in lines:
            try:
                coords = line[0] if len(line) > 0 and isinstance(line[0], (list, tuple, np.ndarray)) else line
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[:4]
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if angle < 25 or angle > 155:
                        checker_h_lines += 1
                    elif 65 < angle < 115:
                        v_lines += 1
            except (TypeError, IndexError, ValueError):
                continue

    # Grid orthogonality: checkered needs BOTH H and V lines
    h_ok = checker_h_lines >= 2
    v_ok = v_lines >= 2
    grid_orth = 1.0 if (h_ok and v_ok) else 0.5 if (h_ok or v_ok) else 0.0
    line_count = (checker_h_lines + v_lines)
    line_score = min(1.0, line_count / 20.0)

    # ── Component 3: Local variance (checkered has high contrast) ──
    lap = cv2.Laplacian(enhanced, cv2.CV_64F)
    lap_var = float(lap.var())
    variance_score = min(1.0, lap_var / 400.0)

    # ── Component 4: Corner regularity (checkered corners are evenly distributed) ──
    regularity = 0.0
    if corners is not None and len(corners) > 10:
        grid_h, grid_w = 4, 4
        cell_h, cell_w = max(1, h_roi // grid_h), max(1, w_roi // grid_w)
        cell_counts = np.zeros((grid_h, grid_w))
        for corner in corners:
            x, y = corner.ravel()
            gi = min(int(y / cell_h), grid_h - 1) if cell_h > 0 else 0
            gj = min(int(x / cell_w), grid_w - 1) if cell_w > 0 else 0
            cell_counts[gi, gj] += 1
        non_empty = int(np.sum(cell_counts > 0))
        regularity = min(1.0, non_empty / 8.0)

    # ── Checkered score: weight orthogonality heavily to filter stripes/noise ──
    checker_score = (0.20 * density_score +
                     0.20 * line_score +
                     0.35 * grid_orth +
                     0.15 * variance_score +
                     0.10 * regularity)
    
    # Apply penalty if no proper grid (prevents stripe false positives)
    if grid_orth < 0.5:
        checker_score *= 0.7

    # ══════════════════════════════════════════════════════════════════════
    # COMBINED SCORE: MAX of wood bump OR checkered mat
    # ══════════════════════════════════════════════════════════════════════
    # Trigger on EITHER hazard type
    score = max(wood_score, checker_score)

    # Hysteresis: once triggered, harder to clear (prevents flicker)
    threshold = min_score - HYSTERESIS if prev_triggered else min_score
    triggered = score >= threshold

    meta = {
        "wood_score": float(wood_score),
        "checker_score": float(checker_score),
        "h_line_count": h_line_count,
        "edge_density": float(edge_density),
        "corners": corner_count,
        "checker_h": checker_h_lines,
        "v_lines": v_lines,
        "variance": lap_var,
    }
    return triggered, float(score), meta
