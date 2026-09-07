"""checkered_mat.py – Ego-frame floor hazard hard-stop detection (topdown RGB).

Detects floor hazards in RS1 top-down RealSense RGB view (sensor frame):
  1. Wood bump / threshold (where wheels get stuck spinning)
  2. Checkered floor mat pattern (door mat area)

Triggers forward hard-stop (fwd_scale=0) when bump or checkered detected.
This is a REFLEX that works WITHOUT SLAM or map-based keepouts.

Design:
    - PRIMARY SOURCE: RS1 color (top-down RealSense RGB / rgbd1)
    - Detects bump via edge detection in near/forward region
    - Detects door mat via brown border around checkered area (HSV); chessboard corners deprecated
    - Analyzes forward region of topdown view (where robot will drive)
    - Detection → fwd_scale=0, allows reverse/turn if rear is clear
    - Tunable thresholds for both bump and checkerboard

Typical use:
    detector = TopdownHazardDetector()
    triggered, reason = detector.check(rs1_color_frame)  # RS1 topdown RGB
    if triggered:
        fwd_scale = 0.0  # Stop forward motion
"""

import cv2
import numpy as np


def detect_brown_border(rgb_roi, min_brown_frac=0.02, min_contour_area=400):
    """Detect dark-brown transition strip / frame around door checkered mat.

    Returns (detected: bool, score: float, brown_px: int).
    Cheap HSV mask + contour; for ~3Hz extras loop, not 30Hz capture.
    """
    if rgb_roi is None or rgb_roi.size == 0:
        return False, 0.0, 0
    hsv = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2HSV)
    # Wood / dark brown border on tan carpet (tuned for household door mat photo)
    lower = np.array([5, 40, 25], dtype=np.uint8)
    upper = np.array([25, 200, 140], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    brown_px = int(np.count_nonzero(mask))
    frac = brown_px / float(mask.size)
    if frac < min_brown_frac:
        return False, frac, brown_px
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, frac, brown_px
    best = 0.0
    h, w = mask.shape[:2]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_contour_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 8 or bh < 8:
            continue
        extent = area / float(max(1, bw * bh))
        aspect = max(bw, bh) / float(min(bw, bh))
        width_cov = bw / float(w)
        score = width_cov * (1.0 + 0.3 * min(aspect, 8) / 8.0) * (0.5 + 0.5 * extent)
        if score > best:
            best = score
    detected = best >= 0.25 and frac >= min_brown_frac
    return detected, float(best), brown_px


# ── Default parameters ──
DEFAULT_CHECKERBOARD_ROWS = 6      # Internal corners (7x7 squares → 6x6 corners)
DEFAULT_CHECKERBOARD_COLS = 6
DEFAULT_FORWARD_FRACTION = 0.4     # Analyze forward 40% of topdown view
DEFAULT_MIN_CORNERS = 4            # Minimum corners to confirm detection
DEFAULT_CORNER_QUALITY = 0.1       # cornerSubPix quality threshold
DEFAULT_BUMP_EDGE_THRESH = 50      # Canny edge threshold for bump detection
DEFAULT_BUMP_MIN_EDGES = 1         # Minimum rows with strong horizontal edges (unused, kept for API compat)


class TopdownHazardDetector:
    """Detects floor hazards in RS1 top-down RealSense RGB (sensor/ego frame).
    
    Detects TWO hazard types in the forward region of topdown view:
      1. Wood bump / threshold (via edge detection)
      2. Checkered floor mat pattern (via corner detection)
    
    Triggers forward hard-stop when either hazard detected.
    
    Attributes:
        checkerboard_rows: Number of internal corner rows to detect
        checkerboard_cols: Number of internal corner columns to detect
        forward_fraction: Fraction of image to analyze in forward direction
        min_corners: Minimum corners required to trigger checkerboard detection
        bump_edge_thresh: Canny edge threshold for bump detection
        bump_min_edges: Minimum edge pixels to confirm bump
        
        detected: True if hazard detected in most recent check()
        detection_reason: 'bump' or 'checkered' or None
        corner_count: Number of corners found (checkered)
        edge_count: Number of edge pixels found (bump)
    """
    
    def __init__(self,
                 checkerboard_rows=DEFAULT_CHECKERBOARD_ROWS,
                 checkerboard_cols=DEFAULT_CHECKERBOARD_COLS,
                 forward_fraction=DEFAULT_FORWARD_FRACTION,
                 min_corners=DEFAULT_MIN_CORNERS,
                 corner_quality=DEFAULT_CORNER_QUALITY,
                 bump_edge_thresh=DEFAULT_BUMP_EDGE_THRESH,
                 bump_min_edges=DEFAULT_BUMP_MIN_EDGES):
        """Initialize topdown hazard detector (bump + checkered).
        
        Args:
            checkerboard_rows: Internal corner rows (N+1 squares → N corners)
            checkerboard_cols: Internal corner columns
            forward_fraction: Fraction of image to analyze in forward direction [0.0, 1.0]
            min_corners: Minimum corners to trigger checkerboard detection
            corner_quality: Corner refinement quality (OpenCV winSize factor)
            bump_edge_thresh: Canny edge threshold for bump detection
            bump_min_edges: Minimum edge pixels to confirm bump
        """
        self.checkerboard_rows = checkerboard_rows
        self.checkerboard_cols = checkerboard_cols
        self.forward_fraction = max(0.1, min(1.0, forward_fraction))
        self.min_corners = max(1, min_corners)
        self.corner_quality = corner_quality
        self.bump_edge_thresh = bump_edge_thresh
        self.bump_min_edges = bump_min_edges
        
        # Detection state
        self.detected = False
        self.detection_reason = None  # 'bump' or 'checkered' or None
        self.corner_count = 0
        self.edge_count = 0
        self.detection_confidence = 0.0
        
        # Checkerboard detection flags
        self._flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                       cv2.CALIB_CB_NORMALIZE_IMAGE +
                       cv2.CALIB_CB_FAST_CHECK)
        
        # Corner refinement criteria
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                         30, 0.001)
        
        # Detection history for temporal filtering (reduce flicker)
        self._history_size = 3
        self._detection_history = []
    
    def check(self, rgb_frame):
        """Check if floor hazard (bump or checkered mat) is visible in topdown RGB.
        
        Args:
            rgb_frame: HxWx3 uint8 RGB image (PRIMARY: RS1 color / topdown RealSense RGB)
        
        Returns:
            (triggered: bool, reason: str or None)
            triggered: True if hazard detected (forward hard-stop)
            reason: 'bump' or 'checkered' or None
        """
        if rgb_frame is None or rgb_frame.size == 0:
            self.detected = False
            self.detection_reason = None
            self.corner_count = 0
            self.edge_count = 0
            self.detection_confidence = 0.0
            return False, None
        
        # Extract forward region of topdown image (where robot will drive)
        h, w = rgb_frame.shape[:2]
        forward_h = int(h * self.forward_fraction)
        if forward_h < 20:  # Minimum usable height
            self.detected = False
            self.detection_reason = None
            return False, None
        
        # Crop to forward region (near/forward area in topdown view)
        forward_region = rgb_frame[:forward_h, :, :]
        
        # ── 1. Brown border around door checkered mat (preferred cue) ──
        # Photo cue: dark brown transition strip framing black/white squares on carpet.
        # Cheap HSV — for ~3Hz extras loop. Named keepout checkered_door is still primary.
        border_hit, border_score, brown_px = detect_brown_border(forward_region)
        self.corner_count = brown_px  # reuse field as brown pixel count for logs
        self.detection_confidence = min(1.0, border_score)
        checkered_detected = border_hit

        gray = cv2.cvtColor(forward_region, cv2.COLOR_RGB2GRAY)

        # ── 2. Check for BUMP (edge detection) ONLY if no brown border ──
        # Wood bump / threshold shows as strong horizontal edges in topdown view
        if not checkered_detected:
            edges = cv2.Canny(gray, self.bump_edge_thresh, self.bump_edge_thresh * 2)
            
            # Look for HORIZONTAL edges that span the image width (not scattered edges)
            # Sum edges along each row → find rows with many edge pixels
            row_edge_counts = np.sum(edges > 0, axis=1)  # Count edge pixels per row
            
            # Bump = continuous horizontal line with strong edges spanning width
            # Require: (1) high % of width covered, (2) multiple consecutive rows
            width_thresh = w * 0.6  # At least 60% of image width (conservative, avoid noise)
            strong_rows = row_edge_counts >= width_thresh
            
            # Count consecutive runs of strong rows (bump = at least 2-3 consecutive rows)
            consecutive = 0
            max_consecutive = 0
            for has_edges in strong_rows:
                if has_edges:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            
            self.edge_count = max_consecutive  # Longest run of consecutive edge rows
            bump_detected = max_consecutive >= 1  # At least 1 row with strong horizontal edge
        else:
            # Skip bump detection if checkerboard found (checkerboards have edges too)
            self.edge_count = 0
            bump_detected = False
        
        # ── 3. Combine detections with temporal filtering ──
        # Priority: checkerboard > bump
        detected_now = checkered_detected or bump_detected
        reason_now = ('brown_border' if checkered_detected else ('bump' if bump_detected else None))
        
        # Temporal filtering: require consistent detection to reduce flicker
        self._detection_history.append(detected_now)
        if len(self._detection_history) > self._history_size:
            self._detection_history.pop(0)
        
        # Trigger if detected in majority of recent frames
        detection_votes = sum(self._detection_history)
        self.detected = detection_votes >= (len(self._detection_history) / 2)
        self.detection_reason = reason_now if self.detected else None
        
        return self.detected, self.detection_reason
    
    def reset(self):
        """Reset detection state and history."""
        self.detected = False
        self.detection_reason = None
        self.corner_count = 0
        self.edge_count = 0
        self.detection_confidence = 0.0
        self._detection_history.clear()
    
    def get_debug_info(self):
        """Get debug information about current detection state.
        
        Returns:
            dict with detection state, corner count, edge count, reason, etc.
        """
        return {
            'detected': self.detected,
            'reason': self.detection_reason,
            'corner_count': self.corner_count,
            'edge_count': self.edge_count,
            'confidence': self.detection_confidence,
            'history': self._detection_history.copy(),
            'params': {
                'checkerboard_rows': self.checkerboard_rows,
                'checkerboard_cols': self.checkerboard_cols,
                'forward_fraction': self.forward_fraction,
                'min_corners': self.min_corners,
                'bump_edge_thresh': self.bump_edge_thresh,
                'bump_min_edges': self.bump_min_edges,
            }
        }


# Legacy alias for backward compatibility
CheckeredMatDetector = TopdownHazardDetector


def check_topdown_hazard(rgb_frame,
                        checkerboard_rows=DEFAULT_CHECKERBOARD_ROWS,
                        checkerboard_cols=DEFAULT_CHECKERBOARD_COLS,
                        forward_fraction=DEFAULT_FORWARD_FRACTION,
                        min_corners=DEFAULT_MIN_CORNERS,
                        bump_edge_thresh=DEFAULT_BUMP_EDGE_THRESH,
                        bump_min_edges=DEFAULT_BUMP_MIN_EDGES):
    """Convenience function for one-shot topdown hazard detection (bump + checkered).
    
    For persistent detection with temporal filtering, use TopdownHazardDetector.
    
    Args:
        rgb_frame: HxWx3 uint8 RGB image (RS1 topdown RGB)
        checkerboard_rows: Internal corner rows
        checkerboard_cols: Internal corner columns
        forward_fraction: Fraction of image to analyze in forward direction
        min_corners: Minimum corners to trigger checkered detection
        bump_edge_thresh: Canny edge threshold for bump detection
        bump_min_edges: Minimum edge pixels to confirm bump
    
    Returns:
        (triggered, reason, corner_count, edge_count)
    """
    detector = TopdownHazardDetector(
        checkerboard_rows=checkerboard_rows,
        checkerboard_cols=checkerboard_cols,
        forward_fraction=forward_fraction,
        min_corners=min_corners,
        bump_edge_thresh=bump_edge_thresh,
        bump_min_edges=bump_min_edges
    )
    
    triggered, reason = detector.check(rgb_frame)
    return triggered, reason, detector.corner_count, detector.edge_count


# Legacy convenience function (for backward compatibility)
def check_checkered_mat(rgb_frame,
                       checkerboard_rows=DEFAULT_CHECKERBOARD_ROWS,
                       checkerboard_cols=DEFAULT_CHECKERBOARD_COLS,
                       forward_fraction=DEFAULT_FORWARD_FRACTION,
                       min_corners=DEFAULT_MIN_CORNERS):
    """Legacy convenience function for checkered mat detection only.
    
    Use check_topdown_hazard() for bump + checkered detection.
    """
    triggered, reason, corner_count, _ = check_topdown_hazard(
        rgb_frame, checkerboard_rows, checkerboard_cols,
        forward_fraction, min_corners
    )
    return triggered, corner_count, 1.0 if triggered else 0.0
