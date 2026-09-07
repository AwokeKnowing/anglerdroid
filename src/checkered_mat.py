"""checkered_mat.py – Ego-frame floor hazard hard-stop detection (topdown RGB).

Detects floor hazards in RS1 top-down RealSense RGB view (sensor frame):
  1. Brown wood border (HSV + rectangular frame) — PRIMARY for door mat
  2. Wood bump / threshold (where wheels get stuck spinning)
  3. Checkered floor mat pattern (fallback for other checkered zones)

Triggers forward hard-stop (fwd_scale=0) when any hazard detected.
This is a REFLEX that works WITHOUT SLAM or map-based keepouts.

Design:
    - PRIMARY SOURCE: RS1 color (top-down RealSense RGB / rgbd1)
    - Detects brown border via HSV color + rectangular contour (door mat with wood frame)
    - Detects bump via edge detection in near/forward region
    - Detects checkerboard via OpenCV corner detection (optional fallback, expensive ~34ms)
    - Analyzes forward region of topdown view (where robot will drive)
    - Detection → fwd_scale=0, allows reverse/turn if rear is clear
    - Tunable thresholds for all three detection types

Typical use:
    detector = TopdownHazardDetector()
    triggered, reason = detector.check(rs1_color_frame)  # RS1 topdown RGB
    if triggered:
        fwd_scale = 0.0  # Stop forward motion
        # reason will be 'brown_border', 'bump', or 'checkered'
"""

import cv2
import numpy as np


# ── Default parameters ──
DEFAULT_CHECKERBOARD_ROWS = 6      # Internal corners (7x7 squares → 6x6 corners)
DEFAULT_CHECKERBOARD_COLS = 6
DEFAULT_FORWARD_FRACTION = 0.4     # Analyze forward 40% of topdown view
DEFAULT_MIN_CORNERS = 4            # Minimum corners to confirm detection
DEFAULT_CORNER_QUALITY = 0.1       # cornerSubPix quality threshold
DEFAULT_BUMP_EDGE_THRESH = 50      # Canny edge threshold for bump detection
DEFAULT_BUMP_MIN_EDGES = 1         # Minimum rows with strong horizontal edges (unused, kept for API compat)

# ── Brown border detection parameters ──
DEFAULT_BROWN_HSV_LOWER = (5, 40, 30)    # HSV lower bound for brown wood border
DEFAULT_BROWN_HSV_UPPER = (25, 255, 180) # HSV upper bound for brown wood border
DEFAULT_BROWN_MIN_PERIMETER = 300        # Minimum perimeter for valid border rectangle
DEFAULT_BROWN_MIN_AREA = 3000            # Minimum area for valid border rectangle
DEFAULT_BROWN_ASPECT_RATIO_MIN = 0.3     # Min width/height ratio for rectangle (allows narrow vertical)
DEFAULT_BROWN_ASPECT_RATIO_MAX = 4.0     # Max width/height ratio for rectangle (allows wide horizontal)


def _detect_brown_border(gray_frame, rgb_frame, 
                        brown_hsv_lower=DEFAULT_BROWN_HSV_LOWER,
                        brown_hsv_upper=DEFAULT_BROWN_HSV_UPPER,
                        min_perimeter=DEFAULT_BROWN_MIN_PERIMETER,
                        min_area=DEFAULT_BROWN_MIN_AREA,
                        aspect_ratio_min=DEFAULT_BROWN_ASPECT_RATIO_MIN,
                        aspect_ratio_max=DEFAULT_BROWN_ASPECT_RATIO_MAX):
    """Detect brown wood border around checkered floor mat (HSV-based).
    
    Looks for a closed rectangular frame of brown wood/transition border
    around a checkered mat on tan carpet.
    
    Args:
        gray_frame: HxW grayscale image
        rgb_frame: HxWx3 RGB image
        brown_hsv_lower: Lower HSV bound for brown wood (H, S, V)
        brown_hsv_upper: Upper HSV bound for brown wood (H, S, V)
        min_perimeter: Minimum perimeter for valid border rectangle
        min_area: Minimum area for valid border rectangle
        aspect_ratio_min: Minimum width/height ratio
        aspect_ratio_max: Maximum width/height ratio
    
    Returns:
        (detected: bool, confidence: float, perimeter: float)
        detected: True if brown rectangular border found
        confidence: 0.0-1.0 based on shape quality
        perimeter: perimeter of detected rectangle (0 if none)
    """
    if rgb_frame is None or rgb_frame.size == 0:
        return False, 0.0, 0.0
    
    # Convert RGB to HSV for brown color detection
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    
    # Create mask for brown color range (wood border)
    brown_mask = cv2.inRange(hsv, brown_hsv_lower, brown_hsv_upper)
    
    # Morphological operations to close gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in brown mask
    contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, 0.0, 0.0
    
    # Look for largest rectangular contour (border frame)
    best_rect = None
    best_confidence = 0.0
    best_perimeter = 0.0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter < min_perimeter:
            continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for 4-sided polygon (rectangle)
        if len(approx) >= 4:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / float(h) if h > 0 else 0.0
            
            # Check aspect ratio (roughly rectangular, not too elongated)
            if aspect_ratio_min <= aspect_ratio <= aspect_ratio_max:
                # Compute rectangularity (how well the contour fills its bounding box)
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0.0
                
                # Confidence based on rectangularity and perimeter
                confidence = rectangularity * min(1.0, perimeter / (min_perimeter * 2.0))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_perimeter = perimeter
                    best_rect = (x, y, w, h)
    
    if best_rect is not None and best_confidence > 0.3:  # Threshold for detection
        return True, best_confidence, best_perimeter
    
    return False, 0.0, 0.0


class TopdownHazardDetector:
    """Detects floor hazards in RS1 top-down RealSense RGB (sensor/ego frame).
    
    Detects THREE hazard types in the forward region of topdown view:
      1. Brown wood border (HSV color + rectangular frame detection) - PRIMARY
      2. Wood bump / threshold (via edge detection)
      3. Checkered floor mat pattern (via corner detection) - FALLBACK
    
    Triggers forward hard-stop when any hazard detected.
    
    Attributes:
        checkerboard_rows: Number of internal corner rows to detect
        checkerboard_cols: Number of internal corner columns to detect
        forward_fraction: Fraction of image to analyze in forward direction
        min_corners: Minimum corners required to trigger checkerboard detection
        bump_edge_thresh: Canny edge threshold for bump detection
        bump_min_edges: Minimum edge pixels to confirm bump
        brown_hsv_lower: HSV lower bound for brown border detection
        brown_hsv_upper: HSV upper bound for brown border detection
        
        detected: True if hazard detected in most recent check()
        detection_reason: 'brown_border' or 'bump' or 'checkered' or None
        corner_count: Number of corners found (checkered)
        edge_count: Number of edge pixels found (bump)
        brown_perimeter: Perimeter of detected brown border
        brown_confidence: Confidence of brown border detection (0.0-1.0)
    """
    
    def __init__(self,
                 checkerboard_rows=DEFAULT_CHECKERBOARD_ROWS,
                 checkerboard_cols=DEFAULT_CHECKERBOARD_COLS,
                 forward_fraction=DEFAULT_FORWARD_FRACTION,
                 min_corners=DEFAULT_MIN_CORNERS,
                 corner_quality=DEFAULT_CORNER_QUALITY,
                 bump_edge_thresh=DEFAULT_BUMP_EDGE_THRESH,
                 bump_min_edges=DEFAULT_BUMP_MIN_EDGES,
                 brown_hsv_lower=DEFAULT_BROWN_HSV_LOWER,
                 brown_hsv_upper=DEFAULT_BROWN_HSV_UPPER,
                 brown_min_perimeter=DEFAULT_BROWN_MIN_PERIMETER,
                 brown_min_area=DEFAULT_BROWN_MIN_AREA):
        """Initialize topdown hazard detector (brown border + bump + checkered).
        
        Args:
            checkerboard_rows: Internal corner rows (N+1 squares → N corners)
            checkerboard_cols: Internal corner columns
            forward_fraction: Fraction of image to analyze in forward direction [0.0, 1.0]
            min_corners: Minimum corners to trigger checkerboard detection
            corner_quality: Corner refinement quality (OpenCV winSize factor)
            bump_edge_thresh: Canny edge threshold for bump detection
            bump_min_edges: Minimum edge pixels to confirm bump
            brown_hsv_lower: HSV lower bound for brown wood border (H, S, V)
            brown_hsv_upper: HSV upper bound for brown wood border (H, S, V)
            brown_min_perimeter: Minimum perimeter for brown border rectangle
            brown_min_area: Minimum area for brown border rectangle
        """
        self.checkerboard_rows = checkerboard_rows
        self.checkerboard_cols = checkerboard_cols
        self.forward_fraction = max(0.1, min(1.0, forward_fraction))
        self.min_corners = max(1, min_corners)
        self.corner_quality = corner_quality
        self.bump_edge_thresh = bump_edge_thresh
        self.bump_min_edges = bump_min_edges
        self.brown_hsv_lower = brown_hsv_lower
        self.brown_hsv_upper = brown_hsv_upper
        self.brown_min_perimeter = brown_min_perimeter
        self.brown_min_area = brown_min_area
        
        # Detection state
        self.detected = False
        self.detection_reason = None  # 'brown_border' or 'bump' or 'checkered' or None
        self.corner_count = 0
        self.edge_count = 0
        self.detection_confidence = 0.0
        self.brown_perimeter = 0.0
        self.brown_confidence = 0.0
        
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
        """Check if floor hazard (brown border, bump, or checkered mat) is visible in topdown RGB.
        
        Args:
            rgb_frame: HxWx3 uint8 RGB image (PRIMARY: RS1 color / topdown RealSense RGB)
        
        Returns:
            (triggered: bool, reason: str or None)
            triggered: True if hazard detected (forward hard-stop)
            reason: 'brown_border' or 'bump' or 'checkered' or None
        """
        if rgb_frame is None or rgb_frame.size == 0:
            self.detected = False
            self.detection_reason = None
            self.corner_count = 0
            self.edge_count = 0
            self.detection_confidence = 0.0
            self.brown_perimeter = 0.0
            self.brown_confidence = 0.0
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
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(forward_region, cv2.COLOR_RGB2GRAY)
        
        # ── 1. Check for BROWN BORDER (HSV + rectangular frame) FIRST ──
        # This is the PRIMARY detector for James' checkered door mat with brown wood border
        brown_detected, brown_conf, brown_perim = _detect_brown_border(
            gray, forward_region,
            brown_hsv_lower=self.brown_hsv_lower,
            brown_hsv_upper=self.brown_hsv_upper,
            min_perimeter=self.brown_min_perimeter,
            min_area=self.brown_min_area
        )
        self.brown_confidence = brown_conf
        self.brown_perimeter = brown_perim
        
        # ── 2. Check for BUMP (edge detection) if no brown border ──
        # Wood bump / threshold shows as strong horizontal edges in topdown view
        if not brown_detected:
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
            # Skip bump detection if brown border found
            self.edge_count = 0
            bump_detected = False
        
        # ── 3. Check for CHECKERED MAT (corner detection) as OPTIONAL FALLBACK ──
        # Only run if brown border and bump both failed (expensive, ~34ms)
        # Kept as fallback for other checkered patterns without brown borders
        checkered_detected = False
        if not brown_detected and not bump_detected:
            pattern_size = (self.checkerboard_cols, self.checkerboard_rows)
            found, corners = cv2.findChessboardCorners(gray, pattern_size, self._flags)
            
            if found and corners is not None:
                # Refine corner positions for better accuracy
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self._criteria)
                
                self.corner_count = len(corners_refined)
                
                # Confidence based on corner count vs expected
                expected_corners = self.checkerboard_rows * self.checkerboard_cols
                self.detection_confidence = min(1.0, self.corner_count / expected_corners)
                
                # Trigger if we found enough corners
                checkered_detected = self.corner_count >= self.min_corners
            else:
                self.corner_count = 0
        else:
            self.corner_count = 0
        
        # ── 4. Combine detections with temporal filtering ──
        # Priority: brown_border > bump > checkered
        detected_now = brown_detected or bump_detected or checkered_detected
        if brown_detected:
            reason_now = 'brown_border'
            self.detection_confidence = brown_conf
        elif bump_detected:
            reason_now = 'bump'
        elif checkered_detected:
            reason_now = 'checkered'
        else:
            reason_now = None
        
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
            dict with detection state, corner count, edge count, brown border info, reason, etc.
        """
        return {
            'detected': self.detected,
            'reason': self.detection_reason,
            'corner_count': self.corner_count,
            'edge_count': self.edge_count,
            'confidence': self.detection_confidence,
            'brown_confidence': self.brown_confidence,
            'brown_perimeter': self.brown_perimeter,
            'history': self._detection_history.copy(),
            'params': {
                'checkerboard_rows': self.checkerboard_rows,
                'checkerboard_cols': self.checkerboard_cols,
                'forward_fraction': self.forward_fraction,
                'min_corners': self.min_corners,
                'bump_edge_thresh': self.bump_edge_thresh,
                'bump_min_edges': self.bump_min_edges,
                'brown_hsv_lower': self.brown_hsv_lower,
                'brown_hsv_upper': self.brown_hsv_upper,
                'brown_min_perimeter': self.brown_min_perimeter,
                'brown_min_area': self.brown_min_area,
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
