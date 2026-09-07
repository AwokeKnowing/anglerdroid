"""checkered_mat_detector.py - Vision-based checkered door mat detector (ego-frame).

Detects checkered floor patterns (alternating black/white squares) in the robot's
near-field view and triggers a forward stop when the mat is detected under/near the robot.

This is an ego-frame safety reflex (no SLAM/map required) that prevents the robot
from driving onto designated keep-out zones marked with checkered mats.

Detection strategy:
    1. Extract ego-frame ROI (bottom portion of webcam / near-field under robot)
    2. Convert to grayscale and enhance contrast
    3. Detect corners (checkered patterns have high corner density at regular intervals)
    4. Score grid regularity and alternating intensity pattern
    5. Trigger when score exceeds threshold

Lightweight for Jetson: pure OpenCV operations (no deep learning).
"""

import numpy as np
import cv2


# Default tunable parameters
DEFAULT_ROI_BOTTOM_FRACTION = 0.6  # Use bottom 60% of frame (floor near robot)
DEFAULT_ROI_TOP_FRACTION = 0.3     # Start from 30% down (skip distant ceiling/walls)
DEFAULT_MIN_SCORE = 0.25           # Minimum score to trigger detection (0.0-1.0)
DEFAULT_MIN_CORNER_DENSITY = 0.08  # Minimum corner density (corners per 100 pixels)
DEFAULT_HYSTERESIS = 0.10          # Score delta for hysteresis (prevents flicker)


def detect_checkered_mat(rgb_frame, roi_top_frac=DEFAULT_ROI_TOP_FRACTION,
                         roi_bottom_frac=DEFAULT_ROI_BOTTOM_FRACTION,
                         min_score=DEFAULT_MIN_SCORE,
                         min_corner_density=DEFAULT_MIN_CORNER_DENSITY,
                         prev_detected=False, hysteresis=DEFAULT_HYSTERESIS,
                         debug=False):
    """Detect checkered pattern in ego-frame near-field view.

    Args:
        rgb_frame: RGB image (HxWx3 uint8) from webcam or topdown camera.
        roi_top_frac: Top boundary of ROI as fraction of frame height (0.0-1.0).
        roi_bottom_frac: Bottom boundary of ROI as fraction of frame height (0.0-1.0).
        min_score: Minimum score to trigger detection (0.0-1.0).
        min_corner_density: Minimum corner density threshold (corners per 100 pixels).
        prev_detected: Previous detection state (for hysteresis).
        hysteresis: Score adjustment for hysteresis (prevents flicker).
        debug: If True, return debug visualization image.

    Returns:
        If debug=False: (detected: bool, score: float, corner_count: int)
        If debug=True: (detected: bool, score: float, corner_count: int, debug_img: array)
    """
    if rgb_frame is None or rgb_frame.size == 0:
        return (False, 0.0, 0, None) if debug else (False, 0.0, 0)

    h, w = rgb_frame.shape[:2]
    if h < 20 or w < 20:
        return (False, 0.0, 0, None) if debug else (False, 0.0, 0)

    # Extract ROI (bottom portion of frame = floor near robot)
    y0 = int(h * roi_top_frac)
    y1 = int(h * roi_bottom_frac)
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    roi = rgb_frame[y0:y1, :]

    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi

    # Enhance contrast with CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Detect corners using goodFeaturesToTrack (fast, reliable for Jetson)
    corners = cv2.goodFeaturesToTrack(
        enhanced,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=3
    )
    corner_count = len(corners) if corners is not None else 0

    # Corner density score (checkered patterns have high corner density)
    roi_area_px = roi.shape[0] * roi.shape[1]
    corner_density = (corner_count / max(1, roi_area_px)) * 100.0
    density_score = min(1.0, corner_density / max(0.01, min_corner_density * 100.0))

    # Grid regularity score via edge detection + Hough lines
    edges = cv2.Canny(enhanced, 50, 150)
    # Lower threshold and minLineLength to catch smaller grids
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15,
                            minLineLength=10, maxLineGap=5)
    
    # Score based on number of detected lines (grid has many straight edges)
    line_count = len(lines) if lines is not None else 0
    # Normalize: expect ~10-30 lines for a good checkered pattern in ROI
    line_score = min(1.0, line_count / 20.0)

    # Detect perpendicular lines (checkered has both horizontal and vertical)
    h_lines = 0
    v_lines = 0
    if lines is not None and len(lines) > 0:
        for line in lines:
            try:
                if isinstance(line, (list, tuple, np.ndarray)) and len(line) > 0:
                    coords = line[0] if isinstance(line[0], (list, tuple, np.ndarray)) else line
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                        # Wider angle ranges to catch more lines
                        if angle < 25 or angle > 155:  # Horizontal-ish
                            h_lines += 1
                        elif 65 < angle < 115:  # Vertical-ish
                            v_lines += 1
            except (TypeError, IndexError, ValueError):
                continue
    
    # Checkered should have both horizontal AND vertical lines
    # Require at least 2 of each to avoid single-line false positives
    h_ok = h_lines >= 2
    v_ok = v_lines >= 2
    grid_orthogonality = 1.0 if (h_ok and v_ok) else 0.5 if (h_ok or v_ok) else 0.0

    # Local variance score (checkered has high local contrast)
    # Use Laplacian variance as proxy for texture complexity
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    lap_var = laplacian.var()
    # Normalize: typical checkered lap_var is 200-1000+
    # But noisy floors can also have high variance, so cap contribution
    variance_score = min(1.0, lap_var / 400.0)

    # Corner regularity: check if corners are somewhat evenly distributed
    # Random noise has corners everywhere; checkered has grid-like distribution
    corner_regularity = 0.0
    if corners is not None and len(corners) > 10:
        # Divide ROI into grid cells and check corner distribution
        h_roi, w_roi = roi.shape[:2]
        grid_h, grid_w = 4, 4
        cell_h, cell_w = h_roi // grid_h, w_roi // grid_w
        cell_counts = np.zeros((grid_h, grid_w))
        for corner in corners:
            x, y = corner.ravel()
            gi = min(int(y / cell_h), grid_h - 1)
            gj = min(int(x / cell_w), grid_w - 1)
            cell_counts[gi, gj] += 1
        # Checkered should have corners in multiple cells (not all in one spot)
        non_empty_cells = np.sum(cell_counts > 0)
        corner_regularity = min(1.0, non_empty_cells / 8.0)  # At least half the cells

    # Combined score (weighted average)
    # Grid orthogonality is strongly weighted but not a hard gate
    # This allows detection even with imperfect line detection while still filtering stripes
    score = (0.20 * density_score + 
             0.20 * line_score + 
             0.35 * grid_orthogonality +
             0.15 * variance_score +
             0.10 * corner_regularity)
    
    # Apply a penalty if orthogonality is low (prevents stripe false positives)
    if grid_orthogonality < 0.5:
        score *= 0.7  # 30% penalty for no proper grid

    # Hysteresis: raise threshold if previously detected (prevent flicker)
    threshold = min_score + (hysteresis if prev_detected else 0.0)
    detected = score >= threshold

    # Debug visualization
    debug_img = None
    if debug:
        debug_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        # Draw corners
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(debug_img, (int(x), int(y)), 2, (0, 255, 0), -1)
        # Draw lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        # Overlay score
        status = "DETECTED" if detected else "clear"
        color = (0, 0, 255) if detected else (0, 255, 0)
        cv2.putText(debug_img, f"{status} score={score:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(debug_img, f"corners={corner_count} density={corner_density:.1f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return (detected, score, corner_count, debug_img) if debug else (detected, score, corner_count)


class CheckeredMatDetector:
    """Stateful checkered mat detector with hysteresis and rolling detection."""

    def __init__(self, roi_top_frac=DEFAULT_ROI_TOP_FRACTION,
                 roi_bottom_frac=DEFAULT_ROI_BOTTOM_FRACTION,
                 min_score=DEFAULT_MIN_SCORE,
                 min_corner_density=DEFAULT_MIN_CORNER_DENSITY,
                 hysteresis=DEFAULT_HYSTERESIS):
        """Initialize detector with tunable parameters.

        Args:
            roi_top_frac: Top boundary of ROI as fraction of frame height (0.0-1.0).
            roi_bottom_frac: Bottom boundary of ROI as fraction of frame height (0.0-1.0).
            min_score: Minimum score to trigger detection (0.0-1.0).
            min_corner_density: Minimum corner density threshold (corners per 100 pixels).
            hysteresis: Score adjustment for hysteresis (prevents flicker).
        """
        self.roi_top_frac = roi_top_frac
        self.roi_bottom_frac = roi_bottom_frac
        self.min_score = min_score
        self.min_corner_density = min_corner_density
        self.hysteresis = hysteresis
        
        self._detected = False
        self._score = 0.0
        self._corner_count = 0
        self._frame_count = 0

    def update(self, rgb_frame, debug=False):
        """Update detector with new frame.

        Args:
            rgb_frame: RGB image (HxWx3 uint8).
            debug: If True, return debug visualization image.

        Returns:
            If debug=False: (detected: bool, score: float, corner_count: int)
            If debug=True: (detected: bool, score: float, corner_count: int, debug_img: array)
        """
        self._frame_count += 1
        
        result = detect_checkered_mat(
            rgb_frame,
            roi_top_frac=self.roi_top_frac,
            roi_bottom_frac=self.roi_bottom_frac,
            min_score=self.min_score,
            min_corner_density=self.min_corner_density,
            prev_detected=self._detected,
            hysteresis=self.hysteresis,
            debug=debug
        )
        
        if debug:
            detected, score, corner_count, debug_img = result
            self._detected = detected
            self._score = score
            self._corner_count = corner_count
            return detected, score, corner_count, debug_img
        else:
            detected, score, corner_count = result
            self._detected = detected
            self._score = score
            self._corner_count = corner_count
            return detected, score, corner_count

    @property
    def detected(self):
        """True if checkered mat currently detected."""
        return self._detected

    @property
    def score(self):
        """Current detection score (0.0-1.0)."""
        return self._score

    @property
    def corner_count(self):
        """Number of corners detected in last frame."""
        return self._corner_count
