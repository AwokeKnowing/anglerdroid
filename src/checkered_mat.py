"""checkered_mat.py – Ego-frame checkered floor-mat hard-stop detection.

Detects a checkered mat in the bottom region of an RGB camera view (sensor frame)
and triggers a forward hard-stop (fwd_scale=0) to prevent driving onto the mat.
This is a REFLEX that works WITHOUT SLAM or map-based keepouts.

Design:
    - Uses OpenCV checkerboard corner detection on RGB frames
    - PRIMARY SOURCE: Webcam RGB (Vision frames[0])
    - Analyzes only the BOTTOM region of the image (where floor/mat is visible)
    - Detection → fwd_scale=0, allows reverse/turn if rear is clear
    - Tunable thresholds for checkerboard size and detection sensitivity

Typical use:
    detector = CheckeredMatDetector()
    triggered = detector.check(webcam_rgb_frame)  # frames[0]
    if triggered:
        fwd_scale = 0.0  # Stop forward motion
"""

import cv2
import numpy as np


# ── Default parameters ──
DEFAULT_CHECKERBOARD_ROWS = 6      # Internal corners (7x7 squares → 6x6 corners)
DEFAULT_CHECKERBOARD_COLS = 6
DEFAULT_BOTTOM_FRACTION = 0.5      # Analyze bottom 50% of image
DEFAULT_MIN_CORNERS = 4            # Minimum corners to confirm detection
DEFAULT_CORNER_QUALITY = 0.1       # cornerSubPix quality threshold


class CheckeredMatDetector:
    """Detects checkered floor mat in RGB camera images (sensor/ego frame).
    
    This detector looks for checkerboard patterns in the bottom portion of
    RGB camera frames and triggers a forward hard-stop when detected.
    
    Attributes:
        checkerboard_rows: Number of internal corner rows to detect
        checkerboard_cols: Number of internal corner columns to detect
        bottom_fraction: Fraction of image height to analyze (from bottom)
        min_corners: Minimum corners required to trigger detection
        corner_quality: Quality threshold for corner refinement
        
        detected: True if mat detected in most recent check()
        corner_count: Number of corners found in most recent check()
        detection_confidence: Detection confidence [0.0, 1.0]
    """
    
    def __init__(self,
                 checkerboard_rows=DEFAULT_CHECKERBOARD_ROWS,
                 checkerboard_cols=DEFAULT_CHECKERBOARD_COLS,
                 bottom_fraction=DEFAULT_BOTTOM_FRACTION,
                 min_corners=DEFAULT_MIN_CORNERS,
                 corner_quality=DEFAULT_CORNER_QUALITY):
        """Initialize checkered mat detector.
        
        Args:
            checkerboard_rows: Internal corner rows (N+1 squares → N corners)
            checkerboard_cols: Internal corner columns
            bottom_fraction: Fraction of image to analyze from bottom [0.0, 1.0]
            min_corners: Minimum corners to trigger detection
            corner_quality: Corner refinement quality (OpenCV winSize factor)
        """
        self.checkerboard_rows = checkerboard_rows
        self.checkerboard_cols = checkerboard_cols
        self.bottom_fraction = max(0.1, min(1.0, bottom_fraction))
        self.min_corners = max(1, min_corners)
        self.corner_quality = corner_quality
        
        # Detection state
        self.detected = False
        self.corner_count = 0
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
        """Check if checkered mat is visible in RGB frame.
        
        Args:
            rgb_frame: HxWx3 uint8 RGB image (PRIMARY: webcam Vision frames[0])
        
        Returns:
            triggered: bool, True if mat detected (forward hard-stop)
        """
        if rgb_frame is None or rgb_frame.size == 0:
            self.detected = False
            self.corner_count = 0
            self.detection_confidence = 0.0
            return False
        
        # Extract bottom region of image
        h, w = rgb_frame.shape[:2]
        bottom_h = int(h * self.bottom_fraction)
        if bottom_h < 20:  # Minimum usable height
            self.detected = False
            return False
        
        # Crop to bottom region (where floor/mat is visible)
        bottom_region = rgb_frame[-bottom_h:, :, :]
        
        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)
        
        # Attempt to find checkerboard corners
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
            detected_now = self.corner_count >= self.min_corners
        else:
            self.corner_count = 0
            self.detection_confidence = 0.0
            detected_now = False
        
        # Temporal filtering: require consistent detection to reduce flicker
        self._detection_history.append(detected_now)
        if len(self._detection_history) > self._history_size:
            self._detection_history.pop(0)
        
        # Trigger if detected in majority of recent frames
        detection_votes = sum(self._detection_history)
        self.detected = detection_votes >= (len(self._detection_history) / 2)
        
        return self.detected
    
    def reset(self):
        """Reset detection state and history."""
        self.detected = False
        self.corner_count = 0
        self.detection_confidence = 0.0
        self._detection_history.clear()
    
    def get_debug_info(self):
        """Get debug information about current detection state.
        
        Returns:
            dict with detection state, corner count, confidence, etc.
        """
        return {
            'detected': self.detected,
            'corner_count': self.corner_count,
            'confidence': self.detection_confidence,
            'history': self._detection_history.copy(),
            'params': {
                'checkerboard_rows': self.checkerboard_rows,
                'checkerboard_cols': self.checkerboard_cols,
                'bottom_fraction': self.bottom_fraction,
                'min_corners': self.min_corners,
            }
        }


def check_checkered_mat(rgb_frame,
                       checkerboard_rows=DEFAULT_CHECKERBOARD_ROWS,
                       checkerboard_cols=DEFAULT_CHECKERBOARD_COLS,
                       bottom_fraction=DEFAULT_BOTTOM_FRACTION,
                       min_corners=DEFAULT_MIN_CORNERS):
    """Convenience function for one-shot checkered mat detection.
    
    For persistent detection with temporal filtering, use CheckeredMatDetector.
    
    Args:
        rgb_frame: HxWx3 uint8 RGB image
        checkerboard_rows: Internal corner rows
        checkerboard_cols: Internal corner columns
        bottom_fraction: Fraction of image to analyze from bottom
        min_corners: Minimum corners to trigger detection
    
    Returns:
        (triggered, corner_count, confidence)
    """
    detector = CheckeredMatDetector(
        checkerboard_rows=checkerboard_rows,
        checkerboard_cols=checkerboard_cols,
        bottom_fraction=bottom_fraction,
        min_corners=min_corners
    )
    
    triggered = detector.check(rgb_frame)
    return triggered, detector.corner_count, detector.detection_confidence
