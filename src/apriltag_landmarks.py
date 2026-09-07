"""apriltag_landmarks.py — AprilTag landmark relocalization stub.

Offline-first AprilTag detection for absolute pose recovery when SLAM tracking
is lost. Designed to run in ~3Hz vision extras loop (NOT on 30Hz capture path).

Architecture:
- Lightweight OpenCV apriltag detector (suitable for Jetson Orin)
- Named landmarks config (e.g. checkered_door) with tag family/id + world pose
- Camera/robot pose estimation relative to detected tags
- Extension point for SLAM relocalization (future integration)

Usage pattern:
    detector = AprilTagLandmarkDetector()
    result = detector.detect_and_localize(rgb_image, camera_intrinsics)
    if result:
        # Use result.tag_id, result.world_pose, result.confidence
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Optional

import cv2
import numpy as np


# ── Config: Named landmarks ──────────────────────────────────────────

@dataclasses.dataclass
class LandmarkConfig:
    """Configuration for one named AprilTag landmark."""
    
    name: str
    tag_family: str  # e.g. "tag36h11", "tag25h9"
    tag_id: int
    world_x: float  # metres, in map frame
    world_y: float  # metres, in map frame
    world_theta: float  # radians, heading in map frame
    tag_size_m: float  # physical tag size in metres (outer black square)
    enabled: bool = True
    
    def __post_init__(self):
        # Normalize theta to [-pi, pi]
        self.world_theta = math.atan2(
            math.sin(self.world_theta), 
            math.cos(self.world_theta)
        )


# Default landmark set (can be overridden via config file)
DEFAULT_LANDMARKS = [
    LandmarkConfig(
        name="checkered_door",
        tag_family="tag36h11",
        tag_id=0,
        world_x=0.0,  # Placeholder: tune after placement
        world_y=0.0,
        world_theta=0.0,
        tag_size_m=0.15,  # 15cm tag (typical household size)
        enabled=True,
    ),
]


# ── Detection result ─────────────────────────────────────────────────

@dataclasses.dataclass
class AprilTagDetection:
    """Single AprilTag detection result with pose estimate."""
    
    tag_id: int
    tag_family: str
    landmark_name: Optional[str]  # None if tag not in landmark registry
    
    # Robot pose estimate in world frame (if landmark is known)
    world_x: Optional[float]  # metres
    world_y: Optional[float]  # metres
    world_theta: Optional[float]  # radians
    
    # Tag pose in camera frame (always available)
    camera_tvec: np.ndarray  # (3,) translation vector [x, y, z] metres
    camera_rvec: np.ndarray  # (3,) rotation vector (Rodrigues)
    
    # Quality metrics
    confidence: float  # 0-1, based on corner detection quality + reprojection error
    decision_margin: float  # AprilTag decoder decision margin (higher = better)
    
    # Image-space corners (4x2 array, clockwise from top-left)
    corners: np.ndarray
    
    # Metadata
    timestamp: float  # monotonic time
    

# ── AprilTag detector ────────────────────────────────────────────────

class AprilTagLandmarkDetector:
    """Lightweight AprilTag detector for landmark relocalization.
    
    Runs offline-first: works without live cameras (tests use fixture images).
    Suitable for ~3Hz side loop (vision extras), NOT 30Hz capture path.
    """
    
    def __init__(
        self,
        landmarks: Optional[list[LandmarkConfig]] = None,
        tag_family: str = "tag36h11",
        debug: bool = False,
    ):
        """Initialize AprilTag detector.
        
        Args:
            landmarks: List of known landmarks. Defaults to DEFAULT_LANDMARKS.
            tag_family: Primary tag family to detect (tag36h11 recommended for Orin).
            debug: Enable debug logging.
        """
        self.landmarks = landmarks if landmarks is not None else DEFAULT_LANDMARKS
        self.tag_family = tag_family
        self.debug = debug
        
        # Build lookup: tag_id -> landmark
        self._landmark_by_id: dict[int, LandmarkConfig] = {}
        for lm in self.landmarks:
            if lm.enabled and lm.tag_family == self.tag_family:
                self._landmark_by_id[lm.tag_id] = lm
        
        # OpenCV ArUco detector for AprilTag (available since OpenCV 4.7+)
        # tag36h11 = DICT_APRILTAG_36h11
        self._detector = None
        self._detector_params = None
        self._init_detector()
        
        if self.debug:
            print(f"apriltag_landmarks: init family={tag_family} "
                  f"landmarks={len(self.landmarks)} enabled={len(self._landmark_by_id)}")
    
    def _init_detector(self):
        """Initialize OpenCV ArUco/AprilTag detector."""
        try:
            # Map AprilTag family to OpenCV ArUco dictionary
            family_map = {
                "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
                "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
                "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
            }
            
            if self.tag_family not in family_map:
                raise ValueError(f"Unsupported tag family: {self.tag_family}")
            
            aruco_dict = cv2.aruco.getPredefinedDictionary(family_map[self.tag_family])
            
            # Detection parameters (tuned for Jetson Orin + household lighting)
            params = cv2.aruco.DetectorParameters()
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 10
            params.minMarkerPerimeterRate = 0.03  # Accept smaller tags
            params.maxMarkerPerimeterRate = 4.0
            params.polygonalApproxAccuracyRate = 0.05
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            
            self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            self._detector_params = params
            
        except AttributeError:
            # Fallback for OpenCV < 4.7 (legacy API)
            family_map = {
                "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
                "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
                "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
            }
            
            if self.tag_family not in family_map:
                raise ValueError(f"Unsupported tag family: {self.tag_family}")
            
            aruco_dict = cv2.aruco.Dictionary_get(family_map[self.tag_family])
            
            params = cv2.aruco.DetectorParameters_create()
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 10
            params.minMarkerPerimeterRate = 0.03
            params.maxMarkerPerimeterRate = 4.0
            params.polygonalApproxAccuracyRate = 0.05
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            
            self._detector = (aruco_dict, params)  # Legacy tuple interface
            self._detector_params = params
    
    def detect_and_localize(
        self,
        image: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> list[AprilTagDetection]:
        """Detect AprilTags and estimate camera/robot pose.
        
        Args:
            image: RGB or grayscale image (HxW or HxWx3 uint8).
            camera_matrix: 3x3 camera intrinsics matrix.
            dist_coeffs: Distortion coefficients (optional, defaults to zero).
        
        Returns:
            List of AprilTagDetection results, sorted by confidence (best first).
        """
        if image is None or image.size == 0:
            return []
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect tags
        try:
            if isinstance(self._detector, tuple):
                # Legacy OpenCV API (< 4.7)
                aruco_dict, params = self._detector
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, aruco_dict, parameters=params
                )
            else:
                # Modern OpenCV API (>= 4.7)
                corners, ids, rejected = self._detector.detectMarkers(gray)
        except Exception as e:
            if self.debug:
                print(f"apriltag_landmarks: detect failed: {e}")
            return []
        
        if ids is None or len(ids) == 0:
            return []
        
        # Default distortion (zero if not provided)
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5, dtype=np.float64)
        
        results = []
        now = time.monotonic()
        
        for i, tag_id in enumerate(ids.flatten()):
            tag_corners = corners[i][0]  # (4, 2) array
            
            # Check if this tag is a registered landmark
            landmark = self._landmark_by_id.get(int(tag_id))
            
            if landmark is None:
                # Unknown tag: still record detection but no world pose
                detection = AprilTagDetection(
                    tag_id=int(tag_id),
                    tag_family=self.tag_family,
                    landmark_name=None,
                    world_x=None,
                    world_y=None,
                    world_theta=None,
                    camera_tvec=np.zeros(3),
                    camera_rvec=np.zeros(3),
                    confidence=0.5,  # Unknown tags get lower confidence
                    decision_margin=0.0,
                    corners=tag_corners,
                    timestamp=now,
                )
                results.append(detection)
                continue
            
            # Estimate pose via solvePnP
            object_points = self._tag_object_points(landmark.tag_size_m)
            
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    tag_corners,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except Exception as e:
                if self.debug:
                    print(f"apriltag_landmarks: solvePnP failed for tag {tag_id}: {e}")
                continue
            
            if not success:
                continue
            
            # Estimate confidence from reprojection error
            proj_corners, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            reproj_error = np.linalg.norm(
                tag_corners - proj_corners.reshape(-1, 2), axis=1
            ).mean()
            confidence = max(0.0, min(1.0, 1.0 - reproj_error / 10.0))
            
            # Compute robot pose in world frame
            # (Tag pose in camera) + (Camera in robot) + (Landmark in world)
            # For now, assume camera = robot (simplified; can refine with extrinsics)
            world_x, world_y, world_theta = self._camera_to_world_pose(
                tvec, rvec, landmark
            )
            
            detection = AprilTagDetection(
                tag_id=int(tag_id),
                tag_family=self.tag_family,
                landmark_name=landmark.name,
                world_x=world_x,
                world_y=world_y,
                world_theta=world_theta,
                camera_tvec=tvec.flatten(),
                camera_rvec=rvec.flatten(),
                confidence=confidence,
                decision_margin=0.0,  # ArUco doesn't expose this; use reproj error
                corners=tag_corners,
                timestamp=now,
            )
            results.append(detection)
            
            if self.debug:
                print(f"apriltag_landmarks: detected {landmark.name} (id={tag_id}) "
                      f"pose=({world_x:.2f}, {world_y:.2f}, {math.degrees(world_theta):.1f}°) "
                      f"conf={confidence:.2f}")
        
        # Sort by confidence (best first)
        results.sort(key=lambda d: d.confidence, reverse=True)
        return results
    
    @staticmethod
    def _tag_object_points(tag_size_m: float) -> np.ndarray:
        """3D coordinates of tag corners in tag frame (tag center = origin).
        
        Returns (4, 3) array: corners clockwise from top-left.
        Z=0 (tag is flat), X=right, Y=down in tag frame.
        """
        s = tag_size_m / 2.0
        return np.array([
            [-s,  s, 0],  # top-left
            [ s,  s, 0],  # top-right
            [ s, -s, 0],  # bottom-right
            [-s, -s, 0],  # bottom-left
        ], dtype=np.float32)
    
    def _camera_to_world_pose(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray,
        landmark: LandmarkConfig,
    ) -> tuple[float, float, float]:
        """Compute robot world pose from tag detection.
        
        Args:
            tvec: (3,) tag translation in camera frame (metres).
            rvec: (3,) tag rotation in camera frame (Rodrigues).
            landmark: Known landmark config with world pose.
        
        Returns:
            (world_x, world_y, world_theta) — robot pose in map frame.
        
        Simplified model: assumes camera = robot center (no extrinsics).
        Future refinement: add RS1/RS2 camera extrinsics for accurate reloc.
        """
        # Tag pose in camera frame
        R_tag_cam, _ = cv2.Rodrigues(rvec)
        t_tag_cam = tvec.flatten()
        
        # Invert: camera pose in tag frame
        R_cam_tag = R_tag_cam.T
        t_cam_tag = -R_cam_tag @ t_tag_cam
        
        # Tag pose in world frame (from landmark config)
        R_world_tag = self._rotation_matrix_2d(landmark.world_theta)
        t_world_tag = np.array([landmark.world_x, landmark.world_y, 0.0])
        
        # Camera (robot) pose in world frame
        # T_world_cam = T_world_tag @ T_tag_cam
        R_world_cam = R_world_tag @ R_cam_tag[:3, :3]  # Only 2D rotation matters
        t_world_cam = R_world_tag @ t_cam_tag + t_world_tag
        
        # Extract 2D pose (X, Y, Theta)
        world_x = float(t_world_cam[0])
        world_y = float(t_world_cam[1])
        
        # Heading from rotation matrix (yaw = atan2(R21, R11))
        world_theta = math.atan2(R_world_cam[1, 0], R_world_cam[0, 0])
        
        return world_x, world_y, world_theta
    
    @staticmethod
    def _rotation_matrix_2d(theta: float) -> np.ndarray:
        """2D rotation matrix (3x3, Z-axis rotation) from heading angle."""
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ], dtype=np.float32)
    
    def get_landmark_by_name(self, name: str) -> Optional[LandmarkConfig]:
        """Lookup landmark by name."""
        for lm in self.landmarks:
            if lm.name == name:
                return lm
        return None


# ── API for vision extras integration ────────────────────────────────

def detect_apriltag_landmarks(
    rgb_image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    landmarks: Optional[list[LandmarkConfig]] = None,
    debug: bool = False,
) -> list[AprilTagDetection]:
    """Convenience function for one-shot AprilTag landmark detection.
    
    Use this in vision extras loop. Creates detector instance per call
    (overhead is ~1ms on Orin, acceptable for 3Hz loop).
    
    Args:
        rgb_image: RGB or grayscale image (HxW or HxWx3 uint8).
        camera_matrix: 3x3 camera intrinsics.
        dist_coeffs: Distortion coefficients (optional).
        landmarks: Known landmarks (defaults to DEFAULT_LANDMARKS).
        debug: Enable debug logging.
    
    Returns:
        List of AprilTagDetection, sorted by confidence.
    """
    detector = AprilTagLandmarkDetector(landmarks=landmarks, debug=debug)
    return detector.detect_and_localize(rgb_image, camera_matrix, dist_coeffs)
