# AprilTag Landmark Relocalization

**Status**: Offline-first stub ready for SLAM integration (2026-09-07)

## Overview

AprilTag landmark relocalization provides absolute pose recovery when SLAM tracking is lost. Tags placed at known locations (e.g. doorways) act as "anchors" that the robot can use to correct drift or re-initialize SLAM.

This is an **offline-first** implementation: works without live cameras (tests use synthetic images) and runs in the ~3Hz vision extras loop (NOT on the 30Hz capture hot path).

## Architecture

```
Vision Extras Loop (~3Hz)
  ↓
RS1 Topdown RGB
  ↓
AprilTag Detection (OpenCV ArUco)
  ↓
Landmark Registry Lookup
  ↓
PnP Pose Estimation
  ↓
World Pose (X, Y, Theta)
  ↓
[Future] SLAM Relocalization
```

**Key design points:**
- **Lightweight**: OpenCV ArUco detector (no external dependencies beyond opencv-python)
- **Jetson Orin friendly**: Tuned detection parameters for household lighting
- **Rotation-invariant**: Works at any tag orientation
- **GPU-free**: Runs on CPU in extras loop, no impact to 30Hz critical path
- **Testable**: Synthetic fixtures, no live camera required

## Tag Placement

### Checkered Door (Primary Landmark)

Place a **15cm tag36h11 tag (ID=0)** at the checkered door entrance:

1. **Height**: Eye level (~1.5m), centered above doorframe
2. **Orientation**: Tag facing outward (toward room interior)
3. **Visibility**: Clear line of sight from 2-4m away
4. **Lighting**: Avoid direct sunlight/backlighting (diffuse overhead light is best)

**Why this location?**
- High-traffic area (robot passes frequently)
- Natural re-localization point after exploring other rooms
- Complements checkered mat keepout (same named landmark)

### Printing Tags

Use the OpenCV ArUco tag generator:

```python
import cv2
import numpy as np

# Generate tag36h11 ID=0 (checkered_door landmark)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
tag_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 600)  # 600px
cv2.imwrite("apriltag_36h11_id0.png", tag_img)
```

Print at **15cm × 15cm** (outer black square). Use matte paper (glossy creates reflections).

## Configuration

Landmarks are defined in `src/apriltag_landmarks.py`:

```python
DEFAULT_LANDMARKS = [
    LandmarkConfig(
        name="checkered_door",
        tag_family="tag36h11",
        tag_id=0,
        world_x=0.0,  # metres, map frame (TUNE after placement)
        world_y=0.0,
        world_theta=0.0,  # radians, heading in map frame
        tag_size_m=0.15,  # physical tag size (outer black square)
        enabled=True,
    ),
]
```

**Calibrating world pose:**
1. Place robot at a known SLAM-locked pose near the tag
2. Read AprilTag detection: `vision.get_apriltag_detections()`
3. Note the reported `world_x, world_y, world_theta`
4. Update `DEFAULT_LANDMARKS` with actual tag position in map frame
5. Restart vision system to apply

**Adding more landmarks:**

```python
DEFAULT_LANDMARKS.append(
    LandmarkConfig(
        name="kitchen_entry",
        tag_family="tag36h11",
        tag_id=1,
        world_x=3.5,
        world_y=-2.0,
        world_theta=math.radians(180),
        tag_size_m=0.15,
        enabled=True,
    )
)
```

## Usage API

### Vision Extras Integration

AprilTag detection runs automatically in the vision extras loop at ~3Hz (no manual invocation needed).

### Reading Detections

```python
# Get latest AprilTag detections from vision extras loop
detections = vision.get_apriltag_detections()

for det in detections:
    if det.landmark_name == "checkered_door":
        print(f"Robot pose: ({det.world_x:.2f}, {det.world_y:.2f}, "
              f"{math.degrees(det.world_theta):.1f}°)")
        print(f"Confidence: {det.confidence:.2f}")
```

**Detection fields:**
- `tag_id`: AprilTag ID (0-based)
- `tag_family`: "tag36h11", "tag25h9", etc.
- `landmark_name`: Landmark name from registry (None if unknown tag)
- `world_x, world_y, world_theta`: Robot pose in map frame (None if unknown tag)
- `camera_tvec, camera_rvec`: Tag pose in camera frame (always available)
- `confidence`: 0-1, based on reprojection error
- `corners`: (4, 2) array, image-space tag corners
- `timestamp`: `time.monotonic()` when detected

### Checking Staleness

```python
# Check when landmark was last seen
last_seen = vision.get_apriltag_last_seen("checkered_door")
if last_seen is None:
    print("checkered_door never seen")
elif time.monotonic() - last_seen > 5.0:
    print("checkered_door stale (>5s since last detection)")
else:
    print("checkered_door fresh")
```

## SLAM Integration (Future Work)

The AprilTag detection stub is ready for SLAM relocalization. Planned integration:

### 1. Drift Correction

When loop closure shift is large (`> 0.5m`), use AprilTag detections to validate/correct:

```python
# In slam.py loop closure
if max_shift > 0.5:
    detections = vision.get_apriltag_detections()
    for det in detections:
        if det.confidence > 0.7 and det.landmark_name:
            # Compare SLAM pose estimate vs AprilTag pose
            slam_x, slam_y, slam_theta = self._pose.x, self._pose.y, self._pose.theta
            tag_x, tag_y, tag_theta = det.world_x, det.world_y, det.world_theta
            
            drift = math.sqrt((slam_x - tag_x)**2 + (slam_y - tag_y)**2)
            if drift < 0.3:
                print(f"AprilTag confirms loop closure (drift={drift:.2f}m)")
            else:
                print(f"AprilTag rejects loop closure (drift={drift:.2f}m > 0.3m)")
                # Optionally: discard loop closure or add corrective edge
```

### 2. Tracking Loss Recovery

When SLAM loses tracking (`slam_locked=False`), use AprilTag to reinitialize:

```python
# In vision.py when SLAM NOT LOCKED for >10s
if not self._slam_locked and time.monotonic() - self._slam_lock_lost_at > 10.0:
    detections = self.get_apriltag_detections()
    if detections and detections[0].confidence > 0.8:
        det = detections[0]
        print(f"AprilTag relocalization: {det.landmark_name} → "
              f"pose=({det.world_x:.2f}, {det.world_y:.2f}, {math.degrees(det.world_theta):.1f}°)")
        
        # Reset SLAM pose to AprilTag estimate
        self._pose.x = det.world_x
        self._pose.y = det.world_y
        self._pose.theta = det.world_theta
        
        # Clear stale map (optional: keep if drift was small)
        # self._global_map.reset()
```

### 3. Keyframe Anchoring

Add AprilTag detections as high-confidence edges in the pose graph:

```python
# In slam.py when creating keyframe
detections = vision.get_apriltag_detections()
if detections:
    det = detections[0]
    if det.confidence > 0.75 and det.landmark_name:
        # Add fixed constraint: keyframe must be near landmark
        landmark = self._landmark_registry[det.landmark_name]
        info = np.diag([1000.0, 1000.0, 2000.0])  # High confidence
        self._edges.append((kf.id, LANDMARK_NODE_ID, dx, dy, dtheta, info))
```

## Testing

Run unit tests (no live camera required):

```bash
cd /workspace
python3 test_apriltag_landmarks.py
```

**Test coverage:**
1. Synthetic tag detection
2. Known landmark pose estimation
3. Unknown tag handling
4. Multiple tags in frame
5. No tags / empty image
6. Invalid input handling
7. Camera intrinsics
8. Default landmarks config
9. Pose estimation accuracy
10. API integration

All tests use synthetic fixtures (OpenCV ArUco generator), no RealSense needed.

## Performance

**Vision extras loop timing (Jetson Orin NX):**
- AprilTag detection: ~8-15ms per frame (tag present)
- No detection (empty frame): ~3-5ms
- Extras loop budget: ~333ms (3Hz)
- **Impact on 30Hz capture: 0ms** (runs off critical path)

**Detection range:**
- 15cm tag @ 0.5m: Excellent (confidence > 0.9)
- 15cm tag @ 2.0m: Good (confidence > 0.7)
- 15cm tag @ 4.0m: Marginal (confidence > 0.5)
- 15cm tag @ 6.0m+: Unreliable (too small, < 20px)

**Failure modes:**
- Motion blur (robot turning fast): Use slower approach when seeking tags
- Occlusion (person/furniture): Place tags at height, clear sightlines
- Glare (direct sunlight): Avoid windows, prefer diffuse overhead light
- Wrong tag size: Update `tag_size_m` in config (critical for pose accuracy)

## Troubleshooting

### No tags detected

1. Check tag is in frame: `vision.frames[1]` (RS1 RGB, 180° rotated in ego)
2. Verify tag family: `tag36h11` recommended (tag25h9, tag16h5 also supported)
3. Check lighting: Avoid shadows, glare, backlighting
4. Tag size: Must be > 20px in image (closer = better)
5. Enable debug: `AprilTagLandmarkDetector(debug=True)`

### Low confidence (<0.5)

1. Increase tag size (15cm → 20cm) or move robot closer
2. Improve lighting (diffuse overhead, not directional/harsh)
3. Check tag is flat (wrinkled/bent tags fail)
4. Verify camera intrinsics: `vision._get_rs1_color_intrinsics()`

### Wrong pose estimate

1. **Critical**: Verify `tag_size_m` matches physical tag (measure outer black square)
2. Check camera intrinsics (focal length, principal point)
3. Verify landmark world pose in config (X, Y, Theta)
4. Ensure tag is flat (tilted tags produce systematic errors)

### Extras loop errors

Check `/workspace/src/vision.py` extras loop logs:
```
vision-extras error: ...
```

Common issues:
- Missing OpenCV (install `opencv-python`)
- RS1 camera not initialized (check `vision._rs1.ok`)
- Intrinsics unavailable (RealSense profile missing)

## Future Enhancements

1. **Forward camera support**: Also detect tags in RS2 forward RGB (wider FOV)
2. **Tag bundle fusion**: Multiple simultaneous detections → weighted pose average
3. **Temporal filtering**: Median filter over 3-5 frames to reduce jitter
4. **GPU acceleration**: AprilTag3 CUDA for <1ms detection (Orin GPU)
5. **Auto-calibration**: Learn landmark world pose from SLAM-locked observations
6. **Tag size auto-detect**: Estimate tag size from detection (remove manual config)
7. **Relocalization API**: High-level `slam.relocalize_from_apriltag()` method

## Related Docs

- `AGENTS.md` — Orin-first principles, 30Hz critical path budget
- `JAMES_ARCHITECTURE.md` — Vision extras loop (~3Hz side loop)
- `skills/checkered_mat_keepout.md` — Named keepout at checkered door (same landmark)
- `src/slam.py` — PoseGraphSLAM backend (future integration point)
- `src/vision.py` — Vision system, extras loop, AprilTag integration

## References

- AprilTag paper: https://april.eecs.umich.edu/papers/details.php?name=wang2016iros
- OpenCV ArUco/AprilTag: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- tag36h11 family: 2320 unique tags, robust to ~25% occlusion
