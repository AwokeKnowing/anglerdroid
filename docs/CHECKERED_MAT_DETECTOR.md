# Checkered Mat Detector (Ego-Frame Vision-Based Safety Reflex)

## Overview

The checkered mat detector is a vision-based safety reflex that prevents the robot from driving onto checkered door mats. It uses ego-frame computer vision (no SLAM required) to detect checkered patterns in the robot's near-field view and triggers an immediate forward stop when a mat is detected under or near the robot.

This is particularly useful for:
- Marking physical keep-out zones with checkered mats (visual barriers)
- Preventing entry to areas where SLAM-based map keepouts are unavailable
- Providing immediate ego-frame safety without relying on global map state

## Design Principles

1. **Ego-Frame Only**: Works entirely in camera space (no SLAM, no map, no localization)
2. **Vision-Based**: Uses RGB webcam to detect checkered patterns via classical CV
3. **Forward Stop**: Immediately zeros forward motion when detected
4. **Escape Allowed**: Backward motion still allowed if rear is clear (same as near-field reflex)
5. **Lightweight**: Pure OpenCV operations (no deep learning) suitable for Jetson real-time loop
6. **Tunable**: All thresholds and ROI parameters are configurable

## Detection Strategy

### Pattern Recognition

The detector uses a multi-component scoring system to identify checkered patterns:

1. **Corner Density**: Checkered patterns have high corner density at regular intervals
   - Uses `cv2.goodFeaturesToTrack()` to detect corners
   - Normalizes by ROI area to get corners-per-pixel metric

2. **Grid Regularity**: Detects straight edges and lines (checkered has grid structure)
   - Canny edge detection + Hough line transform
   - Scores based on number of detected lines

3. **Local Contrast**: Checkered patterns have high local variance (alternating squares)
   - Laplacian variance as proxy for texture complexity
   - High variance indicates alternating dark/light regions

### Combined Score

The final score is a weighted average:
```
score = 0.5 × corner_density + 0.3 × line_score + 0.2 × variance_score
```

Detection triggers when `score >= min_score` (default 0.25).

### Hysteresis

To prevent flicker (rapid on/off transitions), the detector uses hysteresis:
- **Trigger threshold**: `min_score` (e.g., 0.25)
- **Clear threshold**: `min_score + hysteresis` (e.g., 0.35)
- Once triggered, detection stays active until score drops below clear threshold

## Implementation

### Module Structure (`src/checkered_mat_detector.py`)

```python
# Stateless detection function
detected, score, corners = detect_checkered_mat(
    rgb_frame,
    roi_top_frac=0.3,      # ROI starts at 30% down frame
    roi_bottom_frac=0.6,   # ROI ends at 60% down frame
    min_score=0.25,
    min_corner_density=0.15,
    hysteresis=0.10
)

# Stateful detector class (maintains history for hysteresis)
detector = CheckeredMatDetector(
    roi_top_frac=0.3,
    roi_bottom_frac=0.6,
    min_score=0.25,
    min_corner_density=0.15,
    hysteresis=0.10
)
detected, score, corners = detector.update(rgb_frame)
```

### Integration (`src/vision.py`)

The detector is integrated into the vision capture loop:

```python
# Initialize detector in Vision.__init__()
self._checkered_detector = CheckeredMatDetector()

# In _capture_loop, after processing depth:
if self._webcam and self._webcam.ok:
    detected, score, corners = self._checkered_detector.update(self._webcam.color)
    self._checkered_mat_detected = detected
    self._checkered_mat_score = score
    self._checkered_mat_corners = corners

# Pass to SafetyGuard
self._safety.update(
    self._persistent_obs, fused_yaw, fused_fwd,
    height_cm=self._persistent_height,
    topdown_near_field=self._topdown_near_field,
    checkered_mat_detected=self._checkered_mat_detected
)
```

### Safety Response (`src/safety.py`)

When `checkered_mat_detected=True`:
- `_fwd_scale = 0.0` (immediate forward stop)
- `_bwd_scale` computed normally from rear obstacles
- `_ang_scale` computed normally from lateral obstacles
- `_near_field_reason = "checkered_mat_detected"` (for logging)

## Tunable Parameters

### ROI (Region of Interest)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `roi_top_frac` | 0.3 | Top boundary of ROI as fraction of frame height (0.0-1.0) |
| `roi_bottom_frac` | 0.6 | Bottom boundary of ROI as fraction of frame height (0.0-1.0) |

**Default ROI**: 30%-60% of frame = bottom-middle portion (floor near robot)

**Tuning guidance**:
- **Smaller ROI** (e.g., 0.4-0.6): Only detect mats very close to robot (more conservative)
- **Larger ROI** (e.g., 0.2-0.7): Detect mats farther ahead (earlier warning)
- **Lower ROI** (e.g., 0.5-1.0): Focus on floor directly under robot

### Detection Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_score` | 0.25 | Minimum combined score to trigger detection (0.0-1.0) |
| `min_corner_density` | 0.15 | Minimum corner density threshold (corners per 100 pixels) |
| `hysteresis` | 0.10 | Score delta for hysteresis (prevents flicker) |

**Tuning guidance**:
- **Lower `min_score`** (e.g., 0.20): More sensitive, may trigger on textured floors
- **Higher `min_score`** (e.g., 0.35): Less sensitive, only strong checkered patterns
- **Higher `min_corner_density`**: Require denser corner grid (stricter)
- **Lower `min_corner_density`**: Accept sparser patterns (more lenient)
- **Higher `hysteresis`**: More stable (harder to flicker), slower to clear
- **Lower `hysteresis`**: More responsive, may flicker on borderline patterns

### Recommended Starting Points

For typical checkered door mats:
- **Square size**: 15-40 pixels (at typical floor distance)
- **min_score**: 0.25 (balanced sensitivity)
- **ROI**: 0.3-0.6 (bottom-middle of frame)

For fine-tuning:
1. Capture sample frames with/without mats
2. Run detector with `debug=True` to visualize corners/lines/score
3. Adjust `min_score` based on observed score distributions
4. Adjust ROI based on camera mounting and floor visible area

## Behavior

### Normal Operation (No Checkered Mat)

- All motion allowed (fwd=1.0, bwd=1.0, ang=1.0)
- Standard floor-obstacle avoidance active
- Robot navigates normally

### Checkered Mat Detected (Ego-Frame Reflex)

1. **Forward motion**: Immediately stopped (fwd_scale=0.0)
2. **Backward motion**: Allowed if rear is clear
3. **Angular motion**: Allowed (can rotate in place)
4. **Logging**: "CHECKERED MAT DETECTED — score=X.XX corners=N"

### Escape Behavior

If the robot triggers the reflex:
- **Rear clear**: Can reverse away from the mat
- **Rear blocked**: Cannot reverse, but can rotate
- **Planning**: Higher-level planner should command reverse when reflex fires

## Testing

Comprehensive unit tests in `test_checkered_mat_detector.py`:

1. ✅ Synthetic checkered pattern → detection triggers
2. ✅ Plain floor (uniform color) → no detection
3. ✅ Noisy/textured floor → no detection
4. ✅ Horizontal stripes → no detection (not checkered)
5. ✅ Vertical stripes → no detection (not checkered)
6. ✅ Small checkered patch (10px squares) → detection triggers
7. ✅ Large checkered pattern (40px squares) → detection triggers
8. ✅ Mixed checkered + plain → detection triggers
9. ✅ Hysteresis prevents flicker
10. ✅ SafetyGuard stops forward when checkered detected
11. ✅ SafetyGuard allows motion when no checkered
12. ✅ Empty/invalid frames handled gracefully
13. ✅ ROI configuration affects detection

Run tests: `python3 test_checkered_mat_detector.py`

## Relationship to Existing Safety

### Near-Field Reflex (Top-Down Camera)

**Top-Down Near-Field**: Detects overhead obstacles (table undersides, hands) <30cm from RS1 camera.

**Checkered Mat**: Detects checkered patterns in ego-frame near-field view (floor under robot).

Both mechanisms:
- Trigger immediate forward stop (fwd_scale=0.0)
- Allow backward/angular motion if clear
- Run BEFORE floor-obstacle logic (high priority reflexes)
- Log clear diagnostic messages

### Paint Keepouts (Map-Frame)

**Paint Keepouts**: User paints no-go zones on global map (SLAM-based, persistent).

**Checkered Mat**: Ego-frame vision-based detection (no SLAM, no map).

Complementary:
- Paint keepouts: Map-frame planning-level avoidance (requires SLAM)
- Checkered mat: Ego-frame reflex-level avoidance (no SLAM required)

**Note**: If paint keepouts were soft-only (score 90 instead of hard keepout), they should be hardened separately. The checkered mat detector is the primary ego-frame visual solution.

## Logging and Diagnostics

### Vision Properties

```python
vision.checkered_mat_detected    # bool: reflex triggered
vision.checkered_mat_score       # float: detection score (0.0-1.0)
vision.checkered_mat_corners     # int: number of corners detected
```

### SafetyGuard Properties

```python
safety.checkered_mat_detected    # bool: reflex active
safety.near_field_reason         # str: "checkered_mat_detected" or None
safety.fwd_scale                 # float: 0.0 when reflex active
```

### Log Messages

**Detection triggered** (every 30 frames while active):
```
vision: CHECKERED MAT DETECTED — score=0.45 corners=120 (vision-based ego-frame reflex)
safety: CHECKERED MAT REFLEX — vision detected checkered mat under/near robot, fwd=0.0, computing bwd/ang normally
```

**Detection cleared**:
```
vision: CHECKERED MAT CLEARED — score=0.15 corners=25 (after 18 frames)
```

## Limitations

### Known Limitations

1. **Lighting Dependent**: Performance degrades in very low light or high glare
2. **Camera Angle**: Assumes webcam has downward/forward view of floor
3. **Square Size**: Very small squares (<5px) may not be reliably detected
4. **Non-Standard Patterns**: Only detects classic black/white checkerboard (not diagonal, rotated, or colored variants)

### Mitigation Strategies

- **Lighting**: CLAHE (adaptive histogram equalization) improves contrast in varied lighting
- **Camera Angle**: ROI parameters tunable to match actual camera mounting
- **Square Size**: Tunable thresholds accommodate different mat designs
- **Pattern Variants**: Detection is based on corner density + grid regularity, not color

## Future Enhancements

### Potential Improvements

1. **Graduated Response**: Scale forward speed by score (0.20→0.5x, 0.25→0.25x, 0.30→0.0x)
2. **Temporal Filtering**: Require N consecutive frames to reduce false positives
3. **Multi-Camera**: Use topdown camera as secondary input (redundant detection)
4. **Pattern Library**: Support more keep-out patterns (stripes, symbols, colors)
5. **Integration with Planner**: HouseBot automatically commands reverse when reflex fires

### Live Deployment

The implementation is complete and tested offline. To deploy:

1. ✅ Code implemented in `src/checkered_mat_detector.py`, `src/vision.py`, `src/safety.py`
2. ✅ Unit tests passing
3. ⏳ Integration testing in sim (if available)
4. ⏳ Live testing with drive DISARMED (place checkered mat, verify detection)
5. ⏳ Live testing with drive ARMED in controlled environment
6. ⏳ Full deployment

**Current Status**: Ready for integration and testing. Drive stays DISARMED until validated.

## Quick Start

### Basic Usage (Default Parameters)

```python
from checkered_mat_detector import CheckeredMatDetector

detector = CheckeredMatDetector()

# In your vision loop:
detected, score, corners = detector.update(rgb_frame)
if detected:
    print(f"Checkered mat detected! Score: {score:.2f}")
    # Trigger forward stop
```

### Custom Parameters (Fine-Tuned)

```python
detector = CheckeredMatDetector(
    roi_top_frac=0.4,        # Start ROI lower (closer to robot)
    roi_bottom_frac=0.7,     # Larger ROI (more of floor visible)
    min_score=0.30,          # Higher threshold (less sensitive)
    min_corner_density=0.20, # Require denser corners
    hysteresis=0.15          # More stable (less flicker)
)
```

### Debug Visualization

```python
detected, score, corners, debug_img = detector.update(rgb_frame, debug=True)
cv2.imshow("Checkered Mat Debug", debug_img)
```

The debug image shows:
- Green dots: Detected corners
- Blue lines: Detected straight edges
- Text overlay: Detection status, score, corner count

## References

- **Use Case**: Vision-based ego-frame keep-out detection (no SLAM required)
- **Files Modified**:
  - `src/checkered_mat_detector.py`: Detector implementation (new)
  - `src/vision.py`: Integration into capture loop
  - `src/safety.py`: Added checkered_mat_detected parameter
  - `test_checkered_mat_detector.py`: Comprehensive unit tests (new)
  - `docs/CHECKERED_MAT_DETECTOR.md`: This documentation (new)
- **Related Issues**: Visual keep-out markers, SLAM-free safety, ego-frame obstacle avoidance
