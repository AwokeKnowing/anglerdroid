# Skill: Checkered Mat Hard-Stop (Brown Border Detection)

## Description
Detect and avoid checkered mat zones (e.g., door mat by front door) that mark:
- Dangerous areas (stairs, ledges)
- Off-limits spaces (human work areas, door transition zones)
- Furniture protection zones

Uses RS1 top-down RGB detection with THREE methods (priority order):
1. **Brown wood border detection** (PRIMARY) - HSV + rectangular frame for door mats
2. **Bump detection** - Edge detection for wood thresholds
3. **Checkerboard corners** (FALLBACK) - OpenCV corner detection (expensive, ~34ms)

Named keepout: `checkered_door` for the front door area.

## Preconditions
- RS1 top-down RGB camera operational
- Brown border detector enabled in vision extras loop (~3Hz)
- `topdown_hazard` reflex active in safety layer

## Procedure
1. **Detection phase** (~3Hz RGB extras loop, NOT 30Hz reflex)
   - RS1 RGB frame captured and rotated 180° (camera mounting)
   - Forward region extracted (default 40% of topdown view)
   - **Brown border detection** (PRIMARY):
     * Convert to HSV color space
     * Mask brown wood range (H=5-25°, S=40-255, V=30-180)
     * Find rectangular contours (perimeter >300px, area >3000px²)
     * Aspect ratio 0.3-4.0 allows both narrow and wide rectangles
     * Confidence based on rectangularity (shape quality)
   - **Bump detection** (if no brown border):
     * Canny edge detection on grayscale
     * Look for horizontal edges spanning >60% of image width
     * At least 1 consecutive row with strong edges
   - **Checkerboard corners** (FALLBACK, if both above fail):
     * OpenCV `findChessboardCorners()` with FAST_CHECK
     * Corner refinement via `cornerSubPix()`
     * Requires minimum 4 corners for 6×6 internal corner grid
   - Vision extras thread sets sticky flag `_topdown_hazard=True`

2. **Reflex response** (consumed by 30Hz capture loop)
   - 30Hz capture reads sticky `_topdown_hazard` flag
   - Safety layer zeros `fwd_scale=0.0` immediately
   - Reverse and angular authority maintained (escape capable)
   - No forward motion until pattern clears field of view (3Hz updates)

3. **Recovery phase**
   - If engaged: reverse 20-30cm to clear pattern
   - If approaching: replan route around keepout zone
   - Log event for map-based keepout persistence (RL learns avoidance)
   - Named keepout `checkered_door` reinforces door area avoidance

## Success Criteria
- Brown border detected in forward region at ~3Hz
- Forward stop within <333ms of detection (3Hz update rate)
- No entry into brown-bordered zone
- Reverse escape functional during reflex
- Named keepout `checkered_door` updated for persistent avoidance
- RL policy learns to avoid area through reward shaping

## Failure Modes
- **Brown color variation**: lighting changes affect HSV → HSV range tuned for typical indoor lighting
- **Pattern noise**: false positive on similar brown textures → conservative (stop is safe)
- **Late detection**: pattern appears very close at ~3Hz → acceptable, reflex still triggers within 333ms
- **Sensor failure**: RGB dropout → fall back to bump/checkerboard or floor obstacle map only
- **Already on mat**: stuck in keepout → stuck recovery prioritizes reverse
- **Expensive checkerboard**: findChessboardCorners ~34ms → demoted to fallback, only runs if brown+bump fail

## Integration Points
- **checkered_mat.py**: brown border + bump + checkerboard detection (RS1 RGB)
  - `_detect_brown_border()`: HSV color + rectangular contour detection
  - `TopdownHazardDetector.check()`: prioritized detection cascade
- **vision.py**: `_vision_extras_loop()` runs at ~3Hz for RGB detections
  - Parallel to 30Hz capture, avoids blocking depth reflexes
  - Sets sticky `_topdown_hazard` flag consumed by capture loop
- **safety.py**: `topdown_hazard` flag → hard-stop reflex (`fwd_scale=0`)
- **keepouts.py**: named keepout `checkered_door` for persistent map-frame avoidance
- **local_executive**: path planning avoids keepout zones
- **RL policy** (future): reward shaping discourages entering keepout zones

## Tunable Parameters
- `brown_hsv_lower`: (5, 40, 30) — HSV lower bound for brown wood
- `brown_hsv_upper`: (25, 255, 180) — HSV upper bound for brown wood
- `brown_min_perimeter`: 300 — minimum rectangle perimeter (pixels)
- `brown_min_area`: 3000 — minimum rectangle area (pixels²)
- `forward_fraction`: 0.4 — analyze forward 40% of topdown view
- `bump_edge_thresh`: 50 — Canny threshold for bump detection
- `checkerboard_rows/cols`: 6×6 — internal corners for checkerboard (FALLBACK)

## Performance
- **Brown border detection**: ~2-5ms (HSV + contours, efficient)
- **Bump detection**: ~1-2ms (Canny edges, cheap)
- **Checkerboard corners**: ~34ms (expensive, FALLBACK only)
- **Overall**: ~3Hz RGB extras loop, 0 impact on 30Hz capture budget
- **Latency**: <333ms detection-to-stop (3Hz update rate)

## Trace Markers
```json
{"event": "brown_border_detected", "confidence": 0.95, "perimeter": 1380, "reason": "brown_border"}
{"event": "topdown_hardstop", "fwd_scale": 0.0, "reason": "topdown_hazard"}
{"event": "keepout_updated", "zone_id": "checkered_door", "world_xy": [2.3, -1.5]}
```
