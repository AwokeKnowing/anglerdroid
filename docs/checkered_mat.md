# Wood Bump + Checkered Mat Detector (Ego-Frame RS1 Hard-Stop)

## Overview

Vision-based safety reflex that hard-stops forward motion when **wood floor bump/lip** or **checkered door mat** is detected ahead of robot. Uses **RS1 top-down RGB** (NOT webcam). Works entirely in ego-frame (no SLAM, no map, no pose). Classical CV for real-time 30Hz loop on Jetson.

**Primary hazards detected:**
- **(a) Wood floor bump/lip**: Texture transition, strong horizontal edge, color shift
- **(b) Checkered door mat**: Alternating black-white grid pattern

## How It Works

### Detection Pipeline

**Input**: RS1 top-down RGB (already rotated 180° in vision.py)
- Forward = right side of frame after rotation
- Backward = left side of frame after rotation

1. **ROI Extraction**: Forward portion (50-90% of width = ~40cm ahead of robot)
2. **Preprocessing**: Grayscale + CLAHE adaptive contrast enhancement

3. **Wood Bump Detection**:
   - **Strong horizontal edges** in mid-region (Canny + Hough lines)
   - **Edge density** in middle rows (skip top/bottom 30% for shadows/robot)
   - **Scoring**: `wood_score = 0.6×h_line_score + 0.4×edge_density`
   - Wood bump = floor level change, texture transition (smooth → rough)

4. **Checkered Mat Detection**:
   - **Corners**: `cv2.goodFeaturesToTrack` → checkered has high corner density at grid intersections
   - **Lines**: Canny + Hough → checkered has perpendicular H+V lines
   - **Contrast**: Laplacian variance → alternating dark/light tiles
   - **Regularity**: Corner distribution across grid → evenly spaced
   - **Scoring**: `checker_score = 0.20×density + 0.20×lines + 0.35×grid_orth + 0.15×variance + 0.10×regularity`
   - Grid orthogonality heavily weighted (requires BOTH H and V lines)
   - Penalty (×0.7) if no proper grid → filters stripes/noise

5. **Combined Score**: `score = MAX(wood_score, checker_score)`
   - Triggers on EITHER hazard type

6. **Trigger**: `score ≥ MIN_SCORE` (default 0.20, lower for subtle wood bump)
7. **Hysteresis**: Once triggered, require `score < MIN_SCORE - HYSTERESIS` to clear (prevents flicker)

### Safety Response

When `checkered_mat=True` (wood bump OR checkered detected):
- **Forward motion**: `fwd_scale = 0.0` (hard stop)
- **Backward motion**: Allowed if rear clear (computed normally from obstacles)
- **Angular motion**: Allowed (can rotate in place)
- **Logging**: "WOOD/CHECKER — score=X.XX wood=X.XX checker=X.XX (RS1 ego-frame hard-stop)"

Same behavior as `topdown_near_field` reflex (table underside detection).

## Tunable Constants

Located at top of `src/checkered_mat.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `ROI_FORWARD_START` | 0.50 | Forward ROI start as fraction of width (0.5 = center/robot body) |
| `ROI_FORWARD_END` | 0.90 | Forward ROI end as fraction of width (0.9 = ~40cm ahead) |
| `MIN_SCORE` | 0.20 | Detection threshold (0.0-1.0, lower for subtle wood bump) |
| `HYSTERESIS` | 0.10 | Score delta for hysteresis (prevents flicker) |
| `MIN_CORNER_DENSITY` | 0.08 | Minimum corner density for checkered (corners per 100 pixels) |
| `HOUGH_THRESHOLD` | 15 | Hough line detection threshold |
| `HOUGH_MIN_LINE_LEN` | 10 | Minimum line length (pixels) |
| `HOUGH_MAX_LINE_GAP` | 5 | Maximum gap in line (pixels) |

### Tuning Guidance

**Increase sensitivity** (detect hazards farther ahead):
- Lower `MIN_SCORE` (e.g., 0.15) — more sensitive, may trigger on textured floors
- Increase `ROI_FORWARD_END` (e.g., 0.95) — larger ROI, detects farther ahead

**Decrease sensitivity** (reduce false positives):
- Raise `MIN_SCORE` (e.g., 0.25) — less sensitive, only strong wood/checkered patterns
- Decrease `ROI_FORWARD_END` (e.g., 0.80) — smaller ROI, only very close hazards

**Adjust ROI for detection distance**:
- Detect closer: lower `ROI_FORWARD_START` (e.g., 0.40) to include more area
- Detect farther: raise `ROI_FORWARD_START` (e.g., 0.60) to skip robot body

## Integration

### Vision (`src/vision.py`)

In capture loop (~line 813):
```python
# Wood bump + checkered mat detection (ego-frame, RS1 RGB, no SLAM)
# Uses RS1 top-down color (already rotated 180° above)
if self._rs1 and self._rs1.ok:
    triggered, score, meta = detect_checkered_mat(
        rgbd1,  # RS1 color rotated 180°
        prev_triggered=self._checkered_prev)
    self._checkered_mat = triggered
    # ... logging: wood_score, checker_score ...

self._safety.update(..., checkered_mat=self._checkered_mat)
```

Properties exposed:
- `vision.checkered_mat` → bool
- `vision.checkered_mat_score` → float (0.0-1.0)

### Safety (`src/safety.py`)

```python
def update(self, obs_map, yaw_delta, fwd_delta, ..., checkered_mat=False):
    if checkered_mat:
        self._near_field_reason = "checkered_mat"
        self._fwd_scale = 0.0  # Hard stop forward
    # ... bwd/ang computed normally from obstacles ...
    
    # Don't override reflex:
    if not topdown_near_field and not checkered_mat:
        self._fwd_scale = _clearance_scale(fwd_clear)
```

## Relationship to Map Keepouts

**Visual detector (primary)**: Ego-frame reflex using RS1 RGB, no SLAM. Works immediately, no prior map needed. Detects wood bump + checkered mat ahead.

**Map paint (backup)**: `keepouts.py` `floor_mat` kind now paints HARD (200) instead of SOFT (90). This is a **backup only** because:
- Requires SLAM pose (**currently unreliable / map keepouts disabled until SLAM locked**)
- Requires user to mark mat location in map
- Disk in map may drift if pose drifts

**Recommendation**: Rely on visual detector (RS1 RGB). Map paint is disabled until SLAM is stable.

### Paint Keepouts Change

In `src/keepouts.py`:
- Removed `"floor_mat"` from `_SOFT_KINDS`
- `paint_value_for_kind("floor_mat")` now returns `PAINT_HARD` (200) instead of `PAINT_SOFT` (90)
- Self-test updated to assert `floor_mat` is HARD

This ensures map-painted checkered_door marks are hard obstacles (>100 = OBS_THRESH) for MPPI/VFH if visual detector misses.

## Testing

Run: `pytest src/test_checkered_mat.py` or `python src/test_checkered_mat.py`

Tests (11 passed):
- ✅ Wood bump/lip triggers detection
- ✅ Checkered pattern (20px, 40px squares) in forward ROI triggers
- ✅ Plain floor does not trigger
- ✅ Noisy floor distinguishable by corner count
- ✅ Horizontal/vertical stripes distinguishable by zero corners
- ✅ Forward ROI exclusion works (backward checkered ignored)
- ✅ SafetyGuard stops forward when triggered
- ✅ SafetyGuard allows motion when clear
- ✅ Empty/invalid frames handled gracefully

## Real-World Usage

### Recommended Mat Specifications
- Square size: 15-40 pixels at typical floor distance (~50-100cm from webcam)
- Pattern: Classic black/white checkerboard (alternating squares)
- Placement: Visible in webcam near-field view (bottom portion of frame)

### Fine-Tuning Procedure
1. Place physical checkered mat in robot's near-field view
2. Monitor detection score and corner count in logs
3. Adjust `MIN_SCORE` to balance sensitivity vs. false positives
4. Adjust ROI if mat is not in expected region
5. Test with drive DISARMED first, then ARMED in controlled environment

### Distinguishing Real Mats from False Positives

Checkered mats have:
- **Moderate corner count** (20-50 corners in ROI)
- **Grid orthogonality** (both H and V lines detected)
- **Even corner distribution** (regularity score >0.5)

False positives (stripes, noise):
- **Stripes**: zero corners, only H or V lines
- **Noise**: very high corner count (>200), poor grid orthogonality

If needed, add corner-count gating: `triggered and 10 < meta["corners"] < 100`.

## Limitations

- **Lighting**: Performance degrades in very low light or high glare (CLAHE helps but not perfect)
- **Camera**: Requires RS1 top-down camera (will not work if RS1 fails)
- **Wood bump**: Subtle transitions may not trigger; strong edges work best
- **Square size**: Very small checkered squares (<5px) may not be detected (recommend 15-40px)
- **Pattern variants**: Only detects classic black/white checkerboard (not rotated 45°, colored, etc.)

## Logging

**Detection triggered** (every 30 frames):
```
vision: WOOD/CHECKER — score=0.45 wood=0.12 checker=0.45 (RS1 ego-frame hard-stop)
safety: CHECKERED MAT REFLEX — vision detected mat under/near robot, fwd=0.0, computing bwd/ang normally
```

**Detection cleared**:
```
vision: WOOD/CHECKER cleared — score=0.15 (after 18 frames)
```

## Architecture

Follows the same pattern as `topdown_near_field` reflex:
- **Vision module** (`checkered_mat.py`): Detector helper function for wood bump + checkered
- **Vision capture loop**: Calls detector on RS1 RGB (after 180° rotation), stores `_checkered_mat` flag
- **Safety update**: Accepts `checkered_mat` parameter, zeros `fwd_scale` when True
- **Clearance scale gate**: `if not topdown_near_field and not checkered_mat: ...`

Keeps `vision.py` from growing: detector logic isolated in `checkered_mat.py` (~200 lines).

## Implementation Files

- `src/checkered_mat.py` — Detector implementation
- `src/vision.py` — Wiring (capture loop + properties)
- `src/safety.py` — Reflex logic (fwd_scale=0)
- `src/keepouts.py` — Map paint hardened (floor_mat → HARD)
- `src/test_checkered_mat.py` — Unit tests
- `docs/checkered_mat.md` — This documentation

## Quick Start

1. **Test detector standalone**:
   ```bash
   pytest src/test_checkered_mat.py
   ```

2. **Tune for your mat**:
   - Edit `MIN_SCORE` in `src/checkered_mat.py` (lower = more sensitive)
   - Edit `ROI_TOP_FRAC`, `ROI_BOTTOM_FRAC` to match camera view

3. **Test with robot** (drive DISARMED):
   - Place mat in webcam view
   - Run main loop, monitor logs for "CHECKERED MAT — score=..."
   - Verify score >0.25 when mat visible, <0.25 when not

4. **Deploy**:
   - Once tuned, test with drive ARMED in controlled environment
   - Verify forward stops when mat detected, reverse/turn still work

## Future Enhancements

- **Graduated response**: Scale `fwd_scale` by score (0.20→0.5x, 0.25→0.25x, 0.30→0.0x)
- **Temporal filtering**: Require N consecutive frames to trigger (reduce false positives)
- **Multi-camera**: Use topdown camera as secondary input (redundant detection)
- **Pattern variants**: Support rotated 45° checkerboard, colored patterns
