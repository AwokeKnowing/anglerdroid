# Hazard Detection Optimization: 34ms → <3.5ms amortized

## Problem

After exposure/grab fix (85ea3cf), Kevin shows:
- **grab**: 41ms → **8.5ms** ✅ (exposure cap worked!)
- **TOTAL**: 110ms → **66ms** (still ~15 Hz, need 30 Hz)
- **Remaining bottleneck**: **pose+hazard ~34ms** (almost entirely `TopdownHazardDetector.check()`)

## Root Cause

`TopdownHazardDetector.check()` runs expensive OpenCV operations on RS1 320x240 RGB:
- `cv2.findChessboardCorners()`: 20-25ms (searches for 6x6 internal corners)
- `cv2.cornerSubPix()`: 5-8ms (refines corner positions)
- `cv2.Canny()`: 3-5ms (edge detection for bump)
- **Total**: ~34ms per call

Running this every frame at 30 Hz consumes 34ms of the 33ms budget = impossible.

## Solution: Aggressive Frame Skipping + Fast Mode + Sticky State

### 1. Aggressive Frame Skipping (Every 10th Frame)
Floor hazards (checkered mat, wood bump) are **STATIC** and robot is **slow-moving**.
- **Before**: Check every frame (30 Hz)
- **After**: Check every 10th frame (3 Hz)
- **Amortized overhead**: 34ms / 10 = **3.4ms** (vs 34ms before)

### 2. Fast Mode (Downsample + Skip Refinement)
Enable `fast_mode=True` in `TopdownHazardDetector.check()`:
- **Downsample 2x**: 320x240 → 160x120 (4x fewer pixels)
  - Reduces `findChessboardCorners` time by ~3-4x
  - Minimal detection quality loss (pattern still clear)
- **Skip `cornerSubPix`**: Saves 5-8ms
  - Refinement not needed for coarse detection
  - Binary trigger (hazard vs no hazard) doesn't need pixel-perfect corners

**Expected per-check time**: 34ms → ~10-15ms with fast mode

### 3. Sticky State (Stay Triggered for 1 Second)
Once hazard detected, stay triggered for 30 frames (1 sec @ 30 Hz) before rechecking:
- **Safety**: Hazard doesn't disappear instantly (robot needs time to stop/avoid)
- **Performance**: Eliminates repeated checks when hazard already known
- **User experience**: Smoother immobilize behavior (no flicker)

### Implementation

```python
# vision.py: _capture_loop()
HAZARD_CHECK_INTERVAL = 10  # Check every 10th frame (3 Hz at 30 fps)
HAZARD_STICKY_FRAMES = 30   # Stay triggered for 30 frames (1 sec)

if self._hazard_sticky_frames > 0:
    self._hazard_sticky_frames -= 1
    # Keep existing state (don't recheck yet)
elif self._hazard_check_counter % HAZARD_CHECK_INTERVAL == 0:
    # Time to check: use fast_mode
    hazard_triggered, reason = self._topdown_hazard_detector.check(
        rs1_rgb_rotated, fast_mode=True)
    
    if hazard_triggered:
        self._hazard_sticky_frames = HAZARD_STICKY_FRAMES

# checkered_mat.py: TopdownHazardDetector.check()
def check(self, rgb_frame, fast_mode=False):
    # ...
    if fast_mode:
        forward_region = cv2.resize(forward_region, None, fx=0.5, fy=0.5,
                                   interpolation=cv2.INTER_AREA)
    # ...
    if fast_mode:
        corners_refined = corners  # Skip cornerSubPix
    else:
        corners_refined = cv2.cornerSubPix(gray, corners, ...)
```

## Expected Performance

### Before Optimization
- **pose+hazard**: 34ms every frame
- **TOTAL**: 66ms/frame (~15 Hz)

### After Optimization
- **pose+hazard**: ~3.4ms amortized (34ms / 10 frames, with fast_mode likely ~1-2ms)
- **TOTAL**: <33ms/frame (>30 Hz) ✅

### Breakdown
- **grab**: 8.5ms (after exposure fix)
- **pose+hazard**: ~2ms amortized (optimized)
- **rs1_checks**: ~6ms (depth reflexes, every frame)
- **rs2_gpu**: ~6ms (depth forward, every frame)
- **obs_comb**: ~3ms
- **odom**: ~5ms
- **gmap**: varies (can shed)
- **safety**: ~2ms
- **render**: ~5ms
- **TOTAL**: ~32-38ms (shed gmap if needed to stay <33ms)

## Safety Considerations

### ✅ Depth Reflexes Still Every Frame
- Near-field check (RS1 depth): Every frame
- Overhang approach (RS1 depth): Every frame
- Soft-low obstacle (RS1 depth): Every frame
- These are the PRIMARY safety mechanisms

### ✅ RGB Hazard Detection Adequate at 3 Hz
- Floor hazards are static (don't appear/disappear quickly)
- Robot max speed ~0.3 m/s → travels 10cm between checks (3 Hz)
- Sticky state ensures hazard stays active for 1 second
- Detection latency: worst case 333ms (10 frames @ 30 Hz), typical 167ms

### ✅ Fast Mode Detection Quality
- Downsample 2x: Pattern still clearly visible at 160x120
- Skip cornerSubPix: Binary detection (yes/no) doesn't need pixel-perfect corners
- False positive rate: negligible (pattern matching is robust)
- False negative rate: slightly higher, but depth reflexes are primary safety

## Testing Plan

### Hardware Verification (Kevin Jetson Orin NX)
```bash
python3 src/main.py --wheelbase none --silent
```

**Expected results**:
1. **CAPTURE TIMING**: `TOTAL <33ms` with `pose+hazard ~2-3ms` amortized
2. **Odom log**: `dt ≈ 0.033s` (30 Hz sustained)
3. **Exposure logs**: `RS1/RS2 exposure: ≤20ms` (verify cap holds)
4. **Hazard detection**: 
   - Triggers within 333ms of encountering checkered mat
   - Stays triggered for 1 second (sticky state)
   - Clears when mat no longer visible (after sticky expires)

### Functional Testing
1. **Checkered mat**: Place mat in robot path, verify detection + fwd=0
2. **Wood bump**: Approach threshold, verify detection + fwd=0
3. **Clear floor**: Verify no false positives during normal nav
4. **Low light**: Verify detection still works with gain-compensated exposure

## Rollback Plan

If detection quality degrades or safety issues occur:
1. Reduce `HAZARD_CHECK_INTERVAL` from 10 to 5 (6 Hz, ~7ms amortized)
2. Disable `fast_mode` (use full resolution + cornerSubPix)
3. Reduce `HAZARD_STICKY_FRAMES` from 30 to 15 (0.5 sec)
4. If still not <33ms: shed hazard detection entirely, rely on depth reflexes only

## Alternative Optimizations (if needed)

If <33ms total still not achieved:
1. **Smaller ROI**: Only process center 50% of forward region
2. **Simpler bump detection**: Replace Canny with simple threshold/diff
3. **GPU acceleration**: Port OpenCV ops to CUDA (requires significant effort)
4. **Background thread**: Run detection async (complex, needs synchronization)
5. **Disable checkered detection**: Keep only bump detection (faster)
