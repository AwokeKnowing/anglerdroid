# Camera Configuration Fix: 11 Hz → 30 Hz Capture

## Problem

After JetPack/library reinstall, Vision capture degraded from **60 fps** to **11 Hz** (~110ms/frame):
- `grab`: 41ms (37%)
- `pose+hazard`: 44ms (40%)
- Other stages: 25ms (23%)

## Root Causes

### 1. Camera Grab Bottleneck (41ms → <5ms target)

**Issue**: RealSense frame acquisition was stalling for 41ms per grab cycle.

**Root causes identified**:
- **Frame queue depth**: Queue size = 2 allowed stale frames to accumulate
- **Long timeout**: 50ms `wait_for_frames()` timeout masked underlying issues
- **Blocking behavior**: `wait_for_frames()` blocks until frameset arrives
- **CRITICAL: Unbounded exposure time**: Auto-exposure can stretch >30ms in low light, making 30 Hz physically impossible

**Pre-reinstall config** (60 fps):
- Streams delivered frames immediately
- Non-blocking grab succeeded on first poll
- No inter-stream sync delays

**Post-reinstall regression** (11 Hz):
- Streams sometimes not ready on poll
- Fallback to 50ms `wait_for_frames()` caused stalls
- Multiple cameras grabbing in parallel compounded delays

**Fix**:
1. **Reduced queue size to 1**: Ensures `poll_for_frames()` always returns the NEWEST frame
2. **Reduced timeout to 5ms**: At 30 Hz (33ms/frame), 5ms grace period is sufficient
3. **Fast-fail behavior**: If frame not ready in 5ms, skip and continue (better to drop a frame than stall)
4. **CRITICAL: Capped exposure time to 20ms**: Set `auto_exposure_priority=0` (frame rate priority) and cap exposure at 20ms max
5. **Increased gain to compensate**: Bump gain from 16 to 18 to maintain image quality in low light

```python
# cameras.py: RSCamera.__init__()
for sens in self.profile.get_device().sensors:
    _set_sensor_opt(sens, rs.option.frames_queue_size, 1)  # Was: 2

# Cap exposure to ensure frame time <33ms (CRITICAL for 30 Hz)
_set_sensor_opt(sensor, rs.option.auto_exposure_priority, 0)  # Frame rate priority
_set_sensor_opt(sensor, rs.option.exposure, 20000)  # 20ms max (µs)
_set_sensor_opt(sensor, rs.option.receiver_gain, 18)  # Was: 16

# cameras.py: RSCamera.grab()
def grab(self, timeout_ms=5):  # Was: 50
    frames = self._pipe.poll_for_frames()  # Non-blocking first
    if not frames:
        frames = self._pipe.wait_for_frames(timeout_ms)  # Brief grace period

# cameras.py: WebCam.__init__()
self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
self._cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Short exposure (~15ms)
```

### 2. Hazard Detection Bottleneck (44ms → ~15ms amortized)

**Issue**: `TopdownHazardDetector.check()` runs expensive OpenCV operations every frame:
- `cv2.findChessboardCorners()`: 20-30ms (searches for 6x6 internal corners)
- `cv2.cornerSubPix()`: 5-10ms (refines corner positions)
- `cv2.Canny()`: 5-10ms (edge detection for bump)

**Analysis**:
- Floor hazards (checkered mat, wood bump) are STATIC
- No need to check every frame at 30 Hz
- Detection at 10 Hz (every 3rd frame) is sufficient for slow-moving robot

**Fix**:
```python
# vision.py: _capture_loop()
HAZARD_CHECK_INTERVAL = 3  # Check every 3rd frame (10 Hz at 30 fps)
if self._hazard_check_counter % HAZARD_CHECK_INTERVAL == 0:
    hazard_triggered, hazard_reason = self._topdown_hazard_detector.check(rs1_rgb_rotated)
```

**Impact**: Reduces from 44ms every frame to ~15ms amortized (44ms / 3).

## Expected Performance

### Before Fix
- **Total**: 110ms/frame (9.0 Hz)
- **grab**: 41ms (37%)
- **pose+hazard**: 44ms (40%)
- **Other**: 25ms (23%)

### After Fix (Target)
- **Total**: <33ms/frame (>30 Hz)
- **grab**: <5ms (aggressive poll + short timeout)
- **pose+hazard**: ~15ms amortized (check every 3rd frame)
- **Other**: ~13ms (rs1_checks, rs2_gpu, odom, gmap, render)

## Stream Configuration

Verified correct configuration matches pre-reinstall expectations:

### RealSense D435/D435i (RS1, RS2)
- **Depth**: 848x480 @ 30 Hz (z16 format)
- **Color**: 320x240 @ 30 Hz (rgb8 format)
- **IR** (if enabled): 848x480 @ 30 Hz (y8 format, stereo pair)

### USB Webcam
- **Capture**: 640x480 @ 30 Hz (MJPG)
- **Resize**: 320x240 RGB (for display)
- **Buffer**: 1 frame (minimize latency)

## Safety Considerations

### Hazard Detection Frame Skipping
- **Risk**: Skipping frames could miss transient hazards
- **Mitigation**: Floor hazards (checkered mat, bump) are STATIC and slow-moving robot ensures detection within 100ms (3 frames @ 30 Hz) is sufficient
- **Fallback**: Depth-based safety checks (near-field, overhang, soft-low) still run every frame

### Camera Grab Fast-Fail
- **Risk**: Failing fast could drop frames
- **Mitigation**: At 30 Hz with queue_size=1, frames should always be ready on poll. If 5ms timeout expires, it indicates a config/hardware issue that needs investigation, not masking with long waits
- **Diagnostic**: Timeout counters log when grabs fail, enabling detection of underlying issues

## Testing

### Hardware Verification Required
Run on Kevin Jetson Orin NX (vision-only, wheelbase=None) with:
```bash
python3 src/main.py --wheelbase none --silent
```

### Expected Results
1. **Odom log**: `dt ≈ 0.033s` (30 Hz)
2. **CAPTURE TIMING report** (every 90 frames):
   - **TOTAL**: <33ms (>30 Hz)
   - **grab**: <5ms
   - **pose+hazard**: ~15ms amortized
3. **Soft-low safety**: Still triggers correctly (test with `fwd=0.3` at obstacle)

### Diagnostic Logs
- Camera init: `cameras: RS <serial> — exposure=Xms gain=Y ae_priority=0`
- Webcam init: `cameras: webcam opened (actual=WxH@FPS exp=X ae=Y)`
- Grab timeouts (should be RARE): `vision: RS1/RS2 grab timeout (count=N)`
- Capture timing (every 90 frames): `RS1/RS2 exposure: Xms, gain: Y` (verify exposure ≤20ms)
- Hazard detection: Triggers/clears as expected

## Rollback Plan

If 30 Hz not achieved or safety regressions occur:
1. Revert `cameras.py` queue_size to 2 and timeout to 50ms
2. Revert `vision.py` HAZARD_CHECK_INTERVAL to 1 (every frame)
3. File detailed bug report with timing data

## Key Insight: Exposure Time is a Hardware Constraint

**James's insight**: If auto-exposure stretches exposure time >33ms, the camera **physically cannot deliver 30 fps** regardless of processing speed.

Example:
- Exposure = 40ms → Camera frame time = 40ms minimum → Max FPS = 25 Hz
- Exposure = 20ms → Camera frame time = 20ms + readout (~3ms) → Can sustain 30+ Hz

**Solution**: Cap exposure at 20ms, increase gain to compensate for low-light scenes. Frame rate priority over perfect exposure.

## Future Optimizations (if needed)

If <33ms total not achieved after these fixes:
1. **Async hazard detection**: Run in background thread (complex, needs synchronization)
2. **GPU hazard detection**: Port OpenCV operations to CUDA (requires cuVSLAM team support)
3. **Reduce RS2 resolution**: 848x480 → 640x480 depth (less USB bandwidth, faster GPU processing)
4. **Disable RS2 color**: Only grab depth (if color not needed for atlas display)
5. **Adjust exposure cap**: If 20ms too dark, try 25ms (still <33ms), or add IR illumination
