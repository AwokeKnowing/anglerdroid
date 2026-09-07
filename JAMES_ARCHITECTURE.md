# James Architecture: Vision Loop Separation

## Overview
This document describes the James architecture refactoring that separates RGB-based detections from the 30Hz capture loop to ensure fresh frame acquisition.

## Architecture Principles

### 1. 30Hz Capture Loop (CRITICAL PATH)
**Must get fresh RS+RGB frames; processing finishes before next frame.**

The capture loop (`vision._capture_loop()`) handles:
- ✅ Camera frame grabs (parallel RS1 + RS2 + webcam)
- ✅ Depth processing (RS1 topdown, RS2 forward)
- ✅ Depth reflexes (near-field, overhang, soft-low obstacles)
- ✅ Pose estimation (wheel + visual odometry, cuVSLAM)
- ✅ Global map updates
- ✅ Safety scaling

**Removed from 30Hz loop:**
- ❌ slow RGB pattern matching (chessboard corners)
- ❌ RGB-based pattern recognition (checkered/bump detection)
- ❌ Face detection (future)
- ❌ Gesture recognition (future)

### 2. Vision Extras Loop (~3fps, OPTIONAL i777)
**RGB image detections that do NOT need 30Hz.**

The extras loop (`vision._vision_extras_loop()`) handles:
- ✅ Brown-border / bump RGB detection (TopdownHazardDetector on RS1 RGB)
- 🔮 Face detection (future: optional i777 detection stream)
- 🔮 Gesture recognition (future)
- 🔮 Person tracking (future)

**Design notes:**
- Runs at ~3fps (independent thread, separate from capture)
- Uses RS1 RGB frames (rotated 180° for correct orientation)
- Detection results stored in shared state (thread-safe via Vision._lock where needed)
- Safety layer reads `_topdown_hazard` flag from extras loop
- Zero impact on 30Hz capture loop timing

### 3. Depth Reflexes (30Hz, STAY ON CAPTURE)
**Fast depth-based checks must stay on 30Hz for immediate response.**

These remain in the capture loop:
- ✅ Near-field reflex (<30cm, any obstacle)
- ✅ Overhang approach (30-70cm, table underside)
- ✅ Soft-low obstacle (35cm-1m, 5-30cm height, dog bed/cushion)

**Why these stay at 30Hz:**
- Pure depth processing (no slow OpenCV corner detection)
- Single-pass optimized (check_topdown_all_in_one)
- Required for immediate safety response
- No expensive pattern matching

## Special Cases

### Door Checkered Mat (`checkered_door` keepout)
- Black/white checkered mat by front door
- Detection at ~3fps in extras loop (not 30Hz)
- Named keepout `checkered_door` persists in map
- RL policy should learn to avoid (high stuck risk)
- Hard-stop reflex when detected (fwd_scale=0)

## Implementation

### Thread Structure
```python
# vision.py
class Vision:
    def start(self):
        # 30Hz capture loop
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # ~3fps extras loop (RGB detections)
        self._extras_thread = threading.Thread(target=self._vision_extras_loop, daemon=True)
        self._extras_thread.start()
```

### Capture Loop (30Hz)
```python
def _capture_loop(self):
    # 1. Grab frames (parallel)
    # 2. Pose update
    # 3. RS1 depth checks (near/overhang/soft-low) - DEPTH ONLY
    # 4. RS2 forward depth
    # 5. Combine obstacles
    # 6. Odometry
    # 7. Global map
    # 8. Safety
    # 9. Render
```

### Extras Loop (~3fps)
```python
def _vision_extras_loop(self):
    # 1. Checkered/bump detection (RS1 RGB)
    # 2. (Future) Face detection
    # 3. (Future) Gesture recognition
    # Note: Detection results stored in _topdown_hazard, _topdown_hazard_reason
```

### State Synchronization
- Extras loop writes: `_topdown_hazard`, `_topdown_hazard_reason`, counts
- Capture loop reads: depth reflex flags
- Safety layer reads: all hazard flags (from both loops)
- Thread-safe: extras loop updates are simple assignments (Python GIL)

## Performance

### Before (30Hz loop with RGB detection)
- Capture loop: ~30-35ms (including findChessboardCorners)
- Frame rate: ~28-30 Hz (borderline)
- Risk of frame drops when checkered pattern detected

### After (30Hz capture + 3fps extras)
- Capture loop: ~25-28ms (pure depth processing)
- Frame rate: solid 30 Hz
- Extras loop: ~300ms budget per check (plenty of time for OpenCV)
- Zero interference between loops

## Future Work (Optional i777)

### i777 Detection Stream
The extras loop is designed to optionally integrate with i777 detection services:
- High-level RGB detections (faces, gestures, objects)
- Runs on separate hardware/process
- Results consumed at ~3fps in extras loop
- Zero impact on 30Hz capture

### Integration Points
```python
# vision._vision_extras_loop()
# TODO: Add i777 detection stream integration
# - Subscribe to i777 face detection topic
# - Parse detection results
# - Update person tracking state
# - Trigger social FSM events
```

## Testing

### Verify 30Hz Capture Performance
1. Run with checkered mat in view
2. Check capture loop timing logs
3. Verify zero findChessboardCorners calls in capture path
4. Confirm solid 30 Hz frame rate

### Verify Extras Loop Detection
1. Place checkered mat in forward region
2. Check `vision_extras:` log messages at ~3fps
3. Verify hard-stop reflex triggers (fwd_scale=0)
4. Confirm detection clears when mat removed

### Verify Depth Reflexes (30Hz)
1. Approach table (overhang detection)
2. Approach dog bed (soft-low detection)
3. Hand over topdown camera (near-field detection)
4. All should trigger within 1-2 frames (<50ms)

## References
- `src/vision.py`: Vision class with dual loops
- `src/checkered_mat.py`: TopdownHazardDetector (RGB-based)
- `skills/checkered_mat_keepout.md`: Skill documentation with RL notes
- `src/safety.py`: SafetyGuard reads hazard flags from both loops


## Door mat cue
Prefer **brown border HSV** around the front-door checkered floor (named keepout `checkered_door`), not chessboard corners on the 30Hz path. RL learns stuck risk.


## Orin embedded performance (James 2026-09-07)

The split above is necessary but not sufficient. On Orin:

- Target the **theoretically minimum physically possible** frame time.
- Keep frames **on GPU**; CPU is for control/orchestration, not bulk pixels.
- Choose **when** to project / deproject / join so the expensive ops see the
  fewest points (decimate early).
- Ship viz/logs to i777 on a **side thread** (downsize→compress→send) with a
  bounded queue — drop frames rather than stall capture.
- Continuously ask: what is each core doing *this* millisecond?

Regressions to watch for after refactors: undecimated NumPy over full RS
pointclouds, per-frame ThreadPoolExecutor create/destroy, AE exposure >33ms,
`wait_for_frames` timeouts that eat multiple periods, atlas `.copy()` on the
critical path when a view would do.
