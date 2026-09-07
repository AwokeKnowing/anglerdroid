# Camera Grab Latency Optimization

## Problem (MEASURED on Kevin Jetson Orin NX)

After PR#22 (comprehensive profiling), real hardware shows:

```
CAPTURE TIMING (last 90 frames): TOTAL=110.8ms (9.0 Hz)
================================================================================
grab               41.4ms   37.4%  ← THE BOTTLENECK
rs1_checks          6.1ms    5.5%
rs2_gpu             6.0ms    5.4%
odom                5.1ms    4.6%
gmap                0.0ms    0.0%  (shed)
render              5.9ms    5.4%
```

**Bottleneck**: Camera grab at **41.4ms (37% of total)**, NOT render as hypothesized.

**Gap to 30 Hz**: 110.8ms → 33ms requires **77ms reduction**. If we cut grab from 41ms to 8ms, that's **33ms saved** — nearly half the gap.

## Hypothesis: Why is grab so slow?

Current grab code uses `ThreadPoolExecutor` to parallelize RS1, RS2, webcam:

```python
_cams = [c for c in (self._webcam, self._rs1, self._rs2) if c]
if len(_cams) <= 1:
    for c in _cams:
        c.grab()
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_cams)) as _ex:
        list(_ex.map(lambda c: c.grab(), _cams))
```

**Possible causes of 41ms grab**:

1. **RealSense SDK blocking wait**: Default `rs2_pipeline.wait_for_frames()` blocks until frame arrives (up to timeout)
2. **Sequential blocking**: Even with ThreadPoolExecutor, if cameras aren't ready, threads block on SDK calls
3. **Frame queue depth**: RealSense queue might be shallow (1-2 frames), causing stalls
4. **Webcam in critical path**: V4L2 webcam grab may be slower than RS cameras
5. **ThreadPoolExecutor overhead**: Creating/destroying executor pool each frame
6. **GIL contention**: Python GIL may serialize "parallel" grabs

## Optimization Strategies

### 1. Non-blocking RealSense Polling ⭐

**Current**: `wait_for_frames(timeout_ms)` blocks until frame ready  
**Better**: `poll_for_frames()` returns immediately if no frame available

```python
frameset = self.pipeline.poll_for_frames()
if frameset:
    # Process frame
else:
    # Use previous frame or skip
```

**Impact**: If RS cameras aren't synced, fastest one doesn't wait for slowest.

### 2. Increase RealSense Frame Queue Depth ⭐

**Current**: Default queue size (likely 1-2 frames)  
**Better**: Set larger queue in config

```python
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# Need to check SDK API for queue size parameter
```

**Impact**: Cameras can capture ahead, reducing wait time in grab loop.

### 3. Drop RGB Webcam from Critical Path ⭐

**Current**: All 3 cameras grabbed in parallel  
**Better**: Grab RS1 + RS2 critical, webcam async or lower priority

```python
# Critical: RS1 (topdown) + RS2 (forward depth/odom)
critical_cams = [self._rs1, self._rs2]
# Non-critical: RGB webcam (just for atlas display)
async_cams = [self._webcam]
```

**Impact**: Don't wait for slower V4L2 webcam if RS cameras are ready.

### 4. Persistent ThreadPoolExecutor ⭐

**Current**: Create/destroy executor each frame  
**Better**: Create once in `__init__`, reuse every frame

```python
self._grab_executor = ThreadPoolExecutor(max_workers=3)
# In loop:
futures = [self._grab_executor.submit(c.grab) for c in _cams]
concurrent.futures.wait(futures, timeout=0.030)  # 30ms timeout
```

**Impact**: Eliminate executor creation overhead (~1-2ms).

### 5. Async Frame Capture Thread

**Current**: Main capture loop blocks on grab  
**Better**: Separate thread continuously grabs frames, main loop reads latest

```python
# Separate thread:
while running:
    for cam in cameras:
        cam.async_grab()  # Non-blocking
        
# Main loop:
frames = [cam.get_latest_frame() for cam in cameras]
```

**Impact**: Decouple grab timing from processing timing.

### 6. Reduce RealSense Resolution/Decimation

**Current**: Full resolution capture  
**Check**: Are we using 848×480 or 640×480? Can we decimate more?

```python
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)  # Half res
```

**Impact**: Faster capture, less data transfer.

### 7. Timeout Tuning

**Current**: Unknown timeout value  
**Better**: Set aggressive timeout (15-20ms) so grab doesn't wait forever

```python
frames = pipeline.wait_for_frames(timeout_ms=20)
```

**Impact**: Don't wait >20ms for a single camera.

## Implementation Plan

### Phase 1: Quick Wins (PR #24)

1. ✅ **Persistent ThreadPoolExecutor**: Eliminate per-frame creation overhead
2. ✅ **Aggressive timeout**: Set 20ms timeout for each camera grab
3. ✅ **Drop webcam if slow**: If webcam grab >10ms, skip it for that frame

**Expected impact**: 5-10ms reduction (41ms → 31-36ms)

### Phase 2: Non-blocking Polling (PR #25)

1. **poll_for_frames()**: Replace blocking wait with non-blocking poll
2. **Frame reuse**: If poll returns nothing, reuse previous frame
3. **Queue depth**: Increase RS queue to 3-5 frames

**Expected impact**: 10-15ms reduction (31ms → 16-21ms)

### Phase 3: Async Capture (if needed)

1. **Separate capture thread**: Continuously grab frames
2. **Main loop reads latest**: No blocking on grab

**Expected impact**: 15-20ms reduction (grab becomes nearly free in main loop)

## Success Criteria

- **Target**: grab <10ms (currently 41ms)
- **Total**: TOTAL <33ms for 30 Hz (currently 111ms)
- **Safety**: RS1 topdown + safety checks never shed

## Testing

1. Deploy to Kevin Jetson Orin NX
2. Run vision-only smoke (wheelbase=None)
3. Collect CAPTURE TIMING output every 90 frames
4. Verify:
   - grab time reduced
   - TOTAL <33ms
   - Soft-low still works
   - No frame drops in RS1 (immobilize ground truth)

## Risks

1. **Frame drops**: Non-blocking polling may miss frames
   - **Mitigation**: Always wait for RS1 (safety critical), poll RS2/webcam
2. **Latency**: Using old frames increases latency
   - **Mitigation**: Measure frame age, drop if >50ms old
3. **Complexity**: Async capture adds threading complexity
   - **Mitigation**: Start with simpler approaches (persistent executor, timeout)
