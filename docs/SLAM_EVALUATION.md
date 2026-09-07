# SLAM Quality Evaluation Guide

This document describes how to evaluate SLAM quality from logs, Rerun recordings, and live metrics for the Kevin anglerdroid.

## Live Monitoring

### Key Metrics to Watch

#### 1. **Pose Tracking Quality**

```python
# From main.py or vision debugging
vis = vision_instance
pose_quality = vis._pose.get_tracking_quality()

# Watch these values:
print(f"Visual accept rate: {pose_quality['visual_accept_rate']:.1%}")
print(f"Wheel-only rate: {pose_quality['wheel_only_rate']:.1%}")
print(f"Time since visual: {pose_quality['time_since_visual']:.1f}s")
print(f"Excessive disagreement: {pose_quality['excessive_disagreement']}")
```

**Good values:**
- Visual accept rate: >60% during normal driving
- Wheel-only rate: <40%
- Time since visual: <2.0s during motion
- Excessive disagreement: <10% of frames

**Bad signs:**
- Visual accept rate dropping below 30% → visual odometry failing
- Time since visual >5s → tracking loss
- Excessive disagreement >20% → wheel slip or carpet issues

#### 2. **Encoder Health**

```python
# From wheelbase instance
wb = wheelbase_instance
enc_health = wb.get_encoder_health()

print(f"Encoder OK: {enc_health['encoder_ok']}")
print(f"Data age: {enc_health['age_s']:.2f}s")
print(f"Mode: {enc_health['mode']}")
```

**Good values:**
- encoder_ok: True
- age_s: <0.1s during motion
- mode: native_can (faster than SDO)

**Bad signs:**
- encoder_ok: False → CAN communication failure
- age_s: >0.3s → fallback to commanded velocity (slip invisible)
- mode: sdo after many native_fails → slower polling

#### 3. **SLAM Statistics**

```python
# From SLAM backend
slam = vision_instance._global_map
stats = slam.stats()

print(f"Keyframes: {stats['keyframes']}")
print(f"Edges: {stats['edges']}")
print(f"Loop closures: {stats['loop_closures']}")
print(f"Memory: {stats['memory_mb']:.1f} MB")
print(f"Last loop shift: {stats['last_loop_shift_m']:.3f}m")
```

**Good values:**
- Keyframes: 50-300 (depends on session length)
- Loop closures: >0 after exploring >2m²
- Last loop shift: <0.10m (good alignment)
- Memory: <400 MB

**Bad signs:**
- Keyframes: >500 (memory pressure, pruning kicking in)
- Loop closures: 0 after 5+ minutes → descriptor matching failing
- Last loop shift: >0.50m → large pose jumps on closure

## Log Analysis

### Parsing SLAM Events

#### Loop Closure Detection

```bash
# Extract loop closure events from logs
grep "slam: loop closure" kevin.log

# Example output:
# slam: loop closure #1  kf42↔kf8  dist=2.34m  desc=0.62  match=0.48
```

Fields:
- `kf42↔kf8`: Keyframe IDs connected
- `dist=2.34m`: Spatial distance in map
- `desc=0.62`: Descriptor similarity (higher = better match, threshold 0.40)
- `match=0.48`: Scan-match score (higher = better, threshold 0.25)

**Good signs:**
- desc >0.50 → strong place recognition
- match >0.35 → good scan alignment
- dist >1.5m → meaningful loop, not just adjacent keyframes

**Bad signs:**
- desc <0.45 → weak match, may be false positive
- match <0.30 → poor geometric alignment

#### Optimization Results

```bash
# Extract optimization results
grep "slam: optimise=" kevin.log

# Example:
# slam: optimise=12.3ms  rebuild=456.7ms  max_shift=0.087m  keyframes=142  edges=148  loops=3
```

Fields:
- `max_shift`: Maximum pose correction applied (metres)
- `keyframes`: Number of keyframes in graph
- `edges`: Odometry + loop edges
- `loops`: Total loop closures

**Good signs:**
- max_shift <0.10m → small corrections, good tracking
- rebuild time <500ms → fast enough for 30 Hz loop

**Bad signs:**
- max_shift >0.50m → large jumps, possible tracking loss or false loop
- rebuild time >1000ms → may cause frame drops

#### Encoder Fallback Events

```bash
# Detect encoder fallback warnings
grep "encoder:" kevin.log | grep -E "(stale|failed|falling back)"

# Example:
# ⚠️  encoder: data stale (age=0.42s), falling back to commanded vel
```

**Action:** Check CAN bus health, ODrive errors, cable connections.

### Tracking Loss Signatures

Signs of tracking loss in logs:

1. **Sudden visual rejection burst:**
   ```
   # High rate of wheel-only frames
   odom: vl=0.0234 vr=0.0256 ... (repeated with small visual updates)
   ```

2. **Encoder SDO read failures:**
   ```
   encoder: SDO read failed 10 times ... backing off
   ```

3. **Safety immobilization:**
   ```
   vision: TOPDOWN LOST — immobilized (rs1_ok=True known_px=245)
   ```

## Rerun Inspection

Kevin logs to `~/.kevin/rerun/live.rrd` by default. Use Rerun viewer (i777 Orin or workstation):

```bash
rerun ~/.kevin/rerun/live.rrd
```

### What to Look For

#### 1. **Trajectory Consistency**

- View the robot trajectory (world-frame path)
- Look for:
  - Smooth curves during turns
  - Straight lines during forward motion
  - **Bad:** Sudden jumps, discontinuities, or jitter

#### 2. **Map Quality**

- Inspect the occupancy grid overlay
- Look for:
  - Consistent obstacle boundaries
  - Floor areas marked as free
  - **Bad:** Ghost obstacles, noisy edges, missing walls

#### 3. **Keyframe Alignment**

If keyframe poses are logged:
- Verify keyframes align with trajectory
- Check loop closure connections (edges)
- **Bad:** Keyframes far from trajectory after loop closure

## Field Evaluation Checklist

### Pre-Deployment

- [ ] Encoder health check: `wb.get_encoder_health()` → encoder_ok=True
- [ ] Visual accept rate baseline: >60%
- [ ] SLAM memory: <200 MB at startup
- [ ] Top-down depth: >5000 known_px in normal lighting

### During Operation (every 5 minutes)

- [ ] Visual accept rate still >40%
- [ ] Time since visual <3s
- [ ] Encoder age <0.2s during motion
- [ ] No repeated "TOPDOWN LOST" warnings

### Post-Session

- [ ] Review loop closures: count, max_shift values
- [ ] Check keyframe count: <500
- [ ] Export keyframes if >1 loop closure: `slam.export_keyframes()`
- [ ] Inspect Rerun recording for trajectory anomalies

## Common Failure Modes & Diagnosis

### 1. **Pose Jumps**

**Symptoms:**
- Large `max_shift` in optimization (>0.5m)
- Sudden trajectory discontinuities in Rerun
- Keepouts misaligned after event

**Causes:**
- False loop closure (weak desc/match scores)
- Accumulated drift before closure
- Encoder slip during maneuver

**Fix:**
- Tighten `LOOP_SCORE_THRESH` (default 0.40 → 0.50)
- Reduce `LOOP_MIN_DIST` to catch loops earlier
- Check carpet slip compensation (`ANGULAR_SLIP_SCALE`)

### 2. **Map Corruption**

**Symptoms:**
- Ghost obstacles in free space
- Obstacle erosion (walls disappearing)
- Persistent noise in occupancy grid

**Causes:**
- GPU gmap desync after loop closure (now fixed)
- Misaligned ego-to-global projection
- Encoder fallback during map update

**Fix:**
- Verify GPU sync: check "GPU gmap synchronized" messages
- Calibrate camera pitch: `vision.request_calibration()`
- Monitor encoder health during mapping

### 3. **Tracking Loss**

**Symptoms:**
- Visual accept rate drops to <20%
- Wheel-only rate >80%
- Time since visual >10s

**Causes:**
- Poor lighting (visual odometry failing)
- Featureless floor (carpet, uniform tile)
- Camera obscured or dirty

**Fix:**
- Check camera for obstructions
- Increase ambient lighting
- Verify camera intrinsics

### 4. **Session Reset / Keepout Wipe**

**Symptoms:**
- Keepouts disappear after restart
- Pose resets to (0, 0, 0)

**Causes:**
- No persistent map origin (expected behavior)
- SLAM session restart (manual or crash)

**Fix (short-term):**
- Re-mark keepouts after each session
- Call `keepouts.clear_all_disks()` before remarking

**Fix (long-term):**
- Implement persistent world frame with SLAM relocalization
- Store keepouts with visual anchors for re-detection

## Metrics Export

### Export SLAM Keyframes

```python
from slam import PoseGraphSLAM
slam = vision._global_map
slam.export_keyframes('~/.kevin/slam_session.json', include_data=True)
```

Output format:
- Keyframe poses, timestamps
- Edge connectivity (odometry + loops)
- Observation thumbnails (base64 PNG)

Use for:
- Offline trajectory analysis
- Reconstruction quality evaluation
- Rerun replay with keyframe overlay

### Export Pose History

```python
history = vision._pose.get_world_history()
np.save('~/.kevin/pose_history.npy', history)
```

Use for:
- Trajectory smoothness analysis
- Odometry drift estimation
- Comparison with ground truth (if available)

## Recommended Alert Thresholds

Configure alerts (e.g., via logging or remote monitoring) for:

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Visual accept rate | <30% | **Critical** |
| Encoder age | >0.5s during motion | **Warning** |
| Time since visual | >5s | **Warning** |
| Loop closure max_shift | >0.50m | **Warning** |
| SLAM memory | >450 MB | **Info** |
| Topdown lost frames | >10 consecutive | **Critical** |

## References

- `pose.py`: Pose estimation, visual/wheel fusion
- `slam.py`: Pose-graph SLAM, loop closure
- `wheelbase.py`: Encoder health, CAN diagnostics
- `vision.py`: Capture loop, SLAM integration
- `TOPDOWN_NEAR_FIELD_REFLEX.md`: Safety-critical depth monitoring
