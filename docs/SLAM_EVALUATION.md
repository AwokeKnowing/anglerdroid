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

## 2026-10-19 — L4 offline candidates (sim-off overnight queue) + nav/SLAM decoupling status
L4 = 80 s sim log (synced RS2 framesets, fixed odom stamps). ATE anchored at the first pose (no fit), plus SE(2)-aligned ATE; scored with
`emu.tools.slamlog.ext_eval` / `baseline.py`. Results in `~/slamlogs/res/`.

| method | n | ATE rmse m | ATE max m | head max ° | aligned m | notes |
|---|---|---|---|---|---|---|
| kslam (live SLAM2) | 1 | 0.113 | 0.246 | 4.6 | 0.086 | full run |
| odometry (EKF) | 1 | 0.130 | 0.199 | 3.5 | 0.044 | full run |
| lgfg fused (DISK+LightGlue VO + iSAM2, s/tau states) | 1 | 0.115 | – | 5.4 | 0.038 | |
| DROID-SLAM RGB-D | 3 | 2.47–2.51 | 7.8–7.9 | 169–174 | 0.073–0.076 | heading flip breaks anchoring; wall p90 54–66 cm |
| MASt3R-SLAM | 1 | 0.066 | 0.068 | 0.85 | 0.012 | **coverage 0.53** (52 keyframe poses, 77 relocalizations), scale 0.866, wall p90 2.0 cm |

MASt3R-SLAM is accurate where it tracks but covers only half of the run, n=1 (the run hung at exit after saving; queue stopped
before r2/r3). Not comparable to the full-run numbers until coverage is fixed. Nothing changed in the live default.

Decoupling (`KEVIN_NAV_DECOUPLE=1`, default off; commits 941601a, 77d9904): driving/planner/envelope/LightNav in the odom frame on a
30 s local grid; SLAM map used for named places only when a live-scan vs map agreement check passes. Goal bench n=2: flag off 6/6
goals arrive; flag on 5/6 + 1 refusal (check fixed after). Injected 1 m/30° error detected (agreement 0.23) but a goal sent ~6 s
after start was accepted (single-scan check aliases at startup). SLAM-off LightNav and 30 s lag tests not completed. Not default.

## 2026-10-19 05:40-06:07 — goal interface step 1 (docs/ARCHITECTURE_LOOPS.md)
`KEVIN_GOAL_SHM=1` (default off): LightNav runs as its own process (`nav/lightnav_proc.py`, fixed 250 ms loop) and writes an
odometry-frame goal block that nav consumes (`nav/goalio.py`). Measured: loop lateness max 4 ms / p99 3.4 ms, 4.00 fps capture,
predictions 220-480 ms, 0 errors.

| goal bench (kitchen/bedroom/couch, final GT m) + LightNav 'find the kitchen' 90 s | goals arrived | LightNav final GT m |
|---|---|---|
| flag on, gshm1-3 | 3/3, 0/3, 3/3 | 1.70, 1.19, 1.56 |
| flag off, base19_1-2 (same morning) | 3/3, 3/3 | 1.84, 1.01 |

gshm2 failed before any LightNav use: robot boxed at the start ('local replan failed', axle clearance 0.06 m) — the place-goal
path does not touch the new code, but cause not identified, so the default stays OFF until a clean n>=3.
MASt3R-SLAM with odometry prior + exit fix prepared (`emu/tools/slamlog/odom_prior.py`, `~/slamlogs/res/queue2.sh`), not run:
it needs ~17 GB GPU and LightNav holds 12.3 GB of 24.5 GB -> sim-off window.

## 2026-10-19 07:41-08:22 PT — MASt3R-SLAM sim-off window (queue2: n=3, with/without wheel-odom prior, 1200 s cap)
Exit-hang fix worked (every run exited, wall 139-184 s on L4, 337-345 s on L3) but left orphaned multiprocessing workers
(killed afterwards; next fix: kill the process group). Real-depth injection not implemented -> not used.

| log | method | n | ATE m | head max ° | aligned m | coverage | wall p90 cm | scale |
|---|---|---|---|---|---|---|---|---|
| L4 80 s | kslam live | 1 | 0.113 | 4.6 | 0.086 | 1.0 | 7.3 | – |
| L4 | odometry | 1 | 0.130 | 3.5 | 0.044 | 1.0 | 12.7 | – |
| L4 | lgfg fused | 1 | 0.115 | 5.4 | 0.038 | 1.0 | 13.9 | – |
| L4 | MASt3R, no prior (r1-r3) | 3 | 0.061-0.066 | 0.85-0.98 | 0.011-0.012 | 0.50-0.53 | 2.0-2.1 | 0.87-0.88 |
| L4 | MASt3R + odom prior (o1-o3) | 3 | 0.056-0.060 | 0.94-1.01 | 0.021-0.046 | **0.05** | 1.8 | 0.85-0.88 |
| L3 150 s | kslam live | 1 | 0.182 | 12.4 | 0.112 | 1.0 | 11.0 | – |
| L3 | odometry | 1 | 0.125 | 10.5 | 0.119 | 1.0 | 8.0 | – |
| L3 | lgfg fused | 1 | 0.194 | 12.1 | 0.043 | 1.0 | 14.4 | – |
| L3 | MASt3R, no prior (r1-r3) | 3 | 1.12 | 2.8 | 0.86 | **0.053** | 33.8 | 0.37 |
| L3 | MASt3R + odom prior (o1-o3) | 3 | 1.12 | 2.8 | 0.86 | **0.053** | 33.8 | 0.37 |

MASt3R ATE/heading are only over the poses it produced (coverage = fraction of RS2 frames with a pose). It is very accurate where
it tracks (L4 heading < 1°, walls 2 cm) but loses tracking at the first fast spin and mostly fails to relocalize
(L3: 14 keyframes, ~290 reloc attempts, identical with/without prior). The odometry initial-pose prior made L4 worse (coverage
0.5 -> 0.05, n=3); prior convention checked against MASt3R's own relative rotations (agree to ~0.01 rad) so it is not a frame bug.
Not a replacement for kslam as is. Next: real depth (metric scale) and feeding the prior into RELOC/keyframe selection, not just init.
