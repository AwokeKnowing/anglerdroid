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

## 2026-10-19 09:18 — sim safety-core frame-age spikes (sim artefact, not the core)
Spikes >100 ms came from emu.world's 1 kHz loop writing camera poses 30-236 ms late under CPU contention on cpus 0,1,4,5 (plus
desktop apps on the depth/core cpus). simpin now keeps Chrome/Grok Bot app off cpus 2,3,6,7 and lowers Chrome/viewer priority:
2 nav restarts x 130 s, 0 frames >100 ms (max 88 / 93 ms) vs 3 >100 ms per 200 s before. Safety core unchanged.

## 2026-10-19 09:18-09:52 PT — goal-shm bench rerun (paired with flag-off), gshm2 trap analysis, MASt3R prep
**gshm2 "boxed-in" (0/3 goals) is not caused by the goal-shm flag. It is a nav failure that already existed, and the root cause is not proven.**
- Kevin was not boxed in at the start. On "go to the kitchen" he first went north, then turned back and looped east
  (GT (3.67,2.24)->(3.27,3.73)->(5.25,2.48)->(5.81,3.4)->(5.38,4.79)). He stopped at (5.17,4.94) at t~19 s, 1.29 m from the kitchen.
  The log then shows "local replan failed" x8, then "unreachable ... axle clearance 0.06 m". Bedroom and couch then failed instantly from that spot.
- The same "unreachable: local planner found no path" result appears in 6 earlier flag-off benches (benchB1, postA2, postA3, postC2,
  r3_cur2, r3_ta1). Place goals do not go through the goal-shm code path.
- Hypothesis, unverified: at the start of that leg the dog was at (2.10,6.51), near the kitchen route. It may have blocked the direct
  route, causing the reroute, and/or left stale occupied cells that closed the corridor. No map snapshot or dog trace exists for gshm2.
  Instrumentation added (3bdc848): the GT trace now logs dog x,y, and gb3 runs use BENCH_MAPSNAP=1, so the next occurrence will show it.
- Separate pre-existing pattern: legs where nav's "arrived in X s" is much shorter than the bench wall time (goal clock restarted
  mid-leg, long detour). Found in 37 of 161 arrived legs across all benches, flag-off included (e.g. r3_ta2, fix61, dcon1).
**Paired bench, both arms with nav launched at nice 0 through nice0.sh (BENCH_RESTART, /tmp/gb3.sh), n=3 each, same session:**
| arm | goals | collisions | place-leg wall s (run1 / run2 / run3) | LightNav "find the kitchen" final GT m |
|---|---|---|---|---|
| KEVIN_GOAL_SHM=1 (gs21-23) | 9/9 | 0 | 8.6,20.5,36.1 / 8.5,15.7,65.1 / 8.5,58.9,44.6 | 0.87, 1.58, 0.34 |
| flag off (base20_1-3) | 9/9 | 1 (bedroom run 3) | 15.6,28.3,26.6 / 8.6,38.1,57.2 / 8.7,49.3,28.3 | 2.52, 0.52, 1.99 |
All runs including earlier ones: flag-on 18/21 goals (the 3 misses are gshm2), LightNav mean 1.21 m (n=6). Flag-off 15/15,
LightNav mean 1.58 m (n=5). These are within noise of each other.
**Default NOT flipped.** Reasons: (a) the viewer chat -> LightNav path was not tested with the flag on (the bench uses nav
/api/command directly); (b) the session end state requires KEVIN_GOAL_SHM unset = off. James should decide. To flip it, change one line, nav/run.py:77.
**MASt3R offline prep, done; NOT run (needs a sim-off window):**
- Files: ~/slamcand/MASt3R-SLAM/main.py (backup main.py.pre_prep2; full diff ~/slamlogs/res/m3s_main_kevin_prep2.patch).
  New env flags, all default off:
  - M3S_ODOM prior is now rotation-only unless M3S_DEPTH is set (the map scale is arbitrary). With M3S_DEPTH, translation is converted to prev-camera
    units (t/s_prev). This is a weak, untested hypothesis for the L4 prior regression (L4 Sim3 scale was 0.87).
  - M3S_ODOM_COAST=1: when tracking fails but the odom rotation since the last keyframe is < HFOV, stay in TRACKING on the odom pose
    instead of switching to RELOC.
  - M3S_ODOM_KF=1: create a keyframe when 1 - odom_angle/HFOV < match_frac_thresh. This is the same threshold MASt3R already uses; no new weight.
  - M3S_DEPTH=<dir>: real depth (uint16 mm, resized to the frame image). It sets the INIT keyframe's Sim3 scale to median(real z / pointmap z)
    and logs KFSCALE per keyframe.
  - Orphan kill: before os._exit, psutil terminates, then kills, all child processes.
- Finding, measured offline from the odom priors: in L3 (16 frames) and L4 (18 frames), consecutive RS2 frames are rotated 20-56 deg apart.
  Each of these coincides with a camera-log gap of 0.13-0.6 s during spins. The KF-overlap bound is 36 deg (1-0.333 x 54.2 deg HFOV),
  so image-only tracking cannot bridge them. This is consistent with L3 losing tracking for good, and motivates COAST.
- Next run: ~/slamlogs/res/queue3.sh. It is SIM-OFF ONLY and refuses to start if emu.world or LightNav is running.
  Variants oc / dc / o / r, L3+L4, n=2, 1200 s cap per run, orphan pkill, ext_eval tum@30.
**End state 09:52 PT:**
- Live defaults: flag-off, NOLOGS=1. Nav pid 224290 at nice 0, started 09:44. World and depthcam at nice 0.
- "stop" sent; Kevin idle (v 0, w 0).
- Safety core frame age (ageprobe, 90 s, n=688): p50 48 / p90 54 / max 77 ms, 0% over 100 ms.
- Viewer 200; cloudflared pid 10874 ready (1 connection). viewer_mode.json unchanged.
- Nothing sent to the Orin; safety untouched.

## 2026-10-19 14:09-14:30 PT — LightNav + MASt3R together with the sim up (James t583): NOT achieved under the 100 ms abort rule
**Found first: the safety core process had been DEAD since 12:28 PT** (nav pid 224290, started 09:44).
- Traceback: `nav/bus.py read: TimeoutError("bus kevin_sc_rs2: writer stuck")`, raised from nav/safety_core.py:296 `B["rs2"].read()`.
  The seqlock reader gives up after 200 tries and raises; core_main does not catch it, so the core process exits. The camera
  process (writer, cpu 7) was still alive.
- Nav kept running and reported core alive=false, but nothing acts on that: no fail-stop, no alert, no respawn. Kevin was idle.
  The wheels belong to the core, so the sim wheels just got no commands. The core was dead ~1 h 45 min.
- Copy of the log: ~/m1_backup/dbg/live_nav_core_crash_1228.log. Restored by a nav restart via nice0.sh at 14:13; core alive, code idle.
- I don't know what load at 12:28 caused it; nothing was running from me then.
- PROPOSALS for James (safety core, NOT applied):
  (a) In the core, treat a bus read timeout as "frame stale -> brake/hold" and keep looping, instead of letting the exception kill the process.
  (b) In nav, when core_proc dies: stop all goals, raise a visible alarm, and only then (optionally) respawn the core.
  (c) The same uncaught-raise path would exist on the bot.
**Watchdog tool:** emu/tools/agewatch.py.
- Same /api/state source as /tmp/ageprobe.py, but runs continuously: one summary line per 10 s and an ALERT line for any max(age1, age2) > 100 ms.
- On an ALERT it SIGTERMs, then SIGKILLs, the process groups listed in /tmp/heavy.pgids.
- Sampling: /api/state takes ~60 ms, so the watchdog sees ~25% of core decisions (the "gaps" column counts the misses).
- At 0.03 s polling (40-80% of decisions seen) alerts were more frequent. The extra nav load may have caused some of them, so it now polls at 0.09 s, the same as ageprobe.
- Log: ~/m1_backup/dbg/agewatch_1019pm.log.
**LightNav low-memory restart (step 2):**
- The 5-6 GB target is not reachable: the bf16 weights alone are 9.1 GB.
- Note: `gpu_memory_utilization` is effectively not the limit, because LightNav sets the KV cache size directly (VLN_KV_CACHE_GIB, default 2 GiB).
- Measured per-process GPU memory (nvidia-smi):
  | LightNav config | GPU memory | request time |
  |---|---|---|
  | default: bf16, CUDA graphs, KV 2 GiB, 0.4 | 12.5-13.4 GB | 0.58-0.64 s |
  | fp8_llm_only + eager + KV 0.5 GiB + 0.25 | 8.27 GB | 0.83-1.07 s |
- The fp8 setup answered "ask find the kitchen" (Kevin moved; waypoints came back), but its nav quality was NOT benched.
- Other GPU users: nav 0.45-0.47 GB; world/desktop graphics ~2.5 GB (3.0 GB total with LightNav off).
- After the test I restored LightNav to the default command (live default, quality-benched): pid 751263, 12.5 GB; a request was verified at 14:24.
**MASt3R capped (step 3): 3 attempts, all aborted by agewatch 10-21 s after launch, during model load (before any frame was tracked):**
- Setup: L3 oc / dc, then oc again.
  - All attempts: nice 19, torch memory fraction 0.22 (5.2 GiB per CUDA process), PYTORCH expandable segments.
  - Attempts 1-2 on cpus 0,1,4,5. Attempt 3 at SCHED_IDLE on a single logical cpu (5), 1 thread, 30 ms/frame throttle.
- No MASt3R GPU memory was ever recorded: it was killed before it registered on the GPU.
- Spikes > 100 ms also happen WITHOUT MASt3R, every time a big model loads or exits:
  LightNav load 14:14 (5 alerts, max 124 ms), LightNav kill+reload 14:22-14:23 (3), LightNav first request 14:24 (112 ms),
  and two near-idle ones (14:15:26 at 108 ms, 14:20:27 at 101 ms, the latter 10 s after a MASt3R kill).
  Steady idle with the 0.09 s probe: 14:18:15-14:20:05, n=600, max 91 ms, 0 alerts.
- The SCHED_IDLE single-cpu attempt still triggered a spike, so CPU share is probably not the mechanism. Hypothesis (untested):
  large GPU allocations and host-to-device weight transfers stall the world/depthcam renders on the same 3090.
- No ATE table: nothing produced poses. Earlier sim-off numbers (queue2) stand.
**Recommendation for full coverage on L3 (fast spins):**
- In L3/L4 the odom priors show 16/18 frame pairs that are 20-56 deg apart, all at camera-log gaps of 0.13-0.6 s during spins. The keyframe-overlap
  limit is 36 deg, so no image-only tracker (MASt3R, kslam visual) can bridge them.
- Full coverage must come from wheel+gyro odometry as the always-on, fixed-rate backbone. Today the fused lgfg/odom estimate already has
  coverage 1.0 on L3, ATE 0.125-0.194.
- A learned SLAM (MASt3R) should be an async correction sensor: map->odom transform plus confidence, used as loop closure / relocalisation,
  exactly the slot in ARCHITECTURE_LOOPS. Run it with the odom coast/keyframe hooks (queue3 "oc"/"dc").
- Fix the RS2 log/frame gaps during spins at the source.
- MASt3R on this box only fits sim-off, or with a second GPU (or the Orin later, read-only now).
- Not yet measured: whether "oc" bridges the L3 spins. That needs one sim-off run: `VARS="oc dc" N=1 bash ~/slamlogs/res/queue3.sh`, ~15 min.
**Tooling committed:** emu/tools/agewatch.py; emu/tools/slamlog/m3s/{queue3.sh, queue3c.sh, m3s_main_kevin_prep2.patch}.
- main.py additions: M3S_MEM_FRAC, M3S_FRAME_SLEEP_MS, and a peak-memory print.
- queue3c: nice 19 + SCHED_IDLE, CPUS/NT options, early kill on reloc storms, per-pid GPU log, kill-file registration.

## 2026-10-19 evening: 30 Hz synced sim stream, L8 log, light methods (MASt3R dropped per James t596)

Sim sensor stream (emu/world.py, commits cda48c9, 2a73f7c):
- Colour was 12.4-14 Hz: the colour/IR/mast/viewer "slow lane" ran as a thread in the world process.  Slow-lane ms/s at
  14 Hz: IR pairs 450, viewer 286, colour 141, mast 105.
- Fixes: IR stereo pairs opt-in (KEVIN_EMU_IR=1; only KEVIN_SLAM=cuv and scripts/bench stereo read them), third-person
  viewer in its own thread at 10 Hz, and the whole lane in its own forked process (KEVIN_EMU_SLOWPROC=1) so it no longer
  shares the GIL with the 1 kHz physics loop that publishes depth poses.  Colour 28.8-30.0 Hz.
- sensorlog keeps frames in RAM and encodes after the recording at nice 19 / 1 thread (encoding 60 img/s took ~1 core).
- Safety frame age (agewatch, nav driving + 30 Hz sensorlog + encoding): max 99 ms, 0 > 100 ms.  Before: max 169 ms
  while recording, 323 ms while encoding.
- Tyre contact (7d65cd1): drive tyres collide as ellipsoids at the tread centreline (cylinder rim-edge contact hopped
  +-6 mm -> yaw ripple + up to 1.5 % bias).  drive 0 pi/2: GT within 0.04 % of the wheel setpoint.

Logs (RS2 frames): L3 8.9 Hz, gap max 0.50 s / 52 deg; L6 12.4 Hz, 0.167 s / 13.4 deg; L7 22.5 Hz, 0.134 s;
L8 (70 s, spin-heavy, nav recording on) 28.8 Hz, gap median 0.033 p99 0.067 max 0.100 s, max 7.3 deg/frame.

L8 results (ext metrics; coverage = fraction of frames with a pose):
| method | ATE rmse | ATE max | worst heading | aligned | runtime | GPU |
|---|---|---|---|---|---|---|
| wheel+gyro odom (live) | 0.217 m | 0.506 m | 7.41 deg | 0.083 m | live | 0 |
| kslam (live, odom+gyro + visual) | 0.088 m | 0.174 m | 3.41 deg | 0.053 m | live (nav) | shared |
| o3d_rgbd RS2 (Open3D hybrid RGB-D VO, odom-prior init, no gating) | diverged (316 km) | - | 180 deg | - | 326 ms/frame CPU (0.1x RT) | 0 |
| o3d_rgbd RS1 | diverged (1.3 km) | - | 180 deg | - | 327 ms/frame | 0 |
o3d_rgbd: Open3D reports success on 2015/2016 steps but drifts upward from frame ~100 and blows up by frame 600; a
step must be gated against the wheel+gyro prior (and projected to planar motion) before it is usable.  Not committed yet.
MASt3R dropped (James t596).
