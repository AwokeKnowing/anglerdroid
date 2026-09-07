# Experiment: self-slam-wheel-imu-prior-v0

**Status**: Hypothesis test (PR open, not merged)  
**Date**: 2026-09-07  
**Branch**: `cursor/self-slam-wheel-imu-prior-v0-9413`

## Problem

`--slam self` visual odometry fails to lock reliably on Kevin (Jetson Orin NX):
- Live smoke shows "SLAM NOT LOCKED init_kf_0/2" especially with `--no-wheelbase`
- Carpet ↔ hard floor slip makes encoder/commanded-vel odom unreliable
- PyCuVSLAM is broken on this JP6 stack (SIGILL / CUDA mismatch)

Visual odometry (GPU ORB tracking) struggles with:
- Low feature environments (black frames, uniform carpet)
- Rapid lighting changes
- Motion blur during fast turns

Current fallback (wheel-only) is insufficient due to slip.

## Hypothesis

Fusing **wheel + IMU outside of vision** as an EKF prediction prior will:
1. Provide robust pose estimate even when visual tracking is lost
2. Allow SLAM to lock reliably by bootstrapping from wheel+IMU prior
3. Apply visual correction only when feature tracking is healthy (gated)
4. Boot from saved pose (`~/.kevin/latest_pose.json`) for crash recovery

## Implementation

### Core module: `src/wheel_imu_prior.py`

Simple 2D EKF tracking pose (x, y, theta) with covariance:
- **Prediction**: Integrates wheel velocities + IMU yaw rate (50/50 blend)
- **Correction**: Applies visual odometry when healthy (Mahalanobis gated)
- **Persistence**: Boots from and saves to `~/.kevin/latest_pose.json`

### Integration: `src/vision.py`

When `--wheel-imu-prior` flag is enabled:
- Replaces `PoseEstimator` with `WheelIMUPrior` (via adapter wrapper)
- Loads saved pose on boot
- Saves pose every 10 seconds
- Visual correction gated on `vis_confidence > 0.1`

### CLI flag

```bash
python src/main.py --wheel-imu-prior --slam self
```

**Default**: OFF (no behavior change for live system)

## A/B Testing on Kevin (Orin)

### Baseline (control)

```bash
# SSH into Kevin
ssh kevin@orin.local

# Run without flag (current behavior)
cd ~/anglerdroid
python src/main.py --slam self --rs1 <serial> --rs2 <serial>

# Monitor SLAM lock status in logs:
# Look for: "🟢 SLAM LOCKED" vs "🔴 SLAM LOCK LOST"
# Watch for: "init_kf_0/2" (not enough keyframes)
```

**Metrics to collect**:
- Time to SLAM lock (seconds from start)
- SLAM lock loss count (per 5-10 min session)
- Keyframe count after 5 min
- Visual accept rate (`get_tracking_quality()`)
- Stuck detection false negatives (robot stuck but not detected)

### Treatment (experiment)

```bash
# Run WITH flag (experiment)
python src/main.py --wheel-imu-prior --slam self --rs1 <serial> --rs2 <serial>

# On startup, check for:
# "vision: EXPERIMENT wheel+IMU prior enabled (self-slam-wheel-imu-prior-v0)"
# "wheel_imu_prior: loaded pose from /home/kevin/.kevin/latest_pose.json"
```

**Metrics to collect** (same as baseline):
- Time to SLAM lock
- SLAM lock loss count
- Keyframe count
- Visual accept rate
- Stuck detection false negatives

**Additional metrics** (experiment-specific):
```python
# In Python REPL or log scraping:
metrics = vision._wheel_imu_prior.get_metrics()
print(metrics)
# Expected keys:
#   'x', 'y', 'theta', 'theta_deg',
#   'cov_x', 'cov_y', 'cov_theta',  # Covariance (uncertainty)
#   'predict_count', 'correct_count',  # Prediction vs correction ratio
#   'boot_source'  # 'json' if loaded from file, None otherwise
```

### Test scenarios

1. **Room loop (5-10 min)**
   - Drive Kevin around a room loop (return to start)
   - Measure drift: distance between start and end pose
   - Baseline: expect 20-50cm drift after 10m loop
   - Treatment: expect <20cm drift (with IMU correction)

2. **Carpet ↔ floor transition**
   - Drive from carpet to hard floor and back
   - Monitor visual accept rate (should drop on uniform carpet)
   - Check if SLAM stays locked during carpet phase
   - Baseline: expect SLAM lock loss or high drift
   - Treatment: expect SLAM to stay locked (wheel+IMU prior)

3. **Lost vision (black hallway, low light)**
   - Drive through low-feature area (dark hallway, uniform wall)
   - Monitor visual accept rate (should drop)
   - Check if pose integration continues
   - Baseline: expect SLAM lock loss
   - Treatment: expect pose to continue from wheel+IMU

4. **Crash recovery**
   - Run Kevin for 5 min, note final pose
   - Kill process (`Ctrl+C`)
   - Restart with `--wheel-imu-prior`
   - Check if pose boots from saved JSON
   - Expected log: "wheel_imu_prior: loaded pose from ..."

### Metrics comparison table

| Metric | Baseline | Treatment | Target Improvement |
|--------|----------|-----------|-------------------|
| Time to SLAM lock | 10-30s | <10s | 50% faster |
| SLAM lock loss/10min | 3-5 | <2 | 50% fewer |
| Keyframes @ 5min | 50-100 | >100 | More stable mapping |
| Visual accept rate | 60-70% | 60-70% | Same (no regression) |
| Room loop drift (10m) | 20-50cm | <20cm | 50% less drift |
| Stuck false negatives | 1-2 | 0 | Eliminated (future) |

## Code changes

**Files added**:
- `src/wheel_imu_prior.py` — core EKF module
- `test_wheel_imu_prior.py` — unit tests
- `docs/experiments/self-slam-wheel-imu-prior-v0.md` — this doc

**Files modified**:
- `src/main.py` — add `--wheel-imu-prior` flag
- `src/vision.py` — wire prior into Vision init, create PriorAdapter wrapper

**No changes to**:
- 30 Hz capture path (prior runs on odom_thread @ 100 Hz)
- GPU rendering (`gpu_render.py`)
- Safety reflexes (`safety.py`)
- SLAM backend (`slam.py` — keyframe logic unchanged)

## Constraints followed

- ✅ Drive stays disarmed (no motor commands in tests)
- ✅ Map-frame keepouts disabled until SLAM locks
- ✅ GPU VO not deleted (prior is complementary, not replacement)
- ✅ Follows existing code style (matches `pose.py`, `imu.py` patterns)
- ✅ Default OFF (no live behavior change without flag)
- ✅ Minimal diff (new module + small integration points)

## Next steps (post A/B)

If experiment shows improvement:
1. **Shadow mode**: Enable by default but log-only (no pose changes)
2. **Gradual rollout**: Enable for autonomous wander only, then full
3. **Tune parameters**: Adjust Q/R noise, IMU blend weight based on data
4. **Stuck detection**: Port from `pose.py` to `wheel_imu_prior.py`
5. **History tracking**: Add trajectory buffer for minimap/rerun

If experiment fails:
1. **Root cause**: Analyze why (IMU noise? EKF tuning? Visual gating too loose?)
2. **Iterate**: Adjust noise parameters, blend weights, gating thresholds
3. **Alternative**: Consider PyCuVSLAM fix or external SLAM (ORB-SLAM3)

## References

- `AGENTS.md` — Orin microsecond vigilance, 30 Hz budget
- `src/pose.py` — baseline wheel+visual fusion
- `src/imu.py` — D435i Motion Module IMU pipeline
- `src/odom_thread.py` — high-rate odometry thread (~100-200 Hz)
- `src/slam.py` — PoseGraphSLAM (keyframe + loop closure)

## Test execution

```bash
# Unit tests (no hardware required)
cd /workspace
PYTHONPATH=/workspace/src:$PYTHONPATH python3 test_wheel_imu_prior.py

# Expected: 9 passed, 0 failed
```

## Contact

- **Owner**: James (via anglerdroid repo)
- **Experiment PR**: `cursor/self-slam-wheel-imu-prior-v0-9413`
- **Slack**: #kevin-robot (for A/B results)
