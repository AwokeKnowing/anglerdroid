# D435i IMU Integration for AnglerDroid

This document describes the integration of the Intel RealSense D435i IMU (Motion Module) into Kevin/Anglerdroid's custom SLAM and visual odometry pipeline.

## Overview

The D435i IMU provides gyroscope (angular velocity) and accelerometer data at high rates (200-250 Hz) to stabilize pose estimation when visual lock is weak. The integration supports both:

1. **Self-VO mode**: IMU gyro yaw rate fused with wheel odometry + visual odometry in `PoseEstimator`
2. **cuVSLAM mode**: Full stereo-inertial odometry via NVIDIA cuVSLAM's VIO pipeline

## Hardware Requirements

### Verified Configuration (Jetson Orin NX)
- **Camera**: Intel RealSense D435i sn=`944622074292` (RS2 forward camera)
- **Firmware**: Motion Module enabled
- **librealsense**: 2.58.4+ built from source with `-DFORCE_RSUSB_BACKEND=ON`
  - Required because JetPack 6 removed hidraw support
  - See: https://github.com/IntelRealSense/librealsense/blob/master/doc/jetson/backend_configuration.md

### Working IMU Rates
- **Accelerometer**: 250 Hz ✅
- **Gyroscope**: 200 Hz ✅
- **Note**: accel @ 200 Hz fails on some units; use 250 Hz for accel

### Probe Script
Test IMU availability with:
```bash
python3 /home/jetbot/.kevin/probe_d435i_imu.py
```

Expected output:
```
IMU streams available:
  Gyroscope: 200 Hz
  Accelerometer: 250 Hz
```

## Architecture

### Separate Pipeline Requirement

**CRITICAL**: The IMU pipeline MUST be separate from depth/color pipelines.

**Why**: Sharing a single RealSense pipeline for depth/color/IMU can cause frame starvation:
- Depth/color run at 30 Hz (slow)
- IMU runs at 200-250 Hz (fast)
- Shared pipeline can drop IMU frames or depth frames unpredictably

**Implementation**: `src/imu.py` opens a dedicated `rs.pipeline` for Motion Module only.

### Integration Points

#### 1. Self-VO Mode (`--slam self`, default)

**Path**: `IMUPipeline` → `PoseEstimator.update()` → complementary filter

**Flow**:
1. `Vision._capture_loop()` grabs IMU frames (non-blocking) at 200 Hz
2. Transform gyro from camera frame to body frame
3. Extract yaw rate (Z-axis rotation in body frame)
4. Pass `imu_yaw_rate` to `PoseEstimator.update()`
5. Blend wheel + visual + IMU using adaptive weights:
   - High visual confidence (≥0.20): IMU weight = 15% (stabilization)
   - Low visual confidence (<0.20): IMU weight = 50% (fallback)

**Fusion equation**:
```python
dtheta_fused = (1 - w) * dtheta_wheel_visual + w * (imu_yaw_rate * dt)
```

Where `w` adapts based on visual confidence.

#### 2. cuVSLAM Mode (`--slam cuvslam`)

**Path**: `IMUPipeline` → `CuVSLAMTracker.register_imu_measurement()` → cuVSLAM VIO

**Flow**:
1. `CuVSLAMTracker.__init__()` configures `OdometryMode.Inertial` if IMU available
2. Before each stereo IR frame, poll IMU at high rate (~6-7 samples per frame)
3. Register all IMU measurements via `tracker.register_imu_measurement()`
4. cuVSLAM performs tightly-coupled stereo-inertial optimization

**IMU Calibration** (default noise parameters in `cuvslam_tracker.py`):
```python
IMU_GYRO_NOISE_DENSITY = 6.067e-03      # rad/(s*sqrt(Hz))
IMU_GYRO_RANDOM_WALK = 3.621e-05        # rad/(s^2*sqrt(Hz))
IMU_ACCEL_NOISE_DENSITY = 3.362e-02     # m/(s^2*sqrt(Hz))
IMU_ACCEL_RANDOM_WALK = 9.826e-04       # m/(s^3*sqrt(Hz))
```

**Note**: For production use, calibrate per-device using [Kalibr](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model).

### Frame Conventions

**Camera frame** (RealSense):
- X = right
- Y = down
- Z = forward
- (Right-hand coordinates)

**Body frame** (robot):
- X = forward
- Y = left
- Z = up
- (REP-103 standard)

**Transform**: Camera pitched 64.4° down relative to body.
- Implemented in `IMUPipeline.get_angular_velocity_body()`
- Maps camera gyro → body yaw rate for pose integration

## Graceful Degradation

The system works without IMU (plain D435 or Motion Module unavailable):

1. **IMU Pipeline**: If Motion Module missing, `IMUPipeline.ok = False`
2. **Self-VO**: `imu_yaw_rate = 0.0` → pure wheel + visual fusion (existing behavior)
3. **cuVSLAM**: Falls back to `OdometryMode.Multicamera` (stereo-only)

No crashes, no required hardware. IMU is purely additive.

## Usage

### Standard Operation (Existing Scripts)

No changes required. IMU is auto-detected and integrated:

```bash
# Self-VO with IMU (if available)
python3 src/main.py --slam self --rs2 944622074292

# cuVSLAM with IMU (if available)
python3 src/main.py --slam cuvslam --rs2 944622074292
```

If D435i has Motion Module → IMU active.
If plain D435 or IMU unavailable → graceful fallback.

### Verification

Check logs on startup:
```
vision: IMU pipeline active (gyro 200 Hz, accel 250 Hz)
vision: cuVSLAM backend active
cuvslam: stereo-inertial mode enabled
```

Or (if unavailable):
```
imu: Motion Module not available on 944622074292: <error>
     → graceful degradation: IMU fusion disabled
cuvslam: stereo-only mode (no IMU)
```

## Testing

### Unit Tests

Run without hardware (mocked IMU):
```bash
python3 test_imu_integration.py
```

Tests cover:
- IMU yaw fusion improves accuracy when visual weak
- Adaptive IMU weight based on visual confidence
- Graceful degradation without IMU
- Zero IMU measurements (no-op)
- Frame transformations

### Live Hardware Tests

Probe IMU:
```bash
python3 /home/jetbot/.kevin/probe_d435i_imu.py
```

Run vision loop with debug logging:
```bash
# Terminal 1: Monitor logs
tail -f /tmp/kevin.log

# Terminal 2: Run main loop
python3 src/main.py --slam self --rs2 944622074292
```

Expected behavior:
- IMU grabs succeed at ~200 Hz (non-blocking polls)
- Pose updates include IMU yaw rate
- When spinning in place (visual weak), IMU stabilizes yaw estimate

## Performance Impact

### CPU / GPU
- **IMU pipeline**: ~0.1-0.2 ms per grab (negligible)
- **Pose fusion**: <0.05 ms additional compute
- **cuVSLAM VIO**: ~1-2 ms extra per frame (GPU-accelerated)

### Memory
- IMU pipeline: ~1 MB (separate pipeline + buffers)

### Latency
- IMU samples are ~5 ms fresher than camera frames (200 Hz vs 30 Hz)
- Reduces yaw drift during fast rotations

## Known Limitations

1. **JetPack 6 Requirement**: Must build librealsense with `FORCE_RSUSB_BACKEND=ON`
2. **Accel @ 200 Hz fails**: Use 250 Hz for accelerometer (driver limitation)
3. **IMU calibration**: Default noise parameters are approximate; device-specific calibration recommended for production
4. **Gravity alignment**: cuVSLAM VIO requires accurate IMU extrinsics; misaligned gravity vector indicates bad extrinsics

## Files Modified

### New Files
- `src/imu.py` — IMU pipeline (separate from depth/color)
- `test_imu_integration.py` — Unit tests with mocked IMU
- `IMU_INTEGRATION.md` — This document

### Modified Files
- `src/pose.py` — Added `imu_yaw_rate` parameter + complementary filter
- `src/vision.py` — Initialize IMU pipeline, grab frames, pass to pose
- `src/cuvslam_tracker.py` — Stereo-inertial mode + IMU registration

## References

- [RealSense D435i Datasheet](https://www.intelrealsense.com/depth-camera-d435i/)
- [librealsense IMU Streams](https://github.com/IntelRealSense/librealsense/blob/master/doc/motion_data.md)
- [cuVSLAM Stereo-Inertial Example](https://github.com/NVlabs/PyCuVSLAM/blob/main/examples/realsense/README.md)
- [Kalibr IMU Calibration](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)

## Troubleshooting

### "Motion Module not available"
- Check firmware: `rs-enumerate-devices | grep -A5 "Motion Module"`
- Verify librealsense built with `FORCE_RSUSB_BACKEND=ON` on JetPack 6
- Try probe script: `/home/jetbot/.kevin/probe_d435i_imu.py`

### IMU grabs return False
- IMU pipeline shared with depth/color? → Use separate pipeline
- Check `rs-sensor-control` for IMU stream health

### Gravity misaligned in cuVSLAM
- Check IMU extrinsics in rig setup
- Verify camera pitch angle (default 64.4°)

### Poor VIO performance
- Validate stereo-only mode works first
- Check IMU noise parameters match hardware
- Ensure IMU frequency matches stream (200 Hz gyro)

## Future Work

- Per-device IMU calibration (Kalibr)
- Accelerometer integration (linear motion during visual loss)
- Complementary filter tuning (adaptive weights based on motion model)
- Rerun logging for IMU data visualization
