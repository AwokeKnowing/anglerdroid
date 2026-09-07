# Capture Loop Comprehensive Profiling

## Status

**Current**: Vision capture running at ~11 Hz (89ms/frame) on Jetson Orin NX (vision-only, wheelbase=None)  
**Target**: 30 Hz sustained (33ms/frame)  
**Gap**: Need to eliminate ~56ms per frame

## Profiling Output

After cherry-picking PR#19 fec0107 (capture FrameBudget shedding RS2/gmap) onto main as 55826bd:
- Silent vision-only smoke shows odom dt≈0.089s (~11 Hz)
- read_hz≈24 Hz (main loop is faster, capture is the bottleneck)
- No "capture TOTAL shed" line appeared (RS2/gmap shedding NOT triggered)
- Soft-low still works (fwd=0.3)

## Instrumentation Added

This PR adds comprehensive per-stage timing to `_capture_loop` that prints every 90 frames:

```
================================================================================
CAPTURE TIMING (last 90 frames): TOTAL=89.0ms (11.2 Hz)
================================================================================
STAGE             MEAN     P95  %TOTAL
--------------------------------------------------------------------------------
grab              X.Xms  X.Xms    X.X%
pose+hazard       X.Xms  X.Xms    X.X%
rs1_checks        X.Xms  X.Xms    X.X%   (near-field + overhang + soft-low + depth_topdown)
rs2_gpu           X.Xms  X.Xms    X.X%   (scatter + 4x morph + readback)
obs_comb          X.Xms  X.Xms    X.X%   (blit + mask operations)
odom              X.Xms  X.Xms    X.X%   (GPU visual odometry, ~6-8ms expected)
gmap              X.Xms  X.Xms    X.X%   (global map update + projection)
safety            X.Xms  X.Xms    X.X%   (safety update + SLAM lock check)
render            X.Xms  X.Xms    X.X%   (GPU atlas render: 3D + cameras + minimap)
================================================================================
Target: 33.3ms/frame (30 Hz). Current: 89.0ms (11.2 Hz)
================================================================================
```

## Expected Breakdown (Hypothesis)

Based on code analysis, expected time distribution:

1. **render** (~25-35ms, ~30-40%): Full 3D voxel terrain + cameras + minimap
   - Most expensive single stage
   - Already GPU-accelerated
   - Could reduce detail or shed under budget

2. **rs2_gpu** (~10-15ms, ~12-18%): 4 morph passes + readback
   - Scatter to texture
   - Dilate × 2, Erode × 2 (morphological closing)
   - GPU→CPU readback
   - Could reduce to 3 passes (dilate×2, erode×1)

3. **gmap** (~8-12ms, ~10-14%): Global map evidence update
   - Per-pixel evidence integration
   - Already GPU-accelerated
   - Could reduce update rate or batch

4. **odom** (~6-8ms, ~8-10%): GPU visual odometry
   - SAD-based yaw estimation
   - Already GPU-accelerated
   - Documented as ~6-8ms

5. **rs1_checks** (~5-10ms, ~6-12%): Multiple passes over RS1 verts
   - check_topdown_near_field: Full vert scan
   - check_topdown_overhang_approach: Full vert scan + projection
   - check_topdown_soft_low_obstacle: Full vert scan + projection + height calc
   - depth_topdown: Orthographic projection
   - **OPTIMIZATION**: Combine into single pass (check_topdown_all_in_one)

6. **grab** (~3-8ms, ~4-10%): Parallel camera grabs
   - Already parallelized with ThreadPoolExecutor
   - Limited by RS SDK frame timing

7. **obs_comb** (~2-4ms, ~2-5%): Observation combining
   - Multiple blit operations
   - Mask operations
   - Could optimize buffer reuse

8. **pose+hazard** (~2-4ms, ~2-5%): Pose + RGB hazard check
   - cuVSLAM or wheel+visual
   - Checkered mat detection

9. **safety** (~1-2ms, ~1-2%): Safety update
   - SLAM lock status check
   - Safety scale updates
   - Immobilize logic

## Action Plan

1. **Run on hardware** to get ACTUAL timing breakdown (not hypothesis)
2. **Identify top 2-3 stages** consuming most time
3. **Optimize based on data**:
   - If **render** dominates (likely): Reduce detail, shed under budget, or optimize shaders
   - If **rs2_gpu** high: Reduce morph iterations (4→3 passes)
   - If **gmap** high: Reduce update rate or batch multiple frames
   - If **rs1_checks** high: Apply combined check optimization (already implemented)
   - If **grab** high: Check RS SDK decimation/resolution settings

## Safety Constraints

**Must keep at full rate (never shed)**:
- RS1 topdown depth (ground truth for immobilize)
- Safety checks (near-field, overhang, soft-low, hazard)
- Safety update

**Can shed under budget** (already implemented):
- RS2 forward depth processing
- Global map updates

**Could shed if needed**:
- Render detail level
- Visual odometry (if wheelbase=None)

## Next Steps

1. Deploy this PR to Kevin Jetson Orin NX
2. Run vision-only smoke test (wheelbase=None)
3. Collect timing output every 90 frames
4. Identify actual bottlenecks from data
5. Implement targeted optimizations based on findings
6. Iterate until 30 Hz sustained (p95 < 33ms)
