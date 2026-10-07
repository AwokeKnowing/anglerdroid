# ModernGL Ego Scatter Fix - Summary

## Problem Statement
ModernGL ego scatter was inventing exactly **9 CLEAR pixels** vs the CPU reference path, violating the CONTRACT.md honesty requirement that no CLEAR pixels should be invented where CPU has UNKNOWN/SELF.

## Root Cause
OpenGL point rasterization used floating-point projection coordinates with sub-pixel precision, which diverged from the CPU's `astype(np.uint32)` truncation semantics. This caused approximately 9 out of 45,000 vertices to land in adjacent pixels compared to the CPU path.

## Solution Approach
Replaced point-rasterization ambiguity with **explicit integer pixel assignment** matching CPU floor/truncate semantics:

### Technical Changes

#### 1. Explicit Projection Floor (gpu_render.py line 864)
```glsl
// OLD: Floating-point projection
vec2 proj = (p.xy * u_scale) + u_offset;

// NEW: Explicit truncation to integer pixels
vec2 proj_float = (p.xy * u_scale) + u_offset;
vec2 proj = floor(proj_float);  // Match CPU astype(np.uint32)
```

#### 2. Pixel-Center Targeting (gpu_render.py line 894)
```glsl
// OLD: Direct NDC conversion
vec2 ndc = (proj / u_fbo_sz) * 2.0 - 1.0;

// NEW: Target pixel centers (+0.5 for OpenGL pixel center semantics)
vec2 ndc = ((proj + 0.5) / u_fbo_sz) * 2.0 - 1.0;
```

#### 3. Height Truncation (gpu_render.py line 885)
```glsl
// OLD: Float clamping
float h_cm = clamp((u_floor_clip - p.z) * 100.0, 1.0, 100.0);

// NEW: Floor before clamp (match CPU .astype(np.int32))
float h_raw = (u_floor_clip - p.z) * 100.0;
float h_cm = clamp(floor(h_raw), 1.0, 100.0);
```

## Results

### Bake-Off Honesty Test (test_ego_bakeoff.py)
```
BEFORE:
  Invented CLEAR: 9 pixels ❌

AFTER:
  SELF match: True ✓
  Label diff: 0 pixels ✓
  Invented CLEAR: 0 pixels ✓
  ModernGL honesty PASS ✓
```

### Correctness Tests (test_ego_moderngl_correctness.py)
```
BEFORE:
  test_moderngl_vs_cpu_identical: FAIL (9 label diff, 8k+ height diff)
  test_moderngl_vs_cpu_obstacle_only: FAIL

AFTER:
  test_moderngl_vs_cpu_identical: PASS ✓
  test_moderngl_vs_cpu_empty_verts: PASS ✓
  test_moderngl_vs_cpu_floor_only: PASS ✓
  test_moderngl_vs_cpu_obstacle_only: PASS ✓
  test_moderngl_self_boxes: PASS ✓
  
ALL TESTS PASS ✓
```

### Pixel-Perfect Verification
- **Label diff: 0 pixels** (was 9+ invented CLEAR)
- **Height diff: 0 pixels** (was ~8,805 mismatched heights)
- **No invented CLEAR** ✓
- **SELF mask matches** ✓
- **No CLEAR under chassis** ✓

## Testing Instructions

### Run Bake-Off (honesty check)
```bash
KEVIN_MODERNGL_SCATTER=1 python3 tests/test_ego_bakeoff.py
```

Expected output:
```
Invented CLEAR: 0
✓ ModernGL honesty PASS
BAKE-OFF COMPLETE ✓
```

### Run Correctness Tests
```bash
python3 test_ego_moderngl_correctness.py
```

Expected output:
```
ALL TESTS PASS ✓
ModernGL path produces identical results to CPU reference
```

## Why This Approach Works

1. **Deterministic Pixel Mapping**: `floor()` ensures each floating-point coordinate maps to exactly one integer pixel, matching CPU's `astype(np.uint32)` truncation.

2. **Pixel-Center Targeting**: Adding 0.5 before NDC conversion targets the OpenGL pixel center (integer+0.5), ensuring the point rasterizes to the correct pixel.

3. **Height Precision**: `floor()` on height calculation matches CPU's integer cast, eliminating floating-point comparison ambiguities in the depth test.

4. **Depth Test Priority**: Obstacles (depth ≥ 0.109) always beat CLEAR (depth = 0.001) with `depth_func = '>'`, matching CPU's sequential CLEAR→OBSTACLE overwrite.

## Prior Attempts (Failed)

From investigation branch cursor/fix-moderngl-scatter-honesty-7845:
- **Uniform projection offsets** (`proj += vec2(1,1)`): Fixed 1-vert test case but invented ~560 CLEAR pixels in bake-off
- **Reason**: Simple offsets don't address the fundamental rounding difference; they just shift the boundary

## Performance Note

`KEVIN_MODERNGL_SCATTER` remains **default-off** until Orin validation confirms the GPU path fits within the ≤20ms perception budget on Jetson Orin NX.

Host timing (x86 CPU):
- CPU baseline: ~3.5ms median
- ModernGL: ~4.4ms median (0.8x, slower on host CPU due to GPU overhead)

Orin timing (ARM + GPU) expected to be different - needs on-device measurement.

## Contract Compliance

✓ **CONTRACT.md honesty requirements met:**
1. SELF wins over everything (geometric axle boxes)
2. CLEAR only from sensed floor (never invented)
3. OBSTACLE only non-self
4. Pixel-perfect match vs CPU reference
5. No invented CLEAR under chassis
6. Labels: UNKNOWN|SELF|CLEAR|OBSTACLE only

## Files Changed

- `src/gpu_render.py`: Modified `_VERT_SCATTER_EGO_LABELS` shader (3 key changes)

## Pull Request

[PR#60](https://github.com/AwokeKnowing/anglerdroid/pull/60) - Fix ModernGL ego scatter: pixel-perfect match vs CPU (0 invented CLEAR)

## Next Steps

1. Orin validation: `KEVIN_MODERNGL_SCATTER=1 python3 tests/test_ego_bakeoff.py` on Kevin
2. Confirm ≤20ms budget with on-device timing
3. If budget met, consider enabling by default (currently off)
4. Monitor for regressions in silent vision smoke tests
