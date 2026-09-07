# Overhang Approach Detection Fix - Summary

## Problem Statement
Kevin crashed the mast under a table. The near-field reflex (<30cm) fired too late when already underneath. Live logs showed `mast=0.00` while `fwd/mid` looked clear.

## Root Cause
- Topdown height map only shows tall cells under mast AFTER entry
- Forward `mast_score` stays ~0 while approaching because footprint clear clears the robot's own footprint
- Near-field reflex (<30cm) is reactive, not proactive

## Solution: Overhang Approach Detection

Added **early warning system** using RS1 topdown depth to detect overhangs at **30-70cm** before commitment.

### Detection Strategy
```
Distance Ranges:
  30-70cm: OVERHANG APPROACH (new) → blocks forward, allows reverse
  <30cm:   NEAR-FIELD (existing)  → blocks forward, allows reverse
  <1cm:    Floor obstacle          → normal clearance scaling
```

### Spatial Cone (RS1 coordinates)
```
Forward cone:
  X: -0.15m to 0.15m (lateral width)
  Y: 0.05m to 0.40m (forward depth)
  Z: 0.30m to 0.70m (distance from camera)
```

## Implementation

### 1. Core Detection (`src/vision.py`)

**New function:**
```python
check_topdown_overhang_approach(verts, near_m=0.30, far_m=0.70, ...)
```
- Checks RS1 point cloud for elevated structures at 30-70cm
- Filters by forward cone (ahead of robot nose, centered laterally)
- Returns: (triggered, count, median_z)
- Integrated into Vision capture loop

**State tracking:**
- `_topdown_overhang_approach` (bool)
- `_overhang_approach_count` (int)
- `_overhang_approach_median_z` (float)

### 2. Safety Integration (`src/safety.py`)

**SafetyGuard reflex:**
```python
if topdown_overhang_approach:
    self._fwd_scale = 0.0
    # bwd/ang computed normally from obstacles
```

**Priority order:**
1. `topdown_hazard` (RGB bump/checkered)
2. `topdown_overhang_approach` (NEW - depth 30-70cm)
3. `topdown_near_field` (depth <30cm)
4. Normal clearance scaling

### 3. Planning Integration (`src/house_bot.py`)

**HouseBot decision logic:**
```python
if overhang_approach:
    fwd_scale = 0.0
    scores["fwd_near"] = 0.0
    scores["fwd_mid"] = 0.0
```
- Blocks COMMIT before near-field fires
- Prevents wander into overhangs

## Test Coverage

### New Tests (`test_overhang_approach.py`)
✅ 13 tests, all passing

| Test | What It Validates |
|------|-------------------|
| 1 | Table at 50cm detected |
| 2 | Rear cone ignored |
| 3 | Near-field handles <30cm |
| 4 | Too far (>70cm) ignored |
| 5 | Open floor no trigger |
| 6 | Mixed scene detection |
| 7 | SafetyGuard stops fwd |
| 8 | SafetyGuard clear allows |
| 9 | HouseBot blocks COMMIT |
| 10 | Noise filtering works |
| 11 | Empty clouds handled |
| 12 | Lateral cone limits |
| 13 | Near-field priority |

### Existing Tests - No Regressions
✅ All 34 existing tests pass
- Near-field: 10/10 ✅
- Checkered mat: 10/10 ✅
- Stuck detection: 4/4 ✅

## Success Criteria ✅

✅ **1. Approaching table zeros fwd / blocks COMMIT before near-field**
- Overhang approach: 30-70cm (early)
- Near-field: <30cm (late)
- Tests confirm correct ordering

✅ **2. Open floor allows wander (no false permanent stop)**
- Open floor tests pass
- Distance filtering (30-70cm) prevents far false positives
- Lateral cone prevents side false positives
- Noise filtering (min 80 pixels) prevents transient spikes

✅ **3. Existing tests pass (no regressions)**
- All 34 tests pass
- Near-field, checkered, stuck unchanged

✅ **4. Brief note of cue and why**
- **Cue:** RS1 topdown depth at 30-70cm in forward cone
- **Why:** 
  - Early: Detects before commitment (30-70cm vs <30cm)
  - Ego-frame: No SLAM dependency
  - Focused: Forward cone + distance range filters noise
  - Robust: Uses existing RS1 hardware

## Before/After Behavior

### Before (Incident)
```
1. Approach table: floor looks clear, mast=0.00
2. HouseBot: fwd_mid=0.85, COMMIT forward
3. Drive under table
4. Near-field fires at <30cm (too late, already underneath)
5. CRASH: mast hits table underside
```

### After (Fixed)
```
1. Approach table: floor looks clear
2. Overhang approach: detects table underside at 50cm
3. SafetyGuard: fwd_scale=0.0 (immediate stop)
4. HouseBot: fwd_near=0.0, fwd_mid=0.0, blocks COMMIT
5. Robot STOPS before entry, reverse/turn available
6. Near-field never fires (stopped early)
```

## Key Insights

1. **Layered defense:** Multiple detection ranges (70cm → 30cm → 1cm)
2. **Ego-frame reflexes:** Work without SLAM (critical during SLAM loss)
3. **Escape allowed:** Reverse/turn available when rear clear
4. **Small focused change:** Single detection function + integration points

## Files Changed

- `src/vision.py` (+88 lines): Detection function + integration
- `src/safety.py` (+11 lines): Reflex integration
- `src/house_bot.py` (+8 lines): Planning integration
- `test_overhang_approach.py` (+494 lines): Comprehensive tests

## Pull Request

PR #12: https://github.com/AwokeKnowing/anglerdroid/pull/12
Branch: `cursor/overhang-approach-detection-9656`

**Status:** Draft (drive DISARMED until field testing confirms safe)

## Next Steps

1. ✅ Code review
2. ⏳ Simulation testing (approach table scenarios)
3. ⏳ Field testing (controlled environment)
4. ⏳ Validate with real table approach
5. ⏳ Mark ready for review & merge

---

**Author:** Cloud Agent  
**Date:** 2026-09-07  
**Incident:** Kevin mast crash under table  
**Fix:** Overhang approach detection (30-70cm early warning)
