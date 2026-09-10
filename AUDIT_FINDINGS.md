# HouseBot / SafetyGuard Hard-Stop Bypass Audit
**Date:** 2026-09-07  
**Branch:** `cursor/audit-recover-safety-e6df`  
**Auditor:** Cloud Agent (Re-arm Criterion #5)

## Executive Summary

**RESULT: ✅ NO BYPASSES FOUND**

All code paths correctly enforce hard safety stops. No forward motion (v > 0) can bypass:
- Near-field reflex (`topdown_near_field` → `fwd_scale=0`)
- Topdown hazard (checkered mat / bump → `fwd_scale=0`)
- Stuck immobilize (wheels spinning → `fwd_scale=0`, `bwd/ang` preserved)
- Overhang approach detection (`fwd_scale` near 0)
- Any condition where `fwd_scale == 0`

---

## Audit Scope

Audited all code paths that could potentially command robot motion:

1. **HouseBot phased recover** (BACK → SPIN → COMMIT)
2. **Escape spin / late divert** commands
3. **LocalExecutive** (VFH and MPPI backends)
4. **Tools API** (`twist()`, `twist_for()`)
5. **People_live social motion** (approach/retreat)
6. **MPPI soft costs** (planning hints vs hard constraints)
7. **Simulation policy** (`MppiSimPolicy`, action masks)

---

## Architecture: Safety Scale Enforcement

### Critical Choke Point: `wheelbase.set_wheel_vels()`

**ALL** motion commands route through this single method (`src/wheelbase.py:610-628`):

```python
def set_wheel_vels(self, left_tps: float, right_tps: float):
    fwd = (left_tps + right_tps) / 2.0
    turn = (right_tps - left_tps) / 2.0
    
    # Safety scales applied HERE (lines 622-626)
    if fwd > 0:
        fwd *= self._safety_fwd      # Forward motion scaled
    elif fwd < 0:
        fwd *= self._safety_bwd      # Reverse motion scaled
    turn *= self._safety_ang         # Angular motion scaled
    
    left_tps = fwd - turn
    right_tps = fwd + turn
    # ... send to motors
```

**Property:** If `_safety_fwd == 0.0`, then any positive `fwd` becomes `0.0`.

---

## Detailed Findings by Component

### 1. HouseBot Phased Recover (src/house_bot.py)

**BACK Phase (lines 196-210):**
- Calls `tools.twist_for(BACK_MPS, 0.25*sign, ...)`  
- `BACK_MPS = -0.22` (negative velocity)
- Applies `_safety_bwd` scale ✓
- **Verdict:** Safe – uses reverse motion

**SPIN Phase (lines 212-227):**
- Calls `tools.twist_for(0.0, SPIN_RAD*sign, ...)`
- Zero forward velocity, pure angular
- **Verdict:** Safe – no forward component

**COMMIT Phase (lines 230-261):**
- Calls `local_executive.set_goal_xy(...)` (line 239-242)
- Goal commands flow through `local_executive.tick()` → VFH/MPPI → `tools.twist()` → `wheelbase.set_wheel_vels()`
- **Verdict:** Safe – all commands apply safety scales

**Re-pin during COMMIT (lines 369-384):**
- Detects `hard_repin` when `fwd_scale < LATE_FWD_SCALE` during COMMIT
- Triggers new `_start_back()` sequence
- **Verdict:** Safe – restarts BACK phase, no forward bypass

### 2. Tools API (src/tools.py, src/wheelbase.py)

**`tools.twist_for()` → `wheelbase.twist_for()` (lines 404-417):**
- Runs on 10Hz thread, calls `wheelbase.twist()` internally
- `twist()` → `set_wheel_vels()` ✓
- **Verdict:** Safe

**`tools.twist()` → `wheelbase.twist()` (lines 363-369):**
- Directly calls `set_wheel_vels()` ✓
- **Verdict:** Safe

### 3. LocalExecutive (src/local_executive.py)

**VFH Backend (lines 180-212):**
- Returns `(fwd, ang)` from `navigator.compute_twist(atlas)`
- Called from `main.py:268` → `tools.twist()` → `set_wheel_vels()` ✓
- **Verdict:** Safe

**MPPI Backend (lines 140-178):**
- Returns `(fwd_mps, ang_rads)` from `mppi.tick()`
- Called from `main.py:268` → `tools.twist()` → `set_wheel_vels()` ✓
- **Note:** MPPI soft costs NOT passed from live code (line 163)
  - Soft costs are planning hints only (prefer directions)
  - Hard safety scales still applied at wheelbase ✓
- **Verdict:** Safe

### 4. People_live Social Motion (src/people_live.py)

**Goal Commands (lines 232-301):**
- All calls use `local_executive.set_goal_xy()` or `.set_wander()`
- Flow: `local_executive.tick()` → `tools.twist()` → `set_wheel_vels()` ✓
- **Verdict:** Safe

**Social Hold Priority (lines 67-68):**
- Flag `social_priority` only yields speaker to faces
- Does not bypass motion safety
- **Verdict:** Safe

### 5. MPPI Soft Costs

**Live Code (`src/mppi_costmap.py:342-398`):**
- `soft_scales` parameter exists but NOT passed from `local_executive._tick_mppi()` (line 163)
- Even if passed, soft costs add planning penalties, don't bypass hard scales
- All MPPI output goes through `wheelbase.set_wheel_vels()` ✓
- **Verdict:** Safe (soft costs are hints, hard scales still enforced)

**Sim Code (`sim/mppi_policy.py`):**
- Uses `DualScales` + `action_mask.apply_action_mask()` (lines 69-89)
- Every return path calls `_mask()` which enforces safety scales
- Existing test: `test_recover_never_forward` validates this ✓
- **Verdict:** Safe

### 6. Simulation Policy Tests (sim/test.py, sim/test_action_mask.py)

Existing tests already validate the safety property:
- `test_housebot_respects_hard_stop` ✓
- `test_commit_never_overrides_fwd0` ✓
- `test_phased_recover_sequence` ✓
- `test_recover_never_forward` (action_mask) ✓
- `test_hard_zero_never_weakened` (action_mask) ✓

**Run result:** All 24 sim tests + 12 action_mask tests pass ✓

---

## Safety Scale Sources (src/vision.py:1002-1018)

Hard stops enforced in `vision.py` by zeroing `_safety._fwd_scale`:

1. **Near-field reflex** (line 176-181):
   - `topdown_near_field` (RS1 topdown sees object <30cm)
   - Sets `_fwd_scale = 0.0`
   - Keeps `_bwd_scale`, `_ang_scale` for escape ✓

2. **Topdown hazard** (line 162-168):
   - RS1 RGB detects bump threshold or checkered mat
   - Sets `_fwd_scale = 0.0`
   - Keeps `_bwd_scale`, `_ang_scale` for escape ✓

3. **Stuck immobilize** (line 962-964):
   - `pose_src.is_stuck` (wheels spinning, no motion)
   - Sets `_fwd_scale = 0.0`
   - Keeps `_bwd_scale`, `_ang_scale` for recover ✓
   - Comment (line 1003): "kill forward only so HouseBot RECOVER can reverse/turn off a lip"

4. **Topdown lost** (line 967-976):
   - No top-down depth reading
   - Sets ALL scales to 0 (`_fwd_scale`, `_bwd_scale`, `_ang_scale`) ✓
   - Total immobilize (safety-critical sensor failure)

---

## Test Coverage

### New Regression Tests (`test_safety_hard_stop_bypass.py`)

Created comprehensive unit tests covering:
1. ✅ `tools.twist_for()` respects `fwd_scale=0`
2. ✅ Reverse allowed when `fwd_scale=0`, `bwd_scale=1.0` (stuck immobilize)
3. ✅ Angular-only motion (escape spin) allowed with `fwd_scale=0`
4. ✅ LocalExecutive MPPI backend respects `fwd_scale=0`
5. ✅ LocalExecutive VFH backend respects `fwd_scale=0`
6. ✅ HouseBot COMMIT phase respects hard stop
7. ✅ All hard-stop scenarios enforced
8. ✅ Near-field reflex allows escape
9. ✅ Checkered mat hard stop blocks forward, allows reverse

**Result:** All 9 tests pass ✓

### Existing Sim Tests (`sim/test.py`)

Confirmed existing tests validate:
- Phased recover sequence respects safety
- COMMIT never overrides `fwd=0`
- HouseBot respects hard stops
- Stress tests (5000 steps × 5 scenarios) all collision-free

**Result:** All 24 sim tests + 12 action_mask tests pass ✓

---

## Audit Conclusions

### ✅ No Bypasses Found

**All code paths correctly enforce hard safety stops:**
- HouseBot BACK/SPIN/COMMIT phases
- Escape spins and late diverts
- LocalExecutive (VFH and MPPI)
- People_live social motion
- Tools API

### ✅ Architecture is Sound

**Single choke point design:**
- `wheelbase.set_wheel_vels()` is the ONLY path to motors
- Safety scales consistently applied: `fwd>0` uses `_safety_fwd`, `fwd<0` uses `_safety_bwd`
- No direct motor access bypasses exist

### ✅ Stuck Immobilize Intent Preserved

**When `fwd_scale=0` due to stuck detection:**
- Forward motion blocked (`fwd *= 0.0 = 0`)
- Reverse motion allowed (`bwd_scale=1.0` preserved)
- Angular motion allowed (`ang_scale=1.0` preserved)
- Enables RECOVER sequence: back out → spin → commit new direction

### ✅ Test Coverage Comprehensive

**Regression tests lock the property:**
- Unit tests prove safety scale enforcement at API level
- Sim tests prove policy respects hard stops during recover/commit
- Action mask tests prove MPPI never returns `v>0` when `fwd_scale=0`

---

## Recommendations

### 1. ✅ COMPLETE: Add regression tests
New `test_safety_hard_stop_bypass.py` provides comprehensive coverage.

### 2. Consider: MPPI soft costs in live code
Currently, `local_executive._tick_mppi()` does NOT pass `soft_scales` to `mppi.tick()`.

**Impact:** Low/None
- Hard safety scales still enforced at wheelbase ✓
- Soft costs only bias planning (prefer open directions)
- Not required for safety, only for smoother navigation

**If implemented:** Pass safety scales from `vision.safety_fwd_scale` to MPPI as soft cost hints.

### 3. ✅ COMPLETE: Document audit findings
This document serves as the audit record.

---

## Re-arm Criterion #5 Status

**Criterion:** "No live code path can force motion through a hard stop"

**Status:** ✅ **SATISFIED**

**Evidence:**
1. Comprehensive code audit completed (all paths checked)
2. Architectural analysis confirms single safety choke point
3. Regression tests prove property holds
4. Existing sim tests validate recover/commit behavior
5. No bypasses discovered

**Recommendation:** Safe to proceed with re-arm after review.

---

## Appendix: Command Flow Diagram

```
┌─────────────────────────────────────────────┐
│  Motion Command Sources                     │
├─────────────────────────────────────────────┤
│  • HouseBot (BACK/SPIN/COMMIT)             │
│  • LocalExecutive (VFH/MPPI)               │
│  • People_live (social goals)              │
│  • Navigator (reactive avoid)              │
│  • Gamepad (manual)                        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  tools.twist() or tools.twist_for()         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  wheelbase.twist() or .twist_for()          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  wheelbase.set_wheel_vels()                 │
│  ┌───────────────────────────────────────┐ │
│  │ SAFETY SCALES APPLIED HERE:           │ │
│  │   if fwd > 0: fwd *= _safety_fwd      │ │
│  │   elif fwd < 0: fwd *= _safety_bwd    │ │
│  │   turn *= _safety_ang                 │ │
│  └───────────────────────────────────────┘ │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  ODrive Motors                              │
│  (no direct access from any other path)    │
└─────────────────────────────────────────────┘
```

**Key Property:** `wheelbase.set_wheel_vels()` is the ONLY path to motors. All safety scales enforced here.

---

## Files Modified/Added

- **NEW:** `test_safety_hard_stop_bypass.py` – Comprehensive unit tests (9 tests, all pass)
- **NO CODE CHANGES REQUIRED** – No bypasses found

---

**Audit completed:** 2026-09-07  
**Branch:** `cursor/audit-recover-safety-e6df`  
**Next step:** Open PR for review
