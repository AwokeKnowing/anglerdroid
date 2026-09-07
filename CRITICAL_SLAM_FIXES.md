# Critical SLAM Fixes - Priority Update from James

**Date:** 2026-09-07  
**Status:** IN PROGRESS  
**Priority:** CRITICAL

## Problem Statement (from James)

**Map-frame keepouts do NOT work — there is no usable SLAM.**

Live logs show `enc=False` (encoder fallback), making odometry completely unreliable. When encoders fail, the system falls back to commanded velocity, which makes wheel slip invisible and breaks all SLAM/pose estimation.

## Root Cause Analysis

### 1. Encoder Failure → No SLAM

When `enc=False` appears in logs:
- Encoder reader thread failed to get valid data (native CAN 0x09 or SDO)
- System falls back to **commanded velocity** (no feedback)
- Wheel slip becomes **invisible** to odometry
- **SLAM cannot work** without reliable wheel odometry

**Location:** `src/wheelbase.py:492` - `_enc_ok` starts False, only becomes True when encoders work

### 2. No Operator Awareness

Before these fixes:
- No clear indication that SLAM was unreliable
- Operators could mark map-frame keepouts even when pose was garbage
- Autonomous navigation would proceed with broken odometry
- No immobilization when tracking failed

### 3. False Sense of Reliability

Previous PR added metrics but didn't address:
- **Making encoders actually work reliably**
- **Safe behavior when SLAM fails**
- **Clear operator signals**

## Critical Fixes Implemented

### ✅ 1. SLAM Lock Detection

**Problem:** No way to know if SLAM is trustworthy  
**Solution:** `vision.slam_locked` property that checks:
- Encoders working (`enc_ok=True`, age <1s)
- Visual odometry OR recent visual (<5s)
- Top-down depth working

**Files:** `src/vision.py:250-290, 990-1010`

**Behavior:**
```python
if not vision.slam_locked:
    print(f"🔴 SLAM NOT LOCKED: {vision.slam_lock_reason}")
    # Map-frame features disabled
```

### ✅ 2. Autonomous Motion Immobilization

**Problem:** System drove autonomously with broken odometry  
**Solution:** Hard immobilize when SLAM not locked

**Files:** `src/vision.py:813-850`

**Behavior:**
```
⚠️  vision: SLAM NOT LOCKED (encoder_failed) — autonomous motion disabled
```

Safety scales forced to 0.0 when:
- Encoders failing
- Visual tracking lost >5s
- Top-down depth lost

### ✅ 3. Keepout Marking Prevention

**Problem:** Operators could mark keepouts when pose was garbage  
**Solution:** Check `slam_locked` before allowing map-frame marks

**Files:** `src/keepouts.py:179-231, 246-248`

**Behavior:**
```bash
$ python -m keepouts mark dog_bed
🔴 SLAM NOT LOCKED — cannot mark map-frame keepouts
   Wait for SLAM lock (check logs for '🟢 SLAM LOCKED') or use --no-slam-check
```

### ✅ 4. Prominent Status Logging

**Problem:** No clear operator signal about SLAM health  
**Solution:** 
- Periodic status every 10s in main loop
- State change logging with emojis
- Reason codes for failures

**Files:** `src/main.py:160-180`, `src/vision.py:360-380`

**Example Output:**
```
🔴 SLAM: NOT LOCKED (encoder_stale_1.2s)
   ⚠️  Map-frame navigation disabled — fix: encoder_stale_1.2s

[encoders recover]

🟢 SLAM LOCKED (was unlocked 12.3s)
   ✓ Encoders working, tracking quality good
```

### ✅ 5. Encoder Startup Diagnostics

**Problem:** Silent encoder failures during startup  
**Solution:** Extra diagnostic logging during initialization

**Files:** `src/wheelbase.py:534-541`

**Behavior:**
```
encoder: native CAN 0x09 not supported, using SDO fallback
⚠️  encoder startup: 5/10 attempts, still failing (native=False, sdo_fails=7)
   Check: CAN bus up? ODrive powered? Cables connected?
```

## What Still Needs Investigation

### 🔍 1. Why Are Encoders Failing?

**Possible causes:**
- CAN bus not brought up properly (`can1` interface down)
- ODrive not powered or in error state
- Cable connection issues
- Native CAN 0x09 not supported by ODrive firmware
- SDO read bus contention with velocity commands

**Next steps:**
1. Check CAN bus status: `ip link show can1`
2. Check ODrive errors with `odrivetool`
3. Verify encoder config in ODrive (ticks per rev, mode)
4. Test native vs SDO encoder reads in isolation
5. Review CAN bus lock contention

### 🔍 2. Visual Odometry on Carpet

**Problem:** Low-texture floors (carpet, uniform tile) have low visual feature density

**Current:** `MIN_VIS_CONFIDENCE = 0.10` threshold

**Possible improvements:**
- Lower confidence threshold for carpet
- Adaptive thresholds based on feature density
- Alternative visual features (optical flow, dense methods)
- Fallback to wheel-only with higher slip tolerance

### 🔍 3. Persistent Local Map

**Problem:** Map doesn't survive session restart

**Short-term:** Document session-local limitations  
**Long-term:** Design persistent map with:
- Relocalization on startup (match current view to saved map)
- Visual anchors (AprilTags) for absolute reference
- Saved keyframe database
- Transform keepouts to persistent frame

## Testing Checklist

### Before Deployment

- [ ] Verify CAN bus up: `ip link show can1` → state UP
- [ ] Verify ODrive accessible: `odrivetool` connect
- [ ] Check encoder config: `odrivetool` → check encoder mode/cpr
- [ ] Test encoder reads: verify `enc=True` in logs within 5s of startup
- [ ] Verify SLAM lock: see "🟢 SLAM LOCKED" in logs

### During Operation

- [ ] Monitor encoder health every 30s: `enc=True` in odom logs
- [ ] Watch for SLAM lock loss: no "🔴 SLAM LOCK LOST" messages
- [ ] Verify autonomous motion works when locked
- [ ] Verify immobilization when unlocked
- [ ] Try marking keepout when unlocked → should reject

### Post-Session

- [ ] Review logs for encoder failures
- [ ] Check SLAM lock downtime duration
- [ ] Identify root cause if encoders failed
- [ ] Document any CAN bus issues

## Log Signatures

### Good (SLAM Working)

```
encoder: reading actual wheel velocities (native CAN)
🟢 SLAM LOCKED (was unlocked 2.3s)
   ✓ Encoders working, tracking quality good
```

### Bad (SLAM Broken)

```
encoder: SDO read failed 10 times (vl=None vr=None) — backing off
🔴 SLAM LOCK LOST: encoder_stale_1.2s
   ⚠️  Map-frame navigation DISABLED until lock restored
⚠️  vision: SLAM NOT LOCKED (encoder_stale_1.2s) — autonomous motion disabled
```

### Keepout Rejection

```
$ python -m keepouts mark dog_bed
🔴 SLAM NOT LOCKED — cannot mark map-frame keepouts
   Wait for SLAM lock (check logs for '🟢 SLAM LOCKED') or use --no-slam-check
```

## API Changes (Backward Compatible)

### New Properties

```python
# Vision
vision.slam_locked -> bool           # True if SLAM reliable
vision.slam_lock_reason -> str       # Reason if not locked

# WheelBase (from earlier PR)
wb.get_encoder_health() -> dict      # Encoder diagnostics
```

### Modified Functions

```python
# Keepouts
keepouts.paint_ego(obs, pose, slam_locked=False)  # Now tracks lock status
keepouts.mark_named(name, pose=None, require_slam_lock=True)  # Checks lock
```

### Pose File Format

`~/.kevin/latest_pose.json` now includes:
```json
{
  "x": 1.23,
  "y": 0.45,
  "yaw": 0.12,
  "t": 1725668520.123,
  "slam_locked": true
}
```

## Files Changed (This Update)

**Critical fixes:**
- `src/vision.py` - SLAM lock detection, immobilization, status logging
- `src/wheelbase.py` - Encoder startup diagnostics
- `src/keepouts.py` - SLAM lock requirement for marking
- `src/local_executive.py` - Pass slam_locked to paint_ego
- `src/main.py` - Periodic SLAM status logging

**Documentation:**
- `CRITICAL_SLAM_FIXES.md` - This document

## Success Criteria

1. ✅ **No map-frame keepouts when SLAM unlocked**
2. ✅ **Operator always sees SLAM status** (logs + future UI)
3. ✅ **Autonomous motion disabled when tracking fails**
4. 🔄 **Encoders work reliably** (needs investigation + fixes)
5. 🔄 **Visual odom works on carpet** (needs tuning)
6. 📋 **Persistent local map** (design documented)

## Next Steps (Priority Order)

1. **Diagnose encoder failures** (CAN bus, ODrive state, firmware)
2. **Test encoder initialization** on actual hardware
3. **Tune visual odometry** for carpet/low-texture
4. **Add UI indicator** for SLAM lock status
5. **Design persistent map** with relocalization
6. **Document operator procedures** for SLAM health monitoring

## Related Issues

- Original: "Improve stable SLAM/reconstruction"
- Priority update: "Map-frame keepouts do NOT work"
- Root cause: `enc=False` → no usable SLAM

---

**Status:** Critical fixes implemented, encoder diagnosis needed.  
**Blocking:** Encoder reliability investigation on hardware.  
**Risk:** Current system cannot do autonomous navigation until encoders work.
