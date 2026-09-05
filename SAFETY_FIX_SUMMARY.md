# WheelBase Command-Stale Safety Fix - Quick Reference

## What Was Fixed

**Problem**: One-shot velocity commands kept robot driving until ODrive watchdog expired (2.0s → red LED / disarm).

**Solution**: Added command-staleness detection that zeros velocity after 0.5s of no new commands, BEFORE watchdog trips.

## Safety Timeline

```
One-shot command sent
         ↓
    [Robot moving]
         ↓
t = 0.5s: ⚠️  SAFETY: Command stale
          → Zero velocity
          → Disable watchdog
          → Transition to IDLE
          → Clear errors
         ↓
    [Robot stopped, clean IDLE state (blue LED)]
         ↓
t = 2.0s: ODrive watchdog does NOT trip ✅
         ↓
    [Remains in clean IDLE, disarm_reason=0]
```

## Key Files

| File | Purpose |
|------|---------|
| `src/wheelbase.py` | Safety implementation (command-stale watcher) |
| `test_stale_command_safety.py` | Automated test script |
| `WHEELBASE_SAFETY.md` | Comprehensive safety architecture docs |
| `HARDWARE_VERIFICATION.md` | Step-by-step hardware test procedures |

## Quick Test

```bash
# Automated test (safe, robot will move 0.5s then stop)
python3 test_stale_command_safety.py
```

Expected output: Safety message at t=0.5s, robot stops, no watchdog trip.

## Normal Operations (Unaffected)

✅ **Gamepad** - 30 Hz updates, no false triggers  
✅ **`twist_for()`** - 10 Hz scripted moves work normally  
✅ **Navigator** - Continuous twist commands unaffected  
✅ **Incline hold** - Torque-based position holding preserved  

## Configuration

```python
# In src/wheelbase.py (class WheelBase)
COMMAND_STALE_TIMEOUT = 0.5     # app safety (adjustable 0.3-0.8s)
# ODrive watchdog = 2.0s         # hardware backstop (fixed)
# idle_zero_timeout_s = 5.0s     # idle transition (configurable)
```

**Timing invariant**: `COMMAND_STALE_TIMEOUT < watchdog_timeout` (strictly enforced)

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Watchdog still trips | Branch not deployed | Verify branch, redeploy |
| False triggers while driving | Thread starvation | Check CPU load, increase timeout |
| Safety never triggers | Watcher thread not started | Check startup logs |
| Robot rolls on incline | Torque threshold too high | Lower `INCLINE_TORQUE_THRESHOLD` |

## PR Status

**Branch**: `cursor/wheelbase-command-stale-safety-faca`  
**PR**: https://github.com/AwokeKnowing/anglerdroid/pull/1  
**Status**: Draft - awaiting hardware verification  

### Before Merge Checklist

- [ ] Run `test_stale_command_safety.py` on robot
- [ ] Complete all 5 tests in HARDWARE_VERIFICATION.md
- [ ] Verify NO watchdog trips
- [ ] Verify NO false triggers during gamepad use
- [ ] Test incline hold behavior

## Questions?

See full documentation:
- **Architecture**: WHEELBASE_SAFETY.md
- **Hardware testing**: HARDWARE_VERIFICATION.md
- **Code**: src/wheelbase.py (lines 28-296)
