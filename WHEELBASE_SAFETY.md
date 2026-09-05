# WheelBase Motion Safety Architecture

## Overview

The WheelBase class implements a **defense-in-depth** safety model with three layers of protection against runaway motion:

1. **Application-level command staleness detection** (primary safety)
2. **ODrive hardware watchdog** (secondary backstop)
3. **Incline-aware idle transition** (prevents freewheel on slopes)

## Safety Layers

### Layer 1: Command Staleness Detection (PRIMARY)

**Purpose**: Detect and halt stale non-zero velocity commands before hardware intervention.

**Mechanism**:
- Every call to `set_wheel_vels()` updates `_last_command_time`
- Background monitor thread checks every 0.2s (5 Hz):
  - Are we in closed-loop control?
  - Is commanded velocity non-zero?
  - Has more than 0.5s elapsed since last command?
- If stale: immediately zero velocity, feed watchdog, print warning

**Timeout**: `COMMAND_STALE_TIMEOUT = 0.5s`

**Trigger message**:
```
⚠️  SAFETY: Command stale (0.51s), zeroing velocity (L=0.50, R=0.50)
```

**Design rationale**: A continuous motion control system should receive regular command updates. If no new command arrives within 0.5s while moving, assume the command source has failed or lost connection, and stop proactively.

### Layer 2: ODrive Watchdog (BACKSTOP)

**Purpose**: Hardware-enforced failsafe if application hangs or crashes.

**Mechanism**:
- Enabled at 2.0s timeout when entering closed-loop control
- Fed every 1.0s by `set_wheel_vels()` (when velocity changed or watchdog due)
- If watchdog expires: ODrive disarms, red LED flashes
- Error code: `WATCHDOG_TIMER_EXPIRED` (0x01000000)

**Timeout**: `2.0s` (configured in `enable_watchdog()` call)

**Design rationale**: Hardware backstop for software failures. Should **never trigger** during normal operation because Layer 1 stops the robot first.

### Layer 3: Incline-Aware Idle (SLOPE SAFETY)

**Purpose**: Transition to IDLE only on flat ground to prevent freewheel on inclines.

**Mechanism**:
- After holding zero velocity for 5s, check motor torque
- If torque > 1.0 Nm: stay in closed-loop control (holding position)
- If torque < 1.0 Nm: safe to IDLE (flat ground)

**Timeout**: `idle_zero_timeout_s = 5.0s` (configurable in constructor)

**Design rationale**: Prevent robot from freewheeling down slopes. Keep motors engaged while holding position on inclines.

## Timing Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│  Motion Command Flow (Normal Operation)                         │
│                                                                  │
│  gamepad (30 Hz) ──┐                                            │
│  navigator (30 Hz) ├─→ set_wheel_vels() → ODrive               │
│  twist_for (10 Hz) ─┘     updates _last_command_time           │
└─────────────────────────────────────────────────────────────────┘

Time after last command:
─────────────────────────────────────────────────────────────────────→

0.0s                    0.5s                    2.0s            5.0s
│                        │                        │               │
│                        │                        │               │
│  Robot moving          │  APP SAFETY            │  WATCHDOG     │  IDLE
│  normally              │  zeros velocity        │  (should      │  transition
│                        │  (Layer 1)             │  never trip)  │  (Layer 3)
│                        │                        │  (Layer 2)    │
│                        │                        │               │
│◄──── Safe Operation ──►│◄── 1.5s margin ───────►│◄── 3.0s ────►│

Legend:
  Layer 1 (0.5s) : Command staleness - app detects & zeros
  Layer 2 (2.0s) : Hardware watchdog - backstop if app fails  
  Layer 3 (5.0s) : Idle transition - torque-aware motor idle
```

## Command Source Behaviors

### Gamepad Control
- **Update rate**: 30 Hz (main loop)
- **Staleness**: Will NOT trigger (commands always fresh)
- **Behavior**: Continuous differential drive from joystick

### `twist_for()` Scripted Moves
- **Update rate**: 10 Hz (internal thread)
- **Staleness**: Will NOT trigger during move
- **Behavior**: Timed moves with ramp in/out, auto-stop at end

### Navigator Autonomous Control
- **Update rate**: 30 Hz (main loop)
- **Staleness**: Will NOT trigger during navigation
- **Behavior**: Continuous twist commands toward goal

### One-Shot Commands (PROTECTED)
- **Update rate**: Once, then silence
- **Staleness**: WILL trigger after 0.5s ✅
- **Behavior**: Robot stops before watchdog trips
- **Examples**:
  ```python
  wb.twist(0.5, 0.0)  # single call
  wb.set_wheel_vels(1.0, 1.0)  # single call
  ```

## Safety Testing

### Unit Test: Stale Command Detection

```bash
python3 test_stale_command_safety.py
```

**Test procedure**:
1. Initialize WheelBase
2. Send single `twist(0.3, 0.0)` command
3. Stop sending further commands
4. Monitor for 8 seconds

**Expected results**:
- `t=0.0s`: Robot starts moving at 0.3 m/s
- `t=0.5s`: Safety message printed, velocity → 0
- `t=5.5s`: Motors transition to IDLE
- `t=never`: ODrive watchdog does NOT trip

### Integration Test: Normal Operation

**Gamepad test**:
1. Drive robot with gamepad for 30 seconds
2. Verify NO safety messages (commands are fresh)
3. Release joystick → robot stops normally
4. Verify idle transition after 5s

**`twist_for` test**:
```python
wb.twist_for(0.5, 0.0, duration_secs=3.0, ramp_in_secs=1.0, ramp_out_secs=1.0)
time.sleep(4.0)  # wait for completion
```
- Verify NO safety messages during 3s move
- Verify robot stops normally at end

### Edge Case Test: Incline Hold

**Setup**: Place robot on 10-15° incline

**Test procedure**:
1. Send `twist(0.0, 0.0)` to hold position
2. Wait for command to go stale (0.5s)
3. Wait for idle timeout (5s)

**Expected results**:
- `t=0.5s`: Command stale detected, velocity zeroed (L=0, R=0)
- `t=5.5s`: Idle watcher checks torque → above threshold
- `t=5.5s`: Message: "Incline detected (torque L=X.XXX R=X.XXX Nm), holding position"
- Robot remains in closed-loop control (does NOT freewheel)

## Tuning Parameters

### `COMMAND_STALE_TIMEOUT`

**Default**: `0.5s`

**Constraints**: Must be `< ODrive watchdog timeout` (strict inequality)

**Considerations**:
- Too short: May false-trigger on legitimate command pauses
- Too long: Reduces safety margin before watchdog
- Recommended: `0.3s` to `0.8s` range

### ODrive Watchdog Timeout

**Default**: `2.0s`

**Constraints**: Must be `> COMMAND_STALE_TIMEOUT + stop_time`

**Current setting**: `enable_watchdog(2.0)` in `set_wheel_vels()`

**Considerations**:
- Must give application time to detect staleness and stop
- Current 1.5s margin is conservative
- Do not reduce below 1.5s without testing

### `idle_zero_timeout_s`

**Default**: `5.0s`

**Constraints**: Should be `> watchdog timeout` to ensure Layer 2 never triggers

**Considerations**:
- Longer: Motors stay engaged longer (battery drain, heat)
- Shorter: More responsive to idle, but less safety margin
- Incline hold may keep motors engaged indefinitely (by design)

## Failure Modes & Mitigation

| Failure Mode | Detection | Mitigation | Result |
|--------------|-----------|------------|--------|
| One-shot command (no follow-up) | Layer 1 (0.5s) | Zero velocity | ✅ Robot stops safely |
| Application hang/crash | Layer 2 (2.0s) | Hardware watchdog | ✅ ODrive disarms |
| CAN bus failure | Layer 2 (2.0s) | Watchdog expires | ✅ ODrive disarms |
| Stale command on incline | Layer 1 → Layer 3 | Zero → torque check | ✅ Holds position |
| Thread deadlock | Layer 2 (2.0s) | Hardware watchdog | ✅ ODrive disarms |
| Infinite loop in main | Layer 2 (2.0s) | Hardware watchdog | ✅ ODrive disarms |

## Monitoring & Debugging

### Safety Trigger Logs

When Layer 1 triggers:
```
⚠️  SAFETY: Command stale (0.51s), zeroing velocity (L=0.50, R=0.50)
```

**Action**: Investigate why command source stopped sending updates.

### Watchdog Trip (Layer 2)

```
Left (node 0): active_errors=0x01000000, disarm_reason=0x01000000
Right (node 1): active_errors=0x01000000, disarm_reason=0x01000000
```

**Action**: This should NEVER happen in production. It means:
- Layer 1 failed to detect staleness, OR
- Application crashed/hung before Layer 1 could act

Investigate application health monitoring.

### Normal Idle Transition (Layer 3)

Flat ground:
```
Flat ground (torque L=0.123 R=0.098 Nm), freewheeling
```

Incline hold:
```
Incline detected (torque L=1.234 R=1.198 Nm), holding position
```

## Code References

| Component | File | Function |
|-----------|------|----------|
| Command staleness | `src/wheelbase.py` | `_stale_command_watcher_loop()` |
| Watchdog feed | `src/wheelbase.py` | `set_wheel_vels()` line ~580 |
| Idle transition | `src/wheelbase.py` | `_try_idle_with_torque_check()` |
| Timing constants | `src/wheelbase.py` | `WheelBase` class constants |
| Test script | `test_stale_command_safety.py` | `test_stale_command_safety()` |

## Safety Certification

This implementation satisfies the safety requirement:

> If someone sets a velocity (e.g. 0.5 m/s) once and then stops sending commands, the **application** must bring wheels to **zero and preferably IDLE before** the ODrive watchdog fires.

**Verified behavior**:
- ✅ Application detects staleness at 0.5s
- ✅ Application zeros velocity immediately
- ✅ ODrive watchdog at 2.0s does NOT trip (1.5s margin)
- ✅ Idle transition at 5.0s (torque-aware)
- ✅ Incline hold preserved (does not freewheel)
- ✅ Normal operation unaffected (no false triggers)

**Design principle**: ODrive watchdog is a **last-resort backstop** for catastrophic failures (app crash, hang, CAN failure), not the primary safety mechanism.
