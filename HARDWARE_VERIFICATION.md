# Hardware Verification Guide - WheelBase Command-Stale Safety

## Pre-Verification Setup

1. **Safety first**: Conduct tests in safe area or with robot on blocks
2. **Required equipment**: 
   - Robot Kevin (or test robot) with ODrive S1 wheels
   - SSH access or direct terminal
   - Clear area of at least 2m for movement tests

## Verification Test Suite

### Test 1: Stale Command Detection (PRIMARY TEST)

**Objective**: Verify command-stale safety triggers at 0.5s, not 2.0s watchdog.

**Setup**:
```bash
cd /path/to/anglerdroid
export CAN_IFACE=can1  # or can0, depending on your robot
python3 test_stale_command_safety.py
```

**Expected output**:
```
[t=0.0s] TEST: Sending one-shot velocity command (0.3 m/s forward)...
         Last commanded velocity sent to motors.
         Now STOPPING command stream (simulating one-shot call)...

Monitoring (watching for safety trigger):
----------------------------------------------------------------------
[t= 0.0s] vel_L=+0.00 vel_R=+0.00 m/s  |  state=CLOSED_LOOP  idle=False  |  cmd_age= 0.0s
[t= 0.5s] vel_L=+0.28 vel_R=+0.28 m/s  |  state=CLOSED_LOOP  idle=False  |  cmd_age= 0.5s
⚠️  SAFETY: Command stale (0.51s), zeroing velocity (L=0.50, R=0.50)
[t= 1.0s] vel_L=+0.00 vel_R=+0.00 m/s  |  state=CLOSED_LOOP  idle=False  |  cmd_age= 1.0s
[t= 5.5s] vel_L=+0.00 vel_R=+0.00 m/s  |  state=CLOSED_LOOP  idle=False  |  cmd_age= 5.5s
Flat ground (torque L=0.123 R=0.098 Nm), freewheeling
[t= 6.0s] vel_L=+0.00 vel_R=+0.00 m/s  |  state=IDLE  idle=True  |  cmd_age= 6.0s
```

**Pass criteria**:
- ✅ Safety message appears around t=0.5-0.6s
- ✅ Velocity drops to zero within 0.1s after safety trigger
- ✅ NO red LED on ODrive (watchdog did NOT trip)
- ✅ Robot stops smoothly, no abrupt disarm
- ✅ Motors transition to IDLE around t=5-6s

**Fail criteria**:
- ❌ Robot continues moving past t=1.0s
- ❌ Red LED flash at t=2.0s (watchdog trip)
- ❌ Error message: `WATCHDOG_TIMER_EXPIRED` (0x01000000)
- ❌ Safety message never appears

---

### Test 2: Gamepad Normal Operation (FALSE-TRIGGER CHECK)

**Objective**: Verify gamepad driving does NOT false-trigger safety during normal use.

**Setup**:
```bash
cd /path/to/anglerdroid/src
python3 wheelbase.py  # runs gamepad demo
```

**Test procedure**:
1. Move right joystick forward → robot drives
2. Hold forward for 5 seconds continuously
3. Release joystick → robot stops
4. Wait 10 seconds (observe idle transition)
5. Repeat with turning, backward motion

**Expected behavior**:
- Robot responds smoothly to joystick
- NO safety messages during continuous driving
- Clean stop when joystick released
- Idle transition after 5s of zero velocity

**Pass criteria**:
- ✅ NO "SAFETY: Command stale..." messages during 5s forward drive
- ✅ Smooth acceleration and deceleration
- ✅ Robot stops immediately when joystick released (not at 0.5s delay)
- ✅ Idle transition after 5s of holding zero

**Fail criteria**:
- ❌ Safety messages during normal driving
- ❌ Robot stutters or stops unexpectedly mid-drive
- ❌ False safety triggers with joystick held

---

### Test 3: `twist_for()` Scripted Moves (INTEGRATION TEST)

**Objective**: Verify scripted moves complete normally without false safety triggers.

**Setup**:
```python
cd /path/to/anglerdroid
python3
>>> import sys
>>> sys.path.insert(0, 'src')
>>> from wheelbase import WheelBase
>>> wb = WheelBase()
```

**Test procedure**:
```python
# Test 1: Short move (2s)
>>> wb.twist_for(0.3, 0.0, duration_secs=2.0, ramp_in_secs=0.5, ramp_out_secs=0.5)
>>> import time
>>> time.sleep(3.0)

# Test 2: Long move (5s)
>>> wb.twist_for(0.3, 0.0, duration_secs=5.0, ramp_in_secs=1.0, ramp_out_secs=1.0)
>>> time.sleep(6.0)

# Test 3: Turn in place
>>> wb.twist_for(0.0, 1.0, duration_secs=3.0)
>>> time.sleep(4.0)

>>> wb.shutdown()
```

**Expected behavior**:
- Each move completes smoothly
- NO safety messages during moves
- Robot stops automatically at end of each move
- Idle transition after each move completes

**Pass criteria**:
- ✅ All three moves complete without interruption
- ✅ NO "SAFETY: Command stale..." during moves
- ✅ Clean stop at end of each duration
- ✅ Idle transition after each move

**Fail criteria**:
- ❌ Safety triggers during move (should not happen)
- ❌ Move interrupted before duration expires
- ❌ Robot continues moving after duration

---

### Test 4: Incline Hold (EDGE CASE)

**Objective**: Verify robot holds position on incline after stale command zeros velocity.

**Setup**: Place robot on 10-15° incline (or test ramp)

**Test procedure**:
```python
from wheelbase import WheelBase
wb = WheelBase()

# Send single zero command (simulates "stop and hold" intent)
wb.twist(0.0, 0.0)

# Wait and observe for 10 seconds
import time
time.sleep(10.0)

wb.shutdown()
```

**Expected behavior**:
- t=0.0s: Robot commanded to hold position (zero velocity)
- t=0.5s: Safety triggers: "Command stale..." (zeros velocity again)
- t=5.5s: Idle watcher checks torque
- t=5.5s: "Incline detected (torque L=X.XXX R=X.XXX Nm), holding position"
- t=10.0s: Robot STILL holding position (closed-loop control)

**Pass criteria**:
- ✅ Robot does NOT roll down incline
- ✅ Safety message appears at t=0.5s
- ✅ "Incline detected" message appears at t=5-6s
- ✅ Motors remain in closed-loop (NOT idle)
- ✅ No watchdog trip

**Fail criteria**:
- ❌ Robot rolls down incline (torque check failed)
- ❌ Motors idle on incline (unsafe freewheel)
- ❌ Watchdog trips

---

### Test 5: One-Shot API Call (REGRESSION TEST)

**Objective**: Verify the ORIGINAL PROBLEM is fixed (one-shot command stops safely).

**Setup**: 
```bash
cd /path/to/anglerdroid
python3
```

**Test procedure**:
```python
import sys
sys.path.insert(0, 'src')
from wheelbase import WheelBase
import time

wb = WheelBase()
time.sleep(1.0)

print("Sending one-shot command: 0.5 m/s forward")
wb.twist(0.5, 0.0)

print("NOT sending any more commands. Watching for 10 seconds...")
start = time.time()
while time.time() - start < 10.0:
    time.sleep(1.0)
    vl, vr = wb.get_wheel_velocities_mps()
    elapsed = time.time() - start
    print(f"t={elapsed:.1f}s: vel_L={vl:.2f} vel_R={vr:.2f}")

wb.shutdown()
```

**Expected output**:
```
Sending one-shot command: 0.5 m/s forward
NOT sending any more commands. Watching for 10 seconds...
t=1.0s: vel_L=0.48 vel_R=0.48
⚠️  SAFETY: Command stale (0.51s), zeroing velocity (L=0.83, R=0.83)
t=2.0s: vel_L=0.00 vel_R=0.00
t=3.0s: vel_L=0.00 vel_R=0.00
Flat ground (torque L=0.089 R=0.102 Nm), freewheeling
t=7.0s: vel_L=0.00 vel_R=0.00
```

**Pass criteria**:
- ✅ Safety triggers at ~0.5-0.6s after command
- ✅ Velocity drops to zero by t=1.0s
- ✅ NO red LED / watchdog trip
- ✅ Idle transition at t=5-6s

**Fail criteria**:
- ❌ Robot continues at 0.5 m/s past t=1.0s
- ❌ Watchdog trips at t=2.0s
- ❌ No safety message

---

## Error Checking

After each test, check ODrive error state:

```python
wb.shutdown()  # prints error report
```

**Expected shutdown output (healthy)**:
```
ODrive error check:
   Left (node 0): no errors
   Right (node 1): no errors
WheelBase shutdown complete
```

**FAILURE indicators**:
```
   Left (node 0): active_errors=0x01000000, disarm_reason=0x01000000
```

This indicates `WATCHDOG_TIMER_EXPIRED` - the fix did NOT work. Report immediately.

---

## Verification Checklist

Complete this checklist before approving the PR:

- [ ] **Test 1**: Stale command detection works (0.5s trigger) ✅
- [ ] **Test 2**: Gamepad normal operation (no false triggers) ✅
- [ ] **Test 3**: `twist_for()` scripted moves work normally ✅
- [ ] **Test 4**: Incline hold behavior preserved ✅
- [ ] **Test 5**: One-shot API call problem is FIXED ✅
- [ ] **Error check**: NO watchdog trips in any test ✅
- [ ] **Battery**: Robot operated at reasonable battery level (>30%) ✅
- [ ] **Documentation**: WHEELBASE_SAFETY.md reviewed ✅

---

## Troubleshooting

### Safety triggers too early (< 0.5s)

**Possible causes**:
- System clock issues (check `date`)
- Thread starvation (check CPU load)

**Action**: Increase `COMMAND_STALE_TIMEOUT` to 0.7s for testing margin

### Safety never triggers

**Possible causes**:
- Thread not started (check for `_start_stale_command_watcher` call)
- Wrong branch deployed
- Python exceptions in watcher loop (check for error prints)

**Action**: Check console for startup messages, verify code deployment

### Watchdog still trips

**Possible causes**:
- Safety trigger is working but too slow
- CAN bus congestion delaying zero command
- Thread scheduling delays

**Action**: Increase ODrive watchdog to 3.0s for debugging margin

### False triggers during gamepad driving

**Possible causes**:
- Gamepad update rate < 2 Hz (unlikely if main loop is 30 Hz)
- Main loop blocked on slow operation

**Action**: Add debug logging to `set_wheel_vels()` to track call rate

---

## Reporting Results

When verification is complete, report in PR #1:

```
Hardware verification complete on robot [Kevin/test-robot-name]:

✅ Test 1: Stale command detected at 0.53s, robot stopped cleanly
✅ Test 2: Gamepad operation normal, no false triggers
✅ Test 3: twist_for() moves completed smoothly
✅ Test 4: Incline hold working, torque check at 5.2s
✅ Test 5: One-shot API call FIXED, robot stopped at 0.51s

Error state: No watchdog trips, no errors on shutdown.
Battery level during tests: 75%

Ready to merge.
```

Or report any failures:

```
❌ Test 1 FAILED: [description of what went wrong]

[Include error logs, console output, and observations]
```
