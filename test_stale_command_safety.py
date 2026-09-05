#!/usr/bin/env python3
"""
Test script for WheelBase command-stale safety mechanism.

This script verifies that the safety fix prevents a one-shot velocity command
from keeping the robot driving until the ODrive watchdog expires.

Expected behavior WITH the fix:
1. Set non-zero velocity once
2. Stop sending commands
3. After 0.5s (COMMAND_STALE_TIMEOUT), WheelBase automatically:
   - Zeros velocity
   - Disables watchdog
   - Transitions to IDLE state
   - Clears any errors
4. Safety message printed: "⚠️  SAFETY: Command stale..."
5. Robot finishes in clean IDLE state (blue LED, disarm_reason=0)
6. ODrive watchdog NEVER trips

Expected behavior WITHOUT the fix (old code):
1. Set non-zero velocity once
2. Stop sending commands
3. Robot continues driving with last velocity
4. After 2.0s, ODrive watchdog expires → red LED / disarm

To run this test:
1. Ensure robot is on blocks or in safe test area
2. python3 test_stale_command_safety.py
3. Observe console output and robot behavior
4. Verify robot stops within ~0.5-0.6s, not 2.0s

SAFETY: This test intentionally triggers the stale-command safety mechanism.
The robot will move briefly (0.5s) then stop automatically.
"""

import os
import sys
import time

# Set CAN interface from env or default
CAN_IFACE = os.environ.get('CAN_IFACE', 'can0')

# Must set before importing wheelbase
os.environ.setdefault('CAN_IFACE', CAN_IFACE)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wheelbase import WheelBase


def test_stale_command_safety():
    """Test that stale commands are caught and zeroed before ODrive watchdog trips."""
    
    print("=" * 70)
    print("WheelBase Stale-Command Safety Test")
    print("=" * 70)
    print()
    print("This test verifies the command-stale safety mechanism:")
    print("1. Send a single non-zero velocity command")
    print("2. Stop sending further commands")
    print("3. Verify robot stops within 0.5s (not 2.0s watchdog timeout)")
    print()
    print("Expected timeline:")
    print("  t=0.0s : Send 0.3 m/s forward command")
    print("  t=0.5s : SAFETY triggers, zeros velocity, disables watchdog,")
    print("           transitions to IDLE, clears errors (blue LED)")
    print("  t=never: ODrive watchdog should NOT trip")
    print()
    print("SAFETY: Robot will move briefly. Ensure safe test area.")
    print()
    input("Press ENTER to start test (Ctrl+C to abort)...")
    print()
    
    # Initialize wheelbase
    print(f"Initializing WheelBase on {CAN_IFACE}...")
    wb = WheelBase(can_interface=CAN_IFACE)
    print()
    
    try:
        # Wait for initialization to complete
        time.sleep(1.0)
        
        # Test 1: Send one-shot non-zero velocity
        print("[t=0.0s] TEST: Sending one-shot velocity command (0.3 m/s forward)...")
        wb.twist(0.3, 0.0)  # 0.3 m/s forward, 0 rad/s turn
        print("         Last commanded velocity sent to motors.")
        print("         Now STOPPING command stream (simulating one-shot call)...")
        print()
        
        # Monitor for 8 seconds to observe the safety behavior
        start_time = time.time()
        last_status_time = start_time
        
        print("Monitoring (watching for safety trigger):")
        print("-" * 70)
        
        while time.time() - start_time < 8.0:
            now = time.time()
            elapsed = now - start_time
            
            # Print status every 0.5s
            if now - last_status_time >= 0.5:
                vl, vr = wb.get_wheel_velocities_mps()
                closed_loop = "CLOSED_LOOP" if wb._is_closed_loop else "IDLE"
                is_idle = wb._is_idle
                
                # Check if we've seen the safety trigger
                time_since_cmd = now - wb._last_command_time
                
                print(f"[t={elapsed:4.1f}s] vel_L={vl:+5.2f} vel_R={vr:+5.2f} m/s  |  "
                      f"state={closed_loop}  idle={is_idle}  |  "
                      f"cmd_age={time_since_cmd:4.1f}s")
                
                last_status_time = now
            
            time.sleep(0.1)
        
        print("-" * 70)
        print()
        
        # Check axis state and errors
        print("Checking axis state and errors:")
        print("-" * 70)
        try:
            with wb.bus_lock:
                left_state = wb.left.read_property('axis0.current_state')
                left_disarm = wb.left.read_property('axis0.disarm_reason')
                left_active = wb.left.read_property('axis0.active_errors')
            
            with wb.bus_lock:
                right_state = wb.right.read_property('axis0.current_state')
                right_disarm = wb.right.read_property('axis0.disarm_reason')
                right_active = wb.right.read_property('axis0.active_errors')
            
            print(f"Left axis:  state={left_state} (1=IDLE), disarm_reason=0x{left_disarm:08X}, active_errors=0x{left_active:08X}")
            print(f"Right axis: state={right_state} (1=IDLE), disarm_reason=0x{right_disarm:08X}, active_errors=0x{right_active:08X}")
            print()
            
            # Check for success
            all_idle = (left_state == 1 and right_state == 1)
            no_disarm = (left_disarm == 0 and right_disarm == 0)
            no_errors = (left_active == 0 and right_active == 0)
            
            if all_idle and no_disarm and no_errors:
                print("✅ SUCCESS: Axes in clean IDLE state (blue LED, no watchdog fault)")
            else:
                print("❌ FAILURE: Axes not in clean state")
                if not all_idle:
                    print("   → Axes not in IDLE state")
                if not no_disarm:
                    print("   → Watchdog fault latched (red LED)")
                    if left_disarm & 0x01000000:
                        print("      Left: WATCHDOG_TIMER_EXPIRED")
                    if right_disarm & 0x01000000:
                        print("      Right: WATCHDOG_TIMER_EXPIRED")
                if not no_errors:
                    print("   → Active errors present")
        except Exception as e:
            print(f"Could not read axis state: {e}")
        
        print("-" * 70)
        print()
        print("Test complete!")
        print()
        print("RESULTS:")
        print("--------")
        print("If the safety fix is working:")
        print("  ✓ You should see 'SAFETY: Command stale...' message around t=0.5s")
        print("  ✓ Velocity should drop to zero around t=0.5-0.6s")
        print("  ✓ State should transition CLOSED_LOOP → IDLE immediately at t=0.5s")
        print("  ✓ NO red LED / ODrive disarm should occur")
        print("  ✓ disarm_reason should remain 0 (no watchdog fault)")
        print("  ✓ Final check should show '✅ SUCCESS' with clean IDLE state")
        print()
        print("If the safety fix is NOT working (old behavior):")
        print("  ✗ Robot would continue moving until t=2.0s")
        print("  ✗ ODrive watchdog would trip (red LED / disarm)")
        print("  ✗ You would see disarm_reason=0x01000000 (WATCHDOG_TIMER_EXPIRED)")
        print("  ✗ Final check would show '❌ FAILURE'")
        print()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down wheelbase...")
        wb.shutdown()
        print("Test cleanup complete.")


if __name__ == "__main__":
    test_stale_command_safety()
