"""
wheelbase.py - Final production WheelBase class
===============================================
Wheelbase = 34 cm
Wheel diameter = 17.13 cm
Left wheel inverted by default

Gamepad capped at 50% speed + deadzone for clean zero
Auto mode commented out (easy to re-enable)
"""

import json
import time
import threading
import subprocess
from typing import Optional

import glob as globmod

import can
import inputs
import simplegamepad
from odrivecan import ODriveAxisCAN
from simplegamepad import SimpleGamepad, haveGamepad
from robot_config import WHEEL_DIAMETER_CM, WHEELBASE_CM


class WheelBase:
    # ── Safety timing constraints ────────────────────────────────────
    # CRITICAL: COMMAND_STALE_TIMEOUT must be strictly less than the ODrive
    # watchdog timeout to ensure the application zeros velocity before the
    # hardware watchdog trips. The watchdog is a last-resort backstop, not
    # the primary safety mechanism.
    #
    # Timing hierarchy (increasing):
    #   1. COMMAND_STALE_TIMEOUT (0.5s) - app detects stale non-zero command
    #   2. ODrive watchdog (2.0s) - hardware backstop if app fails
    #   3. IDLE_ZERO_TIMEOUT (5.0s default) - motors idle after holding zero
    #
    # Margin: 1.5s between command-stale and watchdog provides robust safety
    # buffer for the app to stop the robot gracefully before hardware disarm.
    COMMAND_STALE_TIMEOUT = 0.5     # seconds - app-level command staleness
    WATCHDOG_FEED_INTERVAL = 1.0    # seconds - keep-alive feed rate
    INCLINE_TORQUE_THRESHOLD = 1.0  # Nm — above this while holding zero = incline
    VEL_SEND_DELTA = 0.02
    TWIST_FOR_INTERVAL = 0.1  # 10 Hz

    def __init__(self,
                 can_interface: str = "can0",
                 wheel_diameter_cm: float = WHEEL_DIAMETER_CM,
                 wheelbase_cm: float = WHEELBASE_CM,
                 idle_zero_timeout_s: float = 5.0,
                 invert_left: bool = True):
        self.can_interface = can_interface
        self.wheel_diameter_cm = wheel_diameter_cm
        self.wheelbase_cm = wheelbase_cm
        self.idle_zero_timeout_s = idle_zero_timeout_s
        self.invert_left = invert_left

        self.wheel_radius_m = wheel_diameter_cm / 200.0
        self.wheelbase_m = wheelbase_cm / 100.0

        self.bus = None
        self.bus_lock = threading.Lock()
        self.left = None
        self.right = None
        self.gamepad: Optional[SimpleGamepad] = None

        self.running = True
        self.idle_thread = None
        self.stale_command_thread = None
        self._is_closed_loop = False
        self._is_idle = True
        self._last_sent_left = None
        self._last_sent_right = None
        self._last_send_time = 0.0
        self._last_command_time = 0.0  # tracks when set_wheel_vels() was last called
        self._zero_vel_since = None
        self._twist_for_lock = threading.Lock()
        self._twist_for_params = None  # (forward_mps, angular_rads, duration_secs, ramp_in_secs, ramp_out_secs, start_time)
        self._safety_fwd = 1.0
        self._safety_bwd = 1.0
        self._safety_ang = 1.0

        print("WheelBase: bring_up_can...")
        self._bring_up_can()
        print("WheelBase: init_odrive...")
        self._init_odrive()
        print("WheelBase: init_gamepad...")
        self._init_gamepad()
        self._start_idle_watcher()
        self._start_stale_command_watcher()
        self._start_twist_for_thread()

        vbus = 0.0
        try:
            with self.bus_lock:
                vbus = self.left.get_vbus_voltage()
        except Exception:
            pass
        batt = int(100 * (vbus - 30.0) / (42.0 - 30.0)) if vbus > 0 else 0
        batt = max(0, min(100, batt))
        print(f"✅ WheelBase ready (wheelbase={wheelbase_cm}cm, max_speed=50%) | Battery: {vbus:.1f}V ({batt}%)")

        self._start_encoder_reader()

    def _bring_up_can(self):
        try:
            result = subprocess.run(["ip", "-d", "link", "show", self.can_interface],
                                    capture_output=True, text=True, timeout=2)
            if "state UP" in result.stdout:
                print(f"   {self.can_interface} is already UP")
                return
        except Exception:
            pass

        print(f"   Bringing up {self.can_interface} (may ask for sudo password)...")
        try:
            subprocess.run(
                ["sudo", "ip", "link", "set", self.can_interface,
                 "up", "type", "can", "bitrate", "250000"],
                check=True)
            print(f"   {self.can_interface} brought up")
        except subprocess.CalledProcessError:
            print(f"   WARNING: Could not auto-bring up {self.can_interface}")

    def _init_odrive(self):
        self.bus = can.Bus(self.can_interface, interface="socketcan")
        with open('flat_endpoints.json', 'r') as f:
            endpoints = json.load(f)['endpoints']

        self.left = ODriveAxisCAN(self.bus, 0, endpoints)
        self.right = ODriveAxisCAN(self.bus, 1, endpoints)

        for axis in (self.left, self.right):
            axis.clear_errors()
            axis.disable_watchdog()

        self.left.set_axis_state(ODriveAxisCAN.AXIS_STATE_IDLE)
        self.right.set_axis_state(ODriveAxisCAN.AXIS_STATE_IDLE)
        time.sleep(0.3)

        self.left.set_axis_state(ODriveAxisCAN.AXIS_STATE_CLOSED_LOOP_CONTROL)
        self.right.set_axis_state(ODriveAxisCAN.AXIS_STATE_CLOSED_LOOP_CONTROL)
        time.sleep(0.3)

        for axis in (self.left, self.right):
            axis.set_vel_ramp_rate(3.0)
            axis.set_vel_limit(4.0)

        for _ in range(20):
            self.left.set_velocity(0.0)
            self.right.set_velocity(0.0)
            time.sleep(0.02)

        # Motors verified — return to idle until first real command.
        # set_wheel_vels() handles re-engage + watchdog on first non-zero velocity.
        for axis in (self.left, self.right):
            axis.set_velocity(0.0)
            axis.disable_watchdog()
            axis.set_axis_state(ODriveAxisCAN.AXIS_STATE_IDLE)

    def _init_gamepad(self):
        if haveGamepad():
            self.gamepad = SimpleGamepad()
            print("   Gamepad connected → manual control (max 50% speed)")
        else:
            print("   No gamepad detected — will poll every 3s")
        self._start_gamepad_poll()

    def _start_gamepad_poll(self):
        t = threading.Thread(target=self._gamepad_poll_loop, daemon=True)
        t.start()

    @staticmethod
    def _find_joystick_event_path():
        """Find a joystick event device path from by-id or by-path."""
        for p in globmod.glob('/dev/input/by-id/*-event-joystick'):
            return p
        for p in globmod.glob('/dev/input/by-path/*-event-joystick'):
            return p
        return None

    def _gamepad_poll_loop(self):
        while self.running:
            if self.gamepad is not None:
                time.sleep(1.0)
                continue
            time.sleep(3.0)
            if not self.running:
                break
            try:
                dev_path = self._find_joystick_event_path()
                if not dev_path:
                    continue
                inputs.devices.gamepads.clear()
                gp_device = inputs.GamePad(inputs.devices, dev_path)
                inputs.devices.gamepads.append(gp_device)
                self.gamepad = SimpleGamepad()
                print(f"Gamepad connected ({gp_device.name}) → manual control (max 50% speed)")
            except Exception as e:
                print(f"Gamepad poll error: {e}")
                self.gamepad = None

    def _check_gamepad_health(self):
        """Check if gamepad is still working; clear it if dead so poll loop reconnects."""
        if self.gamepad is None:
            return
        if simplegamepad._monitorError:
            print(f"Gamepad lost ({simplegamepad._monitorError}), will reconnect...")
            inputs.devices.gamepads.clear()
            self.gamepad = None

    def _start_idle_watcher(self):
        self.idle_thread = threading.Thread(target=self._idle_watcher_loop, daemon=True)
        self.idle_thread.start()

    def _idle_watcher_loop(self):
        while self.running:
            if (self._zero_vel_since is not None
                    and not self._is_idle
                    and self._is_closed_loop
                    and time.time() - self._zero_vel_since >= self.idle_zero_timeout_s):
                self._try_idle_with_torque_check()
            time.sleep(0.5)

    def _start_stale_command_watcher(self):
        """Start monitoring thread for command staleness detection.
        
        This is a critical safety feature: if we're in closed-loop mode with
        non-zero commanded velocity and no new motion command arrives within
        COMMAND_STALE_TIMEOUT, we immediately zero velocity and feed the
        watchdog to bring the robot to a safe stop BEFORE the ODrive hardware
        watchdog would trip.
        """
        self.stale_command_thread = threading.Thread(
            target=self._stale_command_watcher_loop, daemon=True)
        self.stale_command_thread.start()

    def _stale_command_watcher_loop(self):
        """Monitor for stale non-zero velocity commands.
        
        Safety invariant: COMMAND_STALE_TIMEOUT < ODrive watchdog timeout.
        If a non-zero velocity is latched but no fresh command arrives within
        COMMAND_STALE_TIMEOUT, immediately command zero velocity (and continue
        feeding watchdog briefly) to stop the robot before ODrive watchdog trips.
        
        This prevents the scenario where:
        1. Someone calls set_wheel_vels(non_zero) once
        2. No further commands arrive
        3. Robot drives until ODrive watchdog expires (red LED / disarm)
        
        Instead:
        1. Command goes stale after 0.5s
        2. App zeros velocity (continues feeding watchdog)
        3. Existing idle logic moves to IDLE after zero-timeout
        4. ODrive watchdog never trips
        """
        while self.running:
            now = time.time()
            
            # Check conditions for stale command detection
            if (self._is_closed_loop 
                    and not self._is_idle
                    and self._last_command_time > 0):
                
                # Are we commanding non-zero velocity?
                has_nonzero_vel = False
                if (self._last_sent_left is not None 
                        and self._last_sent_right is not None):
                    has_nonzero_vel = (abs(self._last_sent_left) >= 0.01 
                                       or abs(self._last_sent_right) >= 0.01)
                
                # Has the command gone stale?
                time_since_command = now - self._last_command_time
                if has_nonzero_vel and time_since_command >= self.COMMAND_STALE_TIMEOUT:
                    print(f"⚠️  SAFETY: Command stale ({time_since_command:.2f}s), "
                          f"zeroing velocity (L={self._last_sent_left:.2f}, "
                          f"R={self._last_sent_right:.2f})")
                    
                    # Immediately zero velocity and feed watchdog
                    try:
                        with self.bus_lock:
                            self.left.set_velocity(0.0)
                            self.right.set_velocity(0.0)
                            self.left.feed_watchdog()
                            self.right.feed_watchdog()
                        self._last_sent_left = 0.0
                        self._last_sent_right = 0.0
                        self._last_send_time = now
                        
                        # Mark as zero velocity so idle watcher can take over
                        if self._zero_vel_since is None:
                            self._zero_vel_since = now
                    except can.CanOperationError as e:
                        print(f"Warning during stale-command safety stop: {e}")
            
            # Check frequently (5 Hz) for responsive safety shutoff
            time.sleep(0.2)

    def _try_idle_with_torque_check(self):
        try:
            with self.bus_lock:
                self.left.set_velocity(0.0)
                self.right.set_velocity(0.0)
                self._last_send_time = time.time()
                torque_l = self.left.read_property('axis0.motor.torque_estimate')

            with self.bus_lock:
                self.left.set_velocity(0.0)
                self.right.set_velocity(0.0)
                self._last_send_time = time.time()
                torque_r = self.right.read_property('axis0.motor.torque_estimate')

            if torque_l is None or torque_r is None:
                print("Could not read torque, staying in closed loop")
                return

            if abs(torque_l) > self.INCLINE_TORQUE_THRESHOLD or abs(torque_r) > self.INCLINE_TORQUE_THRESHOLD:
                print(f"Incline detected (torque L={torque_l:.3f} R={torque_r:.3f} Nm), holding position")
                self._zero_vel_since = time.time()
                return

            with self.bus_lock:
                self.left.set_velocity(0.0)
                self.right.set_velocity(0.0)
                self.left.disable_watchdog()
                self.right.disable_watchdog()
                self.left.set_axis_state(ODriveAxisCAN.AXIS_STATE_IDLE)
                self.right.set_axis_state(ODriveAxisCAN.AXIS_STATE_IDLE)
            self._is_closed_loop = False
            self._is_idle = True
            print(f"Flat ground (torque L={torque_l:.3f} R={torque_r:.3f} Nm), freewheeling")
        except can.CanOperationError as e:
            print(f"Warning during idle transition: {e}")

    def twist(self, forward_mps: float, angular_rads: float):
        """Differential drive: forward m/s, angular rad/s (instant)."""
        v_l = forward_mps - (angular_rads * self.wheelbase_m / 2)
        v_r = forward_mps + (angular_rads * self.wheelbase_m / 2)
        left_tps = v_l / (2 * 3.1415926535 * self.wheel_radius_m)
        right_tps = v_r / (2 * 3.1415926535 * self.wheel_radius_m)
        self.set_wheel_vels(left_tps, right_tps)

    def _start_twist_for_thread(self):
        self._twist_for_thread = threading.Thread(target=self._twist_for_loop, daemon=True)
        self._twist_for_thread.start()

    def _twist_for_loop(self):
        """10 Hz loop: run twist_for profile; new call overrides previous."""
        while self.running:
            t0 = time.monotonic()
            with self._twist_for_lock:
                params = self._twist_for_params
            if params is None:
                time.sleep(self.TWIST_FOR_INTERVAL)
                continue
            forward_mps, angular_rads, duration_secs, ramp_in_secs, ramp_out_secs, start_time = params
            elapsed = time.monotonic() - start_time
            if elapsed >= duration_secs:
                with self._twist_for_lock:
                    self._twist_for_params = None
                self.stop()
                time.sleep(self.TWIST_FOR_INTERVAL)
                continue
            if elapsed < ramp_in_secs and ramp_in_secs > 0:
                frac = elapsed / ramp_in_secs
                fwd = forward_mps * frac
            elif elapsed >= duration_secs - ramp_out_secs and ramp_out_secs > 0:
                ramp_out_elapsed = elapsed - (duration_secs - ramp_out_secs)
                frac = 1.0 - ramp_out_elapsed / ramp_out_secs
                fwd = forward_mps * max(0.0, frac)
            else:
                fwd = forward_mps
            self.twist(fwd, angular_rads)
            time.sleep(max(0, self.TWIST_FOR_INTERVAL - (time.monotonic() - t0)))

    def twist_for(self, forward_mps: float, angular_rads: float,
                  duration_secs: float = 2.0, ramp_in_secs: float = 1.0, ramp_out_secs: float = 1.0):
        """
        Timed differential drive: forward m/s and angular rad/s for duration_secs.
        Ramp in: forward velocity 0 → target over ramp_in_secs (angular at target from start).
        Ramp out: forward velocity target → 0 over ramp_out_secs (angular constant until end).
        Runs on 10 Hz timer. New call overrides any in-progress twist_for.
        """
        with self._twist_for_lock:
            self._twist_for_params = (
                float(forward_mps), float(angular_rads),
                float(duration_secs), float(ramp_in_secs), float(ramp_out_secs),
                time.monotonic(),
            )

    def is_twist_for_active(self) -> bool:
        """True while a twist_for profile is running."""
        with self._twist_for_lock:
            return self._twist_for_params is not None

    def set_safety_scales(self, fwd, bwd, ang):
        self._safety_fwd = max(0.0, min(1.0, float(fwd)))
        self._safety_bwd = max(0.0, min(1.0, float(bwd)))
        self._safety_ang = max(0.0, min(1.0, float(ang)))

    def cancel_twist_for(self):
        """Cancel any in-progress twist_for (e.g. when gamepad takes over)."""
        with self._twist_for_lock:
            self._twist_for_params = None

    def get_wheel_velocities_mps(self):
        """Return (v_left, v_right) in m/s from encoder feedback.

        Reads are done by a background thread at ~30 Hz via native CAN
        Get_Encoder_Estimates (fast) with SDO fallback.  This method just
        returns the latest cached value — zero-cost for the caller.

        Falls back to commanded velocities if encoder reads are stale
        (>300 ms since last successful read) or never started.
        """
        circ = 2.0 * 3.1415926535 * self.wheel_radius_m
        with self._enc_lock:
            if self._enc_ok:
                age = time.monotonic() - self._enc_last_good
                if age < 0.3:
                    vl_raw = self._enc_vel[0]
                    vr_raw = self._enc_vel[1]
                    vl = (-vl_raw if self.invert_left else vl_raw) * circ
                    vr = vr_raw * circ
                    return vl, vr
        # Fallback: commanded velocity
        sl = self._last_sent_left
        sr = self._last_sent_right
        if sl is None or sr is None:
            return 0.0, 0.0
        l_tps = -sl if self.invert_left else sl
        r_tps = sr
        return l_tps * circ, r_tps * circ

    # ── Encoder reader thread ────────────────────────────────────────

    @property
    def battery_pct(self):
        """Cached battery percentage (updated every ~5s in encoder thread)."""
        return self._batt_pct

    def _start_encoder_reader(self):
        self._enc_vel = [0.0, 0.0]    # [left_tps, right_tps] from encoder
        self._enc_lock = threading.Lock()
        self._enc_ok = False
        self._enc_native = True        # try native CAN 0x09 first
        self._enc_native_fails = 0
        self._enc_consec_fails = 0     # consecutive read failures (any mode)
        self._enc_last_good = 0.0      # time.monotonic() of last successful read
        self._batt_pct = -1            # unknown until first read
        self._batt_next = 0.0          # read immediately on first loop
        t = threading.Thread(target=self._encoder_reader_loop, daemon=True)
        t.start()

    def _encoder_reader_loop(self):
        _sdo_fail_count = 0
        while self.running:
            vl = vr = None
            try:
                if self._enc_native:
                    with self.bus_lock:
                        vl = self.left.get_encoder_vel_fast()
                        vr = self.right.get_encoder_vel_fast()
                    if vl is None or vr is None:
                        self._enc_native_fails += 1
                        if self._enc_native_fails >= 3:
                            self._enc_native = False
                            print("encoder: native CAN 0x09 not supported, "
                                  "using SDO fallback")
                    else:
                        self._enc_native_fails = 0

                if not self._enc_native:
                    with self.bus_lock:
                        vl = self.left.get_encoder_vel_sdo()
                    with self.bus_lock:
                        vr = self.right.get_encoder_vel_sdo()
                    if vl is None or vr is None:
                        _sdo_fail_count += 1
                        if _sdo_fail_count in (5, 50, 500):
                            print("encoder: SDO read failed %d times "
                                  "(vl=%s vr=%s)" % (_sdo_fail_count, vl, vr))

                if vl is not None and vr is not None:
                    with self._enc_lock:
                        self._enc_vel[0] = vl
                        self._enc_vel[1] = vr
                        self._enc_last_good = time.monotonic()
                        self._enc_consec_fails = 0
                        if not self._enc_ok:
                            self._enc_ok = True
                            mode = "native CAN" if self._enc_native else "SDO"
                            print("encoder: reading actual wheel velocities "
                                  "(%s)" % mode)
                    _sdo_fail_count = 0
                else:
                    with self._enc_lock:
                        self._enc_consec_fails += 1
                        if (self._enc_ok and
                                self._enc_consec_fails >= 10):
                            self._enc_ok = False
                            print("encoder: reads failing, falling back to "
                                  "commanded velocity")
            except Exception as e:
                if self._enc_consec_fails < 3:
                    print("encoder: exception: %s" % e)

            now = time.monotonic()
            if now >= self._batt_next:
                try:
                    with self.bus_lock:
                        v = self.left.get_vbus_voltage()
                    if v > 0:
                        self._batt_pct = max(0, min(100,
                            int(100 * (v - 30.0) / (42.0 - 30.0))))
                except Exception:
                    pass
                self._batt_next = now + 5.0

            time.sleep(0.030)

    def set_wheel_vels(self, left_tps: float, right_tps: float):
        """Direct wheel control (turns/s). Safety-scaled, deduplicates, manages idle.
        
        This method is the ONLY entry point for velocity commands. Every call
        updates _last_command_time, which is monitored by the stale-command
        watcher to ensure continuous command flow during motion.
        """
        # Record this command timestamp for staleness monitoring
        self._last_command_time = time.time()
        
        fwd = (left_tps + right_tps) / 2.0
        turn = (right_tps - left_tps) / 2.0
        if fwd > 0:
            fwd *= self._safety_fwd
        elif fwd < 0:
            fwd *= self._safety_bwd
        turn *= self._safety_ang
        left_tps = fwd - turn
        right_tps = fwd + turn

        is_zero = abs(left_tps) < 0.01 and abs(right_tps) < 0.01

        if is_zero:
            left_tps = right_tps = 0.0
            if self._zero_vel_since is None:
                self._zero_vel_since = time.time()
            if self._is_idle:
                return
        else:
            self._zero_vel_since = None
            if not self._is_closed_loop:
                print("Re-engaging motors from idle...")
                with self.bus_lock:
                    self.left.clear_errors()
                    self.right.clear_errors()
                    self.left.set_axis_state(ODriveAxisCAN.AXIS_STATE_CLOSED_LOOP_CONTROL)
                    self.right.set_axis_state(ODriveAxisCAN.AXIS_STATE_CLOSED_LOOP_CONTROL)
                    self.left.set_velocity(0.0)
                    self.right.set_velocity(0.0)
                    self.left.enable_watchdog(2.0)
                    self.right.enable_watchdog(2.0)
                self._is_closed_loop = True
                self._is_idle = False
                self._last_sent_left = None
                self._last_sent_right = None

        actual_left = -left_tps if self.invert_left else left_tps
        actual_right = right_tps

        now = time.time()
        dl = abs(actual_left - self._last_sent_left) if self._last_sent_left is not None else float('inf')
        dr = abs(actual_right - self._last_sent_right) if self._last_sent_right is not None else float('inf')
        vel_changed = dl >= self.VEL_SEND_DELTA or dr >= self.VEL_SEND_DELTA
        watchdog_due = (now - self._last_send_time) >= self.WATCHDOG_FEED_INTERVAL

        if vel_changed or watchdog_due:
            with self.bus_lock:
                self.left.set_velocity(actual_left)
                self.right.set_velocity(actual_right)
            self._last_sent_left = actual_left
            self._last_sent_right = actual_right
            self._last_send_time = now

    def stop(self):
        self.set_wheel_vels(0.0, 0.0)

    def _read_axis_errors(self, axis, label):
        try:
            active = axis.read_property('axis0.active_errors')
            disarm = axis.read_property('axis0.disarm_reason')
            drv = axis.read_property('axis0.last_drv_fault')
            parts = []
            if active:
                parts.append(f"active_errors=0x{active:08X}")
            if disarm:
                parts.append(f"disarm_reason=0x{disarm:08X}")
            if drv:
                parts.append(f"drv_fault=0x{drv:08X}")
            if parts:
                print(f"   {label}: {', '.join(parts)}")
            else:
                print(f"   {label}: no errors")
        except Exception as e:
            print(f"   {label}: could not read errors ({e})")

    def shutdown(self):
        self.running = False
        time.sleep(0.1)
        with self.bus_lock:
            print("ODrive error check:")
            self._read_axis_errors(self.left, "Left (node 0)")
            self._read_axis_errors(self.right, "Right (node 1)")
            try:
                self.left.set_velocity(0.0)
                self.right.set_velocity(0.0)
                self.left.clear_errors()
                self.right.clear_errors()
                self.left.disable_watchdog()
                self.right.disable_watchdog()
                self.left.set_axis_state(ODriveAxisCAN.AXIS_STATE_IDLE)
                self.right.set_axis_state(ODriveAxisCAN.AXIS_STATE_IDLE)
            except can.CanOperationError as e:
                print(f"Warning during shutdown CAN commands: {e}")
        if self.bus:
            self.bus.shutdown()
        print("WheelBase shutdown complete")


# ====================== DEMO (Gamepad only - auto mode commented out) ======================
if __name__ == "__main__":
    bot = WheelBase()

    print("\n=== WheelBase Demo ===")
    print("Move right joystick for manual control (max 50% speed)")
    print("Release stick → clean zero (deadzone applied)")
    print("Ctrl+C to quit\n")

    last_print = time.time()

    try:
        while True:
            left_tps = 0.0
            right_tps = 0.0

            if bot.gamepad:
                vels = bot.gamepad.diffDrive()   # {'left': -1..1, 'right': -1..1}

                left_norm = vels['left']
                right_norm = vels['right']

                if abs(left_norm) < 0.08:
                    left_norm = 0.0
                if abs(right_norm) < 0.08:
                    right_norm = 0.0

                left_tps = left_norm * 0.5
                right_tps = right_norm * 0.5

            bot.set_wheel_vels(left_tps, right_tps)
            bot._check_gamepad_health()

            if time.time() - last_print > 1.0:
                if bot.gamepad:
                    gp = bot.gamepad
                    rx = gp.abs_state.get('RX', '?')
                    ry = gp.abs_state.get('RY', '?')
                    thread_ok = gp.gamepadThread.is_alive()
                    err = simplegamepad._monitorError
                    status = "OK" if thread_ok and not err else f"THREAD={'alive' if thread_ok else 'DEAD'} err={err}"
                    print(f"Gamepad → L={left_tps:5.2f}  R={right_tps:5.2f}   "
                          f"RX={rx:>6} RY={ry:>6}   "
                          f"idle={bot._is_idle} cl={bot._is_closed_loop}   "
                          f"[{status}]")
                else:
                    print(f"No gamepad   idle={bot._is_idle} cl={bot._is_closed_loop}")
                last_print = time.time()

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bot.shutdown()
