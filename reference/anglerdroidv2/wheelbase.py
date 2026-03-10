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


class WheelBase:
    WATCHDOG_FEED_INTERVAL = 1.0
    INCLINE_TORQUE_THRESHOLD = 1.0  # Nm — above this while holding zero = incline
    VEL_SEND_DELTA = 0.02
    TWIST_FOR_INTERVAL = 0.1  # 10 Hz

    def __init__(self,
                 can_interface: str = "can0",
                 wheel_diameter_cm: float = 17.13,
                 wheelbase_cm: float = 34.0,
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
        self._is_closed_loop = False
        self._is_idle = True
        self._last_sent_left = None
        self._last_sent_right = None
        self._last_send_time = 0.0
        self._zero_vel_since = None
        self._twist_for_lock = threading.Lock()
        self._twist_for_params = None  # (forward_mps, angular_rads, duration_secs, ramp_in_secs, ramp_out_secs, start_time)

        self._bring_up_can()
        self._init_odrive()
        self._init_gamepad()
        self._start_idle_watcher()
        self._start_twist_for_thread()

        print(f"✅ WheelBase ready (wheelbase={wheelbase_cm}cm, max_speed=50%)")

    def _bring_up_can(self):
        try:
            result = subprocess.run(["ip", "-d", "link", "show", self.can_interface],
                                    capture_output=True, text=True, timeout=2)
            if "state UP" in result.stdout:
                print(f"   {self.can_interface} is already UP")
                return
        except Exception:
            pass

        print(f"   Bringing up {self.can_interface}...")
        try:
            subprocess.check_call(["sudo", "ip", "link", "set", self.can_interface,
                                   "up", "type", "can", "bitrate", "250000"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

    def cancel_twist_for(self):
        """Cancel any in-progress twist_for (e.g. when gamepad takes over)."""
        with self._twist_for_lock:
            self._twist_for_params = None

    def set_wheel_vels(self, left_tps: float, right_tps: float):
        """Direct wheel control (turns/s). Deduplicates sends and manages idle."""
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
