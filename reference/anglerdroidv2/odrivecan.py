"""
odrivecan.py - FINAL PRODUCTION (shared burst feeding fix)
=======================================================
- vel_ramp_rate = 3.0
- Shared burst feeding after BOTH nodes are in closed loop
- No more starvation during startup
"""

import json
import struct
import can
import time
from typing import Union, Optional


class ODriveAxisCAN:
    AXIS_STATE_IDLE = 1
    AXIS_STATE_CLOSED_LOOP_CONTROL = 8

    CMD_RX_SDO = 0x04
    CMD_TX_SDO = 0x05
    CMD_SET_AXIS_STATE = 0x07
    CMD_SET_INPUT_VEL = 0x0d
    CMD_SET_INPUT_TORQUE = 0x0e
    CMD_CLEAR_ERRORS = 0x18

    DEFAULT_VEL_RAMP_RATE = 3.0
    DEFAULT_VEL_LIMIT = 4.0
    DEFAULT_WATCHDOG_TIMEOUT_S = 2.5

    STARTUP_IDLE_WAIT_S = 0.8
    STARTUP_CLOSED_LOOP_WAIT_S = 0.6
    JOINT_BURST_COUNT = 90          # shared burst for both nodes
    BURST_SLEEP_S = 0.03
    LOOP_RATE_HZ = 50.0

    MAX_VEL_TURNS_PER_S = 8.0
    MAX_TURN_RATE = 5.0
    VEL_STEP = 0.5

    BATTERY_MIN_V = 30.0
    BATTERY_MAX_V = 42.0

    format_lookup = {
        'bool': '?', 'uint8': 'B', 'int8': 'b',
        'uint16': 'H', 'int16': 'h',
        'uint32': 'I', 'int32': 'i',
        'uint64': 'Q', 'int64': 'q',
        'float': 'f'
    }

    SEND_RETRIES = 5
    SEND_RETRY_DELAY = 0.02

    def __init__(self, bus: can.BusABC, node_id: int, endpoints: dict):
        self.bus = bus
        self.node_id = node_id
        self.endpoints = endpoints
        self.name = f"Node {node_id}"

    def _safe_send(self, msg: can.Message):
        for attempt in range(self.SEND_RETRIES):
            try:
                self.bus.send(msg)
                return
            except can.CanOperationError:
                if attempt < self.SEND_RETRIES - 1:
                    time.sleep(self.SEND_RETRY_DELAY)
                else:
                    raise

    def flush_bus(self):
        while self.bus.recv(timeout=0) is not None:
            pass

    def write_property(self, property: str, value: Union[int, float, bool] = True):
        if property not in self.endpoints:
            return
        ep_id = self.endpoints[property]['id']
        ep_type = self.endpoints[property]['type']
        if ep_type == 'function':
            data = struct.pack('<BHB', 0x01, ep_id, 0)
        else:
            fmt = self.format_lookup[ep_type]
            data = struct.pack('<BHB' + fmt, 0x01, ep_id, 0, value)
        self._safe_send(can.Message(
            arbitration_id=(self.node_id << 5 | self.CMD_RX_SDO),
            data=data,
            is_extended_id=False
        ))
        time.sleep(0.005)

    def read_property(self, property: str) -> Optional[Union[int, float, bool]]:
        if property not in self.endpoints:
            return None
        ep_id = self.endpoints[property]['id']
        ep_type = self.endpoints[property]['type']
        fmt = self.format_lookup[ep_type]
        self.flush_bus()
        self._safe_send(can.Message(
            arbitration_id=(self.node_id << 5 | self.CMD_RX_SDO),
            data=struct.pack('<BHB', 0x00, ep_id, 0),
            is_extended_id=False
        ))
        for msg in self.bus:
            if msg.is_rx and msg.arbitration_id == (self.node_id << 5 | self.CMD_TX_SDO):
                break
        else:
            return None
        _, _, _, val = struct.unpack_from('<BHB' + fmt, msg.data)
        return val

    def clear_errors(self):
        self._safe_send(can.Message(
            arbitration_id=(self.node_id << 5 | self.CMD_CLEAR_ERRORS),
            data=b'',
            is_extended_id=False
        ))
        time.sleep(0.25)

    def set_axis_state(self, state: int):
        self._safe_send(can.Message(
            arbitration_id=(self.node_id << 5 | self.CMD_SET_AXIS_STATE),
            data=struct.pack('<I', state),
            is_extended_id=False
        ))

    def set_velocity(self, vel: float, torque_ff: float = 0.0):
        self._safe_send(can.Message(
            arbitration_id=(self.node_id << 5 | self.CMD_SET_INPUT_VEL),
            data=struct.pack('<ff', vel, torque_ff),
            is_extended_id=False
        ))

    def set_torque(self, torque_nm: float):
        self._safe_send(can.Message(
            arbitration_id=(self.node_id << 5 | self.CMD_SET_INPUT_TORQUE),
            data=struct.pack('<f', torque_nm),
            is_extended_id=False
        ))

    def set_vel_ramp_rate(self, rate: float):
        self.write_property('axis0.controller.config.vel_ramp_rate', rate)
        print(f"  [{self.name}] vel_ramp_rate → {rate} turns/s²")

    def set_vel_limit(self, limit: float):
        self.write_property('axis0.controller.config.vel_limit', limit)

    def feed_watchdog(self):
        if 'axis0.watchdog_feed' in self.endpoints:
            self.write_property('axis0.watchdog_feed')

    def enable_watchdog(self, timeout: float = DEFAULT_WATCHDOG_TIMEOUT_S):
        self.write_property('axis0.config.enable_watchdog', True)
        self.write_property('axis0.config.watchdog_timeout', timeout)
        print(f"  [{self.name}] Watchdog ENABLED ({timeout}s)")

    def disable_watchdog(self):
        self.write_property('axis0.config.enable_watchdog', False)

    def get_vbus_voltage(self) -> float:
        v = self.read_property('vbus_voltage')
        return float(v) if v is not None else 0.0

    def get_battery_level(self, min_v: float = BATTERY_MIN_V, max_v: float = BATTERY_MAX_V) -> int:
        v = self.get_vbus_voltage()
        level = int(100 * (v - min_v) / (max_v - min_v))
        return max(0, min(100, level))


# ====================== INTERACTIVE DEMO ======================
if __name__ == "__main__":
    import sys
    import select
    import tty
    import termios

    print("=== ODrive S1 Botwheels - Production Demo (shared burst fix) ===")

    bus = can.Bus("can0", interface="socketcan")

    with open('flat_endpoints.json', 'r') as f:
        endpoints = json.load(f)['endpoints']

    left = ODriveAxisCAN(bus, 0, endpoints)
    right = ODriveAxisCAN(bus, 1, endpoints)

    try:
        # === SHARED STARTUP PHASE ===
        print("Putting both axes into IDLE...")
        left.clear_errors()
        right.clear_errors()
        left.set_axis_state(left.AXIS_STATE_IDLE)
        right.set_axis_state(right.AXIS_STATE_IDLE)
        time.sleep(1.0)

        print("Entering CLOSED_LOOP_CONTROL on both...")
        left.set_axis_state(left.AXIS_STATE_CLOSED_LOOP_CONTROL)
        right.set_axis_state(right.AXIS_STATE_CLOSED_LOOP_CONTROL)
        time.sleep(0.8)

        # Apply ramp rate
        left.set_vel_ramp_rate(ODriveAxisCAN.DEFAULT_VEL_RAMP_RATE)
        right.set_vel_ramp_rate(ODriveAxisCAN.DEFAULT_VEL_RAMP_RATE)
        left.set_vel_limit(ODriveAxisCAN.DEFAULT_VEL_LIMIT)
        right.set_vel_limit(ODriveAxisCAN.DEFAULT_VEL_LIMIT)

        # === SHARED BURST FEEDING (both nodes fed together) ===
        print("Joint burst-feeding zero velocity to BOTH nodes (3 seconds)...")
        for _ in range(ODriveAxisCAN.JOINT_BURST_COUNT):
            left.set_velocity(0.0)
            left.feed_watchdog()
            right.set_velocity(0.0)
            right.feed_watchdog()
            time.sleep(ODriveAxisCAN.BURST_SLEEP_S)

        # Enable watchdog only after both are happy
        left.enable_watchdog(3.0)   # Node 0 gets extra margin
        right.enable_watchdog(2.5)

        print("\nBattery: L={left.get_battery_level():3d}%   R={right.get_battery_level():3d}%")
        print("\nW/↑ forward   S/↓ back   A/← left   D/→ right   SPACE stop   Q quit\n")

        forward = 0.0
        turn = 0.0
        last_print = time.time()

        def get_key():
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                r, _, _ = select.select([sys.stdin], [], [], 0.01)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        ch += sys.stdin.read(2)
                    return ch
                return None
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        while True:
            key = get_key()
            if key:
                if key in ('w', 'W', '\x1b[A'): forward += ODriveAxisCAN.VEL_STEP
                elif key in ('s', 'S', '\x1b[B'): forward -= ODriveAxisCAN.VEL_STEP
                elif key in ('a', 'A', '\x1b[D'): turn += ODriveAxisCAN.VEL_STEP
                elif key in ('d', 'D', '\x1b[C'): turn -= ODriveAxisCAN.VEL_STEP
                elif key == ' ': forward = turn = 0.0
                elif key.lower() == 'q': break

                forward = max(-ODriveAxisCAN.MAX_VEL_TURNS_PER_S, min(ODriveAxisCAN.MAX_VEL_TURNS_PER_S, forward))
                turn = max(-ODriveAxisCAN.MAX_TURN_RATE, min(ODriveAxisCAN.MAX_TURN_RATE, turn))

            vel_l = forward + turn
            vel_r = forward - turn

            left.set_velocity(vel_l)
            right.set_velocity(vel_r)
            left.feed_watchdog()
            right.feed_watchdog()

            if time.time() - last_print > 1.0:
                print(f"Vel L={vel_l:5.1f}  R={vel_r:5.1f}   Batt L={left.get_battery_level():3d}%")
                last_print = time.time()

            time.sleep(1.0 / ODriveAxisCAN.LOOP_RATE_HZ)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("=== CLEAN SHUTDOWN ===")
        for axis in (left, right):
            axis.set_velocity(0.0)
            axis.set_axis_state(axis.AXIS_STATE_IDLE)
            axis.disable_watchdog()
        bus.shutdown()
        print("✅ Motors idle, watchdog disabled.")
