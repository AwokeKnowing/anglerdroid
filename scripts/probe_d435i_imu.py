#!/usr/bin/env python3
"""Probe D435i Motion Module + a few accel/gyro samples. Does not touch drive."""
from __future__ import annotations
import sys
import time

TARGET_SN = "944622074292"

def main() -> int:
    import pyrealsense2 as rs
    print(f"pyrealsense2 file={rs.__file__}")
    ctx = rs.context()
    target = None
    for d in ctx.devices:
        sn = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name)
        sensors = [s.get_info(rs.camera_info.name) for s in d.sensors]
        print(f"device: {name} sn={sn} sensors={sensors}")
        if sn == TARGET_SN:
            target = d
    if target is None:
        print(f"FAIL: D435i sn={TARGET_SN} not found")
        return 1
    names = [s.get_info(rs.camera_info.name) for s in target.sensors]
    if "Motion Module" not in names:
        print("FAIL: Motion Module not present")
        return 2
    print("OK: Motion Module present")

    # Accel: 250/63 Hz; Gyro: 400/200 Hz on this FW
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(TARGET_SN)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    pipe.start(cfg)
    print("IMU pipeline started (accel@250 gyro@200)")
    accel_n = gyro_n = 0
    t0 = time.time()
    try:
        while time.time() - t0 < 3.0 and (accel_n < 5 or gyro_n < 5):
            frames = pipe.wait_for_frames(timeout_ms=2000)
            for f in frames:
                if not f.is_motion_frame():
                    continue
                data = f.as_motion_frame().get_motion_data()
                st = f.get_profile().stream_type()
                line = f"ts={f.timestamp:.3f} x={data.x:.6f} y={data.y:.6f} z={data.z:.6f}"
                if st == rs.stream.accel and accel_n < 5:
                    print(f"ACCEL {line}")
                    accel_n += 1
                elif st == rs.stream.gyro and gyro_n < 5:
                    print(f"GYRO {line}")
                    gyro_n += 1
    finally:
        pipe.stop()
    print(f"samples accel={accel_n} gyro={gyro_n}")
    if accel_n == 0 or gyro_n == 0:
        print("FAIL: no IMU samples")
        return 3
    print("SUCCESS: IMU streaming")
    return 0

if __name__ == "__main__":
    sys.exit(main())
