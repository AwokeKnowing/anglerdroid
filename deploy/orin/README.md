# Orin Deployment Files

System-level configuration files for Kevin Orin NX (captured from live `jetbot` host 2026-10-15).

## Files

### can1-odrive.service
Systemd oneshot unit to bring up CAN interface `can1` (CANable USB, gs_usb driver) at 250 kbit/s for ODrive motor controllers.

**Install:**
```bash
sudo cp can1-odrive.service /etc/systemd/system/
sudo systemctl daemon-reload
```

**Current state on jetbot:** Loaded but **disabled** (inactive/dead). `can1` interface is brought up manually or via `scripts/canable_watch.sh`.

**Optional (enable auto-start at boot):**
```bash
sudo systemctl enable can1-odrive.service
sudo systemctl start can1-odrive.service
```

**Verify:**
```bash
ip link show can1
# Expected: UP, state UP, bitrate 250000, qlen 1000
```

**NOTE:** Do not auto-enable until validated safe for your robot. Unit retries up to 10 times with 0.5s delays if `can1` device not ready at boot.

### 99-realsense-libusb.rules
Intel RealSense udev rules for libusb backend (D400/L500/T200 series). Sets device permissions (`MODE:="0666"`, `GROUP:="plugdev"`) and IIO sensor workarounds for motion modules.

**Install:**
```bash
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug/replug RealSense cameras after installing.

### 99-realsense-d4xx-mipi-dfu.rules
RealSense MIPI devices and DFU (firmware update) rules. Used for MIPI camera modules and firmware recovery mode.

**Install:**
```bash
sudo cp 99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Installation Summary

To install all files on a fresh Orin:

```bash
cd ~/anglerdroid/deploy/orin

# udev rules (RealSense cameras)
sudo cp 99-realsense-*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

# CAN interface (systemd unit)
sudo cp can1-odrive.service /etc/systemd/system/
sudo systemctl daemon-reload

# OPTIONAL: auto-enable CAN at boot (validate first)
# sudo systemctl enable can1-odrive.service
```

## Related Docs

- **`docs/orin_setup.md`** — full Orin setup guide (JetPack, PyTorch, RealSense, CAN)
- **`docs/bringup.md`** — bring-up checklist for real robot
- **`scripts/canable_watch.sh`** — CANable USB recovery script (manual cron)
- **`scripts/build_orin_cuda.sh`** — CUDA verification for planner

## Notes

- These files reflect the **live jetbot Orin** configuration as of 2026-10-15.
- `can1-odrive.service` is currently **disabled** on jetbot (manual bring-up preferred until validated).
- RealSense cameras **must be on USB 3** (Bus 002, 10 Gb/s) — see `docs/orin_setup.md` USB requirement.
- IMU requires librealsense built from source with `FORCE_RSUSB_BACKEND=ON` (JetPack 7 removed kernel hiddraw).
