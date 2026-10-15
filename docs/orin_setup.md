# Kevin Orin NX Setup

**Phase 0 truth capture (2026-10-15):** Rebuildable host configuration from live `jetbot` Orin.

This page documents the real hardware/software configuration on the production Kevin Orin NX. Use this to rebuild a fresh machine or validate an existing one.

**NOTE:** Root `README.md` and `Dockerfile` still reference **JetPack 6 / CUDA 12** and are **STALE**. Ignore those; this document reflects the actual running system (JP 7.2 / CUDA 13.2).

## Platform

| Component | Value | Notes |
|-----------|-------|-------|
| Hostname | `jetbot` | |
| Board | NVIDIA Jetson Orin NX | |
| JetPack | **7.2-b187** | `nvidia-jetpack` meta-package |
| L4T / Tegra | **R39.2** (REVISION 2.0) | `/etc/nv_tegra_release`, GCID 45755727 |
| Ubuntu | **24.04.4 LTS** (noble) | |
| Kernel | 6.8.12-1021-tegra | PREEMPT, oot variant |
| Arch | aarch64 | |
| CUDA toolkit | **13.2** | `cuda-toolkit-13-2` package |
| Python | **3.12.3** | `/usr/bin/python3.12` |
| Power mode | **40W** | `nvpmodel -q` mode 4 |

## PyTorch / Planner venvs

Kevin uses **two separate venvs** with a borrowed torch path:

| Path | Role | Contents |
|------|------|----------|
| `~/.venv-kplan` | **Primary CUDA torch install** | torch **2.14.1+cu132**, torchvision **0.29.1+cu132**, triton **3.8.0**, ninja |
| `~/.venv-kevinnav` | **Nav/planner venv** | Resolves torch from kplan site-packages (symlink or PATH-based borrow) |
| `~/anglerdroid/.venv` | App/RealSense python | librealsense bindings (no torchvision) |

### PyTorch Installation

**Critical:** Use PyTorch built for **Jetson Orin / CUDA 13.2**, NOT generic pip wheels.

1. Install torch in `~/.venv-kplan`:
   ```bash
   python3.12 -m venv ~/.venv-kplan
   source ~/.venv-kplan/bin/activate
   pip install torch==2.14.1+cu132 torchvision==0.29.1+cu132 --index-url https://download.pytorch.org/whl/cu132
   pip install triton ninja
   ```

2. Create nav venv that borrows torch:
   ```bash
   python3.12 -m venv ~/.venv-kevinnav
   # kevinnav resolves torch from kplan's site-packages (via symlink or PYTHONPATH)
   ```

3. Verify CUDA:
   ```bash
   ~/.venv-kplan/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
   # Expected: True Jetson Orin ...
   ```

4. Build MPPI CUDA kernel:
   ```bash
   cd ~/anglerdroid
   KEVIN_PY=~/.venv-kevinnav/bin/python scripts/build_orin_cuda.sh
   ```

**Resources:**
- https://download.pytorch.org/whl/cu132
- https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/

## RealSense Cameras

### Hardware

| Item | Value |
|------|-------|
| D435i serial | **944622074292** |
| Firmware | **5.15.1** |
| Role | RS2 (forward-facing) |
| USB requirement | **USB 3** (10 Gb/s port or hub) |

**BLOCKER (2026-10-14):** D435i currently on USB 2.1 (Bus 001, 480M) instead of Bus 002 USB 3. This breaks 848×480 depth, IMU "not available", and RS1 frame delivery. **Plug both cameras into Orin's 10 Gb/s USB 3 port** (or USB 3 hub). Verify with `lsusb -t` → cameras under `Bus 002 ... 10000M`.

Second camera (D435 non-i, top-down RS1): **absent** at snapshot time. Role assigned by product name when present.

### librealsense RSUSB Build (IMU Fix)

JetPack 7 / L4T R39 removed kernel `hiddraw` → native HID/IIO IMU paths broken. The pip `pyrealsense2` wheel enumerated Stereo+RGB only (no Motion Module).

**Solution:** Build librealsense **v2.58.4** from source with `-DFORCE_RSUSB_BACKEND=ON`:

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git librealsense_build/librealsense
cd librealsense_build/librealsense
git checkout v2.58.4

mkdir build && cd build
cmake .. -DFORCE_RSUSB_BACKEND=ON \
         -DBUILD_PYTHON_BINDINGS=ON \
         -DPYTHON_EXECUTABLE=~/anglerdroid/.venv/bin/python \
         -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
sudo make install
sudo ldconfig
```

This installs:
- `/usr/local/lib/librealsense2.so*`
- `/usr/local/bin/rs-enumerate-devices`
- Python bindings into `~/anglerdroid/.venv`

Add to `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

Or create `/etc/ld.so.conf.d/librealsense.conf`:
```
/usr/local/lib
```
Then `sudo ldconfig`.

**Proven IMU rates:**
- Accel: 250 or 63 Hz (NOT 200 — causes "Couldn't resolve requests")
- Gyro: 400 or 200 Hz

Use `cfg.enable_device("944622074292")` + accel@250 + gyro@200 in nav code. Prefer **separate IMU pipeline** from depth/color.

### udev Rules

Install RealSense udev rules (copied in `deploy/orin/`):

```bash
sudo cp deploy/orin/99-realsense-libusb.rules /etc/udev/rules.d/
sudo cp deploy/orin/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug/replug cameras after installing rules.

## CAN (ODrive)

### Interface

| Item | Value |
|------|-------|
| Interface | `can1` |
| Bitrate | 250 kbit/s |
| Hardware | CANable USB (gs_usb driver) |
| Device ID | `1d50:606f` (OpenMoko Geschwister Schneider) |

### Systemd Unit

`deploy/orin/can1-odrive.service` brings up `can1` at boot.

**Current state on `jetbot`:** Loaded but **disabled**; `can1` interface manually brought up (or via `scripts/canable_watch.sh`).

To install:
```bash
sudo cp deploy/orin/can1-odrive.service /etc/systemd/system/
sudo systemctl daemon-reload
# OPTIONAL (auto-start at boot; may be risky until validated):
# sudo systemctl enable can1-odrive.service
```

Manual bring-up:
```bash
sudo modprobe gs_usb
sudo ip link set can1 down 2>/dev/null || true
sudo ip link set can1 up type can bitrate 250000
sudo ip link set can1 txqueuelen 1000
```

Verify:
```bash
ip link show can1
# Expected: UP, state UP, qlen 1000
```

### Watch Scripts

Repo includes USB recovery scripts (currently manual cron, not auto-enabled):

- `scripts/canable_watch.sh` — monitors CANable USB, soft-resets if gone
- `scripts/wifi_dongle_watch.sh` — monitors RTL8812AU wifi dongle

**Do NOT auto-enable in docs** until validated safe.

## Calibration

`src/calibration.json` holds robot-specific calibration (wheel scales, track, camera mounts). **Not present** on Orin checkout at snapshot time — auto-cal writes it on first 10s park. Twin stays nominal; only real robot gets calibration.json.

See `docs/bringup.md` step 7 (ruler tests) for calibration procedure.

## Pre-flight / Bring-up

Before running Kevin on the Orin:

1. **Verify CUDA:** `scripts/build_orin_cuda.sh` (exits if GPU unusable)
2. **Pre-flight checks:** `scripts/bringup_check.py` (USB 3, RealSense 30 Hz, IMU, CAN, safety, calibration)
3. **Bring-up:** `scripts/bringup_run.sh 0.3` (enforces CUDA, starts nav at port 8080)

See `docs/bringup.md` for full checklist (wheels-up, e-stop, hand-over-lens, floor tests).

## Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| Cameras on USB 2 | **BLOCKER** | Plug into USB 3 port (Bus 002) |
| Dockerfile / README stale | Doc debt | Ignore; use this page for JP 7.2 / CUDA 13.2 |
| `src/calibration.json` missing | Expected | Auto-cal writes on first park |
| `can1-odrive.service` disabled | Current state | Manual `ip link` or enable after validation |

## Related Docs

- **`docs/bringup.md`** — step-by-step bring-up checklist (pre-flight → wheels-up → floor)
- **`docs/fullstack.md`** — repo layout and architecture
- **`deploy/orin/README.md`** — udev/systemd deployment instructions
- **`scripts/build_orin_cuda.sh`** — CUDA verification + MPPI kernel build

## Serials / Inventory

Known serials as of 2026-10-15:

| Device | Serial | Role | Notes |
|--------|--------|------|-------|
| D435i | 944622074292 | RS2 (forward) | FW 5.15.1, IMU working after RSUSB build |
| D435 (non-i) | (absent) | RS1 (top-down) | Not present at snapshot time |

**Phase 1/2 (future):** `robots/kevin001/inventory.json` with serials, power mode, MAC addresses. Not implemented yet.
