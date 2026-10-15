#!/bin/bash
# Bootstrap a fresh Kevin Orin NX to match docs/orin_setup.md
# Usage: scripts/bootstrap_orin.sh [--install-system]
# Safe to re-run (idempotent where possible). See docs/orin_setup.md for manual steps.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_SYSTEM=0
[[ "$1" == "--install-system" ]] && INSTALL_SYSTEM=1

echo "=== Kevin Orin Bootstrap ==="
echo

# ──────────────────────────────────────────────────────────────────────────────
# 1. Preconditions: Jetson / aarch64 / JetPack / CUDA
# ──────────────────────────────────────────────────────────────────────────────
echo "Step 1: Platform checks"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: Not aarch64 ($(uname -m)). This script is for Jetson Orin NX only."
    exit 1
fi

if [[ ! -f /etc/nv_tegra_release ]]; then
    echo "WARNING: /etc/nv_tegra_release not found. Assuming non-Jetson platform."
else
    TEGRA=$(head -1 /etc/nv_tegra_release)
    echo "Tegra: $TEGRA"
fi

if dpkg -l nvidia-jetpack 2>/dev/null | grep -q '^ii'; then
    JP_VER=$(dpkg -l nvidia-jetpack | grep '^ii' | awk '{print $3}')
    echo "JetPack: $JP_VER"
else
    echo "WARNING: nvidia-jetpack meta-package not found. Cannot verify JetPack version."
fi

if [[ ! -d /usr/local/cuda ]]; then
    echo "ERROR: /usr/local/cuda not found. Install CUDA toolkit first:"
    echo "  sudo apt install cuda-toolkit-13-2"
    echo "See docs/orin_setup.md for JetPack 7.2 / CUDA 13.2 setup."
    exit 1
fi

CUDA_VER=$(nvcc --version 2>/dev/null | grep 'release' | awk '{print $5}' | sed 's/,//' || echo "unknown")
echo "CUDA: $CUDA_VER (nvcc)"
echo

# ──────────────────────────────────────────────────────────────────────────────
# 2. Ensure venvs: ~/.venv-kplan (torch), ~/.venv-kevinnav (nav, borrows torch)
# ──────────────────────────────────────────────────────────────────────────────
echo "Step 2: Python venvs"

PY=/usr/bin/python3.12
if [[ ! -x "$PY" ]]; then
    PY=/usr/bin/python3
    if [[ ! -x "$PY" ]]; then
        echo "ERROR: Python 3.12 or 3 not found at /usr/bin/python3.12 or /usr/bin/python3"
        exit 1
    fi
fi
echo "Python: $PY ($($PY --version 2>&1))"

VENV_KPLAN="$HOME/.venv-kplan"
VENV_KEVINNAV="$HOME/.venv-kevinnav"

# Create venv-kplan if missing
if [[ ! -d "$VENV_KPLAN" ]]; then
    echo "Creating $VENV_KPLAN..."
    $PY -m venv "$VENV_KPLAN"
else
    echo "venv-kplan exists: $VENV_KPLAN"
fi

# Install/upgrade torch + torchvision in kplan
echo "Installing torch 2.14.1+cu132 + torchvision 0.29.1+cu132 in venv-kplan..."
"$VENV_KPLAN/bin/pip" install --upgrade pip setuptools wheel
"$VENV_KPLAN/bin/pip" install \
    torch==2.14.1+cu132 \
    torchvision==0.29.1+cu132 \
    --index-url https://download.pytorch.org/whl/cu132

# Install triton + ninja in kplan
echo "Installing triton + ninja in venv-kplan..."
"$VENV_KPLAN/bin/pip" install triton ninja

# Verify torch in kplan
if ! "$VENV_KPLAN/bin/python" -c 'import torch; print("torch OK:", torch.__version__)' 2>/dev/null; then
    echo "ERROR: torch import failed in venv-kplan"
    exit 1
fi

# Create venv-kevinnav if missing
if [[ ! -d "$VENV_KEVINNAV" ]]; then
    echo "Creating $VENV_KEVINNAV..."
    $PY -m venv "$VENV_KEVINNAV"
else
    echo "venv-kevinnav exists: $VENV_KEVINNAV"
fi

# Ensure kevinnav can import torch by symlinking kplan's site-packages
KPLAN_SITE="$VENV_KPLAN/lib/python3.12/site-packages"
KEVINNAV_SITE="$VENV_KEVINNAV/lib/python3.12/site-packages"

# Fallback to python3.x if python3.12 not present
if [[ ! -d "$KPLAN_SITE" ]]; then
    KPLAN_SITE=$(find "$VENV_KPLAN/lib" -maxdepth 1 -type d -name 'python3.*' | head -1)/site-packages
fi
if [[ ! -d "$KEVINNAV_SITE" ]]; then
    KEVINNAV_SITE=$(find "$VENV_KEVINNAV/lib" -maxdepth 1 -type d -name 'python3.*' | head -1)/site-packages
fi

# Create symlinks for torch modules in kevinnav (borrows from kplan)
for mod in torch torchvision triton; do
    if [[ -e "$KPLAN_SITE/$mod" ]] && [[ ! -e "$KEVINNAV_SITE/$mod" ]]; then
        echo "Symlinking $mod from kplan to kevinnav..."
        ln -s "$KPLAN_SITE/$mod" "$KEVINNAV_SITE/$mod"
    fi
done

# Verify torch in kevinnav
if ! "$VENV_KEVINNAV/bin/python" -c 'import torch; print("torch OK:", torch.__version__)' 2>/dev/null; then
    echo "ERROR: torch import failed in venv-kevinnav (borrow from kplan failed)"
    exit 1
fi

# Ensure ninja in kevinnav bin (symlink from kplan)
if [[ ! -e "$VENV_KEVINNAV/bin/ninja" ]] && [[ -e "$VENV_KPLAN/bin/ninja" ]]; then
    echo "Symlinking ninja from kplan to kevinnav..."
    ln -s "$VENV_KPLAN/bin/ninja" "$VENV_KEVINNAV/bin/ninja"
fi

echo "Venvs ready:"
echo "  kplan:    $VENV_KPLAN (torch primary)"
echo "  kevinnav: $VENV_KEVINNAV (borrows torch from kplan)"
echo

# ──────────────────────────────────────────────────────────────────────────────
# 3. Install system files: udev rules + systemd unit (requires --install-system)
# ──────────────────────────────────────────────────────────────────────────────
echo "Step 3: System files (udev, systemd)"

if [[ $INSTALL_SYSTEM -eq 0 ]]; then
    echo "Skipping system file installation (pass --install-system to enable)."
    echo
    echo "To install manually (requires sudo):"
    echo "  sudo cp $REPO/deploy/orin/99-realsense-*.rules /etc/udev/rules.d/"
    echo "  sudo udevadm control --reload-rules"
    echo "  sudo udevadm trigger"
    echo
    echo "  sudo cp $REPO/deploy/orin/can1-odrive.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  # OPTIONAL (auto-start CAN at boot; currently disabled on live jetbot):"
    echo "  # sudo systemctl enable can1-odrive.service"
    echo
else
    echo "Installing system files (requires sudo)..."
    
    # udev rules
    if ! sudo cp "$REPO/deploy/orin/99-realsense-libusb.rules" /etc/udev/rules.d/; then
        echo "ERROR: Failed to copy 99-realsense-libusb.rules"
        exit 1
    fi
    if ! sudo cp "$REPO/deploy/orin/99-realsense-d4xx-mipi-dfu.rules" /etc/udev/rules.d/; then
        echo "ERROR: Failed to copy 99-realsense-d4xx-mipi-dfu.rules"
        exit 1
    fi
    
    echo "Reloading udev rules..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    # systemd CAN unit
    if ! sudo cp "$REPO/deploy/orin/can1-odrive.service" /etc/systemd/system/; then
        echo "ERROR: Failed to copy can1-odrive.service"
        exit 1
    fi
    
    echo "Reloading systemd..."
    sudo systemctl daemon-reload
    
    echo "System files installed."
    echo "  udev rules: /etc/udev/rules.d/99-realsense-*.rules"
    echo "  systemd:    /etc/systemd/system/can1-odrive.service"
    echo
    echo "NOTE: can1-odrive.service is NOT enabled (consistent with live jetbot)."
    echo "To enable auto-start at boot (optional, validate first):"
    echo "  sudo systemctl enable can1-odrive.service"
    echo
fi

# ──────────────────────────────────────────────────────────────────────────────
# 4. Build/verify CUDA MPPI kernel
# ──────────────────────────────────────────────────────────────────────────────
echo "Step 4: Build CUDA MPPI kernel"

export KEVIN_PY="$VENV_KEVINNAV/bin/python"
if ! "$REPO/scripts/build_orin_cuda.sh"; then
    echo "ERROR: CUDA build/verify failed. See output above."
    echo "Fix: Ensure PyTorch is built for Jetson/JetPack 7.2 with CUDA 13.x support."
    echo "See docs/orin_setup.md for torch installation instructions."
    exit 1
fi

echo

# ──────────────────────────────────────────────────────────────────────────────
# 5. Next steps
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Bootstrap complete ==="
echo
echo "Next steps:"
echo
echo "1. RealSense cameras:"
echo "   - Plug both D435i + D435 into USB 3 port (Bus 002, 10 Gb/s)"
echo "   - Verify with: lsusb -t  (cameras should be under 'Bus 002 ... 10000M')"
echo "   - Build librealsense RSUSB (IMU support): see docs/orin_setup.md 'librealsense RSUSB Build'"
echo
echo "2. CAN interface:"
echo "   - Plug CANable USB (1d50:606f)"
echo "   - Manual bring-up: sudo modprobe gs_usb && sudo ip link set can1 up type can bitrate 250000"
echo "   - Or enable systemd unit: sudo systemctl enable --now can1-odrive.service"
echo
echo "3. Robot inventory:"
echo "   - Set KEVIN_ROBOT=kevin001 (or create robots/<robot_id>/inventory.json for new robot)"
echo
echo "4. Pre-flight check:"
echo "   KEVIN_ROBOT=kevin001 $VENV_KEVINNAV/bin/python scripts/bringup_check.py"
echo
echo "5. Bring-up (capped run):"
echo "   KEVIN_ROBOT=kevin001 scripts/bringup_run.sh 0.3"
echo
echo "See docs/bringup.md for full checklist (wheels-up, e-stop, hand-over-lens, floor tests)."
