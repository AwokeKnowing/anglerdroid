#!/bin/bash
# Build and verify CUDA extensions for Kevin MPPI planner on Jetson Orin.
# Must run on GPU — CPU fallback forbidden. Validates torch.cuda, builds MPPI kernel, fails loudly if GPU unusable.
set -e

echo "=== Kevin CUDA build/verify (Jetson Orin) ==="

# Detect Jetson platform
if [ ! -f /etc/nv_tegra_release ] && [ "$(uname -m)" != "aarch64" ]; then
    echo "WARNING: Not on Jetson aarch64 platform (continuing anyway for dev hosts)"
fi

# Python from KEVIN_PY or ~/.venv-kevinnav/bin/python
PY=${KEVIN_PY:-$HOME/.venv-kevinnav/bin/python}
if [ ! -x "$PY" ]; then
    echo "ERROR: Python not found at $PY"
    echo "Set KEVIN_PY or create ~/.venv-kevinnav venv"
    exit 1
fi

echo "Python: $PY"
echo

# Check torch.cuda.is_available() and actual device usability
echo "Checking PyTorch CUDA..."
$PY -c '
import torch
import sys

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() returned False")
    print("Fix: Install PyTorch built for Jetson/JetPack 7.2 with CUDA 13.x support")
    print("  - Check https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/")
    print("  - Ensure JetPack and CUDA toolkit are installed")
    sys.exit(1)

print(f"torch.cuda.is_available(): True")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device 0: {torch.cuda.get_device_name(0)}")

# Verify we can actually allocate and run on GPU (not just "available")
try:
    x = torch.zeros(100, device="cuda")
    y = x + 1.0
    y = y.cpu()
    assert y[0].item() == 1.0
    print("Device usability: OK (alloc + compute succeeded)")
except Exception as e:
    print(f"ERROR: GPU device unusable despite is_available()==True: {e}")
    print("Fix: Check dmesg for CUDA driver errors, verify nvidia-smi, reboot Orin")
    sys.exit(1)
'

if [ $? -ne 0 ]; then
    echo
    echo "=== CUDA NOT USABLE ==="
    exit 1
fi

echo
echo "Checking ninja (required for CUDA extension build)..."
NINJA_PATH=$(dirname "$PY")/ninja
if [ ! -x "$NINJA_PATH" ]; then
    echo "WARNING: ninja not found in venv bin/ ($NINJA_PATH)"
    echo "Checking system PATH..."
    if ! command -v ninja &>/dev/null; then
        echo "ERROR: ninja not found (required by torch.utils.cpp_extension)"
        echo "Fix: pip install ninja  (or apt install ninja-build)"
        exit 1
    else
        echo "ninja found in system PATH: $(command -v ninja)"
    fi
else
    echo "ninja found in venv: $NINJA_PATH"
fi

echo
echo "Building MPPI CUDA fused kernel..."
# Import mppi_cuda to trigger load_inline build (JIT compiles if not cached)
$PY -c '
import sys, os
# Ensure ninja is on PATH (venv bin first)
py_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = py_bin + os.pathsep + os.environ.get("PATH", "")

print(f"PATH prefix: {py_bin}")

# Suppress verbose build output unless VERBOSE=1
verbose = os.environ.get("VERBOSE", "0") == "1"
if not verbose:
    import warnings
    warnings.filterwarnings("ignore")

try:
    # This triggers torch.utils.cpp_extension.load_inline -> nvcc + ninja
    from nav.kplan import mppi_cuda
    print("MPPI CUDA kernel built successfully (or loaded from cache)")
except Exception as e:
    print(f"ERROR: Failed to build MPPI CUDA kernel: {e}")
    print("Common causes:")
    print("  - nvcc not found (install CUDA toolkit matching JetPack)")
    print("  - ninja not found (pip install ninja)")
    print("  - torch built without CUDA support for this arch")
    sys.exit(1)
'

if [ $? -ne 0 ]; then
    echo
    echo "=== MPPI CUDA BUILD FAILED ==="
    exit 1
fi

echo
echo "=== CUDA BUILD/VERIFY PASSED ==="
echo "Kevin planner will run on GPU (CPU fallback disabled)"
exit 0
