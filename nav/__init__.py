"""Kevin full-stack navigation runtime (see docs/fullstack.md).

Runs unchanged on the Orin and on the Jetson emulator (KEVIN_HAL=sim): hardware is reached only through the
existing anglerdroid drivers in src/ (cameras.RSCamera, imu.IMUPipeline, wheelbase.WheelBase)."""
import os, sys
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
SRC = _SRC
