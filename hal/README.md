# Hardware abstraction seam

The robot code (src/, perception/, nav/) talks to hardware only through three third-party APIs:
`pyrealsense2` (RS1 top-down D435, RS2 forward D435i + its IMU), `python-can` (ODrive S1 over CAN,
CANSimple protocol) and `inputs` (gamepad). The HAL swaps those *libraries*, not the robot code.

| KEVIN_HAL | what it does |
|-----------|--------------|
| `real` (default) | nothing — the real packages and devices are used (Orin). |
| `sim` | `hal/sim` is prepended to `PYTHONPATH` and `hal/sim/bin` to `PATH`: emulated `pyrealsense2`, `can`, `inputs` and `ip`, all fed by `emu/world.py` (shared memory + localhost UDP). |

Single switch: `scripts/kevin-hal <cmd...>` (reads KEVIN_HAL). Nothing in src/ or nav/ checks for sim.
Ground truth is published on a separate shm block (`kevin_gt`) that only evaluation tools read.
