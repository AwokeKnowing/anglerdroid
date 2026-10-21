"""Run emu.world's real loop (physics + render thread + CamFX) in a sandbox next to a live emulator: shm blocks get a
't_' prefix, info/truth/GT-map files go to /dev/shm/t_*, CAN on another port, no viewer.  Prints the world's own
'cam Hz' lines for N seconds.  python bench/slam/world_selftest.py [secs]"""
import os, sys, threading, time
os.environ.update(KEVIN_EMU_INFO="/dev/shm/t_kevin_emu.json", KEVIN_EMU_TRUTH="/dev/shm/t_kevin_emu_truth.json", KEVIN_CAN_PORT="47910",
                  MUJOCO_GL="egl", KEVIN_EMU_VIEW="0")
os.environ.setdefault("KEVIN_EMU_MODEL", "full"); os.environ.setdefault("KEVIN_EMU_NOISE", "realistic")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path[:0] = [REPO]
import numpy as np
from emu import shm
_B, _I = shm.Block.__init__, shm.ImuRing.__init__
shm.Block.__init__ = lambda self, name, layout, create=False: _B(self, "t_" + name, layout, create)
shm.ImuRing.__init__ = lambda self, name, create=False: _I(self, "t_" + name, create)
_sz = np.savez_compressed
np.savez_compressed = lambda f, *a, **k: _sz(f.replace("/dev/shm/kevin_", "/dev/shm/t_kevin_") if isinstance(f, str) else f, *a, **k)
from emu import world
import argparse
a = argparse.Namespace(env="casita", seed=0, subpix=0.08, holes=0.004, min_z=0.175, band=True, tire="stock", ir=1, rt=1.0)
w = world.World(a)
threading.Thread(target=w.run, daemon=True).start()
time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 35)
for n in ("t_kevin_rs1", "t_kevin_rs2", "t_kevin_imu", "t_kevin_gt", "t_kevin_mast"):
    try: os.unlink("/dev/shm/" + n)
    except FileNotFoundError: pass
os._exit(0)
