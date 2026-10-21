"""Stationary timing bench on the REAL robot: the full nav stack (brain + safety core + RS2 camera process) on the real
RealSenses / IMU, with NO wheel base.  The CAN bus is never opened, so the ODrives (axis state, watchdog, e-stop) are
untouched and Kevin cannot move.  Behaviour of the core is unchanged; only its wheel I/O is a stub:
  - wheel speeds read 0 (robot at rest), no motor currents, twist() only records the command;
  - CAN_IFACE is set to a non-existent interface so nav's emergency_zero() fails instead of sending.
Timing caveat: the real WheelBase's encoder/watchdog threads and CAN sends (~30 Hz each) are absent here.
Usage (Orin, repo root):  ~/.venv-kevinnav/bin/python hal/nomotor/run_nomotor.py --port 8080 [nav.run args]
The stub is installed at import time of THIS file, which multiprocessing 'spawn' re-imports in every child (__mp_main__),
so the safety core process gets the stub too."""
import os, sys, threading, time, types
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(REPO, "src"), REPO):
    if p not in sys.path: sys.path.insert(0, p)
os.environ["CAN_IFACE"] = "nomotor0"
os.environ.setdefault("KEVIN_HAL", "real")


class _Axis:
    def write_property_nowait(self, *a, **k): pass


class WheelBase:
    """No-motor stand-in for src/wheelbase.WheelBase (the calls nav makes)."""
    nomotor = True

    def __init__(self, *a, **k):
        self.left, self.right = _Axis(), _Axis(); self.bus_lock = threading.Lock(); self._batt_pct = 0.0
        self.last_cmd = (0.0, 0.0, 0.0); self.n_twist = 0; self.n_nonzero = 0
        print("nomotor: WheelBase stub (CAN never opened, motors untouched)", flush=True)

    def twist(self, v, w):
        self.last_cmd = (float(v), float(w), time.monotonic()); self.n_twist += 1
        if abs(v) > 1e-3 or abs(w) > 1e-3: self.n_nonzero += 1

    def twist_for(self, v, w, *a, **k): self.twist(v, w)
    def is_twist_for_active(self): return False
    def cancel_twist_for(self): pass
    def set_safety_scales(self, *a): pass
    def set_wheel_vels(self, *a): pass
    def get_wheel_velocities_mps(self): return 0.0, 0.0
    def get_motor_currents(self): return None
    def get_encoder_health(self): return dict(encoder_ok=True, age_s=0.0, consecutive_fails=0, mode="nomotor", native_fails=0)
    def get_encoder_snapshot(self): return dict(vl=0.0, vr=0.0, tps_l=0.0, tps_r=0.0, ok=1.0, age=0.0)
    @property
    def battery_pct(self): return self._batt_pct
    def stop(self): self.twist(0.0, 0.0)
    def shutdown(self): pass


_m = types.ModuleType("wheelbase"); _m.WheelBase = WheelBase; _m.NOMOTOR = True
sys.modules["wheelbase"] = _m

if __name__ == "__main__":
    import nav.run
    sys.argv = ["nav.run"] + sys.argv[1:]
    nav.run.main()
