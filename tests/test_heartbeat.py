"""Command heartbeat (nav/hw.py) unit test with a fake wheelbase: repeats only while drive() is fresh, then one zero, then silence.
python tests/test_heartbeat.py"""
import os, sys, threading, time, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.modules.setdefault("twin_geometry", types.SimpleNamespace(RS2_T=__import__("numpy").zeros(3), RS2_R=__import__("numpy").eye(3), RS2_PITCH_DOWN_DEG=0, AXLE_H_M=0.1))
import nav.hw as HW

class FakeWB:
    def __init__(self): self.calls = []
    def twist(self, v, w): self.calls.append((time.monotonic(), v, w))

hw = object.__new__(HW.Hardware); hw.wb = FakeWB(); hw._run = True
hw._drv_lock = threading.Lock(); hw._cmd = (0.0, 0.0, 0.0); hw._hb_zeroed = True; hw.hb_stats = dict(sent=0, zeroed=0)
threading.Thread(target=hw._hb_loop, daemon=True).start()
# brain at a jittery ~4 Hz (250 ms gaps, < 0.3 s fresh window) for 2 s: wheels must see >= ~18 Hz commands
t0 = time.monotonic()
while time.monotonic() - t0 < 2.0: hw.drive(0.4, 0.1); time.sleep(0.25)
c = [x for x in hw.wb.calls if x[0] >= t0]; gaps = [b[0] - a[0] for a, b in zip(c, c[1:])]
print("fresh: %d cmds in 2 s (%.1f Hz), max gap %.0f ms" % (len(c), len(c) / 2, 1e3 * max(gaps)))
assert len(c) / 2 >= 18 and max(gaps) < 0.1
# brain dies: last drive at t1; heartbeat may repeat < 0.3 s, then exactly one zero, then nothing (watchdog stays the net)
t1 = hw._cmd[2]; time.sleep(1.5)
after = [x for x in hw.wb.calls if x[0] > t1]
nz = [x for x in after if x[1] != 0.0]; z = [x for x in after if x[1] == 0.0]
print("dead brain: last nonzero %.0f ms after last drive(), zero sent at %.0f ms, %d zeros, %d calls after zero" % (
    1e3 * (max(x[0] for x in nz) - t1) if nz else 0, 1e3 * (z[0][0] - t1), len(z), len([x for x in after if x[0] > z[0][0]])))
assert all(x[0] - t1 < HW.Hardware.HB_FRESH_S + 0.01 for x in nz) and len(z) == 1 and z[0][0] - t1 < 0.40
print("PASS")
