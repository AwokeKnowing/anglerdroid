"""Live-sim test of nav/bump.py (needs emu.world running; stops nav.run while it drives).
    KEVIN_HAL=sim scripts/kevin-hal python tests/sim_bump_reverse.py [--fresh]
1. normal driving in the open (calibration + false-positive check): straight, reverse, spins, arcs ~50 s
2. scripted REVERSE into an obstacle placed 10 cm behind the rear edge (unseen side): must detect, rear, quickly
3. forward into an obstacle: must detect, front
Ground truth: emu.world contact counter.  Kevin is stopped at the end; restart nav afterwards (restart_nav.sh)."""
import json, math, os, sys, time
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path[:0] = [R, os.path.join(R, "src")]
import nav  # noqa
from emu import gt
from nav.hw import Hardware
from nav.bump import BumpDetector
import twin_geometry as TG

fresh = "--fresh" in sys.argv
cal = os.path.expanduser("~/.kevin/bump_cal_simtest.json")
if fresh and os.path.exists(cal): os.remove(cal)
hw = Hardware(drive=True); det = BumpDetector(TG.TRACK_M, cal_path=cal)
last = [0.0, 0.0]; log = []

def tick(v, w):
    t, verts = hw.wait_rs1(300)
    if t is None: return None
    vl, vr = hw.wheel_vels(); gz = hw.yaw_rate()
    ev = det.update(t, tuple(last), (vl, vr), gz, hw.motor_currents(), getattr(hw, "accel_batch", None)); hw.accel_batch = None
    if ev: v = w = 0.0
    hw.drive(v, w); last[:] = [v, w]
    return ev

def run(v, w, secs, stop_on_event=True):
    t0 = time.monotonic(); evs = []
    while time.monotonic() - t0 < secs:
        ev = tick(v, w)
        if ev:
            s = gt.state(); evs.append((time.monotonic(), ev, s["collisions"]))
            if stop_on_event: break
    return evs

def settle(s=1.0): run(0.0, 0.0, s)
def teleport(x, y, yaw):
    gt.ctl("teleport", x=x, y=y, yaw=yaw); time.sleep(0.2); det.reset(float("inf")); settle(1.0); det.reset(-float("inf"))   # relocation is not a contact

S = json.load(open("/dev/shm/kevin_sim.json")); x0, y0, a0 = S["start"]
gt.ctl("clear_obstacles"); teleport(x0, y0, a0); settle(1.0)
c0 = gt.state()["collisions"]; d0 = (gt.state().get("dog") or {}).get("contacts", 0)
# ---- 1. normal driving
fp = []
seq = [(0.3, 0, 2.0), (0, 0, 1), (-0.15, 0, 1.5), (0, 0, 1), (0, 1.2, 2.0), (0, -1.2, 2.0), (0.25, 0.6, 2.5), (0, 0, 1),
       (-0.12, -0.4, 1.5), (0.4, 0, 1.2), (0, 0, 1.0), (0, 2.0, 1.5), (0.2, -0.8, 2.0), (-0.15, 0, 1.2), (0, 0, 1)]
rounds = 0
while (not det.armed or rounds < 2) and rounds < 6:
    teleport(x0, y0, a0)
    for v, w, s in seq: fp += run(v, w, s, stop_on_event=False)
    rounds += 1
c1 = gt.state()["collisions"]; d1 = (gt.state().get("dog") or {}).get("contacts", 0)
print("normal driving: %d rounds, armed=%s n=%d sigma_I=%s sigma_a=%s, events=%d (real contacts %d, dog %d), max z_I %.1f max z_A %.1f"
      % (rounds, det.armed, det.n_moving, det.state()["sigma_I"], det.state()["sigma_a"], len(fp), c1 - c0, d1 - d0, det.stats["max_zI"], det.stats["max_zA"]))
for _, ev, _c in fp: print("   FP?", ev["dir"], ev["why"])
det.save()
res = {}
# ---- 2/3. scripted contacts
for name, v, off in (("reverse", -0.15, -1), ("forward", 0.15, +1)):
    teleport(x0, y0, a0); settle(0.5)
    edge = -0.2737 if off < 0 else max(p[0] for p in TG.FOOTPRINT_SAFE)
    dcen = edge + off * (0.15 + 0.10)                      # box half-size 0.15, 10 cm gap
    gt.ctl("obstacle", i=0, x=x0 + dcen * math.cos(a0), y=y0 + dcen * math.sin(a0)); settle(0.5)
    cA = gt.state()["collisions"]; t0 = time.monotonic(); t_contact = None; det_ev = None; t_det = None
    while time.monotonic() - t0 < 5.0:
        ev = tick(v, 0.0)
        s = gt.state()
        if t_contact is None and s["collisions"] > cA: t_contact = time.monotonic()
        if ev: det_ev, t_det = ev, time.monotonic(); break
    settle(1.0); gt.ctl("clear_obstacles")
    lat = None if (t_contact is None or t_det is None) else t_det - t_contact
    res[name] = dict(detected=det_ev is not None, dir=det_ev and det_ev["dir"], why=det_ev and det_ev["why"], gt_contact=t_contact is not None,
                     latency_s=None if lat is None else round(lat, 3), travel_after_contact_m=None if lat is None else round(abs(v) * max(lat, 0), 3))
    print(name, json.dumps(res[name]))
settle(0.5); hw.drive(0.0, 0.0); hw.close()
ok = len(fp) == 0 and res["reverse"]["detected"] and "rear" in (res["reverse"]["dir"] or "") and res["forward"]["detected"] and "front" in (res["forward"]["dir"] or "")
json.dump(dict(fp=len(fp), res=res, sigma_I=det.state()["sigma_I"]), open(os.path.expanduser("~/m1_backup/dbg/bump_test.json"), "w"))
print("PASS" if ok else "FAIL")
