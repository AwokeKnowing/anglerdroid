"""Perception under-read probe (sim): core sd0 (and brain governor clearance) vs GT clearance of the SAFE footprint.
Static: robot in the lane, a 1.2 m wall of test boxes beside it (side) or a box face ahead (front) at known GT clearances;
3 s of core decisions each.  Moving: drive 0.25 m/s past the wall at 6 cm.  -> per tick (gt, core sd0, brain clr)."""
import json, math, sys, time, threading, urllib.request as u
sys.path[:0] = ['/home/james/gits/awokeknowing/anglerdroid/src', '/home/james/gits/awokeknowing/anglerdroid']
import numpy as np
from emu import gt
from emu.tools.scbench import live as LV
S = LV.S; F = LV.F; X0, Y0, HB = LV.X0, LV.Y0, LV.HB
def st(): return json.load(u.urlopen('http://localhost:8090/api/state', timeout=5))
def setup(boxes):
    LV.cmd('stop'); gt.ctl('clear_obstacles'); time.sleep(0.3); gt.ctl('teleport', x=X0, y=Y0, yaw=0.0); time.sleep(0.8)
    for i, b in enumerate(boxes): gt.ctl('obstacle', i=i, x=b[0], y=b[1])
    time.sleep(1.5); return np.concatenate([LV.box_pts(*b) for b in boxes])
def collect(P, T, drive=None):
    out = []; last = None; t_end = time.monotonic() + T; tb = 0
    if drive: LV.cmd(drive)
    while time.monotonic() < t_end:
        sq, r = LV.ST.try_read()
        if sq is not None and r['f'][0] != last:
            last = r['f'][0]; g = gt.state(); c = LV.clear(P, g['x'], g['y'], g['yaw'])
            bc = None
            if time.monotonic() - tb > 0.2:
                tb = time.monotonic()
                try: bc = st()['safety']['last'].get('clearance')
                except Exception: pass
            out.append((c, float(r['f'][F['sd0']]), bc, float(r['f'][F['v_m']])))
        time.sleep(0.003)
    if drive: LV.cmd('stop')
    return out
R = {}
for side_cm in (3.0, 4.0, 5.0, 6.0, 8.0):
    yb = Y0 + LV.HW + side_cm / 100 + HB; P = setup([(X0 - 0.15 + 0.3 * k, yb) for k in range(4)]); R['side%.0f' % side_cm] = collect(P, 3.0)
for f_cm in (4.0, 6.0):
    P = setup([(X0 + LV.XF + f_cm / 100 + HB, Y0)]); R['front%.0f' % f_cm] = collect(P, 3.0)
yb = Y0 + LV.HW + 0.06 + HB; P = setup([(X0 + 0.3 + 0.3 * k, yb) for k in range(5)]); R['move6'] = collect(P, 6.0, 'drive 0.25 0 5.5')
gt.ctl('clear_obstacles'); LV.cmd('stop')
json.dump(R, open('/home/james/slamlogs/res/noiseprobe_1020.json', 'w'))
for k, L in R.items():
    a = np.array([(x[0], x[1]) for x in L if np.isfinite(x[1]) and x[0] < 0.2]); b = np.array([(x[0], x[2]) for x in L if x[2] is not None and x[0] < 0.2])
    if not len(a): print(k, 'no data'); continue
    e = a[:, 0] - a[:, 1]
    eb = (b[:, 0] - b[:, 1]) if len(b) else np.array([np.nan])
    print('%-7s n %3d GT %.1f cm | core under-read mean %.2f sd %.2f p99 %.2f max %.2f cm | brain under-read mean %.2f p99 %.2f (n %d)' % (k, len(a), 100 * np.median(a[:, 0]), 100 * e.mean(), 100 * e.std(), 100 * np.percentile(e, 99), 100 * e.max(), 100 * np.nanmean(eb), 100 * np.nanpercentile(eb, 99), len(b)))
