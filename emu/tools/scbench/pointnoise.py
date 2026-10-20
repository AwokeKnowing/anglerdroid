"""Per-point depth noise of the safety core's input points (sim): the core's 5 Hz viz snapshot (its merged point set,
robot frame) beside a test wall at a known GT position.  Per point: under-read u = wall face - point (world, along the
wall normal; + = point closer to the robot than the wall).  Still (3 s) and slow drift past the wall (drive 0.10 m/s).
-> per snapshot: mean / sd / min of u; per 1 cm wall bin: temporal sd across snapshots; the tick-min vs the bin means."""
import json, math, sys, time
sys.path[:0] = ['/home/james/gits/awokeknowing/anglerdroid/src', '/home/james/gits/awokeknowing/anglerdroid']
import numpy as np
from emu import gt
from emu.tools.scbench import live as LV
from nav import bus
X0, Y0, HB, HW = LV.X0, LV.Y0, LV.HB, LV.HW
VIZ = bus.Block('viz', bus.layouts((480, 848))['viz'])
def snaps(T, face, xs, drive=None):
    out, last = [], None; t_end = time.monotonic() + T
    if drive: LV.cmd(drive)
    while time.monotonic() < t_end:
        sq, r = VIZ.try_read()
        if sq is not None and float(r['t'][0]) != last:
            last = float(r['t'][0]); n = int(r['n'][0]); P = r['pts'][:n].astype(float).copy(); g = gt.state()
            c, s = math.cos(g['yaw']), math.sin(g['yaw']); W = np.c_[g['x'] + c * P[:, 0] - s * P[:, 1], g['y'] + s * P[:, 0] + c * P[:, 1]]
            m = (np.abs(W[:, 1] - face) < 0.05) & (W[:, 0] > xs[0]) & (W[:, 0] < xs[1])
            out.append(dict(t=last, x=W[m, 0].tolist(), u=(face - W[m, 1]).tolist(), v=g['v']))
        time.sleep(0.01)
    if drive: LV.cmd('stop')
    return out
def setup(side):
    LV.cmd('stop'); gt.ctl('clear_obstacles'); time.sleep(0.3); gt.ctl('teleport', x=X0, y=Y0, yaw=0.0); time.sleep(0.8)
    yb = Y0 + HW + side + HB; B = [(X0 - 0.45 + 0.3 * k, yb) for k in range(6)]
    for i, b in enumerate(B): gt.ctl('obstacle', i=i, x=b[0], y=b[1])
    time.sleep(1.5); return yb - HB, (X0 - 0.6, X0 + 1.2)
R = {}
face, xs = setup(0.06); R['still6'] = snaps(4.0, face, xs)
face, xs = setup(0.06); R['drift6'] = snaps(5.0, face, xs, 'drive 0.10 0 4.5')
face, xs = setup(0.04); R['still4'] = snaps(4.0, face, xs)
gt.ctl('clear_obstacles'); LV.cmd('stop'); gt.ctl('teleport', x=3.6, y=2.2, yaw=1.5708)
json.dump(R, open('/home/james/slamlogs/res/pointnoise_1020.json', 'w'))
for k, L in R.items():
    L = [d for d in L if len(d['u']) > 20]
    if not L: print(k, 'no points'); continue
    allu = np.concatenate([d['u'] for d in L]); mins = np.array([max(d['u']) for d in L])
    within = np.array([np.std(d['u']) for d in L])
    # temporal per-point noise: 1 cm wall bins, mean u per bin per snapshot, sd across snapshots minus nothing
    X = np.concatenate([d['x'] for d in L]); U = allu; I = np.concatenate([[i] * len(d['u']) for i, d in enumerate(L)])
    b = np.floor(X / 0.01).astype(int); tsd = []
    for bb in np.unique(b):
        q = b == bb; per = [U[q & (I == i)].mean() for i in np.unique(I[q])]
        if len(per) >= 4: tsd.append(np.std(per))
    print('%-7s snaps %2d pts/snap %4d | u mean %.2f sd %.2f p1 %.2f p99 %.2f max %.2f cm | within-snap sd %.2f | per-bin temporal sd median %.2f p90 %.2f cm | snap max-u mean %.2f sd %.2f' % (
        k, len(L), len(allu) // len(L), 100 * U.mean(), 100 * U.std(), 100 * np.percentile(U, 1), 100 * np.percentile(U, 99), 100 * U.max(), 100 * within.mean(),
        100 * np.median(tsd) if tsd else np.nan, 100 * np.percentile(tsd, 90) if tsd else np.nan, 100 * mins.mean(), 100 * mins.std()))
