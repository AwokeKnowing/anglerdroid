"""Minimal live sim confirmation of the wired designed-decel core (sim only; uses the mocap test boxes, 0.30 m cubes).
python -m emu.tools.scbench.live <scenario> [n]      scenarios: headon30 headon45 wall45 squeeze4 wedge
Per run: teleport into the clear lane (X0, Y0, yaw 0), place boxes, 'drive v w T', poll GT (~150 Hz) and the core state
block (every decision).  Reports: GT collisions (delta), min GT clearance of the SAFE footprint to the boxes, core codes,
interventions, plan_bugs delta, decide-time from the core log if KEVIN_SAFETY_CORE_LOG is set."""
import json, math, os, sys, threading, time, urllib.request as u
sys.path[:0] = ['/home/james/gits/awokeknowing/anglerdroid/src', '/home/james/gits/awokeknowing/anglerdroid']
import numpy as np
from emu import gt
from nav import bus, safety_core as S
X0, Y0 = float(os.environ.get('X0', 3.6)), float(os.environ.get('Y0', 2.2)); HB = 0.15
HW = float(np.abs(S.FOOTPRINT_SAFE[:, 1]).max()); XF = float(S.FOOTPRINT_SAFE[:, 0].max())
L = bus.layouts((480, 848)); ST = bus.Block('state', L['state']); F = {n: i for i, n in enumerate(S.STATE_FIELDS)}
def cmd(t): return u.urlopen(u.Request('http://localhost:8090/api/command', data=json.dumps({"text": t}).encode(), headers={'Content-Type': 'application/json'}), timeout=10).read()[:80]
def box_pts(cx, cy, step=0.005):
    s = np.arange(-HB, HB + 1e-9, step); e = np.full_like(s, HB)
    return np.concatenate([np.c_[s, e], np.c_[s, -e], np.c_[e, s], np.c_[-e, s]]) + (cx, cy)
def clear(P, x, y, yaw):
    d = P - (x, y); c, s = math.cos(yaw), math.sin(yaw); B = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]
    return float(S.sd(B).min())
def scenario(name):
    """-> (start pose, boxes [(x, y)], v, w, T, pass(rec))"""
    if name.startswith('headon'):
        v = int(name[6:]) / 100; return (X0, Y0, 0.0), [(X0 + 2.2, Y0)], v, 0.0, 3.0 / v + 2, lambda r: r['coll'] == 0 and S.MARGIN - 0.01 <= r['min_clear'] < 0.30
    if name == 'wall45':   # 1.2 m wall of 4 boxes, its face 4.5 cm beside the safe footprint (margin 3.5): drive past untouched
        yb = Y0 + HW + 0.045 + HB; return (X0, Y0, 0.0), [(X0 + 0.6 + 0.3 * k, yb) for k in range(4)], 0.30, 0.0, 7.0, \
            lambda r: r['coll'] == 0 and r['interv'] == 0 and r['dx'] > 1.8
    if name == 'squeeze4':  # gap = safe width + 2 x 4 cm, 1.0 m ahead: pass through untouched
        o = HW + 0.04 + HB; return (X0, Y0, 0.0), [(X0 + 1.0, Y0 + o), (X0 + 1.0, Y0 - o)], 0.25, 0.0, 7.0, \
            lambda r: r['coll'] == 0 and r['interv'] == 0 and r['dx'] > 1.5
    if name == 'wedge':    # box face 1.5 cm ahead of the front (inside the 3.5 cm margin, no contact), spin in place: front-right corner closes -> refuse
        return (X0, Y0, 0.0), [(X0 + XF + 0.015 + HB, Y0 + 0.10)], 0.0, 0.6, 3.0, \
            lambda r: r['coll'] == 0 and r['deepen'] <= 0.002 and r['codes'].get('collide', 0) > 0
    if name == 'wedge_out':  # same in-margin start, reverse out: must pass (never collide); the lane has furniture ~0.6 m behind (brake there is correct)
        return (X0, Y0, 0.0), [(X0 + XF + 0.015 + HB, Y0 + 0.10)], -0.15, 0.0, 2.5, \
            lambda r: r["coll"] == 0 and r["deepen"] <= 0.002 and r["dx"] < -0.15 and r["codes"].get("collide", 0) == 0
    raise SystemExit('unknown scenario ' + name)
def run(name):
    (x, y, yaw), boxes, v, w, T, ok = scenario(name)
    cmd('stop'); gt.ctl('clear_obstacles'); time.sleep(0.5); gt.ctl('teleport', x=x, y=y, yaw=yaw); time.sleep(1.0)
    for i, (bx, by) in enumerate(boxes): gt.ctl('obstacle', i=i, x=bx, y=by)
    time.sleep(2.0); P = np.concatenate([box_pts(*b) for b in boxes]); g0 = gt.state(); c_start = clear(P, g0['x'], g0['y'], g0['yaw'])
    G, C, run_ = [], [], [True]
    def poll():
        last = None
        while run_[0]:
            s = gt.state(); G.append((s['x'], s['y'], s['yaw'], s['collisions']))
            sq, r = ST.try_read()
            if sq is not None and r['f'][0] != last: last = r['f'][0]; C.append(r['f'].copy())
            time.sleep(0.004)
    th = threading.Thread(target=poll); th.start(); time.sleep(0.2); f0 = None
    cmd('drive %.2f %.2f %.1f' % (v, w, T)); time.sleep(T + 1.5); cmd('stop'); time.sleep(0.5); run_[0] = False; th.join()
    G = np.array(G); C = np.array(C); cl = np.array([clear(P, *g[:3]) for g in G[::3]])
    moving = C[(np.abs(C[:, F['v_cmd']]) > 1e-3) | (np.abs(C[:, F['w_cmd']]) > 1e-3)]
    codes = {}
    for k in moving[:, F['code']].astype(int): codes[S.CODES[k]] = codes.get(S.CODES[k], 0) + 1
    interv = int(((np.abs(moving[:, F['v_out']] - moving[:, F['v_cmd']]) > 1e-6) | (np.abs(moving[:, F['w_out']] - moving[:, F['w_cmd']]) > 1e-6)).sum())
    r = dict(scn=name, coll=int(G[-1, 3] - G[0, 3]), start_clear=round(c_start, 4), min_clear=round(float(cl.min()), 4), end_clear=round(float(cl[-1]), 4),
             deepen=round(max(0.0, min(c_start, S.MARGIN) - float(cl.min())), 4), dx=round(float(G[-1, 0] - G[0, 0]), 3), dyaw=round(float(G[-1, 2] - G[0, 2]), 3),
             codes=codes, interv=interv, ticks=len(moving), plan_bugs=int(C[-1, F['plan_bugs']] - C[0, F['plan_bugs']]) if len(C) else None,
             period_max_ms=round(1e3 * float(np.nanmax(C[1:, F['period']])), 1) if len(C) > 1 else None)
    r['PASS'] = bool(ok(r)); gt.ctl('clear_obstacles'); return r
if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3; out = []
    for k in range(n):
        r = run(sys.argv[1]); out.append(r); print(json.dumps(r), flush=True)
    print('%s: %d/%d PASS' % (sys.argv[1], sum(r['PASS'] for r in out), n), flush=True)
