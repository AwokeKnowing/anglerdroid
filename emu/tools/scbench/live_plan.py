"""Live sim: planner-side clearance (PLAN_CLEARANCE = margin + perception allowance), planner-driven ('forward 2').
python -m emu.tools.scbench.live_plan <scenario> [n]    scenarios: wall gap4 gappass[:cm]
Per run: teleport into the lane, place test boxes, restart nav (fresh map at this pose), 'forward 2', wait for the goal.
Reports: GT contacts, min GT clearance of the SAFE footprint to the boxes, core interventions (moving ticks where the
core output != the brain command) and codes, plan_bugs delta, distance travelled, goal events."""
import json, math, os, subprocess, sys, threading, time, urllib.request as u
sys.path[:0] = ['/home/james/gits/awokeknowing/anglerdroid/src', '/home/james/gits/awokeknowing/anglerdroid']
import numpy as np
from emu import gt
from emu.tools.scbench import live as LV
S, F, X0, Y0, HB, HW = LV.S, LV.F, LV.X0, LV.Y0, LV.HB, LV.HW
REPO = '/home/james/gits/awokeknowing/anglerdroid'
def st(): return json.load(u.urlopen('http://localhost:8090/api/state', timeout=5))
def restart_nav():
    subprocess.run(['bash', '-c', 'cd %s && source emu/tools/simpin.sh && NOLOGS=1 bash ~/m1_backup/dbg/restart_nav.sh && bash emu/tools/simpin.sh apply' % REPO],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 240:
        try:
            s = st()
            if s.get('mode') == 'idle' and (s['safety'].get('core') or {}).get('code') in ('idle', 'ok'): time.sleep(4); return True
        except Exception: pass
        time.sleep(1)
    return False
def scenario(name):
    if name == 'wall':      # 1.5 m wall (5 boxes) 7 cm beside the safe footprint (outside margin + allowance + under-read)
        yb = Y0 + HW + 0.07 + HB; return [(X0 + 0.15 + 0.3 * k, yb) for k in range(5)]
    if name.startswith('gapw'):  # barrier of 6 boxes (3 per side, 0.9 m each) across the lane with the gap in the middle
        g = float(name.split(':')[1]) / 100 if ':' in name else 0.073
        o = HW + g + HB; return [(X0 + 1.0, Y0 + sg * (o + 0.3 * k)) for sg in (1, -1) for k in range(3)]
    if name.startswith('gap'):
        g = 0.04 if name == 'gap4' else float(name.split(':')[1]) / 100 if ':' in name else 0.073
        o = HW + g + HB; return [(X0 + 1.0, Y0 + o), (X0 + 1.0, Y0 - o)]
    raise SystemExit('unknown ' + name)
def run(name, T=40.0):
    boxes = scenario(name); LV.cmd('stop'); gt.ctl('clear_obstacles'); time.sleep(0.3); gt.ctl('teleport', x=X0, y=Y0, yaw=0.0); time.sleep(1.0)
    for i, b in enumerate(boxes): gt.ctl('obstacle', i=i, x=b[0], y=b[1])
    time.sleep(1.0)
    if not restart_nav(): return dict(scn=name, error='nav not ready')
    LV.ST = LV.bus.Block('state', LV.L['state'])     # nav restart recreates the shared-memory state block
    P = np.concatenate([LV.box_pts(*b) for b in boxes]); G, C, go = [], [], [True]
    def poll():
        last = None
        while go[0]:
            s = gt.state(); G.append((s['x'], s['y'], s['yaw'], s['collisions']))
            sq, r = LV.ST.try_read()
            if sq is not None and r['f'][0] != last: last = r['f'][0]; C.append(r['f'].copy())
            time.sleep(0.004)
    th = threading.Thread(target=poll); th.start(); time.sleep(0.3)
    s0 = st(); ev0 = len(s0['events']); LV.cmd('forward 2'); t0 = time.monotonic(); modes = []; BR = []
    while time.monotonic() - t0 < T:
        s = st()
        if not modes or modes[-1] != s['mode']: modes.append(s['mode'])
        BR.append((time.monotonic(), s['safety'].get('governor')))     # brain governor (v_in, w_in, v_out, w_out, need2, scale, near, n_pts, n_map)
        if s['mode'] == 'idle' and time.monotonic() - t0 > 2: break
        time.sleep(0.05)
    LV.cmd('stop'); time.sleep(0.5); go[0] = False; th.join(); s = st()
    G = np.array(G); C = np.array(C); cl = np.array([LV.clear(P, *g[:3]) for g in G[::2]])
    mv = C[(np.abs(C[:, F['v_cmd']]) > 1e-3) | (np.abs(C[:, F['w_cmd']]) > 1e-3)]
    codes = {}
    for k in mv[:, F['code']].astype(int): codes[S.CODES[k]] = codes.get(S.CODES[k], 0) + 1
    interv = int(((np.abs(mv[:, F['v_out']] - mv[:, F['v_cmd']]) > 1e-6) | (np.abs(mv[:, F['w_out']] - mv[:, F['w_cmd']]) > 1e-6)).sum())
    ev = [e[1] for e in s['events'][ev0:] if any(w in str(e) for w in ('arriv', 'unreach', 'stuck', 'going', 'escape', 'give', 'no path'))]
    det = []          # intervention ticks and plan_bugs increments: code, sd0 (cm), brain cmd -> core out
    iv = (np.abs(C[:, F['v_out']] - C[:, F['v_cmd']]) > 1e-6) | (np.abs(C[:, F['w_out']] - C[:, F['w_cmd']]) > 1e-6)
    ib = np.r_[False, np.diff(C[:, F['plan_bugs']]) > 0]
    for k in np.flatnonzero((iv & ((np.abs(C[:, F['v_cmd']]) > 1e-3) | (np.abs(C[:, F['w_cmd']]) > 1e-3))) | ib)[:12]:
        det.append(dict(k=int(k), bug=bool(ib[k]), code=S.CODES[int(C[k, F['code']])], sd0=round(100 * float(C[k, F['sd0']]), 2),
                        cmd=[round(float(C[k, F['v_cmd']]), 3), round(float(C[k, F['w_cmd']]), 3)], out=[round(float(C[k, F['v_out']]), 3), round(float(C[k, F['w_out']]), 3)],
                        vm=[round(float(C[k, F['v_m']]), 3), round(float(C[k, F['w_m']]), 3)], free=round(float(C[k, F['free']]), 3), need=round(float(C[k, F['need']]), 3),
                        age=[round(1000 * float(C[k, F['age1']]), 1), round(1000 * float(C[k, F['age2']]), 1)],
                        brain=min(BR, key=lambda q: abs(q[0] - float(C[k, F['t_dec']])))[1:] if BR else None, brain_dt=round(min(abs(q[0] - float(C[k, F['t_dec']])) for q in BR), 3) if BR else None,
                        prev_cmd=[round(float(C[k - 1, F['v_cmd']]), 3), round(float(C[k - 1, F['w_cmd']]), 3)] if k else None))
    y_at_gap = None
    if name.startswith('gap'):
        k = np.argmin(np.abs(G[:, 0] - (X0 + 1.0))); y_at_gap = round(float(G[k, 1] - Y0), 3) if abs(G[k, 0] - X0 - 1.0) < 0.05 else None
    return dict(scn=name, coll=int(G[-1, 3] - G[0, 3]), min_clear_cm=round(100 * float(cl.min()), 2), dx=round(float(G[-1, 0] - X0), 3), dy=round(float(G[-1, 1] - Y0), 3),
                y_at_gap=y_at_gap, core_interv=interv, codes=codes, ticks=len(mv), plan_bugs=int(C[-1, F['plan_bugs']] - C[0, F['plan_bugs']]),
                gov=(s['safety'].get('last') or {}).get('state'), modes=modes, events=ev[-3:], detail=det, t=round(time.monotonic() - t0, 1))
if __name__ == '__main__':
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for k in range(n): print(json.dumps(run(sys.argv[1])), flush=True)
    gt.ctl('clear_obstacles')
