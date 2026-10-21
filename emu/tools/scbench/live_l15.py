"""L15 wedge recreate (needs world with KEVIN_EMU_TELEPORT_KEEPZ=1, nav with KEVIN_ESCAPE=1): kitchen tile -> start teleport
(keep-z: wheels 12 mm into the carpet, MuJoCo ejects Kevin), then 'go to kitchen'.  python -m emu.tools.scbench.live_l15 [n] [T]"""
import json, math, sys, time, urllib.request as u
sys.path[:0] = ['/home/james/gits/awokeknowing/anglerdroid/src', '/home/james/gits/awokeknowing/anglerdroid']
from emu import gt
def cmd(t): return u.urlopen(u.Request('http://localhost:8090/api/command', data=json.dumps({"text": t}).encode(), headers={'Content-Type': 'application/json'}), timeout=10).read()[:100]
def st(): return json.load(u.urlopen('http://localhost:8090/api/state', timeout=5))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3; T = float(sys.argv[2]) if len(sys.argv) > 2 else 120; K = (4.3, 5.9)
for k in range(n):
    cmd('stop'); gt.ctl('teleport', x=4.3, y=5.9, yaw=0.0); time.sleep(2.0)
    g0 = gt.state(); gt.ctl('teleport', x=3.877, y=1.566, yaw=1.238); time.sleep(3.0); g1 = gt.state()
    c0 = g1['collisions']; s0 = st(); b0 = (s0['safety'].get('core') or {}).get('brakes'); pb0 = (s0['safety'].get('core') or {}).get('plan_bugs')
    print('run %d: after teleport GT (%.3f, %.3f, %.2f) z-ejection collisions +%d, last_contact %s' % (k, g1['x'], g1['y'], g1['yaw'], c0 - g0['collisions'], g1.get('last_contact')), flush=True)
    cmd('go to kitchen'); t0 = time.monotonic(); modes = []; mind = 9; G = []
    while time.monotonic() - t0 < T:
        s = st(); g = gt.state(); G.append((g['x'], g['y'])); d = math.hypot(g['x'] - K[0], g['y'] - K[1]); mind = min(mind, d)
        if not modes or modes[-1] != s['mode']: modes.append(s['mode'])
        if s['mode'] == 'idle' and time.monotonic() - t0 > 3: break
        time.sleep(0.25)
    g = gt.state(); s = st(); core = s['safety'].get('core') or {}
    ev = [e for e in s['events'][-12:] if any(w in str(e) for w in ('arriv', 'stuck', 'escape', 'unreach', 'give'))]
    r = dict(run=k, t=round(time.monotonic() - t0, 1), arrived_dist=round(math.hypot(g['x'] - K[0], g['y'] - K[1]), 2), collisions=g['collisions'] - c0,
             modes=modes, core_brakes=None if b0 is None else core.get('brakes', 0) - b0, plan_bugs=None if pb0 is None else core.get('plan_bugs', 0) - pb0, events=ev[-4:])
    print(json.dumps(r), flush=True); cmd('stop'); time.sleep(1)
