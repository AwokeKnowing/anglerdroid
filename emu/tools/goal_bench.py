"""Reverse-driving / success benchmark in the live sim.  python bench.py <tag>
(Copy of ~/m1_backup/dbg/bench.py, the r3 goal bench; writes ~/m1_backup/dbg/bench_<tag>.json.)
Teleports Kevin to the env start, restarts nav (fresh SLAM), runs goals kitchen -> bedroom -> couch, then teleports
back, restarts nav and runs LightNav 'find the kitchen' (90 s).  Ground truth from emu.world (body-frame v).
BENCH_DOG=keepout|off|on (default keepout, BENCH_DOG_R=1.5 m around the start pose); needs emu.world with the 'dog' ctl."""
import json, math, os, subprocess, sys, time, urllib.request
sys.path.insert(0, '/home/james/gits/awokeknowing/anglerdroid')
from emu import gt
TAG = sys.argv[1]; OUT = os.path.expanduser('~/m1_backup/dbg/bench_%s.json' % TAG)
def post(t):
    r = urllib.request.Request("http://localhost:8095/chat", data=json.dumps({"text": t, "who": "bench"}).encode(), method="POST", headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))["reply"]
def nav(): return json.load(urllib.request.urlopen("http://localhost:8090/api/state", timeout=5))
NRES = [0]
def reset():
    NRES[0] += 1; subprocess.run('cp ~/.kevin/emu/live_nav.log ~/m1_backup/dbg/nav_%s_pre%d.log' % (TAG, NRES[0]), shell=True)
    S = json.load(open('/dev/shm/kevin_sim.json')); x, y, a = S['start']
    # dog option (BENCH_DOG): keepout (default) = dog wanders but never within BENCH_DOG_R (1.5 m) of the start pose;
    # off = dog parked for the whole bench; on = unrestricted.  (The world also moves the dog off any teleport target.)
    md = os.environ.get("BENCH_DOG", "keepout"); R_ = float(os.environ.get("BENCH_DOG_R", "1.5"))
    r_ = gt.ctl("dog", enabled=(md != "off"), keep_out=([[x, y, R_]] if md == "keepout" else []))
    print("dog:", r_, flush=True)
    gt.ctl("teleport", x=x, y=y, yaw=a); time.sleep(2)
    subprocess.run([os.path.expanduser('~/m1_backup/dbg/restart_nav.sh')], stdout=subprocess.DEVNULL); time.sleep(3)
def run(cmd, place, T, vln=False):
    s0 = gt.state(); c0 = s0['collisions']
    try: b0 = (nav().get('bump') or {}).get('stats', {}).get('events', 0)
    except Exception: b0 = 0
    rev = fwd = trev = 0.0; t0 = time.time(); t0m = time.monotonic(); tl = t0; modes = {}
    print(cmd, '->', post(cmd), flush=True)
    while time.time() - t0 < T:
        time.sleep(0.05); s = gt.state(); GTL.append([time.monotonic(), s.get('x'), s.get('y'), s.get('v'), s.get('collisions'), (s.get('dog') or {}).get('contacts'), s.get('yaw'), s.get('last_contact'), s.get('w')]); tn = time.time(); dt = tn - tl; tl = tn; v = s['v']
        if v < -0.02: rev += -v * dt; trev += dt
        elif v > 0.02: fwd += v * dt
        try: n = nav()
        except Exception: continue
        if v < -0.02: modes[n['mode']] = modes.get(n['mode'], 0) + (-v * dt)
        done = (not (n.get('vln') or {}).get('active')) if vln else n['mode'] == 'idle'
        if done and tn - t0 > 3: break
    if vln: post('stop'); time.sleep(1)
    s = gt.state(); S = json.load(open('/dev/shm/kevin_sim.json')); p = S['places'][place]
    r = dict(t0m=t0m, t1m=time.monotonic(), cmd=cmd, secs=round(time.time() - t0, 1), fwd_m=round(fwd, 2), rev_m=round(rev, 2), rev_pct=round(100 * rev / max(rev + fwd, 1e-6), 1), rev_s=round(trev, 1),
             rev_by_mode={k: round(v, 2) for k, v in modes.items()}, gt_dist=round(math.hypot(s['x'] - p[0], s['y'] - p[1]), 2), collisions=s['collisions'] - c0, result=nav()['msg'])
    nb = nav().get('bump') or {}; r.update(bumps=(nb.get('stats') or {}).get('events', 0) - b0, bump_last=nb.get('last'), bump_armed=nb.get('armed'), max_zI=(nb.get('stats') or {}).get('max_zI'))
    print(json.dumps(r), flush=True); return r
R = []; GTL = []
reset()
for pl in ('kitchen', 'bedroom', 'couch'): R.append(run('go to the ' + pl, pl, 150))
reset(); R.append(run('ask find the kitchen', 'kitchen', 90, vln=True))
post('stop'); json.dump(R, open(OUT, 'w'), indent=1); json.dump(GTL, open(OUT.replace('.json', '_gt.json'), 'w'))
tf = sum(r['fwd_m'] for r in R); tr = sum(r['rev_m'] for r in R)
print('TOTAL fwd %.2f m rev %.2f m (%.1f %%), reverse time %.1f s, collisions %d, bumps %d' % (tf, tr, 100 * tr / max(tf + tr, 1e-6), sum(r['rev_s'] for r in R), sum(r['collisions'] for r in R), sum(r.get('bumps', 0) for r in R)))
