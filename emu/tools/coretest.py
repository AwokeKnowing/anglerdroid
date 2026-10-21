# sim-only fault injection for the safety core (copied from /tmp/coretest.py).  python emu/tools/coretest.py A|B
"""safety-core fault injection (sim only). A: stuck RS2 writer (SIGSTOP camproc + seq left odd) 2 s.  B: kill -9 core.
F: camera freeze (sim depth cameras SIGSTOPped 2 s).  C: motion start after idle (core tick period)."""
import json, os, signal, sys, time, subprocess, urllib.request as u
sys.path.insert(0, '/home/james/gits/awokeknowing/anglerdroid/src'); sys.path.insert(0, '/home/james/gits/awokeknowing/anglerdroid')
from nav import bus
from nav.safety_core import STATE_FIELDS, CODES
from emu import gt
import numpy as np
L = bus.layouts((480, 848)); ST = bus.Block('state', L['state']); RS2 = bus.Block('rs2', L['rs2'])
def cmd(t):
    r = u.Request('http://localhost:8090/api/command', data=json.dumps({"text": t}).encode(), headers={'Content-Type': 'application/json'}); return u.urlopen(r, timeout=10).read()[:120]
def nst(): return json.load(u.urlopen('http://localhost:8090/api/state', timeout=5))
nav = int(subprocess.check_output(['pgrep', '-f', 'nav.run --port 8090']).split()[0])
kids = [int(x) for x in subprocess.check_output(['pgrep', '-P', str(nav)]).split()]
aff = {k: subprocess.check_output(['taskset', '-cp', str(k)]).decode().split(':')[1].strip() for k in kids}
core = [k for k in kids if aff[k] == '3'][0]; cam = [k for k in kids if aff[k] == '7'][0]
print('nav', nav, 'core', core, 'camproc', cam)
def rec(T, out, ev=None):
    t_end = time.monotonic() + T; last = None
    while time.monotonic() < t_end:
        sq, r = ST.try_read()
        if sq is not None:
            f = r['f']
            if f[0] != last:
                last = f[0]; d = dict(zip(STATE_FIELDS, f)); s = gt.state() if len(out) % 3 == 0 else None
                out.append(dict(t=time.monotonic(), t_dec=d['t_dec'], age1=d['age1'], age2=d['age2'], v_cmd=d['v_cmd'], v_out=d['v_out'], w_out=d['w_out'],
                                code=CODES[int(d['code'])] if 0 <= d['code'] < len(CODES) else '?', gt_v=None if s is None else s['v'], ev=ev))
        time.sleep(0.002)
mode = sys.argv[1]; R = []
if mode == 'A':
    print(cmd('forward 2')); rec(1.0, R, 'pre')
    for _ in range(40):
        rec(0.1, R, 'pre')
        if R and abs(R[-1]['v_out']) > float(os.environ.get('VMIN', '0.1')): break
    print('moving: v_out %.2f' % R[-1]['v_out'])
    os.kill(cam, signal.SIGSTOP); bumped = False
    if not int(RS2.hdr[0]) & 1: RS2.hdr[0] += 1; bumped = True       # writer "stopped mid-write"
    t_inj = time.monotonic(); print('injected stuck writer (seq odd, bumped=%s)' % bumped)
    rec(2.0, R, 'stall')
    if bumped: RS2.hdr[0] += 1
    os.kill(cam, signal.SIGCONT); t_rel = time.monotonic(); rec(3.0, R, 'post'); print(cmd('stop'))
    st = [x for x in R if x['ev'] == 'stall']; post = [x for x in R if x['ev'] == 'post']
    first_stale = next((x for x in st if x['age2'] > 0.1), None)
    print('core alive after:', os.path.exists('/proc/%d' % core), 'ticks during stall', len(st), 'max tick gap %.0f ms' % (1e3 * max(np.diff([x['t_dec'] for x in R]))))
    print('stall codes', sorted(set(x['code'] for x in st)), 'v_out during stall max %.3f' % max(abs(x['v_out']) for x in st))
    if first_stale: i = R.index(first_stale); print('first stale tick: %.0f ms after inject, age2 %.0f ms, v_out %.3f code %s; previous tick v_out %.3f code %s' % (1e3 * (first_stale['t_dec'] - t_inj + (first_stale['t'] - first_stale['t_dec']) * 0), 1e3 * first_stale['age2'], first_stale['v_out'], first_stale['code'], R[i-1]['v_out'], R[i-1]['code']))
    rec_ok = next((x for x in post if x['code'] not in ('inop_age', 'inop_cadence')), None)
    print('recovered: %s %.0f ms after release' % (rec_ok['code'] if rec_ok else None, 1e3 * (rec_ok['t'] - t_rel) if rec_ok else -1))
    print('gt v pre %.2f | during stall %s' % (max(x['gt_v'] or 0 for x in R if x['ev'] == 'pre'), [round(x['gt_v'], 2) for x in st if x['gt_v'] is not None][:12]))
elif mode == 'B':
    print(cmd('forward 2')); rec(1.0, R, 'pre')
    for _ in range(40):
        rec(0.1, R, 'pre')
        if R and abs(R[-1]['v_out']) > float(os.environ.get('VMIN', '0.1')): break
    print('moving: v_out %.2f' % R[-1]['v_out']); t_k = time.monotonic(); os.kill(core, signal.SIGKILL); print('kill -9 core', core)
    G = []
    while time.monotonic() - t_k < 12:
        s = gt.state(); n = nst(); c = n['safety']['core'] or {}
        G.append((round(time.monotonic() - t_k, 2), round(s['v'], 3), n['mode'], n['core_alarm'], c.get('alive'), c.get('restarts')))
        time.sleep(0.05)
    for g in G[:40:2] + G[40::20]: print(g)
    print('events', [e[1] for e in nst()['events'][-6:]])
    print(cmd('forward 1'))   # must be refused while alarmed / or accepted after recovery
    time.sleep(1); print('events', [e[1] for e in nst()['events'][-3:]]); print(cmd('stop'))
elif mode == 'F':   # camera freeze: the sim depth cameras (emu.depthcam: RS1 + RS2 lanes) SIGSTOPped 2 s while driving
    dc = int(subprocess.check_output(['pgrep', '-f', 'emu.depthcam']).split()[0])
    print(cmd('forward 2')); rec(1.0, R, 'pre')
    for _ in range(40):
        rec(0.1, R, 'pre')
        if R and abs(R[-1]['v_out']) > float(os.environ.get('VMIN', '0.1')): break
    print('moving: v_out %.2f' % R[-1]['v_out']); e0 = nst().get('ego_shared'); os.kill(dc, signal.SIGSTOP); t_inj = time.monotonic()
    rec(2.0, R, 'stall'); n1 = nst(); os.kill(dc, signal.SIGCONT); t_rel = time.monotonic(); rec(3.0, R, 'post'); n2 = nst(); print(cmd('stop'))
    st = [x for x in R if x['ev'] == 'stall']; first0 = next((x for x in st if abs(x['v_out']) < 1e-3), None)
    print('core alive after:', os.path.exists('/proc/%d' % core), 'ticks during freeze', len(st), 'codes', sorted(set(x['code'] for x in st)))
    if first0: print('first zero output %.0f ms after freeze (code %s, age1 %.0f ms)' % (1e3 * (first0['t'] - t_inj), first0['code'], 1e3 * first0['age1']))
    print('v_out during freeze after 150 ms max %.3f' % max([abs(x['v_out']) for x in st if x['t'] - t_inj > 0.15] or [0]))
    gv = [x['gt_v'] for x in st if x['gt_v'] is not None]; print('gt v during freeze', [round(v, 2) for v in gv[:12]])
    print('brain ego_shared before/at end of freeze/after', e0, n1.get('ego_shared'), n2.get('ego_shared'), 'brain frame_timeout', n1['timing'].get('frame_timeout', {}).get('n'))
    rec_ok = next((x for x in R if x['ev'] == 'post' and x['code'] not in ('inop_age', 'inop_cadence', 'no_depth')), None)
    print('recovered: %s %.0f ms after release' % (rec_ok['code'] if rec_ok else None, 1e3 * (rec_ok['t'] - t_rel) if rec_ok else -1))
elif mode == 'C':   # motion starts after idle (wheelbase idles axes after 5 s of zero): core tick period must stay < 33 ms
    n = int(os.environ.get('NC', '6')); S = []
    for i in range(n):
        cmd('stop'); time.sleep(6.5)                     # > idle_zero_timeout_s (5 s): axes go IDLE, next motion re-engages
        o = []; t0 = time.monotonic(); cmd('drive 0.2 0 1.0'); rec(2.5, o, ev='C%d' % i); R += o
        td = np.array([x['t_dec'] for x in o]); per = np.diff(td) * 1000
        mv = [x['t'] - t0 for x in o if x['v_out'] > 0.02]; gv = [x['t'] - t0 for x in o if x['gt_v'] is not None and x['gt_v'] > 0.02]
        S.append(dict(i=i, ticks=len(td), per_max_ms=round(float(per.max()), 1), per_p50_ms=round(float(np.median(per)), 1), n_over_33=int((per > 33.4).sum()),
                      t_vout_s=round(mv[0], 3) if mv else None, t_gtv_s=round(gv[0], 3) if gv else None)); print(S[-1], flush=True)
    cmd('stop'); R.append(dict(summary=S))
json.dump(R, open('/home/james/m1_backup/dbg/coretest_%s.json' % mode, 'w'))
