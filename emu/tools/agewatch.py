"""Continuous safety-core frame-age watchdog for heavy sim-side jobs (read-only on nav: polls /api/state).
python -m emu.tools.agewatch <log> [--kill-file F] [--period 0.03]
Every core decision (new t_dec) records age1/age2 (camera frame age at decision).  Every 10 s one summary line.
Any age > 0.100 s: ALERT line, and every process group id listed in the kill file (one per line) gets SIGTERM, then
SIGKILL after 3 s -- the heavy job is aborted, the sim stack is not touched."""
import json, os, signal, sys, time, urllib.request
log = sys.argv[1]; kf = None; per = 0.03
if '--kill-file' in sys.argv: kf = sys.argv[sys.argv.index('--kill-file') + 1]
if '--period' in sys.argv: per = float(sys.argv[sys.argv.index('--period') + 1])
f = open(log, 'a', buffering=1); last = None; W = []; N = 0; NB = 0; MX = 0.0; tw = time.time(); miss = 0
def kill():
    if not kf or not os.path.exists(kf): return
    for l in open(kf).read().split():
        try: os.killpg(int(l), signal.SIGTERM); f.write('%s ABORT SIGTERM pgid %s\n' % (time.strftime('%T'), l))
        except Exception as e: f.write('abort %s: %s\n' % (l, e))
    time.sleep(3)
    for l in open(kf).read().split():
        try: os.killpg(int(l), signal.SIGKILL)
        except Exception: pass
f.write('%s agewatch start period %.3f kill-file %s\n' % (time.strftime('%T'), per, kf))
while True:
    try: c = json.load(urllib.request.urlopen('http://localhost:8090/api/state', timeout=2))['safety']['core']
    except Exception as e: f.write('%s state error %s\n' % (time.strftime('%T'), e)); time.sleep(1); continue
    if c['t_dec'] != last:
        if last is not None and c['t_dec'] - last > 0.06: miss += 1        # skipped >=1 decision between polls
        last = c['t_dec']; a = max(c['age1'] or 0, c['age2'] or 0); W.append(a); N += 1; MX = max(MX, a)
        if a > 0.100:
            NB += 1; f.write('%s ALERT age %.0f ms (age1 %.0f age2 %.0f)\n' % (time.strftime('%T'), 1e3 * a, 1e3 * c['age1'], 1e3 * c['age2'])); kill()
    if time.time() - tw >= 10:
        if W: f.write('%s n %d max %.0f ms >100: %d | total n %d max %.0f >100: %d | gaps %d\n' % (time.strftime('%T'), len(W), 1e3 * max(W), sum(x > 0.1 for x in W), N, 1e3 * MX, NB, miss))
        W = []; tw = time.time()
    time.sleep(per)
