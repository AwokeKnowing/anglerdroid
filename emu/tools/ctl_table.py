import json, os, sys, numpy as np
D = os.path.expanduser('~/m1_backup/dbg/')
def gtm(t, v, w):
    t, v, w = map(lambda a: np.asarray(a, float), (t, v, w)); ok = np.isfinite(v) & np.isfinite(w); t, v, w = t[ok], v[ok], w[ok]
    dt = np.diff(t); g = (dt > 1e-3) & ((np.abs(v[1:]) > 0.02) | (np.abs(w[1:]) > 0.05))
    dv = np.diff(v)[g] / dt[g]; dw = np.diff(w)[g] / dt[g]; s = 0; fl = 0
    for x in w:
        if x > 0.15: fl += s == -1; s = 1
        elif x < -0.15: fl += s == 1; s = -1
    T = dt[g].sum(); return np.sqrt(np.mean(dv ** 2)), np.sqrt(np.mean(dw ** 2)), fl / max(T, 1e-6) * 60
def ctl(files):
    E = {}; Tm = {}
    for f in files:
        if not os.path.exists(f): continue
        j = json.load(open(f)); c = j.get('ctl') or {}
        for k, v in c.items():
            if isinstance(v, list) and len(v) in (3, 4): E.setdefault(k, []).append(v)
        for k, x in (j.get('timing') or {}).items():
            if isinstance(x, dict) and 'p50' in x: Tm.setdefault(k, []).append(x)
    e = {k: (sum(x[0] * x[-1] for x in L) / max(sum(x[-1] for x in L), 1)) for k, L in E.items()}
    tm = {k: (sum(x['mean'] * x['n'] for x in L) / sum(x['n'] for x in L), float(np.median([x['p50'] for x in L]))) for k, L in Tm.items()}
    return e, tm
def row(M, kind, i):
    if kind == 'bench':
        b = D + 'bench_ctl_%s%d' % (M, i)
        if not os.path.exists(b + '.json'): return None
        R = json.load(open(b + '.json')); G = json.load(open(b + '_gt.json'))
        goals = sum(1 for r in R[:3] if r['result'].startswith('arrived') or 'short of' in r['result']); secs = sum(r['secs'] for r in R[:3])
        coll = sum(r['collisions'] for r in R); sm = gtm([g[0] for g in G], [g[3] if g[3] is not None else np.nan for g in G], [g[8] if g[8] is not None else np.nan for g in G])
        e, tm = ctl([b + '_ctl_pre2.json', b + '_ctl_end.json']); res = '%d/3 goals %.0fs' % (goals, secs)
        rt = [r['rtf'] for r in R if 'rtf' in r]; rtf = '%.2f' % min(rt) if rt else '-'
    else:
        f = D + 'drift_ctl_%s_fridge%d' % (M, i)
        if not os.path.exists(f + '.jsonl'): return None
        L = [json.loads(l) for l in open(f + '.jsonl')]; c = json.load(open(f + '_ctl.json')) if os.path.exists(f + '_ctl.json') else {}
        coll = c.get('collisions'); sm = gtm([l.get('tm', l['t']) for l in L], [l.get('v', np.nan) for l in L], [l.get('w', np.nan) for l in L])
        e, tm = ctl([f + '_ctl.json']); res = '%.0fs %.1fm %s' % (L[-1]['t'], L[-1]['path'], (c.get('msg') or '')[:14])
        rtf = '%.2f' % c['rtf'] if 'rtf' in c else '-'
    g = lambda k, i=1: tm[k][i] if k in tm else float('nan')
    per = g('ctl_period_ms') if 'ctl_period_ms' in tm else g('period_ms')
    f2c = g('frame_to_iss_ms') if 'frame_to_iss_ms' in tm else g('frame_to_cmd_ms')
    return [M, kind[0] + str(i), res, coll, rtf, '%.0f' % per, '%.0f' % g('percep_period_ms') if 'percep_period_ms' in tm else '%.0f' % g('period_ms'), '%.0f' % f2c,
            '%.0f' % g('sample_to_effect_ms') if 'sample_to_effect_ms' in tm else '-', '%.1f' % g('track_err_cm', 0) if 'track_err_cm' in tm else '-',
            '%.2f/%.2f' % (e.get('pred_pos', np.nan) * 100, e.get('raw_pos', np.nan) * 100), '%.2f/%.2f' % (e.get('pred_yaw', np.nan), e.get('raw_yaw', np.nan)),
            '%.3f/%.3f' % (e.get('pred_v', np.nan), e.get('raw_v', np.nan)), '%.3f/%.3f' % (e.get('pred_w', np.nan), e.get('raw_w', np.nan)),
            '%.2f' % sm[0], '%.2f' % sm[1], '%.1f' % sm[2]]
H = ['mode', 'run', 'result', 'coll', 'sim RTF min', 'ctl per p50', 'RS1 per p50', 'frame->cmd p50', 'sample->eff p50', 'track cm mean', 'pos cm pred/raw', 'yaw deg pred/raw', 'v pred/raw', 'w pred/raw', 'rms dv/dt', 'rms dw/dt', 'w flips/min']
print('| ' + ' | '.join(H) + ' |'); print('|' + '---|' * len(H))
for M in sys.argv[1:] or ['pre', 'cmd']:
    for kind in ('bench', 'fridge'):
        for i in (1, 2, 3):
            r = row(M, kind, i)
            if r: print('| ' + ' | '.join(str(x) for x in r) + ' |')
