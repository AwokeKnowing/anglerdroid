"""python benchsafe2.py <safelog> <label> <tag>... : per-variant table (3-D GT gap from emu/tools/gtgap) + event classes"""
import json, sys, numpy as np, collections
sys.path.insert(0, '/home/james/gits/awokeknowing/anglerdroid/emu/tools'); import gtgap as GG
F = GG.load(); M = 0.035
SL = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; T = np.array([r[0] for r in SL]); DRV = ('path', 'backout', 'turn')
label, tags = sys.argv[2], sys.argv[3:]
tot = collections.Counter(); gaps3, gaps2 = [], []; EV = []; per_goal = collections.defaultdict(lambda: [0, 0])
for tag in tags:
    R = json.load(open('bench_%s.json' % tag)); GT = json.load(open('bench_%s_gt.json' % tag)); gt = np.array([g[0] for g in GT])
    for r in R:
        per_goal[r['cmd']][1] += 1; per_goal[r['cmd']][0] += r['result'].startswith('arrived')
        i0, i1 = np.searchsorted(T, r['t0m']), np.searchsorted(T, r['t1m']); rows = SL[i0:i1]
        drv = [k for k, x in enumerate(rows) if x[17] in DRV]
        tot['min'] += sum(min(rows[k + 1][0] - rows[k][0], 0.2) for k in drv if k + 1 < len(rows)) / 60
        prev = 'ok'
        for k in drv:
            s = rows[k][9]; tot['ticks'] += 1; tot['nok'] += s != 'ok'
            if s != 'ok' and prev == 'ok':
                x = rows[k]; j = min(np.searchsorted(gt, x[1]), len(GT) - 1); g = GT[j]
                g3, g2 = GG.gap(F, g[1], g[2], g[6]), GG.gap2d(F, g[1], g[2], g[6])
                near = [(n, p[0][2]) for n, p in zip(('rs1', 'rs2', 'map'), x[18:21] if len(x) > 20 else (None,) * 3) if p]
                src = min(near, key=lambda q: q[1]) if near else (None, None)
                kind = 'no_depth' if s == 'no_depth' else ('stop' if x[10] == 0.0 else 'limit')
                clr = x[11]
                if s == 'no_depth': cls = 'no_depth (stale/invalid frame)'
                elif g3 < M: cls = 'A real: 3-D gap < 3.5 cm'
                elif g2 < M: cls = 'B overhang: 2-D gap < 3.5 cm, 3-D clear'
                elif clr is not None and clr <= M + 0.005 and g2 > 0.10: cls = 'C perceived at margin, GT >= 10 cm (perception)'
                elif clr is not None and clr <= M + 0.005: cls = 'D at margin, GT 3.5-10 cm'
                else: cls = 'E envelope sweep (clr > margin): speed/approach'
                EV.append(dict(tag=tag, cmd=r['cmd'], kind=kind, cls=cls, v=x[3], w=x[4], clr=clr, g3=g3, g2=g2, src=src[0], sd=src[1], mode=x[17]))
            prev = s
        j0, j1 = np.searchsorted(gt, r['t0m']), np.searchsorted(gt, r['t1m'])
        mv = [g for g in GT[j0:j1] if g[6] is not None and abs(g[3] or 0) > 0.02]
        gaps3 += [GG.gap(F, g[1], g[2], g[6]) for g in mv]; gaps2 += [GG.gap2d(F, g[1], g[2], g[6]) for g in mv]
        tot['coll'] += r['collisions']; tot['goals'] += r['result'].startswith('arrived'); tot['n'] += 1
ne = len(EV); c = collections.Counter(e['kind'] for e in EV)
print('| %s | %d runs | goals %d/%d | driving %.1f min | events %d (stop %d / limit %d / no_depth %d) = %.2f/min | non-ok ticks %.1f%% | 3-D GT min %.3f p5 %.3f, <3.5 cm %d/%d | 2-D min %.3f p5 %.3f | collisions %d |' % (
    label, len(tags), tot['goals'], tot['n'], tot['min'], ne, c['stop'], c['limit'], c['no_depth'], ne / max(tot['min'], 1e-6), 100 * tot['nok'] / max(tot['ticks'], 1),
    min(gaps3), np.percentile(gaps3, 5), sum(g < M for g in gaps3), len(gaps3), min(gaps2), np.percentile(gaps2, 5), tot['coll']))
print('   per goal:', {k.replace('go to the ', ''): '%d/%d' % tuple(v) for k, v in per_goal.items()})
for cl, n in sorted(collections.Counter(e['cls'] for e in EV).items()):
    E = [e for e in EV if e['cls'] == cl]
    print('   %-48s %3d | |v| med %.2f | src %s | modes %s' % (cl, n, np.median([abs(e['v']) for e in E]), dict(collections.Counter(e['src'] for e in E)), dict(collections.Counter(e['mode'] for e in E))))
json.dump(EV, open('events_%s.json' % label, 'w'))
