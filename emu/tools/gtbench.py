"""python gtbench.py <label> <tag>... : ground-truth table for goal benches (works for any safety variant):
goals arrived/attempted, contacts (world collisions + dog contacts), 3-D and 2-D footprint gap to the GT world
(emu/tools/gtgap) over samples where Kevin moves (|v| > 2 cm/s or |w| > 0.1 rad/s): min and 5th percentile."""
import json, sys, numpy as np
sys.path.insert(0, '/home/james/gits/awokeknowing/anglerdroid/emu/tools'); import gtgap as GG
F = GG.load(); label, tags = sys.argv[1], sys.argv[2:]
arr = att = col = 0; g3, g2 = [], []; fwd = 0.0
for tag in tags:
    R = json.load(open('bench_%s.json' % tag)); GT = json.load(open('bench_%s_gt.json' % tag))
    for r in R: att += 1; arr += r['result'].startswith('arrived'); col += r['collisions']; fwd += r['fwd_m']
    for g in GT[::2]:
        if g[1] is None: continue
        if abs(g[3] or 0) > 0.02 or abs(g[8] or 0) > 0.1: g3.append(GG.gap(F, g[1], g[2], g[6])); g2.append(GG.gap2d(F, g[1], g[2], g[6]))
g3, g2 = np.array(g3), np.array(g2)
print("| %s | goals %d/%d | fwd %.1f m | contacts %d | 3-D gap min %.3f p5 %.3f | 2-D gap min %.3f p5 %.3f | n %d |" % (
    label, arr, att, fwd, col, g3.min(), np.percentile(g3, 5), g2.min(), np.percentile(g2, 5), len(g3)))
