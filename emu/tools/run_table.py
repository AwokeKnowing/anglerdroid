"""Drift/goal table from emu.eval output dirs (per-env run.json files, not summaries).
python emu/tools/run_table.py DIR[:label] ... [--envs room maze hall]
  drift = |gt - slam| at the end / ground-truth path length; yaw = max heading error; g = goals reached; c = contact events"""
import glob, json, math, os, statistics as st, sys
a = sys.argv[1:]; envs = ["room", "maze", "hall"]
if "--envs" in a: i = a.index("--envs"); envs = a[i + 1:]; a = a[:i]
W = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
for arg in a:
    d, _, lab = arg.partition(":"); lab = lab or os.path.basename(d.rstrip("/")); out = []
    for e in envs:
        r = []
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(d), "%s_s*/run.json" % e))):
            R = json.load(open(f)); tr = R["track"]; res = R["results"]
            L = sum(math.hypot(p["gt"][0] - q["gt"][0], p["gt"][1] - q["gt"][1]) for p, q in zip(tr, tr[1:])) or 1
            fe = math.hypot(tr[-1]["gt"][0] - tr[-1]["est"][0], tr[-1]["gt"][1] - tr[-1]["est"][1])
            yw = max(abs(math.degrees(W(p["gt"][2] - p["est"][2]))) for p in tr)
            r.append((f.split("_s")[-1][0], 100 * fe / L, yw, sum(x["ok"] for x in res), len(res), sum(x["collisions"] for x in res)))
        if r: out.append("%s %.1f%% [%.1f-%.1f] yaw%.0f g%d/%d c%d s%s" % (e, st.median(x[1] for x in r), min(x[1] for x in r), max(x[1] for x in r),
                                                                       st.median(x[2] for x in r), sum(x[3] for x in r), sum(x[4] for x in r), sum(x[5] for x in r), "".join(x[0] for x in r)))
    print(lab, "|", " ; ".join(out))
