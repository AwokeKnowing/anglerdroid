"""Real-log sanity maps (no ground truth): the shared occmap mapper posed by each method's trajectory, from zero,
in the robot's start frame; one panel per method + map sharpness (fraction of known cells that are occupied: a
blurred/double-walled map from drift has more, thicker occupied cells) and loop-closure gap (end vs start pose; the
real run ended with a drive back to its start, nav goal (-0.37, 0.03) in odom).
python bench/slam/real_maps.py DATA RESULTS OUT.png"""
import glob, json, math, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import occmap
from eval import load_tum
seq, res, out = sys.argv[1:4]; panels = []; rows = []
for p in sorted(glob.glob(os.path.join(res, "*.tum"))):
    name = os.path.basename(p)[:-4]; t, T = load_tum(p)
    if len(t) < 10: continue
    g = occmap.build(seq, T, t, gt_shape=(450, 450), gt_org=(-9.0, -9.0))
    k = g.known(); occ_frac = float(g.occ()[k].mean()) if k.any() else 0
    xy = T[:, :2, 3]; gap = float(np.hypot(*(xy[-1] - xy[0])))
    rows.append(dict(method=name, occ_frac_of_known=round(occ_frac, 3), known_m2=round(float(k.sum()) * occmap.RES ** 2, 1), end_minus_start_m=round(gap, 3)))
    img = np.full(g.L.shape + (3,), 128, np.uint8); img[g.free()] = (235, 235, 235); img[g.occ()] = (30, 30, 30); img = img[::-1].copy()
    rc = g.rc(xy); pts = np.stack([rc[:, 1], g.L.shape[0] - 1 - rc[:, 0]], 1).astype(np.int32); cv2.polylines(img, [pts], False, (0, 0, 255), 1)
    cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2); panels.append(img)
while len(panels) % 3: panels.append(np.full_like(panels[0], 255))
cv2.imwrite(out, np.vstack([np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]))
json.dump(rows, open(os.path.join(res, "real_maps.json"), "w"), indent=1); [print(json.dumps(r)) for r in rows]
