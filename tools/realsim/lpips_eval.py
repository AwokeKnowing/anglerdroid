"""Masked LPIPS(alex)/SSIM of camera-model sim frames vs the real mast frames (people + OSD masked), + side-by-sides.
usage: lpips_eval.py <sim_dir with <look>_mast_sim.png> <real_dir> <out_dir>"""
import json, os, sys
import numpy as np, cv2, torch, lpips
from skimage.metrics import structural_similarity as ssim
S, REAL, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
LOOKS = {"20261021-211921_m0": [(300, 250, 600, 500)], "20261021-212158_g2_after": [(0, 50, 520, 600)],
         "20261021-213609_t3_turnR_after": [(350, 220, 440, 560), (280, 340, 600, 600)],
         "20261021-213640_t4_turn75_after": [], "20261021-213708_t5_turn90_after": [], "20261021-213719_t5_turn90_after": []}
LP = lpips.LPIPS(net="alex", verbose=False)
def T(x, m): return torch.tensor(x * m[..., None]).permute(2, 0, 1)[None].float() / 127.5 - 1
def lab(im, t):
    im = im.copy(); cv2.rectangle(im, (0, 0), (len(t) * 10 + 10, 24), (0, 0, 0), -1); cv2.putText(im, t, (5, 17), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA); return im
res = []
for n, rects in LOOKS.items():
    real = cv2.imread(os.path.join(REAL, n, "mast_rgb.png"))[..., ::-1]; sim = cv2.imread(os.path.join(S, n + "_mast_sim.png"))[..., ::-1]
    m = np.ones(real.shape[:2], bool)
    for x0, y0, x1, y1 in rects + [(0, 0, 500, 26)]: m[y0:y1, x0:x1] = False
    with torch.no_grad(): lp = float(LP(T(real, m), T(sim, m)))
    g1, g2 = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY)
    r = dict(look=n, lpips_alex_masked=round(lp, 3), ssim_masked=round(float(ssim(g1, g2, data_range=255, full=True)[1][m].mean()), 3), mask_frac=round(float(m.mean()), 2))
    if n.endswith("_m0"):
        mc = m.copy(); mc[:, 640:] = False; mc[440:, 200:] = False; mc[570:, :] = False
        with torch.no_grad(): r["lpips_alex_cycles_mask"] = round(float(LP(T(real, mc), T(sim, mc))), 3)
    sh = real.copy(); sh[~m] = (sh[~m] * 0.25).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, "side_%s.jpg" % n[9:]), np.concatenate([lab(sh, "REAL mast " + n[9:15] + " (people masked)"), lab(sim, "SIM ovrtx RTPT + camera model  LPIPS %.3f" % lp)], 1)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    res.append(r); print(json.dumps(r))
mm = float(np.mean([r["lpips_alex_masked"] for r in res]))
json.dump(dict(looks=res, mast_lpips_mean=round(mm, 3), baseline_cycles_prototype_lpips=0.370), open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
print("mean %.3f" % mm)
