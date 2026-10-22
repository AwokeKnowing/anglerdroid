"""Real | sim (camera-model output) montage for a render dir.  usage: montage.py <poses.json> <render_dir> <out.jpg> look-substrings..."""
import json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colour_match import sim_frame, cw_luma  # noqa: E402
J = json.load(open(sys.argv[1]))["looks"]; R, out = sys.argv[2:4]; sel = sys.argv[4:]
names = [n for s in sel for n in J if s in n and os.path.exists(os.path.join(R, n + "_mast_hdr.npy"))]
reals = {n: cv2.imread(os.path.join(J[n]["dir"], "mast_rgb.png")) for n in J if os.path.exists(os.path.join(R, n + "_mast_hdr.npy"))}
ae = float(np.median([cw_luma(r[..., ::-1]) for r in reals.values()]))
def lab(im, t):
    im = im.copy(); cv2.rectangle(im, (0, 0), (len(t) * 9 + 8, 20), (0, 0, 0), -1); cv2.putText(im, t, (4, 15), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA); return im
rows = []
for n in names:
    sim = sim_frame(np.load(os.path.join(R, n + "_mast_hdr.npy")), ae)[..., ::-1]
    rows.append(np.hstack([lab(cv2.resize(reals[n], (480, 360)), "REAL mast " + n[9:15]), lab(cv2.resize(sim, (480, 360)), "SIM ovrtx RTPT + camera model")]))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out, len(rows))
