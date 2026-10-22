"""Colour/brightness match of renders vs real mast frames per surface class (wall, accent/damask, ceiling, carpet).
Classes come from raycasting the walls model at the fitted pose (people/furniture pixels are outliers -> medians).
Renders go through the camera model first (sim_frame), so this compares what the robot would see.
usage: colour_match.py <poses.json> <render_dir> [look-substring ...]  -> prints JSON per look + mean"""
import json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poses as P  # noqa: E402
import camera_model as CM  # noqa: E402

AE_TARGET = None  # set by sim_frame from the real frames' median centre-weighted luma (camera-level, not per frame)
MAST_P = dict(CM.MAST, awb_strength=1.0,
              awb_bias=(0.95, 1.0, 0.935),   # measured: real wall median G/R 1.055, G/B 1.07 after the camera's AWB (6 looks)
              veil_k=0.0, veil_sigma=60.0)   # glare off: measured no effect on the ceiling/wall ratio (k 0-0.15); kept as a knob of the cheap wide lens (not yet measured)

def cw_luma(img):
    H, W = img.shape[:2]; w = np.exp(-(((np.arange(H)[:, None] - H / 2) / (0.35 * H)) ** 2 + ((np.arange(W)[None] - W / 2) / (0.35 * W)) ** 2))
    y = (img / 255.0 * [0.2126, 0.7152, 0.0722]).sum(-1); return float((y * w).sum() / w.sum())

def sim_frame(hdr, ae_target, p=MAST_P, f=252.95, cx=399.75, cy=299.75, seed=1):
    q = dict(p, ae_target=ae_target)
    x = CM.veiling_glare(hdr[..., :3].astype(np.float32), q.get("veil_k", 0.0), q.get("veil_sigma", 60.0))
    x = CM.vignette(x, q["vignette"], f, cx, cy); x = cv2.GaussianBlur(x, (0, 0), q["mtf_sigma"])
    return CM.Sensor(q, seed).frame(x, steady=True)

def classes(mesh, p):
    cam = P.camgeo.Cam(p, 800, 600); d, fid = P.camgeo.raycast(mesh, cam, step=1); n = mesh.face_normals
    nn = np.zeros(fid.shape + (3,)); ok = fid >= 0; nn[ok] = n[fid[ok]]
    wall = ok & (np.abs(nn[..., 2]) < 0.3); diag = wall & (np.abs(nn[..., 0]) > 0.3) & (np.abs(nn[..., 1]) > 0.3)
    return {"wall": wall & ~diag, "damask": diag, "carpet": fid == -1, "ceiling": fid == -2}

if __name__ == "__main__":
    J = json.load(open(sys.argv[1]))["looks"]; R = sys.argv[2]; sel = sys.argv[3:]
    names = [n for n in J if any(s in n for s in sel) and os.path.exists(os.path.join(R, n + "_mast_hdr.npy"))]
    reals = {n: cv2.imread(os.path.join(J[n]["dir"], "mast_rgb.png"))[..., ::-1] for n in names}
    ae = float(np.median([cw_luma(r) for r in reals.values()])); mesh = P.camgeo.load_casita(); acc = {}
    for n in names:
        sim = sim_frame(np.load(os.path.join(R, n + "_mast_hdr.npy")), ae); out = {}
        for k, m in classes(mesh, J[n]["mast"]).items():
            m = m.copy(); m[:26, :500] = False
            if m.sum() < 2000: continue
            a, b = np.median(reals[n][m], 0), np.median(sim[m], 0); out[k] = dict(real=a.round(1).tolist(), sim=b.round(1).tolist())
            acc.setdefault(k, []).append((a, b))
        print(n[9:], json.dumps(out))
    print("MEAN", json.dumps({k: dict(real=np.mean([a for a, _ in v], 0).round(1).tolist(), sim=np.mean([b for _, b in v], 0).round(1).tolist()) for k, v in acc.items()}), "ae_target", round(ae, 3))
