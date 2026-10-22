"""Render the reference looks (poses.json from final_poses.py) with ovrtx and save linear HDR (.npy, float16) + LDR png.
usage: render_looks.py <casita.usda> <poses.json> <outdir> [look-substring ...]   env RL_WARM (frames per pose, 24), RL_SET (settings json)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovrtx_render import RtxSim  # noqa: E402
from usd_scene import cam_matrix  # noqa: E402
usda, pj, out = sys.argv[1:4]; sel = sys.argv[4:]; os.makedirs(out, exist_ok=True)
J = json.load(open(pj))["looks"]; names = [n for n in J if not sel or any(s in n for s in sel)]
warm = int(os.environ.get("RL_WARM", 24))
S = {k: tuple(v) for k, v in json.loads(os.environ.get("RL_SET", "{}")).items()}
t0 = time.time(); sim = RtxSim(usda, ["mast", "rs2_640"], settings=S, log=os.path.join(out, "ovrtx.log")); print("startup %.1fs" % (time.time() - t0), flush=True)
from PIL import Image
for n in names:
    L = J[n]; sim.set_cams({"/World/Cams/mast": cam_matrix(L["mast"]), "/World/Cams/rs2_640": cam_matrix(L["rs2"])})
    t1 = time.time()
    for k in range(warm): o = sim.step(read=("HdrColor", "LdrColor") if k == warm - 1 else ())
    dt = (time.time() - t1) / warm
    for prod in ("mast", "rs2_640"):
        h = o[(prod, "HdrColor")]; l = o[(prod, "LdrColor")]
        np.save(os.path.join(out, "%s_%s_hdr.npy" % (n, prod)), h[..., :3].astype(np.float16))
        Image.fromarray(l[..., :3]).save(os.path.join(out, "%s_%s_ldr.png" % (n, prod)))
    print(n, "%.1f ms/step (both products)" % (dt * 1e3), "hdr mean", float(h[..., :3].mean()), flush=True)
sim.close()
