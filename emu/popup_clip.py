"""Record one pop-up-box safety stop as clip frames: python -m emu.popup_clip OUTDIR [D]"""
import math, os, subprocess, sys, time
import numpy as np
from emu.eval import start, stop, REPO, ROBPY
from emu import gt
from emu.safety_eval import lane, run_drive, FRONT, BOX
out = sys.argv[1]; D = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
os.makedirs(out, exist_ok=True)
w, n = start("room", 0, 8090, out)
z = np.load("/dev/shm/kevin_gt_map.npz"); p0 = lane(z["occ"].astype(bool), z["origin"], float(z["res"]), 4.0)
rec = subprocess.Popen(["scripts/kevin-hal", ROBPY, os.path.join(REPO, "emu", "tools", "clip_rec.py"), os.path.join(out, "frames"), "8090", "10"], cwd=REPO,
                       env=dict(os.environ, KEVIN_HAL="sim"), stdout=open(os.path.join(out, "rec.log"), "w"), stderr=subprocess.STDOUT)
time.sleep(4)
def during(t, s, ctx):
    if "placed" not in ctx and s["v"] > 0.55:
        c, sn = math.cos(s["yaw"]), math.sin(s["yaw"]); d = FRONT + D + BOX
        gt.ctl("obstacle", i=0, x=s["x"] + c * d, y=s["y"] + sn * d); ctx["placed"] = 1
rows, iv, ctx = run_drive(8090, p0, 0.6, 4.0, during)
time.sleep(2); open(os.path.join(out, "frames", "STOP"), "w").close(); time.sleep(2); rec.kill()
print("popup clip: placed=%s collisions=%d interventions=%d vmax=%.2f" % ("placed" in ctx, int(rows[-1, 5]), iv, rows[:, 4].max()), flush=True)
stop(n, w)
import cv2, glob
fs = sorted(glob.glob(os.path.join(out, "frames", "*.jpg")))
if fs:
    h, wd = cv2.imread(fs[0]).shape[:2]; v = cv2.VideoWriter(os.path.join(out, "popup.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (wd, h))
    [v.write(cv2.imread(f)) for f in fs]; v.release(); print("frames", len(fs))
