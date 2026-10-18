"""MASt3R-SLAM (Murai/Dexheimer/Davison, CVPR 2025; calibration-free dense monocular SLAM on MASt3R two-view priors,
generic central-camera ray model, retrieval loop closure + global GN) on one camera stream of a recorded sim log.
Run uncalibrated (config/base.yaml) — no intrinsics given.  Keyframe poses only.
-> RESULTS/<seq>/mast3rslam_<cam>.tum (camera frame, scale-free) + .json
python bench/slam/run_mast3rslam.py DATA/<seq> RESULTS/<seq> mast|rs2_color      (venv-mast3r)"""
import json, os, sys, time, subprocess
import numpy as np
seq, res, cam = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]; os.makedirs(res, exist_ok=True)
F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True); T = F["t"]
name = "m3s_%s_%s" % (os.path.basename(seq), cam)
img_dir = os.path.join(res, "_m3s_imgs", name); os.makedirs(img_dir, exist_ok=True)
for k, i in enumerate(F["i"].astype(int)):             # RGBFiles wants *.png; cv2 decodes the jpg by content
    dst = os.path.join(img_dir, "%06d.png" % k)
    if not os.path.exists(dst): os.symlink(os.path.join(seq, cam, "%06d.jpg" % i), dst)
repo = os.environ.get("MAST3R_SLAM_DIR", os.path.expanduser("~/slambench/ext/MASt3R-SLAM"))
cfg = os.environ.get("M3S_CONFIG", "config/base.yaml")
t0 = time.time()
r = subprocess.run([sys.executable, "main.py", "--dataset", img_dir, "--config", cfg, "--no-viz", "--save-as", "slambench"], cwd=repo,
                   stdout=open(os.path.join(res, "_%s.log" % name), "w"), stderr=subprocess.STDOUT)
wall = time.time() - t0
out = os.path.join(repo, "logs", "slambench", name + ".txt")
D = np.loadtxt(out, ndmin=2)
idx = np.clip(np.round(D[:, 0] * 30.0).astype(int), 0, len(T) - 1); D[:, 0] = T[idx]   # RGBFiles stamps = k/30
np.savetxt(os.path.join(res, "mast3rslam_%s.tum" % cam), D, fmt="%.6f")
fps = [float(l.split()[-1]) for l in open(os.path.join(res, "_%s.log" % name)) if l.startswith("FPS:")]
json.dump(dict(frame="cam:" + cam, scale_free=True, per_frame=False, rc=r.returncode,
               note="MASt3R-SLAM uncalibrated (%s), %d keyframes of %d frames" % (cfg, len(D), len(T)),
               runtime="%.0f s wall for %d frames; tracker FPS %.1f (last log)" % (wall, len(T), fps[-1] if fps else float("nan"))),
          open(os.path.join(res, "mast3rslam_%s.json" % cam), "w"), indent=1)
ply = os.path.join(repo, "logs", "slambench", name + ".ply")
print("mast3rslam done", cam, len(D), "keyframes", round(wall), "s", "ply" if os.path.exists(ply) else "")
