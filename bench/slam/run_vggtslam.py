"""VGGT-SLAM 2.0 (MIT-SPARK, RSS 2026; uncalibrated feed-forward dense RGB SLAM, SL(4)/factor-graph submaps, loop
closure) on one camera stream of a recorded sim log.  No intrinsics are given: VGGT estimates them; we log its focal
estimate per keyframe to check self-calibration against the withheld twin truth.
-> RESULTS/<seq>/vggtslam_<cam>.tum (camera frame, scale-free) + .json (focal estimates, runtime)
python bench/slam/run_vggtslam.py DATA/<seq> RESULTS/<seq> mast|rs2_color [submap]   (venv-vggt, cwd = VGGT-SLAM repo)"""
import json, os, sys, time, glob
import numpy as np
seq, res, cam = sys.argv[1], sys.argv[2], sys.argv[3]; sub = int(sys.argv[4]) if len(sys.argv) > 4 else 16
seq = os.path.abspath(seq); res = os.path.abspath(res); os.makedirs(res, exist_ok=True)
F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
img_dir = os.path.join(res, "_vggt_imgs_" + cam); os.makedirs(img_dir, exist_ok=True)
for i, t in zip(F["i"].astype(int), F["t"]):
    dst = os.path.join(img_dir, "%.4f.jpg" % t)
    if not os.path.exists(dst): os.symlink(os.path.join(seq, cam, "%06d.jpg" % i), dst)
repo = os.environ.get("VGGT_SLAM_DIR", os.path.expanduser("~/slambench/ext/VGGT-SLAM")); os.chdir(repo); sys.path.insert(0, repo)
import vggt_slam.map as VM
Ks = []; _dec = VM.decompose_camera
def dec(P):
    out = _dec(P); Ks.append(np.asarray(out[0]).tolist()); return out
VM.decompose_camera = dec
tum_raw = os.path.join(res, "vggtslam_%s.raw.txt" % cam)
sys.argv = ["main.py", "--image_folder", img_dir, "--max_loops", "1", "--min_disparity", "50", "--conf_threshold", "25", "--lc_thres", "0.95",
            "--submap_size", str(sub), "--log_results", "--skip_dense_log", "--log_path", tum_raw]
import torch
import main as M
t0 = time.time(); torch.cuda.reset_peak_memory_stats()
M.main()
wall = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 2**30
D = np.loadtxt(tum_raw, ndmin=2)
np.savetxt(os.path.join(res, "vggtslam_%s.tum" % cam), D, fmt="%.6f")
# VGGT predicts intrinsics at its input resolution (518 px wide); express the focal in the 640 px source image
fx = [k[0][0] for k in Ks]; cx = [k[0][2] for k in Ks]
scale = 640.0 / (2 * np.median(cx)) if Ks else float("nan")
json.dump(dict(frame="cam:" + ("mast" if cam == "mast" else "rs2_color"), scale_free=True, per_frame=False,
               note="VGGT-SLAM 2.0 uncalibrated, submap %d, %d keyframes of %d frames" % (sub, len(D), len(F["t"])),
               runtime="%.0f s wall for %d frames (%.1f FPS of input, %.2f GB peak torch)" % (wall, len(F["t"]), len(F["t"]) / wall, peak),
               focal_px_640=float(np.median(fx) * scale) if Ks else None, focal_px_640_iqr=[float(np.percentile(fx, 25) * scale), float(np.percentile(fx, 75) * scale)] if Ks else None),
          open(os.path.join(res, "vggtslam_%s.json" % cam), "w"), indent=1)
print("vggtslam done", cam, len(D), "poses", wall, "s", flush=True); sys.stdout.flush()
os._exit(0)
