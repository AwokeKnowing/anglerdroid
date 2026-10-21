"""LingBot-Map (Robbyant 2026, Apache-2.0; streaming feed-forward geometric-context transformer, VGGT lineage,
uncalibrated) on one camera stream of a recorded sim log.  SDPA attention (no FlashInfer JIT), bf16 trunk,
streaming mode with keyframe KV caching.
-> RESULTS/<seq>/lingbot_<cam>.tum (camera frame, scale-free, c2w) + .json (focal estimate, runtime, peak mem)
python bench/slam/run_lingbot.py DATA/<seq> RESULTS/<seq> mast|rs2_color [stride]   (venv-mast3r: torch 2.5)"""
import argparse, json, os, sys, time
import numpy as np
seq, res, cam = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]; stride = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(res, exist_ok=True)
F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
img_dir = os.path.join(res, "_lingbot_imgs_" + cam); os.makedirs(img_dir, exist_ok=True)
keep = list(range(0, len(F["t"]), stride))
for k in keep:
    dst = os.path.join(img_dir, "%06d.jpg" % k)
    if not os.path.exists(dst): os.symlink(os.path.join(seq, cam, "%06d.jpg" % int(F["i"][k])), dst)
repo = os.environ.get("LINGBOT_DIR", os.path.expanduser("~/slambench/ext/lingbot-map")); sys.path.insert(0, repo); os.chdir(repo)
sys.argv = ["demo.py"]
import torch, demo
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
ISZ = int(os.environ.get("LINGBOT_SIZE", "392"))
args = argparse.Namespace(mode="streaming", image_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024, num_scale_frames=8,
                          kv_cache_sliding_window=int(os.environ.get("LINGBOT_WIN", "16")), use_sdpa=True, camera_num_iterations=4,
                          model_path=os.environ.get("LINGBOT_CKPT", os.path.expanduser("~/slambench/ckpt/lingbot-map/lingbot-map.pt")))
images, paths, _ = demo.load_images(image_folder=img_dir, image_size=ISZ, patch_size=14)
dev = torch.device("cuda"); model = demo.load_model(args, torch.device("cpu")); model.aggregator = model.aggregator.to(dtype=torch.bfloat16); model = model.to(dev)   # cast on CPU first: fits next to LightNav
n = images.shape[0]; kfi = int(os.environ.get("LINGBOT_KFI", max(2, (n + 319) // 320)))
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    pred = model.inference_streaming(images, num_scale_frames=8, keyframe_interval=kfi, output_device=torch.device("cpu"))
wall = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 2**30
ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"].float(), images.shape[-2:])     # c2w (see demo.postprocess)
ext = ext.reshape(-1, 3, 4).cpu().numpy(); intr = intr.reshape(-1, 3, 3).cpu().numpy()
from scipy.spatial.transform import Rotation as Rot
names = sorted(os.listdir(img_dir)); ts = [F["t"][int(os.path.splitext(p)[0])] for p in names]
with open(os.path.join(res, "lingbot_%s.tum" % cam), "w") as f:
    for t, E in zip(ts, ext):
        q = Rot.from_matrix(E[:, :3]).as_quat(); f.write("%.4f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (t, *E[:, 3], *q))
Wp = images.shape[-1]; fx = float(np.median(intr[:, 0, 0])) * 640.0 / Wp
json.dump(dict(frame="cam:" + cam, scale_free=True, per_frame=False, note="LingBot-Map streaming (SDPA, bf16), stride %d, keyframe_interval %d, KV window %d, input %d px, %d frames" % (stride, kfi, args.kv_cache_sliding_window, ISZ, n),
               runtime="%.1f s for %d frames = %.1f FPS, %.2f GB peak torch" % (wall, n, n / wall, peak), focal_px_640=fx), open(os.path.join(res, "lingbot_%s.json" % cam), "w"), indent=1)
print("lingbot done", cam, n, "frames", round(wall, 1), "s", round(peak, 2), "GB fx640", round(fx, 1), flush=True)
