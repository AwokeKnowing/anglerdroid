"""Map-building video from a life24 run's frames/ (4 tiles: GT | slam2 | RTAB-Map | odom).
python bench/slam/life24_video.py ~/slambench/res24h/seed0 [--fps 10]  -> RUN/map_building.mp4 (re-run any time while the run goes)"""
import argparse, glob, os, cv2
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--fps", type=float, default=10); a = ap.parse_args()
fs = sorted(glob.glob(os.path.join(os.path.expanduser(a.run), "frames", "*.png")))
if not fs: raise SystemExit("no frames")
h, w = cv2.imread(fs[0]).shape[:2]; out = os.path.join(os.path.expanduser(a.run), "map_building.mp4")
for cc in ("avc1", "mp4v"):
    vw = cv2.VideoWriter(out + ".tmp.mp4", cv2.VideoWriter_fourcc(*cc), a.fps, (w, h))
    if vw.isOpened(): break
for f in fs:
    im = cv2.imread(f)
    if im is not None: vw.write(cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im)
vw.release(); os.replace(out + ".tmp.mp4", out); print(out, len(fs), "frames", cc)
