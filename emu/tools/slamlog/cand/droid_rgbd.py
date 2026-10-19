"""RGB-D DROID-SLAM on a Kevin export dir (rgb/*.jpg, depth/*.png16 mm registered to colour, K.txt).
Settings = the repo's ETH3D RGB-D evaluation defaults.  Writes TUM lines 'export_index tx ty tz qx qy qz qw' (camera-to-world)."""
import sys, os, glob, time, argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'droid_slam'))
import numpy as np, torch, lietorch, cv2
import torch.nn.functional as F
from droid import Droid

def stream(d, depth=True):
    K = np.loadtxt(os.path.join(d, 'K.txt')); fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for f in sorted(glob.glob(os.path.join(d, 'rgb', '*.jpg'))):
        n = int(os.path.basename(f)[:-4]); im = cv2.imread(f); h0, w0, _ = im.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0))); w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        im = torch.as_tensor(cv2.resize(im, (w1, h1))[:h1 - h1 % 8, :w1 - w1 % 8]).permute(2, 0, 1)
        z = torch.as_tensor(cv2.imread(os.path.join(d, 'depth', '%06d.png' % n), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0)
        z = F.interpolate(z[None, None], (h1, w1)).squeeze()[:h1 - h1 % 8, :w1 - w1 % 8]
        k = torch.as_tensor([fx, fy, cx, cy]); k[0::2] *= w1 / w0; k[1::2] *= h1 / h0
        yield n, im[None], (z if depth else None), k

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('datapath'); p.add_argument('out'); p.add_argument('--mono', action='store_true')
    p.add_argument('--weights', default='droid.pth'); p.add_argument('--buffer', type=int, default=1024); p.add_argument('--image_size', default=[240, 320])
    p.add_argument('--disable_vis', action='store_true', default=True); p.add_argument('--beta', type=float, default=0.5)
    p.add_argument('--filter_thresh', type=float, default=2.0); p.add_argument('--warmup', type=int, default=8); p.add_argument('--keyframe_thresh', type=float, default=3.5)
    p.add_argument('--frontend_thresh', type=float, default=16.0); p.add_argument('--frontend_window', type=int, default=20); p.add_argument('--frontend_radius', type=int, default=2)
    p.add_argument('--frontend_nms', type=int, default=1); p.add_argument('--stereo', action='store_true'); p.add_argument('--backend_thresh', type=float, default=22.0)
    p.add_argument('--backend_radius', type=int, default=2); p.add_argument('--backend_nms', type=int, default=3); p.add_argument('--motion_damping', type=float, default=0.5)
    p.add_argument('--upsample', action='store_true'); p.add_argument('--asynchronous', action='store_true')
    p.add_argument('--frontend_device', default='cuda'); p.add_argument('--backend_device', default='cuda')
    a = p.parse_args(); torch.multiprocessing.set_start_method('spawn')
    droid = None; ids = []; t0 = time.time()
    for n, im, z, k in stream(a.datapath, not a.mono):
        if droid is None: a.image_size = [im.shape[2], im.shape[3]]; droid = Droid(a)
        droid.track(len(ids), im, z, intrinsics=k); ids.append(n)
    t_track = time.time() - t0
    def s2():
        for i, (n, im, z, k) in enumerate(stream(a.datapath, not a.mono)): yield i, im, k
    traj = droid.terminate(s2()); t_all = time.time() - t0
    with open(a.out, 'w') as f:
        for n, T in zip(ids, traj): f.write('%d %s\n' % (n, ' '.join('%.7f' % v for v in T)))
    print('DROIDTIME frames %d track %.1f s (%.1f fps) total %.1f s gpu %.2f GB' % (len(ids), t_track, len(ids) / t_track, t_all, torch.cuda.max_memory_allocated() / 1e9))
