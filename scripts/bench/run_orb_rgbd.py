"""ORB-SLAM3 RGB-D (colour + depth, no IR, no viewer, stock ORB-SLAM3 TUM RGB-D settings: 1000 features, scale 1.2,
8 levels, FAST 20/7, ThDepth 40) on a Kevin export dir (emu.tools.slamlog.export_rgbd).  Builds the yaml from K.txt.
  python scripts/bench/run_orb_rgbd.py <exportdir> <out_tum.txt>
Binary: sim/orbslam3/rgbd/rgbd_kevin (= ORB-SLAM3 Examples/RGB-D/rgbd_tum with viewer off and no real-time sleep)."""
import os, sys, time, subprocess, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); ORB = os.path.join(os.path.dirname(REPO), 'ORB_SLAM3')
d, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
K = np.loadtxt(d + '/K.txt'); lst = [l.split() for l in open(d + '/list.txt')]
import cv2; h, w = cv2.imread('%s/rgb/%s.jpg' % (d, lst[0][1])).shape[:2]
y = open(ORB + '/Examples/RGB-D/TUM1.yaml').read().split('\n'); rep = {'Camera1.fx': K[0, 0], 'Camera1.fy': K[1, 1], 'Camera1.cx': K[0, 2], 'Camera1.cy': K[1, 2],
       'Camera1.k1': 0, 'Camera1.k2': 0, 'Camera1.p1': 0, 'Camera1.p2': 0, 'Camera1.k3': 0, 'Camera.width': w, 'Camera.height': h, 'Camera.fps': 30,
       'RGBD.DepthMapFactor': 1000.0, 'Stereo.b': 0.050}
def fmt(k, v): return '%s: %s' % (k, str(int(v)) if k in ('Camera.width', 'Camera.height', 'Camera.fps') else '%.6f' % v)
y = [fmt(ln.split(':')[0], rep[ln.split(':')[0]]) if ':' in ln and ln.split(':')[0] in rep else ln for ln in y]
wd = '/tmp/orb_rgbd_%d' % os.getpid(); os.makedirs(wd, exist_ok=True); open(wd + '/cam.yaml', 'w').write('\n'.join(y))
with open(wd + '/assoc.txt', 'w') as f:
    for t, n in lst: f.write('%s rgb/%s.jpg %s depth/%s.png\n' % (t, n, t, n))
env = dict(os.environ, LD_LIBRARY_PATH='%s/lib:%s/lib:%s' % (os.path.expanduser('~/.local/share/mamba/envs/orbslam3'), ORB, os.environ.get('LD_LIBRARY_PATH', '')))
t0 = time.time()
p = subprocess.run([REPO + '/sim/orbslam3/rgbd/rgbd_kevin', ORB + '/Vocabulary/ORBvoc.txt', wd + '/cam.yaml', d, wd + '/assoc.txt'], cwd=wd, env=env, stdout=open(wd + '/log.txt', 'w'), stderr=subprocess.STDOUT)
wall = time.time() - t0
os.replace(wd + '/CameraTrajectory.txt', out)
n = sum(1 for _ in open(out)); lost = open(wd + '/log.txt').read().count('Fail to track')
print('orb_rgbd frames %d poses %d wall %.1f s (%.1f ms/frame) rc %d log %s' % (len(lst), n, wall, 1000 * wall / len(lst), p.returncode, wd + '/log.txt'))
