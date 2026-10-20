# Not a candidate as-is: diverges on L8 (no odom gating). Kept for reference; see docs/SLAM_EVALUATION.md 2026-10-19.
"""Light RGB-D candidate: Open3D frame-to-frame RGB-D odometry (hybrid photometric+geometric, CPU) on a Kevin export dir
(rgb/*.jpg, depth/*.png mm registered to colour, K.txt), initialised by the wheel+gyro odom prior (odom_prior.npy, the
relative camera motion between consecutive frames).  A VO step is accepted only if Open3D converges; else the odom step
is used (the backbone).  No loop closure, no keyframes.  Writes TUM 'export_index tx ty tz qx qy qz qw' (camera-to-world).
  python o3d_rgbd.py <exportdir> <odom_prior.npy> <out.txt> [stride]"""
import sys, os, glob, time, numpy as np, cv2, open3d as o3d
from scipy.spatial.transform import Rotation as Rot
d, pp, out = sys.argv[1:4]; stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
K = np.loadtxt(d + "/K.txt"); P = np.load(pp)
fs = sorted(glob.glob(d + "/rgb/*.jpg"))[::stride]
h, w = cv2.imread(fs[0]).shape[:2]
I = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
opt = o3d.pipelines.odometry.OdometryOption(depth_max=4.0, depth_diff_max=0.07)
J = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
def load(f):
    n = int(os.path.basename(f)[:-4]); c = o3d.geometry.Image(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
    z = o3d.geometry.Image(cv2.imread(d + "/depth/%06d.png" % n, cv2.IMREAD_ANYDEPTH).astype(np.uint16))
    return n, o3d.geometry.RGBDImage.create_from_color_and_depth(c, z, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=True)
t0 = time.time(); n0, r0 = load(fs[0]); T = P[n0].copy() if np.isfinite(P[n0, 0, 0]) else np.eye(4)
L = [(n0, T.copy())]; ok = fb = 0
for f in fs[1:]:
    n, r = load(f)
    prior = np.linalg.inv(P[n0]) @ P[n] if np.isfinite(P[n0, 0, 0]) and np.isfinite(P[n, 0, 0]) else np.eye(4)
    # Open3D convention: transform maps source(r) points into target(r0) frame = T_0<-1 = prior
    s, Tr, _ = o3d.pipelines.odometry.compute_rgbd_odometry(r, r0, I, prior, J, opt)
    if s: ok += 1
    else: Tr = prior; fb += 1
    T = T @ Tr; L.append((n, T.copy())); n0, r0 = n, r
wall = time.time() - t0
with open(out, "w") as fo:
    for n, T in L:
        q = Rot.from_matrix(T[:3, :3]).as_quat(); fo.write("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (n, *T[:3, 3], *q))
print("o3d_rgbd frames %d vo_ok %d odom_fallback %d wall %.1f s (%.1f ms/frame)" % (len(L), ok, fb, wall, 1000 * wall / len(L)))
