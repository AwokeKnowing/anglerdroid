"""python -m emu.tools.slamlog.export_rgbd <logdir> <cam> <outdir> [every]: colour jpg + depth registered to colour
(png16 mm) + list.txt (stamp_s file) for third-party RGB-D SLAM / reconstruction (ACE0, DROID, ...)."""
import os, sys, shutil
import cv2, numpy as np
from .log import Log
d, cam, out = sys.argv[1:4]; ev = int(sys.argv[4]) if len(sys.argv) > 4 else 1
L = Log(d); os.makedirs(out + "/rgb", exist_ok=True); os.makedirs(out + "/depth", exist_ok=True); lst = open(out + "/list.txt", "w")
for i, (t, f) in enumerate(L.frames[cam][::ev]):
    n = "%06d" % i; shutil.copy(os.path.join(d, f + ".jpg"), "%s/rgb/%s.jpg" % (out, n))
    cv2.imwrite("%s/depth/%s.png" % (out, n), np.round(L.depth_in_color(f) * 1000).astype(np.uint16)); lst.write("%.6f %s\n" % (t, n))
K, _ = L.K(cam, "color"); np.savetxt(out + "/K.txt", K)
