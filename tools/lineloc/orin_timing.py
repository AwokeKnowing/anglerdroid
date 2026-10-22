"""Orin CPU timing of lineloc on the stationary real mast frames (read-only; nice 19; no robot I/O)."""
import time, math, numpy as np, cv2, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lineloc import Camera, LineMap, LineLoc
cv2.setNumThreads(1)
FIT = np.array([3.815, 1.406, math.radians(95.8)])
c = Camera(196.78, 196.78, 319.5, 239.5, (-0.0262, 0, 0, 0), t_bc=(0.0291, 0, 0.99115)); c.R_bc = Camera.tilt(math.radians(2.16), math.radians(-3.22)) @ c.R_bc
M = LineMap("casita_lines.npz"); ims = [cv2.imread(f"real_{k}.png", 0) for k in range(5)]
P = np.diag([0.03 ** 2, 0.03 ** 2, math.radians(1) ** 2])
for sc in (0.5, 0.8):
    l = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, sc); T = []
    for r in range(10):
        for im in ims: t = time.perf_counter(); l.detect(im); T.append((time.perf_counter() - t) * 1e3)
    print(f"LSD scale {sc}: med {np.median(T):.1f} p95 {np.percentile(T, 95):.1f} max {np.max(T):.1f} ms")
for name, kw in [("V", dict(use_h=False)), ("VH", dict(use_h=True, h_junction=True, h_minlen=40, sigma_map=0.02))]:
    for sc in (0.5, 0.8):
        loc = LineLoc(M, c, lsd_scale=sc, **kw); T = []; TL = []
        for r in range(10):
            for im in ims:
                t = time.perf_counter(); loc.segments(im); TL.append((time.perf_counter() - t) * 1e3)
                t = time.perf_counter(); x, Pp, info = loc.update(im, FIT, P); T.append((time.perf_counter() - t) * 1e3)
        print(f"lineloc {name} lsd{sc}: total med {np.median(T):.1f} p95 {np.percentile(T, 95):.1f} max {np.max(T):.1f} ms (LSD part med {np.median(TL):.1f}) ok={info.get('ok')} pose=({x[0]:.3f},{x[1]:.3f},{math.degrees(x[2]):.2f})")
