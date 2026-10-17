"""cuVSLAM (pycuvslam 16, default OdometryConfig except rectified stereo; SlamConfig defaults) on a gen_stereo log.
python run_cuvslam.py LOG.npz [stereo|rgbd] -> ATE/RPE vs GT (same start), ms/frame, lost frames"""
import sys, json, math, time, numpy as np
sys.path[:0] = [__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), __import__('os').path.join(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), 'src')]
import cuvslam
from cuvslam_tracker import rig_from_twin, _quat
L = dict(np.load(sys.argv[1])); mode = sys.argv[2] if len(sys.argv) > 2 else "stereo"
I = json.loads(str(L["intr"])); H, W = L["ir1"].shape[1:]
cams = []
for k in (1, 2):
    c = cuvslam.Camera(size=[W, H], principal=[I["ppx"], I["ppy"]], focal=[I["fx"], I["fy"]])
    R, t = rig_from_twin("rs2_ir%d" % k); c.rig_from_camera = cuvslam.Pose(rotation=_quat(R), translation=t.tolist()); cams.append(c)
if mode == "rgbd": cams = cams[:1]
rig = cuvslam.Rig(cams)
if mode == "rgbd":
    oc = cuvslam.Tracker.OdometryConfig(odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
         rgbd_settings=cuvslam.Tracker.OdometryRGBDSettings(depth_scale_factor=1000.0, depth_camera_id=0, enable_depth_stereo_tracking=False))
else:
    oc = cuvslam.Tracker.OdometryConfig(rectified_stereo_camera=True)
tr = cuvslam.Tracker(rig, oc, cuvslam.Tracker.SlamConfig())
def to2d(p):
    tx, ty, tz = p.translation; qx, qy, qz, qw = p.rotation
    return (tz, -tx, -math.atan2(2 * (qx * qz + qw * qy), 1 - 2 * (qx * qx + qy * qy)))
E, G, ms, lost = [], [], [], 0; first = None
for i in range(len(L["t"])):
    ts = int(L["t"][i] * 1e9); t0 = time.perf_counter()
    if mode == "rgbd": pe, sp = tr.track(ts, [L["ir1"][i]], depths=[L["depth"][i]])
    else: pe, sp = tr.track(ts, [L["ir1"][i], L["ir2"][i]])
    ms.append((time.perf_counter() - t0) * 1e3)
    p = sp if (sp is not None and getattr(sp, "translation", None) is not None) else (pe.world_from_rig.pose if pe.world_from_rig is not None else None)
    if p is None: lost += 1; continue
    q = to2d(p)
    if first is None: first = (q, L["gt"][i])
    # express relative to the first tracked frame, anchored at that frame's GT
    q0, g0 = first
    dx, dy = q[0] - q0[0], q[1] - q0[1]; c, s = math.cos(g0[2] - q0[2]), math.sin(g0[2] - q0[2])
    E.append((g0[0] + c * dx - s * dy, g0[1] + s * dx + c * dy, g0[2] + q[2] - q0[2])); G.append(L["gt"][i])
E, G = np.array(E), np.array(G); ex = np.hypot(*(E[:, :2] - G[:, :2]).T)
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
path = np.r_[0, np.cumsum(np.hypot(*np.diff(G[:, :2], axis=0).T))]
print(json.dumps(dict(log=sys.argv[1], m="cuvslam_" + mode, env=str(L["env"]), lvl=str(L.get("lvl","ideal")), sun=bool(L.get("sun",False)), frames=len(L["t"]), lost=lost, path=round(float(path[-1]), 1), ate=round(float(np.sqrt((ex ** 2).mean())), 3),
      final=round(float(ex[-1]), 3), max=round(float(ex.max()), 3), yawmax=round(math.degrees(np.abs([wrap(a) for a in E[:, 2] - G[:, 2]]).max()), 1),
      ms=round(float(np.mean(ms)), 2), ms_p99=round(float(np.percentile(ms, 99)), 1))))
