"""ORB-SLAM3 stereo (repo build sim/orbslam3/orbslam3_track, ORB-SLAM3 default stereo settings: 1250 features,
scale 1.2, 8 levels, FAST 20/7, ThDepth 40; intrinsics/baseline = sim calibration; no speckle filter, no viewer,
no atlas save) on a gen_stereo2 log.  python run_orb.py LOG.npz -> same metrics as run_cuvslam.py"""
import sys, os, json, math, time, socket, struct, subprocess, numpy as np
sys.path[:0] = [__import__('os').path.join(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), 'src')]
import twin_geometry as TG
from scipy.spatial.transform import Rotation as Rot
REPO = __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')); ORB = os.path.join(os.path.dirname(REPO), 'ORB_SLAM3')
PREFIX = os.path.expanduser('~/.local/share/mamba/envs/orbslam3')
L = dict(np.load(sys.argv[1])); I = json.loads(str(L['intr'])); H, W = L['ir1'].shape[1:]
y = open(os.path.join(REPO, 'sim/orbslam3/D435i.yaml')).read()
for k, v in (('Camera1.fx', I['fx']), ('Camera1.fy', I['fy']), ('Camera1.cx', I['ppx']), ('Camera1.cy', I['ppy']), ('Camera.width', W), ('Camera.height', H), ('Camera.fps', 10), ('Stereo.b', I['baseline'])):
    y = '\n'.join((k + ': ' + (str(int(v)) if k in ('Camera.width', 'Camera.height', 'Camera.fps') else '%.4f' % v)) if ln.startswith(k + ':') else ln for ln in y.split('\n'))
y = '\n'.join(ln for ln in y.split('\n') if 'SaveAtlas' not in ln)
yp = '/tmp/orb_bench_%d.yaml' % os.getpid(); open(yp, 'w').write(y)
port = 5591; env = dict(os.environ, ORBSLAM3_VIEWER='0', ORBSLAM3_SPECKLE_K='0', LD_LIBRARY_PATH='%s/lib:%s/lib:%s' % (PREFIX, ORB, os.environ.get('LD_LIBRARY_PATH', '')))
tum = os.path.expanduser('~/.kevin/orbslam3/traj_tum.txt')
if os.path.exists(tum): os.remove(tum)
p = subprocess.Popen([REPO + '/sim/orbslam3/orbslam3_track', ORB + '/Vocabulary/ORBvoc.txt', yp, str(port), '127.0.0.1', '5592'], env=env, stdout=open('/tmp/orb_bench.log', 'w'), stderr=subprocess.STDOUT)
for _ in range(120):
    try: s = socket.create_connection(('127.0.0.1', port)); break
    except OSError: time.sleep(1)
t0 = time.perf_counter()
for i in range(len(L['t'])):
    body = struct.pack('<QHH5f', int(L['t'][i] * 1e9), W, H, I['fx'], I['fy'], I['ppx'], I['ppy'], I['baseline']) + L['ir1'][i].tobytes() + L['ir2'][i].tobytes()
    s.sendall(struct.pack('<I', len(body)) + body)
s.shutdown(socket.SHUT_WR); p.wait(timeout=1800); wall = time.perf_counter() - t0
tb, Rb = TG.frame('rs2_ir1'); Tbc = np.eye(4); Tbc[:3, :3] = Rb; Tbc[:3, 3] = tb; Tcb = np.linalg.inv(Tbc)
D = np.loadtxt(tum, ndmin=2) if os.path.exists(tum) else np.zeros((0, 8))
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
E, G = [], []; first = None; tl = L['t']
for r in D:
    i = int(np.argmin(np.abs(tl - r[0]))); Tc = np.eye(4); Tc[:3, :3] = Rot.from_quat(r[4:8]).as_matrix(); Tc[:3, 3] = r[1:4]
    Tb = Tbc @ Tc @ Tcb; q = (Tb[0, 3], Tb[1, 3], math.atan2(Tb[1, 0], Tb[0, 0]))
    if first is None: first = (q, L['gt'][i])
    q0, g0 = first; dx, dy = q[0] - q0[0], q[1] - q0[1]; c, sn = math.cos(g0[2] - q0[2]), math.sin(g0[2] - q0[2])
    E.append((g0[0] + c * dx - sn * dy, g0[1] + sn * dx + c * dy, g0[2] + q[2] - q0[2])); G.append(L['gt'][i])
res = dict(log=sys.argv[1], m='orbslam3_stereo', env=str(L['env']), lvl=str(L.get('lvl', 'ideal')), sun=bool(L.get('sun', False)), frames=len(tl), tracked=len(E), ms=round(wall / len(tl) * 1e3, 1))
if len(E) > 10:
    E, G = np.array(E), np.array(G); ex = np.hypot(*(E[:, :2] - G[:, :2]).T)
    res.update(ate=round(float(np.sqrt((ex ** 2).mean())), 3), final=round(float(ex[-1]), 3), max=round(float(ex.max()), 3), yawmax=round(math.degrees(np.abs([wrap(a) for a in E[:, 2] - G[:, 2]]).max()), 1))
print(json.dumps(res))
