"""Head-on stop test: python stoptest.py <tag> v1 [v2 ...]   (box obs0 2.2 m ahead, drive straight, GT polled ~200 Hz)"""
import subprocess, json, math, os, sys, threading, time, urllib.request
sys.path.insert(0, '/home/james/gits/awokeknowing/anglerdroid')
from emu import gt
TAG = sys.argv[1]; VS = [float(a) for a in sys.argv[2:]]
X0, Y0, YAW, DBOX = float(os.environ.get('X0', 3.6)), float(os.environ.get('Y0', 2.2)), float(os.environ.get('YAW', 0.0)), float(os.environ.get('DBOX', 2.2))
SL = os.path.expanduser('~/m1_backup/dbg/safelog.jsonl')
def post(t):
    r = urllib.request.Request("http://localhost:8095/chat", data=json.dumps({"text": t, "who": "bench"}).encode(), method="POST", headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))["reply"]
for v in VS:
    post('stop'); gt.ctl('clear_obstacles'); gt.ctl('teleport', x=X0, y=Y0, yaw=YAW); time.sleep(2.0)
    import subprocess; subprocess.run([os.path.expanduser('~/m1_backup/dbg/restart_nav.sh')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(4)
    W, DL = float(os.environ.get('W', 0.0)), float(os.environ.get('DL', 0.0))
    if abs(W) > 1e-3:
        Rr = v / W; th = DBOX / Rr; px, py, nx, ny = Rr * math.sin(th), Rr * (1 - math.cos(th)), -math.sin(th), math.cos(th)
    else: px, py, nx, ny = DBOX, 0.0, 0.0, 1.0
    px, py = px + DL * nx, py + DL * ny
    bx, by = X0 + px * math.cos(YAW) - py * math.sin(YAW), Y0 + px * math.sin(YAW) + py * math.cos(YAW); gt.ctl('obstacle', i=0, x=bx, y=by); time.sleep(1.5)
    off = os.path.getsize(SL); G = []; run = [True]
    def poll():
        while run[0]:
            s = gt.state(); G.append([time.monotonic(), s['t'], s['x'], s['y'], s['yaw'], s['v'], s['w'], s['collisions'], s['drives'], s.get('last_contact')]); time.sleep(0.003)
    th = threading.Thread(target=poll); th.start(); time.sleep(0.3)
    t_cmd = time.monotonic(); print(v, post('drive %.2f %.2f %.1f' % (v, W, 3.0 / v + 2)), flush=True)
    time.sleep(3.0 / v + 3); run[0] = False; th.join(); post('stop')
    L = [json.loads(l) for l in open(SL).read()[off:].splitlines() if l.strip()] if True else []
    with open(SL) as f: f.seek(off); L = [json.loads(l) for l in f if l.strip()]
    json.dump(dict(w=W, dl=DL, v=v, t_cmd=t_cmd, box=[bx, by], x0=[X0, Y0, YAW], gt=G, nav=L), open(os.path.expanduser('~/m1_backup/dbg/stop/%s_%.2f.json' % (TAG, v)), 'w'))
    os.environ.get('W') and None
    s = G[-1]; print('end x=%.3f collisions=%d gap_bumper_to_face=%.3f' % (s[2], s[7] - G[0][7], (DBOX - 0.15) - ((s[2] - X0) * math.cos(YAW) + (s[3] - Y0) * math.sin(YAW)) - 0.126), flush=True)
gt.ctl('clear_obstacles'); post('stop')
