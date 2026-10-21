import json, os, subprocess, sys, time, urllib.request as U
sys.path.insert(0, '/home/james/gits/awokeknowing/anglerdroid')
from emu import gt
def post(t): return json.load(U.urlopen(U.Request("http://localhost:8095/chat", json.dumps({"text": t, "who": "bench"}).encode(), {"Content-Type": "application/json"}), timeout=60))["reply"]
S = json.load(open('/dev/shm/kevin_sim.json')); x, y, a = S['start']
gt.ctl("dog", enabled=True, keep_out=[[x, y, 1.5]]); gt.ctl("teleport", x=x, y=y, yaw=a); time.sleep(2)
subprocess.run([os.path.expanduser('~/m1_backup/dbg/restart_nav.sh')], stdout=subprocess.DEVNULL); time.sleep(3)
out = open(os.path.expanduser('~/m1_backup/dbg/dbg_gov.jsonl'), 'w')
for cmd in sys.argv[1:]:
    print(cmd, post(cmd), flush=True); t0 = time.time()
    while time.time() - t0 < 120:
        time.sleep(0.1); n = json.load(U.urlopen("http://localhost:8090/api/state", timeout=5))
        out.write(json.dumps([time.time() - t0, n['mode'], n['msg'][:40], n['safety'].get('governor'), (n['safety'].get('core') or {}).get('code'), (n['safety'].get('core') or {}).get('sd0')]) + "\n"); out.flush()
        if n['mode'] == 'idle' and time.time() - t0 > 3: break
    print(n['msg'], flush=True)
post('stop')
