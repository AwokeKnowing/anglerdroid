cd /home/james/gits/awokeknowing/anglerdroid
~/.venv-kevinrobot/bin/python -c "from emu import gt; import json; S=json.load(open('/dev/shm/kevin_sim.json')); x,y,a=S['start']; gt.ctl('teleport',x=x,y=y,yaw=a)"; sleep 2
(source emu/tools/simpin.sh; ~/m1_backup/dbg/restart_nav.sh; emu/tools/simpin.sh apply) >/dev/null 2>&1; sleep 4
~/.venv-kevinrobot/bin/python - <<'PY'
import json,urllib.request,time
from emu import gt
def cmd(t): urllib.request.urlopen(urllib.request.Request('http://localhost:8090/api/command',json.dumps({'text':t}).encode(),{'Content-Type':'application/json'})).read()
def st(): return json.loads(urllib.request.urlopen('http://localhost:8090/api/state').read())
cmd("go to the kitchen"); t=time.time(); c0=gt.state()['collisions']
while time.time()-t<40:
    time.sleep(0.5); s=st()
    if s['mode'] not in ('path','backout','turn') and time.time()-t>2: break
g=gt.state(); print((s.get('ctl') or {}).get('mode','pre'), round(time.time()-t,1), s['msg'][:70], 'coll', g['collisions']-c0); cmd("stop")
PY
