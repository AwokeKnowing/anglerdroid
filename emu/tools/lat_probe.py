# drive goals back and forth for ~60 s, then print the latency breakdown
import json,urllib.request,time,sys
def cmd(t): urllib.request.urlopen(urllib.request.Request('http://localhost:8090/api/command',json.dumps({'text':t}).encode(),{'Content-Type':'application/json'})).read()
def st(): return json.loads(urllib.request.urlopen('http://localhost:8090/api/state').read())
t0=time.time()
for g in ["2.5 0","0.5 0","2.5 0.3","0.5 0"]:
    cmd("go to "+g); t=time.time()
    while time.time()-t<20:
        time.sleep(0.5)
        if st().get('mode') not in ('path','backout') and time.time()-t>2: break
cmd("stop"); s=st(); t=s['timing']
for k in ('rx_age_ms','percep_ms','percep_period_ms','snap_age_ms','ctl_compute_ms','ctl_wait_ms','ctl_period_ms','frame_to_iss_ms','frame_to_cmd_ms','sample_to_effect_ms','loop_ms','mppi_ms','map_refresh_ms'):
    if k in t: print('%-20s mean %6.1f p50 %6.1f p99 %6.1f n %d'%(k,t[k]['mean'],t[k]['p50'],t[k]['p99'],t[k]['n']))
print(s['msg'][:80])
