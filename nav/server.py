"""Tiny command/telemetry server (stdlib only).  GET / page, GET /api/state, GET /api/map.png,
POST /api/command {"text": "..."}  <- the hook a conversational layer will call later."""
from __future__ import annotations
import json, math, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2

PAGE = r"""<!doctype html><meta charset=utf-8><title>Kevin</title>
<style>body{font:14px system-ui;margin:0;display:flex;height:100vh;background:#111;color:#ddd}
#l{flex:1;position:relative}canvas{width:100%;height:100%;image-rendering:pixelated}
#r{width:360px;padding:10px;display:flex;flex-direction:column;gap:8px;border-left:1px solid #333}
input{font:15px monospace;padding:6px;background:#222;color:#fff;border:1px solid #555}
pre{font:11px monospace;white-space:pre-wrap;margin:0;overflow:auto}</style>
<div id=l><canvas id=c></canvas></div><div id=r>
<input id=q placeholder="go to kitchen | forward 1 | turn left 90 | label sofa | stop" autofocus>
<div>click map = go there, shift-click = label here</div><pre id=ev></pre><pre id=st></pre></div>
<script>
const c=document.getElementById('c'),x=c.getContext('2d');let S=null,img=new Image(),meta=null;
async function cmd(t){await fetch('/api/command',{method:'POST',body:JSON.stringify({text:t})})}
q.onkeydown=e=>{if(e.key=='Enter'){cmd(q.value);q.value=''}};
function v(){const W=c.clientWidth,H=c.clientHeight;c.width=W;c.height=H;return [W,H]}
let view={cx:0,cy:0,s:60};
function w2s(p){const [W,H]=[c.width,c.height];return [W/2+(p[0]-view.cx)*view.s,H/2-(p[1]-view.cy)*view.s]}
function s2w(a,b){const [W,H]=[c.width,c.height];return [(a-W/2)/view.s+view.cx,-(b-H/2)/view.s+view.cy]}
c.onwheel=e=>{view.s*=e.deltaY<0?1.15:1/1.15;e.preventDefault()};
c.onclick=e=>{const r=c.getBoundingClientRect(),p=s2w(e.clientX-r.left,e.clientY-r.top);
 if(e.shiftKey){const n=prompt('label');if(n)cmd(`label ${n} at ${p[0].toFixed(2)} ${p[1].toFixed(2)}`)}else cmd(`go to ${p[0].toFixed(2)} ${p[1].toFixed(2)}`)};
async function tick(){try{S=await (await fetch('/api/state')).json();view.cx=S.pose[0];view.cy=S.pose[1];
 ev.textContent=S.events.map(e=>e[1]).join('\n');
 st.textContent=JSON.stringify({mode:S.mode,pose:S.pose.map(a=>+a.toFixed(2)),cmd:S.cmd,slam:S.slam,safety:S.safety,timing:S.timing,plan:S.plan_timing},null,1);}catch(e){}}
function draw(){v();x.fillStyle='#000';x.fillRect(0,0,c.width,c.height);
 if(meta&&img.complete){const a=w2s([meta.x0,meta.y1]),b=w2s([meta.x1,meta.y0]);x.drawImage(img,a[0],a[1],b[0]-a[0],b[1]-a[1])}
 if(S){if(S.path){x.strokeStyle='#0f0';x.beginPath();S.path.forEach((p,i)=>{const q=w2s(p);i?x.lineTo(...q):x.moveTo(...q)});x.stroke()}
  for(const [n,p] of Object.entries(S.places)){const q=w2s(p);x.fillStyle='#fc0';x.fillRect(q[0]-3,q[1]-3,6,6);x.fillText(n,q[0]+5,q[1]-5)}
  const r=w2s(S.pose),th=S.pose[2];x.strokeStyle='#4af';x.beginPath();x.arc(r[0],r[1],0.21*view.s,0,7);x.moveTo(...r);
  x.lineTo(r[0]+Math.cos(th)*0.35*view.s,r[1]-Math.sin(th)*0.35*view.s);x.stroke()}
 requestAnimationFrame(draw)}
async function mapl(){const m=await (await fetch('/api/map.json')).json();meta=m;img.src='/api/map.png?'+Date.now()}
setInterval(tick,200);setInterval(mapl,1000);mapl();draw();
</script>"""


def map_png(brain, half=10.0):
    g = brain.slam.grid; x, y, _ = brain.slam.pose()
    c0 = np.floor((np.array([x - half, y - half]) - g.origin) / g.res).astype(int); n = int(2 * half / g.res)
    c0 = np.clip(c0, 0, g.n - n)
    L = g.L[c0[1]:c0[1] + n, c0[0]:c0[0] + n]; seen = g.seen[c0[1]:c0[1] + n, c0[0]:c0[0] + n]
    img = np.full(L.shape, 60, np.uint8); img[seen] = 200; img[L > 0.7] = 20
    x0, y0 = g.origin + c0 * g.res
    ok, png = cv2.imencode(".png", np.flipud(img))
    return png.tobytes(), dict(x0=float(x0), y0=float(y0), x1=float(x0 + n * g.res), y1=float(y0 + n * g.res))


def serve(brain, port=8080):
    cache = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, body, ctype="application/json", code=200):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/": return self._send(PAGE.encode(), "text/html")
            if p == "/api/state": return self._send(json.dumps(brain.state(), default=float).encode())
            if p == "/api/map.json":
                cache["png"], cache["meta"] = map_png(brain); return self._send(json.dumps(cache["meta"]).encode())
            if p == "/api/map.png":
                if "png" not in cache: cache["png"], cache["meta"] = map_png(brain)
                return self._send(cache["png"], "image/png")
            self._send(b"{}", code=404)

        def do_POST(self):
            if self.path != "/api/command": return self._send(b"{}", code=404)
            n = int(self.headers.get("Content-Length", 0)); d = json.loads(self.rfile.read(n) or b"{}")
            c = brain.command(d.get("text", ""))
            self._send(json.dumps(c).encode())

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True, name="http").start()
    print("[nav] UI on http://0.0.0.0:%d" % port, flush=True)
    return srv
