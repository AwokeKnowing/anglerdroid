"""M1 multi-client check: N headless WS clients on the viewer; compare state seq/pose, chat broadcast, latency.
python tests/ws_clients.py [url=ws://localhost:8095/ws] [n=3]"""
import json, sys, threading, time
from websockets.sync.client import connect
URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8095/ws"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rx = [dict(seqs={}, n=0, log_seen=None, first=None) for _ in range(N)]; stop = time.time() + 12; token = "where  "  # chat probe
def run(i):
    with connect(URL, max_size=None, open_timeout=15) as ws:
        if i == 0:
            threading.Timer(4.0, lambda: ws.send(json.dumps(dict(type="chat", text="where", who="probe0")))).start()
        while time.time() < stop:
            try: m = json.loads(ws.recv(timeout=2))
            except TimeoutError: continue
            if m.get("type") != "state": continue
            t = time.time(); rx[i]["n"] += 1; rx[i]["seqs"][m["seq"]] = (t, json.dumps([m.get("kevin"), m.get("pose"), m.get("mode")]))
            if rx[i]["log_seen"] is None and any(r[1] == "probe0" for r in (m.get("log") or [])): rx[i]["log_seen"] = t
            rx[i]["clients"] = m.get("clients")
ths = [threading.Thread(target=run, args=(i,)) for i in range(N)]; [t.start() for t in ths]; [t.join() for t in ths]
common = set.intersection(*[set(r["seqs"]) for r in rx])
same = sum(len({rx[i]["seqs"][s][1] for i in range(N)}) == 1 for s in common)
skew = [max(rx[i]["seqs"][s][0] for i in range(N)) - min(rx[i]["seqs"][s][0] for i in range(N)) for s in common]
skew.sort()
print("url %s clients %d msgs %s rate %.1f Hz" % (URL, N, [r["n"] for r in rx], rx[0]["n"] / 12))
print("common seqs %d identical payload %d/%d; inter-client arrival skew p50 %.1f ms p99 %.1f ms" % (len(common), same, len(common), 1e3 * skew[len(skew) // 2], 1e3 * skew[int(len(skew) * .99) - 1]))
print("chat from client0 seen in shared log by:", [r["log_seen"] is not None for r in rx], "server client count", [r.get("clients") for r in rx])
