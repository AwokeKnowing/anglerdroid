"""python -m nav.cli [--url http://localhost:8080] ["command"]   (no command -> interactive prompt)"""
import json, sys, time, urllib.request
url = "http://localhost:8080"; a = sys.argv[1:]
if a[:1] == ["--url"]: url, a = a[1], a[2:]
def send(t):
    r = urllib.request.urlopen(urllib.request.Request(url + "/api/command", json.dumps(dict(text=t)).encode(), method="POST"))
    print(json.loads(r.read())); time.sleep(0.25)
    print(" ->", json.loads(urllib.request.urlopen(url + "/api/state").read())["msg"])
if a: send(" ".join(a))
else:
    while True:
        try: t = input("kevin> ")
        except EOFError: break
        if t.strip(): send(t)
