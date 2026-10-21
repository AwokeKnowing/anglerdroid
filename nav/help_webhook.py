"""Help / voice webhook client (stdlib only).  Never prints the URL or the key.
Config: env KEVIN_HELP_WEBHOOK_URL / KEVIN_HELP_WEBHOOK_KEY, else the file ~/.kevin/help_webhook.env (KEY=VALUE lines,
chmod 600, not in git; path override KEVIN_HELP_WEBHOOK_ENV).
Auth: 'Authorization: Bearer <key>' first; if the endpoint answers 401/403 it retries once with 'X-Webhook-Key: <key>' and
keeps whichever worked for the rest of the process.
help(): rate-limited to one POST per component per RATE_S (600 s), persisted in ~/.kevin/help_rate.json so restarts of
the supervisor do not spam.
  python -m nav.help_webhook test      one connectivity POST (component "test"; prints the HTTP status only)
"""
from __future__ import annotations
import json, os, socket, sys, time, urllib.error, urllib.request

RATE_S = 600.0
_ENV = os.path.expanduser(os.environ.get("KEVIN_HELP_WEBHOOK_ENV", "~/.kevin/help_webhook.env"))
_RATE = os.path.expanduser("~/.kevin/help_rate.json")
_auth_mode = ["bearer"]


def _cfg():
    url, key = os.environ.get("KEVIN_HELP_WEBHOOK_URL"), os.environ.get("KEVIN_HELP_WEBHOOK_KEY")
    if (not url or not key) and os.path.exists(_ENV):
        for ln in open(_ENV):
            k, _, v = ln.strip().partition("=")
            v = v.strip().strip('"').strip("'")
            if k == "KEVIN_HELP_WEBHOOK_URL" and not url: url = v
            if k == "KEVIN_HELP_WEBHOOK_KEY" and not key: key = v
    return url, key


def configured():
    u, k = _cfg(); return bool(u and k)


def post(payload, timeout=8.0):
    """-> (http_status or None, note).  Never raises."""
    url, key = _cfg()
    if not url or not key: return None, "webhook not configured (%s)" % _ENV
    body = json.dumps(payload, default=str).encode()
    for mode in ([_auth_mode[0]] + [m for m in ("bearer", "x-webhook-key") if m != _auth_mode[0]]):
        h = {"Content-Type": "application/json", "User-Agent": "kevin-help/1"}
        h.update({"Authorization": "Bearer " + key} if mode == "bearer" else {"X-Webhook-Key": key})
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h, method="POST"), timeout=timeout) as r:
                _auth_mode[0] = mode; return r.status, mode
        except urllib.error.HTTPError as e:
            if e.code in (401, 403): continue
            return e.code, mode
        except Exception as e:
            return None, "%s: %s" % (mode, e.__class__.__name__)
    return 401, "both auth headers rejected"


def _rate_ok(component):
    try: st = json.load(open(_RATE))
    except Exception: st = {}
    now = time.time()
    if now - float(st.get(component, 0)) < RATE_S: return False
    st[component] = now
    os.makedirs(os.path.dirname(_RATE), exist_ok=True)
    tmp = _RATE + ".tmp"; json.dump(st, open(tmp, "w")); os.replace(tmp, _RATE)
    return True


def help(component, error, attempts=0, log_tail="", state=None, robot=None, force=False):
    """POST {"kind": "help", ...} unless this component already asked within RATE_S.  -> (status, note)"""
    if not force and not _rate_ok(component): return None, "rate-limited (one per %d s per component)" % RATE_S
    robot = robot or os.environ.get("KEVIN_ROBOT", "kevin001")
    return post(dict(kind="help", robot=robot, component=component, error=error, attempts=attempts,
                     log_tail=log_tail[-4000:] if log_tail else "", state=state or {}, host=socket.gethostname(),
                     time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


if __name__ == "__main__":
    if sys.argv[1:2] == ["test"]:
        s, note = post(dict(kind="help", robot=os.environ.get("KEVIN_ROBOT", "kevin001"), component="test",
                            error="webhook connectivity test, no action needed", time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        print("HTTP status:", s, "(auth %s)" % note if s else note)
    else:
        print(__doc__)
