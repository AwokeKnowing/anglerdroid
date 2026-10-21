"""Chat agent: plain English -> validated Kevin commands (the goals.parse grammar is the typed API).

  1. Exact grammar first: if goals.parse() understands the text it goes straight to the brain (stop never waits on a model).
  2. Otherwise a local LLM (any OpenAI-compatible /v1/chat/completions server; default ollama on :11434) is asked for
     JSON {"commands": [...], "reply": "..."}.  Every command is re-parsed with goals.parse and must be one of the
     allowed kinds; anything else is dropped.  No valid command -> polite clarification, never motion.

env: KEVIN_LLM_URL (default http://localhost:11434/v1), KEVIN_LLM_MODEL (default qwen2.5:7b-instruct), KEVIN_LLM_KEY (opt)."""
from __future__ import annotations
import json, os, re, time, urllib.request
from nav import goals as G

ALLOWED = {"stop", "goto_place", "goto_xy", "relative", "turn", "face", "label", "forget", "where", "places", "save", "ask"}
URL = os.environ.get("KEVIN_LLM_URL", "http://localhost:11434/v1").rstrip("/")
MODEL = os.environ.get("KEVIN_LLM_MODEL", "qwen2.5:7b-instruct")

SYSTEM = """You are the command interpreter for Kevin, a small indoor wheeled robot. Convert the user's message into robot commands.
Reply with ONLY a JSON object: {"commands": [<command strings>], "reply": "<one short sentence to the user>"}.
Command strings (exact syntax, lowercase):
  go to <place>          drive to a known named place. Known places: %(places)s
  go to <x> <y>          drive to map coordinates in metres
  forward <m>            drive straight ahead <m> metres (e.g. forward 1)
  back <m>               reverse <m> metres (max ~1)
  turn left <deg> | turn right <deg>   rotate in place ("spin around"/"turn around" = turn left 180, a full spin = turn left 360)
  face <deg>             face an absolute map heading in degrees
  label <name>           remember the current position as <name>
  forget <name>          delete a remembered place
  where                  report position (use for "where are you?", "what's your location?")
  places                 list known places (use for "what places do you know?", "where can you go?")
  stop                   stop immediately
  ask <instruction>      visual navigation by the camera model, for targets that are NOT a known place and must be found by
                         looking (e.g. "go find the dog" -> "ask go to the dog", "approach the red chair" -> "ask approach the red chair")
Rules: "a bit"/"a little" = 0.5 m; "a few steps" = 1 m. Use a known place name only if it is in the list (closest match). Several steps in
order are allowed. If the message is not a robot instruction, is unclear, or unsafe, return "commands": [] and ask a short clarifying
question in "reply". Never invent coordinates.
Examples: "where are you?" -> {"commands": ["where"], "reply": "Here's where I am."}
"go to the bedroom then the kitchen" -> {"commands": ["go to bedroom", "go to kitchen"], "reply": "Bedroom first, then the kitchen."}"""


def _strip(t):
    t = t.strip().rstrip(".!?").strip()
    return re.sub(r"^(please|kevin|hey kevin|ok kevin)[ ,]+", "", t, flags=re.I)


def exact(text):
    """grammar pass-through: dict if goals.parse understands it (and it is a real command), else None."""
    c = G.parse(text)
    return None if c["kind"] == "error" else c


def llm(text, places=(), timeout=20.0):
    body = dict(model=MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[dict(role="system", content=SYSTEM % dict(places=", ".join(places) or "none")), dict(role="user", content=text)])
    h = {"Content-Type": "application/json"}
    if os.environ.get("KEVIN_LLM_KEY"): h["Authorization"] = "Bearer " + os.environ["KEVIN_LLM_KEY"]
    rq = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(), h)
    r = json.loads(urllib.request.urlopen(rq, timeout=timeout).read())
    out = r["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", out, re.S)
    return json.loads(m[0] if m else out)


def validate(cmds, places=()):
    """-> list of (text, parsed dict) that are in the allowed command set; unknown places dropped."""
    ok = []
    for s in (cmds or [])[:4]:
        if not isinstance(s, str): continue
        s = s.strip().rstrip(".!?").strip()
        c = G.parse(s)
        if c["kind"] not in ALLOWED: continue
        if c["kind"] == "goto_place" and places:
            n = c["name"].lower().strip(); P = [p.lower() for p in places]
            if not (n in P or any(n in p or p in n for p in P)): continue
        if c["kind"] == "relative" and not (-1.5 <= c["d"] <= 5.0): continue
        ok.append((s.strip().lower(), c))
    return ok


def interpret(text, places=()):
    """-> dict(cmds=[(text, parsed)], reply=str, via='grammar'|'llm'|'none', ms=float). Pure apart from the LLM call."""
    t0 = time.monotonic(); t = _strip(text)
    c = exact(t)
    if c is not None and c["kind"] == "goto_place" and (re.search(r"\b(and|then)\b", c["name"]) or (places and not validate(["go to " + c["name"]], places))):
        c = None                                         # 'go to the dog' / 'go find X': not a known place -> let the LLM pick (e.g. ask)
    if c is not None: return dict(cmds=[(t.lower(), c)], reply=None, via="grammar", ms=(time.monotonic() - t0) * 1e3)
    if re.fullmatch(r"(stop|halt|freeze|wait|stop it|stop now|emergency stop|whoa)( kevin)?", t, re.I):
        return dict(cmds=[("stop", dict(kind="stop"))], reply=None, via="grammar", ms=(time.monotonic() - t0) * 1e3)
    try: j = llm(t, places)
    except Exception as e:
        return dict(cmds=[], reply="I can't reach my language model right now (%s). Try an exact command like 'go to kitchen' or 'forward 1'." % type(e).__name__, via="none", ms=(time.monotonic() - t0) * 1e3)
    v = validate(j.get("commands"), places)
    reply = str(j.get("reply") or "").strip()[:200]
    if not v and not reply: reply = "Sorry, I didn't understand. You can say things like 'go to the kitchen', 'turn left', or 'forward 1 metre'."
    if not v and j.get("commands"): reply = (reply + " (I couldn't turn that into a safe command.)").strip()
    return dict(cmds=v, reply=reply or None, via="llm", ms=(time.monotonic() - t0) * 1e3)
