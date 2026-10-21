"""Live LLM acceptance suite for nav.agent (needs the local LLM: KEVIN_LLM_URL/KEVIN_LLM_MODEL).
python tests/test_agent_phrasings.py   -> prints per-case result + score; exit 0 if >= 18/20 and every nonsense case is motion-free."""
import math, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nav import agent as A

PLACES = ["kitchen", "living room", "bedroom", "bathroom", "front door", "hallway"]
def kind(k, **kw):
    def chk(c):
        if c["kind"] != k: return False
        for a, v in kw.items():
            x = c.get(a)
            if callable(v):
                if not v(x): return False
            elif isinstance(v, float):
                if x is None or abs(x - v) > 1e-3: return False
            elif x != v: return False
        return True
    return chk
deg = lambda d: math.radians(d)
place = lambda n: kind("goto_place", name=lambda x: x and (n in x.lower()))
left = lambda d: kind("turn", rad=lambda r: r is not None and abs(r - deg(d)) < 0.05)
right = lambda d: kind("turn", rad=lambda r: r is not None and abs(r + deg(d)) < 0.05)
spin = kind("turn", rad=lambda r: r is not None and (abs(abs(r) - math.pi) < 0.05 or abs(abs(r) - 2 * math.pi) < 0.05))
fwd = lambda lo, hi: kind("relative", d=lambda d: d is not None and lo <= d <= hi)
back = lambda lo, hi: kind("relative", d=lambda d: d is not None and -hi <= d <= -lo)
ask = lambda w: kind("ask", text=lambda t: t and w in t.lower())

CASES = [  # (phrase, [checks for the first N commands])
    ("head to the kitchen", [place("kitchen")]),
    ("can you go over to the bedroom please", [place("bedroom")]),
    ("take me to the front door", [place("front door")]),
    ("I'd like you to drive into the living room", [place("living room")]),
    ("spin around", [spin]),
    ("turn around", [spin]),
    ("back up a bit", [back(0.2, 1.0)]),
    ("move forward two meters", [fwd(2.0, 2.0)]),
    ("go ahead about half a metre", [fwd(0.4, 0.6)]),
    ("rotate 45 degrees to the right", [right(45)]),
    ("turn left a quarter turn", [left(90)]),
    ("remember this spot as the couch", [kind("label", name=lambda n: n and "couch" in n)]),
    ("this spot is the charging dock, remember it", [kind("label", name=lambda n: n and "dock" in n)]),
    ("go find the dog", [ask("dog")]),
    ("approach the red chair", [ask("chair")]),
    ("where are you right now?", [kind("where")]),
    ("what places do you know?", [kind("places")]),
    ("freeze!", [kind("stop")]),
    ("go to the bathroom and then come back to the kitchen", [place("bathroom"), place("kitchen")]),
    ("face 180 degrees", [kind("face", rad=lambda r: r is not None and abs(abs(r) - math.pi) < 0.05)]),
]
NONSENSE = ["what is the capital of France", "blorp zingle wump", "tell me a joke", "drive off the balcony and jump", "how are you feeling today"]
MOTION = {"goto_place", "goto_xy", "relative", "turn", "face", "drive", "ask"}

if __name__ == "__main__":
    ok = 0; lat = []
    print("model", A.MODEL, "url", A.URL)
    for ph, chks in CASES:
        r = A.interpret(ph, PLACES); lat.append(r["ms"])
        cs = [c for _, c in r["cmds"]]
        good = len(cs) >= len(chks) and all(ch(c) for ch, c in zip(chks, cs))
        ok += good
        print("%s %-55s -> %-40s %s %5.0f ms  %s" % ("PASS" if good else "FAIL", ph, " | ".join(t for t, _ in r["cmds"]) or "-", r["via"], r["ms"], (r["reply"] or "")[:60]))
    nm_ok = 0
    for ph in NONSENSE:
        r = A.interpret(ph, PLACES); moving = [t for t, c in r["cmds"] if c["kind"] in MOTION]
        g = not moving and bool(r["reply"]); nm_ok += g
        print("%s NONSENSE %-46s -> %-30s reply=%s" % ("PASS" if g else "FAIL", ph, " | ".join(moving) or "no motion", (r["reply"] or "")[:70]))
    t0 = time.monotonic(); r = A.interpret("stop", PLACES); st = (time.monotonic() - t0) * 1e3
    print("stop via %s in %.2f ms" % (r["via"], st))
    lat.sort(); print("SCORE %d/%d phrasings, nonsense %d/%d motion-free, llm latency p50 %.0f ms max %.0f ms" % (ok, len(CASES), nm_ok, len(NONSENSE), lat[len(lat) // 2], lat[-1]))
    sys.exit(0 if ok >= 18 and nm_ok == len(NONSENSE) and r["via"] == "grammar" else 1)
