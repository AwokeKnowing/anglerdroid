"""Typed command language -> goals in map coordinates.  One parser shared by CLI, web page and (later) the
conversational layer: anything that can produce a string can drive the robot.

  go to <place> | go to <x> <y> [heading_deg] | forward <m> | back <m> | turn left|right <deg> | face <deg>
  drive <v> <w> [secs]   (direct velocity, still safety-filtered)
  stop | label <name> [at <x> <y>] | forget <name> | places | where | save [map] | load <map>
  ask <instruction>      (VLN hook: LightNav-0 / future chat model)"""
from __future__ import annotations
import json, math, os, re, time
import numpy as np

MAPS = os.path.expanduser(os.environ.get("KEVIN_MAPS", "~/.kevin/maps"))


class Places:
    def __init__(self): self.d: dict[str, list[float]] = {}
    def set(self, name, x, y, th=0.0): self.d[name.lower().strip()] = [round(float(x), 3), round(float(y), 3), round(float(th), 3)]
    def get(self, name):
        n = name.lower().strip()
        if n in self.d: return self.d[n]
        hits = [k for k in self.d if n in k or k in n]          # 'kitchen' matches 'kitchen table'
        return self.d[hits[0]] if len(hits) == 1 else None
    def remove(self, name): return self.d.pop(name.lower().strip(), None) is not None


def save_map(name, grid, places, pose):
    d = os.path.join(MAPS, name); os.makedirs(d, exist_ok=True)
    with grid.lock:
        np.savez_compressed(os.path.join(d, "grid.npz"), L=grid.L, seen=grid.seen, origin=grid.origin, res=grid.res)
    json.dump(dict(places=places.d, pose=list(pose), saved=time.time()), open(os.path.join(d, "places.json"), "w"), indent=1)
    return d


def load_map(name, grid, places):
    d = os.path.join(MAPS, name); z = np.load(os.path.join(d, "grid.npz"))
    with grid.lock:
        grid.L[:] = z["L"]; grid.seen[:] = z["seen"]; grid.origin = z["origin"]; grid.version += 1
    grid.refresh_likelihood()
    j = json.load(open(os.path.join(d, "places.json"))); places.d = j["places"]
    return j.get("pose")


_NUM = r"(-?\d+(?:\.\d+)?)"


def parse(text: str):
    """-> dict(kind=..., ...) or dict(kind='error', msg=...). Pure function (unit-testable)."""
    t = text.strip().lower().replace(",", " ")
    if not t: return dict(kind="error", msg="empty")
    if t in ("stop", "halt", "cancel", "s"): return dict(kind="stop")
    if t in ("places", "list"): return dict(kind="places")
    if t in ("where", "pose", "where am i"): return dict(kind="where")
    m = re.fullmatch(r"(?:go ?to|goto|go|g)\s+" + _NUM + r"\s+" + _NUM + r"(?:\s+" + _NUM + r")?", t)
    if m: return dict(kind="goto_xy", x=float(m[1]), y=float(m[2]), heading=None if m[3] is None else math.radians(float(m[3])))
    m = re.fullmatch(r"(?:go ?to|goto|go|g)\s+(?:the\s+)?(.+)", t)
    if m: return dict(kind="goto_place", name=m[1])
    m = re.fullmatch(r"(?:forward|fwd|f)\s+" + _NUM + r"\s*m?", t)
    if m: return dict(kind="relative", d=float(m[1]))
    m = re.fullmatch(r"(?:back|backward|b)\s+" + _NUM + r"\s*m?", t)
    if m: return dict(kind="relative", d=-float(m[1]))
    m = re.fullmatch(r"turn(?:\s+(left|right))?\s*" + _NUM + r"?\s*(?:deg)?", t)
    if m:
        a = float(m[2]) if m[2] else 90.0
        return dict(kind="turn", rad=math.radians(-a if m[1] == "right" else a))
    m = re.fullmatch(r"drive\s+" + _NUM + r"\s+" + _NUM + r"(?:\s+" + _NUM + r")?", t)
    if m: return dict(kind="drive", v=float(m[1]), w=float(m[2]), secs=float(m[3] or 1.0))
    m = re.fullmatch(r"face\s+" + _NUM, t)
    if m: return dict(kind="face", rad=math.radians(float(m[1])))
    m = re.fullmatch(r"label\s+(.+?)\s+at\s+" + _NUM + r"\s+" + _NUM, t)
    if m: return dict(kind="label", name=m[1], x=float(m[2]), y=float(m[3]))
    m = re.fullmatch(r"(?:label|name|this is)\s+(?:the\s+)?(.+)", t)
    if m: return dict(kind="label", name=m[1])
    m = re.fullmatch(r"forget\s+(.+)", t)
    if m: return dict(kind="forget", name=m[1])
    m = re.fullmatch(r"save(?:\s+(\S+))?", t)
    if m: return dict(kind="save", name=m[1] or "default")
    if t in ("relocalize", "reloc", "where am i really"): return dict(kind="relocalize")
    m = re.fullmatch(r"load\s+(\S+)", t)
    if m: return dict(kind="load", name=m[1])
    m = re.fullmatch(r"(?:ask|vln)\s+(.+)", t)
    if m: return dict(kind="ask", text=m[1])
    return dict(kind="error", msg="unknown command: %r (try: go to kitchen | forward 1 | turn left 90 | label sofa | stop)" % text)
