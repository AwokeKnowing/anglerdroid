#!/usr/bin/env python3
"""One 5s Kevin batch: look, optional twist, log before/after pose."""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request

BASE = "http://192.168.50.107:8091"
OUT = pathlib.Path("/tmp/kevin_pilot")
LOG = pathlib.Path(__file__).resolve().parent / "out" / "pilot_cmds.jsonl"


def _get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=8) as r:
        return json.loads(r.read().decode())


def _post(path: str, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def _look(tag: str):
    OUT.mkdir(exist_ok=True)
    for cam in ("all", "webcam", "rs1", "rs2"):
        with urllib.request.urlopen(f"{BASE}/look/{cam}.jpg", timeout=8) as r:
            (OUT / f"{cam}_{tag}.jpg").write_bytes(r.read())


def _log(rec: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=float, default=None)
    ap.add_argument("--ang", type=float, default=None)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--look-only", action="store_true")
    ap.add_argument("--tag", default="now")
    args = ap.parse_args()
    st0 = _get("/status")
    _look(args.tag)
    print("STATUS", json.dumps(st0))
    if args.look_only or args.fwd is None:
        return
    if not st0.get("topdown_ok"):
        print("ABORT topdown_lost")
        return
    if (args.fwd or 0) > 0 and st0.get("safety", {}).get("fwd", 0) < 0.35:
        print("ABORT safety_fwd", st0.get("safety"))
        return
    body = {
        "forward_mps": float(args.fwd or 0.0),
        "angular_rads": float(args.ang or 0.0),
        "duration_secs": float(args.dur),
    }
    r = _post("/twist", body)
    print("TWIST", r.get("applied"), r.get("ok"))
    time.sleep(float(args.dur) + 1.4)
    st1 = _get("/status")
    _look(args.tag + "b")
    p0, p1 = st0.get("pose") or {}, st1.get("pose") or {}
    dx = p1.get("x", 0) - p0.get("x", 0)
    dy = p1.get("y", 0) - p0.get("y", 0)
    rec = {
        "t_wall": time.time(),
        "bag": "wander_pilot_c",
        "cmd": "twist_for",
        **body,
        "pose0": p0,
        "pose1": p1,
        "dist_m": (dx * dx + dy * dy) ** 0.5,
        "dyaw_deg": p1.get("yaw_deg", 0) - p0.get("yaw_deg", 0),
        "safety0": st0.get("safety"),
        "safety1": st1.get("safety"),
        "applied": r.get("applied"),
    }
    _log(rec)
    print("AFTER", json.dumps(st1.get("pose")), "saf", st1.get("safety"),
          "bag", st1.get("bag"))
    print("dist=%.3f dyaw=%.1f" % (rec["dist_m"], rec["dyaw_deg"]))


if __name__ == "__main__":
    main()
