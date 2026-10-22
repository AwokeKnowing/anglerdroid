#!/usr/bin/env python3
"""i777 host SLAM view: casita walls prior + door keepout into the live Rerun.

Does not drive Kevin. Loads the house STL, logs wall segments and a hard
front-door disk to the native Rerun on :9877 so we can mark that area as
no-go once live pose is aligned.

  KEVIN_WALLS_STL_PRIOR  default: the resolved world mesh (src/world_params.py <- worlds/casita/world.json)
  RERUN_CONNECT          default: rerun+http://127.0.0.1:9877/proxy
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import importlib.util

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

if SRC not in sys.path: sys.path.append(SRC)
import world_params  # noqa: E402  the resolved house mesh (worlds/casita/world.json)
DEFAULT_STL = world_params.walls_stl()
# Casita south door opening (visit_wander portal median).
DOOR_XY = (-0.60, -2.81)
DOOR_R_M = 0.90
KEEP_PATH = os.path.expanduser("~/.kevin/keepouts.json")


def _circle_xy(cx, cy, r, n=48, z=0.04):
    pts = []
    for i in range(n + 1):
        a = 2.0 * math.pi * i / n
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a), z])
    return pts


def _write_keepout_json():
    os.makedirs(os.path.dirname(KEEP_PATH), exist_ok=True)
    data = {
        "version": 1,
        "note": "i777 casita frame. front_door is hard. Align to Kevin session before sync.",
        "named": [
            {"id": "front_door", "label": "front door opening", "kind": "hard"},
            {"id": "checkered_door", "label": "checkered area by door", "kind": "floor_mat"},
        ],
        "polygons_map_m": [],
        "disks_map_m": [
            {
                "id": "front_door",
                "x": DOOR_XY[0],
                "y": DOOR_XY[1],
                "r_m": DOOR_R_M,
                "kind": "hard",
                "paint": 200,
            }
        ],
    }
    with open(KEEP_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("slam_host: wrote %s" % KEEP_PATH)
    return data


def main():
    stl = os.environ.get("KEVIN_WALLS_STL_PRIOR", "").strip() or DEFAULT_STL
    os.environ["KEVIN_WALLS_STL_PRIOR"] = stl
    connect = os.environ.get(
        "RERUN_CONNECT", "rerun+http://127.0.0.1:9877/proxy")

    spec = importlib.util.spec_from_file_location(
        "walls_stl_prior",
        os.path.join(SRC, "perception", "walls_stl_prior.py"),
    )
    walls_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walls_mod)
    import rerun as rr

    prior = walls_mod.WallsSTLPrior()
    if not prior.is_loaded():
        print("slam_host: walls prior did not load (%s)" % stl)
        return 1
    segs = prior.get_wall_segments()
    print("slam_host: %d wall segments from %s" % (len(segs), stl))

    _write_keepout_json()

    rr.init("kevin_anglerdroid")
    rr.connect_grpc(connect)
    rr.set_time("host", timestamp=time.time())

    strips = []
    colors = []
    for x0, y0, x1, y1 in segs:
        strips.append([[float(x0), float(y0), 0.02],
                       [float(x1), float(y1), 0.02]])
        colors.append([210, 190, 140])
    rr.log("slam/walls", rr.LineStrips3D(strips, colors=colors), static=True)
    rr.log(
        "slam/keepout/front_door",
        rr.LineStrips3D([_circle_xy(DOOR_XY[0], DOOR_XY[1], DOOR_R_M)],
                        colors=[[220, 40, 40]]),
        static=True,
    )
    rr.log(
        "slam/keepout/front_door_pt",
        rr.Points3D([[DOOR_XY[0], DOOR_XY[1], 0.08]],
                    radii=0.08, colors=[[220, 40, 40]]),
        static=True,
    )
    print("slam_host: logged walls + front_door keepout to %s" % connect)
    print("slam_host: door disk (%.2f, %.2f) r=%.2fm hard" % (
        DOOR_XY[0], DOOR_XY[1], DOOR_R_M))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
