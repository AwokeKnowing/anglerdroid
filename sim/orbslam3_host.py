#!/usr/bin/env python3
"""i777 ORB-SLAM3 host: spawn C++ tracker, log pose + casita walls into Rerun.

Does not drive Kevin. Pair with KEVIN_ORBSLAM3_HOST=<i777> on the robot so RS2
stereo IR arrives on TCP :5571. Pose JSON lands on UDP :5572.

  python3 sim/orbslam3_host.py          # Pangolin map viewer + Rerun
  ORBSLAM3_VIEWER=0 python3 sim/orbslam3_host.py   # Rerun only
"""
from __future__ import annotations

import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORB = os.path.expanduser(os.environ.get(
    "ORB_SLAM3_ROOT", os.path.join(os.path.dirname(REPO), "ORB_SLAM3")))
PREFIX = os.path.expanduser(os.environ.get(
    "ORB_SLAM3_PREFIX", os.path.expanduser("~/.local/share/mamba/envs/orbslam3")))
TRACK = os.path.join(REPO, "sim", "orbslam3", "orbslam3_track")
VOC = os.path.join(ORB, "Vocabulary", "ORBvoc.txt")
YAML = os.path.join(REPO, "sim", "orbslam3", "D435i.yaml")
LISTEN_PORT = int(os.environ.get("KEVIN_ORBSLAM3_PORT", "5571"))
POSE_PORT = int(os.environ.get("KEVIN_ORBSLAM3_POSE_PORT", "5572"))
RERUN_CONNECT = os.environ.get("RERUN_CONNECT", "rerun+http://127.0.0.1:9877/proxy")

# Same as cuvslam_tracker: RS2 pitched down; ORB world is first-camera frame.
# Floor guess: our_x = cam_z, our_y = -cam_x (OpenCV X-right Y-down Z-forward).
CAMERA_PITCH_DEG = 64.4


def _log_walls():
    slam_host = os.path.join(REPO, "sim", "slam_host.py")
    if os.path.isfile(slam_host):
        env = os.environ.copy()
        env["RERUN_CONNECT"] = RERUN_CONNECT
        subprocess.run([sys.executable, slam_host], check=False, env=env)


def _spawn_tracker():
    if not os.path.isfile(TRACK):
        print("orbslam3_host: missing %s — run sim/orbslam3/build.sh" % TRACK)
        return None
    if not os.path.isfile(VOC):
        print("orbslam3_host: missing vocab %s" % VOC)
        return None
    env = os.environ.copy()
    lib = os.path.join(PREFIX, "lib")
    orblib = os.path.join(ORB, "lib")
    env["LD_LIBRARY_PATH"] = "%s:%s:%s" % (
        lib, orblib, env.get("LD_LIBRARY_PATH", ""))
    env.setdefault("DISPLAY", os.environ.get("DISPLAY", ":1"))
    if env.get("ORBSLAM3_VIEWER", "") != "0":
        env["ORBSLAM3_VIEWER"] = "1"
    cmd = [TRACK, VOC, YAML, str(LISTEN_PORT), "127.0.0.1", str(POSE_PORT)]
    print("orbslam3_host: exec %s viewer=%s DISPLAY=%s"
          % (" ".join(cmd), env.get("ORBSLAM3_VIEWER"), env.get("DISPLAY")))
    return subprocess.Popen(cmd, env=env)


def _listen_pose():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", POSE_PORT))
    sock.settimeout(0.5)
    return sock


def main():
    os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":1"))
    try:
        import rerun as rr
        rr.init("kevin_anglerdroid", spawn=False)
        rr.connect_grpc(RERUN_CONNECT)
        print("orbslam3_host: rerun %s" % RERUN_CONNECT)
    except Exception as e:
        print("orbslam3_host: rerun unavailable (%s)" % e)
        rr = None

    _log_walls()
    proc = _spawn_tracker()
    sock = _listen_pose()
    print("orbslam3_host: pose UDP :%d  (waiting for tracker + Kevin IR)"
          % POSE_PORT)

    stop = {"n": 0}

    def _sig(sig, frame):
        stop["n"] = 1

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    n = 0
    last = 0.0
    path = []
    while not stop["n"]:
        if proc is not None and proc.poll() is not None:
            print("orbslam3_host: tracker exited %s" % proc.returncode)
            break
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        if data.startswith(b"ORBP") and len(data) >= 8:
            npt = struct.unpack_from("<H", data, 4)[0]
            need = 8 + npt * 12
            if npt and len(data) >= need and rr is not None:
                pts = np.frombuffer(data, dtype="<f4", offset=8, count=npt * 3)
                pts = np.ascontiguousarray(pts.reshape(-1, 3), dtype=np.float32)
                try:
                    rr.log("slam/orb/map", rr.Points3D(pts, radii=0.012,
                                                      colors=[80, 180, 255]))
                except Exception:
                    pass
            continue
        if data.startswith(b"ORBI") and len(data) >= 8:
            nj = struct.unpack_from("<I", data, 4)[0]
            jpg = data[8:8 + nj]
            if rr is not None and jpg:
                try:
                    import cv2
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                                       cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        rr.log("slam/orb/ir_left", rr.Image(img))
                except Exception:
                    pass
            continue
        try:
            msg = json.loads(data.decode("ascii"))
        except Exception:
            continue
        n += 1
        ok = bool(msg.get("ok"))
        x = float(msg.get("x", 0))
        y = float(msg.get("y", 0))
        z = float(msg.get("z", 0))
        yaw = float(msg.get("yaw", 0))
        # Floor-frame guess (same mapping as cuVSLAM rig → our world).
        floor_x = z
        floor_y = -x
        if rr is not None:
            try:
                rr.set_time("orb", sequence=int(msg.get("n", n)))
                rr.log("slam/orb/cam", rr.Points3D([[x, y, z]], radii=0.05,
                                                   colors=[255, 80, 40]))
                rr.log("slam/orb/floor_xy",
                       rr.Points3D([[floor_x, floor_y, 0.05]], radii=0.06,
                                   colors=[255, 200, 40]))
                path.append([floor_x, floor_y, 0.02])
                if len(path) > 4000:
                    path = path[-4000:]
                if len(path) >= 2:
                    rr.log("slam/orb/path", rr.LineStrips3D([path],
                                                           colors=[255, 160, 40]))
            except Exception:
                pass
        now = time.monotonic()
        if now - last >= 2.0:
            print("orbslam3_host: n=%d ok=%s state=%s cam=(%.2f,%.2f,%.2f) "
                  "floor=(%.2f,%.2f) yaw=%.1fdeg pitch_ref=%.1f"
                  % (n, ok, msg.get("state"), x, y, z, floor_x, floor_y,
                     math.degrees(yaw), CAMERA_PITCH_DEG))
            last = now

    sock.close()
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
    print("orbslam3_host: done (%d poses)" % n)


if __name__ == "__main__":
    main()
