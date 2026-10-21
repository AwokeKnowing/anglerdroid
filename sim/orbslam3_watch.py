#!/usr/bin/env python3
"""Keep ORB-SLAM3 host + a slow Kevin wander alive. Map snapshots land in
~/.kevin/orbslam3/. Does not disable safety or brown-border keepout.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST_PY = os.path.join(REPO, "sim", "orbslam3_host.py")
VENV_PY = os.path.join(REPO, ".venv", "bin", "python")
KEVIN = "192.168.50.107"
GOAL = "http://%s:8091/goal" % KEVIN
HEALTH = "http://%s:8091/health" % KEVIN
HB = os.path.expanduser("~/.kevin/orbslam3/heartbeat.txt")
STATE = os.path.expanduser("~/.kevin/orbslam3/watch_state.json")
HOST_LOG = os.path.expanduser("~/.kevin/orbslam3/host.log")
KEVIN_RESTARTS_MAX = 3
_FLOOR_RE = re.compile(r"floor=\(([-\d.]+),([-\d.]+)\)")


def _http_json(url, payload=None, timeout=3.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _pgrep(pat):
    try:
        out = subprocess.check_output(["pgrep", "-f", pat], text=True)
        return [int(x) for x in out.split() if x.strip()]
    except subprocess.CalledProcessError:
        return []


def _host_alive():
    return bool(_pgrep("sim/orbslam3_host.py")) and bool(
        _pgrep("sim/orbslam3/orbslam3_track"))


def _spawn_host():
    env = os.environ.copy()
    env["DISPLAY"] = env.get("DISPLAY") or ":1"
    env["ORBSLAM3_VIEWER"] = "1"
    env["RERUN_CONNECT"] = env.get(
        "RERUN_CONNECT", "rerun+http://127.0.0.1:9877/proxy")
    py = VENV_PY if os.path.isfile(VENV_PY) else "python3"
    log = open(os.path.expanduser("~/.kevin/orbslam3/host.log"), "a")
    subprocess.Popen(
        [py, "-u", HOST_PY], cwd=REPO, env=env,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print("orbslam3_watch: spawned host", flush=True)


def _kevin_main_alive():
    cmd = (
        "pgrep -f '/home/jetbot/anglerdroid/.venv/bin/python -u main.py' "
        ">/dev/null"
    )
    return subprocess.call(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         "jetbot@%s" % KEVIN, cmd]) == 0


def _restart_kevin():
    script = r"""
pkill -INT -f '/home/jetbot/anglerdroid/.venv/bin/python -u main.py' || true
sleep 6
pkill -TERM -f '/home/jetbot/anglerdroid/.venv/bin/python -u main.py' || true
sleep 2
cd /home/jetbot/anglerdroid/src
export KEVIN_EGO_EVERY=3 KEVIN_EGO_PLAN=1 KEVIN_LIVE_OBS=1 KEVIN_SPEAK_VOL=0.70 CAN_IFACE=can0
nohup /home/jetbot/anglerdroid/.venv/bin/python -u main.py \
  --slam self --wheel-imu-prior --rerun-hz 8 \
  --rerun-connect rerun+http://192.168.50.198:9877/proxy \
  --orbslam3-host 192.168.50.198 \
  --rs1 815412070676 --rs2 944622074292 --rgb1 /dev/video0 \
  --auto-local --local-planner insect --wander --people \
  >> /home/jetbot/.kevin/logs/armed-insect-orbslam3.log 2>&1 &
echo restarted:$!
"""
    subprocess.call(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         "jetbot@%s" % KEVIN, "bash -s"],
        input=script.encode())
    print("orbslam3_watch: restarted Kevin wander", flush=True)


def _nudge_off_mat(bearing_deg=85.0):
    # Do not resume_wander immediately: that clears the turn and he
    # reverse-pins on the brown door mat again.
    _http_json(GOAL, {"type": "relative_bearing", "bearing_deg": bearing_deg,
                      "distance_m": 0.7, "label": "away-from-mat"})
    time.sleep(12.0)
    _http_json(GOAL, {"type": "resume_wander"})
    print("orbslam3_watch: nudge turn %.0f then wander" % bearing_deg, flush=True)


def _host_floor():
    try:
        with open(HOST_LOG, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 6000))
            tail = f.read().decode("utf-8", "replace")
        last = None
        for m in _FLOOR_RE.finditer(tail):
            last = (float(m.group(1)), float(m.group(2)))
        return last
    except Exception:
        return None


def _load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return {"kevin_restarts": 0, "last_pose": None, "stuck_n": 0}


def _save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f)


def main():
    os.makedirs(os.path.expanduser("~/.kevin/orbslam3"), exist_ok=True)
    st = _load_state()
    last_nudge = time.time()
    print("orbslam3_watch: start", flush=True)
    while True:
        now = time.time()
        host_ok = _host_alive()
        if not host_ok:
            _spawn_host()
            time.sleep(8)
            host_ok = _host_alive()
        kevin_ok = _kevin_main_alive()
        if not kevin_ok and st.get("kevin_restarts", 0) < KEVIN_RESTARTS_MAX:
            st["kevin_restarts"] = int(st.get("kevin_restarts", 0)) + 1
            _restart_kevin()
            _save_state(st)
            time.sleep(12)
            kevin_ok = _kevin_main_alive()
        health = _http_json(HEALTH) if kevin_ok else {}
        floor = _host_floor()
        stuck = False
        if floor is not None:
            prev = st.get("last_pose")
            if prev and abs(floor[0] - prev[0]) < 0.04 and abs(floor[1] - prev[1]) < 0.04:
                st["stuck_n"] = int(st.get("stuck_n", 0)) + 1
            else:
                st["stuck_n"] = 0
            st["last_pose"] = list(floor)
            stuck = st["stuck_n"] >= 3
            _save_state(st)
        line = ("t=%.0f host=%s kevin=%s health=%s restarts=%d stuck=%s floor=%s\n"
                % (now, host_ok, kevin_ok, bool(health.get("ok", True) or health),
                   st.get("kevin_restarts", 0), stuck, floor))
        with open(HB, "w") as f:
            f.write(line)
        # Frozen pose or periodic: yaw off the door mat, then wander.
        if kevin_ok and (stuck or (now - last_nudge) > 400):
            _nudge_off_mat(85.0 if int(now) % 2 == 0 else -95.0)
            last_nudge = now
            st["stuck_n"] = 0
            _save_state(st)
        time.sleep(20)


if __name__ == "__main__":
    main()
