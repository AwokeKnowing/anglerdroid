import json, math, os, subprocess, sys, time
sys.path.insert(0, __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')))
from emu.eval import http, cmd, REPO, SIMPY
from emu.shm import read_info
from emu import gt
tag, pitch = sys.argv[1], sys.argv[2]; O = os.path.expanduser("~/.kevin/emu/rehearse_" + tag); os.makedirs(O, exist_ok=True)
subprocess.call(["pkill", "-f", "emu.world"]); subprocess.call(["pkill", "-f", "nav.run"]); subprocess.call(["pkill", "-f", "viewer.py"]); time.sleep(2)
cal = O + "/calib.json"; os.path.exists(cal) and os.remove(cal)
w = subprocess.Popen([SIMPY, "-m", "emu.world", "--env", "room", "--seed", "0"], cwd=REPO, env=dict(os.environ, MUJOCO_GL="egl", KEVIN_EMU_NOISE="realistic", KEVIN_EMU_RS2_PITCH=pitch), stdout=open(O + "/world.log", "w"), stderr=subprocess.STDOUT)
time.sleep(10)
n = subprocess.Popen(["scripts/bringup_run.sh", "0.3"], cwd=REPO, env=dict(os.environ, KEVIN_HAL="sim", KEVIN_PY=os.path.expanduser("~/.venv-kevinrobot/bin/python"), KEVIN_CALIB=cal, KEVIN_PORT="8092", KEVIN_RECORD=O + "/log"), stdout=open(O + "/run.out", "w"), stderr=subprocess.STDOUT)
for _ in range(120):
    try: http(8092, "/api/state"); break
    except Exception: time.sleep(0.5)
time.sleep(12)                                    # parked: auto-cal windows
V = []; W = []
def watch(sec):
    t0 = time.time()
    while time.time() - t0 < sec: s = gt.state(); V.append(abs(s["v"])); W.append(abs(s["w"])); time.sleep(0.02)
cmd(8092, "drive 0.6 0 3"); watch(4); time.sleep(3)
cmd(8092, "drive 0 2.0 2"); watch(3); time.sleep(3)
cmd(8092, "drive -0.5 0 2"); watch(3); cmd(8092, "stop"); time.sleep(4)
st = http(8092, "/api/state")
n.terminate(); time.sleep(2); subprocess.call(["pkill", "-f", "nav.run"]); w.terminate()
C = json.load(open(cal)) if os.path.exists(cal) else None
print(json.dumps(dict(tag=tag, pitch=pitch, v_max_gt=round(max(V), 3), w_max_gt=round(max(W), 3), calib=C and C.get("rs2_autocal"), safety=st.get("safety"))))
