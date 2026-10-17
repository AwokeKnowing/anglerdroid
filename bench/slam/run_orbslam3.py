"""ORB-SLAM3 stereo (repo build sim/orbslam3/orbslam3_track, default stereo settings from sim/orbslam3/D435i.yaml,
calibrated with the twin IR intrinsics/baseline; viewer off) on the RS2 IR pair of a recorded sim log.  CPU only.
-> RESULTS/<seq>/orbslam3_stereo.tum (rs2_ir1 optical frame, metric) + .json
python bench/slam/run_orbslam3.py DATA/<seq> RESULTS/<seq>      (any python with numpy+cv2)"""
import json, os, sys, time, socket, struct, subprocess
import numpy as np, cv2
seq, res = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); os.makedirs(res, exist_ok=True)
REPO = os.environ.get("KEVIN_REPO", os.path.expanduser("~/gits/awokeknowing/anglerdroid"))
ORB = os.environ.get("ORB_SLAM3_DIR", os.path.expanduser("~/gits/awokeknowing/ORB_SLAM3"))
PREFIX = os.path.expanduser("~/.local/share/mamba/envs/orbslam3")
M = json.load(open(os.path.join(seq, "meta.json"))); I = M["intr"]["rs2_ir"]; B = M["intr"]["ir_baseline_m"]; W, H = I["w"], I["h"]
F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
y = open(os.path.join(REPO, "sim/orbslam3/D435i.yaml")).read()
vals = {"Camera1.fx": I["fx"], "Camera1.fy": I["fy"], "Camera1.cx": I["ppx"], "Camera1.cy": I["ppy"], "Camera.width": W, "Camera.height": H,
        "Camera.fps": int(round(M.get("cam_hz", 15))), "Stereo.b": B}
y = "\n".join((ln.split(":")[0] + ": " + (str(int(vals[ln.split(":")[0]])) if ln.split(":")[0] in ("Camera.width", "Camera.height", "Camera.fps") else "%.4f" % vals[ln.split(":")[0]]))
              if ln.split(":")[0] in vals else ln for ln in y.split("\n") if "SaveAtlas" not in ln)
yp = os.path.join(res, "_orb.yaml"); open(yp, "w").write(y)
port = int(os.environ.get("ORB_PORT", "5691"))
env = dict(os.environ, ORBSLAM3_VIEWER="0", ORBSLAM3_SPECKLE_K="0", LD_LIBRARY_PATH="%s/lib:%s/lib:%s" % (PREFIX, ORB, os.environ.get("LD_LIBRARY_PATH", "")))
tum = os.path.expanduser("~/.kevin/orbslam3/traj_tum.txt")
if os.path.exists(tum): os.remove(tum)
p = subprocess.Popen([REPO + "/sim/orbslam3/orbslam3_track", ORB + "/Vocabulary/ORBvoc.txt", yp, str(port), "127.0.0.1", str(port + 1)],
                     env=env, stdout=open(os.path.join(res, "_orb.log"), "w"), stderr=subprocess.STDOUT)
for _ in range(180):
    try: s = socket.create_connection(("127.0.0.1", port)); break
    except OSError: time.sleep(1)
t0 = time.perf_counter()
for i, t in zip(F["i"].astype(int), F["t"]):
    a = cv2.imread(os.path.join(seq, "rs2_ir1", "%06d.png" % i), 0); b = cv2.imread(os.path.join(seq, "rs2_ir2", "%06d.png" % i), 0)
    body = struct.pack("<QHH5f", int(t * 1e9), W, H, I["fx"], I["fy"], I["ppx"], I["ppy"], B) + a.tobytes() + b.tobytes()
    s.sendall(struct.pack("<I", len(body)) + body)
s.shutdown(socket.SHUT_WR); p.wait(timeout=3600); wall = time.perf_counter() - t0
D = np.loadtxt(tum, ndmin=2) if os.path.exists(tum) and os.path.getsize(tum) else np.zeros((0, 8))
if len(D): D[:, 0] = D[:, 0] / (1e9 if D[0, 0] > 1e6 else 1.0)
np.savetxt(os.path.join(res, "orbslam3_stereo.tum"), D, fmt="%.6f")
json.dump(dict(frame="cam:rs2_ir1", scale_free=False, per_frame=True, note="ORB-SLAM3 stereo, twin-calibrated IR pair, default ORB params",
               runtime="%.1f ms/frame wall incl. png decode (%d frames, CPU)" % (wall / len(F["t"]) * 1e3, len(F["t"]))),
          open(os.path.join(res, "orbslam3_stereo.json"), "w"), indent=1)
print("orb done", len(D), "poses of", len(F["t"]), "frames", round(wall, 1), "s")
