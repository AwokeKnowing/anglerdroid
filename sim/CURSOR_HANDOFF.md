# Kevin Newton sim handoff (2026-09-09 PT)

Open this repo on **i777** at `/home/james/gits/awokeknowing/anglerdroid`.
Newton venv: `.venv-newton` (already installed: Newton 1.5.1 + examples + rerun-sdk + opencv-python-headless).
House STL original: `/home/james/Desktop/casitahouse-walls.stl`
Repaired STL: `sim/casitahouse-walls-fixed.stl`
Entry: `sim/newton_kevin.py` (+ `--drive`) and `sim/newton_drive.py`.

## Goal (James)

Wander successfully with **good coverage of all rooms**, no stuck states.
Great SLAM is optional if wandering + a heuristic (frontier / least-visited / open-direction) works.
Walls must stop the nose before it crosses. Drive on the real Jetson stays **disarmed** until James says otherwise.

## Robot geometry (authoritative)

Origin = axle midpoint between hub motors. +X forward, +Y left, +Z up.
- Body 33×33×15 cm. Rear face x=−0.18 m, front x=+0.15 m.
- Caster hangs 8 cm behind rear face (x=−0.26 m).
- Mast to 90 cm, then forward 18 cm so cameras sit above the axle.
- Hub motors ~18 cm diameter, 6 cm wide; wheelbase ~34 cm.

## What works

1. **Wall collision for depth**: full 3k-tri STL as MuJoCo collider stalls. House is **86 thin boxes** (6 cm) from the repaired STL (`house_mesh.py`). Visual STL stays a site. Depth hits the boxes.
2. **Head-on wall stop**: RS1 top-down sees a wall slab; nose stopped ~0.33 m, `nose_crossed_wall=False`. Evidence: `kevin_headon_stopped.png`, `kevin_rs1_wall.png`.
3. **Pose fusion**: live `PoseEstimator` (IMU + wheel + visual). Chassis truth does **not** feed SLAM. Visual miss/over-match → IMU+wheel. Slip: `ANGULAR_SLIP_SCALE=0.92`, depth noise ~2% z + 3 mm.
4. **Open-floor slide (partial)**: when wheels commanded but body stalls on open floor (front >1.5 m), a 0.16 m house-xy free-joint slide syncs `joint_q`/`joint_qd`/MuJoCo `qpos` and clears warm-start. First slide held; later pin near (−1.35, −3.9) still snaps west inside the solver (~50 snaps). He wandered SW room only (~9.3 m path), never a second room.

## Current blocker

Leave the first room. Heuristic aims east then doorway, but a later free-joint pin snaps pose west after slides. Do not weaken wall stop while fixing the pin. Prefer fixing what SolverMuJoCo reads on the next step (`joint_q`/`qpos`/`qacc_warmstart`) over more slides.

## How to run

```bash
cd /home/james/gits/awokeknowing/anglerdroid
.venv-newton/bin/python sim/newton_kevin.py --drive --viewer gl --device cuda:0
# or the long wander helpers:
.venv-newton/bin/python sim/kevin_wander_220.py
```

DISPLAY=:1, GPU RTX 3090. machineId for tools: `ea5c5a69-7b00-4c3c-8c20-d4faf78f079e`.

## Live robot (Jetson Kevin)

- Back online as of 2026-09-08 evening; hostname/user `jetbot`.
- Drive **DISARMED**. Do not re-arm without James.
- Smoke had ego honesty zeros; keep `KEVIN_MODERNGL_SCATTER` off (invents CLEAR vs CPU).
- Face gallery / `~/.kevin` stay on device, not in git.

## Do not

- Raw STL as MuJoCo collider
- Feed chassis truth into SLAM
- Land ModernGL scatter until honesty matches CPU
- Re-arm the live robot
- Commit `.venv-newton/`, `sim/runs/`, or the 200MB `kevin_newton_drive.gif`

## Key files

- `sim/newton_drive.py` — sensors, commander, wall boxes, pose slide, gif compose
- `sim/newton_kevin.py` — chassis build, GL/Rerun entry
- `sim/house_mesh.py` — STL → wall boxes
- `sim/visit_wander.py` — coverage heuristic
- `src/pose.py`, `src/odom_thread.py`, `src/slam.py`, `src/perception/` — live stack the sim calls

## Latest stats snapshot

See `sim/kevin_newton_drive_stats.txt` from the last wander (SW room only; snap-back still present).
