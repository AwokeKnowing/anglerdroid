# Kevin full stack: emulator, SLAM, goals, safety (progress notes, 2026-10-13)

Status: **work in progress**, committed on branch `fullstack-sim-slam-2026-10-14` (2026-10-14). Timings are measured on i777 (RTX 3090) against the emulator, not on the Orin.

## Layout
- `hardware/tires/v3_4/`: tire v3.4 (STL, generators, renders, README, deps/).
- `emu/`: Jetson emulator.
  - `world.py`: MuJoCo twin plus 8 environments (`envs.py`), rendered with EGL. Sensor data goes out over shared memory (seqlock), CANSimple runs over UDP, and ground truth and scenario control (teleport, obstacles, blind) are on UDP port 47801.
  - `eval.py`: goal tours. `safety_eval.py`: adversarial safety tests (fresh nav per test). `gt.py`: ground-truth client.
  - `autocal_eval.py` + `autocal_probe.py`: forward-camera auto-cal convergence across randomized mounts (`KEVIN_EMU_RS2_RAND=1` or `KEVIN_EMU_RS2_PITCH/ROLL/DZ`). `popup_clip.py`: one pop-up-box stop as a clip.
  - `viewer/`: `viewer.py` (3-D web viewer + chat box, :8095) and `live.sh` (live casita sim + nav + viewer; `~/.kevin/emu/live.sh` is a thin wrapper to it).
  - `tools/`: `clip_rec.py` (clip frames), `tour_clip.sh OUT envs…` (tour + recorder), `run_table.py DIR[:label]…` (drift/goal table from run.json files), `fit_odom.py` (fit odometry motion-model alphas -> `src/odom_noise.json`), `diskguard.py` (rotate our own regenerable data), `rehearse_bringup.py` (bringup_run.sh against the emulator), `ab_queue_example.sh` (nohup A/B queue ending with live.sh).
- `scripts/bench/`: offline SLAM benchmarks (`bench.py` odom/slam2/KISS on recorded scans, `run_cuvslam.py`, `run_orb.py`, `gen_stereo2.py`/`rgbd_rec.py` log generators, `slam_replay.py`).
- `scripts/bringup_check.py` (read-only pre-flight), `scripts/bringup_run.sh` (capped real run + replay log), `docs/bringup.md` (checklist).
- Data never lives in the repo: runs, logs, recordings, clips and tables go to `~/.kevin/emu/` (i777) and `~/kevin_logs/` (robot); see `.gitignore`.
- `hal/sim/`: drop-in `pyrealsense2`, `can`, `inputs` and `ip`. `scripts/kevin-hal` picks real or sim with one switch (`KEVIN_HAL`).
- `nav/`: the runtime. The same code runs on the Orin and in sim.
  - `hw.py`: the only module that touches drivers. It uses the existing `src/` RSCamera, IMUPipeline and WheelBase.
  - `perceive.py`: RS1 ego labels (the robot's own `label_rs1_ego`), RS2 depth to obstacle and floor points, and the self-view mask.
  - `slam.py`: wheel + gyro odometry (zero-velocity gyro-bias update), a 2 cm log-odds map, and correlative scan matching (bilinear likelihood field, odometry prior) in its own thread at about 5 Hz.
  - `plan.py`: kplan (vendored in `nav/kplan`). A 12 m window of the map at 1 cm, refreshed in place so the CUDA graph stays valid. Global path from fastpath A*, MPPI via CUDA-graph replay.
  - `safety.py`: velocity-dependent collision envelope (details below).
  - `goals.py`: typed command language, places, and map save/load (`~/.kevin/maps`).
  - `server.py`: web UI and `/api/command` (the hook for conversational AI). `cli.py`: command-line client.
  - `vln.py`: LightNav-0 hook (`ask <instruction>`).
  - `run.py`: the 30 Hz loop (hard caps `KEVIN_VMAX`/`KEVIN_WMAX`; `KEVIN_RECORD=<dir>` replay log; `KEVIN_AUTOCAL`).
  - `slam2.py`: default SLAM (GPU correlative matching, chi-square match test with motion-model covariance from `src/odom_noise.json`; `KEVIN_SLAM2_TEST=gate` = old check).
  - `autocal.py`: forward D435i pitch/roll/height from gravity + floor RANSAC while parked, 5 deg / 3 cm sanity limits, writes `src/calibration.json` (sim: `~/.kevin/emu/calib_sim.json`).
  - `record.py` / `replay.py`: per-tick log and offline replay through odometry + slam2.
  - `selfmask.py`: calibrates the self-view mask.

```
RS1 frame (30 Hz clock) -> odom(wheels+gyro) -> ego labels -> MPPI (CUDA graph) -> safety envelope -> WheelBase
          | 5 Hz scans (RS2 + RS1)                     ^ path / window (0.8 s, background)
          v                                            |
   SLAM thread: CSM + 2 cm log-odds map ------> planner thread (map window, A*, MapCPU)
web / CLI / VLN -> goals.parse -> Brain.command   (go to <place|x y>, forward, turn, label, save/load, drive, ask, stop)
```

## Run
```
MUJOCO_GL=egl ~/.venv-kevinsim/bin/python -m emu.world --env apartment        # emulator (omit on the robot)
KEVIN_HAL=sim scripts/kevin-hal ~/.venv-kevinrobot/bin/python -m nav.run --port 8080
~/.venv-kevinrobot/bin/python -m nav.cli "label sofa"   |  "go to sofa" | "forward 1" | "turn left 90" | "stop"
browser: http://<host>:8080   (click = go there, shift-click = label)
~/.venv-kevinsim/bin/python -m emu.eval --envs room apartment ...     # goal tours  -> ~/.kevin/emu/eval
~/.venv-kevinsim/bin/python -m emu.safety_eval --env room              # safety     -> ~/.kevin/emu/safety
```
- Robot venv: `~/.venv-kevinrobot` (angler-brain site-packages plus moderngl, cuvslam, ninja).
- Sim venv: `~/.venv-kevinsim`.

## Measured (i777, emulator)
### Loop timing (apartment tour)
| Metric | Value |
|---|---|
| Control period | 33.3 ms mean (p99 40 ms, max 56) |
| Loop compute | 6.8 ms mean (p99 11) |
| Frame stamp to wheel command | 33 ms mean (p99 44) |
| MPPI graph replay | 1.0 ms (p99 1.7) |
| New goal (A* + upload) | 1–3 ms |
| Map-window refresh (background) | 15 ms |
| Scan match (background) | 9–16 ms |

### Unmodified robot code under emulation
- `src/main.py` ran at 30.0 fps with about 9 ms processing.

### Goal tours
- **Apartment** (5 goals by label, 20.9 m, before the SLAM fix): 4 of 5 succeeded, 0 collisions. Live SLAM error: mean 6.4 cm, max 12 cm.
- **SLAM offline replay** of the same log, after the fix (2 cm map, bilinear CSM with odometry prior, RS2 points within 3 m, 40-scan warm-up):

  | | Mean error | Max error | Max yaw error |
  |---|---|---|---|
  | SLAM | 2.7 cm | 6.9 cm | 3.5° |
  | Odometry only | 5.3 cm | 14.6 cm | 4.6° |

  The old 5 cm matcher was worse than odometry. It drifted toward the corner of the search window on score ties.
- **Multi-environment tour** (live, after the SLAM fix; logs in `~/.kevin/emu/eval/<env>_s0/`, summary in `~/.kevin/emu/eval_all.log`). Goal tracking is **weak**:

  | Env | Success | Physics contacts | SLAM error mean / max | Odometry error, final |
  |---|---|---|---|---|
  | room | 2/4 | 0 | 3.5 / 8 cm | 4.6 cm (22 cm peak) |
  | apartment | 0/4 | 380 | 11 / 33 cm | 46 cm |
  | corridor | 2/4 | 347 | 0.5 / 1.76 m | 5.2 m (wheels slipping while pinned) |
  | office | 1/4 | 0 | 1.8 / 4 cm | 4 cm |

  Failure causes, in order:
  1. Goals more than 6 m away fall outside the 12 m planning window. The window edge is treated as a wall, so they are reported as "inside obstacle". Fix: a coarse global A* on the full map, or a larger window.
  2. "Arrived" is declared on the SLAM pose while the robot is 0.4–0.9 m off, because the goal was snapped to a reachable point near the place.
  3. Pinned-against-wall episodes cause hundreds of contact events. **In planner mode the envelope does not prevent these.** Likely cause: side and rear contacts outside the RS1/RS2 view while the robot rotates. The envelope only sees forward depth and the RS1 ego window.

  Doors and other environments were not finished at the time of writing.

### Safety envelope
`emu.safety_eval`, room, straight `drive 0.6`, no planner:
- **Normal driving**, parallel to a wall at 8, 12 and 15 cm side clearance: 0 interventions, 0 collisions.
- **Box pops up D m ahead at about 0.57 m/s:** no collision for D = 1.2, 0.8, 0.5, 0.35 or 0.25 m. The robot stops with a 0–4 mm gap. At D = 0.15 m it collides; that is below the physical stopping distance (about 0.18 m).
- **Crossing box** (lead 0.9, 0.6 or 0.4 m): 0 collisions.
- **Blind tests:**
  - RS1 + RS2 dead: standstill in 0.44 s, 0.15 m travelled.
  - RS1 alone dead: same as above.
  - RS2 alone dead: no stop, by design. RS1 still gives valid depth. Policy question for James.
- **Envelope design:** the commanded (v, w) is swept along the footprint (SIL12 + 1.5 cm) for its stopping distance. That distance uses the larger of commanded and measured speed, a = 1.5 m/s² and 0.10 s reaction time. The command is scaled down only if the sweep hits fresh depth. If depth is stale (>0.25 s) or less than 5 % of the ego window is valid, the robot is immobilized.
- **Interventions during planner tours are still high** (about 40 per goal). The MPPI uses a 3 cm clearance and its own optimistic a_brake of 1.96. Fix still to do: make the MPPI v_cap consistent with the envelope (a = 1.5 m/s²).

### LightNav-0 (Qwen3-VL-4B, Apache-2.0)
- Serving on i777: vLLM at 40 % GPU, port 8050.
- Latency: 887 ms on warm-up, then **150–190 ms per step** when warm (batch 1).
- Output: 10 SE(2) waypoints, about 1.5 m ahead.
- **Fit:** use it as a high-level goal proposer at 1–5 Hz, never in the 30 Hz loop. `nav/vln.py` turns its last waypoint into a map goal, and MPPI plus safety stay in charge.
- **Not done:** a semantic success evaluation on emulator camera frames. The latency probe used a placeholder image. It will not fit the Orin alongside the stack without quantization; that is unverified.

## Findings for the real robot
1. **Encoder staleness bug (likely real):** `odrivecan.poll_encoder_vels` read the head of a queue flooded with about 1220 cyclic frames per second, so readings were 1–5 s stale. Fixed so it drains the queue and keeps the newest value. Backup is in `~/.kevin/backups/fullstack/`. Verify on hardware with candump.
2. **RS1 x-offset:** `TD_X_OFFSET = -75` puts RS1 4 cm ahead of the axle; the twin URDF puts it 0.7 cm behind. In sim, ego obstacles appear 4.0 cm too far, measured against a box at 0.30, 0.45 and 0.60 m. One of the two is wrong. Measure on the robot.
3. **Wheel geometry:** `robot_config` has wheel radius 0.085 and track 0.354; the twin has 0.08565 and 0.362.
4. **Self-view:** RS1 sees the robot's own front corners and wheels as obstacles. `nav/selfmask.py` calibrates a mask (run it once on the real robot). Pixels inside the footprint are always dropped.
5. **Drive response:** the emulated ODrive velocity loop (twin gains) takes about 0.6 s to start and overshoots by 20 % (0.6 → 0.74 m/s) with an undershoot on stop. If the real robot does the same, retune `vel_gain` / integrator.
6. **`cameras.find_rgb_device`:** now honors `KEVIN_RGB_DEV` (sim sets it to `none`). Backup includes the prior uncommitted changes.

## What transfers to the Orin
- Unchanged: all of `nav/`, plus `hal` set to real (no sim modules on the path).
- The kplan CUDA extension and libfastpath rebuild on aarch64 automatically (gcc `-mcpu=native`, torch `load_inline`). Needs ninja.
- Expect MPPI to be about 3–5× slower on the Orin (still under 10 ms). Re-measure.
- Calibrate the self mask. Resolve the TD_X_OFFSET question first.

## Gaps / next steps
- Multi-environment tour results (running).
- cuVSLAM as an alternative odometry source (installed, not yet wired or evaluated).
- MPPI/envelope speed-rule consistency.
- Map relocalization on load (currently assumes the saved pose).
- LightNav semantic evaluation on emulator RGB.
- Goal 7 (RL): not started. Tracking is weak, but the failures above are planner-window and blind-side issues, not something RL fixes. Fix those first, then re-measure.

## Twin = source of truth (James 6:13), 2026-10-13
Single source: `scripts/gen_twin_geometry.py` parses the twin URDF per `KEVIN_TIRE` into `src/twin_geometry.json`; `src/twin_geometry.py` loads it and applies the calibration hook `src/calibration.json` (or `KEVIN_CALIB`): wheel_r_scale, track_scale, per-frame dxyz/drpy_deg, IMU gyro_bias/scale (reality→model; currently identity).
Changed values (backup `~/.kevin/backups/fullstack/twin_1814/`):

| file | constant | old | new |
|---|---|---|---|
| robot_config | WHEEL_DIAMETER_CM | 17.0 | 17.13 |
| robot_config | WHEEL_HALFTRACK_CM | 17.7 | 18.1 |
| robot_config | WHEELBASE_M (track) | 0.354 | 0.362 |
| robot_config | WHEEL_RADIUS_M | 0.085 | 0.08565 |
| robot_config | BODY_FRONT/AFT/CASTER_AFT/WID cm | 18/18/20/33 | 12.6/23.1/27.4/30.8 |
| robot_config | MAST_CX_CM | -13 | -15.75 |
| vision | TD_X_OFFSET | -75 | -79 (RS1 0.7 cm behind axle) |
| vision | FW_TD_X_DELTA | 132 | 136 |
| vision | TD_FLOOR_CLIP | 0.91+dz | rs1 height-0.041 = 0.9066 |
| cuvslam_tracker | CAMERA_HEIGHT_M / SETBACK / PITCH | 0.97 / 0.13 / 64.4 | 0.4679 / 0.130066 / 63.0 |
| nav/kplan/core.py | track / a_brake | 0.303 / 1.96 | 0.362 / 1.5 |
| nav/hw.py, slam.py, safety.py | RS2 pose, IMU rotation, track, SIL12 | literals | from twin_geometry |

Check (room, 19.2 m): odom drift 0.22 % (yaw sign of the IMU-rotated gyro verified), SLAM 0.31 %, 0 collisions, 2/4 goals.
Not yet derived from twin: perception fast_ego80/ego_rs1/gpu_render literals, pose.py, emu/world.py mounts (hard-coded from same URDF).
Domain randomization plan: per episode sample wheel_r ±1 %, track ±1 %, camera extrinsics ±5 mm/±1°, gyro bias ±0.01 rad/s and scale ±2 %, depth subpix 0.05–0.2, holes 0–3 %, min-Z 0.15–0.3, ODrive delay/overshoot; same keys as calibration.json.

## RealSense depth model (James 6:15)
Sources: Intel RealSense D400 Series Datasheet (doc 337029): D435/D435i baseline 50 mm, depth FOV 87°×58° (±3°), RGB FOV 69°×42°, min-Z ~0.28 m at 1280×720 (scales with resolution, ~0.175 m at 848×480); depth origin = left IR imager; RGB ~15 mm from left imager. Intel whitepaper "Tuning depth cameras for best performance": RMS = Z²·subpixel/(f·B), subpixel ≈ 0.08; invalid left band width = f·B/Z px.
Code (src/cameras.py): depth 848×480 z16 + color 640×480 rgb8 @30 Hz, IR1/IR2 y8 for VSLAM (RS2 emitter off). **No rs2::align** — the code consumes raw depth in the left-IR (depth) frame; only `decimation_filter` is used (no spatial/temporal/hole-filling). Sim therefore emits unaligned depth from the depth optical frame (twin), color from the twin color frame.
emu/world.py now: noise σ = Z²·subpix/(f·B) (f = 433 px at 848×480), left invalid band per pixel u < f·B/Z, min-Z 0.175, max 8 m, occlusion-edge shadow, random dropouts. Flags: `--subpix 0.08 --holes 0.004 --min-z 0.175 --no-band` (domain randomization). At 2 m σ ≈ 1.5 cm (old model 0.6 cm).
Not modeled: dark/shiny-surface holes by material, projector pattern, IR emitter interaction. No real captured frame exists (searched ~/.kevin and repo; bags deleted on the Orin), so no real-vs-sim comparison yet.

## 2026-10-13 evening: camera-mapping audit, cuVSLAM rig, slam2

### Camera mapping (audited against twin URDF `urdf/kevin.urdf`, serials, code)
| consumer | physical camera | twin frame | pose old -> new |
|---|---|---|---|
| RS1 everywhere (vision TD_*, nav ego, safety, emu rs1_*) | D435 sn 815412070676, mast top, lenses down | realsense_d435_{depth,infra1,infra2,color}_optical | (-0.007, 0, 0.862) base / 0.948 m above floor, looking straight down; IR2 at x=-0.057; color x=+0.008 (unchanged) |
| RS2 everywhere (vision FW_*, nav Rs2Cloud/SLAM match, emu rs2_*) | D435i sn 944622074292, low forward, 27° down | realsense_d435i_*_optical | (-0.130, 0, 0.382) base / 0.468 m, pitch 27° down; IR2 y=-0.05 |
| IMU (imu.py, nav hw.yaw_rate, odom_thread) | D435i BMI | realsense_d435i_imu_optical | odom_thread pitch -64.4 -> -63.0, imu.py default 64.4 -> 63.0 (twin) |
| cuvslam_tracker | was RS2 IR pair only, with 0.97 m (~RS1 height) + 0.13 m/64.4° (RS2) mixed, and the quaternion pitched the camera UP | now rig of 4 cameras: rs1_ir1, rs1_ir2, rs2_ir1, rs2_ir2 from twin, + rs2_imu | see rig table below |
| serial->role (main.py, nav/hw.py) | chosen by sorted serial order (works only by luck: 8154.. < 9446..) | `twin_geometry.realsense_roles()`: D435I => RS2, D435 => RS1, env overrides | fixed |

Git history (all 70+ branches): cuvslam_tracker has only ever been a single-RealSense (RS2 IR1+IR2) rig since d3f0a73 (2026-09-05); no two-RealSense rig existed. Now `KEVIN_CUVSLAM_RIG=dual` (default, both RealSenses, RS1 IR enabled in vision with emitter ON for safety depth) | `rs2` (old single pair). NB: pycuvslam 16.0 on i777 rejects the existing `OdometryConfig(...)` kwargs (pre-existing API mismatch, both modes) — not fixed (cuVSLAM work stopped per James).
cuVSLAM rig (frame X right, Y down, Z fwd, origin floor under axle), verified == URDF chain to 1e-5:
| cam | t_rig (m) | view dir (rig) |
|---|---|---|
| rs1_ir1 | (0, -0.9476, -0.0070) | (0, 1, 0) down |
| rs1_ir2 | (0, -0.9476, -0.0570) | (0, 1, 0) |
| rs2_ir1 | (0, -0.4679, -0.1301) | (0, 0.454, 0.891) |
| rs2_ir2 | (0.05, -0.4679, -0.1301) | (0, 0.454, 0.891) |
| rs2_imu | (0.0055, -0.4777, -0.1382) | — |
Emulator now renders RS1 IR1/IR2 too (both devices stream ir1/ir2).

### SLAM history: why "morph matching" nearly worked
src/gpu_render odom_gpu (SAD over morph-closed BEV images) + src/slam.py thumbnail IoU: dense, morphologically closed obstacle images give a wide smooth score basin; failed on frame-to-frame drift (no map), yaw-only, featureless floors.
Old nav/slam.py diagnosis: 668/817 matches rejected as "no gain" (likelihood field saturated: score ~0.98 at the odom pose), 42 "big" corrections rejected => yaw drift from long maneuvers was locked out.

### nav/slam2.py (new, default `KEVIN_SLAM=2`; `KEVIN_SLAM=1` = old)
Keyframe hit-count grid 4 cm -> morph close -> distance transform -> exp field (12 cm coarse / 4 cm fine); brute-force torch GPU search (±24 cm, ±4°, then fine); degeneracy gate (score curvature eigen-directions); keyframes 20 cm/8°; loop closure (old KF within 2 m, >4 m path ago, wide search ±0.6 m/±10°, score>0.65, plausibility vs 3 % drift); Gauss-Newton pose graph (scipy sparse); planning GridMap rebuilt in place. Key bug found: half-cell grid_sample offset accumulated 14 cm drift; fixed (cell centres).
Offline replay on recorded emulator runs (`~/.kevin/emu/replay2.py`, recordings `~/.kevin/emu/rec/*.pkl`):
| env | dist m | odom max/final | old slam live mean/max | slam2 mean/max | loops |
|---|---|---|---|---|---|
| room | 27 | 0.145/0.022 | 0.050/0.097 | 0.028/0.069 | 7 |
| apartment | 42 | 1.21/1.20 | 0.119/0.279 | 0.039/0.132 | 6 |
| corridor | 57 | 2.15/2.13 (197 wall contacts -> wheel slip) | 0.526/1.75 | ~0.46/0.97 | 12-14 |
| clutter | 8 | 0.227/0.227 | 0.059/0.176 | 0.167/0.427 (worse) | 0 |
| office | 66 | 0.85/0.40 | 0.393/0.664 | 0.207/0.50 | 16 |
Runtime i777 (RTX 3090): match+insert 20-30 ms mean per scan (5 Hz, async), loop search ~30 ms, optimize 18-65 ms, rebuild p99 ~0.3-0.5 s (async). Orin NX estimate ~4-6x slower GPU -> ~100 ms/scan: run at 2-5 Hz or halve NPTS/search window.
Open: corridor degeneracy + odom slip under contact, clutter regression, live tours with slam2 not yet complete.

## 2026-10-13 night: contacts, self-view phantoms, Orin timing (progress log)
- **Contacts → 0** (s2eval_b, 6 envs partial): poster removed from the corridor doorway (it hung at mast height); ego body mask no longer hides jambs; safety + kplan use `FOOTPRINT_SAFE` (the sim deck corners stick out ~3 cm past SIL12); safety margin 0.035 + penetration-sum rule; |w|≤0.8 when clearance <0.10. Corridor repro 64 → 0 contacts.
- **But goals collapsed** in s2eval_b (apartment 1/6, corridor 1/6, office 1/4, room 2/4; clutter 2/2, maze 2/2): "stuck" / "local planner found no path". Root cause of the recurring **clearance 0.020**: RS1 ego self-view. With the body mask cut to the bare SIL12, the wheels (y ±0.207), gate caster and deck rear corners, plus flickering speckle at the robot silhouette, came through as obstacles at sd 0–0.02. That is permanently inside the 0.035 margin, so safety stays 'limited' in open space. The sim self-mask (`nav/data/self_mask_sim.npy`, 17:23) was stale.
- **Fix (nav/perceive.py):**
  - Body mask = FOOTPRINT_SAFE + 1 cm, plus wheel bands (|x|<0.095, 0.15<|y|<0.245) and a rear caster/deck box (−0.30<x<−0.185, |y|<0.175).
  - Within 5 cm outside the hull, an obstacle pixel is kept only if it is part of a blob (≥3/8 neighbors) and was an obstacle last frame too.
  - The sim self-mask was recalibrated (30 px).
  - Open-space min ego sd is now ≥0.05 (it was −0.009 to 0.02).
  - Trade-off: on the real robot, an obstacle in the wheel bands within ~4 cm of a tire is invisible to RS1. RS2 and the map memory still cover the front.
- **Orin (jetbot, JetPack R39, 40 W mode), slam2 kernels:**
  - Synthetic benchmark with 2000 pts: the per-scan track search took 23.6 ms, and the loop/wide search took 43.3 ms.
  - Environment: torch 2.14.1+cu130 in `~/.venv-kplan`, with cv2/scipy in `~/kevin-debug/s2bench/deps`. PyTorch warns it has no native build for this GPU; the kernels ran via JIT.
  - Comparison: i777 measured ~8–30 ms per scan. The Orin's async match costs ≈1 scan period, which fits.
  - The benchmark used the box copy of slam2.py (19:02). The search code is the same as on i777.
  - Not yet measured on the Orin: optimize() (the benchmark's edge format was wrong) and the full pipeline on real frames.
- **Planner fixes (20:30 PT):**
  - `nav/globalplan.py`:
    - Inflation fallbacks went from 0.22/0.17/0.12 to 0.24/0.21. The robot half-width is 0.207, and 0.12 planned through 0.3 m gaps, so MPPI could never follow the plan.
    - A goal in a pocket the robot can't reach now snaps to the closest *reachable* cell; before, it failed and the robot kept chasing the old goal.
  - `nav/run.py`: after 2 no-progress events, the global replan uses 0.30 inflation and logs "goal pocket too tight: aiming for closest reachable point". The state now exposes `safety.last` (for the clip overlays).
  - Room result: 0/1 (s2eval_c) → 2/4 with 3 contacts (s2eval_d, after the perception fix) → 3/4 with 0 contacts (s2eval_e, after the inflation fix). Door was fixed. Couch was still stuck: its goal sits in a 0.5 m gap between the table and the couch, and the evaluator's closest reachable point is 0.71 m away.
- **Clips:** `~/.kevin/emu/clip_rec.py` (i777) polls nav and sim cameras at 5 Hz. Each frame is split into three panels: RS2 color, RS1 top-down labels, and the slam2 map with pose, MPPI path, goal, and the footprint+margin envelope (colored by safety state). `~/.kevin/emu/tour_clip.sh OUT envs…` runs an eval and the recorder together. The box's `/workspace/clips/encode.sh` encodes the frames to H.264 720p.
- **Recovery fix (20:45 PT, nav/run.py):**
  - After reversing out, if the global path leaves more than 60° off the current heading, the robot first turns to face it, using turn mode with the R275 check and a 6 s timeout, then resumes the path.
  - Why: in s2eval_f, MPPI dithered for a long time (v ~0.03) while facing 180° away from "center", and safety was 'ok'.
  - Room result: 1/4 in s2eval_f, before the fix; 3/4 with 0 contacts in s2eval_g, after it. Couch remains an honest fail: the robot gets within 0.13 m of the evaluator's reachable-distance radius, but its stop rule only trusts dg<0.35 to the goal.
  - Run-to-run variance is large: the same code gave room 3/4 (e) and 1/4 (f).
- **Full 8-env tours, slam2 (s2eval_g, 20:42–21:04 PT, 9-goal tours, all fixes above):**

  | Env | Goals reached | Contacts | Notes |
  |---|---|---|---|
  | room | 3/4 | 0 | couch is in the table/couch gap |
  | corridor | 5/6 | 0 | |
  | clutter | 2/2 | 0 | |
  | office | 1/4 | **367** | see below |
  | maze | 1/3 | 3 | slam2 lost: 2.2–3.0 m error |
  | doors | 2/6 | 0 | 0.95 door "arrived" with 0.39 m SLAM error, so not ok |
  | casita | 1/2 | 0 | |
  | apartment | aborted | — | nav died at 10 s with SIGABRT (see crash note) |

  - Total: **15/27 (56%)**, well short of the ≥90% target.
  - Office contacts: the robot drove under a desk while slam2 was 0.43 m off. Neither camera can see the desktop (~0.75 m): it is closer to RS1 than the D435's min-z, so it reads as no-depth, and it is above RS2's FOV. Only the table legs reach the map. **Overhang gap:** treat RS1 no-depth blobs (too close to the camera) as obstacles. This matters for the real robot too.
- **Forward cone (James):** RS2 points *do* reach slam2 and the map. `run.py` concatenates them into each submitted scan, and slam2 matches on RS2. The clips simply didn't draw them. The bottom-left panel now shows the robot-frame top-down view: RS1 ego in grey/red, RS2 floor in green, RS2 obstacles in orange, and the FOV cone. The map panel draws the cone as well.
- **Crash dialog (James):**
  - What crashed: `/var/crash/_usr_bin_python3.12.1000.crash` at 20:45:45 PT was `nav.run`, with SIGABRT ("terminate called without an active exception").
  - Cause: on SIGTERM, `sys.exit` runs interpreter finalization while native threads (WheelBase/HAL, torch) are still joinable, which triggers std::terminate.
  - Fix: `main()` now calls `os._exit(0)` after `Brain.run`'s finally (wheels zeroed, CAN down).
  - MuJoCo already renders headless (`MUJOCO_GL=egl`).

## 2026-10-13 21:10 PT: overhangs, slam2 gating
- **Emulator bug (overhangs):** MuJoCo's near clip defaults to 0.01×model extent, which is 0.20 m in the office. Desk tops 0.18–0.20 m below the RS1 lens were clipped, so the camera saw *through* them to the floor. That is how the mast hit the desk (367 contacts). `emu/world.py` now fixes `visual/map znear` at about 1 cm, and `--min-z 0.175` produces the real D435 invalid band.
- **Too-close rule (nav/perceive.py `Ego._too_close`):**
  - Per-pixel rays are learned from valid returns.
  - A no-depth blob of at least 150 px is re-inserted as points at 0.17 m (an overhang at about 0.74 m height) when it is:
    - outside the left stereo-occlusion band (8% of columns), and
    - present in 3 consecutive frames.
  - Floor holes and speckle don't pass the size and persistence tests. Office probe: open floor and a 30 cm clearance gave 0 px; at and under the desk edge it gave 1920–2240 px.
- **Overhang edges:**
  - Obstacle pixels at least 0.55 m high are kept inside the body mask (except the mast/e-stop column) and in the near ring. The RS1 sees a desk edge only as a thin line, which the ring filter used to drop.
  - Remaining gap: a desk edge becomes visible only about 2 cm before the footprint reaches it (sd 0.004 at 0.15 m). RS1's FOV at desk height is about ±0.2 m, and RS2 looks down. It is mapped once seen.
- **slam2 odometry gate (maze/doors aliasing):**
  - A match may move the pose by at most 0.06 + 0.08·d m and 2.5° + 0.05·d, where d = odometry travelled since the last accepted match. Otherwise it is rejected (`stats.gated`).
  - The wide re-acquire radius grows with d (0.25 → 1.0 m) instead of a fixed ±1 m with a near-flat prior.
  - Replays (mean / max m): clutter 0.052/0.133, apartment 0.06/0.14, office 0.105/0.20, corridor 0.47/1.71 (was 0.39/1.45; the wheel-slip episode).
- **No-depth = obstacle in the safety path (James, 21:12 PT):**
  - `Ego._too_close` runs on every RS1 frame, before labelling, so safety sees it directly; it does not depend on map memory.
  - Rays come from RS1 intrinsics, so no valid return is needed first, e.g. a hand over the lens at boot. Measured returns override them.
  - Legitimately invalid pixels are excluded: the decimation border (4 px) and the left stereo-occlusion band (8%). The self-mask/body is masked downstream.
  - A persistent no-depth blob (≥150 px, 3 frames) is inserted as its floor "shadow" (rays at 0.80 m, so OBSTACLE). Whatever blocks those rays is somewhere along them, so the cells under it are not provably free.
  - If more than 50% of the non-legit pixels are no-depth, `occluded` is set and `valid_fraction()` returns 0, so safety goes `no_depth` and the robot is immobilized.
  - Offline test on recorded office frames (`/tmp/tc_off.py`):

    | Case | Result |
    |---|---|
    | open floor | 0 px |
    | 5% random holes | 0 |
    | 1-frame blob | 0 |
    | hand over whole lens | occluded → immobilized |
    | hand over 57% | occluded → immobilized |
    | box close (40×50 px) | 2000 px → 600 obstacle points beside the robot |
    | desk edge | 2066 px → obstacles at sd 0.024 (inside the 0.035 margin → blocked) |

  - Known limitation: a large black or IR-absorbing floor patch at least 150 px, persistent, would read as an obstacle (fail-safe). A per-site floor-hole mask would be the fix if it shows up on the real robot.
- **Doors regression (s2eval_h doors 1/6):** the overhang keep-rule (h ≥ 0.55 m inside the body) also kept door-jamb pixels in the +1 cm band, so the robot froze in the 0.55 m door. The rule is now strictly inside the hull (−1 cm).
- **Maze root cause:** odometry lost about 0.5 rad on the first spin, while gt yaw went 0 → 0.5 rad in 0.25 s. The yaw error stayed at 39° and slam2 built a rotated map (it followed odometry exactly). `hw.yaw_rate()` now uses the mean of *all* 200 Hz gyro samples since the last call (it used only the latest one). `imu.grab` drains 256 frames (was 32). Odometry dt clamp went from 0.2 to 0.6 s. Likely cause: a loop stall during the first spin. Not yet proven.

## Emulator sensor-realism audit (James, 21:44 PT)
What the emulator modeled **before** this change: ideal unless noted.

| Item | Before (honest) | Now: `KEVIN_EMU_NOISE=realistic` | `pessimistic` |
|---|---|---|---|
| Wheel encoder | vel + N(0, 0.002 turn/s); pos exact float, no quantization | per-wheel radius error N(0, 0.5%); pos integrated from reported vel, quantized 1/16384 turn | radius error 1.5% |
| Track / pivot skid | physical only (MuJoCo tire–floor friction 0.9, caster 0.35–0.7); robot's track = sim track exactly | + effective-track error N(0, 1.5%) on the differential | 4% |
| Contact / carpet slip | physical contacts only | tire μ ×0.9 | tire μ ×0.6 (real skid in pivots and contacts) |
| CAN latency / jitter | none (10 ms cyclic UDP, arrival = stamp) | **not modeled yet** (robot stamps on arrival) | not modeled |
| Gyro white noise | N(0, 0.003 rad/s) per 200 Hz sample = 0.017 °/s/√Hz (BMI055 datasheet: 0.014) | same | 2× |
| Gyro bias | constant N(0, 0.002 rad/s) per axis | N(0, 0.2 °/s), the residual after ZUPT (BMI055 zero-rate offset is up to ±1 °/s) | N(0, 1 °/s) |
| Gyro bias random walk | none | 3e-5 rad/s/√s | 1.5e-4 |
| Gyro scale error | none | N(0, 1%) | N(0, 3%) |
| Thermal drift | none | 0 | 2e-6 rad/s² (ramp) |
| Accel noise | N(0, 0.03 m/s²) (not used by nav) | same | 2× |
| IMU–camera time offset | none | +3 ms | +10 ms |
| Camera timestamp jitter | none | N(0, 1 ms) | N(0, 3 ms) |
| Depth noise | stereo Z²·subpix/(f·B), holes, left invalid band, min-Z (and the near-clip bug fixed tonight) | same | same |

- Per-episode draws are seeded by `--seed`, and the world log prints them.
- References: BMI055 datasheet (Bosch BST-BMI055-DS000; gyro noise density 0.014 °/s/√Hz, zero-rate offset ±1 °/s, sensitivity tolerance ±1%). Diff-drive systematic errors: Borenstein & Feng, UMBmark (1995): wheel-diameter and wheelbase errors of about 0.5–2% dominate.
- No-IMU ablation: `KEVIN_NO_IMU=1` makes `hw.yaw_rate()` return None, so odometry uses wheel-differential yaw.

## 21:55 Uneven floors, thresholds, outdoor envs, negative obstacles, sunlight (in progress)
- `emu/world.py::_floor_model` (on for KEVIN_EMU_NOISE=realistic/pessimistic, KEVIN_EMU_FLOOR=1|2, and always for outdoor envs):
  - indoor: MuJoCo heightfield (2.5 cm grid): carpet lumps +-1.5/3 mm over 10-30 cm + sag/bow +-3/5 mm over ~1 m
    (apartment realistic: 6.6 mm total span); rounded threshold capsules 6-12 mm proud in every wall gap
    (`envs.B.wall` records `env.thresholds`). Single floor friction 0.9 / rolling 0.005 indoors — per-room floor
    types (carpet/wood/tile) NOT done yet.
  - outdoor (`env.terrain`, 1 cm grid): `sidewalk` (expansion joints 8 mm x 6 mm every 1.5 m, 1 % cross-fall,
    8 mm heaved-slab lip, 15 cm curb to asphalt road, 1.2 m curb-cut ramp 1:7, facade, posts, bin);
    `parking` (3 % slope, 7 potholes 2-4 cm deep / 20-40 cm, gravel patch, parked cars, poles); `lawn`
    (2-4 cm bumps over 0.25-0.8 m, 6 % slope, friction 0.55, rolling 0.03 condim 6, 2500 visual-only grass blades
    4-9 cm that occlude depth but not wheels, tree, bench, 4 cm bed edging).
  - curb drop counted as a collision (`env.hazard`, last_contact = drop-off) and rasterized into GT occupancy.
- `KEVIN_EMU_SUN=1`: projector washed out -> stereo subpixel error x2, blob dropouts (smooth-noise mask, ~25-45 %
  of low-texture area) + 3 % random holes.
- `nav/perceive.py`: negative obstacles + tilt leveling (`Ego._cliff`): per-frame robust floor-plane fit on RS1
  points near the nominal floor (0.946 m), cloud leveled by the fitted pitch/roll, points >= 5 cm below the plane
  for 2 frames in blobs >= 40 px re-inserted at 0.80 m along their rays -> OBSTACLE. Offline:

  | case | cliff px | obstacle pts | note |
  |---|---|---|---|
  | flat floor | 0 | 191 (baseline) | |
  | curb 15 cm drop | 14211 | 4566 | detected |
  | pothole 3 cm | 0 | 191 | drivable, as intended |
  | pitch 6 deg | 0 | 191 | was 4311 phantom pts before leveling |
  | roll 6 deg | 0 | 191 | was 2638 before leveling |
  | grass blades +-2 cm | 0 | 2092 | **open: grass reads as obstacle** |
- Runs queued on i777: s2eval_i (indoor) -> noise ablation (now includes the floor model for realistic/pessimistic)
  -> outdoor.sh (realistic; stock / soft_v3_2 / stock+sun; out_stock, out_soft, out_sun) -> thr_clip.sh
  (doors+apartment with thresholds, clips `~/.kevin/emu/clips/<env>_thresholds`).

## 22:05 Phantom/overhang/grass fixes, rugs, hfield orientation bug
- s2eval_i (all 21:10 fixes, ideal noise) final: apartment 2/9 (2 contacts), casita 1/2, clutter 1/2 (1), corridor 3/6,
  doors 1/6, maze 3/3 (14), office 1/4 (204), room 3/4 -> **15/36 (42 %)**, worse than s2eval_g. Clips:
  /workspace/clips/<env>_tour_s2eval_i.mp4.
- Apartment phantom root cause (live RS1 frames + offline): a permanent ~241 px no-depth blob at the right image edge
  (a twin part inside min-Z) passed the too-close rule every frame -> ~80 synthetic obstacle points 0.7 m behind the
  robot, and those were also submitted to slam2 -> remembered map phantoms around/behind the robot (map points
  feed the safety envelope) -> 'no progress: reversing out' loops; nav clearance median 0.17 m vs GT ~0.44 m.
  Fixes: (a) that blob is a legit-invalid region in sim (KEVIN_HAL=sim); (b) synthetic too-close / drop-off obstacles
  go to the safety layer only: Ego labels the raw cloud for the map (`Ego.map_obs`), run.py submits map_obs.
- Office 204 contacts root cause (offline MuJoCo at the logged pose): `chassis_mast` vs desk top edge (env102,
  z 0.71-0.75) with the desk top over the rear deck. RS1 sees the desk (z~0.2 m) but `label_rs1_ego` marks the
  robot's own boxes SELF ("SELF wins"), so the overhang was hidden. Fix: SELF or OBSTACLE >= 55 cm inside the hull
  (minus mast column) = overhang (open floor: 0 such px). Contact poses now sd -0.13 (85-130 px inside); open 0.37.
  Contacts counted 204 because the touch flickers in/out while turning (one touch event, many counts).
- Grass: low (< 12 cm) obstacle px with sparse 7x7 neighbourhood (< 55 %) dropped when they are >= 1.5 % of the
  known ground (grassy view). Offline lawn: obstacles 553 -> 66, sd 0.06 -> 0.386; bench (30 cm) and 4 cm edging
  kept; indoor frames / all safety tests unchanged (grass_px 0).
- BUG fixed: MuJoCo hfield row 0 is the +y edge -> terrains were mirrored in y (sidewalk road/curb on the wrong side).
  Fixed before the outdoor runs started (ablation 'ideal' has no hfield; realistic/pessimistic runs pick up the fix).
- Indoor floor types: hard floor mu 0.7 (realistic) / 0.5 (pessimistic polished tile), rolling 0.002; rugs/carpet
  (random ~45 % of places, 1.2-2 x 0.9-1.6 m, 6-10 mm edge, mu 1.0, rolling 0.01-0.03, condim 6).
- Offline outdoor perception checks (robot placed on terrain): curb edge cliff 12114 px; curb-cut ramp and 8 mm lip
  0 cliff px / no obstacles; parking flat/gravel clear; lawn edge of world reads as a drop (correct).
- Queue: noise ablation (running) -> outdoor (stock/soft/sun) -> threshold clips -> s2eval_j (all 8 indoor, fixes).
- Eval tables: `~/.kevin/emu/tables.py <dirs>` (odom-only vs slam2 end error and drift % of path, safety states).

## 22:45 Noise ablation results, reverse-driving fix
- Noise ablation (room/maze/apartment, --tour 4 --seed 3; realistic/pessimistic include the uneven floor, thresholds,
  rugs; started after the phantom fixes but BEFORE the overhang-SELF/grass/MPPI changes):

  | cfg | goals | contacts | odom-only end err / drift (room, maze, apt) | slam2 end err / drift |
  |---|---|---|---|---|
  | ideal | 9/11 | 0 | 0.10 m 0.4 %, 0.17 m 0.3 %, 0.25 m 0.8 % | 0.03 m 0.1 %, 0.04 m 0.1 %, 0.08 m 0.2 % |
  | realistic | 5/11 | 0 | 0.32 m 1.4 %, 0.26 m 0.5 %, 1.54 m 9.1 % | 0.05 m 0.2 %, 0.26 m 0.5 %, 1.44 m 8.4 % |
  | pessimistic | 1/11 | 16 | 6.47 m 34.6 %, 6.51 m 19.9 %, 2.00 m 3.8 % | 5.30 m 28.3 %, 5.02 m 15.3 %, 1.28 m 2.5 % |
  | realistic no-IMU | 3/11 | 1 | 0.19 m 1.0 %, 0.45 m 1.0 %, 2.17 m 9.6 % | 0.09 m 0.5 %, 0.11 m 0.2 %, 1.56 m 6.9 % |
  Honest reading: slam2 only beats odometry when it is already close; realistic apartment and all pessimistic runs
  lose the map (slam2 follows the drifting odometry: gate + wide search not enough). Apartment ideal improved 2/9 ->
  3/4 with the phantom fix.
- Reverse driving: tracks show the robot moving BACKWARD 46-75 % of moving time (GT displacement vs heading; room
  ideal 73 %). Offline closed-loop MPPI test (/tmp/revsim.py, kinematic, 5x4 room) reproduces it:
  old: start facing away 72 % reverse (fail), nose-in pocket 78 % (fail), goal behind 99 % (fail), side 0 % ok.
  Fix (nav/kplan): reverse cap v_min -0.3 -> -0.2 m/s; reverse cost x(0.5+4 near-goal + rev_w 4 x clearance factor)
  /0.2; heading cost head_w 1.5 x clearance x (1-cos(th - bearing to carrot)) beyond 0.35 m from it (CUDA kernel,
  runtime Limits.rev_w/head_w; Triton twin uses the same constants). Recovery reverse-out + turn-to-path unchanged.
  new: away 2.3 % reverse ok 8.1 s, pocket 4.4 % ok 8.2 s, side 0 % ok, goal behind 0.7 % (stops 0.16 m short).
- Outdoor bug: 1 cm outdoor heightfield (630k vertices) dropped the emulator cameras to 1.1-1.4 Hz -> out_stock
  sidewalk (0/5) and parking are invalid. Outdoor grid now 2 cm (joints one cell wide; terrain wavelengths scaled
  by the grid), render cost ~= indoor hfield. Stock sidewalk/parking re-queued as out_stock2 (out2.sh, after s2eval_j).

## cuVSLAM baseline offline (Oct 13, ~23:30 PT)
- pycuvslam 16: `horizontal_stereo_camera` -> `rectified_stereo_camera` in src/cuvslam_tracker.py. Rig accepted.
- Runner: ~/.kevin/emu/run_cuvslam.py LOG.npz [stereo|rgbd]. Default configs; RS2 rig from twin, sim intrinsics.
- room (s2eval_i GT track, 27 m, 1368 frames at 10 Hz, clean re-rendered RS2 IR/depth):
  stereo ATE 0.130 m (max 0.26, yaw max 5.6 deg, 4.4 ms/frame, 0 lost); RGB-D ATE 0.104 (max 0.24, 6.4 ms).
  For reference, wheel odometry (realistic corruption) ATE 0.402 and slam2 0.304.
  Caveat: the images carry no IR noise, and cuVSLAM is vision-only here.
- slam2 was reverted to its simplest form (the UNC weighting variant was removed; a copy is in backups). tire_mu is now realistic 1.0 / pessimistic 0.8.

## Disk management + fair visual-SLAM bench + cuVSLAM live (Oct 13, ~23:30-01:00 PT)
- Disk (i777 /): 819 GB used (95 %) -> 799 GB (92 %). Freed: pip cache 14 GB, uv cache (15.5 GiB listed, ~3 GB real, rest hardlinked into venvs),
  superseded stereo/room.npz 2.1 GB, /tmp scratch 0.9 GB, encoded raw clip frames. <=80 % is NOT reachable from our files alone (see list for James in report).
- ~/.kevin/emu/diskguard.py: below 100 GB free, rotate OUR oldest regenerable artifacts (stereo2/*.npz, rgbd/run*, kevin_clips/*.tar,
  raw clip frames that already have a packed tar; nothing touched <30 min); below 30 GB refuse writes. Wired into clip_rec.py (every 100 frames;
  frames stored 2/3 size), rgbd_rec.py (every 200 frames; frames stored at half res, stored_decimation=2), gen_stereo2.py (at start).
- gen_stereo2.py: re-render along rec/<env>.json GT, 424x240 10 Hz compressed archive (stereo2/, ~0.15-0.7 GB each), noise:
  realistic IR 2 DN / pessimistic 4 DN + gain flicker + blur; D4xx depth model (subpix 0.08/0.16, holes 0.4/2 %, band, edge shadows); sun mode.
  Finding: half-res kills cuVSLAM stereo (room realistic ATE 1.28 m vs 0.29-0.46 m at 848x480), so bench2.sh renders full res into scratch2/,
  evaluates, deletes (disk peak ~1 GB).  -> bench2.jsonl
- ORB-SLAM3 offline: run_orb.py (repo tracker, ORB defaults, sim intrinsics, no speckle filter, no atlas save; ~/.kevin/orbslam3 backed up first).
  Default ORB-SLAM3 stereo repeatedly loses track on the sim scenes and starts new maps; the TUM export then mixes maps (ATE meaningless, report tracked %).
- cuVSLAM is nondeterministic run to run (room realistic stereo ATE 0.29 vs 0.46 m on identical frames).
- Live: KEVIN_SLAM=cuv -> nav/vslam_cuv.py (cuVSLAM default pose; slam2 grid with matching+loop closure off; odom only between frames / on loss).
  Queued ~/.kevin/emu/cuv.sh after out3_DONE (8 indoor envs ideal -> s2eval_cuv, then room/maze/apartment realistic -> cuv_realistic). Untested live yet.

## Live cuVSLAM v1 + regression diagnosis (Oct 14, ~00:20-02:00 PT)
- s2eval_cuv (KEVIN_SLAM=cuv, ideal noise): 8/36 goals vs slam2 s2eval_j 22/36. Pose error far above offline (room 1.32 m vs 0.13 offline).
  Room: fine (<0.15 m) for 70 s, then ~23-30 deg yaw error during in-place spins that never recovers; v_lost 0.
  Suspect: emulator IR read noise comes from a FIXED 8-frame bank at fixed pixels = static image-space texture that pulls VO toward
  zero motion (offline re-renders used fresh noise and were fine). Fix (queued): per-frame random 2-D shift of the bank.
- maze/doors/office contacts in s2eval_j: camera rate fell to 3.7 / 13 / 12 Hz (others ~28-30) while offline GPU jobs ran in parallel.
  Fixes: offline GPU jobs paused during tours; nav/run.py speed bound <= 5 cm and 0.35 rad per depth frame (no effect at 30 Hz).
- Lawn stuck (stock tire): emulator GT jumped 1.4 m / -114 deg in the first 10 s, then 0 m.  Not reproduced passively or with
  open-loop drive (both tires move).  Spawn reference changed to the highest terrain under the footprint; rerun queued.
- Lawn sun no_depth: the edge-shadow rule (|dz| > 8 % on both sides of every edge) blanked thin blades twice.  Replaced (queued) by the
  geometric one-sided D4xx occlusion test (right-image column suffix-min, ~6 ms).
- Queue: ~/.kevin/emu/fix_round.sh (after cuv_DONE and bench2_DONE): apply world_shadowfix.py, safety_eval regression,
  s2eval_k maze/doors/office, lawn stock + sun, s2eval_cuvk (6 envs), s2eval_s2k (slam2 room/apartment/corridor).

## IMU use (Oct 14, 06:10 PT)
- slam2 already uses the IMU in the standard way: nav/slam.py Odom = translation from wheel encoders, heading from the D435i gyro
  (all 200 Hz samples since the last tick, twin IMU frame), bias learned by EMA while wheels are stopped and nothing is commanded
  (zero-velocity update). Wheel-difference yaw only when KEVIN_NO_IMU=1. No weights or slip detector. Unchanged.
- Emulator gyro errors already modeled: white noise, turn-on bias, scale error, bias random walk, thermal ramp (pessimistic).
- cuVSLAM stereo-inertial: KEVIN_CUV_IMU=1 -> OdometryMode.Inertial, rig IMU from twin rs2_imu, D435i default noise densities
  (cuvslam_tracker constants), raw samples registered in strict time order up to each frame (nav/vslam_cuv.py).
- New env "hall" (14 x 10 m open space, 4 pillars).  A/B queue: ~/.kevin/emu/ab_imu.sh -> ab_{realistic,pessimistic}_{gyro,wheels},
  ab_cuvimu{0,1}.
- fix_round results: safety_eval all pass (0 collisions); s2eval_k contacts maze 74->0, doors 43->0, office 28->0 (camera 29.7-29.9 Hz);
  s2eval_cuvk (IR noise fix) 10/32 vs 6/32 before; lawn still 0 m (stock + sun) - unresolved.

## Furnished casita + full-detail twin + live web viewer (2026-10-14)
- `emu/envs.py` casita: couch, loveseat, coffee table, kitchen table + 4 chairs, 2 beds, desk + monitor + chair; 12 labeled places;
  hard floor (tile plates) in kitchen, laundry/back door, front door; lumpy carpet elsewhere (~1 cm step at the edges); `dog=True`.
- Wall section raster now uses raw `trimesh.intersections.mesh_plane` segments (Path3D.discrete chained open segments into
  phantom diagonals across the living room, in the GT map too).  Wall collision = run-length boxes per height slab
  (MuJoCo collides meshes as convex hulls; the house mesh is visual-only now).
- `emu/world.py`: `KEVIN_EMU_MODEL=lite|full`; full twin's RealSense housing/lens meshes are hidden from the sensor cameras
  (group 5; they enclose the camera origins -> 1 cm hits -> stereo shadow -> no_depth).  Yorkie (mocap, soft) wanders/pauses/
  startles; `dog_contacts` counted separately.  Casita hfield at 5 cm cells (2.5 cm -> 4 Hz cameras).
- `KEVIN_EMU_VIEW=1`: third-person chase render (~7-10 Hz, own model copy, brighter light) with camera frustums (cyan = depth
  FOV, yellow = RGB FOV) and live textures on the far planes (colorized depth / RGB) -> `/dev/shm/kevin_view.jpg`,
  `/dev/shm/kevin_sim.json` (GT pose, dog, places); `KEVIN_EMU_VIEW_DIR` saves frames for videos.
- `~/.kevin/emu/live.sh` + `viewer.py` (:8095, cloudflared quick tunnel): 3-D view (drag/zoom), SLAM map panel, chat ->
  parser -> nav `/api/command`.  "follow the dog" re-goals 0.7 m short of the dog's GT pose every 1.5 s (sim-only cheat).
- Known gaps: SLAM drifts badly in the casita (tour GT error up to 3.8 m); recurring wall-corner contact at env (3.6, 4.7)
  on the kitchen -> living-room route; 2 dog contacts while following; MPPI rollouts not exposed by nav.

### Casita fix round (2026-10-14 PM)
- Startup twist: free (IDLE) wheels let Kevin roll ~7 cm / 19 deg on the lumpy carpet before nav armed the drives (not the
  dog: reproduced with the dog off).  IDLE hub motors now have ~0.1 Nm drag/cogging (`ODriveNode.torque`).  Tour start yaw
  error 0.32 -> 0.015 rad.  A slower creep (~0.005 rad/s at standstill) remains once nav runs with the drives in closed loop;
  pure physics with closed loop at 0 holds (checked 30 s), so it comes from the command path - open.
- Dog: treats Kevin like an obstacle (keeps >= 0.6 m from its centre, never targets/plans through that disc, also when
  running).  Old rule (startle < 0.35 m, ignored while running) let it trot into a stopped Kevin.  Dog contacts log Kevin's
  speed.  Viewer "follow the dog" holds 1.1-1.4 m.
- Chase camera: ray-cast from Kevin to the eye against walls/furniture, eye pulled in before the first hit.
- Viewer: processed obstacle input inset (RS1 floor-leveled ego labels, blank-within-0.6 m in magenta, RS2 cone, footprint,
  drop-off pixel count) via the same nav.perceive code; viewer runs under scripts/kevin-hal.
- Wall contacts now logged per contact (`CONTACT t= ... v= w=`) in world.log.

## Round: creep, scan-match gating, back-out, IMU stamps (Oct 14)
- Standstill creep: MuJoCo soft-contact creep on carpet slope. Fix: wheel `frictionloss` 0.1 Nm, `opt.noslip_iterations` (KEVIN_EMU_NOSLIP, default 10), IDLE torque 0. Parked 40 s yaw creep 0.11 -> 0.005 rad.
- slam2 match acceptance (KEVIN_SLAM2_TEST=chi2): scan-match covariance from the score Hessian (Laplace), innovation vs predicted pose, accept if Mahalanobis^2 < 11.34 (chi2 99%, 3 dof), then Kalman update. Old gate kept as `gate`.
- Wedged recovery (nav/run.py): breadcrumb trail; when the local planner fails, reverse along the trail at 0.10 m/s (safety filter still applies), retry every 0.25 m up to 0.8 m. Untested yet.
- IMU (src/imu.py): samples stamped with the sensor timestamp (same clock as depth frames) instead of host poll time; batched samples previously shared one stamp and cuVSLAM dropped them. Falls back to host time if the domains differ by >1 s.
- cas_full3 (chi2, before noslip/back-out): 6/9 goals, 1 wall contact (back door, 6.20,8.06), 0 dog, SLAM max 0.29 m (was 0.81).
- Viewer style (emu/world.py, VIEW only): each camera is a small solid pyramid (apex = optical centre, 8 cm deep for depth, 10 cm see-through for colour) with the live image on a small plane 3.5 cm behind the apex; full FOV drawn as thin translucent lines to VIEW_CLIP = 0.2..1.2 m with faint near/far rectangles. The big far-plane image boards are gone.
- Viewer style revised (James): no pyramid bodies. Each camera = thin translucent frustum edge lines from the lens to 1.2 m + faint near (0.2 m) / far rectangles; the live image fills the frustum cross-section 10 cm (depth) / 9 cm (colour) in front of the lens, camera-oriented.
- cas_full4 (all fixes: noslip/frictionloss, chi2, back-out, IMU stamps, frustum-line view): 5/9 (2 of them "closest reachable"), 0 wall contacts, 0 dog, SLAM max 0.84 m / final 0.80 m (worse than cas_full3 0.29 m), back-out never triggered. ab3 A/B: IMU arm beats wheels-only in realistic noise, but both arms are below the earlier ab_ runs -> regression/variance to investigate.
