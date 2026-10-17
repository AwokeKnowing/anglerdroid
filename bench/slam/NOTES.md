# Kevin SLAM / reconstruction research + benchmark notes (branch slambench-2026-10-17)

Status: WIP, 2026-10-17 (America/Los_Angeles). Only measured numbers are listed in the results tables. Research rows
are taken from papers/READMEs, and those rows are marked as claims.

## 1. Candidate survey (2024–2026)

| system | license | GPU / rate (claims unless marked measured) | self-calib | wheel / IMU fusion | loop closure | map output | maintained |
|---|---|---|---|---|---|---|---|
| **cuVSLAM / PyCuVSLAM** (NVIDIA, 2025) | NVIDIA community (free; check terms for commercial use) | Orin real time, multi-cam; **measured 7–11 ms/frame on the 3090** | no (needs calibrated rig) | stereo-inertial mode; wheel only through external fusion | yes (SLAM mode, localization map) | poses + landmarks | yes (Isaac ROS 3.x) |
| **nvblox** (NVIDIA) | Apache-2.0 | Orin real time | n/a (mapping only) | takes any pose source | n/a | TSDF/ESDF mesh, 2D costmap slice | yes |
| **ORB-SLAM3** | GPL-3.0 | CPU; **measured 180 ms/frame here** (offline, niced, contended, png decode included) | no | VI (IMU); no wheel upstream | yes (DBoW2, multi-map atlas) | sparse | low activity (community forks) |
| **VGGT-SLAM 2.0** (MIT-SPARK, RSS 2026) | BSD-2 code; VGGT-1B weights under Meta's license (check terms) | claim 8.4 FPS on 3090, 3.5 FPS on Jetson Thor; **measured 13–20 FPS input throughput with keyframing, 5.3–5.7 GB** | yes (predicts intrinsics; SL(4) submaps cope with projective ambiguity) | none (pure RGB) | yes (SALAD retrieval) | dense point cloud per submap; FOUND-IT scene graph on top | yes (real-time code June 2026) |
| **MASt3R-SLAM** (CVPR 2025) | code CC BY-NC-SA; MASt3R checkpoints naver non-commercial | claim 15 FPS (about 7 FPS reported on a 3090-class card) | yes (generic central-camera ray model) | none | yes (ASMK retrieval) | dense pointmaps / ply | moderate |
| MASt3R-Fusion (2025) | research | 3090-class | yes | IMU + GNSS, online calibration | yes | dense | new |
| DROID-SLAM / DPVO / DPV-SLAM | BSD | 3090: DROID about 20 FPS (high VRAM), DPVO 60+ FPS | no | DBA-Fusion adds IMU, wheel and GNSS to DROID | DPV-SLAM yes | sparse/semi-dense | maintenance mode |
| MegaSaM (2025) | Apache | offline only | yes (focal + depth) | no | global BA | dense depth per frame | research |
| GS-SLAMs (SplaTAM, MonoGS, Photo-SLAM, GS-ICP…) | mostly non-commercial | 3090, 1–10 FPS | no | no | some | Gaussian splats (photoreal) | research churn |
| Kimera / Hydra (MIT-SPARK) | BSD | CPU real time | no | VIO + external odom | yes | mesh + 3D scene graph | Hydra active |
| VIW-Fusion / wheel-VIO (e.g. OpenVINS + wheel) | GPL / BSD | CPU | online extrinsic and wheel calibration | **yes: visual-inertial-wheel** | VIW-Fusion yes | sparse | moderate |

## 2. Measured: casita_spins (realistic sim: PBR textures, decor, sun and lamps, CamFX lens/noise/AE; path 9.31 m, 3013° turned, 1615 frames at 15 Hz)
Metrics:
- ate_xy, final_xy and yaw are anchored at the start pose, in metres and degrees.
- yaw_per_360 is the heading error per 360° turned.
- ate_aligned is SE3 for metric methods and Sim3 for scale-free ones.
- Rows marked keyframes-only have coverage None.

| method | ate_xy | final_xy | yaw_max | yaw_final | yaw/360 | rpe1m_med | ate_aligned | runtime |
|---|---|---|---|---|---|---|---|---|
| odom_wheel | 0.19 | 0.395 | 6.95 | 6.94 | 0.83 | 0.020 | 0.128 | – |
| odom_wheel_gyro | 0.092 | 0.191 | 5.07 | 0.65 | 0.61 | 0.019 | 0.067 | – |
| slam2 (current) | 0.035 | 0.066 | 4.08 | 0.17 | 0.49 | 0.011 | 0.028 | 62.8 ms/scan CPU |
| cuvslam_stereo | 0.025 | 0.024 | 5.42 | 0.06 | 0.65 | 0.006 | 0.023 | 7.0 ms/frame (p99 13.9) |
| cuvslam_stereo_imu | 0.019 | 0.012 | 1.01 | 0.72 | 0.12 | 0.012 | 0.017 | 8.8 ms/frame (p99 17.5) |
| cuvslam_rgbd | 0.020 | 0.007 | 9.16 | 0.42 | 1.09 | 0.006 | 0.019 | 11.1 ms/frame |
| orbslam3_stereo | 0.005 | 0.001 | 0.39 | 0.07 | 0.05 | 0.004 | 0.004 | 180 ms/frame CPU offline |
| vggtslam_mast (uncalib, submap 8) | 0.567 | 0.736 | 14.32 | 7.74 | 1.8 | 0.372 | 0.374 (Sim3) | 79 s / 1615 fr, 5.3 GB |
| vggtslam_rs2_color (uncalib, submap 8) | 0.138 | 0.131 | 14.32 | 14.3 | 1.74 | 0.295 | 0.126 (Sim3) | 121 s / 1615 fr, 5.7 GB |

Self-calibration check, VGGT-SLAM median focal estimate vs withheld twin truth (640 px wide):
- rs2_color: 646.6 vs 625.2 (+3.4%).
- mast: 325.8 vs 240.3 (+36%; true HFOV 106.2°, which is outside VGGT's training distribution).

VGGT used submap 8 because submap 16 ran out of GPU memory with LightNav using 14.5 GB.


## 2b. Measured: casita_tour (31.31 m, 3289° turned, 2495 frames, 166 s; one rug-edge bump at t≈47 s: pitch 11.8°, roll −5.7°, GT yaw-rate spike 1.68 rad/s vs wheel 0.51)

| method | coverage | ate_xy | final_xy | yaw_max | yaw_final | yaw/360 | rpe1m_med | ate_aligned | runtime |
|---|---|---|---|---|---|---|---|---|---|
| odom_wheel | 1.0 | 2.355 | 3.963 | 53.07 | 42.7 | 5.81 | 0.012 | 1.142 | – |
| odom_wheel_gyro | 1.0 | 0.586 | 0.998 | 14.17 | 13.53 | 1.55 | 0.012 | 0.361 | – |
| slam2 | 1.0 | 0.184 | 0.352 | 11.48 | 7.63 | 1.26 | 0.020 | 0.126 | 96.2 ms/scan |
| cuvslam_stereo | 1.0 | 2.08 | 2.451 | 47.44 | 41.16 | 5.19 | 0.008 | 0.924 | 7.8 ms/frame |
| cuvslam_stereo_imu | 1.0 | 0.386 | 0.636 | 29.71 | 28.56 | 3.25 | 0.016 | 0.386 | 9.8 ms/frame |
| cuvslam_rgbd | 1.0 | 2.492 | 3.838 | 43.84 | 38.41 | 4.8 | 0.005 | 1.129 | 13.0 ms/frame |
| orbslam3_stereo | 0.998 | 0.138 | 0.001 | 6.59 | 0.01 | 0.74 | 0.005 | 0.114 | 304 ms/frame CPU offline (contended) |
| vggtslam_mast | kf | 0.852 | 0.328 | 64.52 | 22.86 | 7.84 | 0.454 | 0.782 (Sim3) | 95 s / 2495 fr |
| vggtslam_rs2_color | kf | 4.051 | 2.76 | 177.73 | 149.23 | 20.41 | 0.85 | 2.856 (Sim3) | 140 s / 2495 fr |

## 2c. Measured: casita_long (45.21 m, 3190° turned, 9001 frames, 600 s cap; route not finished, robot stalled after ~300 s)

| method | coverage | ate_xy | final_xy | yaw_max | yaw_final | yaw/360 | rpe1m_med | ate_aligned | runtime |
|---|---|---|---|---|---|---|---|---|---|
| odom_wheel | 1.0 | 1.776 | 2.2 | 20.33 | 10.8 | 2.29 | 0.010 | 0.504 | – |
| odom_wheel_gyro | 1.0 | 0.991 | 1.22 | 8.23 | 1.31 | 0.93 | 0.006 | 0.465 | – |
| slam2 | 1.0 | 0.592 | 0.796 | 5.21 | 3.32 | 0.59 | 0.019 | 0.353 | 99.6 ms/scan |
| cuvslam_stereo | 1.0 | 6.924 | 7.998 | 58.43 | 47.05 | 6.59 | 0.011 | 2.216 | 7.8 ms/frame |
| cuvslam_stereo_imu | 1.0 | 10.421 | 12.147 | 132.31 | 122.67 | 14.93 | 0.022 | 2.632 | 8.5 ms/frame |
| cuvslam_rgbd | 1.0 | 7.948 | 9.189 | 60.2 | 56.7 | 6.79 | 0.006 | 2.528 | 12.3 ms/frame |
| orbslam3_stereo | 0.128 | 2.077 | 4.508 | 82.73 | 81.73 | 23.84 | 0.003 | 1.812 | CRASHED at t≈80 s (Sophus SO3::exp NaN) |
| vggtslam_mast | kf | 7.555 | 7.367 | 154.82 | 76.22 | 20.5 | 0.768 | 2.555 (Sim3) | 112 s / 9001 fr |
| vggtslam_rs2_color | kf | 5.748 | 2.525 | 179.9 | 1.13 | 22.17 | 1.066 | 3.102 (Sim3) | 196 s / 9001 fr |

VGGT-SLAM focal self-calibration, median estimate vs truth:

| sequence | rs2_color (truth 625.2) | mast (truth 240.3) |
|---|---|---|
| spins | 646.6 | 325.8 |
| tour | 620.7 | 327.7 |
| long | 661.6 | 333.3 |

The rs2_color estimate is within −1% to +6%. The mast estimate is consistently about 36–39% too long, which means the 106° lens is read as about 88–90°.

### Diagnosis of the visual failures on tour/long
- **Self-occlusion.** The 27°-pitched RS2 sees Kevin's own deck in the bottom of the frame: IR rows 365 and below (24% of the image) and colour rows about 400 and below never change. Static features on the robot body pull VO toward "no motion", especially during turns.
  - A per-sequence auto self-mask (temporal row variance) was added to run_cuvslam.py (CUVSLAM_SELFMASK=1). Its tour runs finished, but the results were not yet evaluated when i777 went offline.
- **Close-range, low-texture views** (pressed against the bed, blank walls) at 120–140 s in long.
- **Bump jolts** (rug edge) hurt stereo-inertial cuVSLAM most: +72° at the 43 s bump in long.
- **Monocular feed-forward SLAM** (VGGT-SLAM) on floor-heavy, rotation-dominant views with sparse keyframes (165–334 kf over 9001 frames) is not usable as an odometry source. Treat it as a global mapper only.
- **slam2** (the current stack, wheel+gyro+scan matching) was the most robust over the long run. ORB-SLAM3 was the most accurate while it lived, but crashed.

## 2d. Our combined estimator: kslam (+ kfuse), plus LingBot-Map. All numbers measured.
Bench files are in `bench/slam/`. Per-sequence tables are in `bench/slam/results/casita_*.md`.

ATE and yaw are anchored at the start pose:
| seq | method | ate_xy m | final_xy m | yaw_max ° | yaw/360 ° | ate_aligned m |
|---|---|---|---|---|---|---|
| spins | kslam | 0.045 | 0.051 | 2.21 | 0.26 | 0.021 |
| spins | slam2 | 0.035 | 0.066 | 4.08 | 0.49 | 0.028 |
| tour | **kslam** | 0.211 | 0.344 | **2.83** | **0.31** | 0.127 |
| tour | slam2 | 0.184 | 0.352 | 11.48 | 1.26 | 0.126 |
| tour | kslam_vo (cuVSLAM translation) | 0.693 | 0.826 | 2.83 | 0.31 | 0.376 |
| long | **kslam** | 0.716 | 0.959 | **4.78** | **0.54** | 0.364 |
| long | kslam_vo | **0.536** | **0.517** | 4.78 | 0.54 | **0.211** |
| long | slam2 | 0.592 | 0.796 | 5.21 | 0.59 | 0.353 |
| tour_p24 (true RS2 pitch 24°, belief 27°) | kslam (self-cal) / no self-cal | 0.232 / 0.223 | | 3.66 / 3.35 | | 0.127 / 0.119 |

LingBot-Map, uncalibrated, streaming at 392 px, KV window 16, SDPA, 5.1–5.2 GB:
| seq | camera | ate_aligned (Sim3) | yaw_max | speed |
|---|---|---|---|---|
| spins | mast | 0.833 | 179.8° | 6.9 FPS |
| spins | rs2_color | 0.729 | 172.6° | 6.9 FPS |
| tour | mast | 0.68 | 178.3° | 6.9 FPS |
| tour | rs2_color | 1.804 | 179.0° | 7.0 FPS |

LingBot-Map focal estimates: mast 252.8 / 255.8 (truth 240.3, +5–6%); rs2_color 582.3 / 609.3 (truth 625.2).

Self-calibration (measured):
- **RS2 pitch** from the floor plane: estimated 23.65° with true 24.0° and belief 27° (p24 sequence). With truth 27° it estimated 26.85 / 26.65 / 26.03 (spins / tour / long).
  - Camera height: 0.466 m vs twin 0.468 m (0.382 m + axle 0.0857 m).
- **Gyro yaw scale**: estimated −0.0073 / −0.0063 / −0.0107. Effective truth is −0.0081 (from the drawn per-axis errors projected through the IMU mount).
- **Mast lens (mastcal)**: rotation-consistency calibration on in-place turns, using kslam heading and no target or map.

| seq | fx (truth 240.34) | k1 (truth −0.050) | k2 (truth 0.008) | HFOV (truth 109.97°) |
|---|---|---|---|---|
| spins | 243.71 | −0.0498 | 0.0085 | 109.01° |
| tour | 244.06 | −0.0483 | 0.0080 | 108.86° |
| long | 244.32 | −0.0443 | 0.0057 | 108.85° |

  HFOV here is the edge-ray angle, including distortion; the pinhole-fx HFOV is 106.18°. For comparison:

| source | fx | error vs truth |
|---|---|---|
| robot belief (100° HFOV) | 268.5 | +11.7% |
| VGGT-SLAM | 326–333 | +36–39% |
| LingBot-Map | 253–256 | +5–6% |
| mastcal | 243.7–244.3 | +1.4–1.7% |

### What the numbers say
1. Heading drift is solved by structure, not by a better matcher. Manhattan wall directions from RS2 depth normals, learned online with no map, cut tour yaw_max from 11.5° (slam2) to 2.8°. The residual check against GT gave median 0.6° and p90 2.0° per observation.
2. Off-the-shelf VO/SLAM on this robot's views is fragile.
   - cuVSLAM drifts 26–123° on tour/long through rug bumps, close low-texture views and floor-dominated frames. A self-mask did not fix it.
   - ORB-SLAM3 is the most accurate while alive (5 mm on spins) but crashed with a Sophus NaN at 80 s on long.
   - Feed-forward models (VGGT-SLAM 2.0, LingBot-Map) lose heading on in-place spins: yaw flips of about 180°, a rotation-only degeneracy at small KV/window sizes.
3. Use learned models for what they are good at:
   - dense geometry and global mapping on i777, not odometry;
   - a focal prior (LingBot-Map +5%). Fine calibration comes from geometry (mastcal +1.5%).
4. Translation is now the main error, from wheel slip on rug edges and the 0.3% radius error.
   - VO translation helped on long (ATE 0.72 → 0.54 m) but hurt on tour (0.21 → 0.69 m).
   - The next step is loop closure / map relocalization edges in an i777 pose graph, not more odometry.

## 3. Architecture (proposal)
- **Robot (Orin): local pose and local map, always autonomous.**
  - EKF on SE(2)+z: wheel odometry is the prior; the gyro covers short windows; ZUPT from cmd.
  - cuVSLAM stereo-inertial on the D435i IR pair (about 9 ms/frame measured on the 3090) as the visual odometry factor.
  - nvblox (or the existing 2D grid) for the local ESDF and costmap in the odom frame.
  - Safety uses only odom-frame data (RS1 top-down), never needs i777.
- **i777 (3090): global SLAM and reconstruction.**
  - Sources: keyframes (JPEG RGB + 16-bit depth PNG, about 2–4 Mbit/s at 3–5 Hz) plus odom poses.
  - Pose graph: odom + cuVSLAM edges + loop closures (VGGT-SLAM 2.0 / MASt3R retrieval).
  - Dense reconstruction: TSDF/mesh, optionally splats; scene graph.
  - Sends back `T_map_odom` corrections (about 1 Hz, tiny) and map tiles.
- **Link loss.** The robot keeps running in the odom frame. The last `T_map_odom` is held and its covariance grows. Goal-directed global navigation pauses once the covariance passes a threshold; local safety is unaffected. On reconnect, apply the correction smoothly (rate-limited), never as a jump inside the planner window.
- **Frames.** map → odom (from i777) → base (robot EKF) → sensors (twin extrinsics, refined online).
- **Auto-calibration.**
  - Mast intrinsics: per-session focal from a feed-forward model is unreliable at 106° (see the +36% above). Use a geometric approach: refine fx, k1, k2 by BA against RS2-depth-anchored landmarks, or run a self-calibrating BA (COLMAP-style) on i777 over a spin sequence, seeded from the 100° belief.
  - RS2 pitch: estimate from the floor plane in depth (RANSAC) every few seconds, plus gravity from the D435i accelerometer.

## 3b. Recommended system ("kslam", ours)
- **On the robot (Orin, CPU-light):**
  - kslam EKF: wheel prior; gyro with online scale/bias; ZUPT; Manhattan heading from RS2 depth at 5 Hz (49 ms/frame in Python offline, not optimised); floor-plane pitch/height self-cal; auto self-occlusion mask.
  - Local 2D/2.5D map in the odom frame.
  - The safety layer stays independent.
- **On i777:**
  - Pose graph: kslam odometry edges; loop-closure edges from place recognition plus metric RGB-D registration (or ORB-SLAM3/cuVSLAM relocalization kept as an edge source, never as the pose owner); Manhattan as a unary heading prior.
  - Dense reconstruction (TSDF / splats) at the corrected poses. LingBot-Map (Apache) or MapAnything-apache can serve as a dense-geometry prior for the mast camera.
  - mastcal is re-run per session to keep mast intrinsics current.
  - Sends `T_map_odom` at about 1 Hz.
  - On link loss the robot continues in the odom frame with growing covariance.

## 4. Unresolved / caveats
- Sim IR stereo has no projector speckle or real rolling-shutter artifacts, so ORB/cuVSLAM numbers are optimistic.
- GPU contention: LightNav vLLM uses 14.5 GB and ollama qwen2.5:7b 6.6 GB. The ollama model was unloaded temporarily for the GPU runs and must be reloaded.
- MASt3R-SLAM: install unfinished (CUDA backend compile interrupted by the 11:30 i777 crash); not measured.
- ABot-Recon, MapAnything, DA3 and AMB3R were not installed (GPU budget and time). AMB3R has no license.
- Live sim restarted once at 15:36 PDT from the slambench worktree (merged with the live branch). Measurements:
  - camera rate 18.5–19.9 Hz with realism, vs 20.0–20.8 Hz before;
  - offline selftest under contention: realism on 16 Hz vs off 20 Hz.
- Decor plants and the lamp base collide, so the GT map changed slightly.
