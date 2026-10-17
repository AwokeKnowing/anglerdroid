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

## 4. Unresolved / caveats
- Sim IR stereo has no projector speckle or real rolling-shutter artifacts, so ORB/cuVSLAM numbers are optimistic.
- GPU contention: LightNav vLLM uses 14.5 GB and ollama qwen2.5:7b 6.6 GB. The ollama model was unloaded temporarily for the GPU runs and must be reloaded.
- Tour and long sequences: in progress.
