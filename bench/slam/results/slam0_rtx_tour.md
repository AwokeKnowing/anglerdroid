# SLAM from zero on the photoreal RTX sim (2026-10-22)

3 runs of route 'tour' (31.5 m, ~3200 deg turned, 161 s), emulator noise 'realistic', seeds 0-2 (different wheel/IMU/depth noise draws; the GT path is the same because the data-generation controller follows GT). Cameras: RTX real-time path tracing at the GT pose of every frame (render_rtx.py). No map/STL used by any method; scores vs sim truth (eval.py, anchored at the start pose) and the twin GT occupancy (occmap.py, shared mapper).

| method | n | ATE xy m, median (min..max) | final xy m | max yaw err deg | RPE/1m m | coverage | map occ IoU | map occ F1 (4 cm tol) | free IoU | fps* |
|---|---|---|---|---|---|---|---|---|---|---|
| _gt_pose_map (mapper upper bound) | 3 | - | - | - | - | - | 0.441 (0.441..0.443) | 0.719 | 0.797 | - |
| slam2 | 3 | 0.308 (0.126..0.675) | 0.238 | 9.6 | 0.032 | 1.00 | 0.160 (0.111..0.255) | 0.387 | 0.657 | 37.7 |
| kslam | 3 | 0.355 (0.156..0.378) | 0.045 | 6.2 | 0.024 | 1.00 | 0.169 (0.167..0.253) | 0.394 | 0.679 | 14.4 |
| rtab_vo | 3 | 0.428 (0.381..1.901) | 0.753 | 9.7 | 0.008 | 0.73 | 0.200 (0.162..0.210) | 0.447 | 0.694 | 10.2 |
| rtab_wio | 3 | 0.432 (0.294..1.015) | 0.003 | 7.3 | 0.028 | 1.00 | 0.152 (0.084..0.193) | 0.378 | 0.638 | 65.0 |
| odom_wheel_gyro | 3 | 0.453 (0.334..1.023) | 0.528 | 7.6 | 0.028 | 1.00 | 0.143 (0.085..0.166) | 0.347 | 0.628 | - |
| viw_fg | 3 | 1.195 (0.959..1.622) | 1.308 | 39.2 | 0.041 | - | 0.112 (0.108..0.127) | 0.285 | 0.599 | 6.8 |
| wi_fg | 3 | 1.260 (1.058..1.749) | 1.502 | 41.7 | 0.042 | - | 0.109 (0.108..0.126) | 0.276 | 0.594 | 1415.2 |
| kiss_icp | 3 | 1.433 (1.427..2.628) | 2.720 | 52.9 | 0.204 | 1.00 | 0.087 (0.086..0.102) | 0.232 | 0.569 | 5.0 |
| odom_wheel | 3 | 2.049 (1.938..2.193) | 3.304 | 19.0 | 0.027 | 1.00 | 0.097 (0.093..0.145) | 0.256 | 0.596 | - |
| orb_rgbd | 3 | 3.105 (3.104..3.109) | 0.182 | 109.2 | 0.010 | 0.93 | 0.089 (0.085..0.092) | 0.221 | 0.582 | 11.1 |
| orb_mono | 3 | 5.426 (4.334..5.929) | 0.071 | 179.7 | 0.658 | 0.87 | - | - | - | 9.5 |
| orb_mono_imu | 3 | 5.604 (5.554..6.553) | 0.708 | 179.9 | 0.143 | 0.83 | 0.053 (0.036..0.063) | 0.162 | 0.231 | 9.1 |
| orb_rgbd_imu | 3 | 6.706 (6.677..6.851) | 0.066 | 180.0 | 0.025 | 0.83 | 0.066 (0.056..0.068) | 0.181 | 0.343 | 10.0 |

\* fps measured with other bench jobs sharing the 8 cores (indicative only); slam2 = scans/s at 5 Hz input; wi_fg = offline graph speed.

IMU-camera clock offset measured by gyro/wheel yaw-rate cross-correlation: 3.46-3.47 ms on all 3 runs (emulator truth 3.0 ms). Real log 20261021_213451: IMU vs wheel-speed residual -42.6 ms (measured).
