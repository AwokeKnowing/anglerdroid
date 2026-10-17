seq casita_long  path 45.21 m, turned 3190.0 deg

| method | coverage | ate_xy | final_xy | yaw_max | yaw_final | yaw_per_360 | rpe1m_med | ate_aligned | scale | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| cuvslam_rgbd | 1.0 | 7.948 | 9.189 | 60.2 | 56.7 | 6.79 | 0.006 | 2.528 | None | 12.3 ms/frame (p99 69.8), lost 0 |
| cuvslam_stereo | 1.0 | 6.924 | 7.998 | 58.43 | 47.05 | 6.59 | 0.011 | 2.216 | None | 7.8 ms/frame (p99 18.0), lost 0 |
| cuvslam_stereo_imu | 1.0 | 10.421 | 12.147 | 132.31 | 122.67 | 14.93 | 0.022 | 2.632 | None | 8.5 ms/frame (p99 27.3), lost 0 |
| kslam | 1.0 | 0.716 | 0.959 | 4.78 | 4.18 | 0.54 | 0.005 | 0.364 | None | 45.52 ms/frame CPU (python, 9001 frames) |
| kslam_vo | 1.0 | 0.536 | 0.517 | 4.78 | 4.18 | 0.54 | 0.01 | 0.211 | None | post-process; slip-flagged steps 18, VO steps used 9000 of 9000 |
| odom_wheel | 1.0 | 1.776 | 2.2 | 20.33 | 10.8 | 2.29 | 0.01 | 0.504 | None |  |
| odom_wheel_gyro | 1.0 | 0.991 | 1.22 | 8.23 | 1.31 | 0.93 | 0.006 | 0.465 | None |  |
| orbslam3_stereo | 0.128 | 2.077 | 4.508 | 82.73 | 81.73 | 23.84 | 0.003 | 1.812 | None | crashed |
| slam2 | 1.0 | 0.592 | 0.796 | 5.21 | 3.32 | 0.59 | 0.019 | 0.353 | None | 99.6 ms/scan (2822 scans) |
| vggtslam_mast | None | 7.555 | 7.367 | 154.82 | 76.22 | 20.5 | 0.768 | 2.555 | 1.1626 | 112 s wall for 9001 frames (80.0 FPS of input, 5.22 GB peak torch) |
| vggtslam_rs2_color | None | 5.748 | 2.525 | 179.9 | 1.13 | 22.17 | 1.066 | 3.102 | 0.0108 | 196 s wall for 9001 frames (45.9 FPS of input, 5.63 GB peak torch) |
