seq casita_spins  path 9.31 m, turned 3013.0 deg

| method | coverage | ate_xy | final_xy | yaw_max | yaw_final | yaw_per_360 | rpe1m_med | ate_aligned | scale | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| cuvslam_rgbd | 1.0 | 0.02 | 0.007 | 9.16 | 0.42 | 1.09 | 0.006 | 0.019 | None | 11.1 ms/frame (p99 21.1), lost 0 |
| cuvslam_stereo | 1.0 | 0.025 | 0.024 | 5.42 | 0.06 | 0.65 | 0.006 | 0.023 | None | 7.0 ms/frame (p99 13.9), lost 0 |
| cuvslam_stereo_imu | 1.0 | 0.019 | 0.012 | 1.01 | 0.72 | 0.12 | 0.012 | 0.017 | None | 8.8 ms/frame (p99 17.5), lost 0 |
| kslam | 1.0 | 0.045 | 0.051 | 2.21 | 0.29 | 0.26 | 0.012 | 0.021 | None | 44.97 ms/frame CPU (python, 1615 frames) |
| kslam_vo | 1.0 | 0.072 | 0.058 | 2.21 | 0.29 | 0.26 | 0.012 | 0.034 | None | post-process; slip-flagged steps 0, VO steps used 1612 of 1614 |
| odom_wheel | 1.0 | 0.19 | 0.395 | 6.95 | 6.94 | 0.83 | 0.02 | 0.128 | None |  |
| odom_wheel_gyro | 1.0 | 0.092 | 0.191 | 5.07 | 0.65 | 0.61 | 0.019 | 0.067 | None |  |
| orbslam3_stereo | 1.0 | 0.005 | 0.001 | 0.39 | 0.07 | 0.05 | 0.004 | 0.004 | None | 180.1 ms/frame wall incl. png decode (1615 frames, CPU) |
| slam2 | 1.0 | 0.035 | 0.066 | 4.08 | 0.17 | 0.49 | 0.011 | 0.028 | None | 62.8 ms/scan (259 scans) |
| vggtslam_mast | None | 0.567 | 0.736 | 14.32 | 7.74 | 1.8 | 0.372 | 0.374 | 2.6192 | 79 s wall for 1615 frames (20.4 FPS of input, 5.30 GB peak torch) |
| vggtslam_rs2_color | None | 0.138 | 0.131 | 14.32 | 14.3 | 1.74 | 0.295 | 0.126 | 1.5309 | 121 s wall for 1615 frames (13.4 FPS of input, 5.66 GB peak torch) |
