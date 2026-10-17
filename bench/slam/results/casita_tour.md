seq casita_tour  path 31.31 m, turned 3289.0 deg

| method | coverage | ate_xy | final_xy | yaw_max | yaw_final | yaw_per_360 | rpe1m_med | ate_aligned | scale | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| cuvslam_rgbd | 1.0 | 2.492 | 3.838 | 43.84 | 38.41 | 4.8 | 0.005 | 1.129 | None | 13.0 ms/frame (p99 26.8), lost 0 |
| cuvslam_stereo | 1.0 | 2.08 | 2.451 | 47.44 | 41.16 | 5.19 | 0.008 | 0.924 | None | 7.8 ms/frame (p99 18.9), lost 0 |
| cuvslam_stereo_imu | 1.0 | 0.386 | 0.636 | 29.71 | 28.56 | 3.25 | 0.016 | 0.386 | None | 9.8 ms/frame (p99 21.7), lost 0 |
| odom_wheel | 1.0 | 2.355 | 3.963 | 53.07 | 42.7 | 5.81 | 0.012 | 1.142 | None |  |
| odom_wheel_gyro | 1.0 | 0.586 | 0.998 | 14.17 | 13.53 | 1.55 | 0.012 | 0.361 | None |  |
| orbslam3_stereo | 0.998 | 0.138 | 0.001 | 6.59 | 0.01 | 0.74 | 0.005 | 0.114 | None | 304.4 ms/frame wall incl. png decode (2495 frames, CPU) |
| slam2 | 1.0 | 0.184 | 0.352 | 11.48 | 7.63 | 1.26 | 0.02 | 0.126 | None | 96.2 ms/scan (596 scans) |
| vggtslam_mast | None | 0.852 | 0.328 | 64.52 | 22.86 | 7.84 | 0.454 | 0.782 | 1.8112 | 95 s wall for 2495 frames (26.2 FPS of input, 5.28 GB peak torch) |
| vggtslam_rs2_color | None | 4.051 | 2.76 | 177.73 | 149.23 | 20.41 | 0.85 | 2.856 | 0.1788 | 140 s wall for 2495 frames (17.8 FPS of input, 5.65 GB peak torch) |
