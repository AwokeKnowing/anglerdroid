# SLAM from zero on the photoreal RTX sim — round 2: 3 genuinely different routes (n = 3)

Routes (bench/slam/record.py, each its own start pose + path, realistic noise, seeds 10-12; RTX colour from
bench/slam/render_rtx.py on the re-dressed scene; depth/wheel/IMU from the MuJoCo twin):

| route | start (x, y, yaw) | path | turned | frames (15 Hz) |
|---|---|---|---|---|
| bedrooms | (5.0, 11.0, 180°) bedroom 2 | 36.6 m | 3350° | 2580 |
| kitchen_loop | (5.6, 4.95, -90°) kitchen table | 39.8 m | 3201° | 2653 |
| door_long | (0.9, 4.0, 0°) front door | 45.2 m | 3405° | 2980 |

## Results (MEASURED, twin ground truth; median over the 3 routes, (min..max))

Map IoU = occupied-cell IoU of the 2D occupancy map built from RS2 depth with each method's poses vs the casita
walls+counters raster (4 cm); "_gt_pose_map" uses the true poses (upper bound of this mapper).

| method | n | ATE xy m, median (min..max) | final xy m | max yaw err deg | RPE/1m m | coverage | map occ IoU | map occ F1 (4 cm tol) | free IoU | fps* |
|---|---|---|---|---|---|---|---|---|---|---|
| _gt_pose_map (mapper upper bound) | 3 | - | - | - | - | - | 0.450 (0.439..0.522) | 0.734 | 0.809 | - |
| slam2 | 3 | 0.156 (0.096..0.221) | 0.103 | 7.7 | 0.024 | 1.00 | 0.288 (0.260..0.349) | 0.586 | 0.754 | 57.1 |
| viw_fg | 3 | 0.167 (0.150..0.207) | 0.049 | 3.3 | 0.039 | - | 0.253 (0.217..0.332) | 0.518 | 0.737 | 4.2 |
| rtab_wio | 3 | 0.224 (0.093..0.467) | 0.111 | 4.5 | 0.022 | 1.00 | 0.253 (0.160..0.295) | 0.535 | 0.750 | 73.6 |
| kslam | 3 | 0.273 (0.192..0.577) | 0.275 | 6.2 | 0.022 | 1.00 | 0.199 (0.138..0.212) | 0.459 | 0.713 | 25.0 |
| odom_wheel_gyro | 3 | 0.275 (0.214..0.476) | 0.335 | 5.3 | 0.022 | 1.00 | 0.208 (0.162..0.228) | 0.468 | 0.713 | - |
| wi_fg | 3 | 0.290 (0.170..0.439) | 0.285 | 4.0 | 0.042 | - | 0.204 (0.181..0.250) | 0.460 | 0.731 | 1139.9 |
| viw_fg_noscale | 3 | 0.648 (0.343..2.100) | 1.127 | 21.4 | 0.045 | - | 0.120 (0.112..0.187) | 0.310 | 0.612 | 4.2 |
| wi_fg_noscale | 3 | 0.716 (0.338..2.200) | 1.225 | 22.4 | 0.047 | - | 0.118 (0.117..0.187) | 0.303 | 0.616 | 840.6 |
| odom_wheel | 3 | 1.208 (0.583..2.333) | 1.929 | 29.5 | 0.026 | 1.00 | 0.109 (0.094..0.157) | 0.276 | 0.603 | - |
| orb_rgbd | 3 | 4.785 (1.994..8.476) | 3.304 | 178.7 | 0.008 | 1.00 | 0.077 (0.056..0.279) | 0.200 | 0.426 | 20.6 |
| rtab_vo | 3 | 5.755 (1.140..6.817) | 7.924 | 9.4 | 0.007 | 0.62 | 0.091 (0.075..0.158) | 0.260 | 0.386 | 8.4 |

\* fps: own wall-clock throughput on i777 with 3 pipelines + renders running in parallel (load ~26 on 8 cores), so
lower than on an idle box; viw_fg is dominated by Open3D RGB-D odometry (~0.45-0.7 s per keyframe on CPU).

## What changed since round 1 and what it did (MEASURED)

* **Online gyro/wheel calibration in viw_fg / wi_fg** (bench/slam/run_viw_fg.py): a 4-state Kalman filter
  [gyro yaw scale ε_g, gyro yaw bias b, wheel yaw scale ε_w (effective track), wheel-diameter mismatch ε_d (UMBmark E_d)]
  on per-keyframe yaw increments gyro-vs-wheel and gyro-vs-visual, priors from specs (ε_g 1% BMI055 sensitivity
  tolerance, ε_d 0.7% from 0.5% radius tolerance; ε_w 5% ASSUMED for soft tyres on carpet). Corrections are applied
  causally before preintegration / to the wheel factor; the IMU bias stays a graph state. Also: static initialisation
  in the rest window before the first command (round 1 took gravity from the spawn transient).
  * ATE median viw_fg 0.648 → 0.167 m, wi_fg 0.716 → 0.290 m; max yaw error 21° → 3-4°.
  * Estimates vs the emulator's drawn (withheld) errors, viw_fg:
    | route | ε_g est / true (base yaw = IMU y) | ε_d est / true (r_R − r_L)/r | ε_w est |
    |---|---|---|---|
    | bedrooms | −0.19% / −0.30% | −0.34% / −0.31% | −5.2% |
    | kitchen_loop | +1.34% / +1.62% | −0.38% / −0.35% | −4.7% |
    | door_long | +1.33% / +1.32% | −0.92% / −0.89% | −2.9% |
  * Lessons on the way (measured, bedrooms): a 2-state gyro-vs-wheel filter (no wheel states) put +3.2% into the gyro
    (truth −0.3%), because the sim's wheels under-report yaw by ~3-5% (soft tyres/scrub); a 3-state filter without ε_d
    still left a 0.002 rad/s bias error (yaw drift 20°) because wheel-diameter mismatch looks like a bias while driving.
* **Dressing**: kitchen, hall, bedrooms, bathrooms, laundry now furnished/textured (tools/realsim/dress_twin_rooms.py,
  all ASSUMED, prefix `twin_` in dressing.json): every solid piece mirrors a MuJoCo twin geom (beds, desk + monitor + chair,
  kitchen table + 4 chairs, plants, wall art/books/cabinet/outlets) so RTX colour agrees with sim depth; plus surface-only
  tile floors (kitchen/laundry/bath zones kept), bath wainscot, hall runner.
* Visual-only methods are still not usable: ORB-SLAM3 RGB-D and RTAB-Map's own RGB-D VO track locally well
  (RPE/1m 0.7-0.8 cm) but lose heading in the in-place spins (max yaw error up to 179°; RTAB VO coverage 0.62).
  The RS2 is pitched 27° down, so most of the image is carpet; at 0.8 rad/s spins the view sweeps ~5°/frame.

## Real log sanity (~/kevin_logs/20261021_213451, no ground truth)
wi_fg (round-2 filter, --no-visual) ends 0.14 m from the start, as round 1 (0.14), kslam 0.21, wheel+gyro odometry 0.20.
Its filter measured the real wheels OVER-reporting yaw by ~6% vs the gyro (ε_w +5.8% ± 1.0%) where the sim's wheels
under-report by 3-5%: the sim's tyre/carpet turning model disagrees with the real robot in sign (open problem).

## Assumed / not measured
* Dressing of the undressed rooms (looks and the twin furniture layout) — not observed in real frames.
* ε_w prior (5%), the wheel factor noise model (spec-based, unchanged from round 1).
* n = 3 = one run per route; seeds differ per route (sensor error draws differ too).
