# lineloc — line-map localisation of Kevin's mast camera against the casita model

2D pose (x, y, yaw) of base_link in the sim/casita world frame (STL normalised so its bbox min is at the origin, the same frame as
`emu.envs.env_casita` and maploc). It runs as an iterated-EKF update on top of wheel odometry, which stays the primary source.

## Per-frame pipeline (fixed cost: capped segments, fixed iteration counts, no periodic work)
1. **Segments.** LSD (`cv2.createLineSegmentDetector`, internal scale 0.5), keep the 200 longest that are over 20 px. Undistort
   only the endpoints (Brown k1/k2), not the whole image.
2. **Gravity classification.** Rays go into the gravity-levelled base frame using the known camera tilt (`Camera.tilt(pitch, roll)`;
   0 in the twin; 2.16° nose-down / −3.22° on the real mast; the RS2 would use its IMU). A 3-D vertical's interpretation plane
   contains the up axis, so a segment is vertical when |n_z| < sin 1.5°. Collinear pieces are merged into one bearing plus an
   elevation span.
3. **Map** (`mapbuild.py`, offline, about 25 s). The STL's crease and boundary edges give 207 vertical edges (corners, as
   x, y, z0, z1) and 342 horizontal edges. A 10 cm visibility grid at the mast height (Embree rays) makes the per-frame
   candidate lookup O(1).
4. **Association** in a small window around the odometry prediction (chi² 99 % gate from the prior covariance):
   - verticals → corners by bearing, plus height-span compatibility at the predicted range;
   - then, in the posterior window, the non-vertical segments ≥ 40 px → wall–floor (z = 0) and wall–ceiling (z = top)
     junction lines, using image point-to-line distance, a 3° angle gate and an overlap check.
5. **Minimal solver.** A linear closed-form 3-bearing resection (`LineLoc.resect`), used in an MSAC with fixed draws for
   relocalisation (`ransac=` / `n_lost=`). It is off by default; see the results for why.
6. **Robust Gauss–Newton** (Huber IRLS) with the odometry prior. Per-residual sd = sqrt(σ_px² + (f·σ_map/range)²) with
   σ_px = 1.5 px and σ_map = 2 cm (map-model error). The posterior covariance is the EKF update.
7. **Acceptance.** At least 2 vertical inliers, at least 3 inliers in total, and an update NIS within chi²(3) 99 %. Otherwise
   the odometry prediction is kept.
8. **Keyframes (fusion layer).** Update only after ≥ 2 cm / 1° of motion or 1 s. Identical views add no information about a
   static map error, so this prevents covariance collapse while standing still.

Odometry noise is maploc's fitted model (`Q(d)`, variance linear in path and rotation).

## Files
- `lineloc.py`: Camera, LineMap, LineLoc (detector, classification, association, resection, MSAC, GN, gating).
- `mapbuild.py`: STL → `casita_lines.npz` (corners, horizontal edges, visibility grid). Defaults (2026-10-21): the resolved
  world mesh (`LL_STL`, else `src/world_params.py` <- `worlds/casita/world.json`, currently `sim/casitahouse31-walls-laser-zfix.stl`)
  and the selected wheels' mast camera height (`src/robot_params.py` `mast_rgb_height_m`, soft_v3_2 0.99115; was a fixed stock 0.97615).
  `casita_lines.npz` is gitignored: rebuild with `python3 mapbuild.py` after a model change.
- `synth.py`: Lambert-shaded ray-cast renders of the model through the twin lens from known poses.
- `test_lineloc.py`: resection exactness, undistort round trip, vertical classification under tilt, synthetic poses (exact map:
  median < 1 cm), and a smoke test of the defaults. Run with `python3 -m pytest -q`.
- `bench.py`: replay of the sim logs (odometry EKF + lineloc), errors vs GT. Configure with `LL_*` env vars. The camera height is
  per log (mast pos + the log's tyre axle height; was a fixed stock +0.08565, 1.5 cm low for soft-tyre logs).
- `poselib_cmp.py`: per-frame comparison against PoseLib PnPL (6-DoF, BSD-3; the line solver LIMAP uses).
- `real_check.py`, `orin_timing.py`: read-only real-robot check and Orin CPU timing.
- `diag*.py`, `ovl.py`, `dbg*.py`: diagnostics used during development.

## Results (sim logs with GT, twin lens, stock-tyre mast height 0.976 m)
Runs were repeated 3× and are deterministic (identical numbers). Timings are on the box (Xeon, Python, 1 process). Every
method was started from GT[0] with the same odometry.
| log (frames, path) | method | localized | pos med / p95 / max (cm) | yaw med / p95 (deg) | ms/frame med / p95 / max (box) |
|---|---|---|---|---|---|
| L19 (450, 5.2 m) | odometry only | – | 3.51 / 3.96 / 5.33 | 0.79 / 1.15 | – |
| | maploc (edge tracking, existing) | 70% | 0.70 / 2.17 / 3.56 | 0.11 / 0.36 | 84 / 165(p90) / – |
| | lineloc V (bearings only) | 97% | 1.16 / 3.85 / 6.37 | 0.19 / 1.42 | 8.0 / 9.8 / 15 |
| | lineloc VH (bearings + floor/ceiling lines) | 93% | 1.52 / 4.49 / 6.69 | 0.16 / 1.12 | 9.9 / 12.2 / 23 |
| L20 (619, 5.1 m) | odometry only | – | 2.37 / 5.97 / 6.02 | 1.32 / 1.99 | – |
| | maploc (edge tracking, existing) | 67% | 0.96 / 3.69 / 4.21 | 0.14 / 0.57 | 82 / 123(p90) / – |
| | lineloc V (bearings only) | 99% | 0.94 / 2.71 / 3.42 | 0.20 / 0.40 | 6.7 / 8.9 / 15 |
| | lineloc VH (bearings + floor/ceiling lines) | 98% | 1.55 / 6.77 / 9.80 | 0.18 / 0.63 | 9.5 / 12.4 / 43 |
| L12 (1243, 24.5 m) | odometry only | – | 24.77 / 50.44 / 51.20 | 1.61 / 3.76 | – |
| | maploc (edge tracking, existing) | 51% | 1.03 / 14.08 / 24.22 | 0.22 / 2.12 | 41 / 95(p90) / – |
| | lineloc V (bearings only) | 86% | 23.55 / 43.71 / 60.21 | 2.31 / 6.34 | 5.8 / 7.2 / 66 |
| | lineloc VH (bearings + floor/ceiling lines) | 91% | 1.97 / 24.54 / 32.14 | 0.22 / 1.86 | 7.8 / 9.7 / 54 |
| L13 (1223, 23.4 m) | odometry only | – | 19.74 / 34.17 / 35.62 | 3.22 / 5.42 | – |
| | maploc (edge tracking, existing) | 46% | 1.27 / 4.88 / 10.70 | 0.24 / 2.24 | 55 / 114(p90) / – |
| | lineloc V (bearings only) | 96% | 1.04 / 4.36 / 8.32 | 0.14 / 0.56 | 6.1 / 7.1 / 16 |
| | lineloc VH (bearings + floor/ceiling lines) | 96% | 1.47 / 4.84 / 8.26 | 0.17 / 0.71 | 8.6 / 10.4 / 59 |

"localized" means accepted updates out of attempted updates. maploc's timings were taken while other jobs were loading the box.
L18 has no odometry (NaN) and was skipped. L12 contains a bump at t ≈ 9.7 s: odometry reads −0.6 cm/frame while GT reads
+4.7 cm, so the robot is ~20 cm off along-track within 0.4 s.

**Per-frame solver comparison** (L19 + L20 + L13 every 3rd frame, prior = GT + N(3 cm, 1°), same detections and associations):
lineloc 2.61 cm median / 6.64 p95, yaw 0.30° / 1.28°; PoseLib PnPL (6-DoF) 2.91 / 28.3 cm, yaw p95 180° (flips);
the prior alone 3.49 / 7.40 cm.

**Orin NX CPU** (13:50 PDT, nomotor HAL running, load average 5–7, `nice 19`, 1 thread, real 640×480 frames):
LSD@0.5 15.8 ms; lineloc V 26 ms; VH 37 ms. With LSD@0.8: V 46 ms, VH 59 ms.

**Bench defaults (2026-10-21).** `bench.py` now passes only the `LL_*` variables that are set, so it runs LineLoc's own
defaults (the table above) with keyframes on (`lineloc.KF_*`; `LL_KF=0` disables).  It used to carry its own defaults
(LSD 0.8, all horizontal edges, σ_map 0, MSAC relocalisation after 10 rejects, no keyframes): L20 3.31 / 60.4 cm instead of
1.55 / 6.77, from covariance collapse while turning in place (sd 3 mm, then 7-12 cm off fixes pass the NIS gate) and a false
MSAC relocalisation (60 cm).  L19's 18 cm at the end with the new map was the same configuration.

**zfix map (casitahouse31-walls-laser-zfix, camera 0.99115 m), deployed defaults, 3 runs each (identical).**  L23-L25 are
fresh soft-tyre clips in the zfix world (counters off); L19/L20 were recorded in the older world, so the old map is their match.
| log | map | localized | pos med / p95 / max (cm) | odometry med (cm) |
|---|---|---|---|---|
| L19 | old / new | 93% / 92% | 1.52 / 4.49 / 6.69  vs  1.86 / 7.19 / 10.97 | 3.51 |
| L20 | old / new | 98% / 96% | 1.55 / 6.77 / 9.80  vs  1.28 / 4.15 / 6.29 | 2.37 |
| L23_zfix | old / new | 91% / 93% | 4.01 / 17.96 / 31.47  vs  2.38 / 7.11 / 10.04 | 25.2 |
| L24_zfix | old / new | 86% / 91% | 3.84 / 8.25 / 16.36  vs  2.61 / 6.66 / 9.92 | 23.9 |
| L25_zfix | old / new | 95% / 95% | 2.34 / 23.01 / 34.70  vs  2.15 / 5.75 / 10.40 | 20.5 |

`test_synthetic_defaults_smoke` fails with the zfix map (frame 264: a 3.3σ prior draw, 16.7 cm off, gives an accepted
9.2 cm fix from only 4 verticals, posterior sd 3.2 cm, Mahalanobis² 17.5).  The limit was left at 7 cm: the fix is
overconfident, which is a lineloc weakness, not a test problem.

## What was learned
- **Vertical bearings alone** are precise when correctly associated (sub-pixel residuals at GT) and give ≈ 1 cm median
  tracking. Range along the viewing direction is weakly observed, though: posterior sd is 2–18 cm along-track in some frames.
  After an along-track odometry fault (the L12 bump) bearings cannot pull the pose back, and tracking stays ~20–60 cm off.
- **Floor/ceiling junction lines** fix that (L12: 23.6 → 2.0 cm median). They cost ~0.5 cm median elsewhere. Some walls in the
  sim render sit ~2.4 px (a few cm) off the walls STL (e.g. the x = 0.12 m wall), and floor tiles and rugs make parallel
  distractors.
- **Relocalisation by MSAC** over 3-bearing resections in a wide window (25 cm / 5°) produced false fixes in sim (L20 max
  54 cm) and on the real frame (a −21 cm / −4° mode). The casita map is not distinctive enough in bearings alone, so it is
  off by default.
- **Unrealistic σ_px (3–5 px)** made results worse. Map-model error is handled by σ_map, not by inflating σ_px.
