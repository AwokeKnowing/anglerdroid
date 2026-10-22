# nav/maploc: mast-camera localisation against the known casita map (baseline, sim-first)

The map is the twin's wall STL (the resolved world mesh: `src/world_params.py` <- `worlds/casita/world.json`, currently sim/casitahouse31-walls-laser-zfix.stl; env `ML_STL` overrides) plus floor (z 0) and ceiling (wall tops + 1 cm, derived from the mesh: 2.1997 for zfix, 2.29 for the old casitahouse-walls-fixed.stl) planes. It has no furniture.
The state is (x, y, yaw) plus, optionally, ln(global map scale). Wheel odometry is the prediction and the metric reference.
Only the forward mast RGB camera is used.

## Method (v4, default `LocalizerSeg`; established model-based edge tracking, RAPiD / Drummond-Cipolla, point-to-line)
- Map edges are the wall mesh's own crease edges (>30 deg) and open-boundary edges, sampled every 2 cm. They are exact 3D, so sub-pixel at any resolution.
  Each frame keeps the points that are visible (an Embree ray reaches the point) and that appear as an image discontinuity: probe rays 2 px either side show a normal change >30 deg or a depth jump >5 %.
- Image edges are LSD segments (von Gioi et al.) with sub-pixel endpoints. They come from the full-resolution 640x480 frame after undistorting with the lens in info.
- The residual is the signed distance to the nearest segment within +-22.5 deg of orientation. It is re-associated every Gauss-Newton iteration.
- The fit is robust (Huber IRLS) Gauss-Newton with the odometry prior in information form.
  When the prior yaw sd is larger than the association window, a robust-cost yaw grid within +-3 sigma seeds the solve. This happens after blur-skipped spins with a turn-scale error.
- A fix is accepted only if it passes a chi2 99 % Mahalanobis gate and the inlier fraction does not drop. No fix is attempted while yaw-rate motion blur exceeds the Huber scale.
- Global map scale is constant with no process noise, a 5 % prior, and a +-2 %/iteration trust region. It is estimated jointly over the whole run.
- Noise models are fitted, not hand-set:
  - odometry variance is proportional to path and turn, fitted by NNLS on 1 s increments vs GT;
  - fix information is scaled so the fix NEES at GT matches chi2(3), leave-one-out (`calib.py`).
- v3 (`Localizer`: pixel render + directional chamfer DT, half resolution) is kept for comparison (`ML_METHOD=dt`).

## Lens and mount (the twin is the source of truth)
- The mast lens is the twin truth lens: DFOV 118 deg, f 240.34 px, k1 -0.05, k2 0.008, from `emu/realism.mast_truth`.
  `emu/world.py` now writes it into info["mast"] (`intr` + `dist`), and `emu/tools/slamlog/log.py: Log.mast_lens()` reads it.
  For logs recorded before this change, `Log.mast_lens()` falls back to the truth lens the sim rendered with. `export.py` writes it to mlx/mast.json.
- The camera is at base_link + twin `frames.rgb.xyz`, with base_link at the axle = the tyre profile's loaded wheel radius
  (`src/tire_profiles.json`): stock 0.08565 m -> camera 0.976 m; soft_v3_2 0.10065 m -> camera 0.991 m (soft tyres, 7fcec63).
  Picked per log: `export.py` copies the log's info["tire"] into poses.npz; a log (or older export) without it predates the
  soft tyres and uses stock (`harness.log_tire / log_axle / tbc(n)`, `make_loc(n)`). `ML_BASE_Z` overrides the base height.
  Live use (`maploc.AXLE_H`) follows the active profile (KEVIN_TIRE, else the json default).
  v1-v3 subtracted the axle height twice, so the camera sat 8.6 cm too low. That was the main cause of their ~-2.5 % scale and forward bias.
- **The real bot still needs a one-time checkerboard calibration of the mast camera:** intrinsics + distortion, and the mount pitch/height.
  In sim, scale is sensitive to the mount: 0.2 deg of pitch moves the scale ~0.8 %, and 5 mm of height moves it ~0.5 %.
  The sim body sits 3-6 mm low (tyre sag) with ~-0.2 deg static pitch (nose up). Feeding those values in removes the residual -1 % bias at GT (`scalecam.py`: 0.996 +- 0.004).

## Results (sim, logs L12/L13/L14, casita, full-res, fixes every 3rd mast frame unless noted)
Position error med/p90/final in m, in the map frame. Clean odometry (sim odometry is unrealistically good):

| log | odom | Slam2 (live) | maploc v4 | maploc 10 Hz |
|---|---|---|---|---|
| L12 | 0.248/0.481/0.509 | 0.336/0.551/0.571 | 0.015/0.119/0.013 | 0.012/0.112/0.010 |
| L13 | 0.197/0.336/0.280 | 0.169/0.255/0.220 | 0.016/0.047/0.052 | 0.014/0.044/0.026 |
| L14 | 0.087/0.272/0.272 | 0.072/0.261/0.303 | 0.023/0.208/0.216 | 0.021/0.190/0.215 |

Injected odometry error (wheel scale, turn scale), maploc position median/p90 (odom median/p90):

| error | L12 | L13 | L14 |
|---|---|---|---|
| +2 %, +3 % | 0.015/0.116 (0.29/1.08) | 0.020/0.054 (0.64/1.05) | 0.029/0.191 (0.43/1.13) |
| -2 %, -3 % | 0.014/0.104 (0.32/1.02) | 0.018/0.051 (0.36/0.61) | 0.024/0.215 (0.85/1.30) |
| +3 %, -4 % | 0.016/0.107 (0.16/1.07) | 0.018/0.053 (0.53/0.95) | 0.022/0.216 (1.14/1.77) |

With the yaw grid, yaw p90 under these errors is 2.7-7.4 deg; spins are skipped for blur. Without the grid, turn-scale error after spins drove yaw p90 to 12-42 deg and lost track.
Slam2 is the live run with clean odometry. It could not be re-run offline under the injected error.

Global map scale (truth 1.0 for the clean map, 0.9709 for the map built 3 % too large):
- clean: -1.17 / -0.22 / +0.39 %;
- x1.03 map: -1.10 / -0.20 / +0.63 %;
- mean over logs: -0.33 % and -0.22 %.
The filter sd (0.05-0.09 %) is over-confident because fixes are correlated and the mount pose is not modelled. The honest per-log uncertainty is the between-log sd, ~0.8 %.
This is consistent with the mount sensitivity above. A static-pitch state (`ML_PITCH_EST=1`) was tried and is not identifiable enough: it estimated -0.45/-0.37/+0.52 deg against a truth of about -0.2 deg.

Runtime: 52 ms median and 114 ms p90 per fix, single process on the box CPU (x86) at full res. That is 10 Hz-capable on one core. The Orin's ARM cores were not measured.

## Run
The working directory must hold `mlx/` and the frames from `python -m nav.maploc.export <log> <work>`.
  python calib.py L12 L13 L14 (ML_CALF=calib.json)
  ML_EVERY=3 ML_SCALE=1.0 ML_OUT=res python run.py L12 L13 L14
  python summ.py res/summary.json
Options:
- ML_ODOM_ERR=ts,ys: injected wheel-scale and turn-scale error;
- ML_MAP_DISTORT: wrong-scale map test;
- ML_METHOD=dt: v3;
- ML_DOWN: downsample factor;
- ML_BASE_Z, ML_PITCH_DEG: mount;
- ML_PITCH_EST=1;
- ML_ZMIN.
Diagnostics:
- gtscale.py: single-frame fixes at GT;
- scalecam.py: range/scale bias at GT, with scale about the floor point below the camera;
- edgebias.py: signed edge residual by class.
Needs trimesh + embreex (ray casting) and opencv (LSD). These are not in the robot venvs; use a separate venv.
