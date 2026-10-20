# nav/maploc: mast-camera localisation against the known casita map (baseline, sim-first)

The map is the twin's wall STL (sim/casitahouse-walls-fixed.stl) plus floor (z 0) and ceiling (z 2.29) planes. It has no furniture.
Pose is (x, y, yaw) and, optionally, ln(global map scale). Wheel odometry is the prediction. Each mast frame is undistorted
with the lens calibration, its structural edges are rendered from the map by Embree ray casting (creases over 30 deg plus depth jumps), and the
pose is fitted to LSD line segments with a directional chamfer distance transform (8 orientation bins, +-1 bin). The fit is robust
(Huber) Gauss-Newton with the odometry prior. A fix is accepted only if it passes a chi2 99% Mahalanobis gate and the inlier
fraction does not drop. No fix is attempted while yaw-rate motion blur exceeds the Huber scale.

Noise models are fitted, not hand-set:
- odometry variance is proportional to path and turn, fitted by NNLS on 1 s increments against GT;
- fix information is scaled so that the fix NEES at GT matches chi2(3). This is leave-one-out across logs (`calib.py` -> `calib.json`).

Run (cwd = a work dir holding `mlx/` and the mast frames from `export.py`, with `mast_truth.json` copied to `mlx/`):
  python calib.py L12 L13 L14; ML_EVERY=3 [ML_SCALE=1.0] ML_OUT=res python run.py L12 L13 L14; python summ.py res
Options:
- ML_SCALE: estimate the global map scale, starting from this value;
- ML_MAP_DISTORT: build a deliberately wrong-scale map (a test);
- ML_ODOM_ERR=ts,ys: synthetic odometry miscalibration;
- ML_LENS=belief: use info.json's 100 deg pinhole instead of the truth lens;
- ML_ZMIN: use only map edges above this height.
Needs trimesh + embreex (ray casting) and opencv (LSD). These are not in the robot venvs; use a separate venv.
