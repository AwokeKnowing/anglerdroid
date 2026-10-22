# casita_fit: mast RGB <-> casita wall model (pose, tilt, lens check)

Fits `sim/casitahouse-walls-fixed.stl` (same frame as `emu/envs.env_casita`: bbox min at the origin, x east,
y north, z up, wall tops z = 2.28) to one upright mast frame by **line reprojection**:

* image lines: OpenCV LSD (`LSD_REFINE_STD`), hand-matched to model 3D edges in `annot_*.json`
  (each entry: model line A-B, image segments, `role` = `fit` or held-out `check`; rejected candidates are listed in `_doc`);
* unknowns: camera pose (x, y, z, yaw, pitch + nose-down, roll) + lens (f, k1 [k2, cx, cy]); Brown radial model as in `emu.realism.mast_truth`;
* residual: every ~15 px sample of a detected segment -> 2D px offset to the projected (lens-aware) model line;
  one robust loss (soft-L1, 1.5 px), no per-line weights;
* model-free check: vanishing point of the image verticals -> pitch/roll (`vp.py`);
* scale: model scale and camera height are one similarity DoF, so the fit reports the camera height in model units;
  stage E fixes it at the resolved `mast_rgb_height_m`, stage F/G free a vertical-only model scale `sz`.

```
PY=.venv-newton/bin/python   # cv2 + scipy + trimesh (embree)
$PY tools/casita_fit/fit.py --image ~/kevin_bench/casita/mast_1600x1200.png --annot tools/casita_fit/annot_1213.json \
    --stl sim/casitahouse-walls-fixed.stl --out ~/kevin_bench/casita/out
$PY tools/casita_fit/overlay.py --image ~/kevin_bench/casita/mast_1600x1200.png --annot tools/casita_fit/annot_1213.json \
    --stl sim/casitahouse-walls-fixed.stl --report ~/kevin_bench/casita/out/fit_report.json --stage A --out ~/kevin_bench/casita/out/overlay.png
```
`overlay.png`: model render at 20 % + RED model key lines (dashed = held-out) + BLUE matched image segments (CYAN = held-out);
`overlay_lines.png`: the same without the fill.

## Result 2026-10-21 12:13 PDT (Kevin parked, odom 0,0; 1600x1200 MJPG = same FOV as 640x480, checked by ORB: scale 1.0001)

| stage | free | rms / med px (204 samples) | x, y (m) | z (model m) | yaw | pitch (down) | roll | f px @1600 | k1 |
|---|---|---|---|---|---|---|---|---|---|
| A | pose6 + f + k1 | 2.93 / 1.08 | 3.81, 1.44 | 1.10 | 95.8 | 2.16 | -3.22 | 492 | -0.026 |
| B | + k2 | 2.90 / 0.79 | 3.82, 1.40 | 1.11 | 95.9 | 2.34 | -3.17 | 496 | -0.010 (k2 -0.005) |
| C | + k2 + cx, cy | 2.84 / 0.81 | 3.76, 1.30 | 1.06 | 95.7 | 2.76 | -3.07 | 511 | 0.007 (pp 789.5, 613.6) |
| D | pose6, twin lens fixed | 7.80 / 4.12 | 3.82, 1.07 | 1.09 | 96.5 | 1.42 | -2.78 | 600.9 (twin) | -0.05 |
| E | z fixed 0.991 | 4.98 / 2.42 | 3.79, 1.40 | 0.991 | 95.5 | 0.01 | -3.67 | 502 | -0.026 |
| F | A + vertical scale sz = 0.980 | 3.03 / 1.05 | 3.82, 1.41 | 1.05 | 96.0 | 2.04 | -3.20 | 498 | -0.025 |

Vertical vanishing point, no model (27 segments, 1.1 deg rms): **pitch 2.24 deg nose-down, roll -3.30 deg**
(-3.31 / 2.28 with the twin lens: insensitive to the lens). Roll < 0 = camera right side up / image verticals lean
top-right; twin nominal mount is pitch 0, roll 0.

Reading it:
* tilt: pitch ~2.2 deg nose-down and roll ~-3.2 deg, agreed by the fit (A-F, model-based) and the VP (model-free).
* lens: every free-lens fit wants f ~ 490-510 px @1600 (196-204 @640, HFOV ~116 deg) with weak barrel (k1 -0.01..-0.03);
  the twin truth (DFOV 118 -> f 600.9 @1600 / 240.3 @640, k1 -0.05, k2 0.008) leaves 2.7x the residual with structured
  errors. f and distance trade off (dolly zoom: D moves the camera 0.35 m back), so treat f as 500 +- 20 px until a checkerboard.
* height / scale: the camera sits 1.05-1.11 model-m above the model floor vs 0.991 m measured, mostly set by the
  ceiling lines (camera-to-wall-top 1.18 model-m in A and F). Either the model's 2.28 m wall tops are ~10 cm higher than
  the real ceiling (real ~2.17 m) or the plan is ~6-10 % large. Only one short floor line is visible (cluttered floor),
  so this is not resolved here: a tape measure of the living-room ceiling height decides it.
* model differences seen in held-out lines: front door edges 25-47 px off (door position in the STL differs or is not
  modelled), partition end 5-7 px (end ~10 cm shallower in reality), counter heights.
