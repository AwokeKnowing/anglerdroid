# Kevin tire profiles (v310): `stock` | `soft_v3_2`

**Default is `stock`.** Keep it that way until James confirms the KevinFlex v3.2 soft tires are actually fitted. Then switch everywhere at once with one variable.

| | switch |
|---|---|
| robot (anglerdroid) | `export KEVIN_TIRE=soft_v3_2` (alias `soft`) before `main.py` / sims / tools; unset or `stock` = old behaviour |
| twin generators | `KEVIN_TIRE=soft python3 mass_model.py && ... gen_full.py && gen_lite.py && bake_atlas.py && export_lite_gltf.py && export_web.py` (writes `*_soft_v3_2.*`, never touches stock outputs) |
| twin drive app | `python3 kevin_drive.py --model full|lite --tire soft` (or env) |
| viewer | `viewer/index.html?tire=soft` (also `&lite=1`, `&tex=2k|4k`) |
| web sim demo | `simready/index.html?tire=soft` |

Single source of truth: `tire_profiles.json`, byte-identical in `anglerdroid/src/` and `kevin-urdf-twin/simready/`, read by `tire_profile.py` (one in each repo). The viewer and web demo carry a 2-line JS copy (radius, URDF, file suffix); edit them together if the numbers change.

## Profile values

| field | stock | soft_v3_2 | status |
|---|---|---|---|
| loaded / rolling radius (axle height) | 0.08565 m | **0.100 m** | soft NOMINAL: re-measure (roll 10 turns, measure distance / 20π) |
| free radius (sim contact) | 0.08565 | 0.1038 | design (gen_softtire_v3_2.py) |
| robot_config WHEEL_DIAMETER_CM | 17.0 (unchanged tape value) | 20.0 | nominal |
| wheel mass | 2.2 kg | 2.37 kg (+0.17 per wheel, total 12.80 → 13.14 kg) | spec |
| ride height vs stock (chassis + every sensor) | 0 | +14.35 mm | design |
| caster spacers | 0 | 14.35 mm (front caster mount z 0.00375 → −0.01060, gate caster yaw z −0.04595 → −0.06030) | design |
| radial stiffness | rigid | ~13 N/mm (MuJoCo solref 0.0446 1, solimp 0.5 0.5 0.001: 3.8 mm static sink at 49 N/wheel, axle settles at 0.0999 m, bounce ~7 Hz) | calibrated to the 12.5–14 N/mm estimate |
| friction | 0.9 0.005 0.0001 | same (unmeasured) | TODO measure |

## SAFETY-CRITICAL (perception follows the ride height)

RS1 looks straight down; open space = depth beyond `TD_FLOOR_CLIP`. RS1 is 0.865 m above the axle, i.e. 0.951 m above the floor on stock tires, so the 0.91 m clip flags anything ≥ ~4.1 cm. On soft tires the floor is 14.35 mm farther. **Running soft tires with `KEVIN_TIRE=stock` makes obstacles up to ~5.5 cm read as floor**; the reverse makes the floor read as a ~1.4 cm obstacle (immobilise or false positives). Under the profile the clip is 0.92435 and FW_CAM_HEIGHT 0.48221 (+14.35 mm). Verify on the robot after fitting: a flat floor must label CLEAR, and a 5 cm block must label OBSTACLE.

## Inventory (file:line, what depends on wheel/tire, what v310 does)

Legend: **S** = now switchable via the profile, **A** = follows automatically (derives from robot_config / profile), **N** = not switched (reason given).

### anglerdroid (i777 `/home/james/gits/awokeknowing/anglerdroid`, line numbers after the v310 edit)
1. `src/tire_profiles.json`, `src/tire_profile.py`: new, the switch. **S**
2. `src/robot_config.py:14-21` WHEEL_DIAMETER_CM 17.0 → profile override (20.0 soft); also exports TIRE_PROFILE and RIDE_DZ_M. **S**
3. `src/robot_config.py` WHEEL_RADIUS_M / WHEEL_DIAMETER: derived (0.085 → 0.100). **A**
4. `src/robot_config.py` WHEEL_HALFTRACK_CM / WHEELBASE_M 0.354: not tire-dependent (hub spacing); tire centre track 0.362 unchanged. **N (n/a)**
5. `src/robot_config.py` WHEEL_LEN_CM 20 / WHEEL_WID_CM 10 self-mask box: 20 cm covers the 20.8 cm soft tire with no margin. **N**, consider 22 cm once fitted.
6. `src/wheelbase.py:26,56,66,441-442,528`: m/s ↔ turn/s and odometry circumference from WHEEL_DIAMETER_CM. **A**
7. `src/wheelbase.py:5`: docstring "17.13 cm". **N** (comment)
8. `src/odrivecan.py:28-29` DEFAULT_VEL_RAMP_RATE 3 turn/s², DEFAULT_VEL_LIMIT 4 turn/s: in turns, so on soft tires they become 1.885 m/s² (was 1.61) and 2.51 m/s (was 2.14). **N**, deliberately left in turns (no flash/behaviour change); scale them by 0.8565 if you want identical m/s.
9. `src/s1-odrive-config.py:27,32` vel_limit 10, vel_ramp_rate 20 (on-board config, turns). **N** (flash-time; same note).
10. `src/safety.py:28` DECEL_MPS2 = ramp·2π·WHEEL_RADIUS_M: 1.602 → 1.885 m/s² (stopping distance model follows the real wheel). **A**
11. `src/torque_tools.py:27,59-60,286-337`: torque ↔ force with WHEEL_RADIUS_M. **A**
12. `src/pose.py:81` PoseEstimator(wheel_radius_m): fed from robot_config by vision.py:795/801. **A**
13. `src/vision.py:73-77` TD_FLOOR_CLIP 0.91 → 0.91 + RIDE_DZ_M. **S, SAFETY**
14. `src/vision.py:100` FW_CAM_HEIGHT 0.467863 → + RIDE_DZ_M (RS2 phys_h floor at 0). **S, SAFETY**
15. `src/vision.py` FW_FLOOR_CLIP 0.15 (forward range, not height), FW_HEIGHT_CLIP 1.30 (above floor via phys_h). **N (n/a)**
16. `src/vision.py:~1543` runtime pitch/height auto-fit (new_cam_h): starts from FW_CAM_HEIGHT. **A**
17. `src/perception/ego_rs1.py:~27` TD_FLOOR_CLIP copy. **S, SAFETY**
18. `src/perception/fast_ego80.py:~42` TD_FLOOR_CLIP copy. **S, SAFETY**
19. `src/perception/fast_ego80.py:~52` FW_CAM_H copy. **S, SAFETY**
20. `src/perception/ego_rs1_fast.py`, `src/gpu_render.py` floor_clip_m: passed in from vision. **A**
21. `src/cuvslam_tracker.py:29` CAMERA_HEIGHT_M 0.97 → + RIDE_DZ_M. **S**
22. `src/wheel_imu_prior.py:15`: docstring example only (0.0762). **N** (comment)
23. `src/imu.py`: no height or wheel constant; IMU height rises 14.35 mm, no code impact. **N (n/a)**
24. `src/orbslam3_ship.py:67`, `src/perception/color_uv_alignment.py`: camera-to-camera extrinsics from the RealSense, independent of the floor. **N (n/a)**
25. `sim/rl_nav/ego80.py:~36,~46` TD_FLOOR_CLIP and FW_CAM_H (bag-matched labeler): must match the tires the bag was recorded with. **S, SAFETY**
26. `sim/rl_nav/calibrate_rs2_pitch.py:26-30` imports FW_CAM_H. **A**
27. `sim/rl_nav/throttle_zones.py:66-81`: wheel radius from robot_config. **A**
28. `sim/rl_nav/kernels.py:63` TRACK_M 0.354, `test_safety_match.py:137`. **N (n/a)**
29. `sim/rl_nav/env.py` / Warp kernels: 2-D ego grids, no wheel radius or camera height. **N (n/a)**
30. `sim/dynamics.py:17` WHEEL_RADIUS_M, WHEELBASE_M. **A**
31. `sim/urdf_twin_fk.py:17-23` DEFAULT_URDF → profile twin URDF (`sim/kevin-urdf-twin/urdf/kevin_soft_v3_2.urdf` added). Wheel radius and camera poses then come from the URDF. **S**
32. `sim/newton_kevin.py:197-201`, `sim/newton_drive.py:2659-2770`: wheel_r and spawn height from urdf_twin_fk; RS1/RS2 poses from FK (line 1825). **A**
33. `sim/newton_drive.py:78` RS1_ORIGIN_Z 0.865 (axle frame): unchanged by the tire. **N (n/a)**; it is overridden from FK when the URDF twin is used.
34. `sim/newton_drive.py:325,470` wheel_r=0.09 defaults; `sim/newton_kevin.py:25-29`, `sim/kevin_wander_220.py:25-29`, `sim/_padfix_offline_run.py:25-29` WHEEL_R 0.09 / MAST_CAM_Z 0.90: legacy box robot. **N** (approximate box model; not the twin)
35. `robot/generate_urdf.py:13-34` legacy URDF: R 0.100 for soft (wheel visual, caster, footprint); body stays fixed to the axle. **S**
36. `robot/anglerdroid.urdf`: generated output (stock; regenerate with KEVIN_TIRE=soft to get the soft one). **N** (generated file not rewritten)
37. `src/globalmap.py`, `src/keepouts.py`, `src/insect_planner.py`: footprint in cm from robot_config, no wheel radius or height. **N (n/a)**
38. tests (`test_ego_*`, `test_soft_low_obstacle*.py:15` TD_FLOOR_CLIP 0.91 copy, `test_wheel_imu_prior.py`, `test_torque_tools.py`): stock fixtures. **N**; run them with the default.

### kevin-urdf-twin (box `/workspace/kevin-urdf-twin`, deployed to i777 as v310)
39. `simready/tire_profiles.json`, `simready/tire_profile.py`: new switch (mirror). **S**
40. `urdf/kevin.urdf`: restored to stock, byte-identical to v307 (v308 had edited it in place to soft). **S**
41. `urdf/kevin_soft_v3_2.urdf`: the v308 soft version (soft tire visual, collision r 0.100, mass 2.37, both caster spacers). **S**
42. `simready/kevinlib.py:28` URDF() default → profile URDF. **S**
43. `simready/simbodies.py:15,26-27` POS caster bodies −caster_spacer; `:62` WHEEL_R (kinematics, cap_tps 2.787 → 2.387 turn/s for 1.5 m/s); body_inertials path. **S**
44. `simready/mass_model.py:~121-125` soft tire element 0.17 kg per wheel (mesh shape); output mass_properties_soft_v3_2.json. **S**
45. `simready/gen_full.py`: contact r = free radius, tire solref/solimp/friction, floor offset, keyframe, assets/full_soft_v3_2/, kevin_full(_vel)_soft_v3_2.xml, urdf/kevin_sim_soft_v3_2.urdf. **S**
46. `simready/gen_lite.py`: same, plus assets/lite_soft_v3_2/, kevin_lite_soft_v3_2.xml, urdf/kevin_lite_soft_v3_2.urdf, lite_parts/skin_tessellation _soft_v3_2.json. **S**
47. `simready/lite_parts.py:241`, `bake_atlas.py:15` (soft atlas re-baked), `export_lite_gltf.py` (1k/2k/4k + kevin_lite_{2k,4k}_soft_v3_2.xml), `export_web.py` (web/kevin_lite_physics_soft_v3_2.xml, kevin_lite_bodies_soft_v3_2{,_1k,_2k}.glb). **S**
48. `simready/kevin_drive.py:27-29,69` --tire flag; controller radius, caps and model path follow. **S**
49. `simready/index.html:70-75,127,137` web demo ?tire=: SW_R, wheel caps, physics XML, bodies GLB. **S**
50. `viewer/index.html:319-322,709,~791` ?tire=: ground −R, URDF, lite GLB, status text. **S**
51. `simready/validate.py`, `compare_models.py`, `compare_render.py`, `lite_stats.py`, `gen_sim_doc.py`, `gen_mass_doc.py`, `gen_bom.py`: stock-only reports (hard-coded 0.08565 floor in the mass doc text). **N**; the soft numbers are in this doc.
52. the kplan 2-D planner sim: no wheel radius or height inputs (footprint only). **N (n/a)**

**Count: 52 entries: 34 switchable (S) or auto-following (A), 18 not switched (N: n/a, legacy, comments, or flagged below).**

## Not made switchable (decide when the tires are on)
- ODrive limits are in turn/s (`odrivecan.py:28-29`, `s1-odrive-config.py`): soft tires give +16.8 % m/s and m/s². They were not rescaled, to avoid any behaviour or flash change.
- Self-mask wheel box (`robot_config` WHEEL_LEN_CM 20) vs the 20.8 cm soft tire.
- Legacy Newton box robots (WHEEL_R 0.09, MAST_CAM_Z 0.90) and test fixtures.
- Friction is unmeasured, so the stock value is kept for soft.
- The soft radius, ride height and stiffness are design values. Measure them after fitting and edit only `tire_profiles.json` (both copies) plus the 2 JS tables.

## Verification (v310)
- Stock outputs regenerated through the new code are byte-identical (140 files: MJCF, URDF, JSON, OBJ, GLB). Robot stock values are identical (0.91, 0.467863258422, 17.0 cm, 0.97).
- Soft: 13.140 kg; axle settles at 0.09987 m; rest pitch −0.03°; gate caster on the floor at q=0. Demo and brake runs pass in full and lite.
- Viewer and web demo load with and without ?tire=soft, with no errors and no 404s.
