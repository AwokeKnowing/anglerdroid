# RealSense D435 / D435i optical layout (Kevin twin)

## Sources
- Intel RealSense D400 Series Datasheet (Rev 015, March 2023), Tables 3-52, 4-16, 4-19
- Official `realsense2_description` `_d435.urdf.xacro` nominal extrinsics
- Figure 10-9 mechanical drawing (same housing for D435 and D435i)

## Housing
| Dim | Nominal |
|-----|---------|
| Width (Y) | 90.00 mm |
| Height (Z) | 25.00 mm |
| Depth (X) | 25.00 mm |
| Mass | 75 g (datasheet) |

D435i shares the same outer housing; difference is internal BMI055/BMI085 IMU. Twin marks D435i with a small top-rear badge.

## Mesh frame
Origin at geometric center of the pill. **+X forward** (out front glass), **+Y left**, **+Z up** (REP-103 when camera faces robot +X).

Front face at `x = +12.500 mm`.

## Optical aperture centers (vs mesh origin)
Depth / left-IR is 17.5 mm from 1/4-20 centerline (Table 4-19). 1/4-20 is on the width centerline ⇒ left IR at `y = +17.500 mm`.

| Aperture | Y (m) | Y (mm) | Basis |
|----------|-------|--------|-------|
| Left IR (infra1 / depth) | 0.017500 | 17.500 | Table 4-19 |
| Right IR (infra2) | -0.032500 | -32.500 | 50 mm baseline (module D430 / ROS) |
| RGB | 0.032500 | 32.500 | ROS `d435_cam_depth_to_color_offset` = +15 mm from left IR |
| IR projector | -0.007500 | -7.500 | Midpoint of stereo baseline (projector between imagers; no separate datasheet X-Y table) |

All apertures at `z = 0` (horizontal midplane of 25 mm height).

Depth optical center lies **4.2 mm behind front glass** (Table 4-16). Aperture cylinders are front-face cosmetics, not the optical center.

## Aperture cylinders
Each cylinder axis = +X. Length 2.0 mm; tip proud of front face by **0.5 mm**.

| | radius | tip x | center x |
|--|--------|-------|----------|
| IR L/R | 2.50 mm | 0.013000 | 0.012000 |
| RGB | 2.50 mm | 0.013000 | 0.012000 |
| Projector | 1.60 mm | 0.013000 | 0.012000 |

Lens opening diameters are **not** tabulated in the datasheet; Ø5.0 mm (IR/RGB) and Ø3.2 mm (projector) match the visible front openings on the peripheral camera.

## Housing loft stations (stadium)
From official `d435.dae` envelope (mesh frame: +X forward):

| Station | X (mm) | Width × Height |
|---------|--------|----------------|
| Back face | −12.50 | 77.4 × 19.2 |
| Back edge loop | −8.00 | 90.0 × 25.0 |
| Front bevel start | +0.50 | 90.0 × 25.0 |
| Front face | +12.50 | 84.5 × 19.5 |

66 verts (≤80). AABB stays 90 × 25 × 25 mm.

## Rear M3 mounting holes
Figure 10-9 / DAE: two M3 points, **45 mm** center-to-center, centered on back, mid-height. Max insert 3 mm.

| Hole | Y (m) | Cylinder |
|------|-------|----------|
| Left (+Y) | +0.0225 | Ø3.0 × 2.0 mm, tip −0.5 mm past back |
| Right (−Y) | −0.0225 | same |

Centers at `x = −0.012` (same proud pattern as front apertures).


## Size check (2026-09-21)
Measured STL AABB vs datasheet Table 3-52:
| Axis | Datasheet | Mesh AABB |
|------|-----------|-----------|
| Width Y | 90.00 mm | **90.000 mm** |
| Height Z | 25.00 mm | **25.000 mm** |
| Depth X | 25.00 mm | **25.000 mm** |

Front bevel station ≈ 84.5 × 19.5 mm (inside envelope). Official ROS `_d435.urdf.xacro` uses collision depth **25.05 mm**; twin uses exact 25.00. Aperture cylinders tip **+0.5 mm** past front glass (cosmetic, as specified). D435i IMU badge sits flush with top (not proud).


## D435i IMU (BMI055 / BMI085)
Nominal extrinsics from Intel `realsense2_description` `_d435i_imu_modules.urdf.xacro` (`use_nominal_extrinsics`).

Official `camera_link` / depth frame (REP-103) → IMU:
`xyz = (-0.01174, -0.00552, 0.0051)`, `rpy = (0, 0, 0)`.

Kevin mesh origin is housing geometric center; depth/`camera_link` sits at left IR in Y (`+0.0175 m`). IMU in mesh / `realsense_d435i` frame:

| | X | Y | Z |
|--|---|---|---|
| m | −0.01174 | +0.01198 | +0.00510 |

Links: `realsense_d435i_imu` (REP-103, same axes as camera body) and `realsense_d435i_imu_optical` (`rpy = −π/2, 0, −π/2`). TF `base_link` → `realsense_d435i` → `realsense_d435i_imu` is the IMU↔base extrinsic.


## Simulation frames (depth / infra / color / projector)
Both `realsense_d435` and `realsense_d435i` expose REP-103 body frames plus optical children for Gazebo / ROS camera simulation (same Y layout as Intel `realsense2_description` nominal extrinsics; mesh origin = housing geometric center):

| Frame | XYZ in mesh (m) | Notes |
|-------|-----------------|-------|
| `*_depth` / `*_infra1` | (0, +0.0175, 0) | Left IR / depth; = Intel `camera_link` in Y |
| `*_infra2` | (0, −0.0325, 0) | Right IR (−50 mm from left) |
| `*_color` | (0, +0.0325, 0) | RGB (+15 mm from left IR) |
| `*_projector` | (0, −0.0075, 0) | IR pattern projector |

Each has `*_optical` with `rpy = (−π/2, 0, −π/2)` (OpenCV: +X right, +Y down, +Z forward). Attach simulated cameras to the optical frames. X = 0 is mid-housing (Intel nominal); true depth optical center is ~8.2 mm forward of mid-housing (4.2 mm behind front glass).
