#!/usr/bin/env python3
"""Generate anglerdroid URDF and body mesh STL.

Origin: centre of the axle between wheels, at wheel-centre height.
  +X = forward, +Y = left, +Z = up.
  Ground plane at z = -R (R = wheel radius).

Run:  python robot/generate_urdf.py
Creates:  robot/anglerdroid.urdf, robot/meshes/body.stl
"""
import struct, os, math

R = 0.0857          # wheel radius (m)
WHEEL_W = 0.05      # wheel width
WHEEL_GAP = 0.01    # frame-to-wheel gap
FRAME_HW = 0.15     # frame half-width (30 cm / 2)
TOTAL_HW = FRAME_HW + WHEEL_GAP + WHEEL_W  # 0.21

Z_BASE = 0.05 - R   # body base in URDF frame
Z_TOP  = 0.22 - R   # body top in URDF frame
X_BACK = -0.15
X_FRONT = 0.13
X_SLOPE = 0.0
Y_HW = 0.16         # body half-width (slightly wider than frame for visual)

MAST_TOP = 1.00 - R
ARM_Z0 = 0.97 - R
ARM_Z1 = 1.00 - R

DIR = os.path.dirname(os.path.abspath(__file__))

# ─── STL generation ──────────────────────────────────────────────────

def _tri_normal(v0, v1, v2):
    e1 = [v1[i] - v0[i] for i in range(3)]
    e2 = [v2[i] - v0[i] for i in range(3)]
    n = [e1[1]*e2[2]-e1[2]*e2[1],
         e1[2]*e2[0]-e1[0]*e2[2],
         e1[0]*e2[1]-e1[1]*e2[0]]
    l = math.sqrt(sum(x*x for x in n)) or 1e-12
    return [x/l for x in n]

def write_binary_stl(path, triangles):
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(triangles)))
        for v0, v1, v2 in triangles:
            n = _tri_normal(v0, v1, v2)
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))
    print(f"  wrote {path} ({len(triangles)} triangles)")

def quad_tris(a, b, c, d):
    return [(a, b, c), (a, c, d)]

def generate_body_stl():
    v = [
        (X_FRONT, -Y_HW, Z_BASE),   # 0 front-bottom-near
        (X_BACK,  -Y_HW, Z_BASE),   # 1 back-bottom-near
        (X_BACK,  -Y_HW, Z_TOP),    # 2 back-top-near
        (X_SLOPE, -Y_HW, Z_TOP),    # 3 slope-top-near
        (X_FRONT,  Y_HW, Z_BASE),   # 4 front-bottom-far
        (X_BACK,   Y_HW, Z_BASE),   # 5 back-bottom-far
        (X_BACK,   Y_HW, Z_TOP),    # 6 back-top-far
        (X_SLOPE,  Y_HW, Z_TOP),    # 7 slope-top-far
    ]
    tris = []
    tris += quad_tris(v[0], v[4], v[5], v[1])   # bottom (outward = -Z)
    tris += quad_tris(v[1], v[5], v[6], v[2])   # back   (outward = -X)
    tris += quad_tris(v[2], v[6], v[7], v[3])   # top    (outward = +Z)
    tris += quad_tris(v[3], v[7], v[4], v[0])   # slope  (outward = +X,+Z)
    tris += quad_tris(v[1], v[2], v[3], v[0])   # near   (outward = -Y)
    tris += quad_tris(v[4], v[7], v[6], v[5])   # far    (outward = +Y)
    write_binary_stl(os.path.join(DIR, "meshes", "body.stl"), tris)

# ─── URDF generation ─────────────────────────────────────────────────

def _box_visual(name, xyz, size, rgba):
    ox, oy, oz = xyz
    sx, sy, sz = size
    return f"""    <visual name="{name}">
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="0 0 0"/>
      <geometry><box size="{sx:.4f} {sy:.4f} {sz:.4f}"/></geometry>
      <material name="{name}_mat"><color rgba="{rgba}"/></material>
    </visual>"""

def _cyl_visual(name, xyz, radius, length, rgba, rpy="1.5708 0 0"):
    ox, oy, oz = xyz
    return f"""    <visual name="{name}">
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rpy}"/>
      <geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>
      <material name="{name}_mat"><color rgba="{rgba}"/></material>
    </visual>"""

def generate_urdf():
    mast_cx = (X_BACK + X_BACK + 0.03) / 2   # near back wall
    mast_h = MAST_TOP - Z_TOP
    mast_z = Z_TOP + mast_h / 2

    arm_x0 = mast_cx + 0.015
    arm_x1 = 0.015
    arm_cx = (arm_x0 + arm_x1) / 2
    arm_len = arm_x1 - arm_x0
    arm_z = (ARM_Z0 + ARM_Z1) / 2

    caster_cx = X_BACK - 0.03
    caster_z = 0.02 - R

    jetson_cx = (X_BACK + 0.01 + X_BACK + 0.11) / 2
    jetson_z = Z_TOP + 0.0125

    rs1_cx = -0.005
    rs1_z = ARM_Z1 + 0.0125
    rs2_cx = 0.0325
    rs2_z = ARM_Z0 - 0.015

    wheel_cy_l = -(FRAME_HW + WHEEL_GAP + WHEEL_W / 2)
    wheel_cy_r = -wheel_cy_l

    urdf = f"""<?xml version="1.0"?>
<robot name="anglerdroid">
  <link name="base_link">
    <!-- Trapezoidal body (STL mesh, origin already at axle centre) -->
    <visual name="body">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="meshes/body.stl"/></geometry>
      <material name="body_mat"><color rgba="0.18 0.18 0.20 1"/></material>
    </visual>

    <!-- Left wheel -->
{_cyl_visual("wheel_left", (0, wheel_cy_l, 0), R*0.97, WHEEL_W, "0.09 0.09 0.10 1")}

    <!-- Right wheel -->
{_cyl_visual("wheel_right", (0, wheel_cy_r, 0), R*0.97, WHEEL_W, "0.09 0.09 0.10 1")}

    <!-- Caster -->
{_box_visual("caster", (caster_cx, 0, caster_z), (0.04, 0.03, 0.04), "0.16 0.16 0.18 1")}

    <!-- Mast -->
{_box_visual("mast", (mast_cx, 0, mast_z), (0.03, 0.03, mast_h), "0.40 0.42 0.45 1")}

    <!-- Camera arm -->
{_box_visual("camera_arm", (arm_cx, 0, arm_z), (abs(arm_len), 0.03, 0.03), "0.34 0.36 0.40 1")}

    <!-- RS1 top-down camera -->
{_box_visual("rs1_topdown", (rs1_cx, 0, rs1_z), (0.09, 0.08, 0.025), "0.12 0.22 0.32 1")}

    <!-- RS2 forward camera -->
{_box_visual("rs2_forward", (rs2_cx, 0, rs2_z), (0.035, 0.06, 0.03), "0.12 0.22 0.32 1")}

    <!-- Jetson NX -->
{_box_visual("jetson_nx", (jetson_cx, 0, jetson_z), (0.10, 0.08, 0.025), "0.16 0.22 0.16 1")}

    <!-- Base footprint (ground reference, transparent) -->
{_box_visual("footprint", (0, 0, -R + 0.001), (0.28, 0.42, 0.002), "0.5 0.5 0.5 0.2")}

    <collision name="body_collision">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="meshes/body.stl"/></geometry>
    </collision>
  </link>
</robot>
"""
    path = os.path.join(DIR, "anglerdroid.urdf")
    with open(path, 'w') as f:
        f.write(urdf)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print("Generating anglerdroid URDF...")
    generate_body_stl()
    generate_urdf()
    print("Done. Open robot/anglerdroid.urdf in a URDF viewer.")
