#!/usr/bin/env python3
"""Generate src/twin_geometry.json from the digital twin URDF(s) - the twin is the source of truth for every
geometric constant the robot code uses (one entry per tire profile in src/tire_profiles.json).

  python scripts/gen_twin_geometry.py [--twin ~/kevin-urdf-twin-serve/kevin-urdf-twin]
Frames are relative to base_link (= axle centre, +x forward, +y left, +z up); heights are above the floor
(= z + wheel loaded radius; ride_dz is exactly the loaded-radius increase)."""
import argparse, json, math, os, xml.etree.ElementTree as ET
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = dict(rs1_depth="realsense_d435_depth_optical", rs1_color="realsense_d435_color_optical",
              rs1_ir1="realsense_d435_infra1_optical", rs1_ir2="realsense_d435_infra2_optical",
              rs2_depth="realsense_d435i_depth_optical", rs2_color="realsense_d435i_color_optical",
              rs2_ir1="realsense_d435i_infra1_optical", rs2_ir2="realsense_d435i_infra2_optical",
              rs2_imu="realsense_d435i_imu_optical", rs2_gyro="realsense_d435i_gyro_optical", rs2_accel="realsense_d435i_accel_optical",
              rgb="rgb_camera_optical", wheel_left="wheel_left", wheel_right="wheel_right")


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def chain(urdf):
    root = ET.parse(urdf).getroot(); par = {}
    for j in root.findall("joint"):
        o = j.find("origin"); xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())]
        r = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
        T = np.eye(4); T[:3, :3] = rpy(*r); T[:3, 3] = xyz
        par[j.find("child").get("link")] = (j.find("parent").get("link"), T)
    def coll_origin(link):
        for l in root.findall("link"):
            if l.get("name") == link:
                c = l.find("collision")
                if c is not None and c.find("origin") is not None: return np.array([float(v) for v in c.find("origin").get("xyz").split()])
        return np.zeros(3)
    def T_base(link):
        T = np.eye(4)
        while link != "base_link":
            p, Tj = par[link]; T = Tj @ T; link = p
        return T
    T_base.coll = coll_origin
    return T_base


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--twin", default=os.path.expanduser("~/kevin-urdf-twin-serve/kevin-urdf-twin")); a = ap.parse_args()
    prof = json.load(open(os.path.join(REPO, "src", "tire_profiles.json")))["profiles"]; out = {"_doc": __doc__, "profiles": {}}
    for name, P in prof.items():
        T_base = chain(os.path.join(a.twin, P["twin_urdf"])); wr = float(P["wheel_r_loaded_m"]); dz = float(P["ride_dz_m"])
        fr = {}
        for k, link in FRAMES.items():
            T = T_base(link); fr[k] = dict(xyz=[round(float(v), 6) for v in T[:3, 3]], R=[[round(float(v), 6) for v in row] for row in T[:3, :3]],
                                           height_above_floor=round(float(T[2, 3]) + wr, 6))
        # track = distance between tire contact centres (collision cylinder centres), not hub-joint origins
        cl = (T_base("wheel_left") @ np.r_[T_base.coll("wheel_left"), 1])[:3]; cr = (T_base("wheel_right") @ np.r_[T_base.coll("wheel_right"), 1])[:3]
        track = abs(cl[1] - cr[1])
        z2 = np.array(fr["rs2_depth"]["R"])[:, 2]                      # optical z (view dir) in base
        out["profiles"][name] = dict(wheel_r_m=wr, ride_dz_m=dz, track_m=round(track, 6), axle_height_m=round(wr, 6),
                                     rs2_pitch_down_deg=round(math.degrees(math.asin(-z2[2])), 3), frames=fr, twin_urdf=P["twin_urdf"])
    p = os.path.join(REPO, "src", "twin_geometry.json"); json.dump(out, open(p, "w"), indent=1); print("wrote", p)
    for n, v in out["profiles"].items():
        print(n, "wheel_r", v["wheel_r_m"], "track", v["track_m"], "rs1", v["frames"]["rs1_depth"]["xyz"], "rs2", v["frames"]["rs2_depth"]["xyz"], "pitch", v["rs2_pitch_down_deg"])


if __name__ == "__main__":
    main()
