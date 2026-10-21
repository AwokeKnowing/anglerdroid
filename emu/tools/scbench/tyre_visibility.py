"""RS1 (top-down) / RS2 (forward) visible ground and smallest visible obstacle next to the hull, from the twin.

  KEVIN_TIRE=stock|soft_v3_2 python -m emu.tools.scbench.tyre_visibility [--json out]
Ray-casts the twin MJCF (visual meshes, group 2 + floor) from each depth camera's twin pose (robot settled on a flat
floor) to floor points and obstacle-top points at a gap d outside FOOTPRINT_SAFE, along outward normals at sample
points around the hull.  A point counts as seen when it is inside the depth FOV (87 x 58 deg) and the first hit along
the ray is the point itself.  The obstacle threshold of the RS1 floor clip (TD_FLOOR_CLIP) is reported separately."""
import json, math, os, sys
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")]
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
import twin_geometry as G
import tire_profile as TP
from nav.safety import _edges

TWIN = os.environ.get("KEVIN_TWIN", os.path.expanduser("~/kevin-urdf-twin-serve/kevin-urdf-twin/simready"))
HF, VF = math.radians(87 / 2), math.radians(58 / 2)


def model():
    fn = "kevin_full.xml" if TP.STOCK else "kevin_full%s.xml" % TP.SUFFIX
    root = ET.parse(os.path.join(TWIN, fn)).getroot(); c = root.find("compiler")
    c.set("meshdir", os.path.join(TWIN, c.get("meshdir", "assets"))); c.set("texturedir", os.path.join(TWIN, c.get("texturedir", "assets")))
    for g in root.iter("geom"):
        if g.get("name") in ("wheel_left_tire", "wheel_right_tire") and g.get("type") == "cylinder":
            r, hw = g.get("size").split()[:2]; g.set("type", "ellipsoid"); g.set("size", "%s %s %s" % (r, r, hw))
    if not any(g.get("name") == "floor" for g in root.iter("geom")):
        ET.SubElement(root.find("worldbody"), "geom", name="floor", type="plane", size="5 5 0.1", contype="1", conaffinity="2", group="0")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")); d = mujoco.MjData(m)
    for i in range(m.ngeom):                   # as emu.world: the RealSense housings enclose the sensor origins
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[i])) or ""
            if mn.endswith(("rs_aluminum", "rs_ir_lens")): m.geom_group[i] = 5
    if m.nkey: mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(3000): mujoco.mj_step(m, d)
    return m, d


def main():
    m, d = model(); b = m.body("base_link").id; Rb = d.xmat[b].reshape(3, 3).copy(); pb = d.xpos[b].copy()
    grp = np.zeros(6, np.uint8); grp[2] = 1; grp[int(m.geom_group[m.geom("floor").id])] = 1; gid = np.zeros(1, np.int32)
    cams = {k: G.frame(k + "_depth") for k in ("rs1", "rs2")}
    def seen(cam, P):                         # P world point
        t, R = cams[cam]; o = pb + Rb @ t; Rw = Rb @ R; q = Rw.T @ (P - o)
        if q[2] <= 0.05 or abs(q[0] / q[2]) > math.tan(HF) or abs(q[1] / q[2]) > math.tan(VF): return False
        v = P - o; L = float(np.linalg.norm(v)); dist = mujoco.mj_ray(m, d, o, v / L, grp, 1, -1, gid)
        return dist < 0 or dist >= L - 0.004
    n, c = _edges(G.FOOTPRINT_SAFE); F = G.FOOTPRINT_SAFE
    pts = []                                   # (label, point on the hull, outward normal)
    for (lab, x, y, nx, ny) in (("front", G.HULL_FRONT_M + 0.0003, 0.0, 1, 0), ("front_left_corner", 0.105, 0.185, 0.81, 0.58),
                                ("left_beside_wheel_axle", 0.0, 0.20998, 0, 1), ("left_wheel_front_quarter", 0.06, 0.20998, 0, 1),
                                ("left_rear_quarter", -0.06, 0.20998, 0, 1), ("rear", -0.2737, 0.0, -1, 0),
                                ("right_beside_wheel_axle", 0.0, -0.20998, 0, -1), ("front_right_corner", 0.105, -0.185, 0.81, -0.58)):
        pts.append((lab, np.array([x, y]), np.array([nx, ny]) / math.hypot(nx, ny)))
    out = dict(tire=TP.NAME, rs1_h=round(G.height("rs1_depth"), 4), rs2_h=round(G.height("rs2_depth"), 4), base_z=round(float(pb[2]), 5),
               pitch_deg=round(math.degrees(math.asin(-Rb[2, 0])), 3))
    try:
        from perception.ego_rs1 import TD_FLOOR_CLIP
        out["rs1_clip_obstacle_h_cm"] = round(100 * (G.height("rs1_depth") - float(TD_FLOOR_CLIP)), 2)
    except Exception as e: out["rs1_clip_err"] = str(e)
    for lab, p, nv in pts:
        r = {}
        # start from the hull boundary along the normal: first gap with a seen floor point
        base = p.copy()
        while (np.r_[base] @ n.T - c).max() < 0: base = base + nv * 0.001
        for cam in ("rs1", "rs2"):
            dmin = None
            for dd in np.arange(0.0, 0.6, 0.005):
                q = base + nv * dd; P = pb + Rb @ np.r_[q, -pb[2] + 0.001]
                if seen(cam, np.r_[P[0], P[1], 0.001]): dmin = round(float(dd) * 100, 1); break
            hmin = {}
            for gap in (0.01, 0.02, 0.035, 0.05, 0.10):
                q = base + nv * gap; hv = None
                for H in np.arange(0.005, 0.60, 0.005):
                    P = pb + Rb @ np.r_[q, 0.0]; P[2] = H
                    if seen(cam, P): hv = round(float(H) * 100, 1); break
                hmin["%gcm" % (gap * 100)] = hv
            r[cam] = dict(floor_visible_from_cm=dmin, min_visible_top_cm=hmin)
        out[lab] = r
    print(json.dumps(out, indent=1))
    if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
