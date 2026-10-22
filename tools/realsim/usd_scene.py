"""Photoreal casita stage for NVIDIA ovrtx (RTX real-time path tracing) -> USDA, built from the ONE world model.

Geometry (no copies): walls = the resolved walls STL (src/world_params.py <- worlds/casita/world.json), normalised like
tools/casita_fit/camgeo.load_casita (bbox min at the origin); counters = world_params.counter_parts() (mat top/body).
Look (materials, lights, furniture, drapes) = tools/realsim/dressing.json; every entry carries 'src' saying what is
MEASURED (from tonight's real frames) vs ASSUMED.  Texture / furniture files live OUTSIDE git (--assets dir, CC0 sources
listed in its manifest); this file only references them.
Cameras: one USD camera + RenderProduct per sensor (mast RGB 800x600 Brown k1 via OmniLensDistortionOpenCvPinholeAPI,
forward RealSense colour 640x480 pinhole), pose written by tools/realsim/ovrtx_render.py at run time.
usage: usd_scene.py --assets DIR --out DIR/casita.usda [--dressing tools/realsim/dressing.json]
"""
import argparse, json, math, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, os.path.join(REPO, "tools", "casita_fit"))


def fl(v): return "%.6g" % v
def v3(a): return "(%s, %s, %s)" % (fl(a[0]), fl(a[1]), fl(a[2]))
def arr3(A): return "[" + ", ".join(v3(a) for a in A) + "]"
def arr2(A): return "[" + ", ".join("(%s, %s)" % (fl(a[0]), fl(a[1])) for a in A) + "]"
def arri(A): return "[" + ", ".join(str(int(a)) for a in A) + "]"


def tri_uv(T, n, tile):
    """planar UVs per triangle corner (metres / tile): vertical faces (u along the wall, v = z), else top-down (x, y)."""
    out = np.zeros(T.shape[:2] + (2,))
    vert = np.abs(n[:, 2]) < 0.7
    t = np.stack([-n[:, 1], n[:, 0]], 1); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    out[vert, :, 0] = (T[vert, :, :2] * t[vert, None, :]).sum(-1); out[vert, :, 1] = T[vert, :, 2]
    out[~vert] = T[~vert, :, :2]
    return out / tile


def mesh_prim(name, T, mat, tile=1.0, ind="    "):
    name = re.sub(r"\W", "_", name)  # USD prim names: [A-Za-z0-9_] only
    T = np.asarray(T, float); n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); L = np.linalg.norm(n, axis=1); keep = L > 1e-10
    T, n = T[keep], n[keep] / L[keep, None]
    P = T.reshape(-1, 3); uv = tri_uv(T, n, tile).reshape(-1, 2); N = np.repeat(n, 3, 0)
    lo, hi = P.min(0), P.max(0)
    return (f'{ind}def Mesh "{name}" (prepend apiSchemas = ["MaterialBindingAPI"])\n{ind}{{\n'
            f'{ind}    uniform token subdivisionScheme = "none"\n{ind}    uniform bool doubleSided = 1\n'
            f'{ind}    float3[] extent = [{v3(lo)}, {v3(hi)}]\n'
            f'{ind}    point3f[] points = {arr3(P)}\n'
            f'{ind}    int[] faceVertexCounts = {arri([3] * len(T))}\n'
            f'{ind}    int[] faceVertexIndices = {arri(range(len(P)))}\n'
            f'{ind}    normal3f[] normals = {arr3(N)} (interpolation = "faceVarying")\n'
            f'{ind}    texCoord2f[] primvars:st = {arr2(uv)} (interpolation = "faceVarying")\n'
            f'{ind}    rel material:binding = </World/Looks/{mat}>\n{ind}}}\n')


def material(name, m, tex_dir):
    """UsdPreviewSurface; m: color (linear rgb) or tex (dir/prefix of ambientCG-style *_Color/_NormalGL/_Roughness) with
    tint (multiplies the colour texture), roughness, emissive, opacity."""
    p = f"/World/Looks/{name}"; s = [f'        def Material "{name}"\n        {{\n            token outputs:surface.connect = <{p}/S.outputs:surface>\n']
    sh = [f'            def Shader "S"\n            {{\n                uniform token info:id = "UsdPreviewSurface"\n']
    tex = m.get("tex")
    if tex:
        s.append(f'            def Shader "uv"\n            {{\n                uniform token info:id = "UsdPrimvarReader_float2"\n'
                 f'                string inputs:varname = "st"\n                float2 outputs:result\n            }}\n')
        def T(tag, suffix, cs, out, extra=""):
            f = os.path.join(tex_dir, tex + suffix)
            if not os.path.exists(f): return False
            s.append(f'            def Shader "{tag}"\n            {{\n                uniform token info:id = "UsdUVTexture"\n'
                     f'                asset inputs:file = @{f}@\n                token inputs:sourceColorSpace = "{cs}"\n'
                     f'                token inputs:wrapS = "repeat"\n                token inputs:wrapT = "repeat"\n'
                     f'                float2 inputs:st.connect = <{p}/uv.outputs:result>\n{extra}                {out}\n            }}\n')
            return True
        tint = m.get("tint", [1, 1, 1])
        if T("col", m.get("color_suffix", "_Color.jpg"), "sRGB", "float3 outputs:rgb", f'                float4 inputs:scale = ({fl(tint[0])}, {fl(tint[1])}, {fl(tint[2])}, 1)\n'):
            sh.append(f'                color3f inputs:diffuseColor.connect = <{p}/col.outputs:rgb>\n')
        if T("nrm", "_NormalGL.jpg", "raw", "float3 outputs:rgb", f'                float4 inputs:scale = ({fl(2 * m.get("bump", 1.0))}, {fl(2 * m.get("bump", 1.0))}, 2, 1)\n'
             f'                float4 inputs:bias = ({fl(-m.get("bump", 1.0))}, {fl(-m.get("bump", 1.0))}, -1, 0)\n'):
            sh.append(f'                normal3f inputs:normal.connect = <{p}/nrm.outputs:rgb>\n')
        if "roughness" not in m and T("rgh", "_Roughness.jpg", "raw", "float outputs:r"):
            sh.append(f'                float inputs:roughness.connect = <{p}/rgh.outputs:r>\n')
    else:
        sh.append(f'                color3f inputs:diffuseColor = {v3(m.get("color", [0.5, 0.5, 0.5]))}\n')
    if "roughness" in m: sh.append(f'                float inputs:roughness = {fl(m["roughness"])}\n')
    if "emissive" in m: sh.append(f'                color3f inputs:emissiveColor = {v3(m["emissive"])}\n')
    if "opacity" in m: sh.append(f'                float inputs:opacity = {fl(m["opacity"])}\n')
    sh.append('                float inputs:metallic = 0\n                token outputs:surface\n            }\n')
    return "".join(s) + "".join(sh) + "        }\n"


def ear_clip(P):
    """triangulate a simple counter-clockwise polygon [[x, y], ...] (ear clipping) -> list of 3-vertex tuples."""
    V = [tuple(map(float, q)) for q in P]; T = []
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    def inside(p, a, b, c): return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0
    while len(V) > 3:
        for i in range(len(V)):
            a, b, c = V[i - 1], V[i], V[(i + 1) % len(V)]
            if cross(a, b, c) <= 1e-12: continue
            if any(inside(q, a, b, c) for q in V if q not in (a, b, c)): continue
            T.append((a, b, c)); del V[i]; break
        else: raise ValueError("ear_clip: polygon not simple / not CCW")
    return T + [tuple(V)]


def box_tris(c, h, yaw=0.0):
    co, si = math.cos(yaw), math.sin(yaw); T = []
    V = {}
    for a in (-1, 1):
        for b in (-1, 1):
            for z in (-1, 1):
                V[(a, b, z)] = (c[0] + co * a * h[0] - si * b * h[1], c[1] + si * a * h[0] + co * b * h[1], c[2] + z * h[2])
    for ax in range(3):
        for sg in (-1, 1):
            q = []
            for u, w in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                k = [0, 0, 0]; k[ax] = sg; o = [i for i in range(3) if i != ax]; k[o[0]] = u; k[o[1]] = w; q.append(V[tuple(k)])
            T += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return T


def drape_tris(x0, y0, x1, y1, z0, z1, depth=0.05, folds_per_m=9, nseg=12):
    """pleated curtain: sinusoidal folds along the rail (wall-parallel), hanging z0..z1."""
    L = math.hypot(x1 - x0, y1 - y0); u = np.array([x1 - x0, y1 - y0]) / L; nrm = np.array([-u[1], u[0]])
    ns = max(8, int(L * folds_per_m * 4)); s = np.linspace(0, L, ns + 1); off = depth * 0.5 * np.sin(2 * math.pi * folds_per_m * s / 1.0)
    zs = np.linspace(z0, z1, nseg + 1); T = []
    for i in range(ns):
        for j in range(nseg):
            def P(ii, jj): return (x0 + u[0] * s[ii] + nrm[0] * off[ii], y0 + u[1] * s[ii] + nrm[1] * off[ii], zs[jj])
            T += [(P(i, j), P(i + 1, j), P(i + 1, j + 1)), (P(i, j), P(i + 1, j + 1), P(i, j + 1))]
    return T


def build(assets, out, dressing):
    import world_params as wp, camgeo
    D = json.load(open(dressing)); tex_dir = os.path.join(assets, "textures")
    walls = camgeo.load_casita(wp.V["walls_stl"]); top = float(walls.bounds[1][2]); X1, Y1 = walls.bounds[1][:2]
    T = walls.triangles; n = walls.face_normals; cz = T[:, :, 2].mean(1)
    cls = np.full(len(T), "wall", object)
    rule = D["walls"]["accent_rule"]
    diag = (np.abs(n[:, 2]) < 0.3) & (np.abs(n[:, 0]) > rule["min_abs_nx"]) & (np.abs(n[:, 1]) > rule["min_abs_ny"])
    cls[diag] = "accent"
    floorf = (n[:, 2] > 0.7) & (cz < 0.02); cls[floorf] = "skip"
    L = []
    L.append('#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\ndef Xform "World"\n{\n')
    L.append('    def Scope "Looks"\n    {\n')
    for name, m in D["materials"].items(): L.append(material(name, m, tex_dir))
    L.append('    }\n')
    L.append('    def Xform "House"\n    {\n')
    L.append(mesh_prim("walls", T[cls == "wall"], "wall_paint", D["materials"]["wall_paint"].get("tile", 1.0)))
    L.append(mesh_prim("accent_walls", T[cls == "accent"], "wallpaper", D["materials"]["wallpaper"].get("tile", 1.0)))
    # floor: the default finish (carpet, with its ridges baked into the texture) everywhere, hard-floor zones of
    # worlds/<world>/world.json "floor_zones" laid 3 mm above it (so ridges show on carpet only)
    FZ = wp.floor_zones()
    L.append(mesh_prim("floor", [((0, 0, 0), (X1, 0, 0), (X1, Y1, 0)), ((0, 0, 0), (X1, Y1, 0), (0, Y1, 0))], FZ["default"], D["materials"][FZ["default"]].get("tile", 1.0)))
    for z in FZ["zones"]:
        L.append(mesh_prim("floor_" + z["name"], [tuple((x, y, 0.003) for x, y in t) for t in ear_clip(z["poly"])], z["material"], D["materials"][z["material"]].get("tile", 1.0)))
    cz_ = top + float(wp.V.get("ceiling_above_wall_top_m", 0.01)) if isinstance(wp.V.get("ceiling_above_wall_top_m"), (int, float)) else top + 0.01
    L.append(mesh_prim("ceiling", [((0, 0, top), (X1, Y1, top), (X1, 0, top)), ((0, 0, top), (0, Y1, top), (X1, Y1, top))], "ceiling", D["materials"]["ceiling"].get("tile", 1.0)))
    bt = D.get("ceiling_battens")
    if bt:
        TT = []
        for k in range(bt["n_min"], bt["n_max"] + 1):
            x = bt["x0"] + bt["pitch"] * k
            if 0 < x < X1: TT += box_tris((x, Y1 / 2, top - bt["h"] / 2), (bt["w"] / 2, Y1 / 2, bt["h"] / 2))
        L.append(mesh_prim("ceiling_battens", TT, "ceiling", 1.0))
    parts = wp.counter_parts(); bym = {}
    for p in parts: bym.setdefault("counter_" + p["mat"], []).extend(wp._part_tris(p))
    for mname, TT in bym.items(): L.append(mesh_prim(mname, TT, mname if mname in D["materials"] else "counter_white", 1.0))
    for i, b in enumerate(D.get("boxes", [])):
        L.append(mesh_prim("box_%02d_%s" % (i, b["name"]), box_tris(b["c"], [v / 2 for v in b["size"]], math.radians(b.get("yaw_deg", 0))), b["mat"], b.get("tile", 1.0)))
    for i, d in enumerate(D.get("drapes", [])):
        L.append(mesh_prim("drape_%02d" % i, drape_tris(*d["rail"], d["z0"], d["z1"], d.get("depth", 0.05)), d.get("mat", "sheer"), 1.0))
    sh_ = D.get("outer_shell")
    if sh_:  # closed blinds / drapes behind every window opening (night): a box just outside the walls' bbox, inward-facing
        g = sh_["margin"]; L.append(mesh_prim("outer_shell", box_tris((X1 / 2, Y1 / 2, top / 2), (X1 / 2 + g, Y1 / 2 + g, top / 2 + 0.005)), sh_["mat"], 1.0))
    L.append('    }\n')
    # furniture: CC0 USD assets referenced by path (outside git)
    L.append('    def Xform "Furniture"\n    {\n')
    for i, f in enumerate(D.get("furniture", [])):
        path = os.path.join(assets, f["asset"]); s = f.get("scale", 1.0)
        sc = s if isinstance(s, list) else [s, s, s]
        L.append(f'        def Xform "f{i:02d}_{f["name"]}" (prepend references = @{path}@)\n        {{\n'
                 f'            double3 xformOp:translate = {v3(f["pos"])}\n            float xformOp:rotateZ = {fl(f.get("yaw_deg", 0))}\n'
                 f'            float3 xformOp:scale = {v3(sc)}\n'
                 f'            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]\n        }}\n')
    L.append('    }\n')
    # lights: ceiling fixtures (sphere light + emissive diffuser disc)
    fx = [l for l in D.get("lights", []) if l.get("fixture")]
    if fx:  # visible fixture diffusers (emissive; the SphereLight inside does the lighting)
        L.append('    def Xform "Fixtures"\n    {\n')
        for i, l in enumerate(fx):
            F = l["fixture"]; TT = wp._part_tris(dict(kind="cyl", name="fx", role="fx", cx=l["xy"][0], cy=l["xy"][1], r=F["r"], yaw=0.0, z0=top - F["h"], z1=top - 0.002, mat="fx"))
            L.append(mesh_prim("fixture_%02d_%s" % (i, l["name"]), TT, F.get("mat", "fixture_diffuser"), 1.0, ind="        "))
        L.append('    }\n')
    L.append('    def Xform "Lights"\n    {\n')
    for i, l in enumerate(D.get("lights", [])):
        z = top - l.get("drop", 0.08)
        L.append(f'        def SphereLight "lamp{i:02d}_{l["name"]}"\n        {{\n            float inputs:radius = {fl(l.get("radius", 0.12))}\n'
                 f'            float inputs:intensity = {fl(l["intensity"])}\n            bool inputs:enableColorTemperature = 1\n'
                 f'            float inputs:colorTemperature = {fl(l.get("kelvin", 2700))}\n            bool inputs:normalize = 1\n'
                 f'            double3 xformOp:translate = ({fl(l["xy"][0])}, {fl(l["xy"][1])}, {fl(z)})\n'
                 f'            uniform token[] xformOpOrder = ["xformOp:translate"]\n        }}\n')
    L.append('    }\n')
    L.append('    def Xform "Cams"\n    {\n')
    for cam in D["cameras"]: L.append(camera_prim(cam))
    L.append('    }\n}\n\n')
    L.append(render_scope(D["cameras"], D["render"]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); open(out, "w").write("".join(L))
    return out


def camera_prim(c):
    """c: name, w, h, f, cx, cy, k1 (Brown on normalised coords == OpenCV k1)."""
    HA = 20.955; fl_ = c["f"] * HA / c["w"]; VA = HA * c["h"] / c["w"]
    s = (f'        def Camera "{c["name"]}"' + (' (prepend apiSchemas = ["OmniLensDistortionOpenCvPinholeAPI"])' if c.get("k1") else "") + '\n        {\n'
         f'            float focalLength = {fl(fl_)}\n            float horizontalAperture = {fl(HA)}\n            float verticalAperture = {fl(VA)}\n'
         f'            float horizontalApertureOffset = {fl((c["cx"] - (c["w"] - 1) / 2) / c["w"] * HA)}\n'
         f'            float verticalApertureOffset = {fl(-(c["cy"] - (c["h"] - 1) / 2) / c["h"] * VA)}\n'
         f'            float2 clippingRange = (0.03, 100)\n')
    if c.get("k1"):
        s += (f'            token omni:lensdistortion:model = "opencvPinhole"\n            int2 omni:lensdistortion:opencvPinhole:imageSize = ({c["w"]}, {c["h"]})\n'
              f'            float omni:lensdistortion:opencvPinhole:fx = {fl(c["f"])}\n            float omni:lensdistortion:opencvPinhole:fy = {fl(c["f"])}\n'
              f'            float omni:lensdistortion:opencvPinhole:cx = {fl(c["cx"])}\n            float omni:lensdistortion:opencvPinhole:cy = {fl(c["cy"])}\n'
              f'            float omni:lensdistortion:opencvPinhole:k1 = {fl(c["k1"])}\n            float omni:lensdistortion:opencvPinhole:k2 = {fl(c.get("k2", 0))}\n')
    s += ('            matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 1, 1) )\n'
          '            uniform token[] xformOpOrder = ["xformOp:transform"]\n        }\n')
    return s


def render_scope(cams, R):
    s = 'def Scope "Render"\n{\n'
    for c in cams:
        s += (f'    def RenderProduct "{c["name"]}"\n    {{\n        rel camera = </World/Cams/{c["name"]}>\n        int2 resolution = ({c["w"]}, {c["h"]})\n'
              f'        rel orderedVars = [<HdrColor>, <LdrColor>]\n        token omni:rtx:rendermode = "Real-Time Path-Tracing"\n')
        for k, (typ, v) in R.items():
            s += f'        {typ} {k} = {json.dumps(v) if isinstance(v, str) else (str(int(v)) if typ in ("bool",) else fl(v) if typ == "float" else int(v))}\n'
        s += ('        def RenderVar "HdrColor"\n        {\n            string sourceName = "HdrColor"\n        }\n'
              '        def RenderVar "LdrColor"\n        {\n            string sourceName = "LdrColor"\n        }\n    }\n')
    return s + "}\n"


def R_world_to_cam(yaw, pitch, roll):
    """Same convention as tools/casita_fit/camgeo.R_world_to_cam (copied so the ovrtx venv needs no trimesh):
    rows = camera axes (x right, y down, z forward) in world coords; pitch nose-down +, roll clockwise in image +."""
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0]); r = np.array([np.sin(yaw), -np.cos(yaw), 0.0]); d = np.array([0.0, 0.0, -1.0])
    cp, sp = np.cos(pitch), np.sin(pitch); f2 = cp * f + sp * d; d2 = -sp * f + cp * d
    cr, sr = np.cos(roll), np.sin(roll); r3 = cr * r + sr * d2; d3 = -sr * r + cr * d2
    return np.stack([r3, d3, f2])


def cam_matrix(p):
    """camgeo pose dict -> USD row-vector local-to-world (camera looks down -Z, +Y up)."""
    r, d, f = R_world_to_cam(p["yaw"], p["pitch"], p["roll"])
    M = np.eye(4); M[0, :3] = r; M[1, :3] = -d; M[2, :3] = -f; M[3, :3] = (p["x"], p["y"], p["z"]); return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--assets", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dressing", default=os.path.join(HERE, "dressing.json")); a = ap.parse_args()
    print(build(a.assets, a.out, a.dressing))
