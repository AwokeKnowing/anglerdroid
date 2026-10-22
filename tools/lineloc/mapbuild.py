"""Build the 2D line map for lineloc from the casita wall STL (offline, once).
  vertical edges  -> corners (x, y, z0, z1)          (main cue: bearings)
  horizontal edges-> 3D segments (A, B)               (secondary cue: image lines at known heights)
  visibility      -> per 10 cm grid cell, which features a mast camera there can see (Embree rays)
python3 mapbuild.py [STL] [CAM_H]   (defaults: the resolved world mesh, the selected wheels' mast camera height)
Frame = sim world = STL normalised with its bbox min at the origin (as emu.envs.env_casita / maploc.StructMap)."""
import sys, json, math, numpy as np, trimesh
CREASE_DEG = 30.0

def load(stl):
    m = trimesh.load(stl); ext = m.bounds; sc = 0.001 if (ext[1] - ext[0]).max() > 100 else 1.0
    return trimesh.Trimesh(m.vertices * sc - ext[0] * sc, m.faces, process=True)

def feature_edges(m):
    ang = m.face_adjacency_angles; E = [m.face_adjacency_edges[ang > math.radians(CREASE_DEG)]]
    # open-boundary edges (used by one face only)
    eu, cnt = np.unique(np.sort(m.edges, 1), axis=0, return_counts=True); E.append(eu[cnt == 1])
    E = np.unique(np.sort(np.vstack(E), 1), axis=0); return m.vertices[E[:, 0]], m.vertices[E[:, 1]]

def merge_collinear(A, B, tol=0.01, ang=2.0):
    """merge chains of collinear touching/overlapping edges into maximal segments"""
    segs = [(a, b) for a, b in zip(A, B)]; changed = True
    while changed:
        changed = False; out = []
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]: continue
            a, b = segs[i]
            for j in range(i + 1, len(segs)):
                if used[j]: continue
                c, d = segs[j]; u = (b - a) / np.linalg.norm(b - a); v = (d - c) / np.linalg.norm(d - c)
                if abs(u @ v) < math.cos(math.radians(ang)): continue
                # c, d on the line through a, b ?
                def off(p): w = p - a; return np.linalg.norm(w - (w @ u) * u)
                if off(c) > tol or off(d) > tol: continue
                t = sorted([0, (b - a) @ u, (c - a) @ u, (d - a) @ u]); s = [(c - a) @ u, (d - a) @ u]
                if min(s) > (b - a) @ u + tol or max(s) < -tol: continue      # disjoint
                lo, hi = min(0, *s), max((b - a) @ u, *s); a, b = a + lo * u, a + hi * u; used[j] = True; changed = True
            out.append((a, b)); used[i] = True
        segs = out
    return np.array([s[0] for s in segs]), np.array([s[1] for s in segs])

def build(stl=None, cam_h=None, cell=0.10, out="casita_lines.npz"):
    """stl None = the resolved world mesh (LL_STL / src/world_params.py); cam_h None = the selected wheels' mast camera
    height (src/robot_params.py mast_rgb_height_m; was a fixed stock 0.97615)."""
    from lineloc import world_stl, mast_height
    stl = stl or world_stl(); cam_h = mast_height() if cam_h is None else cam_h
    print("mapbuild:", stl, "cam_h %.5f" % cam_h)
    m = load(stl); A, B = feature_edges(m); d = B - A; L = np.linalg.norm(d, axis=1); dz = np.abs(d[:, 2]) / np.maximum(L, 1e-9)
    vm = (dz > 0.995); hm = (dz < 0.02) & (L > 0.05)
    VA, VB = merge_collinear(A[vm], B[vm]); keep = np.linalg.norm(VB - VA, axis=1) > 0.25; VA, VB = VA[keep], VB[keep]
    # corners: (x, y, z0, z1); merge stacked vertical edges at the same xy (gaps < 5 cm)
    V = np.c_[(VA[:, :2] + VB[:, :2]) / 2, np.minimum(VA[:, 2], VB[:, 2]), np.maximum(VA[:, 2], VB[:, 2])]
    HA, HB = merge_collinear(A[hm], B[hm]); keep = np.linalg.norm(HB - HA, axis=1) > 0.30; HA, HB = HA[keep], HB[keep]
    print("vertical edges", len(V), "horizontal edges", len(HA))
    # visibility grid
    lo = m.bounds[0][:2] - 0.2; hi = m.bounds[1][:2] + 0.2; nx, ny = np.ceil((hi - lo) / cell).astype(int)
    gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + .5) * cell, lo[1] + (np.arange(ny) + .5) * cell, indexing="ij")
    C = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, cam_h)]
    # inside free space: a downward-ish ray box test -> require camera not inside a wall: horizontal rays in 8 dirs all hit something
    ints = m.ray
    dirs8 = np.array([[math.cos(a), math.sin(a), 0] for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)])
    o = np.repeat(C, 8, 0); dd = np.tile(dirs8, (len(C), 1)); hit = ints.intersects_any(o, dd).reshape(-1, 8)
    inside = hit.sum(1) >= 2
    # feature sample points (3 per vertical, 5 per horizontal), pushed 1 cm toward the camera at test time
    def vis(P, groups):
        """P: (k,3) sample points, groups: feature id per sample -> bool (cells, features)"""
        nf = groups.max() + 1; out = np.zeros((len(C), nf), bool); idx = np.where(inside)[0]
        for s in range(0, len(idx), 400):
            ci = idx[s:s + 400]; o = np.repeat(C[ci], len(P), 0); tgt = np.tile(P, (len(ci), 1)); dv = tgt - o
            dist = np.linalg.norm(dv, axis=1); dv /= dist[:, None]
            loc, ri, _ = ints.intersects_location(o, dv, multiple_hits=False)
            hd = np.full(len(o), np.inf); hd[ri] = np.linalg.norm(loc - o[ri], axis=1)
            ok = (hd > dist - 0.02).reshape(len(ci), len(P)) & (dist.reshape(len(ci), len(P)) < 9.0)
            for f in range(nf): out[ci, f] = ok[:, groups == f].any(1)
        return out
    Pv, gv = [], []
    for i, (x, y, z0, z1) in enumerate(V):
        for t in (0.15, 0.5, 0.85): Pv.append([x, y, z0 + t * (z1 - z0)]); gv.append(i)
    Ph, gh = [], []
    for i, (a, b) in enumerate(zip(HA, HB)):
        for t in (0.1, 0.3, 0.5, 0.7, 0.9): Ph.append(a + t * (b - a)); gh.append(i)
    visV = vis(np.array(Pv), np.array(gv)); visH = vis(np.array(Ph), np.array(gh))
    np.savez_compressed(out, V=V, HA=HA, HB=HB, visV=np.packbits(visV, 1), visH=np.packbits(visH, 1), nV=len(V), nH=len(HA),
                        lo=lo, cell=cell, shape=np.array([nx, ny]), inside=inside, cam_h=cam_h)
    print("cells", len(C), "inside", inside.sum(), "mean visible V/H", visV[inside].sum(1).mean(), visH[inside].sum(1).mean())

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None, cam_h=float(sys.argv[2]) if len(sys.argv) > 2 else None)
