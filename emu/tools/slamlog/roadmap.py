"""Topological roadmap prototype (offline; docs/SLAM_LIFELONG.md section 11).
  PYTHONPATH=.:src python -m emu.tools.slamlog.roadmap <out.json> <log> [<log> ...]   (pose source: nav's live pose = kslam)
Per log:
 1. occupancy (5 cm) from the levelled scan cache (lslam_replay.Scans) at the live kslam pose: obstacle hits + free cells
    along each ray (sensor -> hit, last 10 cm excluded); occupied = hit evidence > free evidence.
 2. Manhattan frame: magnitude-weighted histogram of wall-gradient orientation mod 90 deg.
 3. free space eroded by the robot radius (twin R_CIRC) -> Zhang-Suen skeleton -> spurs shorter than the clearance at
    their junction pruned.  Nodes: junction = branch point (>= 3 skeleton neighbours, clustered within its clearance),
    end = skeleton end (room end / dead end), door = clearance local minimum along an edge below half the clearance of
    both its ends (a pinch).  Edges: skeleton paths between nodes, length, heading; snapped to the Manhattan axis when the
    deviation is within atan(clearance / length) (the corridor admits no finer heading).
 4. same pipeline on the GT wall section (sim/casitahouse-walls-fixed.stl at 0.5 m) restricted to the explored area ->
    GT nodes.  Metrics: nodes matched within MATCH_M; wrong edges = edges crossing a GT wall cell; routes between random
    node pairs (Dijkstra) valid = no GT wall within R_CIRC along the route.
 5. node relocalization: every revisit of a node (trajectory leaves > 2 m and returns within NODE_R) matches the scan at
    closest approach against the node's local submap (2 cm GridMap from the first visit's scans) with nav.slam.Matcher
    (+-0.3 m / +-8 deg); error = drift since the first visit vs GT, before and after the correction.
MATCH_M / NODE_R (0.5 m / 0.5 m) are evaluation radii (assumed), not estimator parameters."""
import sys, os, math, json
import numpy as np, cv2, networkx as nx
import twin_geometry as G
from .log import Log
from . import lslam_replay as LR
from nav.slam import GridMap, Matcher, transform
from nav.lslam.model import compose, between, wrap

RES = 0.05; MATCH_M = 0.5; NODE_R = 0.5; RR = float(G.R_CIRC)


def thin(img):
    """Zhang-Suen thinning (bool image)."""
    I = np.pad(img.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = [np.roll(np.roll(I, -dy, 0), -dx, 1) for dy, dx in ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]
            B = sum(P); A = sum(((P[k] == 0) & (P[(k + 1) % 8] == 1)).astype(np.uint8) for k in range(8))
            c = (P[0] * P[2] * P[4] == 0) & (P[2] * P[4] * P[6] == 0) if step == 0 else (P[0] * P[2] * P[6] == 0) & (P[0] * P[4] * P[6] == 0)
            m = (I == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any(): I[m] = 0; changed = True
        if not changed: return I[1:-1, 1:-1].astype(bool)


NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
def nbrs(S, y, x):
    return [(y + dy, x + dx) for dy, dx in NB if 0 <= y + dy < S.shape[0] and 0 <= x + dx < S.shape[1] and S[y + dy, x + dx]]


def manhattan(occ):
    o = cv2.GaussianBlur(occ.astype(np.float32), (5, 5), 1.0); gx = cv2.Sobel(o, cv2.CV_32F, 1, 0); gy = cv2.Sobel(o, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy).ravel(); ang = (np.arctan2(gy, gx).ravel()) % (math.pi / 2)
    h, e = np.histogram(ang, 90, (0, math.pi / 2), weights=mag); h = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5) / 5, "valid")
    return float(e[np.argmax(h)] + (e[1] - e[0]) / 2)


def graph(free, theta_m, origin):
    """free: bool grid (rows = y).  Returns networkx graph with node attrs (xy, kind, clear) and edge attrs (len, heading, snapped, path)."""
    clr = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * RES
    S = thin(clr > RR)
    # prune spurs shorter than the clearance at their junction (repeat until stable)
    for _ in range(20):
        deg = {p: len(nbrs(S, *p)) for p in zip(*np.nonzero(S))}; removed = False
        for p, d in deg.items():
            if d != 1 or not S[p]: continue
            path = [p]; prev = None; cur = p
            while True:
                nb = [q for q in nbrs(S, *cur) if q != prev and q not in path]
                if len(nb) != 1 or len(nbrs(S, *nb[0])) >= 3: junc = nb[0] if len(nb) == 1 else None; break
                prev, cur = cur, nb[0]; path.append(cur)
                if len(path) > 400: junc = None; break
            if junc is not None and len(path) * RES < clr[junc]:
                for q in path: S[q] = False
                removed = True
        if not removed: break
    pts = list(zip(*np.nonzero(S))); deg = {p: len(nbrs(S, *p)) for p in pts}
    key = [p for p in pts if deg[p] >= 3 or deg[p] == 1]
    # cluster branch pixels within their clearance into one node
    nid = {}; nodes = []
    for p in sorted(key, key=lambda p: -clr[p]):
        if p in nid: continue
        k = len(nodes); kind = "junction" if deg[p] >= 3 else "end"; nodes.append(dict(rc=p, kind=kind, clear=float(clr[p])))
        for q in key:
            if q not in nid and deg[q] == deg[p] if kind == "end" else (q not in nid and deg[q] >= 3):
                if math.hypot(q[0] - p[0], q[1] - p[1]) * RES <= max(clr[p], RES): nid[q] = k
        nid[p] = k
    Gr = nx.Graph()
    xy = lambda rc: (origin[0] + (rc[1] + 0.5) * RES, origin[1] + (rc[0] + 0.5) * RES)
    for k, n in enumerate(nodes): Gr.add_node(k, xy=xy(n["rc"]), kind=n["kind"], clear=n["clear"])
    seen = set()
    for p in key:
        for q0 in nbrs(S, *p):
            if q0 in nid or (p, q0) in seen: continue
            path = [p, q0]; prev, cur = p, q0
            while cur not in nid:
                nb = [q for q in nbrs(S, *cur) if q != prev and q not in path[-3:]]
                if not nb: break
                prev, cur = cur, nb[0]; path.append(cur)
                if len(path) > 2000: break
            if cur not in nid or nid[cur] == nid[p]: continue
            seen.add((cur, path[-2])); a, b = nid[p], nid[cur]
            cl = np.array([clr[q] for q in path]); L = len(path) * RES
            # door: pinch along the edge
            i = int(np.argmin(cl[1:-1])) + 1 if len(cl) > 2 else 0
            pa, pb = xy(path[0]), xy(path[-1]); hd = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
            dev = wrap(hd - theta_m); dev = (dev + math.pi / 4) % (math.pi / 2) - math.pi / 4
            snapped = abs(dev) <= math.atan2(float(cl.max()), max(L, RES))
            if Gr.has_edge(a, b) and Gr[a][b]["len"] <= L: continue
            Gr.add_edge(a, b, len=L, heading=hd - dev if snapped else hd, snapped=bool(snapped), path=[xy(q) for q in path[::4]] + [xy(path[-1])],
                        pinch=dict(xy=xy(path[i]), clear=float(cl[i])) if 0 < i and cl[i] < 0.5 * min(cl[0], cl[-1]) else None)
    # doors become nodes on their edge
    for a, b, d in list(Gr.edges(data=True)):
        if d["pinch"]:
            k = Gr.number_of_nodes() + 1000; Gr.add_node(k, xy=d["pinch"]["xy"], kind="door", clear=d["pinch"]["clear"])
            Gr.add_edge(a, k, **{**d, "len": d["len"] / 2, "pinch": None}); Gr.add_edge(k, b, **{**d, "len": d["len"] / 2, "pinch": None}); Gr.remove_edge(a, b)
    return Gr, clr, S


def gt_walls(L, shape, origin):
    import trimesh
    stl = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "sim", "casitahouse-walls-fixed.stl")
    m = trimesh.load(stl); ext = m.bounds; sc = 0.001 if (ext[1] - ext[0]).max() > 100 else 1.0; off = -ext[0] * sc
    sec = m.section(plane_origin=[0, 0, (0.5 - off[2]) / sc], plane_normal=[0, 0, 1]); occ = np.zeros(shape, np.uint8)
    for e in sec.entities:
        P = sec.vertices[e.points][:, :2] * sc + off[:2]; ij = np.round((P - origin) / RES - 0.5).astype(np.int32)
        cv2.polylines(occ, [ij.reshape(-1, 1, 2)], False, 1, 1)
    return occ.astype(bool)


def build(lg):
    L = Log(lg); N = L.nav(); sc = LR.Scans(L).cache(LR.cache_path(lg)); n = len(sc.f2)
    tn = N["t"]; pose = N["pose"]; th = np.unwrap(pose[:, 2])
    P = lambda t: (np.interp(t, tn, pose[:, 0]), np.interp(t, tn, pose[:, 1]), np.interp(t, tn, th))
    # est (map frame) -> GT frame: the start pose (map frame = this run's start pose)
    t0 = max(sc.f2[0][0], tn[0], L.gt_t[0]); g0 = L.gt(t0); e0 = P(t0); A = between(np.array(e0), np.array(g0))   # T_gt_est s.t. g = compose(e0->g0)
    def to_gt(p): return compose(np.array(g0), between(np.array(e0), np.array(p)))
    frames = []
    for i in range(n):
        t, pts, pl = sc.get(i)
        if t < t0 or t > min(tn[-1], L.gt_t[-1]) or len(pts) < 50: continue
        frames.append((t, to_gt(P(t)), L.gt(t), pts))
    allp = np.array([f[1][:2] for f in frames]); origin = allp.min(0) - 4.0; size = allp.max(0) + 4.0 - origin
    shape = (int(size[1] / RES) + 1, int(size[0] / RES) + 1); hit = np.zeros(shape, np.float32); fre = np.zeros(shape, np.float32)
    for t, p, g, pts in frames:
        W = transform(pts, tuple(p)); ij = np.floor((W - origin) / RES).astype(np.int32)
        np.add.at(hit, (ij[:, 1], ij[:, 0]), 1.0)
        r = np.hypot(pts[:, 0], pts[:, 1]); s = np.arange(0.05, 3.0, RES)
        F = (pts[:, None, :] / np.maximum(r, 1e-3)[:, None, None]) * s[None, :, None]; ok = s[None, :] < (r[:, None] - 0.10)
        Fw = transform(F[ok].astype(np.float32), tuple(p)); ij = np.floor((Fw - origin) / RES).astype(np.int32)
        np.add.at(fre, (ij[:, 1], ij[:, 0]), 1.0 / max(1, len(pts) // 200))
    occ = hit > np.maximum(fre, 0.5); free = (fre > 0) & ~cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return L, frames, occ, free, origin


def reloc(L, frames, Gr):
    """node relocalization on revisits."""
    tr = np.array([f[1][:2] for f in frames]); out = []
    for k, d in Gr.nodes(data=True):
        dist = np.hypot(*(tr - np.array(d["xy"])).T); inside = dist < NODE_R
        if not inside.any(): continue
        # visits = maximal runs inside, separated by leaving > 2 m
        visits = []; cur = None; far = True
        for i in range(len(frames)):
            if inside[i] and far: cur = [i]; visits.append(cur); far = False
            elif inside[i] and cur is not None: cur.append(i)
            if dist[i] > 2.0: far = True
        if len(visits) < 2: continue
        v0 = visits[0]; i0 = v0[int(np.argmin(dist[v0]))]; anchor = np.array(frames[i0][1])
        sub = GridMap(8.0, 0.02, (-4.0, -4.0))
        for i in v0: sub.integrate(tuple(between(anchor, np.array(frames[i][1]))), frames[i][3].astype(np.float32), np.zeros((0, 2), np.float32))
        sub.refresh_likelihood()
        for v in visits[1:]:
            i = v[int(np.argmin(dist[v]))]; t, p, g, pts = frames[i]; g0 = frames[i0][2]
            guess = between(anchor, np.array(p)); m = Matcher(sub, lin=0.3, lin_step=0.03, ang=math.radians(8), ang_step=math.radians(1), prior=0.0)
            Pm = pts[np.random.default_rng(0).choice(len(pts), min(1000, len(pts)), replace=False)].astype(np.float32)
            best, bs, med, s0 = m.match(Pm, tuple(guess)); ok = bs > med
            truth = between(np.array(g0), np.array(g))            # relative GT motion first visit -> now
            def err(rel): return float(math.hypot(rel[0] - truth[0], rel[1] - truth[1])), float(abs(math.degrees(wrap(rel[2] - truth[2]))))
            eb = err(guess); ea = err(np.array(best)) if ok else eb
            out.append(dict(node=int(k), kind=d["kind"], t=float(t), before_m=eb[0], before_deg=eb[1], after_m=ea[0], after_deg=ea[1], accepted=bool(ok)))
    return out


def crosses(path, wall, origin, rad):
    """True if a GT wall cell lies within rad of the polyline."""
    dm = cv2.distanceTransform((~wall).astype(np.uint8), cv2.DIST_L2, 5) * RES
    for a, b in zip(path[:-1], path[1:]):
        for s in np.linspace(0, 1, max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / RES) + 1)):
            x = a[0] + s * (b[0] - a[0]); y = a[1] + s * (b[1] - a[1]); j, i = int((x - origin[0]) / RES), int((y - origin[1]) / RES)
            if 0 <= i < wall.shape[0] and 0 <= j < wall.shape[1] and dm[i, j] < rad: return True
    return False


def evaluate(lg):
    L, frames, occ, free, origin = build(lg); th_m = manhattan(occ)
    Gr, clr, S = graph(free, th_m, origin)
    wall = gt_walls(L, occ.shape, origin)
    # GT graph: same pipeline on GT free space = explored area (dilated) minus GT walls
    expl = cv2.dilate(free.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    gfree = expl & ~cv2.dilate(wall.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    Gg, _, _ = graph(gfree, 0.0, origin)
    est = {k: np.array(d["xy"]) for k, d in Gr.nodes(data=True)}; gt = {k: np.array(d["xy"]) for k, d in Gg.nodes(data=True)}
    used = set(); matched = 0
    for k, x in gt.items():
        c = [(np.linalg.norm(x - y), j) for j, y in est.items() if j not in used]
        if c and min(c)[0] < MATCH_M: used.add(min(c)[1]); matched += 1
    wrong = [(a, b) for a, b, d in Gr.edges(data=True) if crosses(d["path"], wall, origin, 0.0 + RES)]
    rng = np.random.default_rng(0); nodes = list(Gr.nodes); routes = []
    comp = max(nx.connected_components(Gr), key=len) if len(Gr) else set()
    cn = list(comp)
    for _ in range(50 if len(cn) > 1 else 0):
        a, b = rng.choice(cn, 2, replace=False); pth = nx.shortest_path(Gr, a, b, weight="len")
        poly = []
        for u, v in zip(pth[:-1], pth[1:]):
            pp = Gr[u][v]["path"]; pp = pp if np.linalg.norm(np.array(pp[0]) - np.array(Gr.nodes[u]["xy"])) < np.linalg.norm(np.array(pp[-1]) - np.array(Gr.nodes[u]["xy"])) else pp[::-1]
            poly += pp
        routes.append(not crosses(poly, wall, origin, RR))
    rel = reloc(L, frames, Gr)
    q = lambda v: dict(n=len(v), p50=round(float(np.median(v)), 3), p95=round(float(np.percentile(v, 95)), 3)) if len(v) else dict(n=0)
    return dict(log=lg, manhattan_deg=round(math.degrees(th_m), 2), nodes=Gr.number_of_nodes(), kinds={k: sum(1 for _, d in Gr.nodes(data=True) if d["kind"] == k) for k in ("junction", "end", "door")},
                edges=Gr.number_of_edges(), snapped=sum(1 for *_, d in Gr.edges(data=True) if d["snapped"]), gt_nodes=len(gt), gt_matched=matched,
                wrong_edges=len(wrong), components=nx.number_connected_components(Gr), largest_component=len(cn),
                routes=dict(n=len(routes), valid=int(sum(routes))),
                reloc=dict(before_m=q([r["before_m"] for r in rel]), after_m=q([r["after_m"] for r in rel]), before_deg=q([r["before_deg"] for r in rel]),
                           after_deg=q([r["after_deg"] for r in rel]), accepted=int(sum(r["accepted"] for r in rel))),
                node_xy=[(round(float(x[0]), 2), round(float(x[1]), 2), Gr.nodes[k]["kind"]) for k, x in est.items()]), (occ, free, wall, S, Gr, Gg, origin)


if __name__ == "__main__":
    out = sys.argv[1]; R = []
    for lg in sys.argv[2:]:
        r, viz = evaluate(lg); R.append(r); print(json.dumps({k: v for k, v in r.items() if k != "node_xy"}), flush=True)
        occ, free, wall, S, Gr, Gg, origin = viz
        img = np.full(occ.shape + (3,), 255, np.uint8); img[free] = (235, 235, 235); img[wall] = (0, 0, 255); img[occ] = (0, 0, 0); img[S] = (0, 160, 0)
        for k, d in Gr.nodes(data=True):
            j, i = int((d["xy"][0] - origin[0]) / RES), int((d["xy"][1] - origin[1]) / RES)
            cv2.circle(img, (j, i), 4, {"junction": (255, 0, 0), "end": (0, 140, 255), "door": (200, 0, 200)}[d["kind"]], -1)
        cv2.imwrite(out.replace(".json", "_%s.png" % os.path.basename(lg.rstrip("/"))), img[::-1])
    json.dump(R, open(out, "w"), indent=1)
