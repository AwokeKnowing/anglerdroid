"""Ground-truth check of nav's planned path (and LightNav's proposed waypoints) against the sim's true walls.
Eval tool only (talks to the running viewer :8095 and nav :8090; robot code never imports it).

  python emu/tools/pathcheck.py [out.png] [--watch SECS]

Draws, in the env (ground-truth) frame: true obstacles (walls mesh sliced at 0.05-0.6 m + box footprints), nav's SLAM
map moved into the env frame with the same nav->env relation the viewer uses, nav's global path, LightNav's last
waypoint chunk, and the robot.  Prints how much of the path lies inside true obstacles / within the robot half-width."""
from __future__ import annotations
import io, json, math, sys, time, urllib.request
import numpy as np, cv2

RES = 0.02; HALF_W = 0.207


def get(u): return urllib.request.urlopen(u, timeout=5).read()


def gt_occupancy(scene, z_lo=0.05, z_hi=0.6):
    x0, y0, x1, y1 = scene["extent"]; x0 -= 0.5; y0 -= 0.5; x1 += 0.5; y1 += 0.5
    W, H = int((x1 - x0) / RES), int((y1 - y0) / RES); occ = np.zeros((H, W), np.uint8)
    px = lambda P: np.round((np.asarray(P)[:, :2] - [x0, y0]) / RES).astype(np.int32)
    for g in scene["geoms"]:
        if g["kind"] == "mesh" and g.get("vertices"):
            import trimesh
            m = trimesh.Trimesh(np.array(g["vertices"]), np.array(g["faces"]), process=False)
            for z in np.arange(z_lo, z_hi, 0.1):
                s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
                if s is None: continue
                for seg in s.discrete: cv2.polylines(occ, [px(seg)], False, 1, 2)
        elif g["kind"] == "box":
            (cx, cy, cz), (sx, sy, sz), a = g["pos"], g["size"], g["yaw"]
            if cz - sz > z_hi or cz + sz < z_lo: continue          # above/below the robot's body band
            c, s = math.cos(a), math.sin(a)
            P = np.array([[cx + c * u - s * v, cy + s * u + c * v] for u, v in ((sx, sy), (-sx, sy), (-sx, -sy), (sx, -sy))])
            cv2.fillPoly(occ, [px(P)], 1)
    return occ.astype(bool), (x0, y0, x1, y1)


def nav2env(kevin, navpose):
    gx, gy, ga = kevin; nx, ny, na = navpose
    def f(x, y):
        dx, dy = x - nx, y - ny; c, s = math.cos(-na), math.sin(-na); u, v = c * dx - s * dy, s * dx + c * dy
        c, s = math.cos(ga), math.sin(ga); return gx + c * u - s * v, gy + s * u + c * v
    return f


def check(out="/tmp/pathcheck.png", viewer="http://localhost:8095", nav="http://localhost:8090"):
    scene = json.loads(get(viewer + "/scene.json")); st = json.loads(get(viewer + "/state.json"))
    n = json.loads(get(nav + "/api/state")); meta = json.loads(get(nav + "/api/map.json"))
    mp = cv2.imdecode(np.frombuffer(get(nav + "/api/map.png"), np.uint8), 0)[::-1]      # row 0 = y0
    occ, (x0, y0, x1, y1) = gt_occupancy(scene)
    S = json.load(open("/dev/shm/kevin_sim.json")); f = nav2env(S["kevin"], n["pose"])
    H, W = occ.shape; img = np.full((H, W, 3), 235, np.uint8)
    # nav map -> env frame (inverse map every env pixel into the nav map image)
    gx, gy, ga = S["kevin"]; nx, ny, na = n["pose"]; d = na - ga; c, s = math.cos(d), math.sin(d)
    YY, XX = np.mgrid[0:H, 0:W]; ex = x0 + (XX + 0.5) * RES - gx; ey = y0 + (YY + 0.5) * RES - gy
    mx = nx + c * ex - s * ey; my = ny + s * ex + c * ey; mres = (meta["x1"] - meta["x0"]) / mp.shape[1]
    ci = ((mx - meta["x0"]) / mres).astype(int); ri = ((my - meta["y0"]) / mres).astype(int)
    ok = (ci >= 0) & (ci < mp.shape[1]) & (ri >= 0) & (ri < mp.shape[0]); v = np.full((H, W), 60, np.uint8); v[ok] = mp[ri[ok], ci[ok]]
    img[v == 200] = (255, 255, 255); img[v == 20] = (150, 150, 255)              # nav: seen free white, occupied pink
    img[occ] = (40, 40, 40)                                                      # truth: dark
    P = lambda xy: (int((xy[0] - x0) / RES), int((xy[1] - y0) / RES))
    res = dict(t=time.strftime("%H:%M:%S"), mode=n.get("mode"), msg=n.get("msg"))
    path = n.get("path") or []
    if path:
        E = np.array([f(*p[:2]) for p in path])
        # densify and measure clearance to true obstacles
        seg = np.concatenate([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 0.01))) for a, b in zip(E[:-1], E[1:])]) if len(E) > 1 else E
        dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * RES
        ij = np.clip(((seg - [x0, y0]) / RES).astype(int), 0, [W - 1, H - 1]); cl = dist[ij[:, 1], ij[:, 0]]
        res.update(path_pts=len(path), path_len_m=round(float(np.sum(np.hypot(*np.diff(E, axis=0).T))), 2) if len(E) > 1 else 0,
                   inside_wall_m=round(float((cl < 0.02).sum() * 0.01), 2), min_clear_m=round(float(cl.min()), 3),
                   under_halfwidth_m=round(float((cl < HALF_W).sum() * 0.01), 2))
        cv2.polylines(img, [np.array([P(p) for p in E], np.int32)], False, (0, 160, 0), 2)
    V = n.get("vln") or {}
    for k in (V.get("sets") or [])[-1:]:
        if k.get("wp_map"): cv2.polylines(img, [np.array([P(f(*p[:2])) for p in k["wp_map"]], np.int32)], False, (220, 120, 0), 2)
    if n.get("goal"): cv2.circle(img, P(f(*n["goal"][:2])), 5, (0, 0, 220), -1)
    kx, ky, ka = S["kevin"]; cv2.circle(img, P((kx, ky)), int(HALF_W / RES), (200, 0, 0), 2)
    cv2.line(img, P((kx, ky)), P((kx + 0.4 * math.cos(ka), ky + 0.4 * math.sin(ka))), (200, 0, 0), 2)
    img = np.ascontiguousarray(img[::-1]); cv2.putText(img, json.dumps(res)[:110], (5, 15), 0, 0.4, (0, 0, 0), 1)
    cv2.imwrite(out, img); return res


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]; out = a[0] if a else "/tmp/pathcheck.png"
    print(json.dumps(check(out)))
