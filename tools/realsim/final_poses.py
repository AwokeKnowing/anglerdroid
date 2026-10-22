"""Assemble the camera poses of the real reference looks from the joint group fits (local_pose.py outputs) and
write poses.json (+ an overlay montage: red = start pose from odometry chain, green = fitted) and the ceiling-lamp
positions measured by back-projecting saturated blobs of the mast frames onto the ceiling plane.
usage: final_poses.py <poses_dir> <pilot_dir>"""
import json, math, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poses as P  # noqa: E402
import world_params as wp  # noqa: E402

O, PIL = sys.argv[1], sys.argv[2]
mesh = P.camgeo.load_casita(); CZ = float(wp.V["ceiling_z"])
out = {"looks": {}, "lamps_obs": []}; tiles = []
for g in ("locA", "locB"):
    G = json.load(open(os.path.join(O, g + ".json")))
    for d, o in zip(G["looks"], G["rel_yaw"]):
        name = os.path.basename(d); b0 = (G["start"][0], G["start"][1], math.radians(G["start"][2]) + o)
        b = (G["base"][0], G["base"][1], math.radians(G["base"][2]) + o)
        # per-look heading: odometry yaw increments of in-place turns on carpet are not exact -> refine this look's
        # heading alone (position shared by the group), +-20 deg, coarse-to-fine, same LSD chamfer score
        DT = P.real_dt(cv2.imread(os.path.join(d, "mast_rgb.png"))); Xp = None; best = (P.score(mesh, b, DT), 0.0)
        for span, stp in ((math.radians(20), math.radians(2)), (math.radians(2.5), math.radians(0.5)), (math.radians(0.6), math.radians(0.1))):
            c0 = best[1]
            for dyw in np.arange(c0 - span, c0 + span + 1e-9, stp):
                v = P.score(mesh, (b[0], b[1], b[2] + dyw), DT)
                if v < best[0]: best = (v, float(dyw))
        s_odo = P.score(mesh, b, DT); b = (b[0], b[1], b[2] + best[1])
        out["looks"][name] = dict(group=g, base=[b[0], b[1], math.degrees(b[2])], dyaw_look_deg=round(math.degrees(best[1]), 2), score_px=[round(s_odo, 2), round(best[0], 2)], mast=P.mast_cam(b), rs2=P.rs2_cam(b), dir=d)
        img = cv2.imread(os.path.join(d, "mast_rgb.png"))
        ov = P.overlay(P.overlay(img, mesh, P.mast_cam(b0), (0, 0, 255)), mesh, P.mast_cam(b), (0, 255, 0))
        cv2.putText(ov, name[9:], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tiles.append(cv2.resize(ov, (400, 300)))
        # lamps: saturated blobs in the upper half -> ceiling plane
        cam = P.camgeo.Cam(P.mast_cam(b), 800, 600); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 2] >= 250) & (hsv[..., 1] < 60)).astype(np.uint8); m[330:] = 0
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            if st[k, 4] < 60: continue
            xn, yn = cam.pix_to_norm(cen[k]); dw = np.array([xn, yn, 1.0]) @ cam.R
            if dw[2] <= 0.05: continue
            s = (CZ - cam.t[2]) / dw[2]; X = cam.t + s * dw
            out["lamps_obs"].append(dict(look=name, px=[round(float(cen[k][0])), round(float(cen[k][1]))], area=int(st[k, 4]), xy=[round(float(X[0]), 2), round(float(X[1]), 2)], range=round(float(s), 2)))
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(os.path.join(O, "fit_overlays.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(out, open(os.path.join(O, "poses.json"), "w"), indent=1)
for l in out["lamps_obs"]: print(l)
