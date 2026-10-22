#!/usr/bin/env python3
"""Draw a fitted casita pose over the mast frame: model render at 20 % opacity, model key lines RED, matched
image segments BLUE (held-out 'check' lines dashed red / cyan), every other visible model edge thin red."""
from __future__ import annotations
import argparse, json, os, sys, numpy as np, cv2
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camgeo import Cam, load_casita, raycast, segment_visible

RED, BLUE, CYAN = (0, 0, 255), (255, 80, 0), (255, 255, 0)


def model_edges(m, min_deg=30.0, min_len=0.05):
    # drop edges of sliver triangles (zero-area faces the STL repair left, they draw long fake rays)
    T = m.triangles; e = np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2).max(1)
    sliver = m.area_faces < 1e-3 * e ** 2
    adj = m.face_adjacency; keep = (m.face_adjacency_angles > np.radians(min_deg)) & ~sliver[adj[:, 0]] & ~sliver[adj[:, 1]]
    E = m.face_adjacency_edges[keep]
    good = np.repeat(~sliver, 3)
    cnt = Counter(map(tuple, np.sort(m.edges[good], 1))); bnd = np.array([e for e, c in cnt.items() if c == 1]).reshape(-1, 2)
    E = np.vstack([E, bnd]).astype(int); V = m.vertices
    return [(V[a], V[b]) for a, b in E if np.linalg.norm(V[b] - V[a]) >= min_len]


def draw_3d(o, m, cam, A, B, col, th, n=80, dashed=False, occlude=True):
    P, vis = segment_visible(m, cam, np.asarray(A, float), np.asarray(B, float), n=n)
    uv, ok = cam.project(P, valid=True)
    vis = (vis if occlude else np.ones(len(P), bool)) & ok
    for i in range(len(uv) - 1):
        if vis[i] and vis[i + 1] and (not dashed or i % 2 == 0):
            cv2.line(o, tuple(np.int32(np.round(uv[i]))), tuple(np.int32(np.round(uv[i + 1]))), col, th, cv2.LINE_AA)


def render(img, m, cam, ann, fill=True, all_edges=True, title=""):
    H, W = img.shape[:2]; o = img.copy()
    if fill:
        dist, fid = raycast(m, cam, step=2)
        N = m.face_normals; col = np.zeros(fid.shape + (3,), np.uint8); wall = fid >= 0
        nn = N[fid[wall]]
        # walls: colour by facing direction (|nx| red-ish, |ny| green-ish), floor brown, ceiling pale
        col[wall] = np.stack([60 + 150 * np.abs(nn[:, 2]), 60 + 170 * np.abs(nn[:, 1]), 60 + 170 * np.abs(nn[:, 0])], -1).astype(np.uint8)
        col[fid == -1] = (40, 90, 140); col[fid == -2] = (230, 230, 200)
        col = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize((fid != -3).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        o[mask] = (0.8 * o[mask] + 0.2 * col[mask]).astype(np.uint8)
    if all_edges:
        for A, B in model_edges(m): draw_3d(o, m, cam, A, B, (0, 0, 200), 1, n=30)
    for ln in ann["lines"]:
        chk = ln["role"] != "fit"
        draw_3d(o, m, cam, ln["A"], ln["B"], RED, 3, dashed=chk, occlude=False)
        for s in ln["segs"]:
            cv2.line(o, (int(round(s[0])), int(round(s[1]))), (int(round(s[2])), int(round(s[3]))), CYAN if chk else BLUE, 3, cv2.LINE_AA)
    cv2.rectangle(o, (0, 0), (W, 100), (0, 0, 0), -1)
    y = 28
    for t in [title, "RED = casita model key lines (solid: used in fit, dashed: held-out check)",
              "BLUE = detected image segments used in fit, CYAN = held-out" + (", fill = model render @20%" if fill else "")]:
        cv2.putText(o, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA); y += 32
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--annot", required=True); ap.add_argument("--stl", default=None, help="default: the resolved world mesh (src/world_params.py)")
    ap.add_argument("--report", required=True); ap.add_argument("--stage", default="A"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    img = cv2.imread(a.image); H, W = img.shape[:2]; m = load_casita(a.stl); ann = json.load(open(a.annot))
    rep = json.load(open(a.report)); st = next(v for k, v in rep["stages"].items() if k.startswith(a.stage))
    cam = Cam(st["params"], W, H); p = st["params"]
    title = "casita fit %s: cam x %.2f y %.2f z %.2f m yaw %.1f pitch %.1f roll %.1f deg f %.0f px | rms %.1f px" % (
        a.stage, p["x"], p["y"], p["z"], np.degrees(p["yaw"]), np.degrees(p["pitch"]), np.degrees(p["roll"]), p["f"], st["fit_rms_px"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, render(img, m, cam, ann, True, True, title))
    cv2.imwrite(a.out.replace(".png", "_lines.png"), render(img, m, cam, ann, False, False, title))


if __name__ == "__main__":
    main()
