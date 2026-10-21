#!/usr/bin/env python3
"""3D hull + SafetyGuard throttle diagram from live robot_config / safety.

Writes sim/rl_nav/out/throttle_zones.html (orbit, cm labels, click a part).
Re-run after changing robot_config.py or safety.py so the numbers stay honest.

  .venv-newton/bin/python sim/rl_nav/throttle_zones.py
  .venv-newton/bin/python sim/rl_nav/throttle_zones.py --serve
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import robot_config as rc  # noqa: E402


def _assign_floats(path: Path, names: tuple[str, ...]) -> dict[str, float]:
    text = path.read_text()
    out: dict[str, float] = {}
    for name in names:
        m = re.search(rf"^{name}\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+))", text, re.M)
        if m is None:
            raise SystemExit(f"{path.name}: missing {name}")
        out[name] = float(m.group(1))
    return out


def _cm_box(cx, cy, sx, sy):
    return {
        "cx": float(cx), "cy": float(cy),
        "sx": float(sx), "sy": float(sy),
        "x0": float(cx - 0.5 * sx), "x1": float(cx + 0.5 * sx),
        "y0": float(cy - 0.5 * sy), "y1": float(cy + 0.5 * sy),
    }


def model() -> dict:
    px = float(rc.EGO_PX_SIZE) * 100.0  # cm per ego pixel
    foot_fwd = (rc.FOOT_X1 - rc.RCX) * px
    foot_aft = (rc.RCX - rc.FOOT_X0) * px
    foot_lat = max(rc.RCY - rc.FOOT_Y0, rc.FOOT_Y1 - rc.RCY) * px
    sf = _assign_floats(
        _REPO / "src" / "safety.py",
        ("MIN_CLEARANCE_PX", "BRAKE_START_PX", "LAT_HARD_PX", "LAT_SAFE_PX"),
    )
    kn = _assign_floats(
        _REPO / "sim" / "rl_nav" / "kernels.py",
        (
            "SCAN_FWD_LX", "SCAN_BWD_LX", "SCAN_SIDE_LY", "SCAN_FACE_HALF",
            "ESCAPE_FWD", "ESCAPE_BWD", "ESCAPE_ANG_FLOOR", "ESCAPE_ANG_HARD",
        ),
    )
    stop_cm = sf["MIN_CLEARANCE_PX"] * px
    brake_cm = sf["BRAKE_START_PX"] * px
    lat_hard = sf["LAT_HARD_PX"] * px
    lat_safe = sf["LAT_SAFE_PX"] * px
    wheel_r = float(rc.WHEEL_DIAMETER_CM) * 0.5
    body_h = 15.0
    mast_z = 90.0
    return {
        "axle": "midpoint between wheel centers. +X forward, +Y left, +Z up.",
        "wheel_diameter_cm": float(rc.WHEEL_DIAMETER_CM),
        "wheel_radius_cm": wheel_r,
        "wheelbase_cm": float(rc.WHEELBASE_CM),
        "body": _cm_box(rc.BODY_CX_CM, rc.BODY_CY_CM, rc.BODY_LEN_CM, rc.BODY_WID_CM),
        "body_h_cm": body_h,
        "wheel_l": _cm_box(rc.WHEEL_L_CX_CM, rc.WHEEL_L_CY_CM, rc.WHEEL_LEN_CM, rc.WHEEL_WID_CM),
        "wheel_r": _cm_box(rc.WHEEL_R_CX_CM, rc.WHEEL_R_CY_CM, rc.WHEEL_LEN_CM, rc.WHEEL_WID_CM),
        "caster": _cm_box(rc.CASTER_CX_CM, rc.CASTER_CY_CM, rc.CASTER_LEN_CM, rc.CASTER_WID_CM),
        "wheel_box_note": "Self-mask is 20×10 cm (fatter than the 17×5 tire).",
        "mast": _cm_box(rc.MAST_CX_CM, rc.MAST_CY_CM, rc.MAST_LEN_CM, rc.MAST_WID_CM),
        "mast_cam_z_cm": mast_z,
        "mast_clear_cm": float(rc.MAST_CLEAR_CM),
        "foot_pad_fwd_cm": float(rc.FOOT_PAD_FWD),
        "foot_pad_bwd_cm": float(rc.FOOT_PAD_BWD),
        "foot_pad_lat_cm": float(rc.FOOT_PAD_LAT),
        "foot_fwd_cm": foot_fwd,
        "foot_aft_cm": foot_aft,
        "foot_lat_cm": foot_lat,
        "scan_fwd_cm": kn["SCAN_FWD_LX"] * 100.0,
        "scan_bwd_cm": abs(kn["SCAN_BWD_LX"]) * 100.0,
        "scan_side_cm": kn["SCAN_SIDE_LY"] * 100.0,
        "stop_cm": stop_cm,
        "brake_cm": brake_cm,
        "lat_hard_cm": lat_hard,
        "lat_safe_cm": lat_safe,
        "escape_fwd": kn["ESCAPE_FWD"],
        "escape_bwd": kn["ESCAPE_BWD"],
        "escape_ang_floor": kn["ESCAPE_ANG_FLOOR"],
        "escape_ang_hard": kn["ESCAPE_ANG_HARD"],
        "fwd_full_x_cm": foot_fwd + brake_cm,
        "bwd_full_x_cm": -(foot_aft + brake_cm),
        "lat_full_y_cm": foot_lat + lat_safe,
        "turn_r_cm": float(rc.TURN_R_M) * 100.0,
        "wander_clear_cm": float(rc.WANDER_CLEAR_CM),
        "ego_px_cm": px,
        "rcx": int(rc.RCX),
        "rcy": int(rc.RCY),
        "foot_x0": int(rc.FOOT_X0),
        "foot_x1": int(rc.FOOT_X1),
        "foot_y0": int(rc.FOOT_Y0),
        "foot_y1": int(rc.FOOT_Y1),
    }


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Kevin throttle zones</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #e8e6e3;
    font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; }
  #stage { position: absolute; inset: 0; }
  #ui { position: absolute; top: 10px; left: 10px; width: min(300px, 46vw);
    max-height: calc(100% - 20px); overflow: auto; z-index: 3;
    background: #161616e6; border: 1px solid #333; border-radius: 8px;
    padding: 12px 14px; box-sizing: border-box; }
  h1 { font-size: 15px; margin: 0 0 8px; font-weight: 600; }
  .sub { color: #aaa; margin: 0 0 10px; }
  table { width: 100%; border-collapse: collapse; margin: 8px 0 12px; }
  th, td { text-align: left; padding: 3px 4px; border-bottom: 1px solid #2a2a2a; }
  th { color: #888; font-weight: 500; }
  td.n { font-variant-numeric: tabular-nums; text-align: right; width: 88px; }
  .tog, .views { display: flex; flex-wrap: wrap; gap: 6px 10px; margin: 8px 0 4px; }
  .tog label { display: flex; gap: 5px; align-items: center; color: #ccc; cursor: pointer; }
  .views button { background: #2a2a2a; color: #ddd; border: 1px solid #444; border-radius: 4px;
    padding: 3px 8px; cursor: pointer; }
  .views button:hover { background: #333; }
  #hint { color: #8ab; margin-top: 8px; }
  #sel { margin-top: 8px; padding: 8px; background: #222; border-radius: 6px; min-height: 2.4em; }
  details { margin: 8px 0; }
  summary { cursor: pointer; color: #ccc; }
  canvas { display: block; }
  .lbl { color: #eee; font: 11px/1.2 ui-sans-serif, system-ui, sans-serif;
    text-shadow: 0 0 4px #000; pointer-events: none; white-space: nowrap; }
</style>
</head>
<body>
<div id="ui">
  <h1>Kevin hull + SafetyGuard</h1>
  <p class="sub">Axle origin. +X forward, +Y left, +Z up. All lengths in cm. Tape from the axle midpoint.</p>
  <div class="views">
    <button type="button" data-view="iso">3/4</button>
    <button type="button" data-view="top">top</button>
    <button type="button" data-view="side">side</button>
    <button type="button" data-view="front">front</button>
  </div>
  <div class="tog" id="togs"></div>
  <details>
    <summary>Tape measure (cm from axle)</summary>
    <table>
      <tr><th>from axle</th><th class="n">cm</th></tr>
      __ROWS__
    </table>
  </details>
  <div id="sel">Click a volume.</div>
  <p id="hint">Drag to orbit, scroll zoom, right-drag pan. Floor grid is 10 cm.</p>
</div>
<div id="stage"></div>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
const M = __MODEL__;
const CM = 0.01;

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const layers = {};
const pick = [];

const stage = document.getElementById("stage");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const cam = new THREE.PerspectiveCamera(50, 1, 0.02, 20);
cam.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
stage.appendChild(renderer.domElement);
const labelR = new CSS2DRenderer();
labelR.domElement.style.position = "absolute";
labelR.domElement.style.inset = "0";
labelR.domElement.style.pointerEvents = "none";
stage.appendChild(labelR.domElement);

const controls = new OrbitControls(cam, renderer.domElement);
const VIEWS = {
  iso:   { pos: [1.35, -1.70, 0.85], tgt: [0.0, 0.0, 0.10] },
  top:   { pos: [0.06, -0.08, 2.55], tgt: [0.0, 0.0, 0.0] },
  side:  { pos: [0.0, -2.05, 0.32],  tgt: [0.0, 0.0, 0.10] },
  front: { pos: [2.05, 0.0, 0.38],   tgt: [0.0, 0.0, 0.10] },
};
function setView(name) {
  const v = VIEWS[name] || VIEWS.iso;
  cam.position.set(...v.pos);
  controls.target.set(...v.tgt);
  cam.lookAt(controls.target);
  controls.update();
}
function fit() {
  const w = stage.clientWidth, h = stage.clientHeight;
  cam.aspect = Math.max(w, 1) / Math.max(h, 1);
  cam.updateProjectionMatrix();
  renderer.setSize(w, h);
  labelR.setSize(w, h);
}
setView("iso");
fit();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 0.85);
sun.position.set(0.6, -0.8, 1.2);
scene.add(sun);

const grid = new THREE.GridHelper(3.0, 30, 0x555555, 0x2a2a2a);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const axes = new THREE.AxesHelper(0.35);
scene.add(axes);

function boxMesh(sx, sy, sz, color, opacity, x, y, z, name, note) {
  const geo = new THREE.BoxGeometry(sx, sy, sz);
  const mat = new THREE.MeshLambertMaterial({
    color, transparent: opacity < 0.99, opacity, depthWrite: opacity > 0.4,
  });
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.userData = { name, note };
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }),
  );
  m.add(edges);
  pick.push(m);
  return m;
}

function addLayer(name, obj, on = true) {
  layers[name] = obj;
  obj.visible = on;
  scene.add(obj);
}

function label(text, x, y, z) {
  const el = document.createElement("div");
  el.className = "lbl";
  el.textContent = text;
  const o = new CSS2DObject(el);
  o.position.set(x, y, z);
  scene.add(o);
}

const body = M.body;
const zWheel = M.wheel_radius_cm * CM;
const zBody = zWheel + 0.02 + (M.body_h_cm * CM) * 0.5;
const hull = new THREE.Group();
hull.add(boxMesh(
  body.sx * CM, body.sy * CM, M.body_h_cm * CM, 0x2b6cb0, 0.78,
  body.cx * CM, body.cy * CM, zBody,
  "body",
  `${body.sx.toFixed(1)} × ${body.sy.toFixed(1)} × ${M.body_h_cm} cm. Center +${body.cx} cm. Front ${body.x1.toFixed(1)} / rear ${body.x0.toFixed(1)}.`,
));
function wheel(w, name) {
  const g = new THREE.CylinderGeometry(M.wheel_radius_cm * CM, M.wheel_radius_cm * CM, w.sy * CM, 24);
  g.rotateX(Math.PI / 2);
  const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0x333333 }));
  m.position.set(w.cx * CM, w.cy * CM, zWheel);
  m.userData = { name, note: `Physical Ø ${M.wheel_diameter_cm} cm. Self-mask box ${w.sx} × ${w.sy} cm at y=${w.cy}.` };
  pick.push(m);
  hull.add(m);
  const ghost = boxMesh(
    w.sx * CM, w.sy * CM, 0.04, 0x888888, 0.25,
    w.cx * CM, w.cy * CM, 0.025,
    name + " mask",
    M.wheel_box_note,
  );
  hull.add(ghost);
}
wheel(M.wheel_l, "left wheel");
wheel(M.wheel_r, "right wheel");
const caster = M.caster;
hull.add(boxMesh(
  caster.sx * CM, caster.sy * CM, 0.04, 0x4a4a4a, 0.85,
  caster.cx * CM, caster.cy * CM, 0.02,
  "caster",
  `Rim at x=${caster.x0.toFixed(1)} cm (200 mm aft of axle).`,
));
const mastH = M.mast_cam_z_cm * CM;
hull.add(boxMesh(
  M.mast.sx * CM, M.mast.sy * CM, mastH, 0x6b7280, 0.9,
  M.mast.cx * CM, M.mast.cy * CM, mastH * 0.5,
  "mast",
  `${M.mast.sx} × ${M.mast.sy} cm post at x=${M.mast.cx}. Cam ~${M.mast_cam_z_cm} cm. Tall-payload clear ${M.mast_clear_cm} cm.`,
));
addLayer("hull", hull, true);

const turn = new THREE.Group();
function ring(rCm, color, op, name, note) {
  const g = new THREE.CylinderGeometry(rCm * CM, rCm * CM, 0.004, 64);
  g.rotateX(Math.PI / 2);
  const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({
    color, transparent: true, opacity: op, depthWrite: false,
  }));
  m.position.set(0, 0, 0.003);
  m.userData = { name, note };
  pick.push(m);
  turn.add(m);
}
ring(M.turn_r_cm, 0x2ecc71, 0.18, "turn disk",
  `Clean in-place turn, ${M.turn_r_cm.toFixed(0)} cm from axle (caster rim).`);
ring(M.wander_clear_cm, 0xf1c40f, 0.12, "wander keep-clear",
  `Wander inflates obstacles by ${M.wander_clear_cm.toFixed(0)} cm. Gaps under ~${(2 * M.wander_clear_cm).toFixed(0)} cm are not entered.`);
addLayer("turn", turn, true);

const foot = new THREE.Group();
const fw = (M.foot_fwd_cm + M.foot_aft_cm) * CM;
const fl = M.foot_lat_cm * 2 * CM;
foot.add(boxMesh(
  fw, fl, 0.012, 0xffffff, 0.12,
  (M.foot_fwd_cm - M.foot_aft_cm) * 0.5 * CM, 0, 0.008,
  "FOOT AABB",
  `Scan hull. Front +${M.foot_fwd_cm.toFixed(1)} cm, aft −${M.foot_aft_cm.toFixed(1)} cm, side ±${M.foot_lat_cm.toFixed(1)} cm. Pads fwd/bwd/lat ${M.foot_pad_fwd_cm}/${M.foot_pad_bwd_cm}/${M.foot_pad_lat_cm} cm.`,
));
addLayer("foot", foot, true);

function slab(x0, x1, y0, y1, z0, z1, color, op, name, note) {
  const sx = Math.abs(x1 - x0), sy = Math.abs(y1 - y0), sz = Math.abs(z1 - z0);
  return boxMesh(
    sx * CM, sy * CM, sz * CM, color, op,
    (x0 + x1) * 0.5 * CM, (y0 + y1) * 0.5 * CM, (z0 + z1) * 0.5 * CM,
    name, note,
  );
}

const yS = M.foot_lat_cm;
const zBand = 6;
const fwd = new THREE.Group();
fwd.add(slab(M.foot_fwd_cm, M.foot_fwd_cm + M.stop_cm, -yS, yS, 0, zBand, 0xc0392b, 0.45,
  "fwd STOP", `scale=0. From FOOT front +${M.foot_fwd_cm.toFixed(1)} for ${M.stop_cm} cm (live MIN_CLEARANCE).`));
fwd.add(slab(M.foot_fwd_cm + M.stop_cm, M.foot_fwd_cm + M.brake_cm, -yS, yS, 0, zBand, 0xe67e22, 0.28,
  "fwd BRAKE", `smoothstep 0→1 over ${M.brake_cm} cm (BRAKE_START). Full +v at x=+${M.fwd_full_x_cm.toFixed(1)} cm.`));
addLayer("fwd", fwd, true);

const bwd = new THREE.Group();
bwd.add(slab(-M.foot_aft_cm - M.stop_cm, -M.foot_aft_cm, -yS, yS, 0, zBand, 0xc0392b, 0.45,
  "bwd STOP", `scale=0. From FOOT aft −${M.foot_aft_cm.toFixed(1)} for ${M.stop_cm} cm (caster pad).`));
bwd.add(slab(-M.foot_aft_cm - M.brake_cm, -M.foot_aft_cm - M.stop_cm, -yS, yS, 0, zBand, 0xe67e22, 0.28,
  "bwd BRAKE", `smoothstep 0→1 over ${M.brake_cm} cm. Full −v at x=${M.bwd_full_x_cm.toFixed(1)} cm.`));
addLayer("bwd", bwd, true);

const lat = new THREE.Group();
const x0 = -M.foot_aft_cm, x1 = M.foot_fwd_cm;
lat.add(slab(x0, x1, yS, yS + M.lat_hard_cm, 0, zBand, 0xc0392b, 0.40,
  "left HARD", `ang=0 within ${M.lat_hard_cm} cm of FOOT side (+${yS.toFixed(1)}).`));
lat.add(slab(x0, x1, yS + M.lat_hard_cm, yS + M.lat_safe_cm, 0, zBand, 0xf1c40f, 0.28,
  "left SAFE", `ang ramps to 1 by ${M.lat_safe_cm} cm. Full yaw at y=+${(yS + M.lat_safe_cm).toFixed(1)}.`));
lat.add(slab(x0, x1, -(yS + M.lat_hard_cm), -yS, 0, zBand, 0xc0392b, 0.40,
  "right HARD", `ang=0 within ${M.lat_hard_cm} cm of FOOT side (−${yS.toFixed(1)}).`));
lat.add(slab(x0, x1, -(yS + M.lat_safe_cm), -(yS + M.lat_hard_cm), 0, zBand, 0xf1c40f, 0.28,
  "right SAFE", `ang ramps to 1 by ${M.lat_safe_cm} cm.`));
addLayer("lat", lat, true);

const esc = new THREE.Group();
esc.add(boxMesh(0.08, 0.08, 0.08, 0x9b59b6, 0.55, 0, 0, 0.28,
  "escape spin",
  `If fwd_scale < ${M.escape_fwd} or bwd_scale < ${M.escape_bwd}, ang ≥ ${M.escape_ang_floor} (or ${M.escape_ang_hard} if laterals ≤ ${M.lat_hard_cm} cm).`));
addLayer("escape", esc, true);

label("axle", 0, 0, 0.20);
label(`+${M.foot_fwd_cm.toFixed(0)} FOOT`, M.foot_fwd_cm * CM, 0, 0.12);
label(`−${M.foot_aft_cm.toFixed(0)} FOOT`, -M.foot_aft_cm * CM, 0, 0.12);
label(`+${M.fwd_full_x_cm.toFixed(0)} full +v`, M.fwd_full_x_cm * CM, 0, 0.08);
label(`${M.bwd_full_x_cm.toFixed(0)} full −v`, M.bwd_full_x_cm * CM, 0, 0.08);
label(`+${M.foot_lat_cm.toFixed(0)} FOOT`, 0, M.foot_lat_cm * CM, 0.10);
label(`+${(M.foot_lat_cm + M.lat_safe_cm).toFixed(0)} full w`, 0, (M.foot_lat_cm + M.lat_safe_cm) * CM, 0.08);

document.querySelector(".views").addEventListener("click", (e) => {
  const name = e.target.dataset.view;
  if (name) setView(name);
});

const names = ["hull", "turn", "foot", "fwd", "bwd", "lat", "escape"];
const togs = document.getElementById("togs");
for (const n of names) {
  const lab = document.createElement("label");
  lab.innerHTML = `<input type="checkbox" ${layers[n].visible ? "checked" : ""} data-k="${n}"/> ${n}`;
  togs.appendChild(lab);
}
togs.addEventListener("change", (e) => {
  const k = e.target.dataset.k;
  if (k && layers[k]) layers[k].visible = e.target.checked;
});

const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const sel = document.getElementById("sel");
renderer.domElement.addEventListener("click", (ev) => {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
  mouse.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
  ray.setFromCamera(mouse, cam);
  const hit = ray.intersectObjects(pick, false)[0];
  if (!hit) { sel.textContent = "Click a volume."; return; }
  const u = hit.object.userData;
  sel.innerHTML = `<b>${u.name}</b><br>${u.note || ""}`;
});

addEventListener("resize", fit);

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene, cam);
  labelR.render(scene, cam);
}
tick();
</script>
</body>
</html>
"""


def _rows(m: dict) -> str:
    items = [
        ("body front / rear", f"+{m['body']['x1']:.1f} / {m['body']['x0']:.1f}"),
        ("body left / right", f"±{m['body']['sy'] * 0.5:.1f}"),
        ("wheelbase (L–R hubs)", f"{m['wheelbase_cm']:.1f}"),
        ("wheel Ø (physical)", f"{m['wheel_diameter_cm']:.2f}"),
        ("wheel mask L/R y", f"±{m['wheel_l']['cy']:.1f}"),
        ("caster aft", f"{m['caster']['x0']:.1f}"),
        ("turn / wander-clear r", f"{m['turn_r_cm']:.0f} / {m['wander_clear_cm']:.0f}"),
        ("FOOT front (scan +x)", f"+{m['foot_fwd_cm']:.1f}"),
        ("FOOT aft (scan −x)", f"−{m['foot_aft_cm']:.1f}"),
        ("FOOT side (scan ±y)", f"±{m['foot_lat_cm']:.1f}"),
        ("sim SCAN matches FOOT", f"{m['scan_fwd_cm']:.1f} / {m['scan_bwd_cm']:.1f} / {m['scan_side_cm']:.1f}"),
        ("fwd/bwd STOP (scale=0)", f"{m['stop_cm']:.1f}"),
        ("fwd/bwd BRAKE (scale=1)", f"{m['brake_cm']:.1f}"),
        ("full +v at x", f"+{m['fwd_full_x_cm']:.1f}"),
        ("full −v at x", f"{m['bwd_full_x_cm']:.1f}"),
        ("lat HARD / SAFE", f"{m['lat_hard_cm']:.1f} / {m['lat_safe_cm']:.1f}"),
        ("full ±w at |y|", f"{m['lat_full_y_cm']:.1f}"),
        ("ego px / axle", f"{m['ego_px_cm']:.0f} cm · ({m['rcx']},{m['rcy']})"),
        ("FOOT px box", f"x {m['foot_x0']}–{m['foot_x1']}  y {m['foot_y0']}–{m['foot_y1']}"),
    ]
    return "\n".join(
        f"<tr><td>{k}</td><td class='n'>{v}</td></tr>" for k, v in items
    )


def write_html(path: Path) -> Path:
    m = model()
    html = _HTML.replace("__MODEL__", json.dumps(m)).replace("__ROWS__", _rows(m))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="html path (default out/throttle_zones.html)")
    ap.add_argument("--serve", action="store_true", help="HTTP on :8766 and open browser")
    ap.add_argument("--open", action="store_true", help="open the html in a browser")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "out" / "throttle_zones.html"
    write_html(out)
    print("wrote", out)
    m = model()
    print("FOOT +%.1f / −%.1f / ±%.1f cm  STOP %.1f  BRAKE %.1f" % (
        m["foot_fwd_cm"], m["foot_aft_cm"], m["foot_lat_cm"], m["stop_cm"], m["brake_cm"]))
    if args.serve:
        root = out.parent

        class H(SimpleHTTPRequestHandler):
            def __init__(self, *a, **k):
                super().__init__(*a, directory=str(root), **k)

            def log_message(self, *_a):
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), H)
        url = f"http://127.0.0.1:{args.port}/{out.name}"
        print("open", url)
        webbrowser.open(url)
        httpd.serve_forever()
    elif args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
