#!/usr/bin/env python3
"""gate_caster_holster.stl — USCaster-style trailing U-fork (2 mm cheeks, 25 mm trail).

Fork frame: yaw shaft at origin; wheel axle at (-0.025, 0, -0.05588).
Side silhouette: rounded axle bottom, nearly-vertical front, and a continuous
~60° (from horizontal) diagonal neck from the swivel collar down/aft into the
trailing U-fork — no flat-deck-then-vertical stairstep.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import box, cylinder, extrude_polygon

try:
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
except ImportError as e:
    raise SystemExit("shapely is required") from e

# --- design constants (meters) ---
AXLE = np.array([-0.02500, 0.0, -0.05588], dtype=float)
WHEEL_W = 0.01981
WHEEL_R = 0.02540
PLATE_T = 0.0020          # side cheek thickness (exact)
CLEARANCE = 0.0025        # gap cheek-inner → wheel face
INNER_Y = WHEEL_W / 2.0 + CLEARANCE   # 0.012405
OUTER_Y = INNER_Y + PLATE_T           # 0.014405

# Rounded cheek end around axle (reference: ~10–12 mm OD, metal under bolt)
R_CHEEK = 0.0115
BOSS_R = 0.0070
BOSS_LEN = CLEARANCE - 0.0003         # stop short of wheel face
AXLE_HOLE_R = 0.0040                  # bolt clearance through cheeks

WHEEL_TOP = float(AXLE[2] + WHEEL_R)  # -0.03048
TIRE_CLEAR = 0.0040                   # ≥3–4 mm air under bridge/neck
BRIDGE_Z_BOT = WHEEL_TOP + TIRE_CLEAR  # -0.02648
BRIDGE_T = 0.0045                     # nominal neck thickness above tire

# Swivel collar / housing (wider than shaft, rounded)
COLLAR_R = 0.0120
COLLAR_Z1 = 0.0025                    # slightly proud of z=0
COLLAR_BLEND_R = min(OUTER_Y - 0.0002, COLLAR_R + 0.0025)
SHAFT_BORE_R = 0.0060

# Cheek front X at mid-height (nearly vertical face ahead of trail)
FRONT_X = 0.0120

# --- ~60° diagonal neck (from horizontal): down/aft from collar into fork ---
# Line: z = DIAG_Z0 + (x - DIAG_X0) * tan(DIAG_DEG)
# Anchored at collar rear near collar top so silhouette blends into cylinder,
# and at axle X the top stays high enough for BRIDGE_T above tire clearance.
DIAG_DEG = 60.0
DIAG_TAN = float(np.tan(np.deg2rad(DIAG_DEG)))
DIAG_X0 = -COLLAR_R                   # rear of collar cylinder
# Choose DIAG_Z0 so z(AXLE[0]) ≈ BRIDGE_Z_BOT + BRIDGE_T
_z_at_axle_target = BRIDGE_Z_BOT + BRIDGE_T  # ≈ -0.02198
DIAG_Z0 = float(_z_at_axle_target - (AXLE[0] - DIAG_X0) * DIAG_TAN)
# → DIAG_Z0 ≈ 0.00054 (into collar body)


def diag_z(x: float) -> float:
    """Top-of-neck Z on the 60° diagonal (valid for x <= DIAG_X0; flat forward)."""
    return DIAG_Z0 + (float(x) - DIAG_X0) * DIAG_TAN


def top_z(x: float) -> float:
    """Outer top silhouette: flat under collar footprint, 60° diagonal aft."""
    x = float(x)
    if x >= DIAG_X0:
        return DIAG_Z0
    return diag_z(x)


# Collar sits down into the neck
COLLAR_Z0 = min(BRIDGE_Z_BOT + 0.0010, DIAG_Z0 - 0.0060)


def _cheek_polygon_xz() -> Polygon:
    """USCaster-like cheek in XZ: round bottom + 60° diagonal top-aft + vertical front.

    Built CCW (outward for +Y extrusion remap): bottom arc (front→under→rear)
    then up the trailing back to the diagonal, along diagonal / flat top to front,
    down the front to the arc.
    """
    ax, az = float(AXLE[0]), float(AXLE[2])
    R = R_CHEEK

    th_front = np.deg2rad(25.0)
    th_rear = np.deg2rad(155.0)
    n_arc = 48
    th_rear_cw = th_rear - 2.0 * np.pi  # clockwise via bottom
    thetas = np.linspace(th_front, th_rear_cw, n_arc)
    arc = [(ax + R * np.cos(t), az + R * np.sin(t)) for t in thetas]

    rear_x = ax + R * np.cos(th_rear)  # ≈ ax - R*cos(25°) ≈ -0.0354
    rear_z_arc = az + R * np.sin(th_rear)
    z_rear_top = top_z(rear_x)        # diagonal meet (~ -0.040)

    # Sample diagonal from rear up to collar rear, then flat to front
    n_diag = 24
    xs_diag = np.linspace(rear_x, DIAG_X0, n_diag)
    diag_pts = [(float(x), top_z(float(x))) for x in xs_diag]

    top_front_z = DIAG_Z0 + 0.0003
    pts = []
    pts.extend(arc)
    # short climb from arc rear onto diagonal (slightly flared aft)
    pts.append((rear_x - 0.0008, 0.5 * (rear_z_arc + z_rear_top)))
    pts.append((rear_x - 0.0005, z_rear_top))
    pts.extend(diag_pts[1:])  # skip duplicate rear
    # flat top under collar / ahead of shaft
    pts.append((0.0, top_front_z))
    pts.append((FRONT_X, top_front_z))
    # front nearly vertical down toward arc meet
    pts.append((FRONT_X, BRIDGE_Z_BOT))
    pts.append((FRONT_X - 0.0010, az + R * 0.55))

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.area <= 0:
        raise RuntimeError(f"invalid cheek polygon: valid={poly.is_valid} area={poly.area}")
    return poly


def _neck_polygon_xz() -> Polygon:
    """Solid diagonal neck web (XZ): clears tire underside, 60° top into collar.

    Spans from ahead of shaft back to where the diagonal meets the tire-clear
    deck — continuous wedge, no horizontal-then-vertical stair.
    """
    # Aft end of web: where diagonal top is just above underside (+ min thickness)
    min_t = 0.0012
    # Solve DIAG_Z0 + (x - DIAG_X0)*DIAG_TAN = BRIDGE_Z_BOT + min_t
    x_aft = DIAG_X0 + (BRIDGE_Z_BOT + min_t - DIAG_Z0) / DIAG_TAN
    # Keep web from eating into cheek rear arc region
    x_aft = max(x_aft, float(AXLE[0]) - 0.0040)  # ~ -0.029

    z_aft_top = top_z(x_aft)
    z_flat = DIAG_Z0 + 0.0003

    pts = [
        (FRONT_X, BRIDGE_Z_BOT),
        (x_aft, BRIDGE_Z_BOT),
        (x_aft, z_aft_top),
    ]
    # along diagonal / flat top forward
    xs = np.linspace(x_aft, DIAG_X0, 16)
    for x in xs[1:]:
        pts.append((float(x), top_z(float(x))))
    pts.append((0.0, z_flat))
    pts.append((FRONT_X, z_flat))

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.area <= 0:
        raise RuntimeError(f"invalid neck polygon: valid={poly.is_valid} area={poly.area}")
    return poly


def _extrude_xz(poly: Polygon, y0: float, y1: float) -> trimesh.Trimesh:
    """Extrude XZ polygon along +Y from y0 to y1."""
    height = float(y1 - y0)
    mesh = extrude_polygon(poly, height=height)
    # extrude_polygon: (x, poly_y=z, extrude_z) → remap to (x, y0+e, z)
    R = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    mesh.apply_transform(R)
    mesh.apply_translation([0.0, y0, 0.0])
    return mesh


def _extrude_cheek(y0: float, y1: float) -> trimesh.Trimesh:
    height = float(y1 - y0)
    assert abs(height - PLATE_T) < 1e-9, f"cheek thickness {height} != {PLATE_T}"
    return _extrude_xz(_cheek_polygon_xz(), y0, y1)


def _diagonal_neck() -> trimesh.Trimesh:
    """Full-width solid neck: 60° top, tire-clear underside, blends to collar."""
    # Span cheek outers so union with cheeks is seamless
    return _extrude_xz(_neck_polygon_xz(), -OUTER_Y, +OUTER_Y)


def _swivel_collar() -> trimesh.Trimesh:
    """Short fat rounded housing at yaw origin (wider than shaft)."""
    h = COLLAR_Z1 - COLLAR_Z0
    m = cylinder(radius=COLLAR_R, height=h, sections=48)
    m.apply_translation([0.0, 0.0, (COLLAR_Z0 + COLLAR_Z1) / 2.0])
    # Soften transition into neck: wider disc near DIAG_Z0
    blend_h = 0.0040
    blend = cylinder(radius=COLLAR_BLEND_R, height=blend_h, sections=48)
    blend.apply_translation([0.0, 0.0, DIAG_Z0 - 0.0005])
    for p in (m, blend):
        if p.volume < 0:
            p.invert()
        trimesh.repair.fix_normals(p)
    return m.union(blend, engine="manifold")


def _boss(y_inner: float, outward_sign: float) -> trimesh.Trimesh:
    """Short axle boss protruding inward from cheek toward wheel face."""
    m = cylinder(radius=BOSS_R, height=BOSS_LEN, sections=32)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cy = y_inner - outward_sign * (BOSS_LEN / 2.0)
    m.apply_translation([AXLE[0], cy, AXLE[2]])
    return m


def _axle_hole_cutter() -> trimesh.Trimesh:
    """Cylinder through both cheeks for axle bolt."""
    h = 2.0 * OUTER_Y + 0.004
    m = cylinder(radius=AXLE_HOLE_R, height=h, sections=32)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation([AXLE[0], 0.0, AXLE[2]])
    return m


def _clean(p: trimesh.Trimesh) -> trimesh.Trimesh:
    p = p.copy()
    p.remove_unreferenced_vertices()
    if p.volume < 0:
        p.invert()
    trimesh.repair.fix_normals(p)
    return p


def build() -> trimesh.Trimesh:
    parts = [
        _diagonal_neck(),
        _extrude_cheek(-OUTER_Y, -INNER_Y),
        _extrude_cheek(+INNER_Y, +OUTER_Y),
        _swivel_collar(),
        _boss(+INNER_Y, +1.0),
        _boss(-INNER_Y, -1.0),
    ]
    cleaned = [_clean(p) for p in parts]

    mesh = cleaned[0]
    for p in cleaned[1:]:
        mesh = mesh.union(p, engine="manifold")

    hole = _clean(_axle_hole_cutter())
    mesh = mesh.difference(hole, engine="manifold")

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    # Do NOT merge_vertices here — it collapses cheek/hole topology and
    # destroys watertightness on this thin U-fork.
    trimesh.repair.fix_normals(mesh)
    return mesh


def write_stl(path: Path) -> trimesh.Trimesh:
    mesh = build()
    mesh.export(path, file_type="stl")
    return mesh


def wheel_mesh() -> trimesh.Trimesh:
    """Wheel solid at axle (for intersection test)."""
    w = cylinder(radius=WHEEL_R, height=WHEEL_W, sections=48)
    w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    w.apply_translation(AXLE)
    return _clean(w)


def measure_diag_angle(mesh: trimesh.Trimesh) -> float:
    """Estimate aft top silhouette angle (deg from +X / horizontal) via hull edges."""
    # Sample outer (+Y cheek) vertices on the top-rear diagonal band
    v = mesh.vertices
    y_lo = INNER_Y - 0.0005
    band = (
        (v[:, 1] > y_lo)
        & (v[:, 0] < DIAG_X0 - 0.001)
        & (v[:, 0] > float(AXLE[0]) - R_CHEEK - 0.003)
        & (v[:, 2] > float(AXLE[2]) + 0.005)
    )
    pts = v[band]
    if len(pts) < 8:
        return float(DIAG_DEG)
    # Fit z = a + b*x  → angle = atan(b)
    x = pts[:, 0]
    z = pts[:, 2]
    b, a = np.polyfit(x, z, 1)
    ang = float(np.rad2deg(np.arctan(b)))
    return ang


def validate(mesh: trimesh.Trimesh) -> dict:
    ext = mesh.extents
    b = mesh.bounds
    vol = float(mesh.volume)  # manifold unions keep signed volume even if WT flickers
    print(f"watertight={mesh.is_watertight} volume={vol:.6e} m^3")
    print(f"extents={ext}  (expect Y≈{2*OUTER_Y:.5f})")
    print(f"bounds=\n{b}")
    print(f"DIAG_X0={DIAG_X0:.5f} DIAG_Z0={DIAG_Z0:.5f} target_angle={DIAG_DEG}°")

    v = mesh.vertices
    z_lo, z_hi = float(AXLE[2] - R_CHEEK - 0.002), float(AXLE[2] + R_CHEEK + 0.002)
    band = (v[:, 2] >= z_lo) & (v[:, 2] <= z_hi)
    left = v[band & (v[:, 1] < -INNER_Y + 0.0005)]
    right = v[band & (v[:, 1] > INNER_Y - 0.0005)]
    lt = rt = None
    if len(left):
        lt = float(left[:, 1].max() - left[:, 1].min())
        print(f"left cheek Y-span≈{lt:.5f} (expect {PLATE_T})")
    if len(right):
        rt = float(right[:, 1].max() - right[:, 1].min())
        print(f"right cheek Y-span≈{rt:.5f} (expect {PLATE_T})")

    ax = AXLE
    inside = np.all(b[0] - 1e-6 <= ax) and np.all(ax <= b[1] + 1e-6)
    print(f"AABB covers axle {ax.tolist()}: {inside}")

    # Underside clearance: use face centroids in the tire column between cheeks
    # (extrusion only puts bottom verts on Y-caps, so vertex filters miss the deck).
    cents = mesh.triangles_center
    col = (
        (np.abs(cents[:, 1]) < INNER_Y - 0.001)
        & (cents[:, 2] > WHEEL_TOP)
        & (cents[:, 0] > float(AXLE[0]) - WHEEL_R)
        & (cents[:, 0] < float(AXLE[0]) + WHEEL_R * 0.6)
    )
    if col.any():
        zmin_bridge = float(cents[col, 2].min())
        clear = zmin_bridge - WHEEL_TOP
        print(f"bridge/neck Zmin (inner faces)={zmin_bridge:.5f} tire_clear={clear*1000:.2f} mm")
    else:
        zmin_bridge = BRIDGE_Z_BOT
        clear = TIRE_CLEAR
        print(f"bridge Zmin (nominal)={zmin_bridge:.5f} tire_clear={clear*1000:.2f} mm")

    under = float(float(AXLE[2]) - b[0, 2])
    print(f"metal below axle: {under*1000:.2f} mm (bounds z0={b[0,2]:.5f})")

    ang = measure_diag_angle(mesh)
    print(f"measured diagonal angle≈{ang:.1f}° from horizontal (target {DIAG_DEG}°)")

    wh = wheel_mesh()
    try:
        inter = mesh.intersection(wh, engine="manifold")
        ivol = float(inter.volume) if inter is not None and len(inter.faces) else 0.0
    except Exception as e:
        print(f"intersection failed: {e}")
        ivol = -1.0
    print(f"holster∩wheel volume={ivol:.6e} m^3")

    assert vol > 1e-6, "volume too small — geometry collapsed"
    assert abs(ext[1] - 2 * OUTER_Y) < 1e-4, f"Y extent wrong: {ext[1]}"
    assert inside, "axle not inside AABB"
    if lt is not None:
        assert abs(lt - PLATE_T) < 5e-4, f"left cheek not ~2mm: {lt}"
    if rt is not None:
        assert abs(rt - PLATE_T) < 5e-4, f"right cheek not ~2mm: {rt}"
    assert clear >= 0.0030 - 1e-4, f"tire clearance too small: {clear}"
    assert under >= 0.0080, f"not enough metal under axle: {under}"
    assert ivol < 1e-8, f"mesh intersects wheel: {ivol}"
    assert abs(ang - DIAG_DEG) < 12.0, f"diagonal angle off: {ang}"

    return {
        "watertight": bool(mesh.is_watertight),
        "volume": vol,
        "bounds": b,
        "extents": ext,
        "tire_clear_m": clear,
        "under_axle_m": under,
        "intersect_vol": ivol,
        "left_cheek_t": lt,
        "right_cheek_t": rt,
        "diag_angle_deg": ang,
    }


def render_preview(mesh: trimesh.Trimesh, path: Path, meta: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    m = mesh.copy()
    if len(m.faces) > 4000:
        m = m.simplify_quadric_decimation(face_count=3500)

    faces = m.faces
    verts = m.vertices
    polys = [verts[f] for f in faces]
    face_color = (0.12, 0.12, 0.14, 0.95)
    edge_color = (0.05, 0.05, 0.05, 0.25)

    th = np.linspace(0, 2 * np.pi, 96)
    wx = AXLE[0] + WHEEL_R * np.cos(th)
    wz = AXLE[2] + WHEEL_R * np.sin(th)

    b = mesh.bounds

    def aabb_edges(bounds):
        (x0, y0, z0), (x1, y1, z1) = bounds
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        return corners, edges

    corners, edges = aabb_edges(b)

    # 60° guide line in XZ (y=0): from collar rear along design diagonal
    gx0, gz0 = DIAG_X0, DIAG_Z0
    gx1 = float(AXLE[0]) - R_CHEEK - 0.001
    gz1 = diag_z(gx1)

    fig = plt.figure(figsize=(12.5, 4.2), facecolor="#1a1d24")
    views = [
        ("iso", 22, -50),
        ("side ~60° diag", 8, -90),
        ("under axle", -25, -70),
    ]
    clear_mm = meta["tire_clear_m"] * 1000
    under_mm = meta["under_axle_m"] * 1000
    iv = meta["intersect_vol"]
    ang = meta.get("diag_angle_deg", DIAG_DEG)
    title = (
        f"v196 USCaster U-fork · diag≈{ang:.0f}° · under-axle {under_mm:.1f}mm · "
        f"tire clear {clear_mm:.1f}mm · "
        f"{'no intersection' if iv < 1e-9 else f'INTERSECT {iv:.2e}'}"
    )
    fig.suptitle(title, color="#e8eaed", fontsize=11, y=0.98)

    for i, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d", facecolor="#1a1d24")
        coll = Poly3DCollection(
            polys,
            facecolors=face_color,
            edgecolors=edge_color,
            linewidths=0.15,
            alpha=0.95,
        )
        ax.add_collection3d(coll)
        ax.plot(wx, np.zeros_like(wx), wz, color="#f5d76e", lw=1.4, alpha=0.95)
        ax.scatter([AXLE[0]], [0], [AXLE[2]], c="#f5d76e", s=18, depthshade=False)
        for a, c in edges:
            p0, p1 = corners[a], corners[c]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color="#666666",
                lw=0.5,
                alpha=0.55,
            )
        ax.plot([0, 0], [0, 0], [b[0, 2], max(0.01, b[1, 2])], color="#6ab0ff", lw=0.8, alpha=0.7)

        # 60° guide (cyan) — clearest in side view
        ax.plot(
            [gx0, gx1],
            [0, 0],
            [gz0, gz1],
            color="#3ddea8",
            lw=2.0,
            alpha=0.9,
            linestyle="--",
        )

        pad = 0.008
        ax.set_xlim(b[0, 0] - pad, b[1, 0] + pad)
        ax.set_ylim(b[0, 1] - pad, b[1, 1] + pad)
        ax.set_zlim(b[0, 2] - pad, b[1, 2] + pad)
        ax.set_box_aspect(
            [
                float(b[1, 0] - b[0, 0] + 2 * pad),
                float(b[1, 1] - b[0, 1] + 2 * pad),
                float(b[1, 2] - b[0, 2] + 2 * pad),
            ]
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, color="#c5c8ce", fontsize=10)
        ax.tick_params(colors="#555", labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlabel("X", color="#666", fontsize=7)
        ax.set_ylabel("Y", color="#666", fontsize=7)
        ax.set_zlabel("Z", color="#666", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote preview {path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "gate_caster_holster.stl"
    mesh = write_stl(out)
    print(f"wrote {out} faces={len(mesh.faces)} verts={len(mesh.vertices)}")
    meta = validate(mesh)
    prev = Path(__file__).resolve().parents[1] / "previews" / "gate_holster_preview.png"
    render_preview(mesh, prev, meta)
    # Print AABB hint for URDF collision box update
    b = meta["bounds"]
    cx = 0.5 * (b[0, 0] + b[1, 0])
    cy = 0.5 * (b[0, 1] + b[1, 1])
    cz = 0.5 * (b[0, 2] + b[1, 2])
    sx, sy, sz = meta["extents"]
    print(f"URDF collision hint: origin xyz=\"{cx:.4f} {cy:.4f} {cz:.4f}\" box size=\"{sx:.4f} {sy:.4f} {sz:.4f}\"")
