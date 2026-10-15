#!/usr/bin/env python3
"""anker_usb_hub_body.stl — Anker hub brick with large perimeter corner radius only.

Link frame (ports face −Y, hanging under mast platform L bar):
  X = 109 mm (length along bar)
  Y =  23 mm (thickness; +Y = back / outboard, −Y = port face)
  Z =  44 mm (hanging height)

Only the four corners *around* the XZ silhouette are rounded (edges parallel to Y,
like the product's vertical corner radii). Face-to-side edges stay sharp.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import box, cylinder

OUT = Path(__file__).resolve().parent / "anker_usb_hub_body.stl"

SX, SY, SZ = 0.109, 0.023, 0.044
R = 0.007  # 7 mm perimeter corners (product ~5–8 mm)


def perimeter_rounded_box(sx: float, sy: float, sz: float, r: float) -> trimesh.Trimesh:
    """Extrude a rounded XZ rectangle along Y — no face↔side edge fillets."""
    assert min(sx, sz) > 2 * r + 1e-6
    hx, hz = sx / 2, sz / 2
    parts: list[trimesh.Trimesh] = [
        box(extents=(sx - 2 * r, sy, sz)),
        box(extents=(sx, sy, sz - 2 * r)),
    ]
    # Four corner cylinders along Y (thickness)
    for sx_s in (-1, 1):
        for sz_s in (-1, 1):
            c = cylinder(radius=r, height=sy, sections=48)
            c.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            )
            c.apply_translation([sx_s * (hx - r), 0.0, sz_s * (hz - r)])
            parts.append(c)

    mesh = trimesh.boolean.union(parts, engine="manifold")
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def build() -> trimesh.Trimesh:
    mesh = perimeter_rounded_box(SX, SY, SZ, R)
    extents = mesh.bounds[1] - mesh.bounds[0]
    print(f"AABB extents={extents} watertight={mesh.is_watertight} "
          f"volume={mesh.volume:.6e} faces={len(mesh.faces)} R={R*1000:.1f}mm")
    return mesh


def write_stl(path: Path = OUT) -> trimesh.Trimesh:
    mesh = build()
    mesh.export(path)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return mesh


if __name__ == "__main__":
    write_stl()
