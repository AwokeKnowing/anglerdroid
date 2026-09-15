#!/usr/bin/env python3
"""Repair the casita wall STL and raycast it as a visual (non-physics) sensor mesh.

The house stays a Newton site: no MuJoCo collider. RS1/RS2 still need wall hits,
so cameras query this welded mesh with backface culling off.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXED_STL = Path(__file__).resolve().parent / "casitahouse-walls-fixed.stl"
DEFAULT_SRC = Path("/home/james/Desktop/casitahouse-walls.stl")

# Last repair report, filled by repair_house_stl / load_repaired_house.
REPORT: dict = {}


def read_binary_stl(path: Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    if raw[:5].lower() == b"solid" and b"\x00" not in raw[:80]:
        raise RuntimeError(f"ASCII STL not supported: {path}")
    n = int.from_bytes(raw[80:84], "little")
    tris = np.empty((n, 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        if off + 50 > len(raw):
            tris = tris[:i]
            break
        off += 12  # stored normal (recomputed on write)
        for k in range(3):
            tris[i, k] = np.frombuffer(raw[off:off + 12], dtype="<f4")
            off += 12
        off += 2
    if len(tris) == 0:
        raise RuntimeError(f"empty STL {path}")
    ext = tris.reshape(-1, 3)
    span = float((ext.max(0) - ext.min(0)).max())
    if span > 50.0:
        tris *= 0.001
    return tris


def write_binary_stl(path: Path, tris: np.ndarray) -> None:
    tris = np.asarray(tris, dtype=np.float32)
    n = int(tris.shape[0])
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 1e-12, nrm / np.maximum(ln, 1e-12), 0.0).astype(np.float32)
    header = b"casitahouse-walls repaired weld+winding+normals".ljust(80, b" ")[:80]
    buf = bytearray()
    buf += header
    buf += int(n).to_bytes(4, "little")
    for i in range(n):
        buf += nrm[i].astype("<f4").tobytes()
        buf += tris[i].astype("<f4").tobytes()
        buf += (0).to_bytes(2, "little")
    Path(path).write_bytes(bytes(buf))


def diagnose_tris(tris: np.ndarray) -> dict:
    tris = np.asarray(tris, dtype=np.float64)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross, axis=1)
    degenerate = area2 < 1e-10
    flat = np.round(tris.reshape(-1, 3), 5)
    # edge winding consistency on unique-ish verts
    edge = defaultdict(int)
    for a, b, c in np.round(tris, 4):
        pts = [tuple(p) for p in (a, b, c)]
        if len(set(pts)) < 3:
            continue
        for u, v in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
            edge[(u, v)] += 1
    seen = set()
    n_consist = n_inconsist = n_bound = n_nonman = 0
    for e, _c in list(edge.items()):
        u, v = e
        if e in seen or (v, u) in seen:
            continue
        seen.add(e)
        seen.add((v, u))
        fwd = edge.get((u, v), 0)
        bak = edge.get((v, u), 0)
        total = fwd + bak
        if total == 1:
            n_bound += 1
        elif total == 2 and fwd == 1 and bak == 1:
            n_consist += 1
        elif total == 2:
            n_inconsist += 1
        else:
            n_nonman += 1
    return {
        "tris": int(len(tris)),
        "raw_verts": int(tris.shape[0] * 3),
        "unique_verts_1e-5": int(len({tuple(p) for p in flat})),
        "degenerate": int(degenerate.sum()),
        "inconsistent_edges": int(n_inconsist),
        "consistent_edges": int(n_consist),
        "boundary_edges": int(n_bound),
        "nonmanifold_edges": int(n_nonman),
        "z_min": float(tris[:, :, 2].min()),
        "z_max": float(tris[:, :, 2].max()),
    }


def _weld(tris: np.ndarray, eps: float = 1e-4):
    flat = tris.reshape(-1, 3)
    q = np.round(flat / eps).astype(np.int64)
    key_to_id = {}
    uniq = []
    idx = np.empty(len(flat), dtype=np.int32)
    for i, k in enumerate(map(tuple, q)):
        j = key_to_id.get(k)
        if j is None:
            j = len(uniq)
            key_to_id[k] = j
            uniq.append(flat[i])
        idx[i] = j
    verts = np.asarray(uniq, dtype=np.float64)
    faces = idx.reshape(-1, 3)
    # drop degenerate (repeated corners) and exact duplicate faces
    keep = []
    seen_f = set()
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a == b or b == c or c == a:
            continue
        key = tuple(sorted((a, b, c)))
        if key in seen_f:
            continue
        seen_f.add(key)
        keep.append((a, b, c))
    faces = np.asarray(keep, dtype=np.int32)
    # drop zero-area after weld
    if len(faces):
        tv = verts[faces]
        area = np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]), axis=1)
        faces = faces[area > 1e-10]
    return verts, faces


def _orient_outward_upward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Consistent winding. Horizontal components point +Z; walls point outward."""
    n_f = len(faces)
    if n_f == 0:
        return faces
    faces = np.array(faces, dtype=np.int32, copy=True)
    emap = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            emap[(u, v)].append(fi)

    # undirected edge -> faces that use it, and whether each walks u->v for key (min,max) hashed later
    undirected = defaultdict(list)
    for (u, v), flist in emap.items():
        undirected[tuple(sorted((u, v)))].extend(flist)

    neigh = [[] for _ in range(n_f)]
    for (u, v), flist in emap.items():
        opp = emap.get((v, u), [])
        for fi in flist:
            for fj in opp:
                if fi != fj:
                    # already opposite directed edges: no flip needed to agree
                    neigh[fi].append((fj, False))
            same = emap.get((u, v), [])
            for fj in same:
                if fj != fi:
                    neigh[fi].append((fj, True))

    visited = np.zeros(n_f, dtype=bool)
    flip = np.zeros(n_f, dtype=bool)
    for seed in range(n_f):
        if visited[seed]:
            continue
        visited[seed] = True
        q = deque([seed])
        while q:
            fi = q.popleft()
            for fj, need_flip in neigh[fi]:
                if visited[fj]:
                    continue
                visited[fj] = True
                # need_flip means fj's edge agrees with fi; flip fj relative to fi
                flip[fj] = flip[fi] ^ bool(need_flip)
                q.append(fj)

    out = faces.copy()
    out[flip] = out[flip, ::-1]

    # per connected component (face adjacency ignoring orientation), choose global flip
    visited[:] = False
    # rebuild adjacency without flip flag
    adj = [[] for _ in range(n_f)]
    for key, flist in undirected.items():
        uniq = list(dict.fromkeys(flist))
        for i, fi in enumerate(uniq):
            for fj in uniq[i + 1:]:
                adj[fi].append(fj)
                adj[fj].append(fi)

    tv = verts[out]
    fn = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    area = np.linalg.norm(fn, axis=1)
    cent = tv.mean(axis=1)

    visited[:] = False
    for seed in range(n_f):
        if visited[seed]:
            continue
        comp = []
        q = deque([seed])
        visited[seed] = True
        while q:
            fi = q.popleft()
            comp.append(fi)
            for fj in adj[fi]:
                if not visited[fj]:
                    visited[fj] = True
                    q.append(fj)
        ids = np.asarray(comp, dtype=np.int32)
        a = area[ids]
        if float(a.sum()) < 1e-12:
            continue
        nrm = fn[ids]
        # horizontal slab vs wall
        nlen = np.linalg.norm(nrm, axis=1)
        abs_z = np.abs(nrm[:, 2]) / np.maximum(nlen, 1e-12)
        mean_abs_z = float(np.average(abs_z, weights=np.maximum(a, 1e-12)))
        if mean_abs_z > 0.65:
            score = float(np.sum(nrm[:, 2]))
        else:
            origin = verts.mean(axis=0)
            away = cent[ids] - origin
            score = float(np.sum(np.einsum("ij,ij->i", nrm, away)))
        if score < 0.0:
            out[ids] = out[ids, ::-1]
    return out


def repair_tris(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    before = diagnose_tris(tris)
    verts, faces = _weld(tris)
    dropped_deg = int(before["tris"] - len(faces))
    faces = _orient_outward_upward(verts, faces)
    # sit on the floor, keep metres
    z0 = float(verts[:, 2].min())
    verts = verts.copy()
    verts[:, 2] -= z0
    # drop a face that is still degenerate after the floor shift (none expected)
    tv = verts[faces]
    area = np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]), axis=1)
    faces = faces[area > 1e-10]
    after_tris = verts[faces]
    after = diagnose_tris(after_tris)
    report = {
        "before": before,
        "after": after,
        "dropped_degenerate_or_duplicate": int(max(0, before["tris"] - after["tris"])),
        "welded_verts": int(len(verts)),
        "z_shift_m": float(z0),
    }
    return verts.astype(np.float32), faces.astype(np.int32), report


def repair_house_stl(src: Path | None = None, dest: Path | None = None) -> Path:
    src = Path(src or DEFAULT_SRC)
    dest = Path(dest or FIXED_STL)
    tris = read_binary_stl(src)
    verts, faces, report = repair_tris(tris)
    write_binary_stl(dest, verts[faces])
    report["src"] = str(src)
    report["dest"] = str(dest)
    REPORT.clear()
    REPORT.update(report)
    print(
        "house mesh repair: tris %d -> %d  unique_verts %d -> %d  "
        "inconsistent_edges %d -> %d  degenerate %d -> %d  wrote %s"
        % (
            before_tris := report["before"]["tris"],
            report["after"]["tris"],
            report["before"]["unique_verts_1e-5"],
            report["after"]["unique_verts_1e-5"],
            report["before"]["inconsistent_edges"],
            report["after"]["inconsistent_edges"],
            report["before"]["degenerate"],
            report["after"]["degenerate"],
            dest,
        )
    )
    # silence unused in case of rewrite
    del before_tris
    return dest


def load_indexed(path: Path):
    tris = read_binary_stl(path)
    # already repaired file: weld again so raycast has indexed mesh, no z shift if already on floor
    verts, faces = _weld(tris, eps=1e-4)
    z0 = float(verts[:, 2].min())
    if abs(z0) > 1e-4:
        verts = verts.copy()
        verts[:, 2] -= z0
    return verts.astype(np.float32), faces.astype(np.int32)


def wall_segments_xy(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """XY segments of mostly-vertical faces (the walls). Shape (E, 2, 2)."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if len(faces) == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)
    tv = verts[faces]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    area = np.linalg.norm(n, axis=1)
    nz = np.abs(n[:, 2]) / np.maximum(area, 1e-12)
    wall = (area > 1e-8) & (nz < 0.4)
    if not np.any(wall):
        return np.zeros((0, 2, 2), dtype=np.float64)
    tw = tv[wall]
    segs = np.concatenate(
        [
            np.stack([tw[:, 0, :2], tw[:, 1, :2]], axis=1),
            np.stack([tw[:, 1, :2], tw[:, 2, :2]], axis=1),
            np.stack([tw[:, 2, :2], tw[:, 0, :2]], axis=1),
        ],
        axis=0,
    )
    # drop vertical edges that collapsed in XY
    elen = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    segs = segs[elen > 1e-4]
    return segs.astype(np.float64)


def _point_seg_dist(points: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Min distance from each point (P,2) to the segment set. Returns (P,)."""
    if len(points) == 0 or len(segs) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    a = segs[:, 0]
    b = segs[:, 1]
    ab = b - a
    ab2 = np.einsum("ij,ij->i", ab, ab)
    ab2 = np.maximum(ab2, 1e-18)
    # (P, E)
    ap = p[:, None, :] - a[None, :, :]
    t = np.einsum("pei,ei->pe", ap, ab) / ab2[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(p[:, None, :] - closest, axis=2)
    return d.min(axis=1)


def body_wall_clearance(
    pos_xy,
    yaw: float,
    segs: np.ndarray,
    front_x: float = 0.15,
    rear_x: float = -0.18,
    half_w: float = 0.165,
) -> tuple[float, float, bool]:
    """Return (min_clearance_m, front_clearance_m, overlap).

    Clearance is the min XY distance from body sample points to wall segments.
    Overlap is true if any wall segment intersects the body rectangle.
    """
    pos_xy = np.asarray(pos_xy, dtype=np.float64)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    local = np.array(
        [
            [front_x, half_w],
            [front_x, -half_w],
            [front_x, 0.0],
            [0.5 * (front_x + rear_x), half_w],
            [0.5 * (front_x + rear_x), -half_w],
            [rear_x, half_w],
            [rear_x, -half_w],
            [rear_x, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    world = np.empty_like(local)
    world[:, 0] = pos_xy[0] + c * local[:, 0] - s * local[:, 1]
    world[:, 1] = pos_xy[1] + s * local[:, 0] + c * local[:, 1]
    dist = _point_seg_dist(world, segs)
    front = _point_seg_dist(world[:3], segs)
    min_d = float(np.min(dist)) if len(dist) else float("inf")
    front_d = float(np.min(front)) if len(front) else float("inf")
    overlap = _rect_segment_overlap(pos_xy, yaw, segs, front_x, rear_x, half_w)
    return min_d, front_d, overlap




def footprint_local_samples(inflate: float = 0.0) -> np.ndarray:
    """Axle-frame corners, edge midpoints, caster, and wheels.

    Body box is 33 cm (rear x=-0.18, front x=+0.15, y=±0.165). Caster sits
    at x=-0.26 on the ground. Wheels are about y=±0.17. inflate grows that
    envelope so a side wall or caster sweep is stopped before the mesh.
    """
    inf = float(inflate)
    rear = -0.18 - inf
    front = 0.15 + inf
    half = 0.165 + inf
    mid_x = 0.5 * (rear + front)
    caster_x = -0.26 - inf
    wheel_y = 0.17 + inf
    cy = 0.05 + 0.5 * inf
    pts = [
        [front, half], [front, -half], [front, 0.0],
        [rear, half], [rear, -half], [rear, 0.0],
        [mid_x, half], [mid_x, -half], [mid_x, 0.0],
        [caster_x, 0.0], [caster_x, cy], [caster_x, -cy],
        [0.0, wheel_y], [0.0, -wheel_y],
        [0.09 + inf, wheel_y], [0.09 + inf, -wheel_y],
        [-0.09 - inf, 0.17], [-0.09 - inf, -0.17],
        [0.15, 0.165], [-0.18, 0.165],
        [0.15, -0.165], [-0.18, -0.165],
    ]
    return np.asarray(pts, dtype=np.float64)


def footprint_world(pos_xy, yaw: float, local: np.ndarray) -> np.ndarray:
    pos_xy = np.asarray(pos_xy, dtype=np.float64)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    local = np.asarray(local, dtype=np.float64)
    world = np.empty_like(local)
    world[:, 0] = pos_xy[0] + c * local[:, 0] - s * local[:, 1]
    world[:, 1] = pos_xy[1] + s * local[:, 0] + c * local[:, 1]
    return world


def _rect_segment_overlap(pos_xy, yaw, segs, front_x, rear_x, half_w) -> bool:
    if len(segs) == 0:
        return False
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    # segments into axle frame
    a = segs[:, 0] - pos_xy
    b = segs[:, 1] - pos_xy
    R = np.array([[c, s], [-s, c]], dtype=np.float64)  # world -> body
    la = a @ R.T
    lb = b @ R.T
    # AABB reject then coarse: either endpoint inside, or segment crosses rect
    xmin, xmax = float(rear_x), float(front_x)
    ymin, ymax = -float(half_w), float(half_w)
    inside = (
        (la[:, 0] >= xmin) & (la[:, 0] <= xmax) & (la[:, 1] >= ymin) & (la[:, 1] <= ymax)
    ) | (
        (lb[:, 0] >= xmin) & (lb[:, 0] <= xmax) & (lb[:, 1] >= ymin) & (lb[:, 1] <= ymax)
    )
    if bool(inside.any()):
        return True
    # Cohen-style: if segment bbox misses rect, skip; else test edges
    seg_min = np.minimum(la, lb)
    seg_max = np.maximum(la, lb)
    maybe = (
        (seg_max[:, 0] >= xmin)
        & (seg_min[:, 0] <= xmax)
        & (seg_max[:, 1] >= ymin)
        & (seg_min[:, 1] <= ymax)
    )
    if not np.any(maybe):
        return False
    # sample the segment; walls are thin and this is a diagnostic, not physics
    cand_a = la[maybe]
    cand_b = lb[maybe]
    for t in (0.25, 0.5, 0.75):
        p = cand_a * (1.0 - t) + cand_b * t
        hit = (
            (p[:, 0] >= xmin) & (p[:, 0] <= xmax) & (p[:, 1] >= ymin) & (p[:, 1] <= ymax)
        )
        if bool(hit.any()):
            return True
    return False


class VisualMeshCaster:
    """Raycast the visual house mesh. Not a physics collider.

    Newton SensorTiledCamera culls backfaces and may only return collision
    shapes. This caster accepts both sides of the wall STL so RS1/RS2 depth
    includes wall hits.
    """

    def __init__(self, verts: np.ndarray, faces: np.ndarray, device: str = "cuda:0"):
        import warp as wp

        self.wp = wp
        self.device = device
        pts = np.ascontiguousarray(verts, dtype=np.float32)
        idx = np.ascontiguousarray(faces.reshape(-1), dtype=np.int32)
        self.mesh = wp.Mesh(
            points=wp.array(pts, dtype=wp.vec3, device=device),
            indices=wp.array(idx, dtype=wp.int32, device=device),
        )
        self._kernel = None
        self._dx_wp = None
        self._dy_wp = None
        self._out_fwd = None
        self._orig_wp = None
        self._dir_wp = None
        self._out_t = None
        self._buf_n = 0

    def _kernel_fn(self):
        if self._kernel is not None:
            return self._kernel
        wp = self.wp

        @wp.kernel
        def _cast_forward(
            mesh_id: wp.uint64,
            origin: wp.vec3,
            qx: wp.float32,
            qy: wp.float32,
            qz: wp.float32,
            qw: wp.float32,
            dx: wp.array2d(dtype=wp.float32),
            dy: wp.array2d(dtype=wp.float32),
            max_t: wp.float32,
            out_fwd: wp.array2d(dtype=wp.float32),
        ):
            y, x = wp.tid()
            rdx = dx[y, x]
            rdy = dy[y, x]
            rdz = wp.float32(-1.0)
            inv = wp.float32(1.0) / wp.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
            lx = rdx * inv
            ly = rdy * inv
            lz = rdz * inv
            # dir in world
            uvx = qy * lz - qz * ly
            uvy = qz * lx - qx * lz
            uvz = qx * ly - qy * lx
            uuvx = qy * uvz - qz * uvy
            uuvy = qz * uvx - qx * uvz
            uuvz = qx * uvy - qy * uvx
            wx = lx + wp.float32(2.0) * (qw * uvx + uuvx)
            wy = ly + wp.float32(2.0) * (qw * uvy + uuvy)
            wz = lz + wp.float32(2.0) * (qw * uvz + uuvz)
            # look = camera -Z
            l0 = wp.float32(0.0)
            l1 = wp.float32(0.0)
            l2 = wp.float32(-1.0)
            luvx = qy * l2 - qz * l1
            luvy = qz * l0 - qx * l2
            luvz = qx * l1 - qy * l0
            luuvx = qy * luvz - qz * luvy
            luuvy = qz * luvx - qx * luvz
            luuvz = qx * luvy - qy * luvx
            lxw = l0 + wp.float32(2.0) * (qw * luvx + luuvx)
            lyw = l1 + wp.float32(2.0) * (qw * luvy + luuvy)
            lzw = l2 + wp.float32(2.0) * (qw * luvz + luuvz)
            ln = wp.sqrt(lxw * lxw + lyw * lyw + lzw * lzw)
            if ln > wp.float32(1.0e-8):
                lxw = lxw / ln
                lyw = lyw / ln
                lzw = lzw / ln
            query = wp.mesh_query_ray(mesh_id, origin, wp.vec3(wx, wy, wz), max_t)
            if query.result and query.t > wp.float32(0.02):
                out_fwd[y, x] = query.t * wp.abs(wx * lxw + wy * lyw + wz * lzw)
            else:
                out_fwd[y, x] = wp.float32(0.0)

        self._kernel = _cast_forward
        return self._kernel


    def _probe_kernel_fn(self):
        if getattr(self, "_probe_kernel", None) is not None:
            return self._probe_kernel
        wp = self.wp

        @wp.kernel
        def _probe_pts(
            mesh_id: wp.uint64,
            pts: wp.array(dtype=wp.vec3),
            dirs: wp.array(dtype=wp.vec3),
            motion_t: wp.array(dtype=wp.float32),
            out_dist: wp.array(dtype=wp.float32),
            out_inside: wp.array(dtype=wp.int32),
            out_motion: wp.array(dtype=wp.int32),
        ):
            i = wp.tid()
            p = pts[i]
            face = int(0)
            fu = float(0.0)
            fv = float(0.0)
            max_d = float(1.6)
            dist = max_d
            q = wp.mesh_query_point_no_sign(mesh_id, p, max_d)
            if q.result:
                cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
                dist = wp.length(cp - p)
            # low caster/belly sample in the same column
            plow = wp.vec3(p[0], p[1], wp.float32(0.06))
            q2 = wp.mesh_query_point_no_sign(mesh_id, plow, max_d)
            if q2.result:
                cp2 = wp.mesh_eval_position(mesh_id, q2.face, q2.u, q2.v)
                d2 = wp.length(cp2 - plow)
                if d2 < dist:
                    dist = d2

            inside = int(0)
            # nearest wall closer than ~5 cm
            if dist < float(0.05):
                inside = int(1)

            # downward ray from above the point: a hit at body height means
            # this XY sits on/through a wall (backface culling is off)
            origin = wp.vec3(p[0], p[1], wp.float32(1.40))
            down = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(-1.0))
            rq = wp.mesh_query_ray(mesh_id, origin, down, wp.float32(1.45))
            if rq.result and rq.t > wp.float32(0.001):
                hz = origin[2] - rq.t
                if hz > wp.float32(0.04) and hz < wp.float32(0.78):
                    inside = int(1)
                    dist = wp.float32(0.0)

            # outward rays from above the point; a close hit at body height
            # is the sample sweeping through the wall surface
            above = wp.vec3(p[0], p[1], p[2] + wp.float32(0.40))
            # 8 directions, slightly downward
            ax0 = wp.float32(1.0)
            ay0 = wp.float32(0.0)
            ax1 = wp.float32(0.70710678)
            ay1 = wp.float32(0.70710678)
            ax2 = wp.float32(0.0)
            ay2 = wp.float32(1.0)
            ax3 = wp.float32(-0.70710678)
            ay3 = wp.float32(0.70710678)
            ax4 = wp.float32(-1.0)
            ay4 = wp.float32(0.0)
            ax5 = wp.float32(-0.70710678)
            ay5 = wp.float32(-0.70710678)
            ax6 = wp.float32(0.0)
            ay6 = wp.float32(-1.0)
            ax7 = wp.float32(0.70710678)
            ay7 = wp.float32(-0.70710678)
            # unrolled so the kernel stays static
            for k in range(8):
                dx = ax0
                dy = ay0
                if k == 1:
                    dx = ax1
                    dy = ay1
                elif k == 2:
                    dx = ax2
                    dy = ay2
                elif k == 3:
                    dx = ax3
                    dy = ay3
                elif k == 4:
                    dx = ax4
                    dy = ay4
                elif k == 5:
                    dx = ax5
                    dy = ay5
                elif k == 6:
                    dx = ax6
                    dy = ay6
                elif k == 7:
                    dx = ax7
                    dy = ay7
                rd = wp.vec3(dx, dy, wp.float32(-0.55))
                rn = wp.sqrt(rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2])
                rd = wp.vec3(rd[0] / rn, rd[1] / rn, rd[2] / rn)
                oq = wp.mesh_query_ray(mesh_id, above, rd, wp.float32(0.50))
                if oq.result and oq.t > wp.float32(0.001):
                    hx = above[0] + rd[0] * oq.t
                    hy = above[1] + rd[1] * oq.t
                    hz = above[2] + rd[2] * oq.t
                    if hz > wp.float32(0.04) and hz < wp.float32(0.78):
                        dxy = wp.sqrt((hx - p[0]) * (hx - p[0]) + (hy - p[1]) * (hy - p[1]))
                        if dxy < wp.float32(0.06):
                            inside = int(1)
                            if dxy < dist:
                                dist = dxy

            # short ray along this sample's motion, before the step lands
            md = dirs[i]
            mt = motion_t[i]
            hit_m = int(0)
            if mt > wp.float32(0.004):
                start = wp.vec3(
                    p[0] - md[0] * wp.float32(0.012),
                    p[1] - md[1] * wp.float32(0.012),
                    wp.float32(0.18),
                )
                mq = wp.mesh_query_ray(mesh_id, start, md, mt + wp.float32(0.03))
                if mq.result and mq.t > wp.float32(0.001) and mq.t < (mt + wp.float32(0.012)):
                    hit_m = int(1)
                    inside = int(1)
                    dist = wp.float32(0.0)
            out_dist[i] = dist
            out_inside[i] = inside
            out_motion[i] = hit_m

        self._probe_kernel = _probe_pts
        return self._probe_kernel

    def probe_footprint(self, points_xy, motion_xy=None, z: float = 0.18):
        """STL query (backfaces on) for footprint samples.

        Returns (dist, inside, motion_hit) arrays. A point is inside/through
        if the nearest wall is < 5 cm, a downward/outward ray from above hits
        a wall at body height, or the motion ray hits before the step.
        """
        wp = self.wp
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        n = int(pts.shape[0])
        if n == 0:
            z0 = np.zeros(0, dtype=np.float32)
            return z0, np.zeros(0, dtype=np.int32), z0.astype(np.int32)
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 0] = pts[:, 0]
        xyz[:, 1] = pts[:, 1]
        xyz[:, 2] = float(z)
        dirs = np.zeros((n, 3), dtype=np.float32)
        mt = np.zeros(n, dtype=np.float32)
        if motion_xy is not None:
            mv = np.asarray(motion_xy, dtype=np.float64).reshape(-1, 2)
            if len(mv) == n:
                for i in range(n):
                    mag = float(np.hypot(mv[i, 0], mv[i, 1]))
                    if mag > 1e-5:
                        dirs[i, 0] = mv[i, 0] / mag
                        dirs[i, 1] = mv[i, 1] / mag
                        mt[i] = mag
        out_d = wp.zeros(n, dtype=wp.float32, device=self.device)
        out_i = wp.zeros(n, dtype=wp.int32, device=self.device)
        out_m = wp.zeros(n, dtype=wp.int32, device=self.device)
        wp.launch(
            self._probe_kernel_fn(),
            dim=n,
            inputs=[
                self.mesh.id,
                wp.array(xyz, dtype=wp.vec3, device=self.device),
                wp.array(dirs, dtype=wp.vec3, device=self.device),
                wp.array(mt, dtype=wp.float32, device=self.device),
            ],
            outputs=[out_d, out_i, out_m],
            device=self.device,
        )
        return out_d.numpy(), out_i.numpy(), out_m.numpy()


    def forward_depth(self, origin, quat_xyzw, dx: np.ndarray, dy: np.ndarray, max_t: float = 12.0) -> np.ndarray:
        wp = self.wp
        dx = np.ascontiguousarray(dx, dtype=np.float32)
        dy = np.ascontiguousarray(dy, dtype=np.float32)
        h, w = dx.shape
        if self._out_fwd is None or tuple(self._out_fwd.shape) != (h, w):
            self._out_fwd = wp.zeros((h, w), dtype=wp.float32, device=self.device)
            self._dx_wp = wp.array(dx, dtype=wp.float32, device=self.device)
            self._dy_wp = wp.array(dy, dtype=wp.float32, device=self.device)
        # Pinhole is constant — do not re-upload dx/dy every grab.
        ox, oy, oz = [float(v) for v in origin]
        qx, qy, qz, qw = [float(v) for v in quat_xyzw]
        wp.launch(
            self._kernel_fn(),
            dim=(h, w),
            inputs=[
                self.mesh.id,
                wp.vec3(ox, oy, oz),
                qx, qy, qz, qw,
                self._dx_wp,
                self._dy_wp,
                float(max_t),
            ],
            outputs=[self._out_fwd],
            device=self.device,
        )
        return self._out_fwd.numpy()


    def _ray_kernel_fn(self):
        if getattr(self, "_ray_kernel", None) is not None:
            return self._ray_kernel
        wp = self.wp

        @wp.kernel
        def _cast_rays(
            mesh_id: wp.uint64,
            origins: wp.array(dtype=wp.vec3),
            dirs: wp.array(dtype=wp.vec3),
            max_t: wp.float32,
            out_t: wp.array(dtype=wp.float32),
        ):
            i = wp.tid()
            d = dirs[i]
            q = wp.mesh_query_ray(mesh_id, origins[i], d, max_t)
            # both sides: a miss from inside the house is retried a hair
            # along the ray so a thin backface still reports
            if q.result and q.t > wp.float32(1.0e-4):
                out_t[i] = q.t
            else:
                nudged = origins[i] + d * wp.float32(0.004)
                q2 = wp.mesh_query_ray(mesh_id, nudged, d, max_t)
                if q2.result and q2.t > wp.float32(1.0e-4):
                    out_t[i] = q2.t + wp.float32(0.004)
                else:
                    out_t[i] = wp.float32(-1.0)

        self._ray_kernel = _cast_rays
        return self._ray_kernel

    def cast_rays(self, origins, dirs, max_t: float = 8.0) -> np.ndarray:
        """Cast world rays at the visual STL. Misses are -1. Backfaces count."""
        wp = self.wp
        o = np.ascontiguousarray(origins, dtype=np.float32).reshape(-1, 3)
        d = np.ascontiguousarray(dirs, dtype=np.float32).reshape(-1, 3)
        n = int(o.shape[0])
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        mag = np.linalg.norm(d, axis=1, keepdims=True)
        mag = np.maximum(mag, 1e-8)
        d = d / mag
        if self._out_t is None or self._buf_n != n:
            self._orig_wp = wp.array(o, dtype=wp.vec3, device=self.device)
            self._dir_wp = wp.array(d, dtype=wp.vec3, device=self.device)
            self._out_t = wp.zeros(n, dtype=wp.float32, device=self.device)
            self._buf_n = n
        else:
            self._orig_wp.assign(o)
            self._dir_wp.assign(d)
        wp.launch(
            self._ray_kernel_fn(),
            dim=n,
            inputs=[
                self.mesh.id,
                self._orig_wp,
                self._dir_wp,
                float(max_t),
            ],
            outputs=[self._out_t],
            device=self.device,
        )
        return self._out_t.numpy()



def wall_collision_boxes(
    verts: np.ndarray,
    faces: np.ndarray,
    thickness: float = 0.05,
    angle_deg: float = 6.0,
    offset_tol: float = 0.05,
    min_len: float = 0.20,
    merge_gap: float = 0.18,
):
    """One thin box per collinear wall run, following the mesh edges.

    Nearly-vertical faces only. Boxes do not span gaps, so a run cannot
    chord across a room. Thickness ~5 cm. Not a mesh collider.
    """
    import math as _math

    segs = wall_segments_xy(verts, faces)
    if len(segs) == 0:
        return []
    # wall height from vertical faces
    tv = np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int32)]
    nrm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    nz = np.abs(nrm[:, 2]) / np.maximum(area, 1e-12)
    wall = (area > 1e-8) & (nz < 0.35)
    z1 = float(tv[wall, :, 2].max()) if np.any(wall) else 2.2
    z1 = max(1.6, min(z1, 2.4))

    a = segs[:, 0]
    b = segs[:, 1]
    d = b - a
    elen = np.linalg.norm(d, axis=1)
    keep = elen > 0.05
    a, b, d, elen = a[keep], b[keep], d[keep], elen[keep]
    direc = d / elen[:, None]
    flip = (direc[:, 0] < -1e-8) | ((np.abs(direc[:, 0]) <= 1e-8) & (direc[:, 1] < 0.0))
    direc = direc.copy()
    direc[flip] *= -1.0
    ang = np.arctan2(direc[:, 1], direc[:, 0])
    nrm2 = np.stack([-direc[:, 1], direc[:, 0]], axis=1)
    mid = 0.5 * (a + b)
    off = np.einsum("ij,ij->i", mid, nrm2)
    ang_bin = np.round(ang / _math.radians(angle_deg)).astype(np.int32)
    off_bin = np.round(off / offset_tol).astype(np.int32)

    boxes = []
    half_t = 0.5 * float(thickness)
    for ab in np.unique(ang_bin):
        sel_ang = np.where(ang_bin == ab)[0]
        for ob in np.unique(off_bin[sel_ang]):
            idx = sel_ang[off_bin[sel_ang] == ob]
            if idx.size == 0:
                continue
            axis = direc[idx].mean(axis=0)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            nn = np.array([-axis[1], axis[0]], dtype=np.float64)
            pts_a = a[idx] @ axis
            pts_b = b[idx] @ axis
            lo = np.minimum(pts_a, pts_b)
            hi = np.maximum(pts_a, pts_b)
            order = np.argsort(lo)
            lo, hi = lo[order], hi[order]
            off_m = float(np.median(off[idx]))
            cur_lo, cur_hi = float(lo[0]), float(hi[0])
            spans = []
            for i in range(1, len(lo)):
                if float(lo[i]) <= cur_hi + merge_gap:
                    cur_hi = max(cur_hi, float(hi[i]))
                else:
                    spans.append((cur_lo, cur_hi))
                    cur_lo, cur_hi = float(lo[i]), float(hi[i])
            spans.append((cur_lo, cur_hi))
            for a0, a1 in spans:
                length = a1 - a0
                if length < min_len:
                    continue
                mid_a = 0.5 * (a0 + a1)
                cxy = axis * mid_a + nn * off_m
                yaw = float(np.arctan2(axis[1], axis[0]))
                boxes.append({
                    "cx": float(cxy[0]),
                    "cy": float(cxy[1]),
                    "yaw": yaw,
                    "hx": 0.5 * length,
                    "hy": half_t,
                    "hz": 0.5 * z1,
                    "cz": 0.5 * z1,
                })
    return boxes


def plus_x_against_boxes(pos_xy, yaw: float, boxes, nose_x: float = 0.15):
    """Nose-to-wall distance along heading, and whether the footprint sits in a box."""
    import math as _math

    if not boxes:
        return float("inf"), False, None
    pos = np.asarray(pos_xy, dtype=np.float64)
    c, s = _math.cos(float(yaw)), _math.sin(float(yaw))
    nose = np.array([pos[0] + c * nose_x, pos[1] + s * nose_x], dtype=np.float64)
    heading = np.array([c, s], dtype=np.float64)
    best = float("inf")
    for b in boxes:
        yaw_b = float(b["yaw"])
        cb, sb = _math.cos(yaw_b), _math.sin(yaw_b)
        # world -> box (x along wall, y thickness)
        rel = nose - np.array([b["cx"], b["cy"]], dtype=np.float64)
        lx = cb * rel[0] + sb * rel[1]
        ly = -sb * rel[0] + cb * rel[1]
        hx = float(b["hx"]) + 0.01
        hy = float(b["hy"])
        # ray nose + t*heading, t>=0, vs box AABB in box frame
        rdx = cb * heading[0] + sb * heading[1]
        rdy = -sb * heading[0] + cb * heading[1]
        t0, t1 = 0.0, 8.0
        hit = True
        for orig, d, lo, hi in ((lx, rdx, -hx, hx), (ly, rdy, -hy, hy)):
            if abs(d) < 1e-9:
                if orig < lo or orig > hi:
                    hit = False
                    break
                continue
            ta, tb = (lo - orig) / d, (hi - orig) / d
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
            if t0 > t1:
                hit = False
                break
        if hit and t1 >= 0.0:
            t_hit = t0 if t0 >= 0.0 else 0.0
            if t_hit < best:
                best = float(t_hit)
    # footprint samples inside any box
    locals = [
        (0.15, 0.0), (0.15, 0.16), (0.15, -0.16),
        (0.0, 0.16), (0.0, -0.16), (0.0, 0.0),
        (-0.18, 0.0), (-0.26, 0.0),
    ]
    crossed = False
    for lx, ly in locals:
        wx = pos[0] + c * lx - s * ly
        wy = pos[1] + s * lx + c * ly
        for b in boxes:
            yaw_b = float(b["yaw"])
            cb, sb = _math.cos(yaw_b), _math.sin(yaw_b)
            rel = np.array([wx - b["cx"], wy - b["cy"]], dtype=np.float64)
            bx = cb * rel[0] + sb * rel[1]
            by = -sb * rel[0] + cb * rel[1]
            if abs(bx) <= float(b["hx"]) + 0.005 and abs(by) <= float(b["hy"]) + 0.005:
                crossed = True
                break
        if crossed:
            break
    if _math.isfinite(best) and best < 0.02:
        crossed = True
    return best, crossed, (float(nose[0]), float(nose[1]))
