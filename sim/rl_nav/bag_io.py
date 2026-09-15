"""Read Kevin raw bags. Prefer meta.intrinsics size — not the 848 default."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_meta(bag: Path) -> dict:
    p = Path(bag) / "meta.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def depth_hw(meta: dict, nbytes: int, cam: str = "rs1") -> tuple[int, int, int]:
    """Return (n_frames, H, W) that divides the z16 blob."""
    n_pix = int(nbytes) // 2
    if n_pix <= 0:
        raise ValueError("empty z16")
    candidates = []
    intr = (meta.get("intrinsics") or {}).get(cam) or {}
    w = int(float(intr.get("width") or 0))
    h = int(float(intr.get("height") or 0))
    if w > 0 and h > 0:
        candidates.append((h, w))
    rs = meta.get("rs") or []
    if len(rs) >= 2:
        candidates.append((int(rs[0]), int(rs[1])))
    for pair in ((480, 640), (480, 848), (640, 480)):
        candidates.append(pair)
    seen = set()
    for h, w in candidates:
        if h <= 0 or w <= 0 or (h, w) in seen:
            continue
        seen.add((h, w))
        hw = h * w
        if n_pix % hw == 0:
            return n_pix // hw, h, w
    raise ValueError("z16 nbytes=%d does not match any HxW (tried %s)" % (
        nbytes, list(seen)))


def mmap_z16(path: Path, meta: dict, cam: str) -> np.ndarray:
    path = Path(path)
    nbytes = path.stat().st_size
    n, h, w = depth_hw(meta, nbytes, cam=cam)
    raw = np.memmap(path, dtype=np.uint16, mode="r", shape=(n, h, w))
    return raw
