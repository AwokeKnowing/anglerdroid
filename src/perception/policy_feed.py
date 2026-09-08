"""Policy feed: honest ego labels + height export for neural policy.

Stable tensor layout for 30 Hz policy consumption (Tesla FSD-style).
See docs/perception/CONTRACT.md.

Layout
------
HxW uint8 labels: UNKNOWN=0, SELF=1, CLEAR=2, OBSTACLE=3
HxW uint8 height: obstacle height above floor in cm (0-100, clamped)

Honesty invariants (enforced by upstream labeling; export is zero-copy):
- SELF wins over all (never CLEAR or OBSTACLE)
- CLEAR is sensed floor only (never invented under chassis)
- OBSTACLE is non-self only
- UNKNOWN remains UNKNOWN when no depth evidence
"""
from __future__ import annotations

import numpy as np
from .labels import UNKNOWN, SELF, CLEAR, OBSTACLE


def export_policy_feed(
    labels: np.ndarray,
    height: np.ndarray,
    *,
    labels_out: np.ndarray | None = None,
    height_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Export honest ego labels + height for neural policy consumption.

    Zero-copy when labels_out/height_out are provided (preferred for 30 Hz).
    Otherwise allocates fresh arrays (test/offline use).

    Parameters
    ----------
    labels : (H, W) uint8
        Ego labels: UNKNOWN=0, SELF=1, CLEAR=2, OBSTACLE=3
    height : (H, W) uint8
        Obstacle height above floor (cm, 0-100 clamped)
    labels_out : (H, W) uint8, optional
        Preallocated output buffer for labels (zero-copy)
    height_out : (H, W) uint8, optional
        Preallocated output buffer for height (zero-copy)

    Returns
    -------
    labels_export : (H, W) uint8
        Policy-ready labels (reference to labels_out if provided)
    height_export : (H, W) uint8
        Policy-ready height (reference to height_out if provided)

    Notes
    -----
    - Honesty preserved: upstream labeling enforces SELF wins, no invented CLEAR
    - Export is a reference or copy; caller must not mutate until next call
    - Layout stable: UNKNOWN=0, SELF=1, CLEAR=2, OBSTACLE=3 (perception.labels)
    - Height encoding: uint8 cm above floor_clip (0 if no obstacle, 100 cap)
    """
    if labels_out is not None:
        np.copyto(labels_out, labels)
        labels_export = labels_out
    else:
        labels_export = np.array(labels, dtype=np.uint8, copy=True)

    if height_out is not None:
        np.copyto(height_out, height)
        height_export = height_out
    else:
        height_export = np.array(height, dtype=np.uint8, copy=True)

    return labels_export, height_export
