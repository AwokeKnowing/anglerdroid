"""Dynamic cell tracking for SLAM outlier filtering.

Tracks per-cell label history to identify transient obstacles (people, dog,
moving chairs) that should not permanently contribute to SLAM keyframes or
loop-closure maps. Uses a sliding window of recent observations to compute
a dynamic confidence score.

Contract:
- Only CLEAR↔OBSTACLE flips count (SELF and UNKNOWN are ignored)
- Dynamic confidence ∈ [0, 1]: 0 = static, 1 = highly dynamic
- Threshold tunable via KEVIN_DYNAMIC_THRESH (default 0.5)

Usage:
    tracker = DynamicTracker(history_len=10)
    for each frame:
        labels = get_ego_labels()  # UNKNOWN|SELF|CLEAR|OBSTACLE
        dynamic_mask = tracker.update(labels)
        # Use dynamic_mask in SLAM: 255 where dynamic, 0 where static
"""
from __future__ import annotations

import os
import numpy as np
from collections import deque

from .labels import UNKNOWN, SELF, CLEAR, OBSTACLE

# How many frames to track for flip detection
HISTORY_LEN = int(os.getenv('KEVIN_DYNAMIC_HISTORY', '10'))

# Flip ratio threshold: ≥0.5 → dynamic
DYNAMIC_THRESH = float(os.getenv('KEVIN_DYNAMIC_THRESH', '0.5'))


class DynamicTracker:
    """Tracks cell label history to identify dynamic obstacles."""

    __slots__ = ('_history', '_history_len', '_dynamic_thresh', '_frame_count')

    def __init__(self, history_len: int = HISTORY_LEN, 
                 dynamic_thresh: float = DYNAMIC_THRESH):
        """
        Parameters
        ----------
        history_len : int
            Number of recent frames to track (default 10).
        dynamic_thresh : float
            Flip ratio threshold for dynamic classification (default 0.5).
        """
        self._history = deque(maxlen=history_len)
        self._history_len = history_len
        self._dynamic_thresh = dynamic_thresh
        self._frame_count = 0

    def update(self, labels: np.ndarray) -> np.ndarray:
        """Update history and return dynamic mask.

        Parameters
        ----------
        labels : (H, W) uint8
            Current frame ego labels (UNKNOWN|SELF|CLEAR|OBSTACLE).

        Returns
        -------
        dynamic_mask : (H, W) uint8
            255 where cell is dynamic (high flip rate), 0 where static.
        """
        h, w = labels.shape
        self._frame_count += 1

        # Store only CLEAR/OBSTACLE (collapse UNKNOWN/SELF → 0, keep 2/3)
        obs_only = np.where((labels == CLEAR) | (labels == OBSTACLE),
                            labels, 0).astype(np.uint8)
        self._history.append(obs_only)

        # Need at least 3 frames for meaningful flip detection
        if len(self._history) < 3:
            return np.zeros((h, w), dtype=np.uint8)

        # Count CLEAR↔OBSTACLE flips per cell
        flip_count = np.zeros((h, w), dtype=np.int16)
        prev = self._history[0]
        for curr in list(self._history)[1:]:
            # Only count flips where both prev and curr are sensed (not 0)
            flipped = (prev > 0) & (curr > 0) & (prev != curr)
            flip_count[flipped] += 1
            prev = curr

        # Dynamic confidence: flip_count / (history_len - 1)
        max_flips = len(self._history) - 1
        confidence = flip_count.astype(np.float32) / max(max_flips, 1)

        # Threshold: ≥0.5 flip ratio → dynamic
        dynamic_mask = np.where(confidence >= self._dynamic_thresh, 
                                255, 0).astype(np.uint8)

        return dynamic_mask

    def get_stats(self) -> dict:
        """Return tracker statistics."""
        return {
            'frame_count': self._frame_count,
            'history_len': len(self._history),
            'history_max': self._history_len,
            'dynamic_thresh': self._dynamic_thresh,
        }

    def reset(self):
        """Clear history (for testing or scene change)."""
        self._history.clear()
        self._frame_count = 0
