"""Kevin Rerun logger — RGB, depth/height, obstacle map (prep; no drive).

Rerun 0.37+ APIs: set_time / connect_grpc / save (not set_time_seconds / connect).
Default: record to ~/.kevin/rerun/live.rrd so headless Orin keeps a replayable log.
Optional: --rerun-connect / --rerun-spawn from main.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Optional

import numpy as np

try:
    import rerun as rr

    HAS_RERUN = True
except (ImportError, TypeError):
    rr = None  # type: ignore
    HAS_RERUN = False

DEFAULT_SAVE = os.path.expanduser("~/.kevin/rerun/live.rrd")


class KevinRerunLogger:
    """Throttle-friendly logger for vision transfer debugging."""

    def __init__(
        self,
        enabled: bool = True,
        *,
        app_id: str = "kevin_anglerdroid",
        save_path: Optional[str] = DEFAULT_SAVE,
        connect_url: Optional[str] = None,
        spawn: bool = False,
        every_n: int = 6,
    ):
        self.enabled = bool(enabled and HAS_RERUN)
        self.every_n = max(1, int(every_n))
        self._n = 0
        self.save_path = save_path
        if not self.enabled:
            print("rerun_log: disabled (missing sdk or --no-rerun)")
            return

        rr.init(app_id)
        if spawn:
            try:
                rr.spawn(connect=True)
                print("rerun_log: spawned viewer")
            except Exception as e:
                print("rerun_log: spawn failed: %s" % e)
        if connect_url:
            try:
                rr.connect_grpc(connect_url)
                print("rerun_log: connect_grpc %s" % connect_url)
            except Exception as e:
                print("rerun_log: connect_grpc failed: %s" % e)
        if save_path and not spawn:
            # File sink is the headless default (Orin has no local viewer need).
            try:
                os.makedirs(os.path.dirname(os.path.expanduser(save_path)) or ".", exist_ok=True)
                path = os.path.expanduser(save_path)
                rr.save(path)
                print("rerun_log: saving %s" % path)
            except Exception as e:
                print("rerun_log: save failed: %s" % e)

    @property
    def due(self) -> bool:
        """True if the next maybe_log call will write (throttle gate)."""
        if not self.enabled:
            return False
        return ((self._n + 1) % self.every_n) == 0

    def maybe_log(
        self,
        *,
        ts: float,
        atlas: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        rs1: Optional[np.ndarray] = None,
        rs2: Optional[np.ndarray] = None,
        obs: Optional[np.ndarray] = None,
        height_cm: Optional[np.ndarray] = None,
        safety: Optional[Mapping[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """Log a tick. Returns True if this call actually wrote entities."""
        if not self.enabled:
            return False
        self._n += 1
        if not force and (self._n % self.every_n) != 0:
            return False
        try:
            # Prefer wall-clock capture time when available.
            if ts and ts > 0:
                rr.set_time("capture", timestamp=float(ts))
            else:
                rr.set_time("capture", timestamp=time.time())
            rr.set_time("tick", sequence=self._n)

            if atlas is not None:
                rr.log("vision/atlas", rr.Image(np.ascontiguousarray(atlas)))
            if rgb is not None:
                rr.log("vision/rgb", rr.Image(np.ascontiguousarray(rgb)))
            if rs1 is not None:
                rr.log("vision/rs1_color", rr.Image(np.ascontiguousarray(rs1)))
            if rs2 is not None:
                rr.log("vision/rs2_color", rr.Image(np.ascontiguousarray(rs2)))
            if obs is not None:
                # Binary / uint8 ego obstacle map (FOOT-cleared persistent_obs).
                o = np.ascontiguousarray(obs)
                if o.ndim == 2:
                    rr.log("maps/obstacle", rr.Image(o))
            if height_cm is not None:
                # Height is cm (1..100). meter=100 → DepthImage interprets units as cm.
                h = np.ascontiguousarray(height_cm)
                if h.ndim == 2:
                    # float32 depth channel keeps sparse zeros as invalid-ish.
                    hf = h.astype(np.float32, copy=False)
                    rr.log("maps/height_cm", rr.DepthImage(hf, meter=100.0))
            if safety:
                rr.log(
                    "safety",
                    rr.Scalars(
                        [
                            float(safety.get("fwd", 1.0)),
                            float(safety.get("bwd", 1.0)),
                            float(safety.get("ang", 1.0)),
                        ]
                    ),
                )
            return True
        except Exception as e:
            # Never let logging take down the 30 Hz loop.
            if self._n < 3 or self._n % 300 == 0:
                print("rerun_log: log err %s" % e)
            return False


def available() -> bool:
    return HAS_RERUN
