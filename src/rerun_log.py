"""Kevin Rerun logger — RGB, depth/height, obstacle map (prep; no drive).

Rerun 0.37+ APIs: set_time / connect_grpc / save (not set_time_seconds / connect).
Default: record to ~/.kevin/rerun/live.rrd so headless Orin keeps a replayable log.
Optional: --rerun-connect / --rerun-spawn from main.

BACKPRESSURE HANDLING (fail-soft for 30 Hz capture loop):
- All logging is async via background thread with bounded queue (maxsize=2)
- When queue is full (network/gRPC sink slow), oldest frame is DROPPED
- Capture loop never blocks on Rerun — put_nowait() fails fast if queue full
- Background thread continues logging when sink recovers
"""

from __future__ import annotations

import os
import time
import threading
import queue
from typing import Any, Mapping, Optional

import numpy as np

try:
    import rerun as rr

    HAS_RERUN = True
except (ImportError, TypeError):
    rr = None  # type: ignore
    HAS_RERUN = False

DEFAULT_SAVE = None  # never unbounded save unless --rerun-save


class KevinRerunLogger:
    """Throttle-friendly logger for vision transfer debugging.
    
    All logging is async (background thread) to prevent gRPC backpressure
    from blocking the 30 Hz capture loop. Queue is bounded (maxsize=2);
    when full, oldest frame is dropped.
    """

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
        self._log_queue: Optional[queue.Queue] = None
        self._log_thread: Optional[threading.Thread] = None
        self._shutdown_event: Optional[threading.Event] = None
        self._drop_count = 0
        self._enqueue_count = 0
        
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
            # NOTE: rr.save() replaces sinks — do not combine with connect_grpc for live viz.
            try:
                os.makedirs(os.path.dirname(os.path.expanduser(save_path)) or ".", exist_ok=True)
                path = os.path.expanduser(save_path)
                rr.save(path)
                print("rerun_log: saving %s" % path)
                if connect_url:
                    print("rerun_log: WARNING save() may replace connect_grpc sink — prefer save_path=None when streaming")
            except Exception as e:
                print("rerun_log: save failed: %s" % e)
        
        # Start background logging thread with bounded queue (maxsize=2)
        self._log_queue = queue.Queue(maxsize=2)
        self._shutdown_event = threading.Event()
        self._log_thread = threading.Thread(target=self._log_worker, daemon=True, name="rerun-logger")
        self._log_thread.start()
        print("rerun_log: async thread started (queue maxsize=2, drop-oldest on full)")

    @property
    def due(self) -> bool:
        """True if the next maybe_log call will write (throttle gate)."""
        if not self.enabled:
            return False
        return ((self._n + 1) % self.every_n) == 0
    
    def _log_worker(self) -> None:
        """Background thread: pull log jobs from queue and write to Rerun.
        
        Runs until shutdown_event is set. Handles gRPC backpressure here
        so the main thread never blocks.
        """
        while not self._shutdown_event.is_set():
            try:
                # Block up to 0.5s for a job (allows clean shutdown)
                log_job = self._log_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # log_job is a dict with all the log data
            try:
                self._do_log(log_job)
            except Exception as e:
                tick = log_job.get("tick", 0)
                if tick < 3 or tick % 300 == 0:
                    print("rerun_log: worker log err %s" % e)
            finally:
                self._log_queue.task_done()
    
    def _do_log(self, log_job: dict) -> None:
        """Perform the actual Rerun logging (called by background thread)."""
        ts = log_job.get("ts")
        tick = log_job.get("tick", 0)
        
        # Set time timeline
        if ts and ts > 0:
            rr.set_time("capture", timestamp=float(ts))
        else:
            rr.set_time("capture", timestamp=time.time())
        rr.set_time("tick", sequence=tick)
        
        # Log all entities
        atlas = log_job.get("atlas")
        if atlas is not None:
            rr.log("vision/atlas", rr.Image(np.ascontiguousarray(atlas)))
        
        rgb = log_job.get("rgb")
        if rgb is not None:
            rr.log("vision/rgb", rr.Image(np.ascontiguousarray(rgb)))
        
        rs1 = log_job.get("rs1")
        if rs1 is not None:
            rr.log("vision/rs1_color", rr.Image(np.ascontiguousarray(rs1)))
        
        rs2 = log_job.get("rs2")
        if rs2 is not None:
            rr.log("vision/rs2_color", rr.Image(np.ascontiguousarray(rs2)))
        
        obs = log_job.get("obs")
        if obs is not None:
            o = np.ascontiguousarray(obs)
            if o.ndim == 2:
                rr.log("maps/obstacle", rr.Image(o))
        
        height_cm = log_job.get("height_cm")
        if height_cm is not None:
            h = np.ascontiguousarray(height_cm)
            if h.ndim == 2:
                hf = h.astype(np.float32, copy=False)
                rr.log("maps/height_cm", rr.DepthImage(hf, meter=100.0))
        
        safety = log_job.get("safety")
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
        
        robot_footprint_underlay = log_job.get("robot_footprint_underlay")
        if robot_footprint_underlay is not None:
            und = np.ascontiguousarray(robot_footprint_underlay)
            if und.ndim == 3 and und.shape[2] >= 3:
                rr.log("vision/robot_foot_underlay", rr.Image(und[:, :, :3]))
        
        robot_footprint_overlay = log_job.get("robot_footprint_overlay")
        if robot_footprint_overlay is not None:
            fov = np.ascontiguousarray(robot_footprint_overlay)
            # RGB only — avoid dual RGBA entities stacking opaque in the same view
            rgb_o = fov[:, :, :3] if (fov.ndim == 3 and fov.shape[2] >= 3) else fov
            rr.log("vision/robot_foot_overlay", rr.Image(rgb_o))
            rr.log("vision/depth_self_mask", rr.Image(rgb_o))

        rs1_mask_overlay = log_job.get("rs1_mask_overlay")
        if rs1_mask_overlay is not None:
            overlay = np.ascontiguousarray(rs1_mask_overlay)
            rgb_o = overlay[:, :, :3] if (overlay.ndim == 3 and overlay.shape[2] >= 3) else overlay
            rr.log("vision/rs1_mask_overlay", rr.Image(rgb_o))
            rr.log("vision/rs1_depth_mask", rr.Image(rgb_o))

        rs1_trust_mask_overlay = log_job.get("rs1_trust_mask_overlay")
        if rs1_trust_mask_overlay is not None:
            # Safety sensor trust/FOV (where "clear" is believed).
            tov = np.ascontiguousarray(rs1_trust_mask_overlay)
            rr.log("vision/rs1_trust_mask_overlay", rr.Image(tov))

        rs1_safety_foot_overlay = log_job.get("rs1_safety_foot_overlay")
        if rs1_safety_foot_overlay is not None:
            # Safety FOOT padded rect (forward-scan box) — orange.
            sof = np.ascontiguousarray(rs1_safety_foot_overlay)
            rgb_o = sof[:, :, :3] if (sof.ndim == 3 and sof.shape[2] >= 3) else sof
            rr.log("vision/rs1_safety_foot", rr.Image(rgb_o))

        safety_foot_overlay = log_job.get("safety_foot_overlay")
        if safety_foot_overlay is not None:
            sof2 = np.ascontiguousarray(safety_foot_overlay)
            rgb_o = sof2[:, :, :3] if (sof2.ndim == 3 and sof2.shape[2] >= 3) else sof2
            rr.log("vision/safety_foot", rr.Image(rgb_o))

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
        rs1_mask_overlay: Optional[np.ndarray] = None,
        rs1_trust_mask_overlay: Optional[np.ndarray] = None,
        rs1_safety_foot_overlay: Optional[np.ndarray] = None,
        robot_footprint_underlay: Optional[np.ndarray] = None,
        robot_footprint_overlay: Optional[np.ndarray] = None,
        safety_foot_overlay: Optional[np.ndarray] = None,
        force: bool = False,
    ) -> bool:
        """Log a tick (non-blocking). Returns True if enqueued for logging.
        
        BACKPRESSURE: When queue is full, drops oldest frame and enqueues new one.
        Never blocks the caller - returns immediately.
        """
        if not self.enabled:
            return False
        self._n += 1
        if not force and (self._n % self.every_n) != 0:
            return False
        
        # Build log job dict (shallow dict, not copying arrays yet)
        log_job = {
            "ts": ts,
            "tick": self._n,
            "atlas": atlas,
            "rgb": rgb,
            "rs1": rs1,
            "rs2": rs2,
            "obs": obs,
            "height_cm": height_cm,
            "safety": safety,
            "rs1_mask_overlay": rs1_mask_overlay,
            "rs1_trust_mask_overlay": rs1_trust_mask_overlay,
            "rs1_safety_foot_overlay": rs1_safety_foot_overlay,
            "robot_footprint_underlay": robot_footprint_underlay,
            "robot_footprint_overlay": robot_footprint_overlay,
            "safety_foot_overlay": safety_foot_overlay,
        }
        
        try:
            # Non-blocking put - if queue full, drop oldest and retry
            self._log_queue.put_nowait(log_job)
            self._enqueue_count += 1
            return True
        except queue.Full:
            # Queue full: drop oldest frame, then enqueue new one
            try:
                dropped = self._log_queue.get_nowait()
                self._drop_count += 1
                if self._drop_count == 1 or self._drop_count % 30 == 0:
                    print("rerun_log: BACKPRESSURE - dropped %d frames (queue full)" % self._drop_count)
            except queue.Empty:
                pass  # Race: queue emptied between full check and get
            
            # Try to enqueue new frame (should succeed now)
            try:
                self._log_queue.put_nowait(log_job)
                self._enqueue_count += 1
                return True
            except queue.Full:
                # Still full (unlikely) - just drop this frame
                self._drop_count += 1
                return False
    
    def shutdown(self, timeout: float = 2.0) -> None:
        """Graceful shutdown: flush queue and stop background thread."""
        if not self.enabled or self._shutdown_event is None:
            return
        
        print("rerun_log: shutting down (enqueued=%d, dropped=%d)..." % (self._enqueue_count, self._drop_count))
        self._shutdown_event.set()
        
        if self._log_thread and self._log_thread.is_alive():
            self._log_thread.join(timeout=timeout)
            if self._log_thread.is_alive():
                print("rerun_log: worker thread did not exit in %.1fs" % timeout)
        
        print("rerun_log: shutdown complete")


def available() -> bool:
    return HAS_RERUN
