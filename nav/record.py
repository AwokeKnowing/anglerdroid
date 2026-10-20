"""Replay log: KEVIN_RECORD=<dir> -> one compressed .npz per control tick (RS1/RS2 depth z16, IMU accel+gyro,
wheel speeds, command, pose, safety state) written by a background thread; drops ticks rather than slowing the loop."""
from __future__ import annotations
import json, os, queue, threading, time
import numpy as np

LIGHT = os.environ.get("KEVIN_RECORD_LIGHT") == "1"


class Recorder:
    def __init__(self, d):
        self.d = os.path.abspath(d); os.makedirs(self.d, exist_ok=True); self.q = queue.Queue(maxsize=8); self.n = 0; self.dropped = 0
        threading.Thread(target=self._w, daemon=True, name="record").start()

    def put(self, **kw):
        try: self.q.put_nowait(kw)
        except queue.Full: self.dropped += 1

    def _w(self):
        while True:
            kw = self.q.get(); i = self.n; self.n += 1
            arrs = {k: np.asarray(v) for k, v in kw.items() if isinstance(v, np.ndarray)}
            # KEVIN_RECORD_LIGHT=1: no images (sensorlog has them); the compressed 848x480 depth made the writer drop
            # ~half the ticks (L11-L13: 10-15 Hz recorded), losing gyro/odom samples
            if LIGHT: arrs = {k: v for k, v in arrs.items() if v.size <= 4096}
            meta = {k: v for k, v in kw.items() if not isinstance(v, np.ndarray)}
            np.savez_compressed(os.path.join(self.d, "t%06d.npz" % i), meta=json.dumps(meta, default=float), **arrs)
