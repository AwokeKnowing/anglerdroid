"""Measure the publish rate and age of emu shm camera blocks (read-only: polls the header, never writes).
python bench/slam/shm_probe.py [secs] [block ...]   default: 60 s, kevin_rs1 kevin_rs2 kevin_mast
Per block: frames, rate, inter-frame interval p50/p95/max (ms), age at arrival (now - frame stamp) p50/p95 (ms)."""
import sys, time, json
import numpy as np
from multiprocessing import shared_memory
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
names = sys.argv[2:] or ["kevin_rs1", "kevin_rs2", "kevin_mast"]
H = {}
for n in names:
    s = shared_memory.SharedMemory(name=n)
    try:
        from multiprocessing import resource_tracker; resource_tracker.unregister(s._name, "shared_memory")
    except Exception: pass
    H[n] = (s, np.ndarray((8,), np.uint64, s.buf, 0))
last = {n: int(h[1][2]) for n, h in H.items()}; arr = {n: [] for n in names}; age = {n: [] for n in names}
t_end = time.monotonic() + secs
while time.monotonic() < t_end:
    for n, (s, h) in H.items():
        c = int(h[2])
        if c != last[n] and not (int(h[0]) & 1):
            now = time.monotonic_ns(); last[n] = c; arr[n].append(now); age[n].append((now - int(h[1])) / 1e6)
    time.sleep(0.0003)
out = {}
for n in names:
    a = np.diff(np.array(arr[n])) / 1e6
    if len(a) < 2: out[n] = dict(frames=len(arr[n])); continue
    out[n] = dict(frames=len(arr[n]), hz=round(1000.0 / a.mean(), 2), int_p50_ms=round(float(np.percentile(a, 50)), 1), int_p95_ms=round(float(np.percentile(a, 95)), 1),
                  int_max_ms=round(float(a.max()), 1), age_p50_ms=round(float(np.percentile(age[n], 50)), 1), age_p95_ms=round(float(np.percentile(age[n], 95)), 1))
print(json.dumps(out))
