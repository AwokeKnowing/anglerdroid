#!/usr/bin/env python3
"""Watchdog for insect-explore training. Restarts a dead/stale trainer."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "sim"
OUT = SIM / "explore"
PY = REPO / ".venv-newton" / "bin" / "python"
HOURS = 20.0
STALL_S = 180.0


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def heartbeat(payload: dict):
    lines = ["%s=%s" % (k, payload[k]) for k in payload]
    _write(OUT / "watchdog_heartbeat.txt", "\n".join(lines) + "\n")


def child_alive(proc) -> bool:
    return proc is not None and proc.poll() is None


def kill_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < 10.0:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def load_deadline(hours=HOURS) -> float:
    p = OUT / "deadline.txt"
    if p.is_file():
        try:
            return float(p.read_text().strip())
        except Exception:
            pass
    end = time.time() + hours * 3600.0
    _write(p, "%.3f\n" % end)
    return end


def launch():
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(SIM) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [str(PY if PY.is_file() else sys.executable), "-u", str(SIM / "explore_train.py")]
    log_path = OUT / "train.log"
    log_f = open(log_path, "a", encoding="utf-8")
    log_f.write("\n--- launch %s ---\n" % time.strftime("%Y-%m-%dT%H:%M:%S"))
    log_f.flush()
    proc = subprocess.Popen(
        cmd, cwd=str(SIM), env=env,
        stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_f, log_path


def hb_unix():
    p = OUT / "heartbeat.txt"
    if not p.is_file():
        return 0.0
    for line in p.read_text().splitlines():
        if line.startswith("unix="):
            try:
                return float(line.split("=", 1)[1])
            except Exception:
                return 0.0
    return 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    end = load_deadline(HOURS)
    _write(OUT / "watchdog.pid", str(os.getpid()) + "\n")
    print(
        "explore watchdog pid=%d deadline_in=%.2fh" % (
            os.getpid(), max(0.0, (end - time.time()) / 3600.0),
        ),
        flush=True,
    )
    launches = 0
    while time.time() < end:
        launches += 1
        proc, log_f, log_path = launch()
        heartbeat({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix": "%.0f" % time.time(),
            "pid": os.getpid(),
            "child": proc.pid,
            "status": "running",
            "launches": launches,
            "hours_left": "%.2f" % ((end - time.time()) / 3600.0),
            "deadline_unix": "%.0f" % end,
        })
        last_byte_t = time.time()
        last_sz = 0
        while time.time() < end:
            rc = proc.poll()
            if rc is not None:
                print("trainer exit", rc, "launch", launches, flush=True)
                break
            try:
                sz = log_path.stat().st_size
            except OSError:
                sz = last_sz
            if sz > last_sz:
                last_sz = sz
                last_byte_t = time.time()
            stale = time.time() - max(last_byte_t, hb_unix() or last_byte_t)
            if stale > STALL_S and time.time() - last_byte_t > STALL_S:
                print("stall, restarting trainer", flush=True)
                kill_tree(proc)
                break
            heartbeat({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "unix": "%.0f" % time.time(),
                "pid": os.getpid(),
                "child": proc.pid,
                "status": "running",
                "launches": launches,
                "hours_left": "%.2f" % ((end - time.time()) / 3600.0),
                "deadline_unix": "%.0f" % end,
                "log_bytes": last_sz,
            })
            time.sleep(5.0)
        try:
            log_f.close()
        except Exception:
            pass
        kill_tree(proc)
        if time.time() >= end:
            break
        time.sleep(2.0)
    heartbeat({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unix": "%.0f" % time.time(),
        "pid": os.getpid(),
        "status": "done",
        "hours_left": "0",
        "deadline_unix": "%.0f" % end,
    })
    print("explore watchdog finished", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
