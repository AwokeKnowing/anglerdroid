#!/usr/bin/env python3
"""24-hour domain-randomization watchdog for Newton casita wander.

Launches `newton_kevin.py --drive` episodes until a deadline (default 24h).
If a child dies or goes silent, it is killed and the next episode starts.
Furniture density, seed, and episode length are randomized each run so
behavior that survives this mix is likelier to transfer to the real house.

Does not need the Cursor agent to keep episodes going. The agent loop only
restarts *this* watchdog if it itself dies, and applies code tweaks between
episodes.
"""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "sim"
OUT = SIM / "overnight"
PY = REPO / ".venv-newton" / "bin" / "python"
HOURS = 24.0
STALL_S = 240.0
CLUTTER = ("sparse", "medium", "dense", "packed", "random")


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def heartbeat(payload: dict):
    lines = ["%s=%s" % (k, payload[k]) for k in payload]
    _write(OUT / "heartbeat.txt", "\n".join(lines) + "\n")
    _write(OUT / "STATUS.txt", "\n".join(lines) + "\n")


def parse_stats(path: Path) -> dict:
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def child_alive(proc) -> bool:
    return proc.poll() is None


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
    while proc.poll() is None and time.time() - t0 < 8.0:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def launch_episode(run_i, seed, clutter, seconds):
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(SIM) + os.pathsep + env.get("PYTHONPATH", "")
    gif = OUT / "last.gif"
    cmd = [
        str(PY if PY.is_file() else sys.executable),
        str(SIM / "newton_kevin.py"),
        "--drive",
        "--seed", str(int(seed)),
        "--clutter", str(clutter),
        "--seconds", str(float(seconds)),
        "--capture-fps", "6",
        "--gif", str(gif),
        "--device", "cuda:0",
    ]
    log_path = OUT / ("run_%04d.log" % run_i)
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("cmd %s\n" % " ".join(cmd))
    log_f.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(SIM),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_f, log_path


def deadline_path():
    return OUT / "deadline.txt"


def load_deadline(hours=HOURS) -> float:
    p = deadline_path()
    if p.is_file():
        try:
            return float(p.read_text().strip())
        except Exception:
            pass
    end = time.time() + hours * 3600.0
    _write(p, "%.3f\n" % end)
    return end


def append_ledger(row: dict):
    with open(OUT / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    end = load_deadline(HOURS)
    pid_path = OUT / "watchdog.pid"
    _write(pid_path, str(os.getpid()) + "\n")
    rng = random.Random(int(time.time()) ^ os.getpid())
    run_i = 0
    ledger = OUT / "ledger.jsonl"
    if ledger.is_file():
        run_i = sum(1 for _ in ledger.open())
    print(
        "overnight watchdog pid=%d deadline_in=%.1fh out=%s"
        % (os.getpid(), max(0.0, (end - time.time()) / 3600.0), OUT),
        flush=True,
    )
    while time.time() < end:
        run_i += 1
        seed = rng.randint(1, 99999)
        clutter = rng.choice(CLUTTER)
        seconds = rng.choice((100.0, 130.0, 160.0))
        wall_budget = max(180.0, seconds / 0.35 + 90.0)
        heartbeat({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix": "%.0f" % time.time(),
            "pid": os.getpid(),
            "status": "starting",
            "run": run_i,
            "seed": seed,
            "clutter": clutter,
            "seconds": seconds,
            "deadline_unix": "%.0f" % end,
            "hours_left": "%.2f" % ((end - time.time()) / 3600.0),
        })
        proc, log_f, log_path = launch_episode(run_i, seed, clutter, seconds)
        last_sz = 0
        last_byte_t = time.time()
        t0 = time.time()
        exit_code = None
        stall = False
        timeout = False
        while True:
            now = time.time()
            if now >= end:
                kill_tree(proc)
                timeout = True
                break
            rc = proc.poll()
            if rc is not None:
                exit_code = rc
                break
            try:
                sz = log_path.stat().st_size
            except OSError:
                sz = last_sz
            if sz > last_sz:
                last_sz = sz
                last_byte_t = now
            if now - last_byte_t > STALL_S:
                stall = True
                kill_tree(proc)
                exit_code = "stall"
                break
            if now - t0 > wall_budget:
                timeout = True
                kill_tree(proc)
                exit_code = "timeout"
                break
            heartbeat({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "unix": "%.0f" % now,
                "pid": os.getpid(),
                "child_pid": proc.pid,
                "status": "running",
                "run": run_i,
                "seed": seed,
                "clutter": clutter,
                "seconds": seconds,
                "wall_s": "%.0f" % (now - t0),
                "log_bytes": sz,
                "deadline_unix": "%.0f" % end,
                "hours_left": "%.2f" % ((end - now) / 3600.0),
            })
            time.sleep(4.0)
        try:
            log_f.close()
        except Exception:
            pass
        stats = parse_stats(OUT / "kevin_newton_drive_stats.txt")
        row = {
            "run": run_i,
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": seed,
            "clutter": clutter,
            "seconds": seconds,
            "exit": exit_code if exit_code is not None else ("timeout" if timeout else "ok"),
            "stall": stall,
            "wall_s": round(time.time() - t0, 2),
            "rooms": stats.get("rooms_visited", ""),
            "visit_frac": stats.get("visit_fraction", ""),
            "path_m": stats.get("path_length_m", ""),
            "rt": "",
            "stuck": stats.get("stuck_recoveries", ""),
            "door": stats.get("door_xy", ""),
            "plan": stats.get("wander_plan", ""),
            "log": str(log_path.name),
        }
        try:
            text = log_path.read_text(errors="replace")
            for line in text.splitlines():
                if "timing sim=" in line and "rt=" in line:
                    for tok in line.split():
                        if tok.startswith("rt="):
                            row["rt"] = tok.split("=", 1)[1]
        except Exception:
            pass
        append_ledger(row)
        snap = OUT / ("run_%04d_stats.txt" % run_i)
        src_stats = OUT / "kevin_newton_drive_stats.txt"
        if src_stats.is_file():
            snap.write_text(src_stats.read_text())
        heartbeat({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix": "%.0f" % time.time(),
            "pid": os.getpid(),
            "status": "idle",
            "run": run_i,
            "last_exit": row["exit"],
            "last_rooms": row["rooms"],
            "last_rt": row["rt"],
            "deadline_unix": "%.0f" % end,
            "hours_left": "%.2f" % ((end - time.time()) / 3600.0),
        })
        print("episode %d done exit=%s rooms=%s rt=%s" % (
            run_i, row["exit"], row["rooms"], row["rt"],
        ), flush=True)
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
    print("overnight watchdog finished", flush=True)
    try:
        pid_path.unlink()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
