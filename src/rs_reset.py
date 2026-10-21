"""RealSense hardware reset + wait for USB re-enumeration (run BEFORE the stack opens the cameras; never while a pipeline
is streaming in another process).
Why (2026-10-21): on the first stack start after another process used the cameras, RS1 re-enumerated on USB ~7 s into
startup (dmesg 'Found UVC 1.50 device ... 435'), the safety core's pipeline died with it and nav refused to start
("safety core did not start (RS1)"); the second start always worked.  A deliberate reset + wait puts every device in
that known-good state first.
  python -m src.rs_reset [serial ...]        (default: all connected RealSenses)"""
import sys, time


def present(rs=None):
    import pyrealsense2 as _rs
    rs = rs or _rs
    try: return {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}
    except Exception: return {}


def wait_present(serials, timeout=30.0, stable_s=2.0):
    """-> (ok, seconds, missing) once every serial has been enumerated continuously for stable_s."""
    t0 = time.monotonic(); since = None
    while time.monotonic() - t0 < timeout:
        miss = [s for s in serials if s not in present()]
        if not miss:
            since = since or time.monotonic()
            if time.monotonic() - since >= stable_s: return True, round(time.monotonic() - t0, 1), []
        else: since = None
        time.sleep(0.25)
    return False, round(time.monotonic() - t0, 1), [s for s in serials if s not in present()]


def reset_and_wait(serials=None, timeout=30.0, log=print):
    """hardware_reset each device, wait until it has dropped off and come back.  -> dict(ok, secs, missing, reset)"""
    import pyrealsense2 as rs
    devs = present(rs); serials = list(serials or devs)
    missing0 = [s for s in serials if s not in devs]
    if missing0: log("rs_reset: not enumerated before reset: %s (waiting)" % missing0)
    done = []
    for s in serials:
        d = devs.get(s)
        if d is None: continue
        try: d.hardware_reset(); done.append(s)
        except Exception as e: log("rs_reset: %s hardware_reset failed: %s" % (s, e))
    t0 = time.monotonic()
    while done and time.monotonic() - t0 < 5.0 and any(s in present(rs) for s in done):   # wait for the drop-off
        time.sleep(0.2)
    ok, secs, miss = wait_present(serials, timeout)
    r = dict(ok=ok, secs=round(time.monotonic() - t0, 1), missing=miss, reset=done)
    log("rs_reset: %s" % r); return r


if __name__ == "__main__":
    print(reset_and_wait(sys.argv[1:] or None))
