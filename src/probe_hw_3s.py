#!/usr/bin/env python3
"""3-second IMU + wheel-encoder go/no-go. Never raises; prints a verdict."""
from __future__ import annotations

import os
import sys
import time
import traceback

DUR = 3.0
RS2 = os.environ.get("KEVIN_RS2", "944622074292")


def _canable_iface():
    """USB CANable (1d50:606f / gs_usb). Name flips can0↔can1 across boots."""
    env = os.environ.get("CAN_IFACE", "").strip()
    if env:
        return env
    for name in sorted(os.listdir("/sys/class/net")):
        if not name.startswith("can"):
            continue
        path = os.path.realpath(os.path.join("/sys/class/net", name, "device"))
        cur = path
        for _ in range(8):
            vend = os.path.join(cur, "idVendor")
            prod = os.path.join(cur, "idProduct")
            if os.path.isfile(vend) and os.path.isfile(prod):
                try:
                    v = open(vend).read().strip()
                    p = open(prod).read().strip()
                except OSError:
                    break
                if v == "1d50" and p == "606f":
                    return name
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return "can1"


CAN = _canable_iface()


def _ok(name, good, detail):
    print("%s %s  %s" % (name, "GO" if good else "NO-GO", detail), flush=True)
    return bool(good)


def probe_imu():
    try:
        from imu import IMUPipeline
    except Exception as e:
        return _ok("imu", False, "import %s" % e)
    imu = None
    try:
        imu = IMUPipeline(RS2)
        if not imu.ok:
            return _ok("imu", False, "pipeline not ok (serial %s)" % RS2)
        t0 = time.monotonic()
        n = 0
        gmax = 0.0
        amax = 0.0
        while time.monotonic() - t0 < DUR:
            if imu.grab():
                n += 1
                gmax = max(gmax, float((imu.gyro ** 2).sum() ** 0.5))
                amax = max(amax, float((imu.accel ** 2).sum() ** 0.5))
            time.sleep(0.002)
        dt = time.monotonic() - t0
        hz = n / max(dt, 1e-6)
        # Sitting: gyro ~0, accel ~9.8. Need samples, not motion.
        good = n >= 80 and 6.0 <= amax <= 20.0
        return _ok(
            "imu",
            good,
            "n=%d  %.0f Hz  |g|_max=%.3f  |a|_max=%.2f  (need n>=80 and |a|~g)"
            % (n, hz, gmax, amax),
        )
    except Exception as e:
        traceback.print_exc()
        return _ok("imu", False, str(e))
    finally:
        try:
            if imu is not None:
                imu.stop()
        except Exception:
            pass


def _bring_can(iface):
    import subprocess
    try:
        out = subprocess.run(
            ["ip", "-br", "link", "show", iface],
            capture_output=True, text=True, timeout=2)
        line = (out.stdout or "").strip()
        if "UP" in line and "DOWN" not in line.split()[1:2]:
            return line
    except Exception:
        line = ""
    try:
        subprocess.run(
            ["sudo", "-n", "ip", "link", "set", iface, "up",
             "type", "can", "bitrate", "250000"],
            capture_output=True, timeout=3)
        subprocess.run(
            ["sudo", "-n", "ip", "link", "set", iface, "txqueuelen", "1000"],
            capture_output=True, timeout=2)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ip", "-br", "link", "show", iface],
            capture_output=True, text=True, timeout=2)
        return (out.stdout or "").strip() or "missing"
    except Exception as e:
        return "err %s" % e


def probe_wheels():
    print("can: using %s (USB CANable 1d50:606f). mttcan is the other can*." % CAN, flush=True)
    print("can: %s" % _bring_can(CAN), flush=True)
    try:
        import wheelbase as wb_mod
    except Exception as e:
        return _ok("wheels", False, "import %s" % e)
    wb = None
    try:
        os.environ["CAN_IFACE"] = CAN
        wb = wb_mod.WheelBase(can_interface=CAN)
    except Exception as e:
        traceback.print_exc()
        return _ok("wheels", False, "WheelBase init: %s" % e)
    try:
        t0 = time.monotonic()
        n = 0
        n_ok = 0
        vbus = -1.0
        ages = []
        vels = []
        while time.monotonic() - t0 < DUR:
            try:
                snap = wb.get_encoder_snapshot()
                n += 1
                if snap.get("ok", 0) > 0.5 and snap.get("age_s", 99) < 0.4:
                    n_ok += 1
                ages.append(float(snap.get("age_s", 99)))
                vels.append((float(snap.get("vl", 0)), float(snap.get("vr", 0))))
            except Exception:
                pass
            try:
                vbus = float(wb.left.get_vbus_voltage() or 0.0)
            except Exception:
                pass
            time.sleep(0.02)
        health = {}
        try:
            health = wb.get_encoder_health()
        except Exception:
            pass
        age_min = min(ages) if ages else 99.0
        sitting = all(abs(a) < 0.05 and abs(b) < 0.05 for a, b in vels) if vels else False
        batt_ok = 28.0 <= vbus <= 44.0
        enc_ok = n_ok >= 20 and age_min < 0.4
        good = batt_ok and enc_ok
        return _ok(
            "wheels",
            good,
            "vbus=%.1fV  enc_ok=%d/%d  age_min=%.3fs  sitting=%s  health=%s"
            % (vbus, n_ok, n, age_min, sitting, health),
        )
    except Exception as e:
        traceback.print_exc()
        return _ok("wheels", False, str(e))
    finally:
        try:
            if wb is not None:
                wb.shutdown()
        except Exception:
            pass


def main():
    print("probe_hw_3s: %.1fs IMU then %.1fs wheels  can=%s rs2=%s" % (
        DUR, DUR, CAN, RS2), flush=True)
    imu_ok = probe_imu()
    wh_ok = probe_wheels()
    ready = imu_ok and wh_ok
    print("READY" if ready else "NOT-READY  (fix before 5 min wander)", flush=True)
    return 0 if ready else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("NOT-READY  probe crashed (caught)", flush=True)
        sys.exit(2)
