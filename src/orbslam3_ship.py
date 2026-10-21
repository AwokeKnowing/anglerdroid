"""Ship RS2 stereo IR to i777 ORB-SLAM3. Off the 30 Hz capture thread.

Capture only memcpy's IR into preallocated camera buffers (existing
capture_ir path). This module JPEG-free TCP-sends those buffers from a
queue-1 side thread so encode/network can never stall grab.

Protocol (little-endian), one TCP frame:
  u32 nbytes
  u64 ts_ns
  u16 w, h
  f32 fx, fy, cx, cy, baseline
  uint8 left[w*h]
  uint8 right[w*h]
"""
from __future__ import annotations

import os
import queue
import socket
import struct
import threading
import time

import numpy as np

FRAME_STRUCT = struct.Struct("<QHH5f")  # after nbytes
DEFAULT_PORT = 5571
DEFAULT_HZ = 20.0
DEFAULT_IR = (848, 480, 424.0, 424.0, 424.0, 240.0, 0.05)


def pack_stereo(ts_ns, left, right, fx, fy, cx, cy, baseline) -> bytes:
    left = np.ascontiguousarray(left, dtype=np.uint8)
    right = np.ascontiguousarray(right, dtype=np.uint8)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("stereo pair must be matching HxW uint8")
    h, w = left.shape
    body = FRAME_STRUCT.pack(int(ts_ns), int(w), int(h),
                             float(fx), float(fy), float(cx), float(cy),
                             float(baseline))
    body += left.tobytes() + right.tobytes()
    return struct.pack("<I", len(body)) + body


def unpack_stereo(frame: bytes):
    if len(frame) < 4:
        raise ValueError("short frame")
    (nbytes,) = struct.unpack_from("<I", frame, 0)
    body = frame[4:]
    if len(body) != nbytes:
        raise ValueError("length mismatch")
    ts_ns, w, h, fx, fy, cx, cy, baseline = FRAME_STRUCT.unpack_from(body, 0)
    pix = int(w) * int(h)
    off = FRAME_STRUCT.size
    left = np.frombuffer(body, dtype=np.uint8, count=pix, offset=off).reshape(h, w)
    right = np.frombuffer(body, dtype=np.uint8, count=pix, offset=off + pix).reshape(h, w)
    return ts_ns, left, right, fx, fy, cx, cy, baseline


def ir_calib_from_profile(profile):
    """Return (w, h, fx, fy, cx, cy, baseline_m) from a D435 stereo IR profile."""
    try:
        import pyrealsense2 as rs
        ir1_s = profile.get_stream(rs.stream.infrared, 1)
        ir2_s = profile.get_stream(rs.stream.infrared, 2)
        ir1 = ir1_s.as_video_stream_profile().get_intrinsics()
        extr = ir2_s.get_extrinsics_to(ir1_s)
        return (ir1.width, ir1.height, ir1.fx, ir1.fy, ir1.ppx, ir1.ppy,
                abs(float(extr.translation[0])))
    except Exception:
        return DEFAULT_IR


class OrbSlam3Ship:
    """Queue-1 TCP publisher of the latest RS2 IR pair."""

    def __init__(self, host, port=DEFAULT_PORT, hz=DEFAULT_HZ, rs_camera=None):
        self.host = host
        self.port = int(port)
        self.dt = 1.0 / max(1.0, float(hz))
        self._rs = rs_camera
        self._q = queue.Queue(maxsize=1)
        self._running = False
        self._pack_thread = None
        self._send_thread = None
        self._sent = 0
        self._drop = 0
        self._calib = DEFAULT_IR

    def start(self):
        if self._running:
            return
        self._running = True
        if self._rs is not None and getattr(self._rs, "profile", None) is not None:
            self._calib = ir_calib_from_profile(self._rs.profile)
        self._pack_thread = threading.Thread(
            target=self._pack_loop, name="orbslam3-pack", daemon=True)
        self._send_thread = threading.Thread(
            target=self._send_loop, name="orbslam3-send", daemon=True)
        self._pack_thread.start()
        self._send_thread.start()
        print("orbslam3_ship: RS2 IR → %s:%d @ %.0f Hz (queue=1, raw TCP)"
              % (self.host, self.port, 1.0 / self.dt))

    def stop(self):
        self._running = False

    def _offer(self, payload):
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass
            self._drop += 1

    def _pack_loop(self):
        while self._running:
            t0 = time.monotonic()
            try:
                cam = self._rs
                if cam is not None and cam.ok and cam.ir_left is not None and cam.ir_right is not None:
                    left = np.empty_like(cam.ir_left)
                    right = np.empty_like(cam.ir_right)
                    np.copyto(left, cam.ir_left)
                    np.copyto(right, cam.ir_right)
                    _, _, fx, fy, cx, cy, baseline = self._calib
                    payload = pack_stereo(
                        int(time.time() * 1e9), left, right, fx, fy, cx, cy, baseline)
                    self._offer(payload)
            except Exception as e:
                if not getattr(self, "_pack_err", 0):
                    print("orbslam3_ship pack: %s" % e)
                self._pack_err = getattr(self, "_pack_err", 0) + 1
            dt = time.monotonic() - t0
            time.sleep(max(0.0, self.dt - dt))

    def _send_loop(self):
        sock = None
        last_log = 0.0
        while self._running:
            if sock is None:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2.0)
                    s.connect((self.host, self.port))
                    s.settimeout(None)
                    sock = s
                    print("orbslam3_ship: connected %s:%d" % (self.host, self.port))
                except Exception as e:
                    time.sleep(1.0)
                    now = time.monotonic()
                    if now - last_log > 10.0:
                        print("orbslam3_ship: waiting for %s:%d (%s)"
                              % (self.host, self.port, e))
                        last_log = now
                    continue
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                sock.sendall(payload)
                self._sent += 1
                if self._sent % 100 == 1:
                    print("orbslam3_ship: sent=%d drop=%d bytes=%d"
                          % (self._sent, self._drop, len(payload)))
            except Exception as e:
                print("orbslam3_ship: send failed (%s) — reconnect" % e)
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


def host_from_env():
    return os.environ.get("KEVIN_ORBSLAM3_HOST", "").strip()


def port_from_env():
    return int(os.environ.get("KEVIN_ORBSLAM3_PORT", str(DEFAULT_PORT)))


def hz_from_env():
    return float(os.environ.get("KEVIN_ORBSLAM3_HZ", str(DEFAULT_HZ)))
