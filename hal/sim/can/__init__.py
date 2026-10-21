"""Emulated python-can for the Jetson emulator.  can.Bus(<channel>, interface="socketcan") returns a bus that
talks to the emulated ODrive S1 nodes inside emu/world.py over a localhost UDP socket (one datagram per CAN
frame).  Same classes / exceptions / semantics the anglerdroid wheelbase + odrivecan modules use."""
from __future__ import annotations
import os, socket, struct, time, select

__version__ = "4.4.2-kevin-emu"
EMULATED = True
_PORT = int(os.environ.get("KEVIN_CAN_PORT", "47800"))
_FMT = struct.Struct("<dIB??8s")      # t, id, dlc, rtr, ext, data


class CanError(Exception): pass
class CanOperationError(CanError): pass
class CanInitializationError(CanError): pass
class CanTimeoutError(CanError, TimeoutError): pass


class Message:
    __slots__ = ("timestamp", "arbitration_id", "is_extended_id", "is_remote_frame", "is_error_frame",
                 "channel", "dlc", "data", "is_fd", "is_rx", "bitrate_switch", "error_state_indicator")

    def __init__(self, timestamp=0.0, arbitration_id=0, is_extended_id=True, is_remote_frame=False,
                 is_error_frame=False, channel=None, dlc=None, data=None, is_fd=False, is_rx=True, **kw):
        self.timestamp = timestamp; self.arbitration_id = arbitration_id; self.is_extended_id = is_extended_id
        self.is_remote_frame = is_remote_frame; self.is_error_frame = is_error_frame; self.channel = channel
        self.data = bytearray(data or b""); self.dlc = len(self.data) if dlc is None else dlc
        self.is_fd = is_fd; self.is_rx = is_rx; self.bitrate_switch = False; self.error_state_indicator = False

    def __repr__(self):
        return "can.Message(id=0x%x, data=%s)" % (self.arbitration_id, bytes(self.data).hex())


class BusABC:
    def __init__(self, channel=None, **kw): self.channel_info = str(channel)
    def __iter__(self):
        while True:
            m = self.recv(1.0)
            if m is not None: yield m
    def __enter__(self): return self
    def __exit__(self, *a): self.shutdown()
    def set_filters(self, filters=None): self._filters = filters
    def flush_tx_buffer(self): pass
    def shutdown(self): pass


class _EmuBus(BusABC):
    def __init__(self, channel=None, interface=None, bustype=None, **kw):
        super().__init__(channel)
        self.channel = channel
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        self._dst = ("127.0.0.1", _PORT)
        self._filters = None
        try:
            self._sock.sendto(_FMT.pack(time.time(), 0x7FF, 0, False, False, b"HELLO\0\0\0"), self._dst)   # register
        except OSError as e:
            raise CanInitializationError("emulated %s: %s" % (channel, e))
        self._closed = False

    def send(self, msg, timeout=None):
        if self._closed: raise CanOperationError("bus closed")
        d = bytes(msg.data)
        try:
            self._sock.sendto(_FMT.pack(time.time(), msg.arbitration_id, len(d), bool(msg.is_remote_frame),
                                        bool(msg.is_extended_id), d.ljust(8, b"\0")), self._dst)
        except OSError as e:
            raise CanOperationError(str(e))

    def _match(self, m):
        if not self._filters: return True
        return any((m.arbitration_id & f.get("can_mask", 0x7FF)) == (f["can_id"] & f.get("can_mask", 0x7FF)) for f in self._filters)

    def recv(self, timeout=None):
        t_end = None if timeout is None else time.monotonic() + timeout
        while True:
            rem = None if t_end is None else max(0.0, t_end - time.monotonic())
            r, _, _ = select.select([self._sock], [], [], rem)
            if not r: return None
            pkt, _ = self._sock.recvfrom(64)
            t, aid, dlc, rtr, ext, data = _FMT.unpack(pkt)
            m = Message(timestamp=t, arbitration_id=aid, is_extended_id=ext, is_remote_frame=rtr,
                        dlc=dlc, data=data[:dlc], channel=self.channel, is_rx=True)
            if self._match(m): return m
            if t_end is not None and time.monotonic() >= t_end: return None

    def shutdown(self):
        if not self._closed:
            self._closed = True; self._sock.close()


def Bus(channel=None, interface=None, bustype=None, **kw):
    return _EmuBus(channel, interface=interface or bustype, **kw)


class Notifier:
    def __init__(self, bus, listeners, timeout=1.0):
        import threading
        self.bus, self.listeners, self._run = bus, listeners, True
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
    def _loop(self):
        while self._run:
            m = self.bus.recv(0.1)
            if m is not None:
                for l in self.listeners: l(m)
    def stop(self, timeout=1.0): self._run = False


interface = type("interface", (), {"Bus": staticmethod(Bus)})
interfaces = {"socketcan": None, "virtual": None}
