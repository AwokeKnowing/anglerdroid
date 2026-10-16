"""Minimal RFC 6455 WebSocket server side for http.server handlers (stdlib only; text frames, ping/pong, close)."""
from __future__ import annotations
import base64, hashlib, struct, threading

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WS:
    def __init__(self, handler):
        h = handler; key = h.headers.get("Sec-WebSocket-Key", "")
        acc = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        # raw status line: BaseHTTPRequestHandler answers HTTP/1.0, which WS clients (and browsers) reject for 101
        h.wfile.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                       "Sec-WebSocket-Accept: %s\r\n\r\n" % acc).encode()); h.wfile.flush()
        self.r = h.rfile; self.w = h.wfile; self.lock = threading.Lock(); self.closed = False
        h.close_connection = True

    def _frame(self, op, data: bytes):
        n = len(data); hdr = bytes([0x80 | op])
        if n < 126: hdr += bytes([n])
        elif n < 65536: hdr += bytes([126]) + struct.pack(">H", n)
        else: hdr += bytes([127]) + struct.pack(">Q", n)
        with self.lock:
            self.w.write(hdr + data); self.w.flush()

    def send(self, text: str):
        if self.closed: raise ConnectionError("closed")
        try: self._frame(0x1, text.encode())
        except Exception: self.closed = True; raise

    def _read(self, n):
        b = self.r.read(n)
        if b is None or len(b) < n: raise ConnectionError("eof")
        return b

    def recv(self):
        """-> str (text message) or None on close. Handles ping/pong and fragmented frames."""
        buf = b""
        while True:
            b0, b1 = self._read(2); op = b0 & 0x0F; n = b1 & 0x7F
            if n == 126: n = struct.unpack(">H", self._read(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._read(8))[0]
            if n > 1 << 20: self.closed = True; return None
            mask = self._read(4) if b1 & 0x80 else None; d = self._read(n)
            if mask: d = bytes(c ^ mask[i & 3] for i, c in enumerate(d))
            if op == 0x8:
                try: self._frame(0x8, b"")
                except Exception: pass
                self.closed = True; return None
            if op == 0x9: self._frame(0xA, d); continue
            if op == 0xA: continue
            buf += d
            if b0 & 0x80: return buf.decode("utf-8", "replace")
