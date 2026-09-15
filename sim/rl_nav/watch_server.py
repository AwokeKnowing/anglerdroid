#!/usr/bin/env python3
"""Serve rl_nav watch page (mp4 + stats). No train work."""
from __future__ import annotations

import argparse
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
HTML = ROOT / "watch.html"
PORT = 8765


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(OUT), **k)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html") and HTML.is_file():
            body = HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if self.path.endswith(".mp4") or self.path.endswith(".json"):
            return
        super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if HTML.is_file():
        shutil.copy2(HTML, OUT / "index.html")
    httpd = ThreadingHTTPServer(("0.0.0.0", int(args.port)), Handler)
    print("rl_nav watch http://127.0.0.1:%d/" % args.port, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
