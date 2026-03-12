"""
reka_server.py – Standalone VLA inference server for Reka Edge.
Runs on a desktop GPU (e.g. 3090). Robot sends atlas frames over HTTP,
gets back a batch of twist actions.

Usage:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements-reka-server.txt
    python reka_server.py --port 8090                    # real model on CUDA
    python reka_server.py --port 8090 --mock             # test without GPU
"""

import argparse
import base64
import json
import os
import re
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

_model = None
_processor = None
_device = None
_request_count = 0

VLA_PROMPT = """You are the navigation controller for a wheeled robot.
You receive a camera image from the robot – a 2x2 grid:
- Top-left: forward-facing RGB camera
- Top-right: depth camera 1 (top-down view, RGB)
- Bottom-left: depth camera 2 (forward view, RGB)
- Bottom-right: obstacle map (green=obstacles from above, red=obstacles ahead, dim blue=unknown areas, black=clear path)

Output exactly 16 navigation steps as a JSON array.
Each step executes for 333 ms and has two fields:
  forward_mps  – forward speed, range -0.3 … 0.3 m/s (positive = forward)
  angular_rads – turn rate, range -1.0 … 1.0 rad/s (positive = turn left)

Rules:
- Black areas in the obstacle map are safe to drive through.
- Coloured areas (green/red/yellow) are obstacles – avoid them.
- Dim-blue areas are unknown – avoid if possible.
- Keep speeds conservative (0.10 – 0.15 m/s).
- If the path is blocked or unclear, output all zeros to stop.

Respond with ONLY the JSON array. No markdown fences, no explanation."""


def load_model(device_str="cuda"):
    global _model, _processor, _device
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = "RekaAI/reka-edge-2603"
    _device = torch.device(device_str)

    print("Loading %s on %s …" % (model_id, device_str))
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16,
    ).eval().to(_device)
    print("Model ready (%.1f GB VRAM)" % (
        sum(p.numel() * p.element_size() for p in _model.parameters()) / 1e9))


def infer(image_bytes, instruction):
    import torch

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.write(fd, image_bytes)
        os.close(fd)

        prompt = VLA_PROMPT + "\n\nCurrent instruction: " + instruction
        messages = [{"role": "user", "content": [
            {"type": "image", "image": tmp_path},
            {"type": "text", "text": prompt},
        ]}]

        inputs = _processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                if val.is_floating_point():
                    inputs[key] = val.to(device=_device, dtype=torch.float16)
                else:
                    inputs[key] = val.to(device=_device)

        with torch.inference_mode():
            sep_id = _processor.tokenizer.convert_tokens_to_ids("<sep>")
            output_ids = _model.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
                eos_token_id=[_processor.tokenizer.eos_token_id, sep_id],
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = _processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.replace("<sep>", "").strip()
        return _parse_actions(text)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def mock_infer(image_bytes, instruction):
    """Keyword-based mock for testing without the model."""
    time.sleep(0.3)
    inst = instruction.lower()
    if "forward" in inst or "straight" in inst or "ahead" in inst:
        return [{"forward_mps": 0.15, "angular_rads": 0.0}] * 16
    if "left" in inst:
        return [{"forward_mps": 0.10, "angular_rads": 0.3}] * 16
    if "right" in inst:
        return [{"forward_mps": 0.10, "angular_rads": -0.3}] * 16
    if "back" in inst:
        return [{"forward_mps": -0.15, "angular_rads": 0.0}] * 16
    if "spin" in inst or "turn around" in inst:
        return [{"forward_mps": 0.0, "angular_rads": 0.8}] * 16
    if "stop" in inst:
        return [{"forward_mps": 0.0, "angular_rads": 0.0}] * 16
    return [{"forward_mps": 0.10, "angular_rads": 0.0}] * 16


def _parse_actions(text):
    """Extract a JSON array of actions from model output, with safe fallback."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            raw = json.loads(match.group())
            out = []
            for a in raw:
                out.append({
                    "forward_mps": max(-0.3, min(0.3, float(a.get("forward_mps", 0)))),
                    "angular_rads": max(-1.0, min(1.0, float(a.get("angular_rads", 0)))),
                })
            if out:
                return out
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass
    print("WARNING: could not parse actions from: %s" % text[:300])
    return [{"forward_mps": 0.0, "angular_rads": 0.0}] * 16


# ── HTTP handler ──────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    infer_fn = None

    def do_POST(self):
        global _request_count
        if self.path != "/infer":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Bad JSON")
            return

        image_b64 = data.get("image", "")
        instruction = data.get("instruction", "")
        if not image_b64 or not instruction:
            self.send_error(400, "Missing 'image' or 'instruction'")
            return

        image_bytes = base64.b64decode(image_b64)

        t0 = time.time()
        actions = self.infer_fn(image_bytes, instruction)
        dt = time.time() - t0
        _request_count += 1

        result = {"actions": actions, "inference_ms": round(dt * 1000, 1)}
        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        print("#%d  %d actions  %.0f ms  | %s" % (
            _request_count, len(actions), dt * 1000, instruction[:60]))

    def do_GET(self):
        if self.path == "/health":
            self._json_ok({"status": "ok", "requests": _request_count})
            return
        self.send_error(404)

    def _json_ok(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="Reka Edge VLA inference server")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--mock", action="store_true", help="Mock inference (no model)")
    ap.add_argument("--device", default="cuda", help="torch device")
    args = ap.parse_args()

    if args.mock:
        print("=== MOCK MODE (no model) ===")
        Handler.infer_fn = staticmethod(mock_infer)
    else:
        load_model(args.device)
        Handler.infer_fn = staticmethod(infer)

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print("Reka VLA server on :%d   (Ctrl+C to quit)" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()
