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

VLA_PROMPT = """You control a wheeled robot indoors. The image is a 2x2 camera grid:
- Top-left: forward RGB camera (main view, use for room layout and target)
- Bottom-left: forward depth camera RGB (obstacle avoidance, close objects appear large)
- Top-right: top-down camera RGB (robot faces RIGHT in this view)
- Bottom-right: obstacle map (robot faces RIGHT, black=clear, colored dots=obstacles)

You are the local planner. Navigate around obstacles to reach the goal. You get fresh images every few seconds.

Output 8 action pairs as: forward_speed,turn_speed separated by semicolons.
Each pair runs for 0.3 seconds. forward_speed: -0.3 to 0.3 (positive=forward, <0.05 won't move). turn_speed: -1.0 to 1.0 (positive=left).

Drive forward: 0.2,0;0.2,0;0.2,0;0.2,0;0.2,0;0.2,0;0.2,0;0.2,0
Turn left: 0,0.5;0,0.5;0,0.5;0,0.5;0,0.5;0,0.5;0,0.5;0,0.5
Arc right: 0.15,-0.3;0.15,-0.3;0.15,-0.3;0.15,-0.3;0.15,-0.3;0.15,-0.3;0.15,-0.3;0.15,-0.3
Stop: 0,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0

Steer around obstacles. Only stop if about to collide or arrived.

Output ONLY the 8 pairs. No text, no brackets, no labels."""


def load_model(device_str="cuda", quantize=True):
    global _model, _processor, _device
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = "RekaAI/reka-edge-2603"
    _device = torch.device(device_str)

    print("Loading %s on %s (quantize=%s) …" % (model_id, device_str, quantize))
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                               use_fast=False)

    load_kwargs = {"trust_remote_code": True}
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    _model = AutoModelForImageTextToText.from_pretrained(
        model_id, **load_kwargs
    ).eval()

    if not quantize:
        _model = _model.to(_device)

    param_bytes = sum(p.numel() * p.element_size() for p in _model.parameters())
    print("Model ready (%.1f GB)" % (param_bytes / 1e9))


def infer(image_bytes, instruction):
    import torch

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.write(fd, image_bytes)
        os.close(fd)

        prompt = VLA_PROMPT + "\n\nInstruction: " + instruction
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
                **inputs, max_new_tokens=150, do_sample=False,
                eos_token_id=[_processor.tokenizer.eos_token_id, sep_id],
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = _processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.replace("<sep>", "").strip()
        print("MODEL RAW: %s" % text[:500])
        return _parse_actions(text), text
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
        return [{"forward_mps": 0.15, "angular_rads": 0.0}] * 8
    if "left" in inst:
        return [{"forward_mps": 0.10, "angular_rads": 0.4}] * 8
    if "right" in inst:
        return [{"forward_mps": 0.10, "angular_rads": -0.4}] * 8
    if "back" in inst:
        return [{"forward_mps": -0.15, "angular_rads": 0.0}] * 8
    if "spin" in inst or "turn around" in inst:
        return [{"forward_mps": 0.0, "angular_rads": 0.6}] * 8
    if "stop" in inst:
        return [{"forward_mps": 0.0, "angular_rads": 0.0}] * 8
    return [{"forward_mps": 0.12, "angular_rads": 0.0}] * 8


def _parse_actions(text):
    """Parse 'f,a;f,a;...' pairs, falling back to JSON array, then zeros."""
    # Try compact format: 0.2,0.0;0.15,-0.3;...
    pairs = re.findall(r"(-?[\d.]+)\s*,\s*(-?[\d.]+)", text)
    if len(pairs) >= 2:
        out = []
        for fwd_s, ang_s in pairs:
            try:
                out.append({
                    "forward_mps": max(-0.3, min(0.3, float(fwd_s))),
                    "angular_rads": max(-1.0, min(1.0, float(ang_s))),
                })
            except ValueError:
                continue
        if out:
            return out

    # Fallback: try JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            raw = json.loads(match.group())
            out = []
            for a in raw:
                if isinstance(a, dict):
                    out.append({
                        "forward_mps": max(-0.3, min(0.3, float(a.get("forward_mps", 0)))),
                        "angular_rads": max(-1.0, min(1.0, float(a.get("angular_rads", 0)))),
                    })
            if out:
                return out
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass

    print("WARNING: could not parse actions from: %s" % text[:300])
    return [{"forward_mps": 0.0, "angular_rads": 0.0}] * 8


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
        out = self.infer_fn(image_bytes, instruction)
        dt = time.time() - t0
        _request_count += 1

        if isinstance(out, tuple):
            actions, raw_text = out
        else:
            actions, raw_text = out, ""

        result = {"actions": actions, "inference_ms": round(dt * 1000, 1),
                  "raw_output": raw_text[:500]}
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
    ap.add_argument("--no-quantize", action="store_true",
                    help="Disable 4-bit quantization (uses more VRAM, not faster)")
    args = ap.parse_args()

    if args.mock:
        print("=== MOCK MODE (no model) ===")
        Handler.infer_fn = staticmethod(mock_infer)
    else:
        load_model(args.device, quantize=not args.no_quantize)
        Handler.infer_fn = staticmethod(infer)

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print("Reka VLA server on :%d   (Ctrl+C to quit)" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()
