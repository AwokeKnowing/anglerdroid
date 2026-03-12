"""
reka_server.py – VLM-based local planner using Reka Edge.
Runs on a desktop GPU (e.g. 3090). Robot sends atlas frames over HTTP,
gets back movement actions.

Approach: Discrete action classification (inspired by PIVOT, DeepMind 2024).
VLMs are good at *choosing* among options, not generating precise numbers.
We define a fixed set of named actions and ask the model to pick one.
Action history is included so the model knows what it just did.

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
ACTIONS_PER_BATCH = 2

# ── Discrete action table ──────────────────────────────────────
# Each action has a name, forward speed (m/s), and turn rate (rad/s).
# The VLM picks from these by number. This is a classification problem,
# which VLMs handle far better than free-form number generation.

ACTION_TABLE = [
    ("FORWARD",       0.20,  0.0),
    ("FORWARD_LEFT",  0.15,  0.35),
    ("FORWARD_RIGHT", 0.15, -0.35),
    ("LEFT",          0.0,   0.5),
    ("RIGHT",         0.0,  -0.5),
    ("SLIGHT_LEFT",   0.20,  0.15),
    ("SLIGHT_RIGHT",  0.20, -0.15),
    ("BACKWARD",     -0.15,  0.0),
    ("STOP",          0.0,   0.0),
]

_action_by_name = {a[0]: a for a in ACTION_TABLE}

def _action_list_str():
    lines = []
    for i, (name, fwd, turn) in enumerate(ACTION_TABLE):
        lines.append("%d. %s" % (i + 1, name))
    return "\n".join(lines)

VLA_PROMPT = """You are the local navigation planner for a wheeled robot.

THE IMAGE shows 4 camera views arranged in a 2x2 grid:
- TOP-LEFT: forward-facing RGB camera (main view of the room)
- BOTTOM-LEFT: forward-facing depth camera (closer obstacles appear larger/brighter)
- TOP-RIGHT: top-down overhead view (robot faces RIGHT in this view)
- BOTTOM-RIGHT: obstacle map (robot faces RIGHT, colored=obstacles, black=clear)

AVAILABLE ACTIONS (pick ONE number):
%s

RULES:
- Pick the single best action number for the current situation.
- Navigate AROUND obstacles, not into them.
- Use depth camera (bottom-left) and obstacle map (bottom-right) to judge proximity.
- If the path ahead is clear, prefer FORWARD or SLIGHT turns.
- If obstacle is close on one side, turn AWAY from it.
- Only pick STOP if you have arrived at the goal or are about to collide head-on.
- Only pick BACKWARD if completely stuck.

Reply with ONLY the action number (1-9), nothing else.""" % _action_list_str()

# Server-side action history (last N actions chosen)
HISTORY_LEN = 4
_action_history = []  # list of action name strings
_current_instruction = ""


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
    global _action_history, _current_instruction
    import torch

    if instruction != _current_instruction:
        _action_history = []
        _current_instruction = instruction

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.write(fd, image_bytes)
        os.close(fd)

        # Build prompt with action history for temporal context
        prompt_parts = [VLA_PROMPT, "", "Task: " + instruction]
        if _action_history:
            recent = _action_history[-HISTORY_LEN:]
            prompt_parts.append("Your recent actions were: " + ", ".join(recent))
            prompt_parts.append("Consider whether these actions are working or if you need to adjust.")

        full_prompt = "\n".join(prompt_parts)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": tmp_path},
            {"type": "text", "text": full_prompt},
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
                **inputs, max_new_tokens=10, do_sample=False,
                eos_token_id=[_processor.tokenizer.eos_token_id, sep_id],
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = _processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.replace("<sep>", "").strip()
        print("MODEL RAW: %s" % text[:200])

        actions, chosen_name = _parse_choice(text)
        _action_history.append(chosen_name)
        if len(_action_history) > HISTORY_LEN:
            _action_history = _action_history[-HISTORY_LEN:]

        return actions, "%s -> %s" % (text, chosen_name)
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
        name = "FORWARD"
    elif "left" in inst:
        name = "FORWARD_LEFT"
    elif "right" in inst:
        name = "FORWARD_RIGHT"
    elif "back" in inst:
        name = "BACKWARD"
    elif "spin" in inst or "turn around" in inst:
        name = "LEFT"
    elif "stop" in inst:
        name = "STOP"
    else:
        name = "FORWARD"
    a = _action_by_name[name]
    action = {"forward_mps": a[1], "angular_rads": a[2]}
    return [action.copy() for _ in range(ACTIONS_PER_BATCH)]


def _parse_choice(text):
    """Parse model output as an action number (1-9). Returns (actions_list, chosen_name)."""
    # Extract any digit from the response
    digits = re.findall(r"\d+", text)
    for d in digits:
        idx = int(d) - 1  # 1-indexed to 0-indexed
        if 0 <= idx < len(ACTION_TABLE):
            name, fwd, turn = ACTION_TABLE[idx]
            action = {"forward_mps": fwd, "angular_rads": turn}
            return [action.copy() for _ in range(ACTIONS_PER_BATCH)], name

    # Try matching action name directly
    upper = text.upper().replace(" ", "_")
    for name, fwd, turn in ACTION_TABLE:
        if name in upper:
            action = {"forward_mps": fwd, "angular_rads": turn}
            return [action.copy() for _ in range(ACTIONS_PER_BATCH)], name

    print("WARNING: could not parse choice from: %s — defaulting to STOP" % text[:300])
    action = {"forward_mps": 0.0, "angular_rads": 0.0}
    return [action.copy() for _ in range(ACTIONS_PER_BATCH)], "STOP"


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
        print("#%d  %d actions  %.0f ms  | %s  ->  %s" % (
            _request_count, len(actions), dt * 1000, instruction[:40],
            raw_text[:60] if raw_text else "?"))

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
    print("Action table: %s" % [a[0] for a in ACTION_TABLE])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()
