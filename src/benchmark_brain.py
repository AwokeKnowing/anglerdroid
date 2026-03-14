#!/usr/bin/env python3
"""
benchmark_brain.py — Verify 3090 can run the full stack at 1 FPS.

Sends synthetic 640x480 JPEG frames (prev + current) to vLLM,
measures time-to-first-token and total response time. Also benchmarks Kokoro TTS.

Usage:
  # Make sure vLLM is running first:
  #   vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000 \
  #     --max-model-len 4096 --enforce-eager --gpu-memory-utilization 0.5 \
  #     --limit-mm-per-prompt '{"image":2,"video":0}'

  python benchmark_brain.py
  python benchmark_brain.py --model Qwen/Qwen2.5-VL-3B-Instruct
  python benchmark_brain.py --no-vision  # text-only, skip images
"""

import argparse
import base64
import json
import time
import numpy as np
from urllib.request import Request, urlopen
from urllib.error import HTTPError

VLLM_URL = "http://localhost:8000/v1/chat/completions"
BUDGET_MS = 1000  # must complete within 1 second (1 FPS)
N_ROUNDS = 20
WINDOW = 4

SYSTEM_PROMPT = (
    "You are a robot. Each frame you receive a camera image and robot state. "
    "Respond with ONLY a function call like twist_for(0, 0) or a dot. "
    "No other text."
)


def make_frame():
    """Generate a random 640x480 JPEG, return base64 string."""
    import cv2
    img = np.random.randint(40, 200, (480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "BENCHMARK", (180, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf.tobytes()).decode('ascii')


def build_messages(conversation, system_prompt):
    """Build messages list, keeping images only in the last user turn."""
    img_indices = [i for i, m in enumerate(conversation)
                   if isinstance(m.get("content"), list)
                   and any(p.get("type") == "image_url" for p in m["content"])]
    strip = set(img_indices[:-1])

    msgs = [{"role": "system", "content": system_prompt}]
    for i, msg in enumerate(conversation):
        if i in strip:
            text_only = [p for p in msg["content"] if p.get("type") != "image_url"]
            if not text_only:
                text_only = [{"type": "text", "text": "(frame)"}]
            msgs.append({"role": msg["role"], "content": text_only})
        else:
            msgs.append(msg)
    return msgs


def call_vllm(url, messages, model, stream=False):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 60,
        "temperature": 0.3,
        "stream": stream,
    }
    req = Request(url,
                  data=json.dumps(body).encode("utf-8"),
                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urlopen(req, timeout=30)

    if stream:
        first_token_ms = None
        chunks = []
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                if first_token_ms is None:
                    first_token_ms = (time.time() - t0) * 1000
                d = json.loads(line[6:])
                delta = d["choices"][0].get("delta", {})
                if "content" in delta:
                    chunks.append(delta["content"])
        total_ms = (time.time() - t0) * 1000
        return "".join(chunks).strip(), first_token_ms or total_ms, total_ms
    else:
        data = json.loads(resp.read())
        total_ms = (time.time() - t0) * 1000
        text = data["choices"][0]["message"]["content"].strip()
        return text, total_ms, total_ms


def benchmark_tts():
    try:
        import os
        from brain_server import _init_kokoro, _kokoro, _HAS_KOKORO
        _init_kokoro()
        from brain_server import _kokoro, _HAS_KOKORO
        if not _HAS_KOKORO:
            print("\n  TTS: kokoro not available, skipping")
            return
        times = []
        for i in range(5):
            t0 = time.time()
            samples, sr = _kokoro.create("Hello, I can see the kitchen ahead.",
                                         voice="af_heart", speed=1.0)
            times.append((time.time() - t0) * 1000)
        avg = sum(times) / len(times)
        print("\n  TTS (Kokoro): %.0f ms avg over %d runs (%.1fs audio)" % (
            avg, len(times), len(samples) / sr))
    except Exception as e:
        print("\n  TTS: error — %s" % e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--vllm-url", default=VLLM_URL)
    args = ap.parse_args()

    vision = not args.no_vision
    vllm_url = args.vllm_url

    print("=" * 60)
    print("AnglerDroid Brain Benchmark")
    print("  model:   %s" % args.model)
    print("  vision:  %s" % vision)
    print("  images:  2 per turn (prev + current)")
    print("  rounds:  %d" % args.rounds)
    print("  budget:  %d ms (1 FPS)" % BUDGET_MS)
    print("=" * 60)

    # pre-generate frames (need rounds+1 to always have a prev frame)
    n_frames = args.rounds + 1
    print("\nGenerating %d test frames..." % n_frames)
    frames = [make_frame() for _ in range(n_frames)]

    # check vLLM is up
    print("Checking vLLM at %s ..." % vllm_url)
    try:
        resp = urlopen(Request(vllm_url.rsplit("/", 3)[0] + "/v1/models"), timeout=5)
        models = json.loads(resp.read())
        names = [m["id"] for m in models.get("data", [])]
        print("  models loaded: %s" % names)
    except Exception as e:
        print("  ERROR: can't reach vLLM — %s" % e)
        return

    conversation = []
    ttft_list = []
    total_list = []
    on_time = 0

    print("\nRunning %d rounds...\n" % args.rounds)
    for i in range(args.rounds):
        prev_b64 = frames[i]
        curr_b64 = frames[i + 1]

        if vision:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + prev_b64}},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + curr_b64}},
                    {"type": "text",
                     "text": "frame: %d\nwz: 0.000 0.000\nSPEECH:\n" % i},
                ],
            }
        else:
            user_msg = {
                "role": "user",
                "content": "frame: %d\nwz: 0.000 0.000\nSPEECH:\n" % i,
            }

        conversation.append(user_msg)

        # keep sliding window: last WINDOW user/assistant pairs
        if len(conversation) > WINDOW * 2:
            conversation = conversation[-(WINDOW * 2):]

        messages = build_messages(conversation, SYSTEM_PROMPT)

        try:
            text, ttft, total = call_vllm(vllm_url, messages, args.model, stream=True)
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            print("  round %2d: HTTP %d — %s" % (i + 1, e.code, body))
            conversation.append({"role": "assistant", "content": "."})
            continue
        except Exception as e:
            print("  round %2d: ERROR — %s" % (i + 1, e))
            conversation.append({"role": "assistant", "content": "."})
            continue

        conversation.append({"role": "assistant", "content": text})

        ttft_list.append(ttft)
        total_list.append(total)
        ok = total <= BUDGET_MS
        if ok:
            on_time += 1

        status = "OK" if ok else "SLOW"
        print("  round %2d: ttft=%4.0fms  total=%4.0fms  [%s]  %s" % (
            i + 1, ttft, total, status, text[:60]))

    # summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if ttft_list:
        print("  TTFT:     avg=%4.0fms  p50=%4.0fms  p95=%4.0fms  max=%4.0fms" % (
            np.mean(ttft_list), np.median(ttft_list),
            np.percentile(ttft_list, 95), np.max(ttft_list)))
        print("  Total:    avg=%4.0fms  p50=%4.0fms  p95=%4.0fms  max=%4.0fms" % (
            np.mean(total_list), np.median(total_list),
            np.percentile(total_list, 95), np.max(total_list)))
        print("  On-time:  %d/%d (%.0f%%)" % (
            on_time, len(total_list), on_time / len(total_list) * 100))
        print("  Verdict:  %s" % (
            "PASS — can sustain 1 FPS" if on_time >= len(total_list) * 0.9
            else "FAIL — too slow for 1s budget"))
    else:
        print("  No successful rounds.")

    benchmark_tts()
    print("")


if __name__ == "__main__":
    main()
