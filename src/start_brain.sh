#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$HOME/angler-brain-venv"
source "$VENV_DIR/bin/activate"


MODEL="Qwen/Qwen3.5-9B"
QUANT="--quantization w8a8_int8"
SGLANG_PORT=30000
ZMQ_PORT=5555
EXTRA_SGLANG=""
NO_STT=""

for arg in "$@"; do
    case $arg in
        --bf16)     QUANT="" ;;
        --int4)     QUANT="--quantization awq_marlin" ;;
        --int8)     QUANT="--quantization w8a8_int8" ;;
        --9b)       MODEL="Qwen/Qwen3.5-9B" ;;
        --4b)       MODEL="Qwen/Qwen3.5-4B" ;;
        --4b-awq)   MODEL="QuantTrio/Qwen3.5-4B-AWQ"; QUANT="" ;;
        --7b)       MODEL="Qwen/Qwen2.5-VL-7B-Instruct" ;;
        --3b)       MODEL="Qwen/Qwen2.5-VL-3B-Instruct" ;;
        --no-stt)   NO_STT="--no-stt" ;;
        *)          EXTRA_SGLANG="$EXTRA_SGLANG $arg" ;;
    esac
done

echo "============================================"
echo " AnglerDroid Brain"
echo "  model:  $MODEL"
echo "  quant:  ${QUANT:-none (pre-quant or bf16)}"
echo "  sglang: localhost:$SGLANG_PORT"
echo "  zmq:    0.0.0.0:$ZMQ_PORT"
echo "============================================"

# Start SGLang in background
echo "Starting SGLang server ..."
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --port $SGLANG_PORT \
    --host 127.0.0.1 \
    --max-running-requests 1 \
    --mem-fraction-static 0.80 \
    --enable-multimodal \
    --chat-template "$SCRIPT_DIR/qwen_nonthinking.jinja" \
    $QUANT \
    $EXTRA_SGLANG &

SGLANG_PID=$!

cleanup() {
    echo "Shutting down ..."
    kill $SGLANG_PID 2>/dev/null || true
    wait $SGLANG_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for SGLang to be ready
echo "Waiting for SGLang to load model ..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$SGLANG_PORT/health" > /dev/null 2>&1; then
        echo "SGLang ready."
        break
    fi
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "SGLang process died. Check logs above."
        exit 1
    fi
    sleep 2
done

# Start brain ZMQ bridge
echo "Starting brain ZMQ bridge ..."
python3 "$SCRIPT_DIR/brain_zmq.py" \
    --port $ZMQ_PORT \
    --sglang-url "http://127.0.0.1:$SGLANG_PORT" \
    --model "$MODEL" \
    $NO_STT
