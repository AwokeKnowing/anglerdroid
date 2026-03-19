#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$HOME/angler-brain-venv"
source "$VENV_DIR/bin/activate"

MODEL="Qwen/Qwen3.5-4B"
QUANT=""
VLLM_PORT=8000
ZMQ_PORT=5555
EXTRA_VLLM=""
NO_STT=""

for arg in "$@"; do
    case $arg in
        --9b)       MODEL="Qwen/Qwen3.5-9B" ;;
        --4b)       MODEL="Qwen/Qwen3.5-4B" ;;
        --4b-awq)   MODEL="QuantTrio/Qwen3.5-4B-AWQ" ;;
        --7b)       MODEL="Qwen/Qwen2.5-VL-7B-Instruct" ;;
        --3b)       MODEL="Qwen/Qwen2.5-VL-3B-Instruct" ;;
        --int8)     QUANT="--quantization compressed-tensors" ;;
        --bf16)     QUANT="" ;;
        --no-stt)   NO_STT="--no-stt" ;;
        *)          EXTRA_VLLM="$EXTRA_VLLM $arg" ;;
    esac
done

echo "============================================"
echo " AnglerDroid Brain"
echo "  model:  $MODEL"
echo "  quant:  ${QUANT:-none (bf16)}"
echo "  vllm:   localhost:$VLLM_PORT"
echo "  zmq:    0.0.0.0:$ZMQ_PORT"
echo "============================================"

CACHE_DIR="$HOME/.cache/vllm-compile"
mkdir -p "$CACHE_DIR"

echo "Starting vLLM server ..."
FLASHINFER_DISABLE_VERSION_CHECK=1 \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor" \
vllm serve "$MODEL" \
    --compilation-config '{"cache_dir":"'"$CACHE_DIR"'"}' \
    --port $VLLM_PORT \
    --host 127.0.0.1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image":1}' \
    --max-num-seqs 1 \
    $QUANT \
    $EXTRA_VLLM &

VLLM_PID=$!

cleanup() {
    echo "Shutting down ..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for vLLM to load model ..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM process died. Check logs above."
        exit 1
    fi
    sleep 2
done

echo "Starting brain ZMQ bridge ..."
python3 "$SCRIPT_DIR/brain_zmq.py" \
    --port $ZMQ_PORT \
    --sglang-url "http://127.0.0.1:$VLLM_PORT" \
    --model "$MODEL" \
    $NO_STT
