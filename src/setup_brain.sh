#!/bin/bash
set -e

# AnglerDroid Brain Setup — SGLang + Qwen3.5-9B
#
# Run on the 3090 machine.
# After setup, start with:
#   ./start_brain.sh            (default: Qwen3.5-9B, INT8)
#   ./start_brain.sh --bf16     (full precision, needs ~18GB VRAM)
#   ./start_brain.sh --7b       (fallback to Qwen2.5-VL-7B)

VENV_DIR="$HOME/angler-brain-venv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " AnglerDroid Brain Setup (SGLang + Qwen3.5)"
echo "============================================"

# 1. Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 2. Upgrade pip + install uv for fast installs
pip install --upgrade pip
pip install uv

# 3. Install SGLang (includes torch, triton, etc.)
echo "Installing SGLang ..."
uv pip install "sglang[all]"

# 4. Install CUDA toolkit for FlashInfer JIT compilation
echo "Installing CUDA nvcc for FlashInfer ..."
uv pip install nvidia-cuda-nvcc-cu12

# 5. Install brain dependencies (ZMQ, STT, TTS, etc.)
echo "Installing brain dependencies ..."
uv pip install -r "$SCRIPT_DIR/requirements-brain.txt"

# 5. Pre-download the model (so first start is fast)
echo "Pre-downloading Qwen3.5-9B ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B', ignore_patterns=['*.bin'])
print('Model cached.')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Activate:  source $VENV_DIR/bin/activate"
echo " Start:     cd $SCRIPT_DIR && ./start_brain.sh"
echo "============================================"
