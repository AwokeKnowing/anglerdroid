#!/usr/bin/env bash
set -e

VENV=~/angler-brain-venv

echo "=== AnglerDroid Brain Server Setup (3090) ==="
echo ""

# system deps
if ! command -v python3 &>/dev/null; then
    echo "Installing python3..."
    sudo apt-get update && sudo apt-get install -y python3 python3-pip
fi

if ! python3 -c "import venv" 2>/dev/null; then
    echo "Installing python3-venv..."
    sudo apt-get update && sudo apt-get install -y python3-venv
fi

# check GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. You need NVIDIA drivers installed."
    echo "  On Ubuntu:  sudo apt install nvidia-driver-560"
    echo "  On cloud:   usually pre-installed"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# venv
echo "Creating venv at $VENV ..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

# pytorch
echo "Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# brain deps (vllm, faster-whisper, numpy)
echo "Installing brain server deps..."
pip install -r requirements-brain.txt

echo ""
echo "=========================================="
echo " Setup complete."
echo "=========================================="
echo ""
echo " To run:"
echo ""
echo "   source $VENV/bin/activate"
echo ""
echo "   # Terminal 1 — vLLM (downloads model on first run):"
echo "   vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \\"
echo "     --max-model-len 4096 --enforce-eager --gpu-memory-utilization 0.5"
echo ""
echo "   # Terminal 2 — brain server:"
echo "   python brain_server.py --port 8090"
echo ""
echo "   # On Jetson:"
echo "   python main.py --brain-url http://<THIS_PC_IP>:8090"
echo ""
