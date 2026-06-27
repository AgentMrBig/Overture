#!/bin/bash
# Run this on a fresh Thunder instance to restore everything
# Usage: bash setup_thunder.sh

set -e

echo "=== Overture Thunder Setup ==="

# Clone repo if not already here
if [ ! -d "Overture" ]; then
    git clone https://github.com/AgentMrBig/Overture.git
fi
cd Overture

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Install all dependencies
pip install -r requirements.txt -q

echo ""
echo "=== Setup complete ==="
echo "Start server : python3 overture_server.py"
echo "Run training : python3 contrastive_pretrain_v3.py"
echo ""
echo "NOTE: Qwen3-235B-A22B will download on first run (~117GB)"
echo "NOTE: Upload your tick data to ~/Overture/data/"
