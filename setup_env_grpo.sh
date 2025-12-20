#!/bin/bash
# ============================================================================
# GRPO Environment Setup Script
# ============================================================================
# Tested configuration:
#   - Python: 3.12
#   - PyTorch: 2.5.1+cu124
#   - CUDA: 12.4
#   - Flash-Attn: 2.8.3
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "CITA Environment Setup"
echo "============================================"

# 1. Create virtual environment
echo ""
echo "[1/6] Creating virtual environment..."
python3.12 -m venv venv_GRPO
source venv_GRPO/bin/activate

# 2. Install PyTorch 2.5.1 with CUDA 12.4
echo ""
echo "[2/6] Installing PyTorch 2.5.1+cu124..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Verify PyTorch installation
echo ""
echo "[3/6] Verifying PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"

# 4. Install requirements
echo ""
echo "[4/6] Installing requirements_grpo.txt..."
pip install -r requirements_grpo.txt

# 5. Install TRL with --no-deps (avoid transformers version conflict)
echo ""
echo "[5/6] Installing TRL 0.22.2 (--no-deps for GRPO support)..."
pip install --no-deps trl==0.22.2

# 6. Install Flash-Attention (pre-built wheel for torch2.5+cu12+cp312)
echo ""
echo "[6/6] Installing Flash-Attention 2.8.3..."
WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

curl -L -o "$WHEEL_NAME" "$WHEEL_URL"
pip install "$WHEEL_NAME"
rm -f "$WHEEL_NAME"

# Final verification
echo ""
echo "============================================"
echo "VERIFICATION"
echo "============================================"
python -c "
import torch
import flash_attn
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA:       {torch.version.cuda}')
print(f'GPU:        {torch.cuda.is_available()}')
print(f'Flash-Attn: {flash_attn.__version__}')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To activate environment:"
echo "  source venv_GRPO/bin/activate"
echo ""
echo "To run GRPO training:"
echo "  python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode micro --use-instruction false"
echo ""
