#!/bin/bash
# ============================================================================
# CITA Environment Setup Script
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

# 0. Install LINUX system dependencies
echo ""
echo "[0/5] Installing system dependencies..."
sudo apt update && sudo apt install -y \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-bibtex-extra \
    texlive-science \
    biber \
    tree

# 1. Install Python 3.12
echo ""
echo "[1/6] Installing Python 3.12..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# 2. Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
python3.12 -m venv venv_CITA
source venv_CITA/bin/activate

# 3. Install PyTorch 2.5.1 with CUDA 12.4
echo ""
echo "[3/6] Installing PyTorch 2.5.1+cu124..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Verify PyTorch installation
echo ""
echo "[4/6] Verifying PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"

# 5. Install requirements
echo ""
echo "[5/6] Installing requirements.txt..."
pip install -r requirements.txt

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
echo "  source venv_CITA/bin/activate"
echo ""
echo "To run training:"
echo "  python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity --use-instruction false"
echo ""
