#!/bin/bash
# ============================================================================
# Environment Setup Script - GRPO Training (UV-based)
# ============================================================================
# Separate venv because GRPO needs trl==0.22.2 + transformers==4.56.2
# (CITA/DPO/PPO use trl==0.11.4 + transformers==4.51.0)
#
# Usage:
#   chmod +x setup_env_grpo.sh
#
#   # GPU Server (GRPO training)
#   ./setup_env_grpo.sh --gpu 2>&1 | tee logs/setup_env_grpo.log
#
#   # Activate
#   source venv_GRPO/bin/activate
# ============================================================================

set -e  # Exit on error
mkdir -p logs

# ============================================================================
# Pinned versions (A100 + CUDA 12.4 + Python 3.12)
# ============================================================================
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"

# ============================================================================
# Parse flags
# ============================================================================
if [ -z "$1" ]; then
    echo "Error: No flag provided"
    echo ""
    echo "Usage:"
    echo "  ./setup_env_grpo.sh --gpu   # GPU Server (Nvidia ONLY)"
    exit 1
fi

# Detect OS
OS="$(uname -s)"

# ============================================================================
# --gpu: GPU Server setup (GRPO training) - Nvidia ONLY
# ============================================================================
if [ "$1" = "--gpu" ]; then
    if [ "$OS" != "Linux" ]; then
        echo "Error: --gpu requires Linux. Detected: $OS"
        exit 1
    fi

    echo "============================================"
    echo "GRPO Environment Setup (UV)"
    echo "============================================"
    echo "Detected OS: $OS"
    echo ""

    # Install system dependencies
    NEED_APT_UPDATE=false
    APT_PACKAGES=""

    if ! command -v tree &> /dev/null; then
        APT_PACKAGES="$APT_PACKAGES tree"
        NEED_APT_UPDATE=true
    fi

    for pkg in jq htop tmux wget curl; do
        if ! command -v "$pkg" &> /dev/null; then
            APT_PACKAGES="$APT_PACKAGES $pkg"
            NEED_APT_UPDATE=true
        fi
    done

    if ! command -v python3.12 &> /dev/null; then
        echo "Adding deadsnakes PPA for Python 3.12..."
        apt-get update
        apt-get install -y software-properties-common
        add-apt-repository -y ppa:deadsnakes/ppa
        APT_PACKAGES="$APT_PACKAGES python3.12 python3.12-venv python3.12-dev"
        NEED_APT_UPDATE=true
    fi

    if [ "$NEED_APT_UPDATE" = true ]; then
        echo "Installing: $APT_PACKAGES"
        apt-get update
        apt-get install -y $APT_PACKAGES
    fi
    echo "Python: $(python3.12 --version)"

    # Install UV if not available
    if ! command -v uv &> /dev/null; then
        echo ""
        echo "Installing UV package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo "UV: $(uv --version)"

    # Create virtual environment
    if [ ! -d "venv_GRPO" ]; then
        echo ""
        echo "Creating virtual environment (Python 3.12)..."
        uv venv --python 3.12 venv_GRPO
    else
        echo "Virtual environment already exists (venv_GRPO)"
    fi

    source venv_GRPO/bin/activate

    # 1. Install PyTorch (CUDA 12.4)
    echo ""
    echo "[1/5] Installing PyTorch ${TORCH_VERSION}+cu124..."
    uv pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${CUDA_INDEX_URL}"

    # 2. Verify PyTorch + CUDA
    echo ""
    echo "[2/5] Verifying PyTorch + CUDA..."
    python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. Nvidia GPU required.')
    exit(1)
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
"

    # 3. Install GRPO requirements
    echo ""
    echo "[3/5] Installing GRPO requirements (UV)..."
    uv pip install -r requirements_grpo.txt

    # 4. Install TRL 0.22.2 with --no-deps (avoid transformers version conflict)
    echo ""
    echo "[4/5] Installing TRL 0.22.2 (--no-deps for GRPO support)..."
    uv pip install --no-deps trl==0.22.2

    # 5. Install Flash-Attention 2
    echo ""
    echo "[5/5] Installing Flash-Attention 2.8.3..."
    GPU_ARCH=$(python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
    echo "GPU compute capability: sm_${GPU_ARCH:-unknown}"

    if [ "$GPU_ARCH" = "80" ] || [ "$GPU_ARCH" = "86" ] || [ "$GPU_ARCH" = "89" ] || [ "$GPU_ARCH" = "90" ]; then
        echo "Using prebuilt FA2 wheel for sm_${GPU_ARCH}..."
        WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        rm -f "$WHEEL_NAME"
        wget -O "$WHEEL_NAME" "$WHEEL_URL"
        uv pip install "$WHEEL_NAME"
        rm -f "$WHEEL_NAME"
    else
        if python -c "import flash_attn; print(f'Flash-Attn {flash_attn.__version__} already installed')" 2>/dev/null; then
            echo "Skipping FA2 build (already installed)."
        else
            echo "WARNING: No prebuilt FA2 wheel for sm_${GPU_ARCH}. Building from source (30-40 min)..."
            echo "Run manually: MAX_JOBS=4 pip install flash-attn --no-build-isolation"
        fi
    fi

    # Final verification
    echo ""
    echo "Verifying GRPO setup..."
    python -c "
import torch

if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)

cc = torch.cuda.get_device_capability()
try:
    import flash_attn
    fa_ver = flash_attn.__version__
except ImportError:
    fa_ver = 'NOT INSTALLED'

import transformers
import trl
import peft

print(f'PyTorch:        {torch.__version__}')
print(f'CUDA:           {torch.version.cuda}')
print(f'GPU:            {torch.cuda.get_device_name(0)}')
print(f'GPU Arch:       sm_{cc[0]}{cc[1]}')
print(f'VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
print(f'Flash-Attn:     {fa_ver}')
print(f'Transformers:   {transformers.__version__}')
print(f'TRL:            {trl.__version__}')
print(f'PEFT:           {peft.__version__}')
print('')
print('SUCCESS: All GRPO components verified')
"

    echo ""
    echo "============================================"
    echo "GRPO Setup Complete! (UV)"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_GRPO/bin/activate"
    echo ""
    echo "To run GRPO training:"
    echo "  python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode micro --use-instruction false"
    echo ""
    exit 0
fi

# Unknown flag
echo "Error: Unknown flag '$1'"
echo ""
echo "Usage:"
echo "  ./setup_env_grpo.sh --gpu   # GPU Server (Nvidia ONLY)"
exit 1
