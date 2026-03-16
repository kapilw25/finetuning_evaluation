#!/bin/bash
# ============================================================================
# Environment Setup Script - CITA Training (UV-based)
# ============================================================================
# Usage:
#   chmod +x setup_env_uv.sh
#
#   # M1 Mac (CPU-based: evaluation, plotting, HF push)
#   ./setup_env_uv.sh --mac 2>&1 | tee logs/setup_env_cpu.log
#
#   # GPU Server (CITA/DPO/PPO/GRPO training)
#   ./setup_env_uv.sh --gpu 2>&1 | tee logs/setup_env_gpu.log
#
#   # Activate
#   source venv_CITA/bin/activate
# ============================================================================

set -e  # Exit on error
mkdir -p logs  # Ensure logs/ exists early (for tee piping)

# ============================================================================
# Pinned versions (A100 + CUDA 12.4 + Python 3.12)
# ============================================================================
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"

# ============================================================================
# Parse flags
# ============================================================================
# Show usage if no flag provided
if [ -z "$1" ]; then
    echo "Error: No flag provided"
    echo ""
    echo "Usage:"
    echo "  ./setup_env_uv.sh --mac   # M1 Mac (CPU-based)"
    echo "  ./setup_env_uv.sh --gpu   # GPU Server (Nvidia ONLY)"
    exit 1
fi

# Detect OS
OS="$(uname -s)"

# ============================================================================
# Common setup (both --mac and --gpu)
# ============================================================================
setup_base() {
    echo "============================================"
    echo "CITA Environment Setup (UV)"
    echo "============================================"
    echo "Detected OS: $OS"
    echo ""

    # Install system dependencies
    if [ "$OS" = "Linux" ]; then
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

        # LaTeX (for paper compilation)
        if ! command -v pdflatex &> /dev/null; then
            APT_PACKAGES="$APT_PACKAGES texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-bibtex-extra texlive-science biber"
            NEED_APT_UPDATE=true
        fi

        if [ "$NEED_APT_UPDATE" = true ]; then
            echo "Installing: $APT_PACKAGES"
            apt-get update
            apt-get install -y $APT_PACKAGES
        fi
    elif [ "$OS" = "Darwin" ]; then
        command -v tree &> /dev/null || brew install tree
        # Python 3.12 — use system python3 on Mac (usually 3.12+)
    fi
    echo "Python: $(python3 --version)"

    # Install UV if not available
    if ! command -v uv &> /dev/null; then
        echo ""
        echo "Installing UV package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add to PATH for current session
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo "UV: $(uv --version)"

    # Create virtual environment with UV
    if [ ! -d "venv_CITA" ]; then
        echo ""
        echo "Creating virtual environment (Python 3.12)..."
        if [ "$OS" = "Linux" ]; then
            uv venv --python 3.12 venv_CITA
        else
            uv venv venv_CITA
        fi
    else
        echo "Virtual environment already exists (venv_CITA)"
    fi

    # Activate virtual environment
    source venv_CITA/bin/activate

    # Install base requirements with UV
    echo ""
    echo "Installing base requirements (UV)..."
    uv pip install -r requirements.txt

    # Create directories
    echo ""
    echo "Creating directories..."
    mkdir -p outputs logs

    echo ""
    echo "Base setup complete."
    echo ""
}

# ============================================================================
# --mac: M1 Mac setup (CPU-based)
# ============================================================================
if [ "$1" = "--mac" ]; then
    setup_base

    echo "============================================"
    echo "M1 Mac Setup Complete! (UV)"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_CITA/bin/activate"
    echo ""
    exit 0
fi

# ============================================================================
# --gpu: GPU Server setup (CITA/DPO/PPO/GRPO training) - Nvidia ONLY
# ============================================================================
if [ "$1" = "--gpu" ]; then
    if [ "$OS" != "Linux" ]; then
        echo "Error: --gpu requires Linux. Detected: $OS"
        exit 1
    fi

    setup_base

    echo "============================================"
    echo "GPU Setup (Linux + Nvidia ONLY)"
    echo "============================================"

    # 1. Install PyTorch (CUDA 12.4)
    echo ""
    echo "[1/4] Installing PyTorch ${TORCH_VERSION}+cu124..."
    uv pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${CUDA_INDEX_URL}"

    # 2. Verify PyTorch + CUDA
    echo ""
    echo "[2/4] Verifying PyTorch + CUDA..."
    python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. Nvidia GPU required.')
    exit(1)
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
"

    # 3. Install GPU requirements
    echo ""
    echo "[3/4] Installing GPU requirements (UV)..."
    uv pip install -r requirements_gpu.txt

    # 4. Install Flash-Attention 2 (pre-built wheel for A100/H100)
    echo ""
    echo "[4/4] Installing Flash-Attention 2.8.3..."
    GPU_ARCH=$(python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
    echo "GPU compute capability: sm_${GPU_ARCH:-unknown}"

    if [ "$GPU_ARCH" = "80" ] || [ "$GPU_ARCH" = "86" ] || [ "$GPU_ARCH" = "89" ] || [ "$GPU_ARCH" = "90" ]; then
        # Ampere/Ada/Hopper — use prebuilt wheel (fast)
        echo "Using prebuilt FA2 wheel for sm_${GPU_ARCH}..."
        WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        rm -f "$WHEEL_NAME"
        wget -O "$WHEEL_NAME" "$WHEEL_URL"
        uv pip install "$WHEEL_NAME"
        rm -f "$WHEEL_NAME"
    else
        # Unknown arch — build from source
        if python -c "import flash_attn; print(f'Flash-Attn {flash_attn.__version__} already installed')" 2>/dev/null; then
            echo "Skipping FA2 build (already installed)."
        else
            echo "WARNING: No prebuilt FA2 wheel for sm_${GPU_ARCH}. Building from source (30-40 min)..."
            echo "Run manually: MAX_JOBS=4 pip install flash-attn --no-build-isolation"
        fi
    fi

    # Final verification
    echo ""
    echo "Verifying GPU setup..."
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
print('SUCCESS: All GPU components verified')
"

    echo ""
    echo "============================================"
    echo "GPU Setup Complete! (UV)"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_CITA/bin/activate"
    echo ""
    echo "To run training:"
    echo "  python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity --use-instruction false"
    echo ""
    exit 0
fi

# Unknown flag
echo "Error: Unknown flag '$1'"
echo ""
echo "Usage:"
echo "  ./setup_env_uv.sh --mac   # M1 Mac (CPU-based)"
echo "  ./setup_env_uv.sh --gpu   # GPU Server (Nvidia ONLY)"
exit 1
