#!/bin/bash
# ============================================================================
# Environment Setup Script - CITA Training (UV-based)
# ============================================================================
# Usage:
#   chmod +x setup_env_uv.sh
#
#   # M1 Mac (CPU-based: evaluation, plotting, HF push)
#   mkdir -p logs && ./setup_env_uv.sh --mac 2>&1 | tee logs/setup_env_cpu.log
#
#   # GPU Server: Ampere/Hopper (A100, H100) OR Blackwell (RTX PRO 4000/5090)
#   mkdir -p logs && ./setup_env_uv.sh --gpu 2>&1 | tee logs/setup_env_gpu.log
#
#   # GPU Server with prebuilt sm_120 FA2 wheel from this repo's GitHub release
#   mkdir -p logs && ./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
#
#   # Activate
#   source venv_CITA/bin/activate
# ============================================================================
#
# Fixes vs. previous version (verified against logs/setup_env_gpu.log):
#   1. PyTorch 2.5.1+cu124 has NO sm_120 (Blackwell) support — auto-detect now
#      installs nightly cu128 for Blackwell GPUs, stable cu124 for Ampere/Hopper.
#   2. FA2 step fell through to a no-op `echo` for sm_120 — now actually builds
#      from source with a version-matched CUDA toolkit (apt-installed if missing).
#   3. Final verifier reported `SUCCESS` even with FA2 missing — now exits non-zero
#      when flash_attn import fails (no SDPA fallback for CITA training).
# ============================================================================

set -e
mkdir -p logs

# ============================================================================
# Pinned versions
# ============================================================================
# Blackwell sm_120 — PyTorch nightly cu128 (stable wheels lack sm_120 kernels).
# Pinned to dev20260407: dev20260228 was rotated off the nightly CDN, dev20260408
# has no paired torchvision build. dev20260407 is the latest with paired torchvision.
TORCH_BLACKWELL_VERSION="2.12.0.dev20260407"
CUDA_INDEX_BLACKWELL="https://download.pytorch.org/whl/nightly/cu128"

# Ampere/Hopper (A100, H100) — stable PyTorch.
TORCH_AMPERE_VERSION="2.5.1"
TORCHVISION_AMPERE_VERSION="0.20.1"
CUDA_INDEX_AMPERE="https://download.pytorch.org/whl/cu124"

# GitHub release tag for prebuilt sm_120 FA2 wheel.
# NOTE: any prebuilt FA2 wheel must match the PyTorch ABI (dev20260407 cu128).
# If a wheel built against a different torch ABI is uploaded, the auto-detect
# below will install it but `import flash_attn` will fail at runtime — in that
# case, delete wheels/flash_attn*.whl and re-run to trigger a fresh source build.
RELEASE_TAG="sm120-cu128-py312"

# Centralized wheel repo: a single GitHub repo hosts the prebuilt sm_120 wheels
# shared across kapilw25's Blackwell projects (factorjepa, walkindia, cita).
# Override with: WHEEL_REPO=owner/repo ./setup_env_uv.sh --gpu --from-wheels
# Set to empty to fall back to `git remote get-url origin` (this repo's releases).
WHEEL_REPO="${WHEEL_REPO:-kapilw25/factorjepa}"

# ============================================================================
# Parse flags
# ============================================================================
FROM_WHEELS=false
for arg in "$@"; do
    if [ "$arg" = "--from-wheels" ]; then
        FROM_WHEELS=true
    fi
done

if [ -z "$1" ]; then
    echo "Error: No flag provided"
    echo ""
    echo "Usage:"
    echo "  ./setup_env_uv.sh --mac                 # M1 Mac (CPU-based)"
    echo "  ./setup_env_uv.sh --gpu                 # GPU Server (Nvidia ONLY)"
    echo "  ./setup_env_uv.sh --gpu --from-wheels   # GPU + prebuilt sm_120 FA2 wheel"
    exit 1
fi

OS="$(uname -s)"

# ============================================================================
# Download prebuilt sm_120 wheels from this repo's GitHub release
# ============================================================================
download_sm120_wheels() {
    local REPO_SLUG
    if [ -n "$WHEEL_REPO" ]; then
        REPO_SLUG="$WHEEL_REPO"
    else
        REPO_SLUG=$(git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||' | sed 's|\.git$||')
    fi
    if [ -z "$REPO_SLUG" ]; then
        echo "FATAL: Cannot detect GitHub repo (WHEEL_REPO unset and no git remote)."
        return 1
    fi

    local API_URL="https://api.github.com/repos/${REPO_SLUG}/releases/tags/${RELEASE_TAG}"
    mkdir -p wheels
    echo "Downloading prebuilt sm_120 wheels from: github.com/${REPO_SLUG}/releases/tag/${RELEASE_TAG}"

    local URLS
    URLS=$(curl -sL "$API_URL" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for asset in data.get('assets', []):
        if asset['name'].endswith('.whl'):
            print(asset['browser_download_url'])
except: pass
" 2>/dev/null)

    if [ -z "$URLS" ]; then
        echo "WARNING: No wheels found in release '${RELEASE_TAG}'."
        echo "Upload wheels first: gh release create ${RELEASE_TAG} wheels/*.whl"
        return 1
    fi

    for url in $URLS; do
        echo "  Downloading: $(basename "$url")"
        wget -q -P wheels/ "$url"
    done
    echo "Downloaded $(ls wheels/*.whl 2>/dev/null | wc -l) wheel(s) to wheels/"
}

# ============================================================================
# Common setup (both --mac and --gpu)
# ============================================================================
setup_base() {
    echo "============================================"
    echo "CITA Environment Setup (UV)"
    echo "============================================"
    echo "Detected OS: $OS"
    echo ""

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

        # LaTeX (paper compilation)
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
        command -v python3.12 &> /dev/null || brew install python@3.12
    fi
    echo "Python: $(python3 --version)"

    if ! command -v uv &> /dev/null; then
        echo ""
        echo "Installing UV package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo "UV: $(uv --version)"

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

    source venv_CITA/bin/activate

    echo ""
    echo "Installing base requirements (UV)..."
    uv pip install -r requirements.txt

    echo ""
    echo "Creating directories..."
    mkdir -p outputs logs wheels

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
# --gpu: GPU Server setup (Nvidia ONLY) — auto-detects Blackwell vs Ampere/Hopper
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

    # Optional: load .env (HF_TOKEN for gated checkpoints, e.g. Llama-3.1-8B-Instruct)
    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
        echo "Loaded .env"
    fi

    # Optional: download prebuilt sm_120 FA2 wheel from this repo's GitHub release
    if [ "$FROM_WHEELS" = true ]; then
        echo ""
        echo "=== Downloading prebuilt wheels ==="
        download_sm120_wheels || {
            echo "Falling back to building from source..."
            FROM_WHEELS=false
        }
    fi

    # 1. PyTorch — auto-detect Blackwell vs Ampere/Hopper
    echo ""
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    echo "Detected GPU: ${GPU_NAME:-unknown}"
    if echo "$GPU_NAME" | grep -qiE "blackwell|rtx.*pro.*(4000|5000|6000)|rtx.*5090|rtx.*5080|rtx.*5070|b100|b200"; then
        echo "[1/4] Installing PyTorch ${TORCH_BLACKWELL_VERSION}+cu128 (Blackwell — pinned nightly)..."
        uv pip install "torch==${TORCH_BLACKWELL_VERSION}" torchvision --index-url "${CUDA_INDEX_BLACKWELL}"
    else
        echo "[1/4] Installing PyTorch ${TORCH_AMPERE_VERSION}+cu124 (Ampere/Hopper — stable)..."
        uv pip install "torch==${TORCH_AMPERE_VERSION}" "torchvision==${TORCHVISION_AMPERE_VERSION}" --index-url "${CUDA_INDEX_AMPERE}"
    fi

    # 2. Verify PyTorch + CUDA
    echo ""
    echo "[2/4] Verifying PyTorch + CUDA..."
    python -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. Nvidia GPU required.')
    sys.exit(1)
cc = torch.cuda.get_device_capability()
arch = f'{cc[0]}{cc[1]}'
supported = [s.replace('sm_', '') for s in torch.cuda.get_arch_list() if s.startswith('sm_')]
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)} (sm_{arch})')
print(f'PyTorch arch list: {torch.cuda.get_arch_list()}')
if arch not in supported:
    print(f'FATAL: GPU sm_{arch} is NOT in PyTorch supported archs. PyTorch will fall back to PTX JIT or fail.')
    print('Per CLAUDE.md no-fallback rule, refusing to proceed. Bump TORCH_*_VERSION in this script.')
    sys.exit(1)
print(f'OK: sm_{arch} supported natively.')
"

    # 3. GPU requirements
    echo ""
    echo "[3/4] Installing GPU requirements (UV)..."
    uv pip install -r requirements_gpu.txt
    export HF_HUB_ENABLE_HF_TRANSFER=1

    # 4. Flash-Attention 2 (prebuilt → arch-based wheel → source build)
    echo ""
    echo "[4/4] Installing Flash-Attention 2..."
    GPU_ARCH=$(python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
    echo "GPU compute capability: sm_${GPU_ARCH:-unknown}"

    if ls wheels/flash_attn*.whl &>/dev/null 2>&1; then
        echo "Installing FA2 from prebuilt wheel in wheels/..."
        uv pip install wheels/flash_attn*.whl
    elif [ "$GPU_ARCH" = "80" ] || [ "$GPU_ARCH" = "86" ] || [ "$GPU_ARCH" = "89" ] || [ "$GPU_ARCH" = "90" ]; then
        echo "Using prebuilt FA2 v2.8.3 wheel for sm_${GPU_ARCH}..."
        WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        rm -f flash_attn*.whl
        if command -v aria2c &> /dev/null; then
            aria2c -x 16 -s 16 -o "$WHEEL_NAME" "$WHEEL_URL"
        else
            wget -O "$WHEEL_NAME" "$WHEEL_URL"
        fi
        uv pip install "$WHEEL_NAME"
        rm -f "$WHEEL_NAME"
    elif [ -n "$GPU_ARCH" ]; then
        # Unknown arch (e.g. sm_120 Blackwell) — check existing install, then build from source
        if python -c "import flash_attn; print(f'Flash-Attn {flash_attn.__version__} already installed')" 2>/dev/null; then
            echo "Skipping FA2 build (already installed)."
        else
            echo "WARNING: No prebuilt FA2 wheel for sm_${GPU_ARCH}. Building from source (30-90 min)..."

            # FA2 source build needs nvcc matching PyTorch's CUDA version exactly
            PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
            echo "PyTorch compiled with CUDA: ${PYTORCH_CUDA}"

            FA2_CUDA_HOME=""
            for CUDA_PATH in "/usr/local/cuda-${PYTORCH_CUDA}" /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda; do
                if [ -f "${CUDA_PATH}/bin/nvcc" ]; then
                    NVCC_VER=$("${CUDA_PATH}/bin/nvcc" --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
                    if [ "${NVCC_VER}" = "${PYTORCH_CUDA}" ]; then
                        FA2_CUDA_HOME="${CUDA_PATH}"
                        echo "Found matching CUDA ${NVCC_VER} toolkit at ${CUDA_PATH}"
                        break
                    fi
                fi
            done

            if [ -z "$FA2_CUDA_HOME" ]; then
                CUDA_PKG="cuda-toolkit-$(echo "${PYTORCH_CUDA}" | tr '.' '-')"
                echo "No CUDA ${PYTORCH_CUDA} toolkit found. Installing ${CUDA_PKG} via apt..."
                apt-get update -qq && apt-get install -y -qq "${CUDA_PKG}" > /dev/null 2>&1 || true
                if [ -f "/usr/local/cuda-${PYTORCH_CUDA}/bin/nvcc" ]; then
                    FA2_CUDA_HOME="/usr/local/cuda-${PYTORCH_CUDA}"
                    echo "Installed CUDA ${PYTORCH_CUDA} at ${FA2_CUDA_HOME}"
                fi
            fi

            if [ -z "$FA2_CUDA_HOME" ]; then
                echo "FATAL: Could not find or install CUDA toolkit ${PYTORCH_CUDA}."
                echo "System nvcc: $(nvcc --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p') (needs ${PYTORCH_CUDA})"
                echo "Install manually: apt-get install cuda-toolkit-$(echo "${PYTORCH_CUDA}" | tr '.' '-')"
                exit 1
            fi

            export CUDA_HOME="${FA2_CUDA_HOME}"
            export PATH="${FA2_CUDA_HOME}/bin:$PATH"
            echo "Using nvcc: $(nvcc --version | grep release)"
            FA2_DIR="/tmp/flash-attention-build"
            rm -rf "$FA2_DIR"
            git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$FA2_DIR"
            cd "$FA2_DIR" && git submodule update --init --recursive && cd -
            echo "Building FA2 wheel for sm_${GPU_ARCH} (this takes 30-90 min)..."
            mkdir -p wheels
            uv pip install pip 2>/dev/null || true
            FLASH_ATTN_CUDA_ARCHS="${GPU_ARCH}" MAX_JOBS=4 NVCC_THREADS=1 \
                pip wheel "$FA2_DIR" --no-build-isolation --no-deps --wheel-dir wheels/ 2>&1 | tee /tmp/fa2_build.log
            uv pip install wheels/flash_attn*.whl
            rm -rf "$FA2_DIR"
            echo "FlashAttention-2 built for sm_${GPU_ARCH}"
            echo "Wheel saved: $(ls wheels/flash_attn*.whl 2>/dev/null | head -1)"

            # Auto-upload wheel to this repo's GitHub release (skip 30-90 min build on future runs).
            # Guarded: requires gh CLI authenticated AND working FA2 import — never publishes broken wheels.
            echo ""
            echo "=== Auto-upload wheel to GitHub release (best-effort) ==="
            UPLOAD_WHEEL=$(ls wheels/flash_attn*.whl 2>/dev/null | head -1)
            if ! command -v gh &> /dev/null; then
                echo "  Skipped: gh CLI not installed. Manual: gh release create ${RELEASE_TAG} ${UPLOAD_WHEEL}"
            elif ! gh auth status &> /dev/null; then
                echo "  Skipped: gh CLI not authenticated. Run 'gh auth login' then re-upload manually."
            elif ! python -c "import flash_attn" 2>/dev/null; then
                echo "  Skipped: FA2 import failed — refusing to publish broken wheel."
            elif [ -z "$UPLOAD_WHEEL" ]; then
                echo "  Skipped: no wheel found in wheels/."
            else
                if [ -n "$WHEEL_REPO" ]; then
                    UPLOAD_REPO="$WHEEL_REPO"
                else
                    UPLOAD_REPO=$(git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||' | sed 's|\.git$||')
                fi
                echo "  Uploading $(basename "$UPLOAD_WHEEL") to ${UPLOAD_REPO} release ${RELEASE_TAG}..."
                if gh release view "${RELEASE_TAG}" --repo "${UPLOAD_REPO}" &> /dev/null; then
                    gh release upload "${RELEASE_TAG}" "$UPLOAD_WHEEL" --repo "${UPLOAD_REPO}" --clobber \
                        && echo "  Uploaded to existing release: github.com/${UPLOAD_REPO}/releases/tag/${RELEASE_TAG}" \
                        || echo "  WARNING: gh release upload failed (non-fatal)."
                else
                    gh release create "${RELEASE_TAG}" "$UPLOAD_WHEEL" --repo "${UPLOAD_REPO}" \
                        --title "FA2 prebuilt wheels (sm_120, cu128, py312)" \
                        --notes "Auto-uploaded by setup_env_uv.sh after source build (torch ${TORCH_BLACKWELL_VERSION})" \
                        && echo "  Created release: github.com/${UPLOAD_REPO}/releases/tag/${RELEASE_TAG}" \
                        || echo "  WARNING: gh release create failed (non-fatal)."
                fi
            fi
        fi
    else
        echo "FATAL: Could not detect GPU arch."
        exit 1
    fi

    # Final verification — STRICT, fails if FA2 missing (no SDPA fallback for CITA training)
    echo ""
    echo "Verifying GPU setup..."
    python -c "
import torch, sys

if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    sys.exit(1)

cc = torch.cuda.get_device_capability()

try:
    import flash_attn
    fa_ver = flash_attn.__version__
except ImportError:
    print('FATAL: Flash-Attention not installed. Per CLAUDE.md no-fallback rule, refusing SUCCESS.')
    sys.exit(1)

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

    # Dependency health check (warning-only — surfaces issues without blocking)
    echo ""
    echo "[Dependency health] uv pip check..."
    set +e
    CHECK_OUT=$(uv pip check 2>&1)
    set -e
    INCOMPAT_COUNT=$(echo "$CHECK_OUT" | grep -cE '^The package')
    if [ "$INCOMPAT_COUNT" -gt 0 ]; then
        echo "WARNING: ${INCOMPAT_COUNT} dependency incompatibilities detected:"
        echo "$CHECK_OUT" | grep -E '^The package' | head -20
        echo "(non-fatal — investigate if training fails)"
    else
        echo "OK — no dependency conflicts"
    fi

    echo ""
    echo "============================================"
    echo "GPU Setup Complete! (UV)"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_CITA/bin/activate"
    echo ""
    echo "To run training (sanity smoke test):"
    echo "  python -u src/train/sft.py --mode sanity --use-instruction false 2>&1 | tee logs/sft_sanity.log"
    echo ""
    echo "Or train all 5 methods × {Instruct, NoInstruct}:"
    echo "  ./scripts/train_all.sh sanity"
    echo ""
    exit 0
fi

# Unknown flag
echo "Error: Unknown flag '$1'"
echo ""
echo "Usage:"
echo "  ./setup_env_uv.sh --mac                 # M1 Mac (CPU-based)"
echo "  ./setup_env_uv.sh --gpu                 # GPU Server (Nvidia ONLY)"
echo "  ./setup_env_uv.sh --gpu --from-wheels   # GPU + prebuilt sm_120 FA2 wheel"
exit 1
