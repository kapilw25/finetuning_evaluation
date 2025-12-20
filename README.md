# CITA: Calibrated Instruction Tuning with Alignment

Comparative study of SFT → DPO/PPO/GRPO → CITA training pipeline on Llama-3.1-8B.

## Installation

**Tested:** Python 3.12, PyTorch 2.5.1+cu124, Flash-Attn 2.8.3, A100-80GB

### Environment 1: venv_CITA (SFT, DPO, PPO, CITA)

**Quick Setup (~2 min)**
```bash
chmod +x setup_env.sh
./setup_env.sh
source venv_CITA/bin/activate
```

**Manual Setup (if script fails)**
```bash
# 1. Create venv
python3.12 -m venv venv_CITA
source venv_CITA/bin/activate

# 2. Install PyTorch (MUST use --index-url for CUDA version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install requirements
pip install -r requirements.txt

# 4. Install flash-attn (choose one):
# Option A: Pre-built wheel (~30 sec)
curl -L -o fa.whl "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
mv fa.whl flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
rm flash_attn*.whl

# Option B: Build from source (~30-40 min, if Option A fails)
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-binary flash-attn

# 5. Verify
python -c "import torch, flash_attn; print(f'PyTorch: {torch.__version__}, Flash-Attn: {flash_attn.__version__}')"
```

### Environment 2: venv_GRPO (GRPO only - requires TRL 0.22.2)

**Quick Setup (~2 min)**
```bash
chmod +x setup_env_grpo.sh
./setup_env_grpo.sh
source venv_GRPO/bin/activate
```

**Manual Setup (if script fails)**
```bash
# 1. Create venv
python3.12 -m venv venv_GRPO
source venv_GRPO/bin/activate

# 2. Install PyTorch (MUST use --index-url for CUDA version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install requirements
pip install -r requirements_grpo.txt

# 4. Install TRL 0.22.2 with --no-deps (avoid transformers version conflict)
pip install --no-deps trl==0.22.2

# 5. Install flash-attn (pre-built wheel)
curl -L -o fa.whl "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
mv fa.whl flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
rm flash_attn*.whl

# 6. Verify
python -c "import torch, flash_attn; print(f'PyTorch: {torch.__version__}, Flash-Attn: {flash_attn.__version__}')"
```

## Training Commands

### Training Modes Summary

| Script | Micro | Sanity | Full |
|--------|-------|--------|------|
| SFT | N/A | ~13 min | ~43 min |
| DPO | N/A | ~31 min | ~103 min |
| PPO | ~1 hour | ~5 hours | ~17 hours |
| GRPO | ~30 min | ~3 hours | ~11 hours |
| CITA | N/A | ~36 min | ~120 min |

### 1. SFT (venv_CITA)
```bash
source venv_CITA/bin/activate

# SANITY: 0.3 epochs (~13 min)
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity --use-instruction false
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity --use-instruction true

# FULL: 1.0 epoch (~43 min)
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full --use-instruction false
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full --use-instruction true
```

### 2a. DPO (venv_CITA)
```bash
source venv_CITA/bin/activate

# SANITY: 0.3 epochs (~31 min)
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction false
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction true

# FULL: 1.0 epoch (~103 min)
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full --use-instruction false
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full --use-instruction true
```

### 2b. PPO (venv_CITA)
```bash
source venv_CITA/bin/activate

# MICRO: 0.05 epochs (~1 hour)
python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode micro --use-instruction false
python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode micro --use-instruction true

# SANITY: 0.3 epochs (~5 hours)
python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction false
python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction true

# FULL: 1.0 epoch (~17 hours) - use TMUX
python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode full --use-instruction false
python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode full --use-instruction true
```

### 2c. GRPO (venv_GRPO - requires TRL 0.22.2)
```bash
source venv_GRPO/bin/activate

# MICRO: 0.05 epochs (~30 min)
python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode micro --use-instruction false
python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode micro --use-instruction true

# SANITY: 0.3 epochs (~3 hours)
python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction false
python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction true

# FULL: 1.0 epoch (~11 hours) - use TMUX
python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode full --use-instruction false
python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode full --use-instruction true
```

### 3. CITA (venv_CITA)
```bash
source venv_CITA/bin/activate

# SANITY: 0.3 epochs (~36 min)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode sanity --use-instruction false
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode sanity --use-instruction true

# FULL: 1.0 epoch (~120 min)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full --use-instruction false
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full --use-instruction true
```

### Full Training Pipeline (FULL mode, A100-80GB)

**NoInstruct Pipeline**
```bash
# 1. SFT NoInstruct (~43 min) - venv_CITA
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full --use-instruction false

# 2a. DPO NoInstruct (~103 min) - venv_CITA
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full --use-instruction false

# 2b. PPO NoInstruct (~17 hours) - venv_CITA (Alternative to DPO)
python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode full --use-instruction false

# 2c. GRPO NoInstruct (~11 hours) - venv_GRPO (Alternative to DPO)
python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode full --use-instruction false

# 3. CITA NoInstruct (~120 min) - venv_CITA
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full --use-instruction false
```

**Instruct Pipeline**
```bash
# 1. SFT Instruct (~43 min) - venv_CITA
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full --use-instruction true

# 2a. DPO Instruct (~103 min) - venv_CITA
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full --use-instruction true

# 2b. PPO Instruct (~17 hours) - venv_CITA (Alternative to DPO)
python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode full --use-instruction true

# 2c. GRPO Instruct (~11 hours) - venv_GRPO (Alternative to DPO)
python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode full --use-instruction true

# 3. CITA Instruct (~120 min) - venv_CITA
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full --use-instruction true
```

### Optuna HP Search (27 trials × 1354 steps, ~20-24 hrs)
```bash
# CITA NoInstruct
python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py \
    --mode full --use-instruction false \
    --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct

# CITA Instruct
python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py \
    --mode full --use-instruction true \
    --base_model kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct
```

## Evaluation

```bash
# Activate environment
source venv_CITA/bin/activate

# 1. ISD (select option 3 for Max)
python comparative_study/05_evaluation/isd/evaluation_embedding.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct

# 2. TruthfulQA (select option 3 for Max)
python comparative_study/05_evaluation/truthfulqa/evaluation.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct

# 3. Conditional Safety (select option 3 for Max)
python comparative_study/05_evaluation/conditional_safety/evaluation.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct

# 4. Length Control (select option 3 for Max)
python comparative_study/05_evaluation/length_control/evaluation.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct

# 5. AQI (select option 2 for Full - NOT Max)
python comparative_study/05_evaluation/AQI/evaluation.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct \
  --batch_size 4
```

### With specific models:
```bash
python comparative_study/05_evaluation/isd/evaluation_embedding.py \
    --models CITA_Instruct CITA_NoInstruct DPO_Instruct DPO_NoInstruct
```

## HuggingFace Models

| Variant | SFT | DPO | PPO | GRPO | CITA |
|---------|-----|-----|-----|------|------|
| NoInstruct | [SFT-NoInstruct](https://huggingface.co/kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct) | [DPO-NoInstruct](https://huggingface.co/kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct) | [PPO-NoInstruct](https://huggingface.co/kapilw25/llama3-8b-pku-PPO-NoInstruct-SFT-NoInstruct) | [GRPO-NoInstruct](https://huggingface.co/kapilw25/llama3-8b-pku-GRPO-NoInstruct-SFT-NoInstruct) | [CITA-NoInstruct](https://huggingface.co/kapilw25/llama3-8b-pku-CITA-NoInstruct-DPO-NoInstruct) |
| Instruct | [SFT-Instruct](https://huggingface.co/kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct) | [DPO-Instruct](https://huggingface.co/kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct) | [PPO-Instruct](https://huggingface.co/kapilw25/llama3-8b-pku-PPO-Instruct-SFT-Instruct) | [GRPO-Instruct](https://huggingface.co/kapilw25/llama3-8b-pku-GRPO-Instruct-SFT-Instruct) | [CITA-Instruct](https://huggingface.co/kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct) |

**Dataset:** [ISD-Instruction-Switch-Dataset](https://huggingface.co/datasets/kapilw25/ISD-Instruction-Switch-Dataset)

## Results

See [logs_training/iter13/observation.md](logs_training/iter13/observation.md) for latest evaluation results.

## Paper (LaTeX)

```bash
# Install LaTeX (one-time)
# macOS:
brew install --cask mactex  # or: brew install basictex
# Ubuntu/Debian:
sudo apt-get install texlive-full

# Compile PDF
cd Overleaf_draft && pdflatex 0_main.tex && bibtex 0_main && pdflatex 0_main.tex && pdflatex 0_main.tex
```

Output: `Overleaf_draft/main.pdf`

## Notes

| Script | Environment | Interactive | Key Args |
|--------|-------------|-------------|----------|
| SFT/DPO/PPO/CITA | venv_CITA | Yes (train vs inference) | `--use-instruction` REQUIRED |
| GRPO | venv_GRPO | Yes (train vs inference) | `--use-instruction` REQUIRED |
| Optuna | venv_CITA | Yes (fresh vs continue) | `--mode mvp/sanity/full` |
| All Evals | venv_CITA | Yes (sanity/full/max menu) | `--models` optional |

**Why two environments?**
- `venv_CITA`: TRL 0.11.4 (SFT, DPO, PPO, CITA)
- `venv_GRPO`: TRL 0.22.2 (GRPO requires GRPOConfig/GRPOTrainer added in TRL 0.14.0+)
