# CITA: Calibrated Instruction Tuning with Alignment

Comparative study of SFT → DPO → CITA training pipeline on Llama-3.1-8B.

## Installation

```bash
# 1. Create venv with Python 3.10
python3.10 -m venv venv_CITA

# 2. Activate
source venv_CITA/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Verify torch
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 5. Install flash-attn (10-40 mins to compile)
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# 6. Verify flash_attn
python -c "import flash_attn; print(f'Flash-Attention: {flash_attn.__version__}')"
```

## Training (FULL mode, A100-40GB)

### NoInstruct Pipeline
```bash
# 1. SFT NoInstruct (~43 min)
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py \
    --mode full --use-instruction false

# 2. DPO NoInstruct (~103 min)
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py \
    --mode full --use-instruction false \
    --base_model kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct

# 3. CITA NoInstruct (~120 min)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py \
    --mode full --use-instruction false \
    --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct
```

### Instruct Pipeline
```bash
# 1. SFT Instruct (~43 min)
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py \
    --mode full --use-instruction true

# 2. DPO Instruct (~103 min)
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py \
    --mode full --use-instruction true \
    --base_model kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct

# 3. CITA Instruct (~120 min)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py \
    --mode full --use-instruction true \
    --base_model kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct
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

| Variant | SFT | DPO | CITA |
|---------|-----|-----|------|
| NoInstruct | [SFT-NoInstruct](https://huggingface.co/kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct) | [DPO-NoInstruct](https://huggingface.co/kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct) | [CITA-NoInstruct](https://huggingface.co/kapilw25/llama3-8b-pku-CITA-NoInstruct-DPO-NoInstruct) |
| Instruct | [SFT-Instruct](https://huggingface.co/kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct) | [DPO-Instruct](https://huggingface.co/kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct) | [CITA-Instruct](https://huggingface.co/kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct) |

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

| Script | Interactive | Key Args |
|--------|-------------|----------|
| SFT/DPO/CITA | Yes (train vs inference) | `--use-instruction` REQUIRED |
| Optuna | Yes (fresh vs continue) | `--mode mvp/sanity/full` |
| All Evals | Yes (sanity/full/max menu) | `--models` optional |
