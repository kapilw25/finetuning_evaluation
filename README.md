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

All evals are interactive - select "full" when prompted.

```bash
# ISD (300 prompts × 10 instructions)
python comparative_study/05_evaluation/isd/evaluation_embedding.py

# Toxicity (3,684 both-unsafe prompts)
python comparative_study/05_evaluation/toxicity/evaluation.py

# TruthfulQA (817 questions × 2 variants)
python comparative_study/05_evaluation/truthfulqa/evaluation.py

# Conditional Safety (500 prompts × 2 variants)
python comparative_study/05_evaluation/conditional_safety/evaluation.py

# Style Transfer (500 prompts × 2 variants)
python comparative_study/05_evaluation/style_transfer/evaluation.py

# AQI (200 samples per category)
python comparative_study/05_evaluation/AQI/evaluation.py
```

### With specific models:
```bash
python comparative_study/05_evaluation/toxicity/evaluation.py \
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

## Notes

| Script | Interactive | Key Args |
|--------|-------------|----------|
| SFT/DPO/CITA | Yes (train vs inference) | `--use-instruction` REQUIRED |
| Optuna | Yes (fresh vs continue) | `--mode mvp/sanity/full` |
| All Evals | Yes (sanity/full menu) | `--models` optional |
