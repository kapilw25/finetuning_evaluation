# GRIT: Geometry‑Aligned Instruction Tuning via Fisher‑Guided Reprojection

This repository contains the finalized GRIT implementation used in our paper. The code is organized as a small library with example scripts for easy reproduction on common instruction‑tuning and classification benchmarks.

## Highlights
- BA convention: ΔW = B A with B ∈ R^{d_out×r}, A ∈ R^{r×d_in}
- Rank‑space Fisher (r×r) via custom autograd hooks (running‑mean stats; no full‑matrix ops)
- Natural‑gradient (NG) preconditioning: ∇B ← ∇B Σ_g^{-1}, ∇A ← Σ_a^{-1} ∇A
- Two‑sided Fisher‑guided reprojection (default), sample‑gated G‑side; dynamic rank adaptation by cumulative energy threshold τ
- Optional curvature and reprojection regularizers (warmup‑gated)

## Repository Layout
```
GRIT/
  grit/
    __init__.py
    autograd.py        # Rank‑space covariance capture (KFACAutogradFunction)
    config.py          # GRITConfig defaults (used across examples)
    manager.py         # GRITManager: instrumentation, inversion, NG, reprojection
    optim.py           # GritOptimizer wrapper (preconditions before step)
    trainer.py         # GritTrainer (HF Seq2SeqTrainer subclass + callback)
  examples/
    train_alpaca.py    # Alpaca (instruction tuning)
    train_dolly.py     # Dolly 15k (instruction/context/response)
    train_boolq.py     # BoolQ (Yes/No), instructionified
    train_qnli.py      # QNLI (entailment), instructionified
    train_gsm8k.py     # GSM8K (math reasoning), instructionified
  README.md            # This file
```

## Installation
- Python 3.10+
- CUDA‑capable GPU (A100/RTX 6000 Ada or similar recommended)

```bash
pip install -r requirements.txt
# If you manage CUDA/torch manually, ensure torch/transformers/bitsandbytes versions are compatible
```

Optional logging/plots:
```bash
pip install wandb matplotlib seaborn
```

## Quickstart
All examples use QLoRA + GRIT by default. They run with 4‑bit NF4, bf16 compute, and LoRA on attention + MLP projections.

```bash
# Alpaca (instruction tuning)
python examples/train_alpaca.py

# Dolly 15k
python examples/train_dolly.py

# BoolQ (classification)
python examples/train_boolq.py

# QNLI (classification)
python examples/train_qnli.py

# GSM8K (math reasoning)
python examples/train_gsm8k.py
```

Two‑sided reprojection is the default (B uses eig(G_cov) when sufficiently sampled). To force an A‑side‑only ablation:
```python
# In your train_*.py before creating GRITManager
config.use_two_sided_reprojection = False
```

## LoRA vs QLoRA (Q-GRIT)
All example training scripts in `examples/` are configured to run on **LoRA modules** by default.  
If you want to use GRIT with **QLoRA (Q-GRIT)** instead, you need to supply a `bnb_config` when loading the model so that quantization is applied.

Example:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
```

Typical places to customize:
- Model: `config.model_id` (e.g., Llama‑3.x 3B/8B)
- Dataset: `config.dataset_name` (or see dataset logic in each example)
- LoRA: `config.lora_rank`, `config.min_lora_rank`, `config.lora_target_modules`
- K‑FAC & reprojection: `kfac_min_samples`, `kfac_update_freq`, `kfac_damping`, `reprojection_freq`

### How to override knobs (reproducibility)
Every example constructs a `GRITConfig()` and then optionally overrides fields before creating the `GRITManager`. You can reproduce paper variants by setting the same fields:

```python
config = GRITConfig()
config.model_id = "meta-llama/Llama-3.2-3B"  # or 8B
config.dataset_name = "tatsu-lab/alpaca"     # or change per task

# LoRA
config.lora_rank = 16
config.min_lora_rank = 8
config.lora_target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# K‑FAC / NG
config.kfac_min_samples = 192
config.kfac_update_freq = 150
config.kfac_damping = 0.003
config.kfac_inversion_device = 'cpu'  # set 'cuda' to invert on GPU

# Reprojection / rank adaptation
config.use_two_sided_reprojection = True   # GRIT‑2s (default); set False for ablation
### Notation (LoRA shapes)
We use ΔW = B A with B ∈ R^{d_out×r}, A ∈ R^{r×d_in} so that BA ∈ R^{d_out×d_in}. This matches the implementation and the paper. Some works write A'B'; both are equivalent with consistent shapes.
### Notation (LoRA shapes)
We use ΔW = B A with B ∈ R^{d_out×r}, A ∈ R^{r×d_in} so that BA ∈ R^{d_out×d_in}. This matches the implementation and the paper. Some works write A'B'; both are equivalent with consistent shapes.

config.reprojection_warmup_steps = 500
config.rank_adaptation_start_step = 500
config.reprojection_freq = 300
config.rank_adaptation_threshold = 0.99

# Training runtime
config.batch_size = 8
config.gradient_accumulation_steps = 4
config.learning_rate = 2e-5
```

HF `Seq2SeqTrainingArguments` are created inside each example; if you need exact reproducibility, set the seed explicitly and keep gradient clipping consistent with the paper:

```python
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
  ..., seed=42, data_seed=42, max_grad_norm=1.5, report_to="none",
)
```

## Reproducing Paper Tables/Figures
- Main results (Table 3): run the four example scripts on Llama‑3.2‑3B and Llama‑3.1‑8B; GRIT‑2s (two‑sided) is the default. For the GRIT‑1s ablation, set `use_two_sided_reprojection=False`. Keep other defaults; see per‑script overrides inside `examples/`.
- Efficiency (Table 4): counts are logged by `GRITManager.neural_reprojection()` at each reprojection event and at training end; we also log final raw ranks per module via `GritCallback`.
- Ablations (Table 5): toggle `use_two_sided_reprojection` and, when desired, delay rank adaptation with `rank_adaptation_start_step` to study fixed‑k vs adaptive.
- Wall‑clock (Table 6): the examples print steps and use HF timers; measure on your hardware (A100/RTX 6000 Ada) to match the paper’s setup.

## Key Configuration Knobs (paper §A.4)
Defaults are in `grit/config.py` and summarized below.

- kfac_update_freq: 50
- kfac_min_samples: 64
- kfac_damping: 1e‑3
- reprojection_freq: 50
- reprojection_k: 8
- use_two_sided_reprojection: True
- grit_cov_update_freq: 15 (per‑module covariance throttling via autograd hooks)
- enable_rank_adaptation: True
- rank_adaptation_threshold (τ): 0.99
- min_lora_rank: 4
- rank_adaptation_start_step: 0
- reprojection_warmup_steps: 0
- ng_warmup_steps: 0
- kfac_inversion_device: cpu (set to 'cuda' to invert on GPU)

Practical guidance (how to change):
- Early training: two‑sided is default, but the G‑side basis is sample‑gated; increase `kfac_min_samples=128–192`, `kfac_damping=0.003–0.007`, and enable warmups (100–150 steps) for stability.
- Longer runs (1–3 epochs): `kfac_min_samples=192`, `kfac_update_freq=150`, `reprojection_warmup_steps=500`, `rank_adaptation_start_step=500`, `reprojection_freq=300`.
- Adaptive scheduling: K‑FAC and reprojection frequencies are adapted heuristically to recent loss trends; tune base frequencies and allow the manager to adjust.
- Covariance throttling: `grit_cov_update_freq` reduces per‑step hook cost by sub‑sampling modules’ updates.
- NG clipping: HF clips after preconditioning. Use `max_grad_norm` as a trust‑region (disable with 0.0 or set a cap guided by recent `||g_nat||`).

## Logging (optional)
Enable W&B to capture spectra/heatmaps and NG norms.
```python
import wandb
wandb.init(project="GRIT", name="run")
# pass report_to="wandb" in HF training args
```
Spectrum/heatmap controls (in `GRITConfig`):
- `log_fisher_spectrum=True`, `log_top_eigs=8`, optional `log_eigs_bar=True`
- `log_eig_heatmaps=True`, limit modules with `log_eig_heatmaps_modules`

Effective‑rank logs at training end (and optionally at each inversion):
- `Fisher/<layer>/final_k_a`, `final_k_b`, `base_r` (for parameter accounting grids)

## Reproducibility
- Seeds and HF arguments are set inside each example; override as needed.
- Hardware: paper runs used A100 40GB; QNLI (8B) additionally tested on RTX 6000 Ada.
- To export final ranks/parameters, collect W&B logs or extend `GritManager` to dump a JSON on `on_train_end`.

## License
See `LICENSE` in this repository.

## Citation
If you use GRIT, please cite the accompanying paper.


