# Plan 9: BF16 Training + Multi-Seed + HuggingFace Integration

**Objective:** Eliminate quantization confound, add statistical rigor, centralize model artifacts

**Status:** READY FOR IMPLEMENTATION
**Priority:** HIGH (blocks paper submission)
**Estimated GPU Cost:** $240 (48 A100 hours)

---

## 🎯 Core Changes (All 6 Training Notebooks)

### ✅ Change 1: Remove Inference Sections
**Rationale:** `inference_evaluation.py` handles all inference centrally

**Files to modify:**
- `01a_SFT_Baseline/Llama3_alignmentDB.ipynb`
- `01b_SFT_GRIT/Llama3_alignmentDB.ipynb`
- `02a_DPO_Baseline/Llama3_alignmentDB.ipynb`
- `02b_DPO_GRIT/Llama3_alignmentDB.ipynb`
- `03a_CITA_Baseline/Llama3_alignmentDB.ipynb`
- `03b_CITA_GRIT/Llama3_alignmentDB.ipynb`

**Action:**
```python
# ❌ DELETE these sections from all notebooks:
# - "Inference" markdown header
# - All cells after "### Inference" up to save section
# - Test case cells (TEST 1, TEST 2, TEST 3, etc.)
# - Multi-turn conversation tests
# - FastLanguageModel.for_inference() calls

# ✅ KEEP only:
# - Training cells
# - Memory stats
# - Model/tokenizer save section
```

---

### ✅ Change 2: Add HuggingFace Push (All Notebooks)

**Location:** Add as FIRST cell after imports

```python
# ===================================================================
# HuggingFace Authentication & Configuration
# ===================================================================
import os
from huggingface_hub import login
from dotenv import load_dotenv
from datetime import datetime

load_dotenv('/lambda/nfs/DiskUsEast1/finetuning_evaluation/.env')
hf_token = os.getenv('HF_TOKEN')
login(token=hf_token)

MODEL_NAME_MAP = {
    "SFT_Baseline": "kapilw/llama3-8b-pku-sft-baseline",
    "SFT_GRIT": "kapilw/llama3-8b-pku-sft-grit",
    "DPO_Baseline": "kapilw/llama3-8b-pku-dpo-baseline",
    "DPO_GRIT": "kapilw/llama3-8b-pku-dpo-grit",
    "CITA_Baseline": "kapilw/llama3-8b-pku-cita-baseline",
    "CITA_GRIT": "kapilw/llama3-8b-pku-cita-grit",
}

# Set for this notebook
RUN_NAME = "SFT_Baseline"  # ← CHANGE PER NOTEBOOK
HF_REPO = MODEL_NAME_MAP[RUN_NAME]
```

**Replace save section:**

```python
# ❌ DELETE
# model.save_pretrained("lora_model_SFT_Baseline")
# tokenizer.save_pretrained("lora_model_SFT_Baseline")

# ✅ ADD
# Update model metadata
model.config.update({
    "training_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "training_steps": trainer_stats.global_step,
    "training_loss": trainer_stats.metrics['train_loss'],
    "dataset": "PKU-SafeRLHF",
    "filtered_samples": 10813,
    "max_steps": 300,
})

# Save locally (backup)
model.save_pretrained(f"lora_model_{RUN_NAME}")
tokenizer.save_pretrained(f"lora_model_{RUN_NAME}")

# Push to HuggingFace
print(f"\n📤 Pushing to HuggingFace: {HF_REPO}")
model.push_to_hub(
    HF_REPO,
    token=hf_token,
    commit_message=f"Training: {trainer_stats.metrics['train_runtime']:.1f}s, Loss: {trainer_stats.metrics['train_loss']:.4f}",
    private=True,
)
tokenizer.push_to_hub(HF_REPO, token=hf_token, private=True)
print(f"✅ Model available at: https://huggingface.co/{HF_REPO}")
```

---

## 🔧 Change 3: BF16 Training (6 New Notebooks)

### Create BF16 versions (suffix `_BF16.ipynb`):

**For GRIT notebooks (3 files):**
- `01b_SFT_GRIT/Llama3_alignmentDB_BF16.ipynb`
- `02b_DPO_GRIT/Llama3_alignmentDB_BF16.ipynb`
- `03b_CITA_GRIT/Llama3_alignmentDB_BF16.ipynb`

```python
# ❌ DELETE quantization config
# bnb_config = BitsAndBytesConfig(...)

# ✅ REPLACE model loading
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",
    torch_dtype=torch.bfloat16,  # No quantization
    token=hf_token,
)

# LoRA config unchanged
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)

# Prepare for training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, lora_config)

# ✅ UPDATE training args
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./outputs/{RUN_NAME}_BF16",
    per_device_train_batch_size=1,   # ← Reduced (BF16 uses 2x memory)
    gradient_accumulation_steps=8,    # ← Increased (keep effective batch=8)
    fp16=False,
    bf16=True,
    # ... rest same
)

# ✅ UPDATE HF_REPO suffix
HF_REPO = MODEL_NAME_MAP[RUN_NAME] + "-bf16"
```

**For Baseline notebooks (Unsloth):**
- `01a_SFT_Baseline/Llama3_alignmentDB_BF16.ipynb`
- `02a_DPO_Baseline/Llama3_alignmentDB_BF16.ipynb`
- `03a_CITA_Baseline/Llama3_alignmentDB_BF16.ipynb`

```python
# ❌ REPLACE
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/llama-3-8b-bnb-4bit",
#     load_in_4bit = True,
# )

# ✅ WITH
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3-8B",
    max_seq_length = 2048,
    dtype = torch.bfloat16,
    load_in_4bit = False,
)

# ✅ UPDATE training args
args = SFTConfig(  # or DPOConfig
    per_device_train_batch_size = 1,   # ← Reduced
    gradient_accumulation_steps = 8,    # ← Increased
    fp16 = False,
    bf16 = True,
    # ... rest same
)

# ✅ UPDATE HF_REPO suffix
HF_REPO = MODEL_NAME_MAP[RUN_NAME] + "-bf16"
```

---

## 🧪 Change 4: CITA Fixes (3 Experiment Notebooks)

**Create 3 new notebooks for CITA_Baseline:**
- `03a_CITA_Baseline/Llama3_alignmentDB_Exp1_AlpacaTemplate.ipynb`
- `03a_CITA_Baseline/Llama3_alignmentDB_Exp2_HigherLR.ipynb`
- `03a_CITA_Baseline/Llama3_alignmentDB_Exp3_LongerTraining.ipynb`

### Experiment 1: Alpaca Template

```python
# ✅ ADD after data loading
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="alpaca")

# ✅ REFORMAT dataset
def format_alpaca_prompt(example):
    system = example['chosen'][0]['content'] if example['chosen'][0]['role'] == 'system' else ""
    user = example['chosen'][1]['content']
    assistant = example['chosen'][2]['content']

    text = f"""Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{system}

{user}

### Response:
{assistant}"""
    return {'text': text}

dataset = dataset.map(format_alpaca_prompt, remove_columns=dataset.column_names)

# Training: lr=5e-6, steps=300 (baseline)
RUN_NAME = "CITA_Baseline_Exp1_AlpacaTemplate"
```

### Experiment 2: Higher LR

```python
# Keep Llama-3 template, change LR
args = DPOConfig(
    learning_rate = 2e-4,  # ← Changed from 5e-6
    # ... rest same
)

RUN_NAME = "CITA_Baseline_Exp2_HigherLR"
```

### Experiment 3: Longer Training

```python
# Keep Llama-3 template, longer training
args = DPOConfig(
    learning_rate = 5e-6,
    max_steps = 1000,  # ← Changed from 300
    # OR
    num_train_epochs = 1,
    max_steps = None,
)

RUN_NAME = "CITA_Baseline_Exp3_LongerTraining"
```

---

## 🎲 Change 5: Multi-Seed Wrapper Script

**New file:** `comparative_study/run_multi_seed.py`

```python
#!/usr/bin/env python3
"""
Multi-seed training automation
Runs each notebook 3x with different seeds for statistical validation
"""
import subprocess
import json
from pathlib import Path
import sys

SEEDS = [42, 123, 456]
NOTEBOOKS = [
    "01a_SFT_Baseline/Llama3_alignmentDB_BF16.ipynb",
    "01b_SFT_GRIT/Llama3_alignmentDB_BF16.ipynb",
    "02a_DPO_Baseline/Llama3_alignmentDB_BF16.ipynb",
    "02b_DPO_GRIT/Llama3_alignmentDB_BF16.ipynb",
    # Add others after validating top performers
]

def inject_seed(notebook_path: Path, seed: int) -> Path:
    """Create notebook copy with injected seed"""
    import nbformat

    nb = nbformat.read(notebook_path, as_version=4)

    # Find training args cell
    for cell in nb.cells:
        if 'TrainingArguments' in cell.source or 'SFTConfig' in cell.source or 'DPOConfig' in cell.source:
            # Inject seed
            cell.source = cell.source.replace(
                'seed = 3407,',
                f'seed = {seed},\n    data_seed = {seed},'
            )
            # Update output dir
            cell.source = cell.source.replace(
                'output_dir = ',
                f'output_dir = "./outputs/seed{seed}/"  # '
            )

    # Find HF_REPO cell
    for cell in nb.cells:
        if 'HF_REPO = MODEL_NAME_MAP' in cell.source:
            cell.source += f'\nHF_REPO += "-seed{seed}"  # Multi-seed run'

    # Save modified notebook
    output_path = notebook_path.parent / f"{notebook_path.stem}_seed{seed}.ipynb"
    nbformat.write(nb, output_path)
    return output_path

def run_notebook(notebook_path: Path):
    """Execute notebook via nbconvert"""
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        '--ExecutePreprocessor.timeout=36000',  # 10 hour timeout
        str(notebook_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def main():
    results_log = Path("outputs/multi_seed_results/execution_log.json")
    results_log.parent.mkdir(parents=True, exist_ok=True)

    execution_log = []

    for notebook in NOTEBOOKS:
        nb_path = Path(f"comparative_study/{notebook}")
        model_name = nb_path.parent.name

        print(f"\n{'='*80}")
        print(f"🚀 Running {model_name} with {len(SEEDS)} seeds")
        print(f"{'='*80}")

        for seed in SEEDS:
            print(f"\n  🎲 Seed {seed}...")

            # Create seed-specific notebook
            seed_nb = inject_seed(nb_path, seed)

            # Execute
            success = run_notebook(seed_nb)

            execution_log.append({
                "notebook": str(nb_path),
                "seed": seed,
                "success": success,
            })

            print(f"    {'✅ Success' if success else '❌ Failed'}")

            # Save log after each run
            with open(results_log, 'w') as f:
                json.dump(execution_log, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Multi-seed execution complete!")
    print(f"📊 Log: {results_log}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
cd /lambda/nfs/DiskUsEast1/finetuning_evaluation
source venv_aqi/bin/activate
python comparative_study/run_multi_seed.py
```

---

## 📊 Change 6: Update Inference Script for BF16 Models

**File:** `comparative_study/05_evaluation/inference_evaluation.py`

```python
# ✅ ADD: Support for HuggingFace-hosted models
MODELS = {
    "Baseline_NoFT": {"type": "base", "path": None},

    # BF16 models (from HuggingFace)
    "SFT_Baseline_BF16": {"type": "hf", "path": "kapilw/llama3-8b-pku-sft-baseline-bf16"},
    "SFT_GRIT_BF16": {"type": "hf", "path": "kapilw/llama3-8b-pku-sft-grit-bf16"},
    "DPO_Baseline_BF16": {"type": "hf", "path": "kapilw/llama3-8b-pku-dpo-baseline-bf16"},
    "DPO_GRIT_BF16": {"type": "hf", "path": "kapilw/llama3-8b-pku-dpo-grit-bf16"},
    "CITA_Baseline_BF16": {"type": "hf", "path": "kapilw/llama3-8b-pku-cita-baseline-bf16"},
    "CITA_GRIT_BF16": {"type": "hf", "path": "kapilw/llama3-8b-pku-cita-grit-bf16"},

    # 4-bit models (local, for comparison)
    "SFT_Baseline_4bit": {"type": "local", "path": BASE_DIR / "01a_SFT_Baseline/lora_model_SFT_Baseline"},
    # ... rest
}

def load_model_with_adapter(model_config):
    """Load model from HuggingFace or local path"""
    if model_config["type"] == "base":
        # Base model without adapter
        return load_base_model()

    elif model_config["type"] == "hf":
        # Load from HuggingFace
        print(f"  📦 Loading from HuggingFace: {model_config['path']}")
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_config['path'],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_config['path'], token=hf_token)

    elif model_config["type"] == "local":
        # Load local adapter (existing code)
        base_model = load_base_model()
        model = PeftModel.from_pretrained(base_model, str(model_config['path']))
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=hf_token)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer
```

---

## 📈 Change 7: Statistical Analysis Notebook

**New file:** `comparative_study/05_evaluation/statistical_analysis.ipynb`

```python
# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path

# Cell 2: Load multi-seed results
results_dir = Path("../../outputs/multi_seed_results")
all_data = []

for json_file in results_dir.glob("*_seed*.json"):
    with open(json_file) as f:
        all_data.append(json.load(f))

df = pd.DataFrame(all_data)
print(f"Loaded {len(df)} results from {df['run_name'].nunique()} models")

# Cell 3: Summary statistics
summary = df.groupby('run_name')['avg_overall_quality'].agg(['mean', 'std', 'count'])
summary['sem'] = summary['std'] / np.sqrt(summary['count'])
summary = summary.sort_values('mean', ascending=False)
print(summary)

# Cell 4: Pairwise t-tests
models = summary.index.tolist()
pvalues = pd.DataFrame(index=models, columns=models)

for i, model1 in enumerate(models):
    for j, model2 in enumerate(models):
        if i < j:
            scores1 = df[df['run_name'] == model1]['avg_overall_quality']
            scores2 = df[df['run_name'] == model2]['avg_overall_quality']
            t_stat, p_val = stats.ttest_ind(scores1, scores2)
            pvalues.loc[model1, model2] = p_val

            if p_val < 0.05:
                sig = '**' if p_val < 0.01 else '*'
                print(f"{model1} vs {model2}: p={p_val:.4f} {sig}")

# Cell 5: Paper-ready figure
fig, ax = plt.subplots(figsize=(14, 8))

groups = {
    'SFT': [m for m in models if 'SFT' in m],
    'DPO': [m for m in models if 'DPO' in m],
    'CITA': [m for m in models if 'CITA' in m],
    'Reference': [m for m in models if 'NoFT' in m],
}

x = 0
x_positions, x_labels = [], []
colors = {'Baseline': '#2E86AB', 'GRIT': '#A23B72', 'Reference': '#888'}

for group_name, model_list in groups.items():
    for model in model_list:
        if model not in summary.index:
            continue

        row = summary.loc[model]
        color = colors['GRIT'] if 'GRIT' in model else colors['Baseline'] if 'Baseline' in model else colors['Reference']

        ax.bar(x, row['mean'], yerr=row['sem'], color=color, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
        ax.text(x, row['mean'] + row['sem'] + 0.2, f"{row['mean']:.2f}", ha='center', fontweight='bold')

        # Significance stars
        best_model = summary.index[0]
        if model != best_model:
            p = pvalues.loc[best_model, model]
            if pd.notna(p):
                if p < 0.01:
                    ax.text(x, row['mean'] + row['sem'] + 0.5, '**', ha='center', fontsize=14)
                elif p < 0.05:
                    ax.text(x, row['mean'] + row['sem'] + 0.5, '*', ha='center', fontsize=14)

        x_positions.append(x)
        x_labels.append(model.replace('_', '\n'))
        x += 1
    x += 0.5

ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=0, ha='center')
ax.set_ylabel('Overall Quality Score (0-10)', fontsize=14, fontweight='bold')
ax.set_title('Inference Quality Comparison\n(Mean ± SEM, n=3 runs)', fontsize=16, fontweight='bold')
ax.set_ylim(0, 10)
ax.grid(axis='y', alpha=0.3)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=colors['Baseline'], label='Baseline'),
    Patch(facecolor=colors['GRIT'], label='GRIT'),
    Patch(facecolor=colors['Reference'], label='Reference'),
], loc='upper right')

ax.text(0.02, 0.98, '* p<0.05  ** p<0.01', transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('Inference_Quality_Results/paper_figure_with_stats.png', dpi=300, bbox_inches='tight')
plt.savefig('Inference_Quality_Results/paper_figure_with_stats.pdf', bbox_inches='tight')
plt.show()
```

---

## 📋 Execution Sequence

### Phase 1: BF16 Validation (Highest Priority)
```bash
# Create BF16 notebooks (copy + modify existing)
# Run top 2 performers first
cd /lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study

# 1. DPO_Baseline (current winner)
cd 02a_DPO_Baseline
jupyter nbconvert --execute --to notebook --inplace Llama3_alignmentDB_BF16.ipynb

# 2. SFT_Baseline (close second)
cd ../01a_SFT_Baseline
jupyter nbconvert --execute --to notebook --inplace Llama3_alignmentDB_BF16.ipynb
```

**Decision point:** If BF16 baselines maintain ~8/10 scores → proceed to GRIT BF16

### Phase 2: GRIT BF16 (Hypothesis Test)
```bash
# Test if quantization broke GRIT
cd 01b_SFT_GRIT
jupyter nbconvert --execute --to notebook --inplace Llama3_alignmentDB_BF16.ipynb

cd ../02b_DPO_GRIT
jupyter nbconvert --execute --to notebook --inplace Llama3_alignmentDB_BF16.ipynb
```

**Decision point:**
- If GRIT BF16 ≈ Baseline BF16 → quantization was the problem ✅
- If GRIT BF16 still fails → GRIT incompatible with preference learning ❌

### Phase 3: CITA Fixes (Parallel)
```bash
cd 03a_CITA_Baseline
jupyter nbconvert --execute --to notebook --inplace Llama3_alignmentDB_Exp1_AlpacaTemplate.ipynb
jupyter nbconvert --execute --to notebook --inplace Llama3_alignmentDB_Exp2_HigherLR.ipynb
jupyter nbconvert --execute --to notebook --inplace Llama3_alignmentDB_Exp3_LongerTraining.ipynb
```

### Phase 4: Multi-Seed (Best 2-3 Configs Only)
```bash
# After identifying winners, run 3x each
python comparative_study/run_multi_seed.py
```

### Phase 5: Statistical Analysis
```bash
cd comparative_study/05_evaluation
jupyter nbconvert --execute --to notebook --inplace statistical_analysis.ipynb
```

---

## 🎯 Success Criteria

### Quantization Hypothesis
- [ ] BF16 baselines maintain quality (7.5-8.0/10)
- [ ] GRIT BF16 ≥ GRIT 4-bit + 1.5 points
- [ ] If yes → publish "quantization breaks GRIT" finding

### CITA Viability
- [ ] At least 1 CITA experiment > 6.5/10
- [ ] If yes → include CITA in final comparison
- [ ] If no → exclude CITA, document why

### Statistical Rigor
- [ ] Variance estimates (SEM) < 0.5 for all models
- [ ] DPO vs SFT: p-value calculated
- [ ] GRIT vs Baseline: p-value < 0.05 (if claiming superiority)

---

## 📊 Expected Outcomes

### Scenario A: Quantization Broke GRIT
```
BF16 Results:
  DPO_GRIT_BF16:    7.95/10 (was 7.14)  → RECOVERED! ✅
  SFT_GRIT_BF16:    7.80/10 (was 7.37)  → RECOVERED! ✅

Conclusion: 4-bit quantization incompatible with K-FAC
Action: Paper section on "Quantization Considerations for Second-Order Optimizers"
```

### Scenario B: GRIT Fundamentally Flawed
```
BF16 Results:
  DPO_GRIT_BF16:    7.20/10 (was 7.14)  → NO CHANGE ❌
  SFT_GRIT_BF16:    7.40/10 (was 7.37)  → NO CHANGE ❌

Conclusion: GRIT doesn't help preference learning
Action: Negative result paper OR pivot to standard LoRA
```

### Scenario C: CITA Salvaged
```
CITA Experiments:
  Exp1 (Alpaca):      6.8/10  → VIABLE ✅
  Exp2 (Higher LR):   6.5/10  → VIABLE ✅
  Exp3 (Longer):      7.0/10  → COMPETITIVE! ✅

Conclusion: Template mismatch was the problem
Action: Include CITA in final comparison
```

---

## ⚠️ Risk Mitigation

### GPU OOM (BF16 uses 2x memory)
- Batch size reduced 2→1
- Gradient accumulation 4→8
- If still OOM: reduce max_seq_length 2048→1536

### Multi-seed failures
- Log failures to `execution_log.json`
- Continue to next seed on failure
- Manual inspection of failed notebooks

### HuggingFace API limits
- Rate limit: 1 upload/10s → add `time.sleep(10)` between pushes
- Storage limit: Delete old checkpoints after successful upload

---

## 📝 Deliverables

1. **12 New BF16 Notebooks** (6 baseline + 6 GRIT)
2. **3 CITA Experiment Notebooks**
3. **Multi-seed automation script** (`run_multi_seed.py`)
4. **Statistical analysis notebook** with publication figure
5. **Updated inference script** supporting HF-hosted models
6. **18 HuggingFace model repos** (6 models × 3 seeds)

---

## 🚀 Next Steps

1. **User reviews this plan** → approve/revise
2. **Claude generates BF16 notebooks** (automated with Edit tool)
3. **User runs Phase 1** (DPO + SFT baselines, ~6 hours)
4. **Review Phase 1 results** → decide on Phase 2
5. **Complete remaining phases** based on findings

**Estimated completion:** 48-72 GPU hours over 1 week
