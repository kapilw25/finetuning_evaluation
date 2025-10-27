# Training Plan: Dataset Migration (PKU → Vaibhaav)

**Date**: 2025-10-26
**Goal**: Switch training dataset to Vaibhaav (natural language instructions)
**Scope**: Phases 1-3 (Code changes, dataset loaders, training)

**Next Step After Completion:** See `next_steps_eval_toxicity.md` for evaluation setup

---

## Problem Summary

### Proposal Claims (2025_Ecliptica.pdf)
- **SFT/DPO**: Train on `(x, y)` pairs **WITHOUT** explicit instructions
- **CITA Innovation**: Introduces explicit instruction conditioning `(I, x, y)`
- **Vision**: "Dynamic alignment correction via **natural language instructions**" (NOT generic categories)

### Current Implementation Issues
❌ **ALL methods** (SFT, DPO, CITA) use metadata-extracted instructions
❌ No test of "instruction vs no-instruction" hypothesis
❌ Instructions are **generic category-based** (NOT natural language alignment directives)
❌ Dataset: PKU-SafeRLHF has limited metadata (only 19 harm categories)

### Solution: Dataset Change
✅ **Training**: Switch to **Vaibhaav/alignment-instructions** (50K samples)
   - Custom natural language instructions per prompt
   - Matches proposal's "dialogue-based alignment correction"

---

## Code Changes Required

### Phase 1: Add New Dataset Loaders

#### File: `comparative_study/0c_utils/data_prep/vaibhaav_loader.py` (NEW)

```python
from datasets import load_dataset
from typing import Dict

def load_vaibhaav_alignment(split: str = "train", max_samples: int = None):
    """Load Vaibhaav/alignment-instructions dataset"""
    dataset = load_dataset("Vaibhaav/alignment-instructions", split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"✅ Loaded {len(dataset)} samples from Vaibhaav/alignment-instructions")
    return dataset

def format_vaibhaav_for_sft(example: Dict) -> Dict:
    """SFT: NO instruction (baseline)"""
    messages = [
        {"role": "user", "content": example["Prompt"]},
        {"role": "assistant", "content": example["Accepted Response"]}
    ]
    return {"messages": messages}

def format_vaibhaav_for_dpo(example: Dict) -> Dict:
    """DPO: NO instruction (baseline)"""
    return {
        "prompt": [{"role": "user", "content": example["Prompt"]}],
        "chosen": [{"role": "assistant", "content": example["Accepted Response"]}],
        "rejected": [{"role": "assistant", "content": example["Rejected Response"]}]
    }

def format_vaibhaav_for_cita(example: Dict) -> Dict:
    """CITA: WITH natural language instruction (proposal innovation)"""
    instruction = example["Instruction generated"]  # Custom per-prompt!

    return {
        "chosen": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": example["Prompt"]},
            {"role": "assistant", "content": example["Accepted Response"]}
        ],
        "rejected": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": example["Prompt"]},
            {"role": "assistant", "content": example["Rejected Response"]}
        ]
    }
```

---

### Phase 2: Update Existing Formatters

#### File: `comparative_study/0c_utils/data_prep/formatters.py`

**Update SFT formatter** (remove instruction):
```python
def format_sft(example: Dict) -> Dict:
    """SFT: NO instruction (baseline)"""
    safe_response, _, _ = get_safe_unsafe_responses(example)

    messages = [
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": safe_response}
    ]
    return {"messages": messages}
```

**Update DPO formatter** (remove instruction):
```python
def format_dpo(example: Dict) -> Dict:
    """DPO: NO instruction (baseline)"""
    safe_response, unsafe_response, _ = get_safe_unsafe_responses(example)

    return {
        "prompt": [{"role": "user", "content": example['prompt']}],
        "chosen": [{"role": "assistant", "content": safe_response}],
        "rejected": [{"role": "assistant", "content": unsafe_response}]
    }
```

**Update CITA formatter** (use PKU metadata for backward compatibility):
```python
def format_cita(example: Dict) -> Dict:
    """CITA: WITH instruction (PKU metadata-based for evaluation)"""
    safe_response, unsafe_response, harmful_categories = get_safe_unsafe_responses(example)
    instruction = synthesize_system_instruction(harmful_categories)

    return {
        "chosen": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": safe_response}
        ],
        "rejected": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": unsafe_response}
        ]
    }
```

---

### Phase 3: Update Training Scripts (Dataset Source)

#### File: `comparative_study/01a_SFT_Baseline/Llama3_BF16.py`

**Current (Line 189-195):**
```python
train_dataset = load_training_dataset(
    split="train",
    max_samples=None,
    method="sft",  # Uses PKU-SafeRLHF
    return_val=False
)
```

**New:**
```python
from vaibhaav_loader import load_vaibhaav_alignment, format_vaibhaav_for_sft

dataset_raw = load_vaibhaav_alignment(split="train")
train_dataset = dataset_raw.map(format_vaibhaav_for_sft, remove_columns=dataset_raw.column_names)
```

---

#### File: `comparative_study/02a_DPO_Baseline/Llama3_BF16.py`

**Current (Line 218-224):**
```python
train_dataset = load_training_dataset(
    split="train",
    max_samples=None,
    method="dpo",  # Uses PKU-SafeRLHF
    return_val=False
)
```

**New:**
```python
from vaibhaav_loader import load_vaibhaav_alignment, format_vaibhaav_for_dpo

dataset_raw = load_vaibhaav_alignment(split="train")
train_dataset = dataset_raw.map(format_vaibhaav_for_dpo, remove_columns=dataset_raw.column_names)
```

---

#### File: `comparative_study/03a_CITA_Baseline/Llama3_BF16.py`

**Current (Line 219-232):**
```python
from data_prep import load_pku_filtered, format_dataset

dataset_raw_train = load_pku_filtered(split="train", ...)
train_dataset = format_dataset(dataset_raw_train, method="dpo")  # ← WRONG
```

**New:**
```python
from vaibhaav_loader import load_vaibhaav_alignment, format_vaibhaav_for_cita

dataset_raw = load_vaibhaav_alignment(split="train")
train_dataset = dataset_raw.map(format_vaibhaav_for_cita, remove_columns=dataset_raw.column_names)
```

**Critical**: CITA now gets **custom natural language instructions** (not generic categories)

---

### Phase 4: Re-train All Models

#### Training Commands (Sequential):

```bash
# 1. SFT Baseline (NO instruction, Vaibhaav dataset)
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full
# Dataset: Vaibhaav/alignment-instructions (50K samples)
# Expected: Lower safety (no instruction), learns from accepted responses

# 2. DPO Baseline (NO instruction, Vaibhaav dataset)
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py \
    --mode full \
    --base_model kapilw25/llama3-8b-pku-sft-baseline-bf16
# Dataset: Vaibhaav/alignment-instructions (50K samples)
# Expected: Higher safety (preference learning), no instruction awareness

# 3. CITA Baseline (WITH natural language instructions, Vaibhaav dataset)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py \
    --mode full \
    --base_model kapilw25/llama3-8b-pku-dpo-sft-bf16
# Dataset: Vaibhaav/alignment-instructions (50K custom instructions!)
# Expected: Highest safety (preferences + natural language instructions + KL regularization)
```

#### Expected Training Results:

| Model | Dataset | Instruction Type | Training Samples | Expected Margin |
|-------|---------|------------------|------------------|-----------------|
| **SFT** | Vaibhaav | ❌ None | 50,001 | ~1.5 |
| **DPO** | Vaibhaav | ❌ None | 50,001 | ~2.95 |
| **CITA** | Vaibhaav | ✅ **Natural language** | 50,001 | ~5.0+ (boost from custom instructions) |

**Key Difference from PKU Training**:
- **Old**: Generic category-based instructions ("Refuse cybercrime, violence")
- **New**: Custom natural language directives ("Focus on discussing ethical considerations, legal consequences, and avoiding providing methods")

---

## Implementation Checklist

### Phase 1-3: Code Changes & Testing
- [✅] Create `loader_vaibhaav.py` (new dataset loader)
- [✅] Update `formatters.py` (remove instructions from SFT/DPO)
- [✅] Update training scripts (switch to Vaibhaav dataset)
- [⏳] Update eval scripts (instruction extraction for CITA)
- [✅] Test Vaibhaav formatters (5 samples each method)

### Phase 4: Re-training (Vaibhaav Dataset)
- [⏳] SFT Baseline (50K samples, NO instruction)
- [⏳] DPO Baseline (50K samples, NO instruction)
- [⏳] CITA Baseline (50K samples, WITH natural language instructions)
- [⏳] Push to HF repos (overwrite old models)

### Phase 5: Evaluation
→ **See `next_steps_eval_toxicity.md` for evaluation setup**

---

## Validation Checklist

### Test Vaibhaav Formatters:

```python
from vaibhaav_loader import load_vaibhaav_alignment
from vaibhaav_loader import format_vaibhaav_for_sft, format_vaibhaav_for_dpo, format_vaibhaav_for_cita

dataset = load_vaibhaav_alignment(split="train")

# Test SFT formatter (NO instruction)
sft_sample = format_vaibhaav_for_sft(dataset[0])
print("SFT:", sft_sample)
# Expected: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

# Test DPO formatter (NO instruction)
dpo_sample = format_vaibhaav_for_dpo(dataset[0])
print("DPO:", dpo_sample)
# Expected: {"prompt": [{"role": "user", ...}], "chosen": [...], "rejected": [...]}

# Test CITA formatter (WITH natural language instruction)
cita_sample = format_vaibhaav_for_cita(dataset[0])
print("CITA:", cita_sample)
# Expected: {"chosen": [{"role": "system", "content": "When responding to..."}, ...]}
# Instruction should be custom per-prompt (not generic categories!)
```

### Compare Training Datasets:

```bash
python3 -c "
from datasets import load_dataset

# Old: PKU-SafeRLHF
pku = load_dataset('PKU-Alignment/PKU-SafeRLHF', split='train')
print(f'PKU samples: {len(pku):,}')
print(f'PKU filtered: ~10,813 (clear safety contrast)')

# New: Vaibhaav
vaibhaav = load_dataset('Vaibhaav/alignment-instructions', split='train')
print(f'Vaibhaav samples: {len(vaibhaav):,}')
print(f'Sample instruction: {vaibhaav[0][\"Instruction generated\"][:100]}...')
"
```

---

## Dataset Comparison

| Aspect | PKU-SafeRLHF (Old) | Vaibhaav (New) |
|--------|-------------------|----------------|
| **Training samples** | 10,813 | 50,001 |
| **Instruction type** | Generic categories | Natural language |
| **Matches proposal** | ⚠️ Partial | ✅ **Full match** |
