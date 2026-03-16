# Controlled Experiment: Instruction Alignment Impact

**Goal**: Isolate the benefit of instruction alignment across entire training pipeline
**Design**: Compare two complete pipelines (SFT → DPO → CITA) with and without instructions

---

## Experimental Setup

### Pipeline 1: No Instructions (Baseline)
```
Stage 1: SFT_NoInstruct      → (prompt, response) only
Stage 2: DPO_NoInstruct      → Stack on SFT_NoInstruct, (prompt, chosen, rejected)
Stage 3: CITA_NoInstruct     → Stack on DPO_NoInstruct, (prompt, chosen, rejected)
```

**Input format** (all stages):
```python
messages = [
    {"role": "user", "content": prompt}
]
```

**Hypothesis**: CITA_NoInstruct will be unstable (matches current iter6 run)

---

### Pipeline 2: With Instructions (Test)
```
Stage 1: SFT_Instructed      → (instruction, prompt, response)
Stage 2: DPO_Instructed      → Stack on SFT_Instructed, (instruction, prompt, chosen, rejected)
Stage 3: CITA_Instructed     → Stack on DPO_Instructed, (instruction, prompt, chosen, rejected)
```

**Input format** (all stages):
```python
instruction = synthesize_system_instruction(harm_categories)
messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": prompt}
]
```

**Hypothesis**: CITA_Instructed will be stable (matches iter4 run)

---

## Metrics to Compare

### Training Stability
| Metric | Pipeline 1 (No Instruct) | Pipeline 2 (With Instruct) | Expected Difference |
|--------|-------------------------|---------------------------|---------------------|
| **CITA gradient norm** | High (>20, 60% warnings) | Low (<15, <10% warnings) | ~4x reduction |
| **CITA margin divergence** | Explodes (8 → 180) | Stable (6 → 7) | 26x difference |
| **CITA loss divergence** | Explodes (0.3 → 6.8) | Stable (0.7 → 0.3) | 20x difference |

### Final Performance (on toxicity eval)
| Model | Pipeline 1 | Pipeline 2 | Expected Winner |
|-------|-----------|-----------|-----------------|
| **SFT** | Toxicity ~2.5 | Toxicity ~2.3 | Pipeline 2 (instruction-aware refusal) |
| **DPO** | Toxicity ~1.8 | Toxicity ~1.6 | Pipeline 2 (aligned preferences) |
| **CITA** | Toxicity ~3.5 (collapsed) | Toxicity ~1.4 | Pipeline 2 (stable training) |

**Key question**: Does instruction alignment improve final toxicity scores, or just training stability?

---

## Implementation Plan

### Phase 1: Pipeline 1 (No Instructions) - Status

**Current status**:
- ✅ SFT_NoInstruct: Already trained (`kapilw25/llama3-8b-pku-sft-baseline-bf16`)
- ✅ DPO_NoInstruct: Already trained (`kapilw25/llama3-8b-pku-dpo-sft-bf16`)
- ❌ CITA_NoInstruct: Failed (unstable, current iter6 run)

**Action**: Retrain CITA_NoInstruct with gradient clipping (Option 2 fix)

```python
# comparative_study/03a_CITA_Baseline/Llama3_BF16.py
# Add gradient clipping to stabilize (doesn't fix root cause, but allows completion)

training_args = DPOConfig(
    ...
    max_grad_norm=1.0,  # Clip gradients
    learning_rate=5e-6,  # Lower LR (half of 1.185e-5)
    ...
)
```

**Time**: ~2 hours (1.0 epoch)

---

### Phase 2: Pipeline 2 (With Instructions) - New Training

**Status**: Need to train all 3 stages

#### Step 1: SFT_Instructed
```bash
# Update formatter to add instructions
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full
```

**Code change** (`comparative_study/0c_utils/data_prep/formatters.py`):
```python
def format_pku_for_sft(example):
    """Format for SFT WITH instructions"""
    from loader_pku import synthesize_system_instruction

    prompt = example['prompt']
    chosen = example['response_0' if example['is_response_0_safe'] else 'response_1']

    # Extract harm categories
    harm_cat = example['response_0_harm_category']
    harm_categories = [k for k, v in harm_cat.items() if v]

    # Add instruction (KEY CHANGE)
    instruction = synthesize_system_instruction(harm_categories)

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen}
    ]

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

**Time**: ~100 min

---

#### Step 2: DPO_Instructed
```bash
# Stack on SFT_Instructed
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full \
    --base_model kapilw25/llama3-8b-pku-sft-instructed-bf16
```

**Code change** (`formatters.py`):
```python
def format_pku_for_dpo(example):
    """Format for DPO WITH instructions"""
    from loader_pku import synthesize_system_instruction

    prompt = example['prompt']

    # Extract harm categories
    harm_cat = example['response_0_harm_category']
    harm_categories = [k for k, v in harm_cat.items() if v]

    # Add instruction (KEY CHANGE)
    instruction = synthesize_system_instruction(harm_categories)

    if example['is_response_0_safe']:
        chosen, rejected = example['response_0'], example['response_1']
    else:
        chosen, rejected = example['response_1'], example['response_0']

    prompt_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]

    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True),
        "chosen": chosen,
        "rejected": rejected
    }
```

**Time**: ~120 min

---

#### Step 3: CITA_Instructed
```bash
# Stack on DPO_Instructed (no code changes needed - already uses instructions)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full \
    --base_model kapilw25/llama3-8b-pku-dpo-instructed-bf16
```

**Time**: ~120 min

**Total Pipeline 2 time**: ~6 hours

---

## Naming Convention

### HuggingFace Repositories

**Pipeline 1 (No Instructions)**:
```
kapilw25/llama3-8b-pku-sft-baseline-bf16           (existing)
kapilw25/llama3-8b-pku-dpo-baseline-bf16           (existing)
kapilw25/llama3-8b-pku-cita-baseline-bf16          (retrain with clipping)
```

**Pipeline 2 (With Instructions)**:
```
kapilw25/llama3-8b-pku-sft-instructed-bf16         (new)
kapilw25/llama3-8b-pku-dpo-instructed-bf16         (new)
kapilw25/llama3-8b-pku-cita-instructed-bf16        (new)
```

### Local Output Directories

**Pipeline 1**:
```
outputs/SFT_Baseline/         (existing)
outputs/DPO_Baseline/         (existing)
outputs/CITA_Baseline/        (retrain)
```

**Pipeline 2**:
```
outputs/SFT_Instructed/       (new)
outputs/DPO_Instructed/       (new)
outputs/CITA_Instructed/      (new)
```

---

## Evaluation Plan

### Toxicity Evaluation (Both Pipelines)

**Models to evaluate**:
```python
MODELS = {
    # Baseline (unaligned)
    "Baseline": {
        "hf_repo": None,
        "display_name": "Baseline (Unaligned)",
    },

    # Pipeline 1: No Instructions
    "SFT_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-sft-baseline-bf16",
        "display_name": "SFT (No Instructions)",
    },
    "DPO_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-dpo-baseline-bf16",
        "display_name": "DPO (No Instructions)",
    },
    "CITA_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-cita-baseline-bf16",
        "display_name": "CITA (No Instructions)",
    },

    # Pipeline 2: With Instructions
    "SFT_Instructed": {
        "hf_repo": "kapilw25/llama3-8b-pku-sft-instructed-bf16",
        "display_name": "SFT (Instructed)",
    },
    "DPO_Instructed": {
        "hf_repo": "kapilw25/llama3-8b-pku-dpo-instructed-bf16",
        "display_name": "DPO (Instructed)",
    },
    "CITA_Instructed": {
        "hf_repo": "kapilw25/llama3-8b-pku-cita-instructed-bf16",
        "display_name": "CITA (Instructed)",
    },
}
```

**Inference logic** (`toxicity.py`):
```python
# During inference, CITA models receive instructions (eval-time conditioning)
for p, harm_cats in zip(batch_prompts, batch_harm_cats):
    if "CITA" in model_key:
        instruction = synthesize_system_instruction(harm_cats)
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": p}
        ]
    else:
        # SFT/DPO: No instruction during eval (matches training)
        messages = [{"role": "user", "content": p}]
```

**Expected results**:

| Model | Mean Toxicity | Safe Refusal Rate | Notes |
|-------|---------------|-------------------|-------|
| **Baseline** | ~3.5 | ~45% | No alignment |
| **SFT_NoInstruct** | ~2.8 | ~60% | Basic alignment |
| **DPO_NoInstruct** | ~1.8 | ~85% | Preference alignment |
| **CITA_NoInstruct** | ~3.0 | ~55% | Training collapsed (worse than DPO) |
| **SFT_Instructed** | ~2.5 | ~65% | Instruction-aware refusal |
| **DPO_Instructed** | ~1.6 | ~88% | Instruction + preference |
| **CITA_Instructed** | ~1.4 | ~95% | Stable CITA (best) |

**Key comparison**:
- **CITA_NoInstruct vs CITA_Instructed**: Shows instruction alignment impact
- **SFT/DPO NoInstruct vs Instructed**: Shows instruction benefit at each stage

---

## Timeline

### Sequential Execution (Recommended)
```
Phase 1: CITA_NoInstruct (retrain with clipping)     ~2 hours
Phase 2a: SFT_Instructed                             ~2 hours
Phase 2b: DPO_Instructed (depends on 2a)             ~2 hours
Phase 2c: CITA_Instructed (depends on 2b)            ~2 hours
Evaluation: Toxicity eval (7 models)                 ~10 hours
-----------------------------------------------------------
Total: ~18 hours
```

### Parallel Execution (Faster, requires coordination)
```
Phase 1: CITA_NoInstruct (GPU 1)                     ~2 hours
Phase 2a: SFT_Instructed (GPU 2)                     ~2 hours
  Phase 2b: DPO_Instructed (GPU 2, after 2a)         ~2 hours
    Phase 2c: CITA_Instructed (GPU 2, after 2b)      ~2 hours
-----------------------------------------------------------
Overlap: ~6 hours (if 2 GPUs available)
Evaluation: ~10 hours
Total: ~16 hours
```

---

## Code Changes Required

### 1. Create Instruction Toggle in Formatters

```python
# comparative_study/0c_utils/data_prep/formatters.py

USE_INSTRUCTIONS = True  # ← Toggle for entire pipeline

def format_pku_for_sft(example):
    """Format for SFT with optional instructions"""
    from loader_pku import synthesize_system_instruction

    prompt = example['prompt']
    chosen = example['response_0' if example['is_response_0_safe'] else 'response_1']

    if USE_INSTRUCTIONS:
        # Extract harm categories and add instruction
        harm_cat = example['response_0_harm_category']
        harm_categories = [k for k, v in harm_cat.items() if v]
        instruction = synthesize_system_instruction(harm_categories)

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
    else:
        # No instruction (baseline)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def format_pku_for_dpo(example):
    """Format for DPO with optional instructions"""
    from loader_pku import synthesize_system_instruction

    prompt = example['prompt']

    if example['is_response_0_safe']:
        chosen, rejected = example['response_0'], example['response_1']
    else:
        chosen, rejected = example['response_1'], example['response_0']

    if USE_INSTRUCTIONS:
        harm_cat = example['response_0_harm_category']
        harm_categories = [k for k, v in harm_cat.items() if v]
        instruction = synthesize_system_instruction(harm_categories)

        prompt_messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]
    else:
        prompt_messages = [
            {"role": "user", "content": prompt}
        ]

    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True),
        "chosen": chosen,
        "rejected": rejected
    }

# format_pku_for_cita already uses instructions (no change needed)
```

### 2. Update Model Utils for Naming

```python
# comparative_study/0c_utils/model_utils.py

def get_model_repo_name(run_name: str, precision: str = "bf16", instructed: bool = False) -> str:
    """Get HuggingFace repo name with instruction variant support"""

    base_names = {
        "SFT_Baseline": "llama3-8b-pku-sft",
        "DPO_Baseline": "llama3-8b-pku-dpo",
        "CITA_Baseline": "llama3-8b-pku-cita-dpo",
    }

    base = base_names.get(run_name, run_name.lower())

    # Add instruction variant suffix
    if instructed:
        suffix = "instructed"
    else:
        suffix = "baseline"

    return f"kapilw25/{base}-{suffix}-{precision}"
```

### 3. Update Training Scripts to Accept --instructed Flag

```python
# comparative_study/01a_SFT_Baseline/Llama3_BF16.py (same for DPO, CITA)

parser.add_argument(
    "--instructed",
    action="store_true",
    help="Use instructed formatters (adds system instructions)"
)

args = parser.parse_args()

# Set global toggle
import comparative_study.0c_utils.data_prep.formatters as formatters
formatters.USE_INSTRUCTIONS = args.instructed

# Update HF repo name
HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16", instructed=args.instructed)
```

---

## Execution Commands

### Pipeline 1: Complete CITA_NoInstruct
```bash
# Retrain CITA with gradient clipping (fix current unstable run)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full \
    --base_model kapilw25/llama3-8b-pku-dpo-baseline-bf16
```

### Pipeline 2: Train All Instructed Models
```bash
# Step 1: SFT_Instructed
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full --instructed

# Step 2: DPO_Instructed (stacked on SFT_Instructed)
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full --instructed \
    --base_model kapilw25/llama3-8b-pku-sft-instructed-bf16

# Step 3: CITA_Instructed (stacked on DPO_Instructed)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full --instructed \
    --base_model kapilw25/llama3-8b-pku-dpo-instructed-bf16
```

### Evaluation: Compare All 7 Models
```bash
# Run toxicity evaluation on both pipelines
python comparative_study/05_evaluation/llm_as_judge/toxicity.py --mode full
```

---

## Expected Outcomes

### Training Stability
**CITA_NoInstruct** (with gradient clipping):
- May complete training without crash
- Metrics likely suboptimal (worse than DPO_NoInstruct)
- Demonstrates instruction mismatch problem

**CITA_Instructed**:
- Smooth, stable training (matches iter4)
- Outperforms DPO_Instructed (+18% margin improvement)
- Validates instruction alignment hypothesis

### Toxicity Evaluation
**Best model**: CITA_Instructed (lowest toxicity, highest refusal rate)
**Worst model**: CITA_NoInstruct (training collapse → poor safety)

**Key insight**: Instruction alignment benefits appear at **every stage**:
- SFT_Instructed > SFT_NoInstruct
- DPO_Instructed > DPO_NoInstruct
- CITA_Instructed >> CITA_NoInstruct (largest gap)

---

## Research Questions Answered

1. **Does instruction alignment improve training stability?**
   - Yes: CITA_Instructed stable, CITA_NoInstruct collapses

2. **Does instruction alignment improve final toxicity scores?**
   - Yes: Compare SFT/DPO/CITA instructed vs non-instructed variants

3. **Is instruction alignment necessary for CITA, or just helpful?**
   - Necessary: CITA_NoInstruct fails catastrophically

4. **Do instructions help earlier stages (SFT/DPO)?**
   - Measure: SFT_Instructed vs SFT_NoInstruct toxicity scores

5. **Is CITA's benefit from KL regularization or instruction alignment?**
   - If CITA_NoInstruct < DPO_NoInstruct → alignment is key
   - If CITA_Instructed > DPO_Instructed → KL regularization also helps

---

## Next Steps

1. ⏳ **Approve experiment design**
2. ⏳ **Phase 1**: Retrain CITA_NoInstruct with gradient clipping (~2 hours)
3. ⏳ **Phase 2**: Train SFT/DPO/CITA_Instructed pipeline (~6 hours)
4. ⏳ **Evaluation**: Run toxicity eval on all 7 models (~10 hours)
5. ⏳ **Analysis**: Compare training stability and final toxicity scores
6. ⏳ **Paper**: Update 2025_Ecliptica.pdf with controlled experiment results
