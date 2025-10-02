# Plan 3: Minimal GRIT Integration with Unsloth

**Status:** ⚠️ Phase 3 (Hybrid) FAILED - Moving to Phase 4b (Pure GRIT)
**Created:** 2025-10-01
**Updated:** 2025-10-02
**Objective:** Inject GRIT (Geometry-Aligned Instruction Tuning) into Unsloth's Llama3 fine-tuning pipeline with minimal code changes

---

## ⚠️ CRITICAL UPDATE: Hybrid GRIT Results (2025-10-02)

### Experimental Results: 200-Step Training

**Configuration Tested:**
- **Baseline:** `unsloth/Llama3_(8B)_Ollama.ipynb` with AdamW optimizer (200 steps)
- **GRIT:** `unsloth_2_Hybrid_Grit/Llama3_8B_GRIT.ipynb` with aggressive GRIT settings (200 steps)

**GRIT Settings (Aggressive):**
```python
grit_config.kfac_update_freq = 5         # Fisher inversions every 5 steps
grit_config.reprojection_freq = 40       # Reprojection every 40 steps
grit_config.kfac_damping = 1e-5          # Low damping for strong effect
grit_config.lambda_kfac = 1e-6           # Minimal regularization
grit_config.kfac_min_samples = 4         # Fast readiness threshold
```

**Verification Logs Confirmed:**
- ✅ Fisher inversions: 40 inversions (every 5 steps as configured)
- ✅ Gradient preconditioning: Applied 195/200 steps (97.5% coverage)
- ✅ Callbacks executing correctly
- ✅ Natural gradients computed and applied

### 🚨 CRITICAL FINDING: Perfect Loss Curve Synchronization

**Observation from TensorBoard (`logs/tensorboard_log5.png`, `log6.png`, `log7.png`):**

The loss curves for **GRIT and Baseline are PERFECTLY IDENTICAL** across all 200 steps:
- Every local minimum matches exactly
- Every local maximum matches exactly
- Every spike and dip synchronized
- Final loss: GRIT=1.045 vs Baseline=1.0439 (0.1% difference)

**This is NOT normal behavior.** Even with similar convergence, different optimizers should show:
- Different noise patterns
- Asynchronous local minima
- Different oscillation phases

### Root Cause Analysis

**Hypothesis: Unsloth's Internal Optimizations Override GRIT Gradients**

Despite GRIT preconditioning being **mechanically functional**, the loss curves suggest:

1. **GRIT modifies gradients** → Verified by logs showing preconditioning applied
2. **BUT Unsloth may recompute/restore gradients** → After callback returns
3. **Result: Preconditioned gradients are discarded** → Optimizer uses original gradients

**Evidence:**
- `FastLanguageModel.get_peft_model()` applies custom gradient checkpointing
- `use_gradient_checkpointing = "unsloth"` may restore gradients from checkpoints
- SFTTrainer callback happens AFTER gradient computation but BEFORE optimizer step
- Unsloth's internal optimizations may reload saved gradients

**Conclusion:** The callback-based hybrid approach is **fundamentally incompatible** with Unsloth's gradient management.

### Decision: Move to Phase 4b (Pure GRIT)

**Rationale:**
1. Hybrid approach shows **zero observable benefit** despite working mechanics
2. Need to test if GRIT itself is effective (separate from Unsloth interference)
3. Pure GRIT uses native `GRITTrainer` with built-in gradient control
4. Scientific rigor requires isolating GRIT from Unsloth's black box

**Next Steps:**
- ~~Phase 3: Hybrid GRIT~~ ❌ **FAILED - Identical curves**
- **Phase 4b: Pure GRIT** → Proceeding with standalone implementation
- Compare Pure GRIT vs Baseline to determine if GRIT provides any benefit

---

## 📁 Project Structure

```
finetuning_evaluation/
├── unsloth/                          # ✅ REFERENCE (DO NOT MODIFY - successful baseline)
│   ├── Llama3_(8B)_Ollama.ipynb      # ✅ Working Unsloth notebook
│   ├── lora_model/                   # ✅ Unsloth fine-tuned LoRA adapters
│   ├── model/                        # ✅ Unsloth GGUF export
│   ├── outputs/                      # ✅ Training logs
│   └── venv_unsloth/                 # ✅ Working environment
│
├── unsloth_2_Hybrid_Grit/            # 🆕 Hybrid: Unsloth + GRIT callback
│   ├── Llama3_(8B)_GRIT.ipynb        # 🆕 GRIT-enhanced notebook (copy + modify)
│   ├── grit_repo/                    # 🆕 Cloned GRIT repository
│   ├── lora_model/                   # 🆕 Hybrid GRIT fine-tuned LoRA adapters
│   ├── model/                        # 🆕 Hybrid GRIT GGUF export for Ollama
│   ├── outputs/                      # 🆕 Hybrid GRIT training logs
│   └── venv_grit/                    # 🆕 Separate venv (optional, can reuse venv_unsloth)
│
├── unsloth_3_Pure_Grit/              # 🆕 Pure GRIT: No Unsloth dependencies
│   ├── train_aqi_grit.py             # 🆕 Pure GRIT training script
│   ├── grit_repo/                    # 🆕 Cloned GRIT repository
│   ├── lora_model/                   # 🆕 Pure GRIT LoRA adapters
│   ├── model/                        # 🆕 Pure GRIT GGUF export
│   └── outputs/                      # 🆕 Pure GRIT training logs
│
├── src/                              # 🆕 Evaluation scripts (compare all 4 models)
│   ├── m01_evaluate_base.py          # Evaluate base model (no fine-tuning)
│   ├── m02_evaluate_unsloth_ft.py    # Evaluate unsloth/lora_model via Ollama
│   ├── m03_evaluate_hybrid_grit.py   # Evaluate unsloth_2_Hybrid_Grit/lora_model via Ollama
│   ├── m04_evaluate_pure_grit.py     # 🆕 Evaluate unsloth_3_Pure_Grit/lora_model via Ollama
│   └── m05_compare_all_models.py     # 🆕 Compare all 4 & calculate delta_AQI
│
└── outputs/
    └── centralized.db                # All evaluation results (single source of truth)
```

### Rationale for Separation

✅ **Keep `unsloth/` as reference:**
- Proven working end-to-end pipeline
- Successful training checkpoint
- Working Ollama deployment (`ollama run unsloth_model`)
- Fallback if GRIT experiments fail

🆕 **Create `unsloth_2_Grit/` for hybrid experiments:**
- Isolated GRIT modifications
- Safe to fail/iterate without breaking baseline
- Independent GGUF export (`ollama run grit_model`)
- Easy A/B comparison via separate Ollama endpoints

🆕 **Create `unsloth_3_Pure_Grit/` for pure GRIT:**
- Full GRIT native implementation (no Unsloth dependencies)
- Uses GRIT's trainer, manager, and optimizer directly
- Research-grade implementation for academic comparison
- Independent from Unsloth optimizations

---

## ~~📋 Phase 3: Hybrid GRIT Implementation~~ ❌ DEPRECATED

**Status:** ❌ Abandoned - Callback approach incompatible with Unsloth
**Reason:** Perfect loss curve synchronization despite verified preconditioning
**See:** Experimental results section above for detailed analysis

This section is kept for historical reference only. Proceeding directly to Phase 4b (Pure GRIT).
```

**Changes:** ✅ Added 2 lines | ❌ Removed 0 lines

---

### Change 8: Saving & GGUF Export (NO CHANGES)

```python
# ✅ KEEP: All Unsloth saving code unchanged
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

if True: model.save_pretrained_gguf("model", tokenizer)
```

**Changes:** ✅ 0 lines | ❌ 0 lines (IDENTICAL!)

---

### Change 9: Ollama Deployment (NO CHANGES)

```python
# ✅ KEEP: All Ollama code unchanged
!curl -fsSL https://ollama.com/install.sh | sh
!ollama create unsloth_model -f ./model/Modelfile
!ollama run unsloth_model
```

**Changes:** ✅ 0 lines | ❌ 0 lines (IDENTICAL!)

---

## 📊 Total Changes Summary

| Component       | Lines Added | Lines Removed | % Changed |
|-----------------|-------------|---------------|-----------|
| Installation    | 3           | 0             | 5%        |
| Model loading   | 0           | 0             | 0%        |
| LoRA setup      | 18          | 0             | 30%       |
| Data prep       | 0           | 0             | 0%        |
| Trainer         | 3           | 0             | 5%        |
| Training        | 0           | 0             | 0%        |
| Inference       | 2           | 0             | 3%        |
| Saving          | 0           | 0             | 0%        |
| Ollama          | 0           | 0             | 0%        |
| **TOTAL**       | **26**      | **0**         | **~8%**   |

---

## 🎯 Key Advantages

### ✅ Pros

1. **Minimal code changes** (26 lines added, 0 removed)
2. **Keep all Unsloth optimizations** (2x faster inference, memory efficiency)
3. **Non-destructive**: Can toggle GRIT on/off with 1 line
4. **Compatible with Unsloth GGUF export**
5. **Easy A/B testing**: Train with/without GRIT callback

### ⚠️ Limitations

1. **Must verify callback timing**: Unsloth may optimize training loop in ways that skip callbacks
2. **Natural gradient preconditioning**: May conflict with `adamw_8bit` optimizer
3. **Hook overhead**: Fisher tracking adds 15-20% memory

---

## 🔬 A/B Testing Setup

```python
# Experiment 1: Baseline (Unsloth only)
trainer_baseline = SFTTrainer(
    model=model,
    # ... standard config ...
    callbacks=[],  # No GRIT
)

# Experiment 2: Unsloth + GRIT
trainer_grit = SFTTrainer(
    model=model,
    # ... same config ...
    callbacks=[GRITCallback(grit_manager, grit_config)],  # With GRIT
)

# Compare metrics
baseline_stats = trainer_baseline.train()
grit_stats = trainer_grit.train()
```

---

## 📅 Implementation Timeline

### Phase 1: Verify Callback Compatibility (30 min)

1. Create minimal test script with `SFTTrainer` + dummy callback
2. Verify callbacks fire at correct training steps
3. Test if Unsloth's optimizations interfere with callback hooks

### Phase 2: Implement GRIT Components (2 hours)

1. Extract `GRITCallback` from GRIT repo
2. Adapt `GRITManager.attach_hooks()` to work with Unsloth's PEFT model
3. Test Fisher matrix accumulation on toy example

### Phase 3: Integration (1 hour)

1. Add 26 lines to existing notebook (create new cell group "GRIT Enhancements")
2. Run full training with callback enabled
3. Verify GGUF export still works

### Phase 4: Train Hybrid GRIT Model (1.5 hours)

1. **Setup Hybrid GRIT Environment** (30 min):
   - Create `unsloth_2_Grit/` directory
   - Copy `unsloth/Llama3_(8B)_Ollama.ipynb` → `unsloth_2_Grit/Llama3_(8B)_GRIT.ipynb`
   - Clone GRIT repo into `unsloth_2_Grit/grit_repo/`
   - Add 26 lines for GRIT integration

2. **Train Hybrid GRIT Model** (1 hour):
   - ✅ Train: Unsloth + GRIT (LoRA + Fisher-guided) in `unsloth_2_Grit/`
   - ❌ Skip: Unsloth baseline (already exists in `unsloth/lora_model/`)
   - Save to `unsloth_2_Grit/lora_model/`
   - Export GGUF to `unsloth_2_Grit/model/`
   - Deploy to Ollama: `ollama create grit_model -f unsloth_2_Grit/model/Modelfile`

### Phase 4b: Train Pure GRIT Model (2 hours)

1. **Setup Pure GRIT Environment** (30 min):
   ```bash
   cd /home/ubuntu/DiskUsEast1/finetuning_evaluation
   mkdir -p unsloth_3_Pure_Grit
   cd unsloth_3_Pure_Grit

   # Clone GRIT repository
   git clone https://anonymous.4open.science/r/iclr2026-submission-18146 grit_repo

   # Create training script
   # (Copy full script from "Pure GRIT Training Script" section above)
   nano train_aqi_grit.py
   ```

2. **Train Pure GRIT Model** (1 hour):
   ```bash
   cd /home/ubuntu/DiskUsEast1/finetuning_evaluation/unsloth_3_Pure_Grit
   python train_aqi_grit.py
   ```

3. **Export to GGUF** (30 min):
   ```bash
   # Option 1: Using llama.cpp
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make -j$(nproc)
   python convert.py ../lora_model --outfile ../model/model-unsloth.gguf --outtype q8_0

   # Create Modelfile
   cd ../model
   cat > Modelfile <<'EOF'
   FROM ./model-unsloth.gguf

   TEMPLATE """Below is an instruction that describes a task. Write a response.

   ### Instruction:
   {{ .Prompt }}

   ### Response:
   {{ .Response }}<|end_of_text|>"""

   PARAMETER stop "<|end_of_text|>"
   PARAMETER temperature 1.5
   PARAMETER min_p 0.1
   EOF

   # Deploy to Ollama
   ollama create pure_grit_model -f ./Modelfile
   ```

### Phase 5: Comparative Evaluation (2.5 hours)

1. **Create Evaluation Scripts** (30 min):
   - `src/m01_evaluate_base.py` - Evaluate base model via Ollama
   - `src/m02_evaluate_unsloth_ft.py` - Evaluate `ollama run unsloth_model`
   - `src/m03_evaluate_grit_ft.py` - Evaluate `ollama run grit_model` (hybrid)
   - `src/m04_evaluate_pure_grit.py` - Evaluate `ollama run pure_grit_model`
   - `src/m05_compare_all_models.py` - Calculate delta_AQI for all 4 models

2. **Run Evaluations** (1.5 hours):
   - Evaluate all 4 models on same AQI test dataset via Ollama endpoints:
     1. Base model (no fine-tuning)
     2. Unsloth fine-tuned (from `unsloth/` - already deployed)
     3. Hybrid GRIT fine-tuned (from `unsloth_2_Grit/` - newly deployed)
     4. Pure GRIT fine-tuned (from `unsloth_3_Pure_Grit/` - newly deployed)

3. **Calculate Deltas** (30 min):
   - `delta_AQI_unsloth = AQI_accuracy_unsloth - baseline_AQI_accuracy`
   - `delta_AQI_hybrid_grit = AQI_accuracy_hybrid_grit - baseline_AQI_accuracy`
   - `delta_AQI_pure_grit = AQI_accuracy_pure_grit - baseline_AQI_accuracy`
   - Measure inference speed deltas
   - Compare: training time, memory, final loss, alignment preservation
   - Save all results to `outputs/centralized.db`

**Total Estimated Time:** 8-9 hours

---

## 🛡️ Risk Mitigation

### Primary Risk
Callbacks may not fire properly due to Unsloth's custom training loop optimizations.

### Fallback Strategy
If callbacks don't work:
- Use Unsloth **only** for data prep (`to_sharegpt`, `standardize_sharegpt`) + GGUF export
- Use standard HuggingFace `Trainer` for training with full GRIT integration
- Still benefit from Unsloth's data processing utilities

---

## 🔍 Validation Metrics

### Evaluation Framework

All models will be evaluated on the **same AQI test dataset** to calculate **delta_AQI** (improvement over base model).

### Model Variants (Ollama Endpoints)

All models evaluated via **Ollama API** for consistent, non-destructive evaluation:

| # | Model Name          | Base Model                      | Training Method         | Location                         | Ollama Endpoint  | Status      |
|---|---------------------|---------------------------------|-------------------------|----------------------------------|------------------|-------------|
| 1 | **Base (No FT)**    | `unsloth/llama-3-8b-bnb-4bit`   | None (control)          | Unsloth repo (HF)                | `llama3:8b` or direct API | ✅ Available |
| 2 | **Unsloth FT**      | `unsloth/llama-3-8b-bnb-4bit`   | Standard LoRA           | `unsloth/lora_model/`            | `unsloth_model`  | ✅ Deployed  |
| 3 | **Unsloth+GRIT FT** | `unsloth/llama-3-8b-bnb-4bit`   | LoRA + Fisher-guided    | `unsloth_2_Grit/lora_model/`     | `grit_model`     | 🆕 To Deploy |
| 4 | **Pure GRIT FT**    | `unsloth/llama-3-8b-bnb-4bit`   | Full GRIT (native)      | `unsloth_3_Pure_Grit/lora_model/`| `pure_grit_model`| 🆕 To Implement|

**Key Point:** All 4 models start from the **same base model** (`unsloth/llama-3-8b-bnb-4bit`), ensuring fair comparison.

### Metrics Table

| Metric                          | Base (No FT) | Unsloth FT | ~~Hybrid GRIT~~ | Pure GRIT | Best Δ |
|---------------------------------|--------------|------------|-----------------|-----------|--------|
| **Training Metrics**            |              |            |                 |           |        |
| Training time (200 steps)       | N/A          | 16.1 min   | ~~16.17 min~~   | ?         | ?      |
| Peak memory (A10 24GB)          | N/A          | 7.37 GB    | ~~7.37 GB~~     | ?         | ?      |
| Final training loss (200 steps) | N/A          | 1.0439     | ~~1.045~~       | ?         | ?      |
| **GRIT Verification**           |              |            |                 |           |        |
| Fisher inversions (200 steps)   | N/A          | N/A        | ~~40 (every 5)~~ | ?        | ?      |
| Preconditioning coverage        | N/A          | N/A        | ~~97.5%~~       | ?         | ?      |
| Loss curve divergence           | N/A          | N/A        | ~~❌ IDENTICAL~~ | ?        | ?      |
| **AQI Domain (Primary)**        |              |            |                 |           |        |
| AQI accuracy (%)                | **baseline** | ?          | ~~SKIPPED~~     | ?         | -      |
| delta_AQI (vs base)             | 0.0          | ?          | ~~N/A~~         | ?         | ?      |
| **Inference Performance**       |              |            |                 |           |        |
| Inference speed (tokens/sec)    | **baseline** | ?          | ~~SKIPPED~~     | ?         | ?      |
| delta_speed (vs base)           | 0.0          | ?          | ~~N/A~~         | ?         | ?      |

**Note:** Hybrid GRIT results crossed out due to experimental failure (identical loss curves).

### Delta AQI Calculation

```python
# Baseline (no fine-tuning)
baseline_AQI_accuracy = evaluate_model(base_model, aqi_test_dataset)
baseline_tokens_per_sec = measure_inference_speed(base_model)

# Fine-tuned models (Hybrid GRIT skipped due to experimental failure)
unsloth_AQI_accuracy = evaluate_model(unsloth_finetuned, aqi_test_dataset)
# hybrid_grit_AQI_accuracy = SKIPPED - identical to baseline
pure_grit_AQI_accuracy = evaluate_model(pure_grit_finetuned, aqi_test_dataset)

# Delta calculations (improvement over base)
delta_AQI_unsloth = unsloth_AQI_accuracy - baseline_AQI_accuracy
# delta_AQI_hybrid_grit = N/A (experiment failed)
delta_AQI_pure_grit = pure_grit_AQI_accuracy - baseline_AQI_accuracy
```

### Success Criteria

- **delta_AQI > 0**: Fine-tuning improved AQI domain performance
- **delta_AQI_pure_grit > delta_AQI_unsloth**: Pure GRIT outperforms standard LoRA
- ~~delta_AQI_hybrid_grit~~ → SKIPPED (experiment failed)
- **delta_speed ≥ -10%**: Inference speed not significantly degraded

---

## 📊 Evaluation Methodology

### Step 1: Prepare AQI Test Dataset

```python
# Load AQI test dataset
from datasets import load_dataset
aqi_test_dataset = load_dataset("your_aqi_dataset", split="test")

# Format for evaluation
def format_aqi_prompt(example):
    return {
        "prompt": f"Predict AQI given: {example['features']}",
        "ground_truth": example['aqi_value']
    }

aqi_eval_data = aqi_test_dataset.map(format_aqi_prompt)
```

### Step 2: Evaluate Base Model (No Fine-Tuning)

```python
from unsloth import FastLanguageModel
import torch
import time

# Load base model WITHOUT any fine-tuning
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Enable fast inference
FastLanguageModel.for_inference(base_model)

# Measure AQI accuracy
def evaluate_aqi_accuracy(model, dataset):
    correct = 0
    total = len(dataset)

    for example in dataset:
        inputs = tokenizer(example['prompt'], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse prediction and compare with ground truth
        if parse_aqi_prediction(prediction) == example['ground_truth']:
            correct += 1

    return (correct / total) * 100

# Measure inference speed
def measure_inference_speed(model, test_samples=100):
    start = time.time()
    total_tokens = 0

    for _ in range(test_samples):
        inputs = tokenizer("Sample prompt", return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50)
        total_tokens += outputs.shape[1]

    elapsed = time.time() - start
    return total_tokens / elapsed

# Run baseline evaluation
baseline_AQI_accuracy = evaluate_aqi_accuracy(base_model, aqi_eval_data)
baseline_tokens_per_sec = measure_inference_speed(base_model)

print(f"Baseline AQI Accuracy: {baseline_AQI_accuracy:.2f}%")
print(f"Baseline Inference Speed: {baseline_tokens_per_sec:.2f} tokens/sec")
```

### Step 3: Evaluate Fine-Tuned Models

```python
# Evaluate Unsloth fine-tuned model
unsloth_model, _ = FastLanguageModel.from_pretrained(
    model_name="outputs/unsloth_finetuned",  # Path to saved LoRA adapters
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(unsloth_model)
unsloth_AQI_accuracy = evaluate_aqi_accuracy(unsloth_model, aqi_eval_data)
unsloth_tokens_per_sec = measure_inference_speed(unsloth_model)

# Evaluate Hybrid GRIT fine-tuned model
hybrid_grit_model, _ = FastLanguageModel.from_pretrained(
    model_name="outputs/hybrid_grit_finetuned",  # Path to Hybrid GRIT LoRA adapters
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(hybrid_grit_model)
hybrid_grit_AQI_accuracy = evaluate_aqi_accuracy(hybrid_grit_model, aqi_eval_data)
hybrid_grit_tokens_per_sec = measure_inference_speed(hybrid_grit_model)

# Evaluate Pure GRIT fine-tuned model
pure_grit_model, _ = FastLanguageModel.from_pretrained(
    model_name="outputs/pure_grit_finetuned",  # Path to Pure GRIT LoRA adapters
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(pure_grit_model)
pure_grit_AQI_accuracy = evaluate_aqi_accuracy(pure_grit_model, aqi_eval_data)
pure_grit_tokens_per_sec = measure_inference_speed(pure_grit_model)
```

### Step 4: Calculate Deltas and Report

```python
# Calculate delta_AQI
delta_AQI_unsloth = unsloth_AQI_accuracy - baseline_AQI_accuracy
delta_AQI_hybrid_grit = hybrid_grit_AQI_accuracy - baseline_AQI_accuracy
delta_AQI_pure_grit = pure_grit_AQI_accuracy - baseline_AQI_accuracy

# Calculate delta_speed
delta_speed_unsloth = unsloth_tokens_per_sec - baseline_tokens_per_sec
delta_speed_hybrid_grit = hybrid_grit_tokens_per_sec - baseline_tokens_per_sec
delta_speed_pure_grit = pure_grit_tokens_per_sec - baseline_tokens_per_sec

# Print comparison table
print("\n" + "="*100)
print("EVALUATION RESULTS: AQI Domain Performance (All 4 Models)")
print("="*100)
print(f"{'Model':<30} {'AQI Acc (%)':<15} {'Δ_AQI':<15} {'Speed (t/s)':<15} {'Δ_Speed':<15}")
print("-"*100)
print(f"{'Base (No FT)':<30} {baseline_AQI_accuracy:>10.2f}   {0.0:>10.2f}   {baseline_tokens_per_sec:>10.2f}   {0.0:>10.2f}")
print(f"{'Unsloth Fine-Tuned':<30} {unsloth_AQI_accuracy:>10.2f}   {delta_AQI_unsloth:>10.2f}   {unsloth_tokens_per_sec:>10.2f}   {delta_speed_unsloth:>10.2f}")
print(f"{'Hybrid GRIT (Unsloth+GRIT)':<30} {hybrid_grit_AQI_accuracy:>10.2f}   {delta_AQI_hybrid_grit:>10.2f}   {hybrid_grit_tokens_per_sec:>10.2f}   {delta_speed_hybrid_grit:>10.2f}")
print(f"{'Pure GRIT':<30} {pure_grit_AQI_accuracy:>10.2f}   {delta_AQI_pure_grit:>10.2f}   {pure_grit_tokens_per_sec:>10.2f}   {delta_speed_pure_grit:>10.2f}")
print("="*100)

# Determine winner
best_delta_aqi = max(delta_AQI_unsloth, delta_AQI_hybrid_grit, delta_AQI_pure_grit)
if best_delta_aqi == delta_AQI_pure_grit:
    winner = "Pure GRIT"
elif best_delta_aqi == delta_AQI_hybrid_grit:
    winner = "Hybrid GRIT (Unsloth+GRIT)"
else:
    winner = "Standard Unsloth"

print(f"\n✅ WINNER: {winner}")
print(f"   Best delta_AQI: +{best_delta_aqi:.2f}%")

# Save results to database
import sqlite3
conn = sqlite3.connect("outputs/centralized.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_results (
        model_name TEXT,
        aqi_accuracy REAL,
        delta_aqi REAL,
        inference_speed REAL,
        delta_speed REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")

cursor.executemany("""
    INSERT INTO evaluation_results
    (model_name, aqi_accuracy, delta_aqi, inference_speed, delta_speed)
    VALUES (?, ?, ?, ?, ?)
""", [
    ("base_no_ft", baseline_AQI_accuracy, 0.0, baseline_tokens_per_sec, 0.0),
    ("unsloth_ft", unsloth_AQI_accuracy, delta_AQI_unsloth, unsloth_tokens_per_sec, delta_speed_unsloth),
    ("hybrid_grit_ft", hybrid_grit_AQI_accuracy, delta_AQI_hybrid_grit, hybrid_grit_tokens_per_sec, delta_speed_hybrid_grit),
    ("pure_grit_ft", pure_grit_AQI_accuracy, delta_AQI_pure_grit, pure_grit_tokens_per_sec, delta_speed_pure_grit),
])

conn.commit()
conn.close()

print("\n✅ Results saved to outputs/centralized.db")
```

---

## 🔬 Model 4: Pure GRIT Implementation

### Overview

Pure GRIT is a **standalone implementation** that uses GRIT's native training pipeline without any Unsloth dependencies. This provides a research-grade baseline to compare against the hybrid Unsloth+GRIT approach.

### GRIT Repository Structure

```
grit_repo/
├── grit/
│   ├── __init__.py
│   ├── autograd.py        # Rank-space covariance capture via PyTorch hooks
│   ├── config.py          # GRITConfig class with all hyperparameters
│   ├── manager.py         # GRITManager orchestrates Fisher updates, reprojection
│   ├── optim.py           # Custom optimizer wrapper for natural gradient
│   └── trainer.py         # GRITTrainer extends HuggingFace Trainer
└── examples/
    ├── train_alpaca.py    # Reference implementation for instruction tuning
    ├── train_dolly.py     # Dolly dataset example
    ├── train_boolq.py     # Boolean Q&A classification
    ├── train_qnli.py      # QNLI classification
    └── train_gsm8k.py     # Math reasoning dataset
```

### GRIT Module Explanations

#### 1. `grit/config.py` - Configuration Management

**Purpose:** Central configuration for all GRIT hyperparameters

**Key Parameters:**
```python
class GRITConfig:
    # Model & Dataset
    model_id: str = "meta-llama/Llama-3.2-3B"
    dataset_name: str = "tatsu-lab/alpaca"
    max_seq_length: int = 1024

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5
    precision: str = "bf16"  # bfloat16 for stability

    # LoRA Settings
    lora_rank: int = 16
    lora_alpha: int = 32  # Typically 2x rank
    lora_dropout: float = 0.0
    lora_target_modules: list = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    min_lora_rank: int = 8   # Minimum rank for adaptation

    # Fisher Matrix & K-FAC
    kfac_update_freq: int = 50        # Update Fisher every N steps
    kfac_min_samples: int = 64        # Min samples for stable estimate
    kfac_damping: float = 0.001       # Damping for numerical stability

    # Reprojection
    reprojection_freq: int = 50       # Reproject weights every N steps
    use_two_sided_reprojection: bool = True

    # Rank Adaptation
    enable_rank_adaptation: bool = True
    rank_adaptation_threshold: float = 0.99  # Keep 99% of Fisher energy

    # Logging
    log_fisher_spectrum: bool = True
    log_top_eigs: int = 8  # Log top 8 eigenvalues
```

#### 2. `grit/autograd.py` - Fisher Information Tracking

**Purpose:** Custom PyTorch autograd hooks to capture activation/gradient statistics for Fisher matrix estimation

**How It Works:**
```python
class FisherHook:
    """Captures covariance matrices for LoRA A and B matrices"""

    def forward_hook(module, input, output):
        """
        Capture activations during forward pass
        Stores: A_cov += activation @ activation.T
        """
        # For LoRA layer: y = W_0(x) + BA(x)
        # Need to track x (input activations)

    def backward_hook(module, grad_input, grad_output):
        """
        Capture gradients during backward pass
        Stores: G_cov += grad_output @ grad_output.T
        """
        # Fisher approximation: F ≈ E[g g^T]
        # where g = gradient of log-likelihood
```

**What It Captures:**
- **Activation covariance:** `A_cov = Σ(x_i @ x_i.T)` where x is input to LoRA layer
- **Gradient covariance:** `G_cov = Σ(g_i @ g_i.T)` where g is gradient w.r.t. output
- **Fisher approximation:** `F ≈ A_cov ⊗ G_cov` (Kronecker product)

#### 3. `grit/manager.py` - GRIT Orchestrator

**Purpose:** Manages Fisher matrix computation, inversion, natural gradient application, and weight reprojection

**Key Methods:**

**`__init__(model, config, device)`**
```python
def __init__(self, model, config, device):
    """
    Initialize GRITManager
    - Identifies all LoRA layers in model
    - Attaches Fisher tracking hooks
    - Prepares storage for covariance matrices
    """
    self.model = model
    self.config = config
    self.device = device
    self._instrument_model()  # Attach hooks to LoRA layers
```

**`_instrument_model()`**
```python
def _instrument_model(self):
    """
    Find all LoRA layers and attach autograd hooks
    Creates storage for:
    - A_cov: activation covariance (for A matrix)
    - G_cov: gradient covariance (for B matrix)
    """
    for name, module in self.model.named_modules():
        if isinstance(module, LoraLayer):
            # Attach forward/backward hooks
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
```

**`update_and_invert_factors()`**
```python
def update_and_invert_factors(self):
    """
    Compute Fisher matrix inverse: F^{-1}

    Steps:
    1. Accumulate covariance matrices from hooks
    2. Add damping for numerical stability: F_damped = F + λI
    3. Compute inverse: F_inv = (F + λI)^{-1}
    4. Optional: Log eigenvalue spectrum for debugging
    """
    for lora_layer in self.lora_layers:
        A_cov = lora_layer.activation_cov
        G_cov = lora_layer.gradient_cov

        # Add damping
        A_cov_damped = A_cov + self.config.kfac_damping * torch.eye(A_cov.size(0))
        G_cov_damped = G_cov + self.config.kfac_damping * torch.eye(G_cov.size(0))

        # Invert
        A_inv = torch.linalg.inv(A_cov_damped)
        G_inv = torch.linalg.inv(G_cov_damped)

        # Store for natural gradient
        lora_layer.A_inv = A_inv
        lora_layer.G_inv = G_inv
```

**`precondition_gradients()`**
```python
def precondition_gradients(self):
    """
    Apply natural gradient: g' = F^{-1} @ g

    For LoRA: ΔW = BA, so gradients transform as:
    - grad_B' = G_inv @ grad_B @ A_inv^T
    - grad_A' = G_inv^T @ grad_A @ A_inv
    """
    for lora_layer in self.lora_layers:
        # Transform B gradient
        grad_B = lora_layer.lora_B.weight.grad
        grad_B_precond = lora_layer.G_inv @ grad_B @ lora_layer.A_inv.T
        lora_layer.lora_B.weight.grad = grad_B_precond

        # Transform A gradient
        grad_A = lora_layer.lora_A.weight.grad
        grad_A_precond = lora_layer.G_inv.T @ grad_A @ lora_layer.A_inv
        lora_layer.lora_A.weight.grad = grad_A_precond
```

**`neural_reprojection()`**
```python
def neural_reprojection(self):
    """
    Project LoRA weights onto principal Fisher eigenvectors

    Steps:
    1. Compute Fisher eigendecomposition: F = U Λ U^T
    2. Calculate cumulative energy: E_r = Σ(λ_1..λ_r) / Σ(all λ)
    3. Find minimal rank: r* = argmin{r : E_r ≥ threshold}
    4. Project weights: W' = U_r* @ U_r*^T @ W
    5. Resize LoRA matrices if rank changed
    """
    for lora_layer in self.lora_layers:
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(lora_layer.fisher_matrix)
        eigenvalues = eigenvalues.flip(0)  # Sort descending
        eigenvectors = eigenvectors.flip(1)

        # Find optimal rank
        cumulative_energy = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum()
        new_rank = (cumulative_energy >= self.config.rank_adaptation_threshold).nonzero()[0].item() + 1
        new_rank = max(self.config.min_lora_rank, new_rank)

        # Project weights
        U_r = eigenvectors[:, :new_rank]
        B_proj = U_r @ (U_r.T @ lora_layer.lora_B.weight)
        A_proj = (lora_layer.lora_A.weight @ U_r) @ U_r.T

        # Update weights
        lora_layer.lora_B.weight.data = B_proj
        lora_layer.lora_A.weight.data = A_proj
```

#### 4. `grit/trainer.py` - GRIT Custom Trainer

**Purpose:** Extends HuggingFace `Trainer` to integrate GRIT updates into training loop

**Key Workflow:**
```python
class GRITTrainer(Trainer):
    def __init__(self, grit_manager, grit_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grit_manager = grit_manager
        self.grit_config = grit_config

    def training_step(self, model, inputs):
        """Override training step to add GRIT operations"""

        # 1. Standard forward + backward pass
        loss = super().training_step(model, inputs)

        # 2. Update Fisher matrix (every kfac_update_freq steps)
        if self.state.global_step % self.grit_config.kfac_update_freq == 0:
            self.grit_manager.update_and_invert_factors()

        # 3. Apply natural gradient preconditioning
        if self.grit_config.use_natural_gradient:
            self.grit_manager.precondition_gradients()

        # 4. Perform reprojection (every reproj_freq steps)
        if self.state.global_step % self.grit_config.reprojection_freq == 0:
            self.grit_manager.neural_reprojection()

        return loss
```

#### 5. `grit/optim.py` - Optimizer Wrapper

**Purpose:** Wraps standard PyTorch optimizer to apply GRIT preconditioning before parameter updates

```python
class GRITOptimizer:
    """Wrapper that applies natural gradient before optimizer step"""

    def __init__(self, optimizer, grit_manager):
        self.optimizer = optimizer
        self.grit_manager = grit_manager

    def step(self):
        # Precondition gradients before optimizer updates parameters
        self.grit_manager.precondition_gradients()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
```

### Pure GRIT Training Script

**Create `unsloth_3_Pure_Grit/train_aqi_grit.py`:**

```python
#!/usr/bin/env python3
"""
Pure GRIT Training Script for AQI Dataset
Based on grit_repo/examples/train_alpaca.py
"""

import sys
sys.path.insert(0, './grit_repo')

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# GRIT imports
from grit.config import GRITConfig
from grit.manager import GRITManager
from grit.trainer import GRITTrainer

# ==================== Configuration ====================
config = GRITConfig(
    # Model (same base as all others)
    model_id="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,

    # Training (match Unsloth baseline)
    batch_size=2,
    gradient_accumulation_steps=4,
    num_epochs=1,
    max_steps=60,  # Same as Unsloth for fair comparison
    learning_rate=2e-4,
    precision="bf16",

    # LoRA (match Unsloth baseline)
    lora_rank=16,
    lora_alpha=32,  # GRIT convention: 2x rank
    lora_dropout=0.0,
    lora_target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    min_lora_rank=8,

    # GRIT-specific Fisher settings
    kfac_update_freq=150,
    kfac_min_samples=64,
    kfac_damping=1e-4,

    # Reprojection
    reprojection_freq=300,
    use_two_sided_reprojection=True,

    # Rank Adaptation
    enable_rank_adaptation=False,  # Disable for fair comparison
    rank_adaptation_threshold=0.99,

    # Logging
    log_fisher_spectrum=True,
    log_top_eigs=8,
)

# ==================== Load Model ====================
print(f"Loading base model: {config.model_id}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==================== Apply LoRA ====================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.lora_target_modules,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)
print(f"\nModel Parameters:")
model.print_trainable_parameters()

# ==================== Load Dataset ====================
print(f"\nLoading dataset: {config.dataset_name}")
dataset = load_dataset(config.dataset_name, split="train")

# Format dataset (same as Unsloth)
def format_alpaca(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"Below is an instruction that describes a task, paired with an input. Write a response.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    full_text = prompt + output + tokenizer.eos_token
    return {"text": full_text}

dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=config.max_seq_length,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=4,
)

print(f"Dataset size: {len(tokenized_dataset)}")

# ==================== Initialize GRIT Manager ====================
print("\nInitializing GRIT Manager...")
grit_manager = GRITManager(
    model=model,
    config=config,
    device="cuda",
)

# ==================== Training Arguments ====================
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=config.num_epochs,
    max_steps=config.max_steps,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=5,
    logging_steps=1,
    save_strategy="steps",
    save_steps=500,
    bf16=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    seed=3407,
    report_to="none",  # Change to "wandb" for logging
)

# ==================== GRIT Trainer ====================
print("\nInitializing GRIT Trainer...")
trainer = GRITTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    grit_manager=grit_manager,
    grit_config=config,
)

# ==================== Train ====================
print("\n" + "="*80)
print("Starting GRIT Training...")
print("="*80)

trainer_stats = trainer.train()

print("\n" + "="*80)
print("Training Completed!")
print("="*80)
print(f"Runtime: {trainer_stats.metrics['train_runtime']:.2f} seconds")
print(f"Final Loss: {trainer_stats.metrics['train_loss']:.4f}")

# ==================== Save Model ====================
print("\nSaving model to ./lora_model")
model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")

print("\n✅ Pure GRIT training complete!")
print(f"LoRA adapters saved to: unsloth_3_Pure_Grit/lora_model/")
```

### Export Pure GRIT to GGUF & Ollama

Since Pure GRIT doesn't have Unsloth's `save_pretrained_gguf()`, we need manual export:

**Option 1: Use llama.cpp directly**
```bash
cd unsloth_3_Pure_Grit

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc)

# Convert to GGUF
python convert.py ../lora_model --outfile ../model/model-unsloth.gguf --outtype q8_0

# Create Modelfile
cd ../model
cat > Modelfile <<'EOF'
FROM ./model-unsloth.gguf

TEMPLATE """Below is an instruction that describes a task. Write a response.

### Instruction:
{{ .Prompt }}

### Response:
{{ .Response }}<|end_of_text|>"""

PARAMETER stop "<|end_of_text|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
EOF

# Deploy to Ollama
ollama create pure_grit_model -f ./Modelfile
```

**Option 2: Temporarily install Unsloth for export only**
```bash
pip install unsloth

python <<EOF
from unsloth import FastLanguageModel

# Load Pure GRIT model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./lora_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Export to GGUF using Unsloth's helper
model.save_pretrained_gguf("model", tokenizer, quantization_method="q8_0")
print("GGUF export complete!")
EOF

# Deploy to Ollama
ollama create pure_grit_model -f ./model/Modelfile
```

---

## 📚 Mathematical Background

### Fisher Information Matrix

The Fisher matrix captures the **curvature of the loss landscape**:

```
F = 𝔼[∇log p(y|x) · ∇log p(y|x)ᵀ]
```

- High Fisher values → parameter is important for current behavior
- GRIT uses this to **preserve alignment** during fine-tuning

### Natural Gradient Update

Instead of standard gradient descent, GRIT uses:

```
θ_{t+1} = θ_t - α · F⁻¹ · ∇L(θ)
```

- Accounts for parameter space geometry
- More efficient optimization in curved loss landscapes

### Two-Sided Reprojection

GRIT projects gradients to avoid interfering with important subspaces:

```
∇' = (I - P_F) · ∇
```

where `P_F` projects onto high-Fisher directions (alignment-sensitive)

---

## 🎓 References

- **GRIT Paper:** Anonymous ICLR 2026 submission
- **Unsloth Documentation:** https://docs.unsloth.ai/
- **TRL SFTTrainer:** https://huggingface.co/docs/trl/en/sft_trainer
- **PEFT LoRA:** https://huggingface.co/docs/peft/en/package_reference/lora

---

## ✅ Next Steps

### Implementation Sequence

1. **Setup** (30 min):
   - Clone GRIT repository
   - Test callback compatibility with Unsloth
   - Verify imports and dependencies

2. **Baseline Evaluation** (30 min):
   - Load `unsloth/llama-3-8b-bnb-4bit` base model
   - Evaluate on AQI test dataset
   - Measure baseline_AQI_accuracy and baseline_tokens_per_sec
   - Store results in `outputs/centralized.db`

3. **Create GRIT-Enhanced Notebook** (1 hour):
   - Copy `unsloth/Llama3_(8B)_Ollama.ipynb` → `unsloth/Llama3_(8B)_GRIT.ipynb`
   - Add 26 lines for GRIT integration
   - Create side-by-side comparison cells

4. **Train Models** (1-2 hours):
   - Train Unsloth baseline (LoRA only)
   - Train Unsloth + GRIT (LoRA + Fisher-guided)
   - Save both models to `outputs/`

5. **Comparative Evaluation** (1 hour):
   - Evaluate both fine-tuned models on AQI test dataset
   - Calculate delta_AQI for each
   - Measure inference speed deltas
   - Generate comparison plots

6. **Documentation** (30 min):
   - Update `outputs/centralized.db` with all results
   - Create `outputs/grit_evaluation.md` summary report
   - Export best model to Ollama

### Expected Outcomes

**Primary Goal:** Quantify **delta_AQI** improvement over base model

**Hypothesis:**
- Standard Unsloth fine-tuning: `delta_AQI_unsloth = +15-25%`
- GRIT-enhanced fine-tuning: `delta_AQI_grit = +20-30%` (better alignment preservation)

**Success Metrics:**
- ✅ `delta_AQI_grit > delta_AQI_unsloth` (GRIT wins)
- ✅ Both deltas > 0 (fine-tuning helps)
- ✅ Inference speed degradation < 10%
- ✅ General knowledge retention > 70%

**Deliverables:**
1. Trained models in `outputs/unsloth_finetuned/` and `outputs/grit_finetuned/`
2. Evaluation results in `outputs/centralized.db`
3. Comparison report in `outputs/grit_evaluation.md`
4. GGUF exports for Ollama deployment
