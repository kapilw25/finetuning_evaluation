# Training Plan: Return to PKU-SafeRLHF Dataset

**Date**: 2025-10-27
**Goal**: Train SFT, DPO, CITA on PKU-SafeRLHF (proven dataset)
**Reason**: Vaibhaav dataset failed (55% accuracy = random guessing)

---

## Dataset Decision

**PKU-SafeRLHF** (chosen):
- ✅ **Proven**: 90% accuracy achievable (287.7× better than Vaibhaav)
- ✅ **CITA improvement**: 16.5% over DPO (margin 6.56 vs 5.52)
- ✅ **Clear preference signal**: One safe, one unsafe response
- ✅ **Increased training data**: 12,035 samples (train+test clear contrast combined)

**Vaibhaav** (abandoned):
- ❌ **Failed**: 55% accuracy (random guessing)
- ❌ **No learning**: CITA margin negative (-0.003)
- ❌ **Versioned**: Tagged as v1_vaibhaav_failed for research traceability

---

## Dataset Configuration

### Training Data: 12,035 samples (Clear Contrast from train+test)
```python
# Combine train and test clear contrast samples
train_clear = 10,813  # From train split
test_clear = 1,222    # From test split
total_train = 12,035  # +11% more data
```

### Evaluation Data: 3,684 samples (Both-unsafe from test)
```python
# Test split only - completely disjoint from training
test_both_unsafe = 3,684  # Harmful prompts for toxicity eval
```

**No overlap**: Clear contrast and both-unsafe are mutually exclusive subsets

---

## Implementation Checklist

### Phase 1: Restore PKU Scripts 
- [✅] Delete Vaibhaav-based training scripts
  - `comparative_study/01a_SFT_Baseline/Llama3_BF16.py`
  - `comparative_study/02a_DPO_Baseline/Llama3_BF16.py`
  - `comparative_study/03a_CITA_Baseline/Llama3_BF16.py`
  - `comparative_study/0c_utils/data_prep/loader_vaibhaav.py`

- [✅] Restore PKU BACKUP scripts
  ```bash
  mv comparative_study/01a_SFT_Baseline/Llama3_BF16_BACKUP.py \
     comparative_study/01a_SFT_Baseline/Llama3_BF16.py

  mv comparative_study/02a_DPO_Baseline/Llama3_BF16_BACKUP.py \
     comparative_study/02a_DPO_Baseline/Llama3_BF16.py

  mv comparative_study/03a_CITA_Baseline/Llama3_BF16_BACKUP.py \
     comparative_study/03a_CITA_Baseline/Llama3_BF16.py
  ```

### Phase 2: Update PKU Loader (Combine train+test) 
- [✅] Modify `loader_pku.py` to load clear contrast from both splits
  ```python
  def load_pku_combined_clear_contrast(val_split: float = 0.1):
      """Load clear contrast from train+test, create 90/10 split"""
      # Load train split clear contrast
      train_data = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
      train_clear = train_data.filter(lambda x: x['is_response_0_safe'] != x['is_response_1_safe'])

      # Load test split clear contrast
      test_data = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
      test_clear = test_data.filter(lambda x: x['is_response_0_safe'] != x['is_response_1_safe'])

      # Combine (10,813 + 1,222 = 12,035)
      from datasets import concatenate_datasets
      combined = concatenate_datasets([train_clear, test_clear])

      # 90/10 split (10,831 train / 1,204 val)
      return combined.train_test_split(test_size=val_split, seed=3407)
  ```

### Phase 3: Update Training Scripts 
- [✅] Update SFT script: Use `load_pku_combined_clear_contrast()`
- [✅] Update DPO script: Use `load_pku_combined_clear_contrast()`
- [✅] Update CITA script: Use `load_pku_combined_clear_contrast()`

### Phase 4: Update model_utils.py (HF repo names) 
- [✅] Change `MODEL_NAME_MAP` to reflect stacked training:
  ```python
  MODEL_NAME_MAP = {
      "SFT_Baseline": "kapilw25/llama3-8b-pku-sft-baseline",    # Base → SFT
      "DPO_Baseline": "kapilw25/llama3-8b-pku-dpo-sft",          # SFT → DPO (stacked)
      "CITA_Baseline": "kapilw25/llama3-8b-pku-cita-dpo",        # DPO → CITA (stacked)
  }
  ```

### Phase 5: Test Compilation
- [✅] `python -m py_compile comparative_study/01a_SFT_Baseline/Llama3_BF16.py`
- [✅] `python -m py_compile comparative_study/02a_DPO_Baseline/Llama3_BF16.py`
- [✅] `python -m py_compile comparative_study/03a_CITA_Baseline/Llama3_BF16.py`

### Phase 6: Convert from Steps-Based to Epoch-Based Training ⏳
**Reference**: https://github.com/kapilw25/finetuning_evaluation/blob/v1_vaibhaav_failed/

**Why Epoch-Based?**
- ✅ Dataset-agnostic (works with any dataset size: 10k or 50k samples)
- ✅ Proportional checkpoints (always saves at 20%, 40%, 60%, 80%, 100%)
- ✅ Adaptive warmup (3% ratio auto-scales with dataset size)
- ✅ Fair comparison (all models see same % of data, not arbitrary step count)

**Current Issue**: Local scripts use fixed `max_steps` (200/1000), but should use `num_train_epochs` (0.1/1.0)

#### 6.1: Update SFT Script (`01a_SFT_Baseline/Llama3_BF16.py`)
- [⏳] **Argparse** (line ~425): Replace `--steps` with `--epochs`
  ```python
  parser.add_argument(
      "--epochs",
      type=float,
      default=None,
      help="Number of training epochs (overrides --mode)"
  )
  ```

- [⏳] **Mode Configuration** (line ~440): Replace steps logic with epochs
  ```python
  if args.epochs is not None:
      num_epochs = args.epochs
      print(f"✅ Custom configuration: {num_epochs} epochs")
  elif args.mode == "sanity":
      num_epochs = 0.1  # 10% of data (~1,083 samples, ~2 min)
      print(f"✅ Sanity check mode: {num_epochs} epochs (~2 minutes)")
  else:
      num_epochs = 1.0  # Full epoch (~10,831 samples, ~17 min)
      print(f"✅ Full training mode: {num_epochs} epochs (~17 minutes)")
  ```

- [⏳] **Steps Per Epoch Calculation** (after dataset loading, line ~210):
  ```python
  # Calculate training steps (needed for checkpoint intervals)
  effective_batch_size = 2 * 4  # per_device=2, grad_accum=4
  steps_per_epoch = len(train_dataset) // effective_batch_size
  total_steps = int(steps_per_epoch * num_epochs)
  checkpoint_interval = int(total_steps * 0.2)  # Save/eval every 20%

  print(f"\n📊 Training Configuration:")
  print(f"   Dataset size: {len(train_dataset):,} samples")
  print(f"   Effective batch size: {effective_batch_size}")
  print(f"   Steps per epoch: {steps_per_epoch:,}")
  print(f"   Total steps: {total_steps:,} ({num_epochs} epochs)")
  print(f"   Checkpoint interval: {checkpoint_interval} steps (20% of training)")
  ```

- [⏳] **Validation Scaling for Sanity Mode** (after val_dataset loading, line ~210):
  ```python
  # Scale validation set for sanity mode (faster evaluation)
  if num_epochs < 1.0:
      val_size_scaled = int(len(val_dataset) * num_epochs)
      val_dataset = val_dataset.select(range(val_size_scaled))
      print(f"⚡ SANITY mode: Scaled validation set to {num_epochs:.1f}x ({len(val_dataset):,} samples)")
  ```

- [⏳] **SFTConfig** (line ~223): Replace `max_steps` with `num_train_epochs` and `warmup_ratio`
  ```python
  training_args = SFTConfig(
      output_dir=str(output_dir),
      per_device_train_batch_size=2,
      gradient_accumulation_steps=4,
      num_train_epochs=num_epochs,  # ← CHANGED: Epoch-based training
      warmup_ratio=0.03,  # ← CHANGED: 3% warmup (auto-scales)
      learning_rate=2e-4,
      logging_steps=1,
      optim="adamw_torch",
      weight_decay=0.01,
      lr_scheduler_type="cosine",
      seed=3407,
      bf16=True,
      gradient_checkpointing=True,
      save_strategy="steps",  # ← Keep "steps" strategy
      save_steps=checkpoint_interval,  # ← CHANGED: Dynamic interval
      save_total_limit=5,
      report_to="tensorboard",
      logging_dir=str(tensorboard_run_dir),
      logging_first_step=True,
      dataloader_num_workers=2,
      dataloader_pin_memory=True,
      max_length=2048,
      packing=False,
      eval_strategy="steps",
      eval_steps=checkpoint_interval,  # ← CHANGED: Dynamic interval
      per_device_eval_batch_size=2,
  )
  ```

- [⏳] **Update function signature** (line ~88): Replace `max_steps` with `num_epochs`
  ```python
  def train_sft_baseline(num_epochs=1.0, output_dir="./outputs/SFT_Baseline", base_model=None, force_skip=False):
      """
      Train SFT baseline with epoch-based training

      Args:
          num_epochs: Number of training epochs (default: 1.0 for full, 0.1 for sanity)
          output_dir: Output directory for checkpoints
          base_model: HuggingFace model ID to load LoRA adapters from (for stacking)
          force_skip: If True, skip training and only run inference
      """
  ```

- [⏳] **Update checkpoint detection logic** (line ~140): Keep using step numbers (no changes needed)
  ```python
  # NOTE: is_training_complete() already works with step-based checkpoints
  # Checkpoint names like "checkpoint-270" → extract step number → compare
  # This works fine with epoch-based training (checkpoints still named by steps)
  if latest_checkpoint and is_training_complete(latest_checkpoint, total_steps):
      # Use total_steps (calculated from num_epochs) instead of max_steps
  ```

- [⏳] **Update time estimates** (line ~480-487): Replace steps-based with epoch-based
  ```python
  # OLD (steps-based)
  print(f"Training will take approximately: {'~12 minutes' if max_steps == 200 else '~62 minutes'}")

  # NEW (epoch-based)
  print(f"Training will take approximately: {'~2 minutes' if num_epochs == 0.1 else '~17 minutes'}")
  ```

- [⏳] **Update main execution call** (line ~518): Pass `num_epochs` instead of `max_steps`
  ```python
  trainer, training_skipped = train_sft_baseline(num_epochs=num_epochs, base_model=args.base_model, force_skip=force_skip)
  ```

#### 6.2: Update DPO Script (`02a_DPO_Baseline/Llama3_BF16.py`)
- [⏳] Apply same changes as SFT (6.1), with adjustments:
  - Function signature: `train_dpo_baseline(num_epochs=1.0, ...)`
  - Effective batch size: `1 * 8 = 8` (per_device=1, grad_accum=8)
  - Learning rate: `1e-5` (Meta's DPO setting)
  - Estimated time: ~2 min (sanity), ~17 min (full)
  - Time estimate: `'~2 minutes' if num_epochs == 0.1 else '~17 minutes'`
  - Main call: `train_dpo_baseline(num_epochs=num_epochs, ...)`

#### 6.3: Update CITA Script (`03a_CITA_Baseline/Llama3_BF16.py`)
- [✅] Apply same changes as SFT (6.1), with adjustments:
  - Function signature: `train_cita_baseline(num_epochs=1.0, ...)`
  - Effective batch size: `1 * 8 = 8` (per_device=1, grad_accum=8)
  - Learning rate: `1.185e-05` (Optuna Trial 2)
  - Beta: `0.1133` (Optuna Trial 2)
  - Lambda KL: `0.001010` (Optuna Trial 2)
  - Estimated time: ~2 min (sanity), ~17 min (full)
  - Time estimate: `'~2 minutes' if num_epochs == 0.1 else '~17 minutes'`
  - Main call: `train_cita_baseline(num_epochs=num_epochs, ...)`

#### 6.4: Test Compilation After Changes ✅
- [✅] `python -m py_compile comparative_study/01a_SFT_Baseline/Llama3_BF16.py`
- [✅] `python -m py_compile comparative_study/02a_DPO_Baseline/Llama3_BF16.py`
- [✅] `python -m py_compile comparative_study/03a_CITA_Baseline/Llama3_BF16.py`

---

## Training Commands

### SANITY Mode (~2 min each, 0.1 epoch)
```bash
# [✅] 1. SFT Baseline (NO instruction, PKU dataset)
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity

# [✅] 2. DPO Baseline (NO instruction, stacked on SFT)
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py \
    --mode sanity \
    --base_model kapilw25/llama3-8b-pku-sft-baseline-bf16

# [✅] 3. CITA Baseline (WITH PKU metadata instructions, stacked on DPO)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py \
    --mode sanity \
    --base_model kapilw25/llama3-8b-pku-dpo-sft-bf16
```

### FULL Training (~17 min each, 1.0 epoch)
```bash
# [✅] 1. SFT Baseline
python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full

# [✅] 2. DPO Baseline
python comparative_study/02a_DPO_Baseline/Llama3_BF16.py \
    --mode full \
    --base_model kapilw25/llama3-8b-pku-sft-baseline-bf16

# [⏳] 3. CITA Baseline
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py \
    --mode full \
    --base_model kapilw25/llama3-8b-pku-dpo-sft-bf16
```

---

## Training Configuration

| Model | Dataset | Instruction | Train Samples | Val Samples | Epochs | Checkpoints |
|-------|---------|-------------|---------------|-------------|--------|-------------|
| **SFT** | PKU | ❌ None | 10,831 | 1,204 | 1.0 | 5 (20% intervals) |
| **DPO** | PKU | ❌ None | 10,831 | 1,204 | 1.0 | 5 (20% intervals) |
| **CITA** | PKU | ✅ Metadata | 10,831 | 1,204 | 1.0 | 5 (20% intervals) |

**Expected Results** (from previous PKU training):
- SFT: margin ~1.5, accuracy ~75%
- DPO: margin ~5.5, accuracy ~90%
- CITA: margin ~6.6, accuracy ~90% (+16.5% over DPO)

---

## Validation Steps

### 1. Check Dataset Counts
```bash
source venv_CITA/bin/activate
python3 -c "
from comparative_study.0c_utils.data_prep.loader_pku import load_pku_combined_clear_contrast

split = load_pku_combined_clear_contrast(val_split=0.1)
print(f'Train: {len(split[\"train\"]):,}')  # Expected: 10,831
print(f'Val: {len(split[\"test\"]):,}')     # Expected: 1,204
"
```

### 2. Test Script Imports
```bash
python3 -c "from comparative_study.01a_SFT_Baseline.Llama3_BF16 import train_sft_baseline"
python3 -c "from comparative_study.02a_DPO_Baseline.Llama3_BF16 import train_dpo_baseline"
python3 -c "from comparative_study.03a_CITA_Baseline.Llama3_BF16 import train_cita_baseline"
```

### 3. Check TensorBoard Metrics
```bash
tensorboard --logdir tensorboard_logs/
# Open http://localhost:6006
# Verify: eval_loss (SFT), rewards/margins (DPO/CITA)
```

---

## Next Steps

1. ✅ Complete Phase 1-5 (restore PKU scripts, update loaders)
2. ⏳ Run SANITY training (validate setup)
3. ⏳ Run FULL training (1.0 epoch each, ~51 min total)
4. ⏳ Evaluate on 3,684 both-unsafe prompts (see `next_steps_eval_toxicity.md`)
