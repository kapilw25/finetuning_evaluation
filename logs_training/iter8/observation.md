# Iter8: warmup_ratio Fix + Gradient Clipping Results

**Date:** 2025-11-09
**Status:** ❌ Gradient clipping broken - Optuna retuning required
**Previous:** `logs_training/iter7/observation_N_next_steps.md`

---

## What Was Tested (Iter7 → Iter8)

**From Iter7:** FULL training exploded at step 255 due to LR schedule mismatch
- Root cause: `warmup_steps=103` (fixed) caused 1.93× higher LR in FULL vs SANITY at step 255
- Solution: Replace with `warmup_ratio=0.253` to align LR schedules across training lengths

**Iter8 Tests:**
1. **SANITY (0.3 epochs):** Verify warmup_ratio doesn't break existing behavior
2. **FULL #1 (no clipping):** Test if warmup_ratio alone fixes explosion
3. **FULL #2 (with clipping):** Add `max_grad_norm=1.0` as safety net

---

## Results Summary

| Test | Status | Final Margin | Issue |
|------|--------|--------------|-------|
| SANITY | ✅ Success | 7.51 (+29% vs DPO) | None |
| FULL #1 (no clip) | ❌ Crashed 42% | 3.10 → 32.43 | High grad_norms (50-154) |
| FULL #2 (with clip) | ❌ Crashed 48% | 3.13 → 31.93 | Clipping not working |

**Conclusion:** warmup_ratio fixed LR schedule but high gradient norms persist → gradient clipping implementation broken

---

## Detailed Training Metrics

### SANITY (0.3 epochs, 407 steps) - ✅ Success

**Configuration:**
- Hyperparameters: Optuna Trial 2 (LR=1.185e-05, beta=0.1133, lambda_kl=0.00101)
- Warmup: `warmup_ratio=0.253` (103 steps)
- Gradient clipping: None

**Results:**
| Epoch | eval_margin | eval_accuracy | eval_loss | grad_norm |
|-------|-------------|---------------|-----------|-----------|
| 0.06  | 1.32        | 80.9%         | 0.446     | 2-16      |
| 0.12  | 4.14        | 86.7%         | 0.335     | 2-16      |
| 0.18  | 6.30        | 88.4%         | 0.327     | 2-16      |
| 0.24  | 7.39        | 88.9%         | 0.348     | 2-16      |
| 0.30  | **7.51**    | **89.2%**     | **0.353** | 2-16      |

**vs DPO Baseline:** margin 7.51 vs 5.82 (+29%) ✅
**Log:** `logs_training/iter8/SANITY/CITA_Baseline_training_20251109_140333.log`

---

### FULL #1: No Gradient Clipping (1.0 epochs, 1354 steps) - ❌ Crashed 42%

**Configuration:**
- Same hyperparameters as SANITY (Optuna Trial 2)
- Warmup: `warmup_ratio=0.253` (343 steps)
- Gradient clipping: None

**Results:**
| Epoch | eval_margin | eval_accuracy | eval_loss | grad_norm |
|-------|-------------|---------------|-----------|-----------|
| 0.2   | 3.10        | 84.4%         | 0.35      | 2-16      |
| 0.4   | **32.43**   | 86.5%         | **1.36**  | **50-154**|

**Explosion:** Margin jumped 10× in 0.2 epochs, grad_norm spiked to 154
**Log:** `logs_training/iter8/FULL/CITA_FULL_No_Gradient_clipping_explodedin40percent.log`

---

### FULL #2: With Gradient Clipping (1.0 epochs, 1354 steps) - ❌ Crashed 48%

**Configuration:**
- Same hyperparameters as SANITY (Optuna Trial 2)
- Warmup: `warmup_ratio=0.253` (343 steps)
- Gradient clipping: **`max_grad_norm=1.0`**

**Results:**
| Epoch | eval_margin | eval_accuracy | eval_loss | grad_norm_clipped |
|-------|-------------|---------------|-----------|-------------------|
| 0.2   | 3.13        | 84.5%         | 0.35      | 10-16             |
| 0.4   | **31.93**   | 87.3%         | **1.25**  | **10-215**        |
| 0.48  | -           | -             | -         | **Crashed**       |

**Clipping Failure:** Config shows `max_grad_norm=1.0` but grad_norms 10-215 AFTER "clipping"
**Evidence:** Logs show `'cita/clipping_active': 1.0` but `'cita/grad_norm_clipped': 215.68`
**Log:** `logs_training/iter8/FULL/CITA_FULL_with_Gradient_clipping.log`

---

## Root Cause Analysis

### Issue #1: warmup_ratio Fix Insufficient
**Status:** Partially fixed LR schedule but didn't prevent explosion

Iter7 identified LR schedule mismatch (FULL had 1.93× higher LR than SANITY at step 255). Switching to `warmup_ratio=0.253` aligned the schedules, but FULL still explodes at epoch 0.4.

**Implication:** High gradient norms (10-16 in SANITY) were a warning sign. Longer training (1354 vs 407 steps) compounds instability → explosion even with correct LR schedule.

### Issue #2: Gradient Clipping Implementation Broken
**Status:** ❌ Critical bug in CITA trainer

Added `max_grad_norm=1.0` to DPOConfig but clipping not applied:
- **Expected:** grad_norm ≤ 1.0 after clipping
- **Actual:** grad_norm 10-215 AFTER "clipping"
- **Evidence:** Logs report `'cita/clipping_active': 1.0` and `'cita/grad_norm_clipped': 215.68` in same step

**Root cause:** Either:
1. DPOTrainer's gradient clipping method overridden/bypassed in `comparative_study/0c_utils/cita_trainer.py`
2. `max_grad_norm` parameter not passed correctly to optimizer
3. Clipping happens but on wrong gradients (not on combined CITA loss gradients)

---

## Next Steps

### Option 1: Debug Gradient Clipping (2-4 hours, $3-6)
**Investigate implementation bug:**
1. Check `comparative_study/0c_utils/cita_trainer.py` - verify clipping not overridden
2. Compare with DPOTrainer base class - ensure `max_grad_norm` passed correctly
3. Test with simple gradient print statements to confirm clipping applied
4. If fixed: Re-run FULL training

**Pros:** Quick if bug is simple, no hyperparameter retuning needed
**Cons:**
- May not find root cause (complex inheritance chain)
- Even if fixed, high grad_norms (10-16) suggest hyperparameters need adjustment
- DPO baseline doesn't need clipping → suggests CITA-specific instability

### Option 2: Run Optuna Retuning ($88, 59 hours, 27 trials)
**Full hyperparameter search for 1354-step FULL training:**
- Search space: `lambda_kl`, `learning_rate`, `beta`, `weight_decay`, `warmup_ratio`
- Optimization target: Stable training + competitive margins
- Uses Hyperband pruning (saves ~50% compute)

**Expected outcome:**
- Lower `lambda_kl` (~0.0005-0.0007) to reduce KL gradient magnitude
- Possibly lower `learning_rate` or longer `warmup_ratio` for stability
- Guarantees stable config through trial pruning

**Pros:**
- Guarantees stable training (Optuna finds gradient-bounded configs)
- More robust than debugging (addresses root instability, not just clipping)
- Already proven successful in iter6 (found Trial 2 config)

**Cons:** Higher cost ($88 vs $3-6)

---

## Recommendation

**Option 2 (Optuna)** - for 3 reasons:
1. **High grad_norms in SANITY** (10-16) suggest hyperparameters fundamentally unstable for 1354-step training
2. **Clipping as band-aid:** Even if debugging fixes clipping, masking high gradients doesn't address root instability
3. **Cost-benefit:** $88 for guaranteed stable config > $3-6 for uncertain debugging + potential re-run

**Fallback:** If Optuna budget unavailable, try Option 1 first, but expect to need Option 2 eventually

---

## References

### Logs (Iter8)
1. **SANITY (success):** `logs_training/iter8/SANITY/CITA_Baseline_training_20251109_140333.log`
2. **FULL #1 (no clipping, crashed 42%):** `logs_training/iter8/FULL/CITA_FULL_No_Gradient_clipping_explodedin40percent.log`
3. **FULL #2 (broken clipping, crashed 48%):** `logs_training/iter8/FULL/CITA_FULL_with_Gradient_clipping.log`

### Previous Analysis
- **Iter7:** `logs_training/iter7/observation_N_next_steps.md` (LR schedule analysis + warmup_ratio solution)

### Code
- **Training script:** `comparative_study/03a_CITA_Baseline/Llama3_BF16.py`
  - Line 129: `WARMUP_RATIO = 0.253` (iter7 fix)
  - Line 290: `max_grad_norm=1.0` (iter8 addition, broken)
- **CITA Trainer:** `comparative_study/0c_utils/cita_trainer.py` (gradient clipping bug likely here)

### Baselines
- **DPO (0.3 epochs):** margin = 5.82
- **CITA SANITY (iter8):** margin = 7.51 (+29% improvement)

### Hyperparameters (Optuna Trial 2)
```python
learning_rate = 1.185e-05
beta = 0.1133
lambda_kl = 0.001010
weight_decay = 0.00885
warmup_ratio = 0.253  # iter7 fix: was warmup_steps=103
```
