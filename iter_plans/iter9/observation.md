# Iter9: Fixed eval_steps + Explosion Detection

**Date:** 2025-11-11

## Problem (iter7/iter8)
- FULL mode explodes at step ~255 (margin: 8→180, loss: 0.3→6.8)
- SANITY mode stable (warmup_ratio=0.253 works for 407 steps)
- Root cause: warmup_ratio fixed, but needed more frequent monitoring

## Changes

### 1. Fixed eval_steps (Llama3_BF16.py:256)
```python
# checkpoint_interval = int(total_steps * 0.2)  # OLD: SANITY=80, FULL=270
checkpoint_interval = 80  # NEW: Fixed (consistent monitoring)
```
- SANITY: 80 steps (unchanged)
- FULL: 80 steps (was 270 → now 17 evals vs 5)
- Rationale: Catch explosion earlier

### 2. Explosion detector (cita_trainer.py:414-419)
```python
if total_norm > 50.0:
    raise ValueError(f"Training exploded (grad_norm={total_norm:.2f})")
```
- Auto-stops training if grad_norm > 50.0
- Saves compute (iter7 reached grad_norm ~100-200)

## Iter8 vs Iter9 Configuration Comparison

**IDENTICAL warmup_ratio:**
- iter9: `Warmup ratio: 25.3%`
- iter8: `Warmup ratio: 25.3%`
- Both use same total_steps: **1,353 steps**

**DIFFERENT eval_steps:**
- **iter9:** `Checkpoint interval: 80 steps` (EXPERIMENT: fixed at 80 for consistent eval)
- **iter8:** `Checkpoint interval: 270 steps` (20% of training)

**Summary:**
- Same warmup configuration (25.3% = 343 warmup steps out of 1,353 total)
- Same all other hyperparameters (LR=1.185e-05, beta=0.1133, lambda_kl=0.001010)
- **ONLY difference:** eval_steps (80 vs 270)

**Evaluation frequency comparison:**
- iter8: 5 evals total (steps 270, 540, 810, 1080, 1350)
- iter9: 17 evals total (every 80 steps)
- **Result:** iter9 has **3.4× more frequent evaluations** than iter8

## Rollback (if fails)
```python
# Llama3_BF16.py:256
checkpoint_interval = int(total_steps * 0.2)  # Uncomment

# cita_trainer.py:414-419
# Delete explosion detector block
```

## Test Commands
```bash
# SANITY (verify no regression)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode sanity \
    --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct

# FULL (main experiment)
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full \
    --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct
```

## Results: Iter9 vs Iter8 FULL Training Comparison

### Iter9 (checkpoint_interval=80) - ❌ EXPLODED at step 496

| Epoch | Step | Margin | Accuracy | Loss  | Status |
|-------|------|--------|----------|-------|--------|
| 0.06  | 80   | 0.16   | 72.6%    | 0.625 | ✅ |
| 0.12  | 160  | 1.25   | 79.1%    | 0.439 | ✅ |
| 0.18  | 240  | 2.56   | 82.8%    | 0.372 | ✅ |
| 0.24  | 320  | 4.89   | 85.2%    | 0.365 | ✅ |
| 0.30  | 400  | 8.58   | 87.5%    | 0.383 | ✅ |
| **0.35** | **480** | **20.27** | **85.4%** | **0.959** | ⚠️ **JUMP** |
| **0.37** | **496** | **-** | **-** | **-** | ❌ **grad_norm=70.65** |

**Explosion:** Training auto-stopped by explosion detector (grad_norm > 50.0)

---

### Iter8 (checkpoint_interval=270) - ❌ EXPLODED at step 542

| Epoch | Step | Margin | Accuracy | Loss  | Status |
|-------|------|--------|----------|-------|--------|
| 0.20  | 270  | 3.13   | 84.5%    | 0.351 | ✅ |
| **0.40** | **540** | **31.93** | **87.3%** | **1.251** | ❌ **EXPLOSION** |

**Explosion:** Crashed at evaluation checkpoint

---

## Critical Findings

**1. Both iter8 and iter9 exploded - warmup_ratio fix NOT sufficient**
- iter8: Exploded at step 540 (epoch 0.40)
- iter9: Exploded at step 496 (epoch 0.37)
- iter9 exploded EARLIER despite same warmup configuration

**2. Explosion warning signs detected in iter9:**
- Epoch 0.30→0.35: Margin jumped 2.4× (8.58→20.27)
- Epoch 0.35→0.37: Grad norms spiked to 70.65
- Loss jumped 2.5× (0.383→0.959)

**3. Frequent eval_steps caught explosion earlier:**
- iter9: Detected at step 496 (saved 860 steps of wasted compute)
- iter8: Detected at step 540 (no early warning)
- Explosion detector worked - auto-stopped at grad_norm=70.65

**4. Step 255 explosion zone (iter7) was cleared:**
- iter9 at step 240: margin=2.56, stable ✅
- iter9 at step 320: margin=4.89, stable ✅
- warmup_ratio=0.253 DID fix iter7's step 255 issue

---

## Conclusion

✅ **warmup_ratio fix SOLVED iter7's step 255 explosion**

❌ **But REVEALED a different instability at ~epoch 0.35-0.40**

⚠️ **Gradient clipping broken in both** (grad_norms 10-70 despite max=1.0)

**Next action:** Run Optuna retuning ($88, 59 hours) to find stable hyperparameters for full 1.0 epoch training.
