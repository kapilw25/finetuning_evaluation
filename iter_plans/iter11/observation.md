# iter11: Two Critical Issues Preventing Paper Rejection

## Issue 1: Unfair Warmup Ratio (iter10 → iter11)

**Log:** `logs_training/iter11/CITA_Adaptive_iter10_UNFAIR_warmup_20to35pct.log` (1 completed, 2 exploded, 1 in-progress @ 26%)

### Problem Discovered

**CITA gets unfair training advantage over SFT/DPO baselines:**

| Method | Warmup Steps | Warmup % | Total Steps |
|--------|-------------|----------|-------------|
| SFT    | 100 (fixed) | **7.4%** | 1354        |
| DPO    | 100 (fixed) | **7.4%** | 1354        |
| CITA   | 342 (ratio=0.253) | **25.3%** | 1354 |

**Impact:** CITA gets **3.4x more warmup** than baselines � any toxicity improvement could be from training stability advantage, not the CITA method itself.

---

## Root Cause: Optuna HP Space Misaligned with Industry Standard

**Industry Standard (WebSearch findings):**
- HuggingFace Alignment Handbook: `warmup_ratio=0.1` (10%)
- Google ML Guide: Warmup d 10% of max_train_steps
- Consensus: **6-10% warmup** for transformers

**Previous CITA Optuna HP Space (iter10):**
```python
warmup_ratio = trial.suggest_float("warmup_ratio", 0.20, 0.35)  # 20-35%, 2-3x standard!
learning_rate = trial.suggest_float("learning_rate", 8e-6, 1.2e-5, log=True)
```

**Trial Results from iter10 run:**
- Trial 0: warmup=22.3%, LR=1.18e-05 � EXPLODED @ step 472 (grad_norm=60.14)
- Trial 1: warmup=30.6%, LR=8.19e-06 � COMPLETED (only survivor due to excessive warmup)
- Trial 2: warmup=22.7%, LR=1.19e-05 � EXPLODED @ ~700

**Observation:** Trial 1 survived ONLY because it had 3x the warmup of SFT/DPO baselines, not because CITA is superior.

---

## Empirical Evidence: Eval Metrics @ Each 20% Checkpoint

**Trial 0** (warmup=22.3%, LR=1.18e-05) - **EXPLODED @ step 472**
| Epoch | Progress | eval_loss | eval_accuracy | eval_margin |
|-------|----------|-----------|---------------|-------------|
| 0.2   | 20%      | 0.3403    | 85.5%         | 3.52        |
| **→** | **EXPLOSION** | **grad_norm=60.14** |

**Trial 1** (warmup=30.6%, LR=8.19e-06) - **COMPLETED** ✓
| Epoch | Progress | eval_loss | eval_accuracy | eval_margin |
|-------|----------|-----------|---------------|-------------|
| 0.2   | 20%      | 0.3953    | 81.1%         | 1.78        |
| 0.4   | 40%      | 0.3230    | 88.0%         | 6.30        |
| 0.6   | 60%      | 0.2711    | 90.2%         | 8.07        |
| 0.8   | 80%      | 0.3129    | 89.0%         | 9.38        |
| 1.0   | 100%     | 0.3312    | **89.2%**     | **9.70**    |

**Trial 2** (warmup=22.7%, LR=1.19e-05) - **EXPLODED @ step ~700**
| Epoch | Progress | eval_loss | eval_accuracy | eval_margin |
|-------|----------|-----------|---------------|-------------|
| 0.2   | 20%      | 0.3325    | 85.5%         | 3.24        |
| 0.4   | 40%      | 0.3311    | **90.1%**     | **11.57**   |
| **→** | **EXPLOSION** | **grad_norm=1.0** | **(BETTER than Trial 1!)** |

**Trial 3** (warmup=24.4%, LR=9.05e-06) - **IN_PROGRESS (step 364, 26%)**
| Epoch | Progress | eval_loss | eval_accuracy | eval_margin |
|-------|----------|-----------|---------------|-------------|
| 0.2   | 20%      | 0.3641    | 83.6%         | 2.45        |
| ...   | ...      | ...       | ...           | ...         |

---

### Critical Insight: Warmup Determines Survival, Not Performance

**Summary of All Trials:**
| Trial | Warmup % | Learning Rate | Status | Max Epoch Reached |
|-------|----------|---------------|--------|-------------------|
| 0     | 22.3%    | 1.18e-05      | EXPLODED | 0.2 (20%)    |
| 1     | **30.6%** | 8.19e-06   | **COMPLETED** | **1.0 (100%)** |
| 2     | 22.7%    | 1.19e-05      | EXPLODED | 0.4 (40%)    |
| 3     | 24.4%    | 9.05e-06      | IN_PROGRESS | 0.2+ (26%)  |

**Pattern:** Trial 1 has the HIGHEST warmup (30.6%) and is the ONLY survivor.

**Trial 2 vs Trial 1 @ 40% checkpoint:**
- Trial 2: 90.1% accuracy, 11.57 margin (warmup=22.7%) → **EXPLODED**
- Trial 1: 88.0% accuracy, 6.30 margin (warmup=30.6%) → **COMPLETED**

**Conclusion:** Trial 2 had BETTER metrics but still exploded due to insufficient warmup. Trial 1 survived with worse metrics because excessive warmup (30.6%) stabilized training. This proves warmup is the confounding variable, not CITA's effectiveness.

---

## Fix Applied

**File**: `comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py:100-108`

```python
# BEFORE (iter10):
warmup_ratio = trial.suggest_float("warmup_ratio", 0.20, 0.35)
learning_rate = trial.suggest_float("learning_rate", 8e-6, 1.2e-5, log=True)

# AFTER (iter11):
warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15)  # Fair: matches SFT/DPO ~10%
learning_rate = trial.suggest_float("learning_rate", 5e-6, 8e-6, log=True)  # Lower to compensate
```

**Rationale:**
1. **Warmup:** 0.05-0.15 centers on 10% (industry standard), matches SFT/DPO's 7.4%
2. **Learning Rate:** Lowered from 8e-6�1.2e-5 to 5e-6�8e-6 to prevent explosions with shorter warmup
3. **Fair Comparison:** CITA now gets same training stability conditions as baselines

---

## Issue 2: Wrong Objective Function

**Log:** `logs_training/iter11/CITA_Adaptive_eval_rewards_chosen_NOT_LOSS.log` (4.75 hours runtime, 2 completed, 1 in-progress @ 31%, 1 unknown)

### Problem: Optuna Optimized for `eval_rewards/chosen` Instead of `eval_loss`

**Original objective function:**
```python
return (
    final_margin,      # Objective 1: maximize margin
    final_accuracy,    # Objective 2: maximize accuracy
    -final_chosen      # Objective 3: minimize chosen reward (WRONG!)
)
```

**Why this is WRONG:**
- `eval_rewards/chosen` = KL divergence from reference policy (more negative = more drift)
- Does NOT penalize overfitting - only measures divergence
- Trial 0 shows the problem: `chosen=-13.28` looked "acceptable" to Optuna, but `eval_loss=0.5026` is terrible

### Empirical Evidence: Trial 0 vs Trial 1

**Trial 0** (warmup=6.56%, LR=7.82e-06) - **OVERFITTED**
| Checkpoint | eval_loss | eval_accuracy | eval_margin | chosen |
|------------|-----------|---------------|-------------|--------|
| 20% | 0.3348 | 86.6% | 4.84 | -3.87 |
| 100% | **0.5026 (+50%)** | 87.6% | **13.01** | **-13.28** |

**Training metrics:** Loss -94.7%, L_KL -186.10 (MASSIVE divergence)

**Trial 1** (warmup=12.08%, LR=5.14e-06) - **HEALTHY**
| Checkpoint | eval_loss | eval_accuracy | eval_margin | chosen |
|------------|-----------|---------------|-------------|--------|
| 20% | 0.3737 | 82.7% | 2.34 | -1.85 |
| 100% | **0.3030 (-19%)** | **88.1%** | 5.25 | -3.86 |

**Training metrics:** Loss -76.9%, L_KL -58.36 (moderate, healthy)

### Why Trial 0 is Scientifically Invalid

**Overfitting indicators:**
1. **eval_loss increasing +50%** while training_loss dropping -94.7%
2. **KL divergence -186** (model drifted massively from reference)
3. **High margin (13.01) achieved through extreme probabilities**, not better separation

**Trial 1 shows healthy training:**
- eval_loss DECREASING consistently (0.3737 → 0.3030)
- KL divergence moderate (-58.36 vs -186.10)
- Margin improving realistically (2.34 → 5.25)

### Fix Applied

**File:** `comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py`

**Modified objective function (Line 436-442):**
```python
# BEFORE
return (
    final_margin,
    final_accuracy if final_accuracy else 0.0,
    -final_chosen if final_chosen else 0.0  # WRONG: doesn't penalize overfitting
)

# AFTER
return (
    final_margin,
    final_accuracy if final_accuracy else 0.0,
    -final_eval_loss if final_eval_loss else 0.0  # CORRECT: penalizes overfitting
)
```

**Why `eval_loss` is correct:**
- DPO loss = `-log(σ(β × margin)) + λ_KL × KL`
- Directly measures training objective
- Increases when model overfits (extreme probabilities)
- Trial 0 would be penalized: `values = [13.01, 0.876, -0.5026]` (high loss penalty)
- Trial 1 would be favored: `values = [5.25, 0.881, -0.3030]` (low loss penalty)

### Action Required

**MUST restart from scratch:** Database contains incompatible objective values (Trials 0-2 use `-chosen`, future trials use `-eval_loss`). Different scales → TPE sampler gets confused. Sunk cost: 4.75 hours (~$7), but prevents wasting 50+ hours learning wrong patterns.

---

## Expected Behavior (Next Run)

**If CITA matches SFT/DPO toxicity with fair warmup:**
- CITA method is scientifically valid
- Improvement comes from contrastive loss, not training tricks

**If CITA requires >10% warmup to match SFT/DPO:**
- This is a valid negative finding
- CITA is more fragile/sensitive than baselines
- Report honestly: "CITA requires extended warmup for stability"

---

## Issue 3: Trial Selection Based on Stability (Visual Analysis)

**After applying fixes from Issues 1 & 2**, a corrected Optuna search was run with fair warmup (5-15%), lower LR (5e-6 to 8e-6), and correct objective `[margin, accuracy, -eval_loss]`. This produced 19 trials with stable training.

**Key Finding:** Trial 5 selected over Trial 0 despite worse raw metrics, due to training stability.

---

### Visual Evidence: TensorBoard Comparisons

#### Plot 1: Trial 0 Instability (tensorboard_2.png)
![Trial 0 Instability](plots/tensorboard_2.png)

**Observation:**
- **eval/loss (bottom-left):** Trial 0 (purple) shows **loss INCREASING** after step 800 (0.318 → 0.516) = training divergence
- **eval/rewards/accuracies (top-left):** Trial 0 reaches 95.3% (highest) but unstable
- **eval/rewards/margins (bottom-right):** Trial 0 achieves 13.5 margin (highest) but likely through extreme probabilities (overfitting)

**Conclusion:** Trial 0 rejected due to loss divergence in second half of training.

---

#### Plot 2: Top 3 Trials vs DPO (tensorboard_2_5_9_vsDPO.png)
![Top Trials vs DPO](plots/tensorboard_2_5_9_vsDPO.png)

**Comparison:**
- **eval/loss (bottom-left):**
  - Trial 2 (orange): 0.282 final, shows **plateau** around steps 600-900
  - Trial 5 (pink): **0.279 final, smoothest descent** (no plateaus)
  - Trial 9 (orange): 0.289 final (higher than Trial 5)
  - DPO (orange): 0.220 baseline

- **eval/rewards/margins (bottom-right):**
  - Trial 2: 8.79 (high but with plateau risk)
  - Trial 5: **6.95 (balanced, stable growth)**
  - Trial 9: 7.29
  - DPO: 5.82 baseline

**Conclusion:** Trial 5 shows best convergence pattern (smooth, monotonic improvement).

---

#### Plot 3: Head-to-Head Trial 2 vs 5 (tensorboard_2_5_vsDPO.png)
![Trial 2 vs 5](plots/tensorboard_2_5_vsDPO.png)

**Direct Comparison:**
- **eval/loss (bottom-left):**
  - Trial 5 (pink): Consistent decrease to **0.279**
  - Trial 2 (orange): Plateaus at 0.28 (stagnation)

- **eval/rewards/accuracies (top-left):**
  - Trial 5 (pink): **89.5%** (2.6pp better)
  - Trial 2 (orange): 86.9%

**Decision:** Trial 5 wins on both stability AND accuracy.

---

#### Plot 4: Full Search Space - All 19 Trials (tensordboard_All_trials_CITA.png)
![All Trials](plots/tensordboard_All_trials_CITA.png)

**Search Space Coverage:**
- **eval/loss (bottom-left):** Wide variance (0.3 to 0.6), showing diverse HP exploration
- **eval/rewards/margins (bottom-right):** Divergent behaviors (5-13 range)
- **eval/rewards/accuracies (top-left):** Most trials cluster 86-89%, Trial 0 outlier at 95%

**Pattern:** Trial 5 (pink) sits in stable cluster with best loss convergence among non-overfitting trials.

---

### Final Selection: Trial 5 Hyperparameters

| Trial | Accuracy | Margin | Loss | Warmup | LR | Status |
|-------|----------|--------|------|--------|----|----|
| 0 | **95.3%** | **13.5** | 0.318→**0.516** | 6.6% | 7.82e-06 | ❌ Divergence |
| 2 | 86.9% | 8.79 | 0.282 | - | - | ❌ Plateau |
| **5** | **89.5%** | **6.95** | **0.279** | **7.49%** | **6.83e-06** | ✅ **Selected** |
| 9 | 86.2% | 7.29 | 0.289 | - | - | ❌ Lower acc |

**Trial 5 Selected For:**
1. **Smoothest training curves** (no plateaus, no divergence)
2. **Best eval_loss convergence** among stable trials (0.279)
3. **Good accuracy** (89.5%, middle of stable cluster)
4. **Balanced margin** (6.95, not extreme like Trial 0's 13.5)
5. **Fair warmup** (7.49% ≈ SFT/DPO's 7.4%)

**Trade-off Accepted:** -5.8pp accuracy vs Trial 0, but gained production reliability (stable training curves suggest robust HPs).

---

## Next Steps

1. **Stop and delete:** Kill current run, delete database + trial outputs
2. **Restart with corrected objective:** Fair warmup (5-15%), Lower LR (5e-6 to 8e-6), Objective `[margin, accuracy, -eval_loss]`
3. **Expected:** Trials with stable eval_loss favored over high-margin overfitters
4. **Monitor:** eval_loss should decrease/stabilize, KL divergence < -100

---

## References

- `logs_training/iter11/CITA_Adaptive_iter10_UNFAIR_warmup_20to35pct.log` - Issue 1 evidence
- `logs_training/iter11/CITA_Adaptive_eval_rewards_chosen_NOT_LOSS.log` - Issue 2 evidence
- `logs_training/iter10/observation.md` - Previous fix (n_startup_trials=10→5)
- `comparative_study/01a_SFT_Baseline/Llama3_BF16.py:265` - SFT warmup_steps=100
- `comparative_study/02a_DPO_Baseline/Llama3_BF16.py:303` - DPO warmup_steps=100
- `comparative_study/03a_CITA_Baseline/Llama3_BF16.py:136` - CITA WARMUP_RATIO=0.253
- WebSearch: HuggingFace Alignment Handbook, Google ML Guide (10% warmup standard)
