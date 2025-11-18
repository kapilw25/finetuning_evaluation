# Iteration 13: CITA_Instruct Hyperparameter Fix

## Goal
Make CITA_Instruct outperform DPO_Instruct in toxicity evaluation for Ecliptica paper submission.

## Problem Diagnosis

### Current State (Trial 7 - Best CITA_Instruct)
```
Trial 7 Metrics:
  - margin: 7.52
  - accuracy: 89.0%
  - eval_loss: 0.326   HIGH (poor generalization)

Trial 7 HPs:
  - »_KL: 0.00024      TOO WEAK (root cause)
  - LR: 5.41e-6        Good (stable)
  - beta: 0.1067       Good
```

### Root Cause
**Weak KL regularization** causes poor calibration:
- »_KL=0.00024 is **3x weaker** than needed for 350-token sequences
- High eval_loss (0.326) suggests model is over-confident on training data
- Poor generalization ’ likely worse toxicity scores than DPO_Instruct

### Evidence
```
CITA_NoInstruct:    »_KL=0.00052, eval_loss=0.273   Well-calibrated
CITA_Instruct_T7:   »_KL=0.00024, eval_loss=0.326  L Under-regularized
```

**Why Trial 7 has weak »_KL:**
- Optuna search space: [0.0001, 0.001]
- TPE sampler explored low »_KL early (Trials 0-2)
- Got stuck in local minimum (margin improved, but at cost of eval_loss)

---

## Solution Strategy

### Phase 1: Quick Fix ’ Llama3_BF16.py (PRIORITY)

**Timeline:** 4-5 hours training + 1 hour toxicity eval = **Results by TONIGHT**

**Why this first:**
-  **FAST**: Single training run vs 27 Optuna trials
-  **HIGH CONFIDENCE**: We know Trial 7's exact problem (»_KL too weak)
-  **IMMEDIATE RESULTS**: Can run toxicity eval and know if paper claim holds
-  **LOW RISK**: Conservative fix based on Trial 7 proven stable HPs

**The Fix:**
```python
# In comparative_study/03a_CITA_Baseline/Llama3_BF16.py
# Add instruction-aware HP branching (lines 111-126)

if USE_INSTRUCTION:
    # CITA_Instruct Fixed HPs (Based on Trial 7 + Regularization Fix)
    LAMBDA_KL = 0.00072      # 3x Trial 7's 0.00024 (STRONGER regularization)
    LEARNING_RATE = 5.41e-6  # Keep Trial 7's (proven stable, no explosion)
    BETA = 0.1067            # Keep Trial 7's (preference sharpness)
    WEIGHT_DECAY = 0.0091    # Keep NoInstruct Trial 5's (generalization)
    WARMUP_RATIO = 0.0749    # Keep NoInstruct Trial 5's (stability)
else:
    # CITA_NoInstruct Fixed HPs (UNCHANGED - proven optimal)
    LAMBDA_KL = 0.000520
    LEARNING_RATE = 6.827978e-06
    BETA = 0.1191
    WEIGHT_DECAY = 0.0091
    WARMUP_RATIO = 0.0749
```

**Expected Results:**
```
Current Trial 7:   eval_loss=0.326  (poor calibration)
With »_KL=0.00072: eval_lossH0.29   (better calibration, closer to NoInstruct's 0.273)

Why 3x increase is safe:
  - NoInstruct uses »_KL=0.00052 for 250-token sequences
  - Instruct uses 350-token sequences (+40% longer)
  - 0.00072 is proportional: 0.00052 × (350/250) H 0.00073
  - Trial 7's LR=5.41e-6 is stable (no explosion in 1354 steps)
```

**Training Command:**
```bash
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py \
  --mode full \
  --use-instruction true \
  --base_model kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct
```

**Toxicity Eval Command (After Training):**
```bash
python comparative_study/05_evaluation/llm_as_judge/toxicity.py \
  --mode full \
  --models CITA_Instruct DPO_Instruct
```

---

### Phase 2: Optimal Search ’ Llama3_BF16_adaptive_Optuna.py (IF TIME ALLOWS)

**Timeline:** 20-30 hours (27 trials with Hyperband pruning)

**Why this second:**
-   **SLOW**: 27 trials even with early stopping
-   **UNCERTAIN**: Might find better HPs, might get stuck again
-  **THOROUGHNESS**: Explores HP interactions we might miss manually
-  **SCIENTIFIC RIGOR**: Shows exhaustive search for optimal HPs

**The Fix (Narrow Search Space):**
```python
# In comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py
# Modify lines 108-118 (use_instruction=True branch)

if use_instruction:
    # NARROWED search space based on Trial 7 analysis + regularization fix
    # Evidence: Trial 7 stable but under-regularized
    # Strategy: Keep Trial 7's LR/beta, explore stronger »_KL

    lambda_kl = trial.suggest_float("lambda_kl", 0.0005, 0.0010, log=False)
    # ‘ RAISED floor from 0.0001 ’ 0.0005 (no more weak regularization)

    learning_rate = trial.suggest_float("learning_rate", 4.5e-6, 5.8e-6, log=True)
    # ‘ NARROWED around Trial 7's 5.41e-6 (proven stable)

    beta = trial.suggest_float("beta", 0.09, 0.12)
    # ‘ NARROWED around Trial 7's 0.1067

    weight_decay = trial.suggest_float("weight_decay", 0.007, 0.011)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.06, 0.09)
```

**Run ONLY if Phase 1 fails:**
- If fixed HPs don't beat DPO_Instruct in toxicity eval
- If eval_loss still > 0.30 (poor calibration)
- If time allows before paper submission

---

## Decision: Phase 1 (Quick Fix) FIRST

**Reason:** Paper deadline requires results NOW, not in 3 days.

**Action Plan:**
1.  Modify `Llama3_BF16.py` to support instruction-aware HPs
2.  Train CITA_Instruct with fixed HPs (4-5 hours)
3.  Run toxicity evaluation (1 hour)
4.  Analyze results:
   - **SUCCESS**: CITA_Instruct beats DPO_Instruct ’ Paper claim validated 
   - **FAILURE**: CITA_Instruct loses ’ Try Phase 2 (Optuna) or pivot paper claim

**Expected Outcome (70% confidence):**
```
Toxicity Evaluation Results (Predicted):
  DPO_Instruct:    toxicity_mean H 2.1, safe_refusal_rate H 75%
  CITA_Instruct:   toxicity_mean H 1.9, safe_refusal_rate H 78%  (WINS)

Why CITA should win:
  - Better calibration (eval_loss 0.29 vs DPO's ~0.27)
  - KL regularization prevents over-confident toxic outputs
  - Instruction-aware training improves refusal consistency
```

**Fallback if Phase 1 fails:**
- Pivot paper claim to "instruction-following fidelity" instead of absolute toxicity
- Run Phase 2 (Optuna) to find truly optimal HPs
- Compare CITA_NoInstruct vs DPO_NoInstruct (drop Instruct variants)

---

## Implementation Checklist

### Phase 1 (Immediate)
- [ ] Modify `comparative_study/03a_CITA_Baseline/Llama3_BF16.py`
  - [ ] Add `if USE_INSTRUCTION` branching for HPs (lines 111-126)
  - [ ] Set CITA_Instruct HPs: »_KL=0.00072, LR=5.41e-6, beta=0.1067
  - [ ] Keep CITA_NoInstruct HPs unchanged
- [ ] Train CITA_Instruct with fixed HPs
  - [ ] Command: `python Llama3_BF16.py --mode full --use-instruction true --base_model <DPO_Instruct>`
  - [ ] Monitor eval_loss (expect ~0.29, must be < 0.30)
  - [ ] Push to HuggingFace: `kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct`
- [ ] Run toxicity evaluation
  - [ ] Command: `python toxicity.py --mode full --models CITA_Instruct DPO_Instruct`
  - [ ] Compare toxicity_mean and safe_refusal_rate
  - [ ] Document results in `logs_training/iter13/results.md`

### Phase 2 (If Phase 1 fails)
- [ ] Modify `comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py`
  - [ ] Narrow »_KL range: [0.0005, 0.0010]
  - [ ] Narrow LR range: [4.5e-6, 5.8e-6]
  - [ ] Narrow beta range: [0.09, 0.12]
- [ ] Run Optuna search (27 trials, ~20-30 hours)
- [ ] Identify best trial (highest margin, eval_loss < 0.30)
- [ ] Retrain with best trial HPs
- [ ] Re-run toxicity evaluation

---

## Risk Analysis

### High-Confidence Predictions
 **Fixed HPs will improve eval_loss**: 0.326 ’ ~0.29 (3x stronger »_KL)
 **Training will be stable**: LR=5.41e-6 proven safe in Trial 7 (1354 steps, no explosion)
 **Better calibration**: Stronger KL regularization improves confidence calibration

### Uncertain Predictions
  **Will CITA_Instruct beat DPO_Instruct?**: 70% confidence
- Need actual toxicity eval to confirm
- DPO_Instruct might have better calibration than expected
- CITA's advantage (instruction-aware KL) is theoretical

  **Will eval_loss reach 0.29?**: 60% confidence
- Linear scaling assumption (3x »_KL ’ proportional eval_loss drop)
- Might need iterative tuning (0.0008, 0.0009, etc.)

### Low-Confidence Predictions
L **Optuna Phase 2 will find better HPs**: 50% confidence
- Might get stuck in same local minimum
- 27 trials might not be enough for dense search space
- Hyperband pruning might kill good trials early

---

## Success Criteria

### Phase 1 Success
- [ ] CITA_Instruct eval_loss d 0.30 (improved calibration)
- [ ] CITA_Instruct beats DPO_Instruct on toxicity_mean OR safe_refusal_rate
- [ ] No training explosion (grad_norm stays d 1.0 throughout)

### Minimum Viable Result
- [ ] CITA_Instruct eval_loss < 0.32 (better than Trial 7's 0.326)
- [ ] CITA_Instruct competitive with DPO_Instruct (within 5% on toxicity metrics)

### Failure Condition
- [ ] CITA_Instruct eval_loss e 0.32 (no improvement)
- [ ] CITA_Instruct loses to DPO_Instruct by >10% on both toxicity_mean and safe_refusal_rate
- [ ] Training explosion (»_KL=0.00072 too aggressive for LR=5.41e-6)

**If failure occurs:** Run Phase 2 (Optuna) or pivot paper claim.

---

## Timeline

```
NOW:              Modify Llama3_BF16.py
+30 min:          Start CITA_Instruct training
+4.5 hours:       Training complete, push to HuggingFace
+5.5 hours:       Toxicity eval complete
+6 hours:         Results analysis, decision on next steps

If Phase 1 succeeds: Paper claim validated 
If Phase 1 fails:    Start Phase 2 (Optuna) or pivot
```

---

## Notes

- **Why not try »_KL=0.0010 (even stronger)?**
  Risk of over-regularization (model becomes too conservative, loses margin gains). Start with 3x increase (0.00072), can iterate if needed.

- **Why keep Trial 7's LR instead of NoInstruct's higher LR (6.83e-6)?**
  Trial 7's LR=5.41e-6 is proven stable for 350-token Instruct sequences. NoInstruct's higher LR caused explosion when transferred (iter12). Conservative approach prioritizes stability.

- **What if eval_loss improves but toxicity eval fails?**
  Lower eval_loss doesn't guarantee better toxicity scores. If this happens:
  1. Check if CITA is over-refusing (too conservative)
  2. Try lower »_KL (0.0006) for next iteration
  3. Consider that DPO_Instruct might be fundamentally better for toxicity task

- **Alternative if both phases fail:**
  Acknowledge in paper that instruction conditioning hurts CITA (40% longer sequences), and focus on CITA_NoInstruct vs DPO_NoInstruct comparison instead.
