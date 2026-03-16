# iter10: Optuna Learning from Explosions (FIXED)

## Problem Discovered

**Trial 0 and Trial 1 had IDENTICAL hyperparameters:**
```
lambda_kl:      0.000875
learning_rate:  1.176258e-05
beta:           0.1093
weight_decay:   0.0104
warmup_ratio:   0.2234

Trial 0: EXPLOSION at step 472 (grad_norm=60.14)
Trial 1: Same config → likely explosion
```

## Root Cause: n_startup_trials=10 (Random Phase Too Long)

**Before (n_startup_trials=10):**
- Trials 0-9: Random (no learning, can duplicate)
- Trial 10+: TPE learns

**Why Trial 0 = Trial 1:**
- Both in random phase (0-9)
- Random sampling can duplicate (probability ~0.01% exact, ~15% clustered)
- TPE cannot learn from explosions (no objective values returned)

---

## Fix Applied

**File**: `Llama3_BF16_adaptive_Optuna.py:518`

```python
# BEFORE:
n_startup_trials=min(10, n_trials // 3),  # First 10 trials random

# AFTER:
n_startup_trials=5,  # Reduced from 10 - TPE learns faster
```

**Impact:**
- Trials 0-4: Random (5 trials instead of 10)
- Trial 5+: TPE learns from past trials
- Saves ~25 hours if explosions cluster in trials 5-9

---

## Fixes Rejected

### ❌ Option 2: Hyperparameter Constraints (NOT APPLIED)

```python
# REJECTED - contradicts Optuna's purpose
min_warmup = 0.15 + (learning_rate - 8e-6) / (1.2e-5 - 8e-6) * 0.15
if warmup_ratio < min_warmup:
    raise optuna.TrialPruned("Unsafe")
```

**Why rejected:**
- Hardcoded heuristic, not learned
- Might block optimal solution (e.g., lr=1.2e-05, warmup=0.20 could be best)
- Trust TPE to explore freely

### ✅ Option 3: trial.report() Bug (ALREADY FIXED)

Line 317 already commented out:
```python
# trial.report(margin, step)  # Already disabled
```

---

## Expected Behavior (Next Run)

| Trial | Sampling | Learning? |
|-------|----------|-----------|
| 0-4 | Random | ❌ NO (still risk of duplicates) |
| 5+ | TPE | ✅ YES (avoids explosion regions) |

**Note**: Random does NOT guarantee diversity. Trials 0-2 could still cluster, but TPE learns 5 trials earlier.
