# Negative KL Issue - Root Cause Analysis

**Date:** 2025-10-25
**File:** `comparative_study/0c_utils/cita_trainer.py:221-223`

---

## Bug Location

```python
# Line 221-223 in cita_trainer.py
kl_chosen = policy_chosen_logps - reference_chosen_logps
kl_rejected = policy_rejected_logps - reference_rejected_logps
loss_kl = (kl_chosen.mean() + kl_rejected.mean()) / 2
```

---

## Mathematical Error

**Current (WRONG) implementation:**
```
KL = mean(log π_θ(y|x) - log π_ref(y|x))
   = mean(log(π_θ / π_ref))
```

**Correct KL divergence:**
```
KL(π_θ || π_ref) = E[π_θ(y|x) · log(π_θ(y|x) / π_ref(y|x))]
                  = ∑_y π_θ(y|x) · [log π_θ(y|x) - log π_ref(y|x)]
```

**Missing:** Probability weighting term `π_θ(y|x)`

---

## Why Negative Values Occur

**Observed values (step 49):**
- `policy_chosen_logps = -97.79`
- `reference_chosen_logps = -93.91`

**Computation:**
```
kl_chosen = -97.79 - (-93.91) = -3.88  ❌
```

**Interpretation:**
- Policy model assigns **lower** probability than reference
- DPO-tuned model is **less confident** than base model
- Mathematically impossible for true KL (KL ≥ 0 always)

---

## Root Cause in Stacked Training

| Model | State | Logprob Magnitude |
|-------|-------|-------------------|
| Reference (base) | Untrained | -93.91 (less negative = more confident) |
| Policy (DPO-tuned) | After 1000 DPO steps | -97.79 (more negative = less confident) |

**Why policy is less confident:**
- DPO training optimizes for **preference margin**, not likelihood
- Model learns to distinguish safe/unsafe, but becomes uncertain overall
- Policy has seen **harder preference pairs** during DPO training

**Result:** `log π_policy < log π_ref` → systematic negative "KL"

---

## Fix Options

### Option 1: True KL Divergence (Theoretically Correct)
```python
# Convert log probs to probs
policy_probs_chosen = torch.exp(policy_chosen_logps)
policy_probs_rejected = torch.exp(policy_rejected_logps)

# KL(π || π_ref) = π · log(π / π_ref)
kl_chosen = policy_probs_chosen * (policy_chosen_logps - reference_chosen_logps)
kl_rejected = policy_probs_rejected * (policy_rejected_logps - reference_rejected_logps)
loss_kl = (kl_chosen.mean() + kl_rejected.mean()) / 2
```

**Pros:** Mathematically rigorous, always ≥ 0
**Cons:** Requires exp() (numerical instability for large negative logprobs)

---

### Option 2: Reverse KL (Practical)
```python
# Reverse KL: KL(π_ref || π)
kl_chosen = reference_chosen_logps - policy_chosen_logps
kl_rejected = reference_rejected_logps - policy_rejected_logps
loss_kl = (kl_chosen.mean() + kl_rejected.mean()) / 2
```

**Pros:** Simple, always positive, standard in PPO/RLHF
**Cons:** Not true forward KL, penalizes different behavior

---

### Option 3: Absolute Difference (Symmetric)
```python
kl_chosen = (policy_chosen_logps - reference_chosen_logps).abs()
kl_rejected = (policy_rejected_logps - reference_rejected_logps).abs()
loss_kl = (kl_chosen.mean() + kl_rejected.mean()) / 2
```

**Pros:** Always positive, symmetric penalty
**Cons:** Not true KL, lacks probabilistic interpretation

---

## Recommended Fix: Option 2 (Reverse KL)

**Justification:**
1. Standard practice in PPO (Schulman et al. 2017)
2. Computationally stable (no exp required)
3. Penalizes policy drift from reference
4. Always positive (solves negative KL issue)

**Code change:**
```python
# Line 221-223 in cita_trainer.py
kl_chosen = reference_chosen_logps - policy_chosen_logps  # SWAP ORDER
kl_rejected = reference_rejected_logps - policy_rejected_logps
loss_kl = (kl_chosen.mean() + kl_rejected.mean()) / 2
```

---

## Impact on Training

**Before fix (negative KL):**
- Loss gradient pushes model AWAY from reference (wrong direction)
- `loss = λ_DPO · L_DPO + λ_KL · (-4.51)` → reduces total loss incorrectly
- Model diverges from reference instead of staying close

**After fix (positive KL):**
- Loss gradient pulls model TOWARD reference (correct direction)
- `loss = λ_DPO · L_DPO + λ_KL · 4.51` → proper regularization
- Model balances DPO objective with staying close to reference

---

## Verification Steps

1. Apply fix to `cita_trainer.py:221-223`
2. Restart training from scratch
3. Monitor `cita/loss_kl` - should be **positive** at all steps
4. Check gradient norm - should stabilize (< 10)
5. Verify margin - should increase without negative samples

**Expected KL values after fix:** 0.5 - 5.0 (typical range for LLM fine-tuning)
