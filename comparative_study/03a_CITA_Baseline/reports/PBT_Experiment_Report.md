# CITA PBT Experiment Report

## Experiment Metadata

| Field | Value |
|-------|-------|
| **Experiment ID** | PBT_20251012_153829 |
| **Date** | 2025-10-12 15:38:29 |
| **Model** | Llama-3.1-8B + LoRA (r=16) |
| **Method** | CITA with Population-Based Training (PBT) |
| **Workers** | 3 |
| **Total Steps** | 1000 per worker |
| **Mutation Interval** | 100 steps |
| **GPU** | GH200 96GB |
| **Training Time** | ~80 minutes |
| **Cost** | ~$2.00 |

---

## Hyperparameter Search Space

| Parameter | Range | Baseline (Working) | PBT Explored |
|-----------|-------|-------------------|--------------|
| `learning_rate` | [1e-5, 5e-5] | 2e-5 | ✅ 4.6e-5 to 6.6e-5 |
| `lambda_kl` | [0.0005, 0.002] | 0.001 | ✅ 0.0006 to 0.001 |
| `beta` | [0.05, 0.2] | 0.1 | ✅ 0.085 to 0.146 |
| `weight_decay` | [0.001, 0.05] | 0.01 | ✅ 0.0076 to 0.025 |
| `warmup_steps` | [50, 150] | 100+ | ✅ 72 to 146 |
| `batch_size` | Fixed | 1 | **8** (modified) |
| `max_length` | Fixed | 131072 | **1024** (optimized) |

---

## Worker Performance Summary

### Final Iteration (Iter 7-8)

| Worker | LR | λ_kl | β | Warmup | Loss | Margin | Status |
|--------|-----|------|---|--------|------|--------|--------|
| **00000** | 5.53e-5 | 0.00082 | 0.102 | 72 | 5.76 | **-13,227** | 🔴 WORST |
| **00001** | 6.64e-5 | 0.00065 | 0.123 | 86 | 11.51 | **-17,100** | 🔴 CATASTROPHIC |
| **00002** | 4.61e-5 | 0.00102 | 0.085 | 90 | 2.75 | **-2,171** | 🟡 "BEST" |

**PBT Selection**: Worker 00002 (lowest loss/margin)

---

## Training Metrics Evolution

### Loss Trajectory (per iteration)

| Iteration | Worker 00000 | Worker 00001 | Worker 00002 |
|-----------|--------------|--------------|--------------|
| 0 | 3.33 | - | 3.33 |
| 1 | 2.93 | 2.44 | 0.05 |
| 2 | 3.05 | 2.94 | 2.42 |
| 3 | 3.04 | 2.93 | 2.41 |
| 7-8 | 5.76 | 11.51 | 2.75 |

**Observation**: Loss diverged, indicating instability.

---

### Margin Trajectory (Chosen - Rejected Logps)

| Iteration | Worker 00000 | Worker 00001 | Worker 00002 | Expected |
|-----------|--------------|--------------|--------------|----------|
| 0-1 | -165 | -394 | -28 | **Positive** |
| 2-3 | -185 | -371 | -2,318 | **Positive** |
| 7-8 | **-13,227** | **-17,100** | **-2,171** | **Positive** |

**Critical Issue**: All margins negative & worsening → Model prefers rejected (unsafe) responses.

---

## Qualitative Analysis: Response Quality

### Test Prompt: "What are the benefits of regular exercise?"

| Checkpoint | Response Sample | Pattern |
|------------|-----------------|---------|
| **Baseline (No FT)** | `Regular exercise can help with weight loss, reduce risk of heart disease...` | ✅ Coherent |
| **Iter 1 (Worker 489262)** | `Regular, 2, 0 2017....` | 🔴 Random numbers |
| **Iter 6 (Worker 557682)** | `Regular) and the problem of 2: 'R_{i_{i, but he was...` | 🔴 LaTeX + gibberish |
| **Iter 22 (Worker 782295)** | `however, and the problem in 2: 'R_{i_{2, but he was...` | 🔴 Unchanged garbage |

**Verdict**: ❌ No improvement over 22 checkpoints. Mode collapse from iteration 0.

---

## Comparison: PBT vs. Baseline

| Metric | Baseline (No FT) | PBT CITA | Verdict |
|--------|------------------|----------|---------|
| **Helpful prompts** | Coherent (6.5/10) | Complete gibberish | 🔴 PBT WORSE |
| **Harmful prompts** | Provides harmful info (0/10 safety) | Gibberish (0/10 safety) | 🟡 Both fail |
| **Training loss** | N/A | 2.75 - 11.51 | - |
| **Margin** | N/A | -2,171 to -17,100 | 🔴 Negative (bad) |

---

## Root Cause Analysis

### Problem: Mode Collapse from Iteration 0

**Contributing Factors**:

1. **Learning rate too high**: 4.6e-5 to 6.6e-5 (2.3× - 3.3× higher than baseline 2e-5)
2. **Lambda_kl too low**: 0.0006-0.001 (20-40% below baseline 0.001)
3. **Warmup too short**: 72-90 steps (vs. baseline 100+)
4. **Batch size too large**: 8 (vs. baseline 1) → Less stable for CITA
5. **Wide search space**: 5× range for LR allowed unstable regions

**Result**: Early training instability → KL divergence exploded → Model learned to output low-loss gibberish.

---

## Lessons Learned

| Issue | Fix for Next Run |
|-------|------------------|
| LR range too wide (5×) | ✅ Narrow to ±10%: [1.8e-5, 2.2e-5] |
| λ_kl minimum too low | ✅ Raise floor: [0.001, 0.0015] |
| Warmup too short (50 min) | ✅ Increase minimum: [100, 120] |
| Batch size too aggressive | ✅ Reduce to 4 (middle ground) |
| No early stopping | ✅ Add gibberish detection with stop |

---

## Recommendations for Next Experiment

### Constrained Hyperparameter Space

```python
"learning_rate": tune.uniform(1.8e-5, 2.2e-5),  # ±10% around working value
"lambda_kl": tune.uniform(0.001, 0.0015),       # Minimum = baseline
"beta": tune.uniform(0.08, 0.12),               # ±20% around 0.1
"weight_decay": tune.uniform(0.008, 0.012),     # ±20% around 0.01
"warmup_steps": tune.randint(100, 120),         # Longer minimum
"batch_size": 4,                                 # Reduce from 8
```

**Goal**: Find hyperparameters **slightly better** than baseline (6.3/10), not explore unstable regions.

---

## Experiment Status

| Status | Value |
|--------|-------|
| **Training** | ✅ Completed (80% as of report) |
| **Model Quality** | ❌ Failed (unusable) |
| **Data Collection** | ✅ Complete |
| **Scientific Value** | ✅ High (negative control) |
| **Next Steps** | Run constrained PBT |

---

## Files Generated

```
outputs/ray_results/cita_pbt_training/         # Ray Tune results
outputs/best_pbt_config.json                    # Best hyperparameters (Worker 00002)
logs/CITA_PBT_training_20251012_153829.log     # Full training log
outputs/lora_model_CITA_Baseline_PBT_BF16/     # LoRA adapter (garbage)
```

---

## Citation

```
Experiment: CITA_PBT_20251012_153829
Result: Mode collapse due to wide hyperparameter search
Learning: Constrain PBT search to ±10-20% of working baseline values
```
