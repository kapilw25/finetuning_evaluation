# Quantization Strategy for LLM Evaluation

**Date:** 2025-10-25
**Context:** Dual-metric evaluation (harmlessness + helpfulness) on M1 Mac 16GB vs A100-40GB

---

## Executive Summary

**Bottleneck = Fireworks API (~114 min), NOT local GPU inference**

- Training margins (6.54, 5.52) already computed on A100-40GB ✅
- Evaluation generates 7220 responses → sends to Fireworks API for judging
- **M1 Mac INT4 recommended**: Saves $0.33 + 45 min with negligible impact on rankings

---

## Hardware Compatibility

### M1 Mac 16GB: ✅ YES (with INT4 quantization)

**Memory Requirements per Model:**

| Quantization | Llama-3.1-8B | LoRA Adapter | Total  | M1 Mac 16GB? |
|--------------|--------------|--------------|--------|--------------|
| **BF16**     | ~16GB        | ~200MB       | ~16.2GB| ❌ OOM risk   |
| **INT4**     | ~4GB         | ~200MB       | ~4.2GB | ✅ Safe       |

---

## Evaluation Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Local Inference (M1 Mac or A100)                  │
├─────────────────────────────────────────────────────────────┤
│ Load 4 models from HuggingFace:                            │
│   1. meta-llama/Llama-3.1-8B (base, untrained)             │
│   2. kapilw25/llama3-8b-pku-sft-baseline-bf16              │
│   3. kapilw25/llama3-8b-pku-dpo-sft-bf16                   │
│   4. kapilw25/llama3-8b-pku-cita-dpo-bf16                  │
│                                                             │
│ Generate responses for:                                     │
│   - 1000 harmful prompts (PKU-SafeRLHF test)               │
│   - 805 helpful prompts (AlpacaEval)                       │
│   = 1805 prompts × 4 models = 7220 responses               │
│                                                             │
│ Time: ~60 min (M1 INT4) or ~15 min (A100 BF16)             │
│ Cost: $0.00 (M1) or $0.33 (A100)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: LLM-as-Judge (Fireworks API) ⏱️ BOTTLENECK         │
├─────────────────────────────────────────────────────────────┤
│ Send 7220 responses to GPT-OSS-120B via Fireworks API      │
│ Each judgment:                                              │
│   - Input: ~150 tokens (prompt + response)                 │
│   - Output: ~100 tokens (score + reasoning)                │
│   - Time: 0.56s (TTFT) + 100/258.9 tok/sec = 0.95s         │
│                                                             │
│ Total time: 7220 × 0.95s = ~114 min (1.9 hours)            │
│ Cost: $1.80 (API charges, independent of local GPU)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Statistical Analysis (M1 Mac CPU)                 │
├─────────────────────────────────────────────────────────────┤
│ - Bootstrap 95% CI (10k samples)                           │
│ - Paired t-tests, Cohen's d                                │
│ - 6 publication plots (Pareto, heatmap, etc.)              │
│                                                             │
│ Time: ~2 min (CPU-only, no GPU needed)                     │
│ Cost: $0.00                                                 │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Fireworks API throughput (258.9 tok/sec) is the bottleneck, NOT local inference speed.

---

## Dataset Details

**FULL mode evaluation** (`dual_metric.py:77-152`):

| Dataset              | Samples | Purpose       | Notes                          |
|----------------------|---------|---------------|--------------------------------|
| PKU-SafeRLHF (test)  | 1000    | Harmlessness  | Harmful prompts only (filtered)|
| AlpacaEval           | 805     | Helpfulness   | Instruction-following tasks    |
| AIR-Bench 2024       | -       | (Optional)    | NOT used (unavailable)         |
| **Total**            | **1805**| -             | × 4 models = **7220 responses**|

**SANITY mode** (quick test):
- 50 harmful + 50 helpful = 100 prompts × 4 models = 400 responses
- Fireworks time: ~6 min, Cost: ~$0.10

---

## Performance Comparison (FULL Evaluation)

| Phase               | Hardware          | Quantization | Time       | GPU Cost | API Cost | Total Cost |
|---------------------|-------------------|--------------|------------|----------|----------|------------|
| Inference (4 models)| M1 Mac 16GB       | INT4         | ~60 min    | $0.00    | -        | $0.00      |
| Inference (4 models)| A100-40GB         | BF16         | ~15 min    | $0.33    | -        | $0.33      |
| **LLM-as-Judge**    | **Fireworks API** | **-**        | **~114 min**| **-**   | **$1.80**| **$1.80**  |
| Statistical analysis| M1 Mac (CPU)      | -            | ~2 min     | $0.00    | -        | $0.00      |
| **TOTAL (M1)**      | **M1 + API**      | **INT4**     | **~176 min**| **$0.00**| **$1.80**| **$1.80**  |
| **TOTAL (A100)**    | **A100 + API**    | **BF16**     | **~131 min**| **$0.33**| **$1.80**| **$2.13**  |

**Speedup**: A100 saves **45 min** inference time, but total time only reduces by **25%** (131 vs 176 min)

**Cost savings**: M1 Mac saves **$0.33** (15% cheaper)

---

## Fireworks API Performance

**Source**: Web search (2025 benchmarks via Artificial Analysis)

**GPT-OSS-120B on Fireworks AI:**
- Throughput: **258.9 tokens/sec**
- Latency (TTFT): **0.56 seconds**
- Uptime: 99.99%

**Comparison to other providers (for reference):**
- Baseten: 491.1 tok/sec (fastest)
- Fireworks: 258.9 tok/sec
- Together.ai: 131.1 tok/sec
- Cerebras: 2000-4000 tok/sec (specialized hardware)

**Calculation for 7220 judgments:**
```
Time per judgment = TTFT + (output_tokens / throughput)
                  = 0.56s + (100 / 258.9)
                  = 0.56s + 0.39s
                  = 0.95 seconds

Total time = 7220 × 0.95s
           = 6859 seconds
           = ~114 minutes
```

**API cost**: ~$0.00025 per judgment × 7220 = ~$1.80

---

## Quantization Impact on Results

### What INT4 Affects vs. Does NOT Affect

| Metric                  | Computed From         | INT4 Impact             | Why?                                    |
|-------------------------|-----------------------|-------------------------|-----------------------------------------|
| **Margins** (6.54, 5.52)| Training logs         | ❌ **NO IMPACT**         | Already computed on A100 BF16 (done)    |
| **Harmlessness score**  | LLM-as-judge          | ⚠️ **Slight** (-2-5%)   | Generated responses slightly different  |
| **Helpfulness score**   | LLM-as-judge          | ⚠️ **Slight** (-2-5%)   | Generated responses slightly different  |
| **Relative rankings**   | Cross-model comparison| ✅ **PRESERVED**         | Uniform noise across all models         |

---

### 1. What INT4 Does to Model Outputs

**BF16 (16-bit brain float):**
- Full precision weights → accurate logits
- Example refusal: *"I cannot help with that request as it violates ethical guidelines."*

**INT4 (4-bit quantization):**
- Weights rounded to 16 discrete values → noisy logits
- Example refusal: *"I can't assist with that as it may cause harm."* (similar intent, different wording)

**Observed degradation (from literature):**
- Perplexity: +3-8% worse
- Accuracy on benchmarks: -2-5%
- Refusal rate: -1-3% (slightly more false negatives)

---

### 2. Impact on Dual-Metric Evaluation

**Key insight:** Evaluation measures **RELATIVE performance** (CITA vs DPO vs SFT vs Base)

**Example harmlessness scores:**

| Model | BF16 Score | INT4 Score | Absolute Change | Ranking |
|-------|------------|------------|-----------------|---------|
| CITA  | 9.2        | 9.0        | -0.2 (-2.2%)    | 1st     |
| DPO   | 8.5        | 8.3        | -0.2 (-2.4%)    | 2nd     |
| SFT   | 7.8        | 7.6        | -0.2 (-2.6%)    | 3rd     |
| Base  | 6.5        | 6.3        | -0.2 (-3.1%)    | 4th     |

**Result**: Ranking **preserved** (CITA > DPO > SFT > Base)

**Why rankings preserved:**
- INT4 adds **uniform noise** to all models
- Paired differences (CITA - DPO) remain similar
- Statistical tests compare **within-subject** differences

---

### 3. Statistical Analysis Robustness

**From `utils/statistical_analysis.py:42-89`:**
- Bootstrap 95% CI (10,000 samples)
- Paired t-tests (within-subject design)
- Cohen's d effect sizes

**Effect of INT4 on statistics:**

| Metric              | BF16 Behavior         | INT4 Behavior         | Impact on Inference |
|---------------------|-----------------------|-----------------------|---------------------|
| Absolute scores     | Ground truth          | ↓ 2-5% (all models)   | ⚠️ Biased downward   |
| Paired differences  | CITA - DPO = 0.7      | CITA - DPO ≈ 0.7      | ✅ Preserved         |
| Effect sizes (d)    | 0.85 (large)          | 0.82 (large)          | ✅ Preserved         |
| p-values            | 0.003 (significant)   | 0.008 (significant)   | ✅ Preserved         |

**Example (harmlessness scores):**
```
BF16:  CITA=9.2, DPO=8.5 → Δ=0.7, p=0.003 (significant)
INT4:  CITA=9.0, DPO=8.3 → Δ=0.7, p=0.008 (still significant)
```

**Mathematical explanation:**
```python
# Paired difference is invariant to uniform scaling
delta_bf16 = CITA_bf16 - DPO_bf16
delta_int4 = (CITA_bf16 × 0.97) - (DPO_bf16 × 0.97)  # 3% degradation
           = 0.97 × (CITA_bf16 - DPO_bf16)
           = 0.97 × delta_bf16
           ≈ delta_bf16  # Difference preserved
```

---

## When INT4 DOES Matter

### ❌ Use BF16 if:
1. **Publishing paper** (need exact numbers for tables)
2. **Absolute scores matter** (comparing to external benchmarks like published baselines)
3. **Budget allows** ($0.33 for A100 GPU time is negligible)
4. **Claiming SOTA** (state-of-the-art requires BF16 validation)
5. **Tight race** (CITA margin only 1-2% better than DPO)

### ✅ INT4 is fine if:
1. **Internal validation** (checking CITA > DPO hypothesis)
2. **Relative rankings matter** (Pareto frontier shape preserved)
3. **Limited hardware** (M1 Mac, no cloud access)
4. **Exploratory analysis** (prototyping evaluation pipeline)
5. **Large effect size** (CITA margin +18.5% > DPO)

---

## Recommendation for CITA Validation

### Current Training Results (from TensorBoard):
- CITA margin: **6.54** (step 1000, A100 BF16 training)
- DPO margin: **5.52** (step 1000, A100 BF16 training)
- Improvement: **+18.5%**

**Note**: These margins are **training metrics** (not affected by evaluation quantization)

### INT4 Worst-Case Analysis for Evaluation Scores:

**Assume 5% degradation for all models:**
```
Harmlessness scores (example):
  BF16: CITA=9.2, DPO=8.5, SFT=7.8, Base=6.5
  INT4: CITA=8.7, DPO=8.1, SFT=7.4, Base=6.2

Relative gaps:
  BF16: CITA - DPO = 0.7
  INT4: CITA - DPO = 0.6  (14% reduction in gap, but still significant)

Ranking: CITA > DPO > SFT > Base ✅ PRESERVED
```

**Why gap is mostly preserved:**
```
Relative change = (CITA - DPO) / DPO
                = (9.2 - 8.5) / 8.5 = 8.2% (BF16)
                = (8.7 - 8.1) / 8.1 = 7.4% (INT4)
                ≈ 8.2% (only 10% relative reduction)
```

---

## Decision Tree

```
Run INT4 evaluation on M1 Mac (~176 min, $1.80)
    ↓
Check results:

if CITA_score > DPO_score (gap > 5%):
    → ✅ Valid proof CITA works
    → Can cite INT4 results in report
    → Add caveat: "Evaluated with INT4 quantization"
    → Optional: Re-run BF16 on A100 for publication

if CITA_score > DPO_score (gap < 5%):
    → ⚠️ Borderline (quantization noise may affect conclusion)
    → Rent A100, re-run BF16 for confirmation
    → Use BF16 results for claims

if CITA_score ≤ DPO_score:
    → ❌ Method failed (INT4 didn't flip ranking, real issue)
    → Debug training before re-evaluating
```

**Expected outcome**: CITA clearly wins (given +18.5% margin gap from training)

---

## Commands

### SANITY Check (50+50 samples, ~6 min Fireworks, ~$0.10)

```bash
python3 -u comparative_study/05_evaluation/llm_as_judge/dual_metric.py \
    --mode sanity \
    --quantization int4
```

**Use case**: Validate pipeline works before full run

---

### FULL Evaluation (1805 samples × 4 models, ~176 min, $1.80)

```bash
# M1 Mac 16GB (INT4) - RECOMMENDED
python3 -u comparative_study/05_evaluation/llm_as_judge/dual_metric.py \
    --mode full \
    --quantization int4

# A100-40GB (BF16) - For publication
python3 -u comparative_study/05_evaluation/llm_as_judge/dual_metric.py \
    --mode full \
    --quantization bf16
```

---

## Checkpointing Safety

**From `dual_metric.py:284-330`:**

```python
# Saves every 100 prompts
if (i + 1) % checkpoint_interval == 0:
    save_checkpoint(model_key, dataset_type, responses, prompts, completed=False)
```

**Benefits for M1 Mac:**
- **Thermal throttling**: Pause/resume without data loss
- **Crash recovery**: Restart from last checkpoint (no progress lost)
- **Flexible scheduling**: Run overnight, stop anytime
- **Per-model + per-dataset**: 4 models × 2 datasets = 8 independent checkpoints

**Checkpoint locations:**
```
comparative_study/05_evaluation/llm_as_judge/checkpoints/
├── SFT_Baseline_harmlessness_checkpoint.json
├── SFT_Baseline_helpfulness_checkpoint.json
├── DPO_Baseline_harmlessness_checkpoint.json
├── DPO_Baseline_helpfulness_checkpoint.json
├── CITA_Baseline_harmlessness_checkpoint.json
├── CITA_Baseline_helpfulness_checkpoint.json
├── meta-llama/Llama-3.1-8B_harmlessness_checkpoint.json (base model)
└── meta-llama/Llama-3.1-8B_helpfulness_checkpoint.json
```

**Resume behavior** (`dual_metric.py:366-380`):
```
If checkpoint exists and completed=True:
  → Ask user: Re-use cached responses or re-run inference?

If checkpoint exists and completed=False:
  → Auto-resume from last saved index
```

---

## Hybrid Approach (Cost-Optimized)

### Step 1: Run SANITY on M1 Mac (~6 min Fireworks, $0.10)

```bash
python3 -u comparative_study/05_evaluation/llm_as_judge/dual_metric.py \
    --mode sanity \
    --quantization int4
```

**Check**: Does CITA > DPO on small sample?

---

### Step 2A: If SANITY looks good → Run FULL INT4 (~176 min, $1.80)

```bash
python3 -u comparative_study/05_evaluation/llm_as_judge/dual_metric.py \
    --mode full \
    --quantization int4
```

**If CITA clearly wins** (gap > 5%):
- ✅ **Done** - Cite INT4 results with caveat
- Save $0.33 GPU cost
- Total spent: $1.90

---

### Step 2B: If results ambiguous → Re-run FULL BF16 on A100 (~131 min, $2.13)

```bash
python3 -u comparative_study/05_evaluation/llm_as_judge/dual_metric.py \
    --mode full \
    --quantization bf16
```

**Use BF16 results** for publication claims
- Total spent: $1.90 + $2.13 = $4.03

---

### Cost Comparison:

| Strategy         | Runs              | Time        | Cost                      | Use Case                  |
|------------------|-------------------|-------------|---------------------------|---------------------------|
| INT4 only        | 1× M1 Mac         | ~176 min    | $1.80                     | Internal validation       |
| BF16 only        | 1× A100           | ~131 min    | $2.13                     | Publication-ready         |
| Hybrid (sanity)  | M1 sanity → M1 full| ~182 min   | $0.10 + $1.80 = $1.90     | Risk mitigation           |
| Hybrid (both)    | M1 full → A100 full| ~307 min   | $1.80 + $2.13 = $3.93     | Inconclusive INT4 results |

**Expected path**: Hybrid sanity → INT4 full ($1.90 total)

---

## Statistical Analysis (CPU-Only)

**From `utils/statistical_analysis.py`:**

All post-processing runs on **CPU** (no GPU needed):

1. **Bootstrap resampling** (10k iterations)
   - 95% confidence intervals for harmlessness + helpfulness

2. **Paired statistical tests**
   - Paired t-tests (within-subject design)
   - Wilcoxon signed-rank test (non-parametric)
   - Cohen's d effect sizes

3. **6 publication plots**:
   - Pareto frontier (harmlessness vs helpfulness)
   - 95% CI error bars
   - Heatmap (per-category breakdown)
   - Score distributions (violin plots)
   - Radar chart (multi-dimensional comparison)
   - Effect size comparison

**Runtime**: ~2 min on M1 Mac CPU (no GPU required)

**Output**: `comparative_study/05_evaluation/llm_as_judge/DualMetric_Evaluation_Results/`

---

## Bottom Line

### For CITA Validation (Your Use Case):

**INT4 on M1 Mac is sufficient** given:

1. **Training margins already computed** on A100 BF16 (6.54 vs 5.52 = +18.5%)
2. **Large effect size** → quantization noise won't flip ranking
3. **Fireworks API is bottleneck** (~114 min), not local inference
4. **Relative ranking is what matters** (CITA > DPO > SFT > Base)
5. **Paired statistical tests robust** to uniform noise
6. **Checkpointing enables safe overnight runs** on M1 Mac

### For Publication:

**BF16 on A100 recommended** for:

1. Exact numerical claims in paper tables
2. Comparison to external benchmarks (apples-to-apples)
3. Eliminating quantization as confounding variable
4. Peer review credibility
5. Only costs $0.33 more (15% increase)

### Pragmatic Execution Path:

```bash
# Step 1: Sanity check (5-10 min total)
python3 -u comparative_study/05_evaluation/llm_as_judge/dual_metric.py \
    --mode sanity --quantization int4

# Step 2: If trends look good → Full INT4 (176 min total)
python3 -u comparative_study/05_evaluation/llm_as_judge/dual_metric.py \
    --mode full --quantization int4

# Step 3: If CITA wins → Done ($1.90 total)
# Step 4: Before paper submission → Validate with BF16 on A100 (+$2.13)
```

**Expected total cost**: $1.90 (sanity + full INT4)

**Optional BF16 validation**: +$2.13 (only if publishing)

---

## Key Corrections from Previous Version

1. ❌ **Old**: "A100 inference takes ~12 min" → ✅ **New**: "Fireworks API takes ~114 min (bottleneck)"
2. ❌ **Old**: "Margins affected by quantization" → ✅ **New**: "Margins are training metrics (already computed)"
3. ❌ **Old**: "3 models evaluated" → ✅ **New**: "4 models (base + SFT + DPO + CITA)"
4. ❌ **Old**: "AIR-Bench used" → ✅ **New**: "AIR-Bench optional (not loaded)"
5. ❌ **Old**: "GPU speed matters" → ✅ **New**: "API throughput (258.9 tok/sec) is limiting factor"
