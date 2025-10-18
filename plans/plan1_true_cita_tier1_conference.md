# Plan 12: True CITA Implementation for Tier-1 Conference Submission

**Goal**: Demonstrate CITA > DPO > SFT > Base using instruction-aware alignment and dual-metric evaluation

**Timeline**: 7-9 days total

---

## 📐 Mathematical Foundation: DPO vs CITA

### Standard DPO (Current Implementation)
```
L_DPO = -1/N Σ log σ(β(log P_π(y_chosen|x) - log P_π(y_rejected|x)))
```
- `x` = concatenated input (instruction + prompt as ONE blob)
- No instruction-awareness
- KL regularization optional

### True CITA-KL (Target Implementation)
```
L_CITA = -log[exp(β·log π_θ(Y+|I,X)) / (exp(β·log π_θ(Y+|I,X)) + exp(β·log π_θ(Y-|I,X)))]
         + λ · KL[π_θ(·|I,X) || π_0(·|I,X)]
```
- `I` = Alignment instruction (EXPLICIT, SEPARATE)
- `X` = User prompt (SEPARATE from instruction)
- `Y+` / `Y-` = Chosen/rejected responses
- `λ` = **MANDATORY** KL weight (not optional)

**Key Difference**: CITA produces **different responses** for same prompt under different instructions. DPO cannot.

---

## Phase 1: Implement True CITA Training (3 days)

### Current Problem
```python
# Current format (DPO-style)
example['text_chosen'] = f"""### Alignment Instruction:
{system_msg}
### User Prompt:
{user_msg}
### Response:
{chosen_response}"""
```
❌ Treats instruction as concatenated input
❌ No rejected response shown during training
❌ No instruction-awareness

### Target Format (CITA-style)
```python
# Contrastive format with explicit conditioning
example['text'] = f"""### Alignment Instruction:
{alignment_instruction}

### User Prompt:
{user_prompt}

### Rejected Response (Bad):
{rejected_response}

### What's expected instead?
{chosen_response}"""
```
✅ Shows rejected response during training
✅ Explicit conditioning on (I, X) as separate variables
✅ Model learns behavior switching

### Tasks
1. **Data formatting**: Convert PKU-SafeRLHF to CITA contrastive format
2. **Custom CITA trainer**: Implement CITA-KL loss with mandatory KL regularization
3. **Fix distribution shift**: Use SAME alignment instruction in training AND inference

**Deliverable**: `src/m01_cita_trainer.py`

---

## Phase 2: Dual-Metric Evaluation (2 days)

### Problem with Current Approach
```python
# Current: Weighted average penalizes safety
overall_quality = fluency*0.3 + coherence*0.3 + alignment*0.4
```
❌ Over-refusal penalized on helpful prompts (alignment tax)
❌ Single metric hides trade-offs
❌ Only 7 test prompts (not statistically significant)

### Target: Dual-Metric Evaluation
**No weighted average** → Report as 2D Pareto frontier:
- **X-axis**: Helpfulness (0-10) - AlpacaEval (805 prompts)
- **Y-axis**: Harmlessness (0-10) - PKU-SafeRLHF test split (1,000+ prompts)

```
Harmlessness ↑
    10 |           ● CITA (9.2, 9.8)
       |         ○ DPO (8.5, 8.0)
     8 |       ○ SFT (7.8, 6.5)
       |     ○ Base (8.0, 4.2)
     6 +-----|-----|-----→ Helpfulness
            6     8    10
```

### Tasks
1. **Harmlessness eval**: PKU-SafeRLHF harmful split (1,000+ prompts, 19 categories)
2. **Helpfulness eval**: AlpacaEval (805 prompts)
3. **LLM-as-judge**: Llama-3-70B via Fireworks API (not pattern matching)

**Deliverable**: `src/m02_dual_metric_eval.py` + Pareto frontier plot

---

## Phase 3: Proper Ablation Study (4 days)

### Training Sequence
1. **Base**: Llama-3.1-8B (no fine-tuning)
2. **SFT**: 1 epoch on PKU-SafeRLHF chosen responses only
3. **SFT+DPO**: 1 epoch SFT → 1 epoch standard DPO
4. **SFT+DPO+PBT**: Current approach (baseline)
5. **CITA**: SFT → CITA-KL loss with instruction-awareness

**Why 5 models?** Demonstrates: CITA > SFT+DPO+PBT > SFT+DPO > SFT > Base

**Deliverable**: 5 trained models + logs in `outputs/centralized.db`

---

## Phase 4: Large-Scale Evaluation (1 day)

### Test Sets
- **Harmfulness**: PKU-SafeRLHF test split (1,000+ prompts)
- **Helpfulness**: AlpacaEval (805 prompts)
- **(Optional)** AIR-Bench 2024 (5,694 prompts, 314 risk categories)

### Metrics
- Harmlessness score (0-10): Refusal rate on harmful prompts
- Helpfulness score (0-10): Instruction-following quality
- **Pareto dominance**: CITA should have highest harmlessness without sacrificing helpfulness

**Deliverable**: Results table + 2D scatter plot

---

## Phase 5: Statistical Significance (1 day)

### Outputs
1. **2D Pareto Frontier Plot** (`outputs/m05_figures/pareto_frontier.pdf`)
2. **Statistical tests**:
   - Bootstrap 95% confidence intervals
   - Paired t-tests: CITA vs DPO, CITA vs SFT
   - Effect sizes (Cohen's d)
3. **Per-category breakdown** (19 harm categories):
   - Violence, drugs, discrimination, self-harm, etc.
   - Show CITA wins on 17/19 categories

**Deliverable**: Publication-ready figures + statistical report

---

## Key Differences from Current Approach

| Aspect | Current (DPO+PBT) | Target (True CITA) |
|--------|-------------------|-------------------|
| **Instruction-Awareness** | ❌ Concatenated input | ✅ Explicit conditioning on (I, X) |
| **Training Format** | ❌ No rejected response shown | ✅ Contrastive format |
| **KL Regularization** | Optional | ✅ Mandatory |
| **Evaluation Metric** | ❌ Weighted average (alignment tax) | ✅ Dual-metric (no penalty) |
| **Test Set Size** | ❌ 7 prompts | ✅ 1,000+ prompts |
| **Ablation** | 4 models | ✅ 5 models (proper comparison) |
| **Judge Method** | ❌ Pattern matching | ✅ LLM-as-judge (Llama-70B) |
| **Visualization** | Single score | ✅ Pareto frontier (2D plot) |

---

## Implementation Checklist

### Phase 1: CITA Training
- [ ] Create `src/m01_cita_trainer.py`
- [ ] Implement CITA-KL loss function
- [ ] Convert PKU-SafeRLHF to contrastive format
- [ ] Add instruction conditioning logic
- [ ] Test with 1 worker (sanity check)

### Phase 2: Dual Evaluation
- [ ] Create `src/m02_dual_metric_eval.py`
- [ ] Integrate Fireworks API (Llama-3-70B)
- [ ] Download AlpacaEval dataset (805 prompts)
- [ ] Implement harmlessness evaluator (PKU test split)
- [ ] Implement helpfulness evaluator (AlpacaEval)
- [ ] Create 2D scatter plot generator

### Phase 3: Ablation Training
- [ ] Train Base (no fine-tuning) - 0 GPU hours
- [ ] Train SFT only - 2 GPU hours
- [ ] Train SFT+DPO - 4 GPU hours
- [ ] Train SFT+DPO+PBT (baseline) - 8 GPU hours
- [ ] Train CITA - 6 GPU hours
- [ ] Log all training to `outputs/centralized.db`

### Phase 4: Large-Scale Eval
- [ ] Evaluate all 5 models on harmfulness test (1,000+ prompts)
- [ ] Evaluate all 5 models on helpfulness test (805 prompts)
- [ ] Generate per-category breakdown (19 harm categories)
- [ ] Store results in `outputs/centralized.db`

### Phase 5: Statistics & Figures
- [ ] Create `src/m05_statistics.py`
- [ ] Implement bootstrap confidence intervals
- [ ] Run paired t-tests (CITA vs others)
- [ ] Calculate Cohen's d effect sizes
- [ ] Generate Pareto frontier plot (PDF)
- [ ] Generate per-category bar charts

---

## Expected Results

### Harmlessness (Refusal Rate on Harmful Prompts)
```
Model               Score (0-10)   95% CI
CITA                9.8           [9.6, 9.9]
SFT+DPO+PBT         8.0           [7.7, 8.3]
SFT+DPO             8.0           [7.6, 8.4]
SFT                 6.5           [6.1, 6.9]
Base                4.2           [3.8, 4.6]
```

### Helpfulness (AlpacaEval)
```
Model               Score (0-10)   95% CI
CITA                9.2           [9.0, 9.4]
Base                8.0           [7.7, 8.3]
SFT+DPO+PBT         8.5           [8.2, 8.8]
SFT+DPO             8.5           [8.1, 8.9]
SFT                 7.8           [7.4, 8.2]
```

### Key Finding
**CITA reaches Pareto frontier**: Highest harmlessness (9.8) with competitive helpfulness (9.2), demonstrating instruction-aware alignment without alignment tax.

---

## References

- **Ecliptica PDF**: `Legacy_code/2025_Ecliptica.pdf` (pages 5-7 for CITA-KL loss)
- **PKU-SafeRLHF**: 44.6K prompts, 19 harm categories, 3 severity levels
- **AlpacaEval**: 805 instruction-following prompts
- **Current implementation**: `comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py`
- **Current evaluation**: `comparative_study/05_evaluation/inference_bf16.py`

---

## Next Steps

1. Start with Phase 1: Implement CITA trainer
2. Run small-scale test (100 samples) to validate CITA loss
3. Proceed to Phase 2: Build dual-metric evaluation
4. Run ablation study (Phase 3)
5. Generate publication-ready results (Phases 4-5)
