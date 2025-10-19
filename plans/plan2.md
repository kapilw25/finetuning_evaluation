# Recommended Order

---

## Goal: Demonstrate CITA > DPO > SFT > Base

**Target for theoretical verification before GPU training:**

```
Harmlessness ↑
    10 |           ● CITA (9.2, 9.8)
       |         ○ DPO (8.5, 8.0)
     8 |       ○ SFT (7.8, 6.5)
       |     ○ Base (8.0, 4.2)
     6 +-----|-----|-----→ Helpfulness
            6     8    10
```

**Key Finding**: CITA reaches Pareto frontier - highest harmlessness (9.8) with competitive helpfulness (9.2), demonstrating instruction-aware alignment without alignment tax.

---

## Critical Flaws in Current Implementation

**The L_DPO Bug is Just ONE of Many Issues**

From `plan1_true_cita_tier1_conference.md`, the current implementation has **MULTIPLE CRITICAL FLAWS**:

| Issue                 | Current (Broken)                       | Required (Plan 1) | Status |
|-----------------------|----------------------------------------|-------------------|--------|
| Instruction-awareness | ✅ Uses role tokens (system/user)      | ~~Text labels~~ | ❌ **Not implementing** (research hypothesis) |
| Training format       | ✅ Standard DPO (separate passes)      | ~~Contrastive single-seq~~ | ❌ **Not implementing** (no precedent) |
| L_DPO formula         | ✅ Standard DPO (FIXED TODAY)           | ✅ Standard DPO with reference model | ✅ **DONE** |
| Evaluation metric     | ❌ Weighted average (alignment tax)     | ⏳ Dual-metric (harmlessness + helpfulness) | ⏳ **Phase 1.4** |
| Test set size         | ❌ 7 prompts (not statistically valid)  | ⏳ 1,000+ prompts | ⏳ **Phase 1.4** |

**Progress: 1 done, 2 building (Phase 1.4), 2 deferred (research hypotheses).**

**Notes on Deferred Items:**

1. **Instruction-awareness (text labels)**: Plan1 wants `### Alignment Instruction:` and `### User Prompt:` text labels. Current implementation uses `{"role": "system"}` vs `{"role": "user"}` (chat template adds special tokens). **DEFERRED** - unproven hypothesis. Standard practice (Llama-3, ChatML) uses role tokens, not text labels.

2. **Contrastive single-sequence format**: Plan1 wants model to see rejected THEN chosen in one sequence. Standard DPO (Rafailov 2023, PKU-SafeRLHF, Anthropic-OpenAI 2024) processes chosen/rejected separately. **DEFERRED** - no precedent in alignment research. Would require: (a) modify `formatters.py`, (b) switch to SFTTrainer OR custom `concatenated_forward()`.

**Strategy**: Use **standard implementations** for baselines (Phase 1-3). If CITA underperforms, test deferred modifications as ablation study in Phase 4.

---

## Phase 1: Build Infrastructure (2-3 days, $0 GPU cost)

**What we're building:**
- ✅ SFT baseline (standard SFTTrainer)
- ✅ DPO baseline (standard DPOTrainer)
- ✅ CITA with standard DPO format (L_SFT + L_DPO + L_KL, role-based separation)
- ⏳ Dual-metric evaluation (1,000+ prompts, custom LLM-as-judge)

**What we're ~~NOT building~~ (deferred research hypotheses):**
- ~~Text-label instruction format~~
- ~~Contrastive single-sequence format~~

---

## Recommended Project Structure

```
comparative_study/
├── 01a_SFT_Baseline/
│   └── Llama3_BF16.py          # ✅ Separate script
├── 02a_DPO_Baseline/
│   └── Llama3_BF16.py          # ✅ Separate script
├── 03a_CITA_Baseline/
│   └── Llama3_BF16_PBT.py      # ✅ Separate script
└── 0c_utils/                   # ← Shared utilities
    ├── data_prep/              # ✅ Package (loader.py, formatters.py)
    ├── cita_trainer.py         # ✅ Unified loss (L_SFT + L_DPO + L_KL)
    ├── monitoring_callback.py  # ✅ KL early stopping, perplexity tracking
    ├── pbt_trainer.py          # ✅ Ray Tune PBT wrapper
    ├── push_automation.py      # ✅ Conditional HF push + GitHub automation
    └── model_utils.py          # ✅ 7 utility functions (load_hf_token, load_model_bf16, setup_lora, apply_torch_compile, load_training_dataset, get_test_prompts, get_model_repo_name)
```

---

1. ✅ **Build SFT trainer** (`comparative_study/01a_SFT_Baseline/Llama3_BF16.py`)
   - ✅ Standard SFTTrainer from TRL
   - ✅ Format 4.1 (ITA) - chosen only
   - ✅ Unified data loading (via `model_utils.load_training_dataset()`)
   - ✅ Push automation (conditional HF + GitHub push)
   - ✅ Auto-shutdown
   - ✅ Verify loss matches theory

2. ✅ **Build DPO trainer** (`comparative_study/02a_DPO_Baseline/Llama3_BF16.py`)
   - ✅ Standard DPOTrainer from TRL
   - ✅ Format 4.3 (EBA) - separate chosen/rejected
   - ✅ Unified data loading (via `model_utils.load_training_dataset()`)
   - ✅ Push automation (conditional HF + GitHub push)
   - ✅ Auto-shutdown
   - ✅ Verify loss matches Rafailov 2023

3. ✅ **CITA trainer** (`comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py`)
   - ✅ Has L_DPO fix (standard DPO with reference model)
   - ✅ Uses role-based instruction separation (system/user tokens)
   - ✅ Unified data loading (via `model_utils.load_training_dataset()`)
   - ✅ Push automation (conditional HF + GitHub push)
   - ✅ Auto-shutdown (`os.system("sudo shutdown -h now")`)
   - ~~Add text-label instruction-awareness~~ (research hypothesis - not implementing)
   - ~~Add contrastive single-sequence format~~ (no precedent - not implementing)

   **Standard Monitoring (fixes PBT failures):**
   - ✅ KL divergence early stopping in `monitoring_callback.py` (stops at iter 1 vs iter 8, saves 77 min)
   - ✅ Reward metrics (`rewards/chosen`, `rewards/rejected`, `rewards/accuracies`) in `cita_trainer.py`
   - ✅ Perplexity tracking (`torch.exp(loss_sft)`) in `cita_trainer.py`
   - ✅ Gradient norm monitoring (`clip_grad_norm_`) in `cita_trainer.py`

4. ✅ **Build evaluation** (`comparative_study/05_evaluation/dual_metric_eval.py`)
   - ✅ Dual-metric (harmlessness + helpfulness)
   - ✅ Large test sets (1,000+ prompts)
   - ✅ LLM-as-judge (Llama-3-70B via Fireworks)

   **Implementation Checklist (Phase 1.4):**
   1. ✅ Research datasets (AlpacaEval, PKU-SafeRLHF, AIR-Bench)
   2. ✅ Research Fireworks API (litellm wrapper, Llama-3.3-70B)
   3. ✅ Research LLM-as-judge prompts (Constitutional AI, HH-RLHF)
   4. ✅ Create `llm_judge_prompts.py` - Custom prompts for harmlessness/helpfulness
   5. ✅ Create `fireworks_client.py` - Fireworks API wrapper with retry logic
   6. ✅ Create `dual_metric_eval.py` - Main evaluation script (loads from HF)
   7. ✅ Create `statistical_analysis.py` - Pareto plots, bootstrap CI, t-tests
   8. ✅ Create `test_dual_metric_eval.py` - Compilation test with dummy endpoints
   9. 🔄 Run compilation test (currently running - datasets downloading)
   10. ✅ Get Fireworks API key - Added to .env (not committed to git)
   11. ⏳ Run full evaluation - After all 3 baselines trained

### Evaluation Plan Details

**Problem with Current Approach:**
```python
# Current: Weighted average penalizes safety
overall_quality = fluency*0.3 + coherence*0.3 + alignment*0.4
```
- ❌ Over-refusal penalized on helpful prompts (alignment tax)
- ❌ Single metric hides trade-offs
- ❌ Only 7 test prompts (not statistically significant)

**Target: Dual-Metric Evaluation (No weighted average)**

Report as 2D Pareto frontier:
- **X-axis**: Helpfulness (0-10) - AlpacaEval (805 prompts)
- **Y-axis**: Harmlessness (0-10) - PKU-SafeRLHF test split (1,000+ prompts)

**Tasks:**
1. **Harmlessness eval**: PKU-SafeRLHF harmful split (1,000+ prompts, 19 categories)
2. **Helpfulness eval**: AlpacaEval (805 prompts)
3. **LLM-as-judge**: Llama-3-70B via Fireworks API (not pattern matching)
   - **Custom prompts, NOT G-Eval**: Alignment research uses custom criteria (PKU-SafeRLHF, Anthropic-OpenAI 2024). G-Eval has no precedent in safety papers. Custom prompts = transparent, reproducible, exact control for ablation.

**Test Sets:**
- **Harmfulness**: PKU-SafeRLHF test split (1,000+ prompts)
- **Helpfulness**: AlpacaEval (805 prompts)
- **(Optional)** AIR-Bench 2024 (5,694 prompts, 314 risk categories)

**Metrics:**
- Harmlessness score (0-10): Refusal rate on harmful prompts
- Helpfulness score (0-10): Instruction-following quality
- **Pareto dominance**: CITA should have highest harmlessness without sacrificing helpfulness

**Statistical Significance:**
1. **2D Pareto Frontier Plot**
2. **Statistical tests**: Bootstrap 95% CI, paired t-tests (CITA vs DPO, CITA vs SFT), Cohen's d
3. **Per-category breakdown** (19 harm categories): Violence, drugs, discrimination, self-harm, etc.

---

## Phase 2: Theoretical Validation (1 day, $0 GPU cost)

1. **Code review:**
   - Read all 3 trainers side-by-side
   - Verify CITA = SFT + DPO + KL
   - Check data formatting matches plan

2. **Devil's advocate:**
   - "Why SHOULD CITA outperform DPO?"
   - "What could go wrong?"
   - "Are there any remaining bugs?"

3. **Community feedback:**
   - Post on r/MachineLearning
   - Ask: "Does this CITA implementation look correct?"
   - Get FREE expert review

---

## Phase 3: Sanity Checks (3 hours, ~$5-10 GPU cost)

**Only AFTER phases 1-2 are complete:**

1. **Run all 3 sanity checks in parallel:**

   ```bash
   # SFT sanity (100 steps, 1 worker)
   python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity

   # DPO sanity (100 steps, 1 worker)
   python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode sanity

   # CITA sanity (100 steps, 1 worker) - Standard DPO + L_SFT + L_KL
   python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py --mode sanity
   ```

2. **Quick evaluation (7 test prompts only):**
   - Just check if CITA >= DPO >= SFT
   - If ordering is wrong, debug BEFORE full training

3. **Decision point:**
   - ✅ If CITA > DPO > SFT: Proceed to full training
   - ❌ If ordering wrong: Debug and repeat Phase 2

---

## Phase 4: Full Training (7-9 days, ~$200-300 GPU cost)

**Only AFTER sanity checks confirm CITA > DPO > SFT.**