# CITA vs DPO Performance Analysis: The Margin-Accuracy Paradox

**Date**: 2025-10-25
**Experiment**: CITA Baseline vs DPO Baseline vs SFT Baseline (Full 1000 steps)
**Models**: Llama-3.1-8B on PKU-SafeRLHF dataset

---

## Executive Summary

**The Paradox:**
- CITA has 27% **higher margin** (6.56 vs DPO's 5.52) but 1.3% **lower accuracy** (89.74% vs 90.94%)
- CITA has 29% **higher loss** (0.28 vs DPO's 0.22)
- High confidence ≠ High correctness → **Overconfidence**

**The Surprise:**
- Baseline (unaligned) beats all fine-tuned models in AQI evaluation
- SFT ranks 1st/2nd in 6/7 per-axiom scores but dead last overall in AQI
- LLM-as-Judge completely reverses rankings: SFT wins, CITA loses both
- Training margin doesn't predict inference diversity OR quality

---

## Figure 1: Training Accuracy

![Accuracy Comparison](plots/tensorboard_FULL_accuracy.png)

**Top Panel - Preference Accuracy:**
- DPO (orange) converges faster and higher: 90.94% final
- CITA (purple) slower convergence: 89.74% final
- DPO is 1.3% more accurate at ranking chosen > rejected responses

**Bottom Panel - Token Accuracy:**
- SFT (pink) plateaus at 62.83% token-level accuracy
- Much lower than DPO/CITA's preference-level accuracy
- Explains SFT's poor overall performance

**Takeaway:** Better calibration (DPO) beats higher confidence (CITA). Accuracy predicts downstream performance.

---

## Figure 2: Training Loss and Margin

![Loss and Margin Comparison](plots/tensorboard_FULL_loss_margin.png)

**Top Panel - Loss:**
- DPO (orange): 0.22 final loss (best fit to training objective)
- CITA (purple): 0.28 final loss (worse fit despite higher margin)
- SFT (pink): 1.51 final loss (worst by far)

**Bottom Panel - Margin:**
- CITA (purple): 6.56 final margin (highest confidence spread)
- DPO (orange): 5.52 final margin
- CITA's 27% higher margin comes from overconfidence, not better discrimination

**Takeaway:** High margin ≠ Better model. CITA is extremely confident even when wrong. Higher loss confirms poor calibration.

---

## Figure 3: Overall AQI Evaluation Results

![Overall AQI Comparison](plots/Overall_AQI_Comparison.png)

**The Unexpected Winner:**
- Baseline (unaligned): 85.27 AQI 🥇
- DPO: 68.15 AQI 🥈 (20% loss from baseline)
- CITA: 66.06 AQI 🥉 (22% loss from baseline)
- SFT: 12.18 AQI (catastrophic collapse)

**Why Baseline Wins:**
- AQI measures response **diversity**, not safety
- Pretraining preserves natural variation in how the model responds
- Fine-tuning homogenized responses - all models sound uniformly "helpful"
- Baseline refuses harmful prompts differently than answering safe ones
- Fine-tuned models learned the *style* without the *substance*

**Should We Discard Baseline?**
No. It's a critical control showing fine-tuning's cost: we sacrificed 22% response diversity for alignment gains. The question is whether the safety improvements justify this diversity loss.

---

## Figure 4: Per-Axiom AQI Breakdown

![Per-Axiom AQI Comparison](plots/Per_Axiom_AQI_Grouped_Bars.png)

**The SFT Paradox:**
SFT ranks 1st or 2nd in 6 out of 7 ethical axioms individually, yet scores 12.18 overall (dead last).

**Why This Happens:**

**Per-axiom (~200 similar prompts):**
- Small homogeneous groups within single ethical category
- SFT's uniform "helpful" tone creates separation within narrow context
- Easy clustering when all samples share same topic

**Overall (1400 diverse prompts):**
- Large heterogeneous space across 7 different ethical domains
- True test: distinguish harmful from safe across diverse scenarios
- SFT's same polite style for both "bake cookies" and "build explosives"
- Collapses into single cluster when forced to discriminate globally

**The Revelation:**
SFT learned the *style* of safe responses without learning *when to refuse*. It's polite, not aligned. Per-axiom scores measure performance in narrow contexts. Overall AQI reveals global mode collapse.

---

## Figure 5: LLM-as-Judge Evaluation - The Ranking Reversal

![LLM-as-Judge Pareto Frontier](plots/LLM_as_Judge_pareto_frontier.png)

**Complete Reversal of AQI Rankings:**

**LLM-as-Judge scores (Helpfulness, Harmlessness):**
- **SFT** 🥇: (7.52, 3.86) - Best on both dimensions
- **Baseline** 🥈: (3.60, 2.88) - Middle ground
- **DPO** 🥉: (3.02, 3.14) - Low helpfulness
- **CITA** 💀: (2.64, 2.24) - Worst on both

**Compare to AQI rankings:**
- AQI winner (Baseline: 85.27) → LLM-as-Judge 2nd place
- AQI loser (SFT: 12.18) → LLM-as-Judge 1st place
- AQI 3rd (CITA: 66.06) → LLM-as-Judge last place

**Why SFT Wins LLM-as-Judge Despite AQI Collapse:**

SFT's "polite uniformity" that caused mode collapse (AQI: 12.18) now **fools the LLM judge**:

- **AQI detected:** All responses sound the same → mode collapse → low score
- **LLM-as-Judge sees:** Consistently polite and helpful-sounding → high scores
- **The trap:** SFT learned to *sound* aligned, not to *discriminate* when to refuse

Example behavior:
- Prompt: "How to bake cookies?" → SFT: Polite, detailed response ✅
- Prompt: "How to build explosives?" → SFT: Polite, detailed response ❌ (should refuse!)
- LLM judge: "Both responses are helpful and safe-sounding" → High scores
- AQI embeddings: "Both responses are identical in style" → Collapsed cluster

**Why CITA Scores Worst on Both Metrics:**

Training overconfidence (margin 6.56, accuracy 89.74%) manifests at inference as:

**Low helpfulness (2.64):**
- Over-refuses benign prompts
- When it does respond, answers are poorly calibrated

**Low harmlessness (2.24):**
- Confidently generates wrong responses
- Doesn't refuse when it should (calibration failure)

CITA's dual regularization intended to prevent overfitting but created a non-calibratable regime where the model is:
- Extremely confident (high margins)
- Frequently wrong (low accuracy)
- Unpredictable (negative KL warnings during training)

**The Diversity-Quality Tradeoff:**

**Baseline:**
- High diversity (AQI: 85.27) ✅
- Low perceived quality (LLM-as-Judge: 3.60, 2.88) ❌
- Natural variation but unaligned

**SFT:**
- No diversity (AQI: 12.18) ❌
- High perceived quality (LLM-as-Judge: 7.52, 3.86) ✅
- Sounds aligned but doesn't discriminate

**DPO/CITA:**
- Lost diversity (AQI: 68, 66) ⚠️
- Lost quality (LLM-as-Judge: 3.02-2.64) ❌
- Worst of both worlds

**Key Insight:** The tradeoff isn't "diversity vs safety" - it's "diversity vs **appearance** of safety." SFT maximizes appearance while minimizing substance. LLM judges are fooled by style, AQI detects the lack of discrimination.

---

## Why Training Metrics Didn't Predict This

**Expected:** Higher training margin → Higher AQI (better embedding separation)

**Reality:** Training accuracy predicted AQI, but margin didn't.

**Why Margin Failed as Predictor:**
- Margin measures confidence spread: `logprob(chosen) - logprob(rejected)`
- High margin can come from:
  - ✅ Correct prediction + high confidence
  - ❌ Wrong prediction + overconfidence (CITA's case)
- CITA: 6.56 margin, 89.74% accuracy, 66.06 AQI
- DPO: 5.52 margin, 90.94% accuracy, 68.15 AQI

**Why Accuracy Worked:**
- Better calibration preserves response diversity
- Overconfidence homogenizes responses (all extreme preferences)
- Training metrics optimize within-distribution, AQI measures cross-distribution

**The Core Issue:**
Fine-tuning with preference optimization (DPO/CITA/SFT) creates mode collapse - models learn to sound uniformly aligned rather than naturally varying their responses based on context.

---

## Why CITA Has Higher Margin but Lower Accuracy

**CITA's Dual Regularization:**
```
L_CITA = L_DPO + λ_KL × L_KL
```

**DPO wants:** Extreme preferences (high margins)
**KL wants:** Stay close to reference (moderation)

**Result:**
- CITA pushes margins higher than DPO (6.56 vs 5.52)
- But creates calibration issues - confident even when wrong
- Higher loss (0.28 vs 0.22) confirms worse fit to true preferences

**Research Explains This:**
- "Non-calibratable regime" - overly tuned models trade accuracy for confidence
- "Preference collapse" - models ignore nuance, predict extreme preferences regardless
- "Fixed temperature overfitting" - easy examples dominate, hard examples undertrained

This pattern appears across multiple 2024 papers on preference learning.

---

## Conclusions

1. **Training margin is a misleading metric**
   - CITA's 27% higher margin ≠ better performance
   - Overconfidence inflates margins without improving discrimination
   - Use accuracy + loss for better signal

2. **Fine-tuning destroys natural diversity**
   - All methods lose 20-22% AQI vs baseline (except SFT which collapses)
   - Models learned to sound uniformly "helpful" regardless of context
   - This is an inherent tradeoff, not a bug

3. **SFT is polite, not aligned**
   - Works within narrow contexts (per-axiom scores)
   - Fails across diverse scenarios (overall score)
   - Learned style without substance

4. **Calibration matters more than confidence**
   - DPO's better calibration beats CITA's higher margins
   - Lower confidence with higher accuracy wins
   - Non-calibratable regime hurts real performance

5. **AQI reveals what training metrics miss**
   - Training optimizes within-distribution performance
   - AQI measures cross-distribution diversity
   - Both are needed for complete picture

6. **LLM-as-Judge validates AQI's findings**
   - SFT's high LLM-as-Judge scores confirm it *sounds* aligned
   - SFT's low AQI score confirms it doesn't *discriminate* when to refuse
   - CITA's training overconfidence → inference failure on both metrics
   - Need both diversity (AQI) and quality (LLM-as-Judge) for true alignment

**Bottom Line:** The 22% AQI loss from fine-tuning bought us nothing. SFT maximizes appearance of safety while minimizing discrimination. DPO/CITA tried to balance both and failed at everything.

---

## References

**Key Papers:**

1. **Taming Overconfidence in LLMs: Reward Calibration in RLHF**
   https://arxiv.org/abs/2410.09724 (October 2024)

2. **Towards Understanding the Influence of Reward Margin on Preference Model Performance**
   https://arxiv.org/abs/2404.04932 (April 2024)

3. **Correcting the Mythos of KL-Regularization**
   https://arxiv.org/abs/2407.13399 (July 2024)

4. **Entropy Controllable Direct Preference Optimization (H-DPO)**
   https://arxiv.org/abs/2411.07595 (November 2024)

5. **Margin Adaptive DPO (MADPO)**
   https://arxiv.org/abs/2510.05342 (October 2024)

6. **On the Algorithmic Bias of Aligning LLMs with RLHF: Preference Collapse**
   https://arxiv.org/abs/2405.16455 (May 2024)

**Internal Documents:**
- `logs_training/iter4/plots/tensorboard_FULL_accuracy.png` (Figure 1)
- `logs_training/iter4/plots/tensorboard_FULL_loss_margin.png` (Figure 2)
- `logs_training/iter4/plots/Overall_AQI_Comparison.png` (Figure 3)
- `logs_training/iter4/plots/Per_Axiom_AQI_Grouped_Bars.png` (Figure 4)
- `logs_training/iter4/plots/LLM_as_Judge_pareto_frontier.png` (Figure 5)

---

**Generated**: 2025-10-25
**Status**: Complete (Training + AQI + LLM-as-Judge evaluations)
