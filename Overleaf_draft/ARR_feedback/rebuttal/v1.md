# ECLIPTICA / CITA — Rebuttal Draft v1

> **Context:** ARR March 2026, Submission #500
> **Scores:** gMno=2, KpHA=3.5, shuX=2
> **Target:** avg >= 3.0 (Findings consideration)

---

# REBUTTAL DRAFT

We thank all reviewers for their detailed feedback. We address each concern below, distinguishing between **evidence already in the paper** (with exact references) and **gaps we acknowledge**.

---

## Response to Reviewer gMno (Score: 2)

### W1: "The paper overstates its novelty. Comparison with SimPO, KTO, multi-objective decoding"

**Existing evidence:**

- Tab. 2 (`4_cita_framework.tex:192-214`) directly compares PPO, GRPO, DPO, CITA on six capability axes. CITA is the only method supporting instruction-conditioned switching + mandatory explicit KL.
- FAQ Q6 (`H_faq.tex:130-144`) explains the core novelty: DPO learns P(y⁺ > y⁻ | X). CITA learns P(y⁺ > y⁻ | **I**, X) — preferences **conditioned on alignment instructions**. This is a different learning objective, not "DPO + KL + instructions."
- FAQ Q12 (`H_faq.tex:264-282`) provides a table comparing CITA against prompting baselines: zero-shot prompting (0.12), few-shot prompting (0.18), DPO + prompting (0.25), **CITA (0.37)** on ECLIPTICA. CITA achieves **3× the score** of zero-shot prompting.
- The introduction (`1_introduction.tex:343-352`) explicitly discusses how ECLIPTICA differs from inference-time alignment methods (SteerLM, arithmetic decoding, inference-time value guidance).

**Gap acknowledged:** SimPO and KTO are not compared. We will add these baselines in revision (both supported by TRL, same training stack). Multi-objective decoding is discussed conceptually but not benchmarked.

---

### W2: "Instruction set has 10 types, very limited. Multilingual missing"

**Existing evidence:**

- The 10 instruction types are **not arbitrary** — they are derived from preference-supervised evidence via 5 independent judge models (`5_isd_dataset.tex:369-392`, Tab. `judge_models`), filtered by BERTScore semantic agreement (3-of-5 rule), and validated by 2 human annotators (both ≥ 4/5 on clarity, actionability, safety). This pipeline is described in `5_isd_dataset.tex:355-423` and Fig. `ecliptica_pipeline`.
- Limitations Section (`9_limitation.tex:20-22`) explicitly acknowledges both the instruction coverage and multilingual gaps.

**Gap acknowledged:** 10 types may not capture all deployment needs. Multilingual evaluation is future work. We state this in Section 8.

---

### W3: "Only Llama-3.1-8B. More models needed"

**Existing evidence:**

- FAQ Q14 (`H_faq.tex:320-334`) discusses scalability indicators: LoRA scales linearly (~0.1% of parameters regardless of model size), larger models follow instructions better, and DPO (which CITA extends) has been validated at 70B scale. We chose 8B for **single-GPU reproducibility**.

**Gap acknowledged:** No multi-model results. This is the most unanimous reviewer concern. We plan to add Llama-3.2-3B and Llama-3.1-70B in revision (see `plan_nextARR.md` Exp 7).

---

### W4: "Trade-off analysis helpfulness/safety should be more detailed"

**Existing evidence:**

- The full benchmark-specific analysis in Appendix F (`F_extended_results.tex:242-577`) provides per-benchmark interpretations with explicit discussion of trade-offs:
  - ECLIPTICA rewards shift magnitude, can favor large stylistic shifts over calibrated ones (`F_extended_results.tex:278-283`)
  - DPO leads on safety-dominant switching but fails on calibration (`F_extended_results.tex:556-570`)
  - CITA prioritizes **consistent multi-axis switching** over single-axis extremes
- Tab. `tradeoffs` (`I_discussion_floats.tex:175-196`) explicitly maps each benchmark's reward structure and failure modes.

---

### W5: "CITA should be tested against SOTA jailbreak attacks"

**Existing evidence:**

- FAQ Q9 (`H_faq.tex:196-208`) discusses mitigations: hierarchical instruction handling, instruction validation, constitutional baselines, audit trails.
- FAQ Q11 (`H_faq.tex:235-253`) reports: "No catastrophic safety failures observed: CITA never produced harmful content when given safety-violating instructions in our red-teaming."

**Gap acknowledged:** No systematic jailbreak benchmarking (GCG, AutoDAN). This is a fair request. We plan to add adversarial robustness evaluation (see `plan_nextARR.md` Exp 8).

---

### W6: "Proofreading. Repeated references, hallucinated reference (OPT-IML)"

**Acknowledged.** We will fix the hallucinated OPT-IML reference, deduplicate repeated citations (TruthfulQA, RLHF), and proofread thoroughly in revision.

---

### W7: "Space manipulation between text and captions"

**Acknowledged.** We will restore standard ACL spacing in revision. The compression was done to fit the 8-page limit after the previous desk rejection required moving Discussion to appendix.

---

## Response to Reviewer KpHA (Score: 3.5)

### W1: "Core distinction from standard instruction following is murky"

**Existing evidence:**

This is the most important conceptual point. Several sections directly address it:

- **FAQ Q12** (`H_faq.tex:264-286`): Direct empirical comparison — zero-shot prompting scores **0.12** vs CITA **0.37** on ECLIPTICA (3× gap). "Prompting operates at the surface level — the model *follows* instructions but doesn't *internalize* them."
- **FAQ Q23** (`H_faq.tex:562-568`): Directly addresses the "routing trick" hypothesis with four evidence lines:
  1. Paraphrase robustness (instruction semantics, not tokens)
  2. Compositional generalization (unseen conjunctions)
  3. Conditional safety/utility coupling (same X, different I moves coupled outcomes)
  4. Anchor ablations as sanity check (Goldilocks regime)
- **Key distinction** (`F_extended_results.tex:565-570`): "Instruction-following maps task instructions to outputs, whereas instruction-alignment requires **counterfactual policy control** under an unchanged user request."
- **TruthfulQA result** (`7_results.tex:116-117`): DPO Δ = +0.001, CITA Δ = **+0.054** (54× better). If CITA were just prompting, it couldn't achieve 54× stronger epistemic calibration switching than DPO.

**What we can add:** A query-scrambled evaluation (Exp 2 in `plan_nextARR.md`) — shuffle X while keeping I fixed. If CITA were a shallow prefix→style map, scrambling X should NOT affect performance. We expect it to **drop dramatically**, proving joint (I,X) dependence.

---

### W2: "No discussion of training cost"

**Existing evidence (already in the paper, Appendix C):**

- Tab. `compute` (`C_implementation.tex:114-130`):

| Metric | PPO | GRPO | DPO | **CITA** |
|--------|-----|------|-----|----------|
| GPU Used | 80GB | 80GB | 40GB | **40GB** |
| Training Time | 17h | 12h | 103min | **120min** |
| GPU Memory | 72GB | 68GB | 39GB | **39GB** |

**CITA costs only 17 minutes more than DPO** (120 vs 103 min) on the same A100-40GB hardware. Same memory footprint. No additional reward model. The Fisher metric is never explicitly computed — the KL anchor approximates it via the standard forward KL, which adds negligible overhead.

- FAQ Q8 (`H_faq.tex:179-191`): Inference overhead is 10-40 tokens (<0.5% of 8K-128K context). "One CITA model replaces N separate DPO models for N deployment contexts."

This evidence was in Appendix C — we acknowledge it should be **surfaced in the main text**.

---

### W3: "Only one model size tested"

Same as gMno W3. **Gap acknowledged.** We plan 3B and 70B experiments.

---

### W4: "Benchmark coverage narrow in domain"

**Existing evidence:**

- ECLIPTICA covers **12 topic categories** (Tab. `prompt_categories`, `5_isd_dataset.tex:173-192`): Technology, Healthcare, Environment, Education, Economics, Social, Ethics, Culture, Science, Business, Personal, Governance — 25 prompts each.
- Additionally, we evaluate on **4 external benchmarks** beyond ECLIPTICA: TruthfulQA, Conditional Safety, Length Control, LITMUS (`7_results.tex:21-52`).

**Gap acknowledged:** Prompts are sourced from HH-RLHF, which skews safety/helpfulness. Creative writing, coding, tutoring prompts are underrepresented. We can expand the evaluation set without retraining (Exp 10 in `plan_nextARR.md`).

---

### W5: "No analysis of interference across many instructions (50, 100)"

**Gap acknowledged.** This is a valid and important question. We do not have scalability analysis beyond 10 instruction types. Planned as Exp 9 in `plan_nextARR.md`.

---

### Comments: "Clarify Table 1 caption / gradient intuition / DPO ignoring instructions / Riemannian language"

All actionable suggestions. We will:
- Clarify that Table 1 responses come from a **single trained checkpoint** per method
- Add 1-2 sentences of gradient intuition in the main text (currently only in Appendix A)
- Explicitly call out that DPO Δ ≈ 0 on TruthfulQA means DPO **effectively ignores** the instruction signal
- Add a concrete analogy for "Riemannian chart" before the formal definition

---

## Response to Reviewer shuX (Score: 2)

### W1: "Limited evaluation scope (single backbone)"

Same as above. **Gap acknowledged.**

---

### W2: "Missing capability evaluation (MMLU, GSM8K, HumanEval)"

**Existing evidence (partial):**

- FAQ Q16 (`H_faq.tex:381`): "We verified CITA doesn't degrade base capabilities (perplexity, coherence)."

**Gap acknowledged:** This claim lacks supporting numbers. We will run lm-eval-harness (MMLU, GSM8K, HumanEval) on all 10 checkpoints + base model. This is cheap (~hours of inference) and directly addresses the concern. Planned as **Exp 1** in `plan_nextARR.md`.

---

### W3: "Unclear novelty over prior work"

**Existing evidence:**

- **The learning objective is different** (`4_cita_framework.tex:128-148`): CITA optimizes **instruction-indexed optima** — a switchable policy family {π_θ(·|I,·)} for I ∈ I — rather than a single implicit policy.
- **The KL is structural, not cosmetic** (`E_ablations.tex:24-44`): Removing KL (λ_KL=0) drops ECLIPTICA from 0.37 → 0.22 and causes training instability. This is a **68% degradation**, not a minor regularization effect.
- **FAQ Q21** (`H_faq.tex:509-542`): DPO's β controls BOTH KL strength AND preference sharpness (coupled). CITA **decouples** them: β for preference sharpness, λ_KL for stability. This decoupling is why CITA achieves 86.7% instruction-alignment efficiency vs DPO's 56.1%.
- **DPO+prompting comparison** (`H_faq.tex:264-282`): DPO + prompting = 0.25, CITA = 0.37 on ECLIPTICA. The gap persists even when DPO gets the same system prompts.

---

### W4: "KL contribution not fully substantiated"

**Existing evidence:**

This is substantiated with an ablation table:

- Tab. `kl_ablation` (`E_ablations.tex:24-44`):

| λ_KL | Reward Margin | ECLIPTICA Score | Stability |
|------|--------------|-----------------|-----------|
| 0 (no KL) | 3.8 | 0.22 | Unstable |
| 0.0001 | 6.2 | 0.33 | Stable |
| **0.00023 (best)** | **7.5** | **0.37** | **Stable** |
| 0.001 | 5.5 | 0.29 | Stable |

- The ablation shows a clear **inverted-U**: too low = regime collapse, too high = over-constraint. The KL anchor is not "standard regularization" — it is the mechanism that keeps **multiple instruction regimes co-located** (`E_ablations.tex:19-22`).

**Gap acknowledged:** This is single-seed. Multi-seed ablation (3 seeds × 4 λ_KL values) is planned as **Exp 4** in `plan_nextARR.md`.

---

### W5: "Theoretical overstatement"

**Partially acknowledged.** The gradient analysis does follow standard DPO formulations, as the reviewer correctly notes. What is specific to CITA is:
- The gradient is **indexed by instruction I** — forming a family of compatible update directions, not one direction (`A_related_and_derivation.tex:209-223`)
- The **self-quenching property** (P⁺→1 ⇒ ‖∇L‖→0) prevents over-steering across competing instructions (`A_related_and_derivation.tex:147-149`)

We will tighten the framing to avoid overclaiming and present these as **properties of the CITA-specific setup** rather than novel mathematical results.

---

### W6: "Lack of statistical rigor (single-seed, no variance)"

**Existing evidence (partial):**

- 95% bootstrap CIs are reported for 3/5 benchmarks (`F_extended_results.tex:503-549`): TruthfulQA (±0.11-0.28), Conditional Safety (±0.01-0.03), Length Control (wider intervals).
- 13 Optuna trials are reported for hyperparameter sensitivity (`E_ablations.tex:120-148`).

**Gap acknowledged:** Training is single-seed. Multi-seed results are needed. Planned as **Exp 4**.

---

### W7: "Benchmark construction concerns (latent biases)"

**Existing evidence:**

- The construction pipeline explicitly mitigates this:
  1. **5 independent judge models** from different families (GPT-4, Gemini, Claude, Llama, Mistral) — Tab. `judge_models` (`5_isd_dataset.tex:369-392`)
  2. **BERTScore semantic agreement** — 3-of-5 agreement rule filters ambiguous cases (`5_isd_dataset.tex:399-401`)
  3. **Human quality gate** — 2 annotators, both ≥ 4/5 on clarity, actionability, safety (`5_isd_dataset.tex:416-422`)
  4. **12 balanced topic categories** — 25 prompts each (`5_isd_dataset.tex:173-192`)

**What we can add:** Per-category × per-instruction performance heatmap to verify no instruction-category bias (Exp 5 in `plan_nextARR.md`).

---

### W8: "Potential shortcut learning"

**Existing evidence:**

- FAQ Q23 (`H_faq.tex:562-568`) directly addresses this with four evidence lines: paraphrase robustness, compositional generalization, conditional safety coupling, and anchor ablations.
- The **TruthfulQA result** is the strongest counter-evidence: switching between "honest uncertainty" and "confident assertion" is a **semantic calibration** task that cannot be solved by a shallow prefix→style map. CITA achieves +0.054 Δ vs DPO's +0.001.

**What we can add:** Query-scrambled evaluation (**Exp 2** in `plan_nextARR.md`) — this is the direct diagnostic the reviewer requested.

---

### W9: "Presentation: repeated high-level framing"

**Acknowledged.** We will tighten the manuscript to reduce redundant framing and foreground the technical contribution.

---

### Comments (1)-(7): Specific experimental suggestions

| Suggestion | Status | Evidence/Plan |
|-----------|--------|---------------|
| (1) MMLU/GSM8K/HumanEval | **GAP** | Exp 1 in plan |
| (2) Controlled KL ablation + multi-seed | **PARTIAL** — ablation exists (`E_ablations.tex:24-97`), multi-seed missing | Exp 4 in plan |
| (3) Query-scrambled / instruction-only eval | **GAP** — FAQ discusses conceptually but no experiment | Exp 2 in plan |
| (4) Instruction distribution balance | **PARTIAL** — 12 categories × 25 each, but no cross-tabulation | Exp 5 in plan |
| (5) Per-instruction stability | **PARTIAL** — aggregate results only | Exp 5 in plan |
| (6) More models | **GAP** | Exp 7 in plan |
| (7) Inter-annotator agreement | **PARTIAL** — 2 annotators ≥4/5 mentioned but no IAA statistic | Need to compute and report |
| (7) Tighten presentation | **Acknowledged** | Revision |

---

## Summary: What We Have vs What We Need

| Category | Evidence In Paper | Gap to Fill |
|----------|------------------|-------------|
| **Novelty vs prompting** | FAQ Q12 table (CITA 0.37 vs prompting 0.12), FAQ Q6, FAQ Q21, Tab. 2 | System-prompt baseline experiment (Exp 3) |
| **KL mandatory** | Full ablation table (λ_KL sweep), 68% degradation at λ_KL=0 | Multi-seed ablation (Exp 4) |
| **Training cost** | Tab. `compute`: CITA = 120 min vs DPO = 103 min, same GPU | Surface this in main text |
| **Capability preservation** | Claim in FAQ Q16, no numbers | MMLU/GSM8K/HumanEval (Exp 1) |
| **Shortcut learning** | FAQ Q23 conceptual arguments, TruthfulQA Δ evidence | Query-scrambled eval (Exp 2) |
| **Statistical rigor** | Bootstrap CIs for 3/5 benchmarks, 13 Optuna trials | Multi-seed training (Exp 4) |
| **Benchmark construction** | 5-judge pipeline, BERTScore, human gate, 12 categories | Per-instruction × per-category heatmap (Exp 5) |
| **Multi-model** | None (acknowledged limitation) | 3B + 70B experiments (Exp 7) |
| **Jailbreak robustness** | FAQ discussion, no empirical test | GCG/AutoDAN evaluation (Exp 8) |
| **SimPO/KTO** | Not compared | Add baselines (Exp 6) |
| **Presentation** | Verbose framing acknowledged | Revision pass |
| **Proofreading** | Hallucinated refs acknowledged | Fix in revision |
