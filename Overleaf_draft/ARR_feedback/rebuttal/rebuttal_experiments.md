# ECLIPTICA / CITA — Rebuttal Experiments Plan

> **Goal:** Run experiments, include results inline in rebuttal, targeting avg score >= 3.0
> **Current scores:** gMno=2, KpHA=3.5, shuX=2
> **Primary target:** Move shuX from 2 → 3+ (most actionable). gMno from 2 → 2.5+ (stretch).
> **Strategy:** Run Tier 1 experiments (all inference-only, ~1 day total), include results as tables in rebuttal. Tier 2 if time permits before deadline.

---

## Tier 1 — HIGHEST ROI, LOWEST COST (all inference-only, ~1 day total)

### Exp 1: Per-Instruction Performance Breakdown [COST: FREE]
- **Answers:** shuX point 5 ("multi-regime stability"), shuX point 4 ("instruction distribution bias")
- **What:** Report ECLIPTICA scores broken down by each of the 10 instruction types. Show that CITA performs well across ALL instructions (not just a subset).
- **Additionally:** Show instruction × category heatmap (10 instructions × 12 topic categories) to prove no instruction-category bias.
- **Compute:** Just post-processing existing evaluation outputs. ~Hours. **No GPU needed.**
- **Output:** 10×1 per-instruction table + 10×12 heatmap
- **Where in paper:** Appendix F or new subsection
- **Why #1:** Zero compute cost. Answers 2 of shuX's concerns with data you already have.
- **Rebuttal Figure:** Grouped bar chart. X-axis: 10 instruction types. Y-axis: ECLIPTICA M1 score. Bars: CITA_I vs DPO_I vs GRPO_I vs PPO_I (4 methods). 95% CI error bars via bootstrap (1,000 resamples over 300 prompts per instruction type). Message: CITA is uniformly strong across all 10 instructions, no single instruction dominates.

### Exp 2: Query-Scrambled Evaluation [COST: ~3-4 hours inference]
- **Answers:** shuX point 3 ("shortcut learning"), shuX point 8 ("potential shortcut learning")
- **What:** Take ECLIPTICA benchmark (3,000 cases). Create 3 diagnostic variants:
  - **(a) Query-scrambled:** Keep I fixed, randomly reassign X across prompts → should DESTROY performance if model truly uses (I,X) jointly
  - **(b) Instruction-only:** Remove X entirely, give only I → should produce generic/degenerate responses
  - **(c) Query-only (NoInstruct baseline):** Remove I, give only X → existing NoInstruct baseline
- **Expected outcome:** (a) drops ECLIPTICA score dramatically (from ~0.37 to ~0.15-0.20), proving joint (I,X) dependence. (b) produces near-random/uniform responses.
- **Compute:** Just inference, no training. ~3-4 hours on A100-40GB.
- **Output:** Table: CITA_I full (0.37) vs scrambled-X (expected ~0.18) vs I-only (expected ~0.10) vs X-only (0.20)
- **Where in paper:** New Section 7 subsection or Appendix E
- **Why #2:** This is shuX's DEEPEST concern. A clean result here is the single strongest evidence you can produce. Novel diagnostic no reviewer has seen before.
- **Rebuttal Figure:** Bar chart with 4 conditions. X-axis: Full (I,X) | Scrambled-X | I-only | X-only. Y-axis: ECLIPTICA M1 score. Single bar per condition for CITA_Instruct. 95% CI error bars via bootstrap (1,000 resamples over 300 prompts). Message: Full (I,X) >> Scrambled-X, proving model learns joint interaction, not shortcut routing.

### Exp 3: Capability Preservation (MMLU / GSM8K / HumanEval) [COST: ~1 day inference]
- **Answers:** shuX point 1 ("missing capability evaluation"), gMno (generalizability)
- **What:** Run lm-eval-harness on ALL 10 checkpoints (SFT_NI, SFT_I, DPO_NI, DPO_I, PPO_NI, PPO_I, GRPO_NI, GRPO_I, CITA_NI, CITA_I) + base Llama-3.1-8B
- **Benchmarks:** MMLU (knowledge), GSM8K (reasoning), HumanEval (coding)
- **Expected outcome:** CITA checkpoints preserve base capability (no degradation from LoRA alignment — LoRA touches only 0.1% of parameters)
- **Compute:** ~2-4 hours per checkpoint on A100-40GB. Total ~1 day. Can parallelize if multiple GPUs.
- **Output:** Table showing all 11 models × 3 benchmarks. Prove no capability tax.
- **Where in paper:** New Table in Section 7 (Results) or Appendix F
- **Why #3:** Direct explicit ask from shuX. Result will likely be "boring" (no degradation), but that's exactly the point — it removes the concern.
- **Rebuttal Figure:** Grouped bar chart. X-axis: 3 benchmarks (MMLU, GSM8K, HumanEval). Bars: Base Llama-3.1-8B vs SFT_I vs DPO_I vs CITA_I (4 key models). 95% CI error bars via bootstrap (1,000 resamples over per-question accuracy). Message: flat bars across all methods = no capability degradation from alignment training.

### Exp 4: System-Prompt Baseline (CITA vs Pure Prompting) [COST: ~6-8 hours inference]
- **Answers:** KpHA weakness 1 ("murky distinction from prompting"), gMno ("novelty"), shuX ("unclear novelty")
- **What:** Take vanilla Llama-3.1-8B-Instruct (Meta's official instruct model, NO CITA training). Run all 5 benchmarks with the SAME 10 alignment instructions as system prompts.
- **Variants to test:**
  - Llama-3.1-8B-Instruct + zero-shot system prompt
  - Llama-3.1-8B-Instruct + few-shot system prompt (3 examples)
  - DPO_Instruct (our trained)
  - CITA_Instruct (our trained)
- **Expected outcome:** CITA >> prompting by a large margin on switching metrics. Prompting may get style right but fail on safety boundary switching and calibration.
- **Compute:** Inference only. ~6-8 hours for all variants × 5 benchmarks.
- **Output:** New row in main results table + dedicated comparison table
- **Where in paper:** Section 7, new paragraph. FAQ Q12 currently claims this but has no real data — replace claim with evidence.
- **Why #4:** Answers ALL 3 reviewers' novelty concern. Requires downloading Llama-3.1-8B-Instruct, slightly more setup than Exp 1-3.
- **Rebuttal Figure:** Radar chart (5 axes = 5 benchmarks). 4 overlaid polygons: Zero-shot prompting vs Few-shot prompting vs DPO_Instruct vs CITA_Instruct. 95% CI shaded bands per axis via bootstrap (1,000 resamples). Message: CITA polygon dominates prompting baselines on all axes, especially safety switching and calibration.

---

## Tier 2 — MEDIUM ROI, REQUIRES RETRAINING (~2-3 days)

### Exp 5: Multi-Seed CITA Ablation (Statistical Rigor) [COST: ~6-24 hours training]
- **Answers:** shuX point 6 ("lack of statistical rigor"), shuX point 2 ("KL contribution not fully substantiated")
- **What:** Train CITA_Instruct with 3 different random seeds. Report mean ± std on all 5 benchmarks.
- **Additionally:** Run the KL ablation (λ_KL = 0, 0.0001, 0.00023, 0.0005) with 3 seeds each to show KL effect is robust.
- **Compute:** 3 seeds × ~2 hours per CITA run = ~6 hours. With KL sweep: 4 values × 3 seeds × 2 hours = ~24 hours.
- **Output:** Updated ablation tables with error bars. Updated main results with confidence intervals.
- **Where in paper:** Appendix E (ablations), main results table
- **Rebuttal Figure:** Line plot with error bands. X-axis: λ_KL values (0, 0.0001, 0.00023, 0.0005). Y-axis: ECLIPTICA M1 score. 3 lines (one per seed), plus bold mean line. 95% CI shaded band around mean via 3-seed variance. Message: inverted-U is reproducible across seeds, KL is structurally necessary, not a lucky seed.

### Exp 6: Multi-Model Scaling — Llama-3.2-3B Only [COST: ~4-6 hours training]
- **Answers:** ALL reviewers (unanimous concern)
- **What:** Repeat full pipeline (SFT → DPO → CITA) on Llama-3.2-3B only. Skip 70B for now (too expensive).
- **Compute:** 3B is cheap (~4-6 hours total pipeline on A100-40GB).
- **Output:** Comparison table: 3B vs 8B across 5 benchmarks.
- **Where in paper:** New Section 7 subsection "Scaling Analysis"
- **Why 3B only:** Addresses "multi-model" concern at minimal cost. Shows CITA generalizes down. 70B saved for full revision.
- **Rebuttal Figure:** Side-by-side grouped bar chart. X-axis: 5 benchmarks. Bar pairs: 3B vs 8B for each of CITA_I and DPO_I. 95% CI error bars via bootstrap (1,000 resamples per benchmark). Message: CITA switching pattern holds at 3B scale, not an 8B artifact.

---

## Tier 3 — LOWER ROI or HIGH COST (only if rebuttal deadline allows, otherwise next submission)

### Exp 7: SimPO / KTO Baselines [COST: ~8 hours training]
- **Answers:** gMno ("comparison with SimPO, KTO")
- **What:** Train SimPO and KTO variants (NoInstruct + Instruct) using same PKU-SafeRLHF data, same LoRA config, same SFT checkpoint. Evaluate on all 5 benchmarks.
- **Implementation:** TRL supports both SimPO and KTO trainers.
- **Compute:** ~2 hours per variant × 4 variants = ~8 hours training + evaluation.
- **Output:** 4 new rows in main results table (SimPO_NI, SimPO_I, KTO_NI, KTO_I)
- **Where in paper:** Section 7, expanded comparison table
- **Why Tier 3:** Only gMno asked. gMno is least likely to change score regardless. Low ROI for rebuttal.
- **Rebuttal Figure:** Heatmap. Rows: 6 methods (SimPO_NI, SimPO_I, KTO_NI, KTO_I, DPO_I, CITA_I). Columns: 5 benchmarks. Cell color: green=best, red=worst (column-normalized). 95% CI in cell text (±) via bootstrap. Message: CITA_I still dominates even against newer preference baselines.

### Exp 8: Jailbreak / Adversarial Robustness [COST: ~1-2 days]
- **Answers:** gMno ("SOTA attacks"), KpHA (implicit safety concern)
- **What:** Run adversarial attack benchmarks on CITA vs DPO:
  - GCG attack (Zou et al., 2023)
  - AutoDAN
  - Instruction-override attempts ("ignore previous instructions")
- **Expected outcome:** CITA's trained instruction-conditioning should be more robust than system-prompt-only approaches to instruction override attacks.
- **Compute:** ~1-2 days depending on attack suite.
- **Output:** Attack success rate table: CITA vs DPO vs prompting
- **Where in paper:** Appendix or new Robustness subsection
- **Rebuttal Figure:** Grouped bar chart. X-axis: 3 attack types (GCG, AutoDAN, Instruction-Override). Y-axis: Attack Success Rate (lower=safer). Bars: CITA_I vs DPO_I vs Llama-Instruct-prompting. 95% CI error bars via bootstrap over attack samples. Message: CITA's trained instruction-conditioning resists attacks better than prompt-only control.

### Exp 9: Instruction Scalability (20, 50, 100 types) [COST: significant]
- **Answers:** KpHA weakness 5 ("interference across many instructions")
- **What:** Expand instruction set from 10 → 20 → 50 types. Retrain CITA. Measure whether switching degrades.
- **Approach:** Synthesize additional instruction types (e.g., academic, legal, medical, satirical, Socratic, etc.) using the same 5-judge pipeline.
- **Compute:** Significant — new dataset creation + retraining for each scale point.
- **Output:** Scalability curve: instruction count vs ECLIPTICA score
- **Where in paper:** New analysis section
- **Rebuttal Figure:** Line plot with error bands. X-axis: Number of instruction types (10, 20, 50). Y-axis: ECLIPTICA M1 score. Lines: CITA_I vs DPO_I. 95% CI shaded band via bootstrap at each scale point. Message: CITA switching degrades gracefully (or holds) as instruction count grows, KL anchor prevents collapse at scale.

### Exp 10: Domain Broadening [COST: ~1 day dataset + inference]
- **Answers:** KpHA ("creative writing, coding, tutoring underrepresented"), Aman's point 3
- **What:** Add prompts from non-safety domains: coding (HumanEval-style), creative writing (WritingPrompts), educational (ELI5), legal (LegalBench subset).
- **Compute:** Dataset creation + evaluation. No retraining needed if using existing CITA checkpoint.
- **Output:** Extended ECLIPTICA with domain-stratified results
- **Where in paper:** Appendix D expanded
- **Rebuttal Figure:** Grouped bar chart. X-axis: domains (Safety, Coding, Creative, Tutoring, Legal). Y-axis: ECLIPTICA M1 score. Bars: CITA_I vs DPO_I. 95% CI error bars via bootstrap over prompts per domain. Message: CITA switching transfers to unseen domains beyond safety, not overfit to HH-RLHF distribution.

---

## Execution Timeline (Rebuttal Sprint)

| Day | Experiments | Compute Type | Deliverable |
|-----|------------|-------------|-------------|
| **Day 1 AM** | Exp 1 (per-instruction breakdown) | Post-processing only | 1 table + 1 heatmap |
| **Day 1 PM** | Exp 2 (query-scrambled eval) | Inference only | 1 killer table |
| **Day 2** | Exp 3 (MMLU/GSM8K/HumanEval) | Inference only | 1 capability table |
| **Day 2 parallel** | Exp 4 (system-prompt baseline) | Inference only | 1 comparison table |
| **Day 3** | Write rebuttal with inline results | Writing | Submit rebuttal |
| **Day 4-5** | Exp 5 (multi-seed) + Exp 6 (3B model) | Training | Bonus tables if deadline allows |

---

## Reviewer-to-Experiment Mapping (re-ranked by ROI)

| Concern | gMno | KpHA | shuX | Experiment | Tier |
|---------|------|------|------|------------|------|
| Per-instruction stability / bias | | | X | **Exp 1** | **T1** |
| Shortcut learning | | | X | **Exp 2** | **T1** |
| Capability preservation | | | X | **Exp 3** | **T1** |
| Novelty vs prompting | X | X | X | **Exp 4** | **T1** |
| Statistical rigor / multi-seed | | | X | Exp 5 | T2 |
| Multi-model scaling | X | X | X | Exp 6 (3B only) | T2 |
| SimPO/KTO baselines | X | | | Exp 7 | T3 |
| Jailbreak robustness | X | | | Exp 8 | T3 |
| Instruction scalability (50+) | | X | | Exp 9 | T3 |
| Domain broadening | | X | | Exp 10 | T3 |
| Proofreading / refs | X | | X | Manual fix | — |

---

## Paper Revision Checklist (Non-Experiment)

- [ ] Fix hallucinated reference (OPT-IML) — gMno
- [ ] Fix repeated references (TruthfulQA cited twice, RLHF cited twice) — gMno
- [ ] Fix figure/table spacing manipulation — gMno
- [ ] Tighten "Riemannian chart" language with concrete intuition — KpHA
- [ ] Add training cost comparison table to main text (data already in Appendix C) — KpHA
- [ ] Reduce repeated high-level framing — shuX
- [ ] Clarify Table 1 caption (single checkpoint vs multiple) — KpHA
- [ ] Add gradient intuition sentence to main text (not just appendix) — KpHA
- [ ] Call out DPO ignoring instruction signal explicitly (DPO Δ=+0.001 on TruthfulQA) — KpHA
- [ ] Add societal impact statement — shuX
- [ ] Anonymize preprint properly — shuX flagged non-anonymization
