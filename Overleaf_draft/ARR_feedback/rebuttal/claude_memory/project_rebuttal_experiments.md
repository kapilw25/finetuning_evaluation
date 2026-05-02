---
name: Rebuttal experiment plan and priorities
description: 10 experiments ranked by ROI for rebuttal sprint, Tier 1 is all inference-only targeting shuX, decided 2026-05-02
type: project
originSessionId: 33ca0014-4198-4082-a482-1e9bff1bb271
---
Kapil decided to execute experiments and include results inline in rebuttal (Aman approved).

**Tier 1 (inference-only, ~1.5 days, DO FIRST):**
1. Exp 1: Per-instruction breakdown — FREE, post-process existing outputs. Answers shuX 4,5.
2. Exp 2: Query-scrambled eval — hours, kills "shortcut learning" (shuX 3,8). HIGHEST impact single experiment.
3. Exp 3: MMLU/GSM8K/HumanEval — ~1 day inference, capability preservation (shuX 1).
4. Exp 4: System-prompt baseline — ~6-8h inference, CITA vs vanilla Llama-Instruct prompting (all 3 reviewers).

**Tier 2 (requires retraining, ~2-3 days):**
5. Exp 5: Multi-seed CITA (3 seeds) — statistical rigor (shuX 2,6).
6. Exp 6: Llama-3.2-3B scaling — multi-model (all reviewers). Skip 70B.

**Tier 3 (deferred, low ROI for rebuttal):**
7-10: SimPO/KTO, jailbreak, instruction scalability, domain broadening.

**Key decision:** SimPO/KTO demoted to Tier 3 despite Aman listing it first — only gMno asked, gMno unlikely to move. Kapil should flag this to Aman.

**Rebuttal format constraint:** OpenReview Official Comments do NOT support images. Rebuttal results must be markdown tables with inline numbers. Plots are for the revised paper only.

**How to apply:** When coding experiments, output results as markdown tables and CSV, not just plots. Every experiment should produce a table suitable for pasting into OpenReview.
