# ACL 2026 Paper Review Simulation: ECLIPTICA/CITA

**Paper:** ECLIPTICA: Instruction-Driven Alignment for Switchable LLM Behavior
**Venue:** ACL 2026 (San Diego, CA, July 2-7, 2026)

---

## All 10 Reviewer Rounds

| # | Reviewer Type | Score | Decision | Strengths | Weaknesses | Critical Question | Missing |
|---|---------------|-------|----------|-----------|------------|-------------------|---------|
| 1 | Alignment/Safety | 6/10 | Weak Accept | Clear problem framing. Mandatory KL insight novel. TruthfulQA 54x compelling. | No Constitutional AI comparison. Safety eval limited to 1 benchmark. Dual-use mitigations untested. | *"How does CITA compare to better system prompt with DPO?"* | Jailbreak resistance, adversarial robustness |
| 2 | Preference Learning | 7/10 | Accept | Loss formulation sound. Gradient self-quenching. KL trust-region elegant. | No convergence analysis. Why KL helps CITA but DPO works without? | *"What's the theoretical insight beyond conditioning on (I,X)?"* | Sample complexity, PAC-Bayes bounds |
| 3 | Instruction Tuning | 7/10 | Accept | ISD well-designed (300x10). Clear following vs alignment distinction. | No FLAN/T0 comparison. Only 30% generalization to novel types. | *"If 60% on similar types, how is this better than few-shot prompting?"* | MT-Bench eval, cross-lingual |
| 4 | **Benchmark/Eval** | **5/10** | **Borderline Reject** | 5 benchmarks, deterministic metrics. No LLM-as-judge. ISD public. | **95.6% undefined in main text**. No error bars/p-values. | *"How is 95.6% computed? What are error bars?"* | Bootstrap CI, multiple seeds |
| 5 | LLM Training | 7/10 | Accept | Clear pipeline. LoRA accessible. Single A100-40GB reproducible. | PPO logs empty. 4-stage pipeline no direct ablation. | *"Skip DPO, train CITA on SFT directly - how much loss?"* | Inference latency, memory footprint |
| 6 | Agentic AI | 8/10 | Strong Accept | "One model, many contracts" practical. Teaser shows switching. | No agent benchmark (SWE-bench). Not tested on tool use. | *"Can you show CITA in agent loop with mid-task instruction changes?"* | Tool-use eval, ReAct integration |
| 7 | **Methodology Critic** | **5/10** | **Borderline Reject** | Same data (PKU-SafeRLHF). Code/models public. | **PPO/GRPO on different hardware**. DPO wins 2/5 including ISD. | *"Fair comparison? PPO 17h, GRPO 12h, CITA 2h."* | PPO/GRPO Optuna tuning |
| 8 | **Theoretical ML** | **4/10** | **Reject** | Gradient analysis correct. Trust-region reasonable. | No identifiability proof. "Switchable policy family" hand-wavy. | *"What guarantees pi_theta factorizes instruction from prompt effects?"* | Convergence bounds, sample complexity |
| 9 | Industry/Applied | 8/10 | Strong Accept | Solves multi-tenant problem. Cost savings (1 vs N models). | No production case study. Latency not measured. | *"30% novel instruction success concerning for production."* | A/B tests, enterprise deployment |
| 10 | Ethics/Responsible AI | 6/10 | Weak Accept | Dual-use discussed. Hierarchical handling proposed. | Mitigations untested. No red-teaming results. | *"Could adversaries use CITA to make models MORE harmful?"* | Red-team eval, safety floor verification |

**Average Score:** 6.3/10

---

## Acceptance Probability

| Outcome | Prob | Reasoning |
|---------|------|-----------|
| **Main Conference** | **35%** | Average 6.3 borderline. Two 5/10 + one 4/10 champion rejection. Missing stats and theory are major flags. |
| **Findings** | **45%** | Strong empirical contribution, public artifacts. Likely AC recommendation if main rejected. |
| **Reject** | **20%** | Theory (4/10) and methodology (5/10) could block if arguing baseline unfairness. |

---

## ROI Comparison: What Helps Most?

| Priority | Improvement | Effort | Acceptance Boost | ROI |
|----------|-------------|--------|------------------|-----|
| 1 | **Define 95.6% metric in main text** | Very Low (edit) | +10 pp | **VERY HIGH** |
| 2 | **Multiple seeds + error bars** | Low (re-run 3x) | +15-20 pp | **HIGH** |
| 3 | **Fair baseline (Optuna for PPO/GRPO)** | Medium | +10 pp | MEDIUM |
| 4 | **DPO+instruction ablation** | Medium | +8 pp | MEDIUM |
| 5 | **Red-team adversarial instructions** | Medium | +5 pp | MEDIUM |
| 6 | **Multi-model (Mistral + 70B)** | HIGH (weeks) | +10 pp | LOW |

**Projected improvement:** 35% → **70-75%** if priorities 1-4 addressed.

---

## Critical Gaps Checklist

### Must Fix (Blocking)

- [✅]95.6% metric definition missing in main text (Added to `7_results.tex:9-15` with Equation 1)
- [ ] No statistical significance / error bars
- [⚠️] Baseline comparison fairness (hardware, tuning effort) — **Addressed in FAQ Q7b-Q7d**: Published HPs for baselines, Optuna only for novel CITA, compute constraints documented (PPO 17h, GRPO 12h per run)

### Should Fix (Strengthen)

- [ ] DPO+instruction ablation (isolate KL contribution)
- [ ] Adversarial instruction robustness

### Nice to Have (Future Work)

- [ ] Agent benchmark (SWE-bench, WebArena)
- [ ] Multi-model generalization (Mistral, Llama-70B)
- [ ] Production deployment case study
