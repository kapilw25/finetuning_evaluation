# 📝 Official Review of Submission500 by Reviewer shuX

> 🕐 Official Reviewby Reviewer shuX20 Apr 2026, 12:38 (modified: 28 Apr 2026, 10:24)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer shuX, AuthorsRevisions

---

## 📄 Paper Summary:

This paper proposes ECLIPTICA, a framework for treating alignment as an instruction-conditioned, runtime-controllable interface, and CITA, a preference optimization method that incorporates instruction conditioning with an explicit KL anchor. The key idea is to enable a single model to switch between multiple behavioral policies (e.g., safety, tone, verbosity) using natural language instructions. The paper also introduces a benchmark that isolates instruction effects by holding prompts constant. Experiments on Llama-3.1-8B show improved instruction sensitivity over DPO, PPO, and GRPO across several benchmarks.

---

## ✅ Summary Of Strengths:

- 🟢 Timely and relevant problem formulation: Treating alignment as a runtime control interface is a compelling direction, particularly for agentic systems and multi-policy deployment settings where a single model must adapt to different behavioral contracts.
- 🟢 Empirical improvements: The method demonstrates consistent gains in instruction sensitivity across multiple benchmarks, particularly on calibration and alignment-quality metrics.
- 🟢 Benchmark contribution: The introduction of a prompt-held-constant benchmark to isolate instruction-conditioned behavior is a meaningful contribution. The dataset construction pipeline is also interesting; if validated carefully (e.g., ruling out shortcut learning), it has the potential to be a useful resource for the community.

---

## ❌ Summary Of Weaknesses:

- 🔴 Limited evaluation scope: Results are restricted to a single backbone (Llama-3.1-8B), limiting claims of generality across architectures and scales.
- 🔴 Missing capability evaluation: The paper does not assess standard capabilities (e.g., MMLU, GSM8K, HumanEval). Given known trade-offs in alignment tuning, it is important to verify that base capabilities are preserved.
- 🔴 Unclear novelty over prior work: The method is closely related to DPO with instruction conditioning and explicit KL regularization. The paper does not clearly demonstrate behavior beyond existing approaches.
- 🔴 KL contribution not fully substantiated: While the paper emphasizes a "mandatory KL" anchor, KL-based stabilization is standard in prior work. The current experiments do not convincingly establish that this component is uniquely necessary.
- 🔴 Theoretical overstatement: The gradient analysis largely follows standard DPO formulations, and the claimed properties do not appear specific to the proposed method.
- 🔴 Lack of statistical rigor: Key ablations are single-seed and do not report variance.
- 🔴 Benchmark construction concerns: Instruction types are derived from query-specific preference pairs and then clustered into fixed categories. This raises the possibility that instruction classes may encode latent biases from their source distribution (e.g., certain instructions correlating with particular query types), which weakens the claim of clean causal control.
- 🔴 Potential shortcut learning: Because both training and evaluation vary instructions while holding the query fixed, the setup does not fully rule out degenerate solutions where the model maps instructions directly to response patterns, rather than learning genuine interaction between instruction and query.
- 🔴 Presentation: The paper contains repeated high-level framing that obscures the core technical contribution.

---

## 💬 Comments Suggestions And Typos:

- 🟡 (1) Evaluate capability preservation:(e.g., MMLU, GSM8K, HumanEval)
- 🟡 (2) Isolate the contribution of KL: Perform a controlled ablation along with a joint (β,λ) sweep and multi-seed results to establish whether KL plays a uniquely necessary role.
- 🟡 (3) Test instruction–query interaction explicitly: Add a diagnostic such as: query-scrambled evaluation (replace X while keeping I fixed), or instruction-only evaluation to verify that the model depends jointly on (I,X) rather than instruction shortcuts.
- 🟡 (4) Analyze instruction distribution: Provide statistics showing that instruction types are balanced across query categories and that their semantics are not biased by the underlying data distribution.
- 🟡 (5) Evaluate multi-regime stability: Report per-instruction performance, loss curves, or interference analysis to demonstrate stable learning across multiple alignment regimes.
- 🟡 (6) Evaluate on more models, variations in backbone and size.
- 🟡 (7) Report inter-annotator agreements for LLM judges as well as humans, even for the rejected samples.
- 🟡 (7) Tighten presentation

---

## 📊 Scores & Ratings

| 🏷️ Category | 🎯 Score |
|---|---|
| 🧠 **Confidence** | 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design. |
| 🔬 **Soundness** | 2.5 |
| 🎆 **Excitement** | 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time. |
| 📋 **Overall Assessment** | 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle. |

---

## 🌍 Limitations And Societal Impact:

The societal impact or broad impact is not mentioned. However the authors do a good job of highlighting the limitations. One cannot be fixed all of them in a single paper, but clearly stating them helps the community

---

## ⚖️ Ethical Concerns:

There are no concerns with this submission

---

## 📎 Additional Metadata

| 🏷️ Field | 📌 Value |
|---|---|
| 🛡️ **Needs Ethics Review** | No |
| 🔄 **Reproducibility** | 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available. |
| 📦 **Datasets** | 4 = Useful: I would recommend the new datasets to other researchers or developers for their ongoing work. |
| 💻 **Software** | 4 = Useful: I would recommend the new software to other researchers or developers for their ongoing work. |
| 🕵️ **Knowledge Of Or Educated Guess At Author Identity** | Yes |
| 📰 **Knowledge Of Paper** | After the review process started |
| 📰 **Knowledge Of Paper Source** | other (specify) |
| 📰 **Knowledge Of Paper Source Other** | One of the reviewer posted a link to the paper |
| 📰 **Impact Of Knowledge Of Paper** | Not at all |
| 📰 **Knowledge Of Authors Guess** | The preprint is not anonymized. |

---

## ✍️ Reviewer Certification:

I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.

## 📜 Publication Ethics Policy Compliance:

I used the Revas tool to check for review issues (https://revas.mbzuai.ac.ae)
