# 📝 Official Review of Submission500 by Reviewer KpHA

> 🕐 Official Reviewby Reviewer KpHA23 Apr 2026, 10:09 (modified: 28 Apr 2026, 10:24)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer KpHA, AuthorsRevisions

---

## 📄 Paper Summary:

This paper tackles a genuinely practical problem: LLM alignment is usually baked into the weights at training time and then frozen, which is increasingly at odds with real-world deployments where the same base model needs to serve different roles (customer support, compliance, creative writing, etc.) with different behavioral postures. The authors propose reframing alignment as an instruction-driven, runtime-switchable control mechanism. Their concrete contributions are: (1) CITA, a training algorithm that conditions preference optimization on explicit alignment instructions and uses a mandatory KL anchor to prevent mode collapse, and (2) ECLIPTICA, a diagnostic benchmark with 3,000 test cases designed to isolate the causal effect of alignment instructions by holding the user prompt fixed while varying only the instruction.

---

## ✅ Summary Of Strengths:

- 🟢 The problem is well-motivated and increasingly relevant. As LLMs get deployed as agentic backbones behind multiple products, the "one model, many behavioral contracts" scenario is not hypothetical—it is happening right now. The paper articulates this deployment pain point clearly.

- 🟢 CITA's objective is theoretically grounded. Making the preference loss conditional on the alignment instruction I (i.e., learning π(Y|I,X)) rather than learning a single implicit preference regime π(Y|X) is a principled departure from standard DPO. The mandatory KL anchor is structurally important here—without it, gradients from competing instructions would interfere or collapse into a single dominant mode. The self-quenching and directional purity analysis in the gradient derivation adds depth.

- 🟢 The ECLIPTICA benchmark is carefully designed. Holding the user prompt X constant while varying only I is the right way to isolate instruction-conditioned alignment from superficial instruction following. The instruction synthesis pipeline (5 judge models → BERTScore filtering → human quality gate) is more rigorous than hand-writing a few tone keywords.

- 🟢 The evaluation is broad. Five different benchmarks (ECLIPTICA, TruthfulQA, Conditional Safety, Length Control, LITMUS) cover truthfulness, safety, verbosity control, and general alignment quality—not just one narrow dimension.

---

## ❌ Summary Of Weaknesses:

- 🔴 The core distinction from standard instruction following is murky. The authors repeatedly claim they are doing "instruction-conditioned alignment" rather than "prompt hacks," but at inference time CITA still concatenates the alignment instruction with the user prompt (Table 1 essentially shows system-prompt-like instructions prepended to the user query). While the training objective is indeed conditional, the runtime switching mechanism is textually indistinguishable from asking a model "Respond concisely and professionally." The paper needs to more clearly articulate what CITA learns that a well-tuned model cannot already do via careful system prompt engineering. The bad cases shown for DPO/PPO/GRPO in Figure 1 look more like alignment failures than inherent limitations of static alignment.

- 🔴 No discussion of training cost. CITA requires preference quadruples (I, X, Y+, Y−) across multiple instruction conditions. How much more data and compute does this need compared to standard DPO? The Fisher metric computation and KL anchoring add overhead—are these negligible or prohibitive at larger scales? Without this, it is hard to judge whether the switchability gains are worth the cost.

- 🔴 Only one model size is tested. All experiments are on Llama-3.1-8B. For a method whose main selling point is deployment flexibility, the absence of results on larger models (e.g., 70B) or smaller ones (e.g., 3B) is a significant gap. Does switchability degrade or improve with scale? The community needs to know.

- 🔴 The benchmark coverage is narrow in domain. The 300 prompts are all drawn from HH-RLHF, which skews toward safety/helpfulness scenarios. Creative writing, coding assistance, and educational tutoring—domains where style switching is arguably even more valuable—are underrepresented. Also, 10 instruction types may not capture the richness of real deployment needs.

- 🔴 No analysis of interference across many instructions. The paper shows the model can switch among 10 instructions, but what happens at 50 or 100? Does CITA's KL anchor still suffice, or do instructions start bleeding into each other? Scalability of the switchable policy space is a crucial question for deployment and is not addressed.

---

## 💬 Comments Suggestions And Typos:

- 🟡 Table 1 is an effective teaser. However, clarify in the caption whether all responses come from a single trained checkpoint or different ones.

- 🟡 The gradient derivation (self-quenching, directional purity) is relegated to the appendix, but its key intuitions deserve a sentence or two in the main text for readers who skip math.

- 🟡 Figure 5 shows CITA_Instruct and CITA_NoInstruct have a clear gap, but DPO_Instruct and DPO_NoInstruct barely differ. If this means DPO effectively ignores the instruction signal, that is a strong finding—call it out explicitly.

- 🟡 4.The "shared manifold neighborhood" / "Riemannian chart" language is elegant but could use a concrete intuition for non-geometry readers. Minor: "shared manifold neighborhood" is used before being formally defined; Figure 3 caption has some rendering issues with ∆θ.

---

## 📊 Scores & Ratings

| 🏷️ Category | 🎯 Score |
|---|---|
| 🧠 **Confidence** | 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings. |
| 🔬 **Soundness** | 4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential. |
| 🎆 **Excitement** | 3.5 |
| 📋 **Overall Assessment** | 3.5 = Borderline Conference |

---

## ⚖️ Ethical Concerns:

There are no concerns with this submission

---

## 📎 Additional Metadata

| 🏷️ Field | 📌 Value |
|---|---|
| 🔄 **Reproducibility** | 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method. |
| 📦 **Datasets** | 4 = Useful: I would recommend the new datasets to other researchers or developers for their ongoing work. |
| 💻 **Software** | 3 = Potentially useful: Someone might find the new software useful for their work. |
| 🕵️ **Knowledge Of Or Educated Guess At Author Identity** | No |
| 📰 **Knowledge Of Paper** | N/A, I do not know anything about the paper from outside sources |
| 📰 **Knowledge Of Paper Source** | N/A, I do not know anything about the paper from outside sources |
| 📰 **Impact Of Knowledge Of Paper** | N/A, I do not know anything about the paper from outside sources |

---

## ✍️ Reviewer Certification:

I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.

## 📜 Publication Ethics Policy Compliance:

I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
