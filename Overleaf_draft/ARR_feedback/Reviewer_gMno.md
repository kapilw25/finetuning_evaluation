# 📝 Official Review of Submission500 by Reviewer gMno

> 🕐 Official Reviewby Reviewer gMno19 Apr 2026, 13:33 (modified: 28 Apr 2026, 10:24)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer gMno, AuthorsRevisions

---

## 📄 Paper Summary:

This paper studies instruction conditioned alignment to enable switchable alignment policies for LLMs.

---

## ✅ Summary Of Strengths:

- 🟢 This paper studies an important problem that fits within the scope of *ACL.
- 🟢 Empirical studies are applied to demonstrate the usefulness of CITA.

---

## ❌ Summary Of Weaknesses:

- 🔴 The paper overstates its novelty. The paper could benefit from comparison with
- 🔴 SimPO, KTO which use regularization to trade-off policies
- 🔴 Multi-objective decoding
- 🔴 Instruction set has 10 types, which is very limited. Multilingual analysis is missing.
- 🔴 Empirical studies uses Llama-3.11-8B. More models should be used for evaluations to demonstrate generalizability.
- 🔴 Trade-off analysis on helpfulness, safety when using switchable policy should be more detailed.
- 🔴 The paper could benefit from comparison with other safety baselines.
- 🔴 CITA should be tested against SOTA attacks such as jailbreak attacks.
- 🔴 The paper needs significant and careful proofreading. The paper contains repeated references such as TruthfulQA, RLHF, hallucinated reference (OPT-IML).

---

## 💬 Comments Suggestions And Typos:

- 🟡 The paper manipulates the space between main body text and figure/table captions, which makes the paper hard to read than it should.

---

## 📊 Scores & Ratings

| 🏷️ Category | 🎯 Score |
|---|---|
| 🧠 **Confidence** | 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings. |
| 🔬 **Soundness** | 2 = Poor: Some of the main claims are not sufficiently supported. There are major technical/methodological problems. |
| 🎆 **Excitement** | 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community. |
| 📋 **Overall Assessment** | 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle. |

---

## ⚖️ Ethical Concerns:

There are no concerns with this submission

---

## 📎 Additional Metadata

| 🏷️ Field | 📌 Value |
|---|---|
| 🛡️ **Needs Ethics Review** | No |
| 🔄 **Reproducibility** | 2 = They would be hard pressed to reproduce the results: The contribution depends on data that are simply not available outside the author's institution or consortium and/or not enough details are provided. |
| 📦 **Datasets** | 3 = Potentially useful: Someone might find the new datasets useful for their work. |
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
