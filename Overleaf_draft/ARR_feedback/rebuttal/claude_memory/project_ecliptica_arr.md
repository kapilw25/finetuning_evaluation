---
name: ECLIPTICA ARR Submission Status
description: ARR March 2026 submission #500, scores 2/3.5/2, rebuttal sprint + revision plan active as of 2026-05-02
type: project
originSessionId: 33ca0014-4198-4082-a482-1e9bff1bb271
---
- **Paper:** ECLIPTICA — instruction-driven runtime-switchable LLM alignment via CITA (Contrastive Instruction-Tuned Alignment)
- **Venue:** ACL Rolling Review, March 2026 cycle, Submission #500
- **Previous:** Desk-rejected Jan 2026 (page limit + single-column appendix). Fixed and resubmitted.
- **Current scores:** gMno=2, KpHA=3.5, shuX=2 (avg=2.5)
- **Target:** avg >= 3.0 for Findings consideration
- **Key result:** CITA achieves 86.7% instruction-alignment efficiency on Llama-3.1-8B across 5 benchmarks

**Why:** Probability of score change from rebuttal alone is ~15-20%. Full revision + resubmit to next venue (EMNLP 2026) has ~45-60% chance.

**How to apply:** Aman approved running new experiments and including results inline in rebuttal. Sprint plan in `Overleaf_draft/ARR_feedback/rebuttal/rebuttal_experiments.md`. Primary target is moving shuX from 2→3 (most actionable reviewer).

**Reviewer personality map:**
- gMno: Short, disengaged review. Score 2/2/2. Unlikely to re-engage. Wants SimPO/KTO, jailbreak tests, proofreading.
- KpHA: Detailed, constructive. Score 3.5/4/3.5. Already positive. Main concern: CITA vs system prompting distinction.
- shuX: Detailed, 7 concrete suggestions. Score 2/2.5/3. MOST MOVABLE. Wants MMLU, query-scrambled eval, multi-seed, capability preservation. Also flagged non-anonymized preprint.
