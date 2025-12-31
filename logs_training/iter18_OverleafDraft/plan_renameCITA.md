# CITA Naming Analysis: Should We Rename?

**Status:** Analysis Complete
**Recommendation:** Keep CITA

---

## Executive Summary

We evaluated renaming **CITA** to a `*PO` name. After literature search:

- **IAPO** - Risky (1 letter from IOPO, anagram of AIPO)
- **ISPO** - Available but 1 letter from InSPO
- **CITA** - Unique, no collision

---

## Namespace Collision Analysis

### Taken Names (Verified)

| Name  | Full Name                                         | Paper                                                                 | Venue      |
|-------|---------------------------------------------------|-----------------------------------------------------------------------|------------|
| IOPO  | Input-Output Preference Optimization              | Empowering LLMs with Complex Instruction Following via IOPO           | ACL 2025   |
| AIPO  | Agreement-aware Iterative Preference Optimization | AIPO: Improving Training Objective for Iterative Preference Opt.      | arXiv 2024 |
| ICPO  | Intrinsic Confidence-Driven Group Relative PO     | ICPO: Intrinsic Confidence-Driven Group Relative PO (Xiaomi)          | 2025       |
| iDPO  | Iterative Direct Preference Optimization          | Iterative Length-Regularized DPO: Improving 7B LMs to GPT-4 Level     | arXiv 2024 |
| InSPO | Intrinsic Self-reflective Preference Optimization | InSPO: Unlocking Intrinsic Self-Reflection for LLM Preference Opt.    | Dec 2025   |

### Candidate Risk Assessment

| Name  | Risk   | Issue                               |
|-------|--------|-------------------------------------|
| IAPO  | HIGH   | 1 letter from IOPO, anagram of AIPO |
| IDPO  | HIGH   | Confusable with iDPO                |
| ISPO  | MEDIUM | 1 letter from InSPO (Dec 2025)      |
| CITA  | NONE   | Unique, no collision                |

---

## ISPO vs CITA: Head-to-Head

### 1) Reviewer Confusion Risk

| Factor                      | ISPO                        | CITA                  | Winner   |
|-----------------------------|-----------------------------|-----------------------|----------|
| Follows *PO convention      | Yes (familiar)              | No (breaks pattern)   | ISPO     |
| "Yet another *PO" fatigue   | HIGH risk                   | LOW - stands out      | **CITA** |
| Near-collision              | InSPO (1 letter, Dec 2025)  | None                  | **CITA** |
| Method clarity in name      | "Switchable" = clear        | "Contrastive" = vague | ISPO     |

**Verdict:** CITA - InSPO collision + reviewer *PO fatigue

### 2) Citation Potential

| Factor                | ISPO                    | CITA                      | Winner   |
|-----------------------|-------------------------|---------------------------|----------|
| Search discoverability| Found with "*PO" queries| Requires specific search  | ISPO     |
| Memorable/unique      | Generic *PO sound       | Unique, pronounceable     | **CITA** |
| Stand-out factor      | Lost in *PO crowd       | Differentiating           | **CITA** |

**Verdict:** CITA - unique names get remembered and cited

### Final Score

| Criterion          | Winner   |
|--------------------|----------|
| Reviewer confusion | **CITA** |
| Citation potential | **CITA** |
| Overall            | **CITA** |

---

## Recommendation

| Rank | Name | Decision                                    |
|------|------|---------------------------------------------|
| 1    | CITA | **KEEP** - Unique, memorable, no collision  |
| 2    | ISPO | Only if reviewers demand *PO naming         |

---

## If Renaming Required (Effort)

| Task                         | Files | Effort |
|------------------------------|-------|--------|
| Replace in .tex files        | 12+   | Medium |
| Modify plot scripts          | 8     | Medium |
| Re-run plots (M1 + GPU)      | -     | High   |

**Total:** ~4-6 hours

---

## Sources

- IOPO: https://arxiv.org/abs/2411.06208
- AIPO: https://arxiv.org/abs/2409.08845
- ICPO: https://arxiv.org/abs/2511.21005
- InSPO: https://arxiv.org/abs/2512.23126
- DPO: https://arxiv.org/abs/2305.18290
- Acronym citation study: https://doi.org/10.1016/j.jss.2025.112277
