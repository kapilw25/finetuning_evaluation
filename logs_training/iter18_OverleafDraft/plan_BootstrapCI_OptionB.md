# Statistical Significance Plan: Error Bars for ACL 2026

## Three Options Compared

| Option | Method | GPU Time | Coding | Statistical Validity |
|--------|--------|----------|--------|---------------------|
| A | Re-train 3× seeds | ~150h (weeks) | None | Gold standard |
| B | Bootstrap CI | ~10h (overnight) | ~1h | Strong (same data) |
| C | Multiple eval runs | ~30h (3× evals) | ~30min | Moderate |

**Selected: Option B (Bootstrap CI)** — Best ROI: strong stats with minimal GPU cost.

---

## Why Option B?

- **Option A** is overkill: re-training 3× seeds takes weeks of GPU
- **Option C** is 3× the GPU cost of B for weaker statistical validity
- **Option B** uses bootstrap resampling: 1 eval run → 1000 virtual resamples

---

## Implementation Steps

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: Modify eval scripts to save per-sample scores (~1h)      │
├──────────────────────────────────────────────────────────────────┤
│ Files: isd/, truthfulqa/, conditional_safety/,                   │
│        length_control/, AQI/ → evaluation.py                     │
│                                                                  │
│ BEFORE:  {"instruction_awareness_score": 0.85}  ← aggregate      │
│ AFTER:   {"per_sample": [0.92, 0.78, 0.89, ...]} ← per-sample    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: Re-run evaluations (~10h GPU, overnight)                 │
├──────────────────────────────────────────────────────────────────┤
│ 5 benchmarks × 10 models = 50 runs                               │
│                                                                  │
│ ISD ──────────► ~3.3h                                            │
│ TruthfulQA ───► ~2.5h                                            │
│ Cond. Safety ─► ~1.7h                                            │
│ Length Ctrl ──► ~1.7h                                            │
│ AQI ──────────► ~0.8h                                            │
│                 ─────                                            │
│           TOTAL: ~10h                                            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: Compute Bootstrap CI (~2 min CPU)                        │
├──────────────────────────────────────────────────────────────────┤
│ for i in 1..1000:                                                │
│     resampled = random.choices(per_sample_scores, k=N)           │
│     means[i] = mean(resampled)                                   │
│                                                                  │
│ CI_95 = [percentile(means, 2.5), percentile(means, 97.5)]        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: Update plots with error bars (~30 min)                   │
├──────────────────────────────────────────────────────────────────┤
│ • generate_combined_plots.py                                     │
│ • eval_utils/plotting.py → add shaded CI to radar chart          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ RESULT: CITA: 95.6% ± 1.2% [94.4%, 96.8%]                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Time Summary

| Step | Task | Time |
|------|------|------|
| 1 | Modify eval scripts | ~1h coding |
| 2 | Re-run evaluations | ~10h GPU (overnight) |
| 3 | Bootstrap CI math | ~2 min CPU |
| 4 | Update plots | ~30 min coding |
| **Total** | | **~1.5h coding + ~10h GPU** |

---

## Note on "2 minutes"

The "2 minutes" refers ONLY to Step 3 (NumPy statistics).
Steps 1-2 are prerequisites because **per-sample scores are not currently saved**.
