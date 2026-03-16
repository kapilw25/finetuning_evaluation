# Bootstrap CI Implementation Status

## Coverage: 3/5 Individual Charts + Heatmap

| Eval | CI Support | Per-Sample Source | Status |
|------|------------|-------------------|--------|
| TruthfulQA | ✓ | Binary `has_uncertainty` difference | Ready |
| Cond. Safety | ✓ | Binary `is_refusal` difference | Ready |
| Length Ctrl | ✓ | Ratio of means (bootstrapped) | Ready |
| ISD | ✗ | Skipped (composite metric) | N/A |
| AQI | ✗ | Cluster-based (CHI, XB) | N/A |

---

## Bug Fix: Per-Sample Metrics Must Match Aggregate

### Original Problem (Error bars extending beyond Y-axis)

| Chart | Bar Shows | Old CI From | **Issue** |
|-------|-----------|-------------|-----------|
| TruthfulQA | rate diff (~0.01) | `uncertainty_total` counts | Wrong units! |
| Cond. Safety | rate diff (~0.4) | `refusal_confidence` | Confidence ≠ binary |
| Length Ctrl | ratio of means | per-prompt ratios | Extreme outliers |
| ISD | fidelity × shift | just fidelity | Missing half metric |

### Fix Applied

| Chart | New Per-Sample Metric | Why It Works |
|-------|----------------------|--------------|
| TruthfulQA | `has_uncertainty` binary diff {-1,0,+1} | Mean = rate difference ✓ |
| Cond. Safety | `is_refusal` binary diff {-1,0,+1} | Mean = rate difference ✓ |
| Length Ctrl | Bootstrap ratio of means directly | Avoids mean(ratios) ≠ ratio(means) ✓ |
| ISD | **Skipped** | Composite metric too complex |

---

## Charts Summary

| Chart | CI Support | Notes |
|-------|------------|-------|
| `truthfulqa_comparison.png` | ✓ | Error bars from binary bootstrap |
| `conditional_safety_comparison.png` | ✓ | Error bars from binary bootstrap |
| `length_control_comparison.png` | ✓ | Error bars from ratio-of-means bootstrap |
| `isd_comparison.png` | ✗ | Skipped (composite metric) |
| `heatmap.png` | ✓ | ±CI values in cells |
| `radar_area.png` | ✗ | Disabled (too cluttered) |
| `aqi_comparison.png` | ✗ | No per-sample data |

---

## Next Steps

```bash
# Re-generate individual plots (now with proper error bars)
python comparative_study/05_evaluation/truthfulqa/evaluation.py --mode full
python comparative_study/05_evaluation/conditional_safety/evaluation.py --mode full
python comparative_study/05_evaluation/length_control/evaluation.py --mode full

# Generate combined plots (heatmap with ±CI)
python comparative_study/05_evaluation/generate_combined_plots.py
```

**Expected Output:**
- 3 individual charts with error bars: TruthfulQA, Cond. Safety, Length Ctrl
- ISD chart: No error bars (composite metric skipped)
- Heatmap: ±CI values for 3/5 evals
- Radar: NO CI bands (too cluttered)
- AQI: No CI (cluster-based metric)

---

## Files Modified

| File | Change |
|------|--------|
| `isd/utils/metrics.py` | Added `per_sample_fidelity` to `ModelMetrics` |
| `isd/evaluation.py` | Saves `per_sample_fidelity`, added bootstrap CI to plot |
| `truthfulqa/evaluation.py` | Added bootstrap CI to comparison plot |
| `conditional_safety/evaluation.py` | Added bootstrap CI to comparison plot |
| `length_control/evaluation.py` | Added bootstrap CI to comparison plot |
| `generate_combined_plots.py` | Added `_load_isd_per_sample()`, disabled radar CI |
| `eval_utils/bootstrap.py` | Bootstrap CI computation (N=1000 resamples) |
| `eval_utils/plotting.py` | Error bars, ±values in plots |
