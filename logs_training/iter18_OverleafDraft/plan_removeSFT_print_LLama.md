# Plan: Remove SFT from Plots + Add MODEL_NAME to Titles

## Summary
- **SFT** is a training stage, not a policy optimization method
- Remove SFT_Instruct & SFT_NoInstruct from all evaluation bar charts
- Add `(Llama-3.1-8B)` to all plot titles for clarity

---

## Files Modified

| File | Change |
|------|--------|
| `eval_utils/model_loader.py:35` | Added `MODEL_NAME = "Llama-3.1-8B"` |
| `eval_utils/__init__.py` | Exported `MODEL_NAME` |
| `eval_utils/plotting.py:21` | Imported `MODEL_NAME` |
| `eval_utils/plotting.py:203-210` | SFT filtering in `generate_comparison_plots()` |
| `eval_utils/plotting.py:257-259` | Auto-append MODEL_NAME to bar chart titles |
| `eval_utils/plotting.py:417-420` | Auto-append MODEL_NAME to box plot titles |
| `eval_utils/plotting.py:460-463` | Auto-append MODEL_NAME to violin plot titles |

---

## Implementation Details

### 1. Central Configuration (`model_loader.py`)
```python
MODEL_NAME = "Llama-3.1-8B"  # Short name for figure titles
```

### 2. SFT Filtering (`plotting.py`)
```python
# Filter out SFT models (SFT is a training stage, not a policy optimization method)
filtered_indices = [i for i, m in enumerate(models) if not m.startswith('SFT')]
```

### 3. MODEL_NAME Auto-Append

| Function | Title Format |
|----------|--------------|
| `generate_comparison_plots` | `{title} (Llama-3.1-8B)` |
| `generate_boxviolin_chart` (Box) | `{title} (Llama-3.1-8B) (Box Plot)` |
| `generate_boxviolin_chart` (Violin) | `{title} (Llama-3.1-8B) (Violin Plot)` |
| `generate_radar_chart_area_based` | Via `model_name` param |
| `generate_combined_heatmap` | Via `model_name` param |

---

## Output Files to Regenerate

Run evaluations on A10-24GB GPU to regenerate:

### Per-Benchmark Bar Charts (SFT removed + MODEL_NAME added)
```
outputs/evaluation/
├── ISD_Evaluation_Embedding/
│   └── isd_comparison.{pdf,png}
├── TruthfulQA_Evaluation/
│   ├── truthfulqa_comparison.{pdf,png}
│   ├── truthfulqa_adaptation_distribution_box.{pdf,png}
│   └── truthfulqa_adaptation_distribution_violin.{pdf,png}
├── Conditional_Safety_Evaluation/
│   ├── conditional_safety_comparison.{pdf,png}
│   ├── conditional_safety_adaptation_distribution_box.{pdf,png}
│   └── conditional_safety_adaptation_distribution_violin.{pdf,png}
├── Length_Control_Evaluation/
│   ├── length_control_comparison.{pdf,png}
│   ├── length_control_adaptation_distribution_box.{pdf,png}
│   └── length_control_adaptation_distribution_violin.{pdf,png}
└── AQI_Evaluation/
    ├── aqi_comparison.{pdf,png}
    └── aqi_per_axiom_comparison.{pdf,png}
```

### Combined Plots (MODEL_NAME added)
```
outputs/evaluation/combined_plots/
├── heatmap.{pdf,png}
├── heatmap_no_ci.{pdf,png}
└── radar_area.{pdf,png}
```

### Files to DELETE (lollipop removed)
```
*_lollipop.{pdf,png}  # Dead code removed - these won't regenerate
```

---

## No Changes Needed

Individual evaluation scripts (`isd/evaluation.py`, etc.) do NOT need modification.
They call `generate_comparison_plots()` which now handles SFT filtering and MODEL_NAME automatically.
