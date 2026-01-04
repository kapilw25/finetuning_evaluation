# Plan: ISD → ECLIPTICA + Metric Notation

## Code Changes

### 1. `generate_combined_plots.py` (lines 50-65)

```python
# BEFORE
EVAL_DIRS = {
    "ISD": EVAL_OUTPUTS_DIR / "ISD_Evaluation_Embedding",
    "TruthfulQA": EVAL_OUTPUTS_DIR / "TruthfulQA_Evaluation",
    "Cond. Safety": EVAL_OUTPUTS_DIR / "Conditional_Safety_Evaluation",
    "Length Ctrl": EVAL_OUTPUTS_DIR / "Length_Control_Evaluation",
    "AQI": EVAL_OUTPUTS_DIR / "AQI_Evaluation",
}

# AFTER
EVAL_DIRS = {
    "ECLIPTICA (M₁)": EVAL_OUTPUTS_DIR / "ISD_Evaluation_Embedding",
    "TruthfulQA (M₂)": EVAL_OUTPUTS_DIR / "TruthfulQA_Evaluation",
    "Cond. Safety (M₃)": EVAL_OUTPUTS_DIR / "Conditional_Safety_Evaluation",
    "Length Ctrl (M₄)": EVAL_OUTPUTS_DIR / "Length_Control_Evaluation",
    "LITMUS (AQI-M₅)": EVAL_OUTPUTS_DIR / "AQI_Evaluation",
}

METRIC_KEYS = {
    "ECLIPTICA (M₁)": "instruction_awareness_score",
    "TruthfulQA (M₂)": "adaptation_score",
    "Cond. Safety (M₃)": "adaptation_score",
    "Length Ctrl (M₄)": "adaptation_score",
    "LITMUS (AQI-M₅)": "aqi_score",
}
```

### 2. `eval_utils/plotting.py` (~line 996)

```python
# BEFORE
def _wrap_eval_labels(eval_names: List[str]) -> List[str]:
    wrapped = []
    for name in eval_names:
        if name == 'TruthfulQA':
            wrapped.append('Truthful\nQA')
        elif name == 'Cond. Safety':
            wrapped.append('Cond.\nSafety')
        elif name == 'Length Ctrl':
            wrapped.append('Length\nCtrl')
        else:
            wrapped.append(name)
    return wrapped

# AFTER
def _wrap_eval_labels(eval_names: List[str]) -> List[str]:
    """Map eval names to short metric labels for heatmap columns."""
    metric_map = {
        'ECLIPTICA (M₁)': 'M₁',
        'TruthfulQA (M₂)': 'M₂',
        'Cond. Safety (M₃)': 'M₃',
        'Length Ctrl (M₄)': 'M₄',
        'LITMUS (AQI-M₅)': 'AQI',
    }
    return [metric_map.get(name, name) for name in eval_names]
```

### 3. `isd/evaluation.py` (~line 439)

```python
# BEFORE
title="ISD: Instruction Awareness (Higher = Better)"

# AFTER
title="ECLIPTICA: Instruction Awareness (Higher = Better)"
```

### 4. `eval_utils/plotting.py` (~line 881) - Comment only

```python
# BEFORE
# ISD is at angle ~0 (rightmost)

# AFTER
# ECLIPTICA is at angle ~0 (rightmost)
```

## Expected Output

| Plot | Labels |
|------|--------|
| Heatmap | M₁, M₂, M₃, M₄, AQI |
| Radar | ECLIPTICA (M₁), TruthfulQA (M₂), Cond. Safety (M₃), Length Ctrl (M₄), LITMUS (AQI-M₅) |

## Execution

```bash
# On GPU server
python comparative_study/05_evaluation/isd/evaluation.py  # Option 1
python comparative_study/05_evaluation/generate_combined_plots.py

# On local
python Overleaf_draft/src/sync_figures.py
```
