# Arxiv Draft Update Plan - CITA Paper

## Overview
Update Overleaf draft to include all 5 trainers (SFT, PPO, GRPO, DPO, CITA) and fix various issues before Arxiv submission.

---

## TODO LIST

### 🔴 1. RADAR PLOT: Change from Pentagon Area to Weighted Average Radius

**Current State:**
- Uses polygon area formula: `Area = 0.5 * sin(2π/n) * Σ(rᵢ * rᵢ₊₁)`
- Shows: CITA 91.3%, DPO 42.7%, GRPO 36.4%, PPO 23.4%, SFT 2.2%

**Required Change:**
- Replace with weighted average of radius (already normalized 0-100%)
- Formula: `weighted_avg = Σ(rᵢ * wᵢ) / Σ(wᵢ)` where weights can be equal

**Files to Modify:**
- `comparative_study/05_evaluation/eval_utils/plotting.py` (lines 990-1091)
  - Modify `calculate_polygon_area()` → `calculate_weighted_radius()`
  - Update `generate_radar_chart_asrea_based()` function
- `comparative_study/05_evaluation/generate_combined_plots.py`

**Output:**
- `outputs/evaluation/combined_plots/radar_area.pdf` → regenerate

---

### 🔴 2. UPDATE NARRATIVE: Replace "wins X/5" with Weighted Average Ranking

**Files to Modify:**
- `Overleaf_draft/7_results.tex` (lines 152-158) - radar figure caption
- `Overleaf_draft/7_results.tex` (lines 41-62) - Table 11 caption
- `Overleaf_draft/8_conclusion.tex` - key findings section

**Current:**
```
CITA wins 3/5 benchmarks with +48.6% margin over DPO
```

**New (example - actual numbers after recalculation):**
```
CITA achieves XX% weighted average (rank #1), outperforming DPO (XX%), GRPO (XX%), PPO (XX%), SFT (XX%)
```

---

###  ✅  3. UPDATE TABLES TO INCLUDE ALL 5 TRAINERS

####  ✅  3.1 Table 2 (Method Comparison) - `1_introduction.tex` lines 116-134
- **Label:** `tab:method_comparison`
- **Current:** DPO vs CITA (2 columns)
- **Required:** Add PPO, GRPO, SFT columns (5 total)
- Properties: Reward Model Required, Instruction-Aware, Behavioral Switching, Mandatory KL, Dynamic Policy, Agent-Compatible

####  ✅  3.2 Table 4 (Training Config) - `2_methodology.tex` lines 79-98
- **Label:** `tab:training_config`
- **Current:** SFT Stage vs CITA Stage (2 columns)
- **Required:** Add DPO, PPO, GRPO columns (5 total)
- **Data Source:** Need hyperparameters for PPO and GRPO training runs

####  ✅  3.3 Table 7 (Feature Comparison) - `4_cita_framework.tex` lines 117-133
- **Label:** `tab:feature_comparison`
- **Current:** DPO vs CITA
- **Required:** Add SFT, PPO, GRPO columns

####  ✅  3.4 Table 12 (Model Variants) - `6_experiments.tex` lines 30-48
- **Label:** `tab:model_variants`
- **Current:** 6 variants (SFT, DPO, CITA × NoInstruct/Instruct)
- **Required:** Add PPO_NI, PPO_I, GRPO_NI, GRPO_I (10 total)

---

### 🔴 4.0: TRAINING PLOTS: Add PPO & GRPO

**Current Plots (only SFT, DPO, CITA):**
- `figures/training/dpo_cita_accuracy.pdf`
- `figures/training/dpo_cita_margins.pdf`
- `figures/training/dpo_cita_loss.pdf`

**TensorBoard Logs Available:**
```
tensorboard_logs/
├── CITA_Instruct_Adaptive_trial_7
├── CITA_NoInstruct_20251116_015238
├── DPO_Instruct_20251116_035213
├── DPO_NoInstruct_20251115_234037
├── GRPO_Instruct_20251220_201022      ← NEW
├── GRPO_NoInstruct_20251220_082628    ← NEW
├── PPO_Instruct_20251220_225226       ← NEW
├── PPO_NoInstruct_20251220_062208     ← NEW
├── SFT_Instruct_20251115_223957
├── SFT_NoInstruct_20251115_212216
```

**Files to Modify:**
- `comparative_study/generate_training_plots.py` (lines 65-72)
  - Add PPO and GRPO to `EVENT_FILES` dict
  - Update `COLORS` and `LINESTYLES` dicts

**Decision Needed:**
- PPO/GRPO may have different metrics (not DPO-style margins)
- May need separate plots like SFT (loss, token_accuracy)
- **REQUIRES GPU SERVER** to verify TensorBoard log structure

---

###  Why do we need it??? isn't `sync_figures.py` taking care of it ?? >> 5.0 EVALUATION PLOTS: Add PPO & GRPO (Figures 5-16)

**CONFIRMED**: Current Overleaf figures have ONLY 3 trainers (SFT, DPO, CITA). Need to update with 5 trainers.

**Script:** `comparative_study/05_evaluation/generate_combined_plots.py`

**Current `METHODS` list (line 96):**
```python
METHODS = ['SFT', 'DPO', 'PPO', 'GRPO', 'CITA']  # Script already supports 5!
```

**Plots to Regenerate (on GPU server):**
- `evaluation/isd_comparison.pdf`
- `evaluation/truthfulqa_comparison.pdf`
- `evaluation/conditional_safety_comparison.pdf`
- `evaluation/length_control_comparison.pdf`
- `evaluation/aqi_comparison.pdf`
- `evaluation/combined_plots/heatmap.pdf`
- `evaluation/combined_plots/radar_area.pdf`

**REQUIRES:** GPU server to regenerate plots with PPO/GRPO data

---

###  ✅  5.0 SYNC FIGURES: Copy Latest from outputs/ to Overleaf_draft/

**Problem:** `Overleaf_draft/figures/evaluation/` has outdated plots (3 trainers only)

**Solution:** Create `sync_figures.py` script to copy fresh figures:

**Source → Destination Mapping:**
```
outputs/evaluation/AQI_Evaluation/           → Overleaf_draft/figures/evaluation/
outputs/evaluation/Conditional_Safety_Evaluation/ → Overleaf_draft/figures/evaluation/
outputs/evaluation/ISD_Evaluation_Embedding/ → Overleaf_draft/figures/evaluation/
outputs/evaluation/Length_Control_Evaluation/ → Overleaf_draft/figures/evaluation/
outputs/evaluation/TruthfulQA_Evaluation/    → Overleaf_draft/figures/evaluation/
outputs/evaluation/combined_plots/           → Overleaf_draft/figures/evaluation/combined_plots/
```

**Script:** `Overleaf_draft/src/sync_figures.py`

---

### 🔴  6.0: EVALUATION PLOTS: Increase Font Size & Bold Black Text

**File:** `comparative_study/05_evaluation/eval_utils/plotting.py`

**Current Font Sizes:**
| Element | Current | Target |
|---------|---------|--------|
| Axis labels | 14 | **16** |
| X-tick labels | 11-12 | **14 bold** |
| Y-tick labels | 11 | **12** |
| Bar annotations | 9-11 | **12 bold** |
| Legend | 10-11 | **12** |

**Changes:**
- Add `fontweight='bold'` to all text elements
- Set `color='black'` explicitly
- Increase font sizes by ~2pt

---

###  ✅  7.0: TABLE 3: Update Hardware Specification

**File:** `Overleaf_draft/2_methodology.tex` (lines 15-32)
**Label:** `tab:infrastructure`

**Current:**
```latex
Hardware & NVIDIA A100-40GB \\
```

**Required:**
```latex
Hardware & NVIDIA A100-80GB \\
```

---

### 8.0: FIX TABLE OVERFLOW (Column Wrap Issues)

####  ✅  8.1 Table 8 (10 Instruction Types) - `5_isd_dataset.tex` lines 67-89  
- **Label:** `tab:instruction_types`
- **Issue:** Two-column text overflow
- **Fix:** Use `tabularx` with `X` columns, or reduce font size, or split table

####  ✅  8.2 Table 11 (Following vs Alignment) - `5_isd_dataset.tex` lines 168-184
- **Label:** `tab:following_vs_alignment`
- **Issue:** Text overflow from left to right column
- **Fix:** Adjust column widths, use `p{width}` columns


---

## QUESTIONS FOR USER

1. **Weighted Average Weights:** Should all 5 evaluations have equal weight (1/5 each)?
#### Answer: Equal but normalized Weights

2. **PPO/GRPO Training Metrics:** Do PPO/GRPO tensorboard logs have same metrics as DPO (accuracy, margins, loss)? Or different ones like SFT (loss, token_accuracy)?
### Look at the events files of tensorboard

3. **Table 4 Hyperparameters:** What are the training hyperparameters for PPO and GRPO? (epochs, learning_rate, etc.)
#### Look at trainign scripts
```
python3 comparative_study/02b_PPO_Baseline/Llama3_BF16.py
python3 comparative_study/02c_GRPO_Baseline/Llama3_BF16.py
```

---

## KEY FILES REFERENCE

| Purpose | File Path |
|---------|-----------|
| Radar plot generation | `comparative_study/05_evaluation/eval_utils/plotting.py` |
| Combined plots script | `comparative_study/05_evaluation/generate_combined_plots.py` |
| Training plots | `comparative_study/generate_training_plots.py` |
| Sync figures | `Overleaf_draft/src/sync_figures.py` (to create) |
| Cleanup unused | `Overleaf_draft/src/cleanup_unused_figures.py` |
| Table 2 | `Overleaf_draft/1_introduction.tex` lines 116-134 |
| Table 3 | `Overleaf_draft/2_methodology.tex` lines 15-32 |
| Table 4 | `Overleaf_draft/2_methodology.tex` lines 79-98 |
| Table 7 | `Overleaf_draft/4_cita_framework.tex` lines 117-133 |
| Table 8 (overflow) | `Overleaf_draft/5_isd_dataset.tex` lines 67-89 |
| Table 11 (overflow) | `Overleaf_draft/5_isd_dataset.tex` lines 168-184 |
| Table 12 | `Overleaf_draft/6_experiments.tex` lines 30-48 |
| Results narrative | `Overleaf_draft/7_results.tex` |
