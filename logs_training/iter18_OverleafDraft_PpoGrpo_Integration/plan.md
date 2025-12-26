# Arxiv Draft Update Plan - CITA Paper

## Overview
Update Overleaf draft to include all 5 trainers (SFT, PPO, GRPO, DPO, CITA) and fix various issues before Arxiv submission.

---

## STATUS LEGEND
| Symbol | Meaning |
|--------|---------|
| 🔴 | Not started |
| 🟡 | Code done, awaiting validation |
| ✅ | Verified in PDF |

---

## TODO LIST - INCOMPLETE

### ✅ 1. RADAR PLOT: Pentagon Area → Average Radius

**File:** `plotting.py` | **Output:** `radar_area.pdf`

- Formula: `avg = Σ(rᵢ) / n` (replaces polygon area)
- Labels updated: "Pentagon Area" → "Average Radius"
#### execute : `python3 comparative_study/05_evaluation/generate_combined_plots.py` to generate updated radar plot
---

### 🔴 2. UPDATE NARRATIVE: "wins X/5" → Weighted Average

**Files:** `7_results.tex`, `8_conclusion.tex`

---

### ✅ 3. TRAINING PLOTS: Add GRPO (PPO skipped - empty logs)

**File:** `generate_training_plots.py`

- Added GRPO colors, event files, metrics
- PPO: ❌ TensorBoard logs empty
- **Generated:** `combined_eval_loss.png`, `grpo_loss.pdf`, etc.

---

### 🔴 4. REGENERATE EVAL PLOTS: Double Fontsize + Bold

**REQUIRES GPU SERVER** - Run each evaluation script to regenerate plots:

| Benchmark | Script |
|-----------|--------|
| ISD | `comparative_study/05_evaluation/isd/evaluation.py` |
| TruthfulQA | `comparative_study/05_evaluation/truthfulqa/evaluation.py` |
| Conditional Safety | `comparative_study/05_evaluation/conditional_safety/evaluation.py` |
| Length Control | `comparative_study/05_evaluation/length_control/evaluation.py` |
<!-- 4. Length Control (select option 3 for Max) -->
| AQI | `comparative_study/05_evaluation/AQI/evaluation.py` | 
<!-- (select option 2 for Full - NOT Max) -->

**Changes Applied (in `plotting.py`):**
- `plt.rcParams['font.weight'] = 'bold'`
- Font sizes ×2 (axis labels 14→28, ticks 11→22, etc.)
- Bar width 50% reduction
- X-axis rotation 45→90°

---

## TODO LIST - COMPLETED

### ✅ 5. UPDATE TABLES: All 5 Trainers

- Table 2 (Method Comparison) - `1_introduction.tex`
- Table 4 (Training Config) - `2_methodology.tex`
- Table 7 (Feature Comparison) - `4_cita_framework.tex`
- Table 12 (Model Variants) - `6_experiments.tex`

---

### ✅ 6. SYNC FIGURES: `sync_figures.py`

Script: `Overleaf_draft/src/sync_figures.py`

---

### ✅ 7. TABLE 3: Hardware A100-40GB → A100-80GB

---

### ✅ 8. TABLE OVERFLOW FIX

- Table 8 (Instruction Types)
- Table 11 (Following vs Alignment)

---

### ✅ 9. GLOBAL BOLD FONT

Added `plt.rcParams['font.weight'] = 'bold'` to 8 plotting scripts.

---

### ✅ 10. HP ABLATION PLOTS: Scale Factors

- `lambda_kl`: ×10⁻⁴ in xlabel
- `learning_rate`: ×10⁻⁶ in xlabel
- Reduced font sizes

---

## KEY FILES REFERENCE

| Purpose | File Path |
|---------|-----------|
| Radar plot | `comparative_study/05_evaluation/eval_utils/plotting.py` |
| Combined plots | `comparative_study/05_evaluation/generate_combined_plots.py` |
| Training plots | `comparative_study/generate_training_plots.py` |
| Sync figures | `Overleaf_draft/src/sync_figures.py` |
