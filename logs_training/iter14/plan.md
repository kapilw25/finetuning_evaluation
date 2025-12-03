# Iter14: Full Evaluation on Max Samples

## Models
SFT_NoInstruct, SFT_Instruct, DPO_NoInstruct, DPO_Instruct, CITA_NoInstruct, CITA_Instruct

## Sample Sizes & Duration

| Eval | Samples | Batch | Duration |
|------|---------|-------|----------|
| TruthfulQA | 1,634 | 8 | **3h 57m** |
| Conditional Safety | 2,444 | 8 | **6h 53m** |
| ISD | 3,000 | 8 | **16h 15m** |
| AQI | 2,800 | 4 | **10h 3m** |
| Length Control | 1,610 | 8 | **10h 24m** |

**Note**: AQI uses Full mode (2,800) instead of Max (20,439) to keep sample sizes proportional (~1,600-3,000 range) and avoid 72-hour bottleneck.

---

## TODO: Training Plots (White Background PDFs)

**Problem**: Current TensorBoard screenshots have black background and group 4 plots together.
**Solution**: Generate individual PDFs with white background using `generate_training_plots.py`

### Available TensorBoard Event Files
```
tensorboard_logs/
├── CITA_Instruct_20251118_031257
├── CITA_Instruct_Adaptive_trial_2
├── CITA_Instruct_Adaptive_trial_7
├── CITA_NoInstruct_20251116_015238
├── DPO_Instruct_20251116_035213
├── DPO_NoInstruct_20251115_234037
├── SFT_Instruct_20251115_223957
├── SFT_NoInstruct_20251115_212216
```

### Metrics to Extract (per event file)
- `eval/rewards/accuracies` → accuracy PDF
- `eval/loss` → loss PDF
- `eval/rewards/margins` → margins PDF

### Figures to Generate (3 metrics × 5 comparisons = 15 PDFs)

| Figure | Comparison | Event Files to Use |
|--------|------------|-------------------|
| Fig 2 | CITA_Instruct Best3Trials vs DPO_Instruct vs CITA_NoInstruct | CITA_Instruct_Adaptive_trial_2, CITA_Instruct_Adaptive_trial_7, CITA_Instruct_20251118_031257, DPO_Instruct_20251116_035213, CITA_NoInstruct_20251116_015238 |
| Fig 3 | DPO_NoInstruct vs CITA_NoInstruct | DPO_NoInstruct_20251115_234037, CITA_NoInstruct_20251116_015238 |
| Fig 4 | DPO_Instruct vs DPO_NoInstruct | DPO_Instruct_20251116_035213, DPO_NoInstruct_20251115_234037 |
| Fig 5 | SFT_NoInstruct vs SFT_Instruct | SFT_NoInstruct_20251115_212216, SFT_Instruct_20251115_223957 |
| Fig 6 | CITA_Instruct AllTrials vs CITA_NoInstruct | All CITA_Instruct_* trials, CITA_NoInstruct_20251116_015238 |

### Output Structure
```
Overleaf_draft/figures/training/pdf/
├── fig2_accuracy.pdf
├── fig2_loss.pdf
├── fig2_margins.pdf
├── fig3_accuracy.pdf
├── fig3_loss.pdf
├── fig3_margins.pdf
... (15 total)
```

### Script to Modify
`comparative_study/05_evaluation/generate_training_plots.py`
- Reads TensorBoard event files directly (not CSV export)
- Uses matplotlib to generate white-background PDFs
- Publication-quality fonts and styling

### Command
```bash
python comparative_study/05_evaluation/generate_training_plots.py
```

---

## Terminal Commands (run on A10 GPU)

```bash
# Activate environment
source venv_CITA/bin/activate

# 1. ISD (select option 3 for Max)
python comparative_study/05_evaluation/isd/evaluation_embedding.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct

# 2. TruthfulQA (select option 3 for Max)
python comparative_study/05_evaluation/truthfulqa/evaluation.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct

# 3. Conditional Safety (select option 3 for Max)
python comparative_study/05_evaluation/conditional_safety/evaluation.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct

# 4. Length Control (select option 3 for Max)
python comparative_study/05_evaluation/length_control/evaluation.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct

# 5. AQI (select option 2 for Full - NOT Max)
python comparative_study/05_evaluation/AQI/evaluation.py \
  --models SFT_NoInstruct SFT_Instruct DPO_NoInstruct DPO_Instruct CITA_NoInstruct CITA_Instruct \
  --batch_size 4
```

---

## Interactive Menu
Each script shows:
```
[1] Sanity
[2] Full
[3] Max Available (100% of dataset - fetches from HF)
```
**Select option 3 for ISD, TruthfulQA, Conditional Safety, Length Control**
**Select option 2 for AQI (Full mode = 2,800 samples)**
