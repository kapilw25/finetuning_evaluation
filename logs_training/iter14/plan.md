# Iter14: Full Evaluation on Max Samples

## Models
SFT_NoInstruct, SFT_Instruct, DPO_NoInstruct, DPO_Instruct, CITA_NoInstruct, CITA_Instruct

## Max Samples (fetched from HF)
| Eval | Max Samples |
|------|-------------|
| ISD | 300 prompts × 10 = 3,000 |
| TruthfulQA | 817 × 2 = 1,634 |
| Conditional Safety | 1,222 × 2 = 2,444 |
| Length Control | 805 × 2 = 1,610 |
| AQI | 20,439 total (7 axioms) |

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

# 5. AQI (select option 3 for Max)
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
**Select option 3 for each eval**
