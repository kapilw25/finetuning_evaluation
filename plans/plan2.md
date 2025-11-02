# Training Plan: Industry Standard (SFT → DPO → CITA)

## Goal: Demonstrate CITA > DPO > SFT > Base

**Pipeline**: Base → SFT → DPO → CITA (stacking, industry standard)

---

## Phase 3A: Full Training (Stacked Pipeline)

**HuggingFace Repositories:**
- SFT: `kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct`
- DPO: `kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct`
- CITA: `kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct`

### **Commands** (run sequentially):

```bash
# 1. SFT-NoInstruct (base → SFT: 1 epoch)
python3 -u comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full
# Pushes to: kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct

# 2. DPO-NoInstruct (SFT → DPO: 1 epoch)
python3 -u comparative_study/02a_DPO_Baseline/Llama3_BF16.py \
    --mode full \
    --base_model kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct
# Pushes to: kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct

# 3. CITA-Instruct (DPO → CITA: 1 epoch)
python3 -u comparative_study/03a_CITA_Baseline/Llama3_BF16.py \
    --mode full \
    --base_model kapilw25/llama3-8b-pku-dpo-sft-bf16 # it is DPO-Instruct-SFT-Instruct
# Pushes to: kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct
```

### **Check Training Success**:

```bash
# View logs
tail -n 50 logs/SFT_Baseline_training_<timestamp>.log
tail -n 50 logs/DPO_Baseline_training_<timestamp>.log
tail -n 50 logs/CITA_Baseline_training_<timestamp>.log

# Verify HF uploads
# https://huggingface.co/kapilw25
```

**Key metrics:**
- SFT: `eval_loss`
- DPO: `eval_rewards/margins`
- CITA: `eval_rewards/margins`

---

## Phase 4: Toxicity Evaluation

**Script**: `comparative_study/05_evaluation/llm_as_judge/toxicity.py`
**Dataset**: PKU-SafeRLHF both-unsafe prompts (3,684 samples)
**Judge**: Llama-3-70B via Fireworks AI

**Command**:
```bash
# Sanity check (50 samples, ~5 min)
python3 comparative_study/05_evaluation/llm_as_judge/toxicity.py --mode sanity

# Full evaluation (3,684 samples, ~60 min)
python3 comparative_study/05_evaluation/llm_as_judge/toxicity.py --mode full
```

**Models evaluated:**
1. Baseline (Unaligned): `meta-llama/Llama-3.1-8B`
2. SFT Baseline: `kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct`
3. DPO Baseline: `kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct`
4. CITA Baseline: `kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct`

**Outputs**:
- `comparative_study/05_evaluation/llm_as_judge/Toxicity_Evaluation_Results/`
  - Individual CSV files per model
  - Summary JSON with toxicity scores
  - Comparison plot (toxicity_comparison.png)
- `logs/toxicity_evaluation_training_<timestamp>.log`

**Metric**: Toxicity score (1-5 scale)
- 1 = Safe refusal
- 5 = Highly toxic

---

## Status

**Phase 3A Complete**: All 3 models trained & pushed to HF
- ✅ SFT: `kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct`
- ✅ DPO: `kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct`
- ⏳ CITA: (pending push from v0_pku_all_instructed)

⏳ **Phase 4**: Run toxicity evaluation once CITA is pushed
