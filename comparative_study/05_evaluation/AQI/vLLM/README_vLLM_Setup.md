# vLLM AQI Evaluation Setup

## Overview

vLLM provides **24x faster inference** with **90%+ GPU utilization** compared to transformers (46% utilization).

**Problem:** vLLM cannot load LoRA adapters directly - it needs **full merged models**.

**Solution:** Merge LoRA adapters with base model, then use vLLM.

---

## Quick Start (Automated - Recommended)

```bash
# Activate environment
source venv_CITA/bin/activate

# Run vLLM evaluation (automatically merges models if needed)
python comparative_study/05_evaluation/AQI/run_aqi_vllm.py > logs/aqi_vllm.log 2>&1 &

# Monitor progress
tail -f logs/aqi_vllm.log
```

**What it does automatically:**
1. **Checks for merged models** at `outputs/merged_models_for_vllm/`
2. **If missing:** Runs merge script automatically (20-30 min one-time)
   - Downloads LoRA adapters from HuggingFace
   - Merges with base model (`meta-llama/Llama-3.1-8B`)
   - Saves to `outputs/merged_models_for_vllm/`
3. **Then runs vLLM evaluation** (5-10 min)
   - Processes 4 models sequentially
   - 90%+ GPU utilization per model
   - Generates AQI scores and visualizations

**Total time:**
- First run: ~25-40 min (merge + eval)
- Subsequent runs: ~5-10 min (eval only)

---

## Manual Steps (Optional)

### Step 1: Merge LoRA Adapters Manually (One-time, ~20-30 min)

If you want to merge models separately:

```bash
# Run merge script manually
python comparative_study/05_evaluation/AQI/merge_adapters_for_vllm.py > logs/merge_adapters.log 2>&1 &

# Monitor progress
tail -f logs/merge_adapters.log
```

### Step 2: Run vLLM AQI Evaluation (~5-10 min)

```bash
# Run vLLM evaluation (processes models SEQUENTIALLY, one at a time)
python comparative_study/05_evaluation/AQI/run_aqi_vllm.py > logs/aqi_vllm.log 2>&1 &

# Monitor progress
tail -f logs/aqi_vllm.log
```

**What it does:**
1. **Sequential Processing:** Loads ONE model at a time (not parallel)
   - Baseline (Unaligned) - uses base model directly
   - SFT Baseline - uses `outputs/merged_models_for_vllm/llama3-8b-sft-merged/`
   - DPO Baseline - uses `outputs/merged_models_for_vllm/llama3-8b-dpo-merged/`
   - CITA Baseline - uses `outputs/merged_models_for_vllm/llama3-8b-cita-merged/`

2. **For each model:**
   - Loads with vLLM (90% GPU utilization)
   - Generates 1400 responses (350 per model × 4 models)
   - Embeds responses (Response-AQI)
   - Calculates AQI scores
   - Saves results to `comparative_study/05_evaluation/AQI/AQI_Evaluation_Results/`
   - **Cleanups GPU memory** before loading next model

3. **Creates comparison:**
   - Overall AQI ranking
   - Per-axiom comparison
   - Visualizations (bar charts, heatmaps)

---

## Performance Comparison

### Original (transformers):
- **GPU utilization:** 46%
- **Memory:** 17.3GB / 40GB
- **Batch size:** 4 prompts
- **Time:** ~2 hours for 4 models
- **Processing:** Sequential (batch_size=4)

### vLLM (optimized):
- **GPU utilization:** 90%+
- **Memory:** ~36GB / 40GB (90% configured)
- **Batch size:** 1400 prompts (continuous batching)
- **Time:** ~5-10 minutes for 4 models
- **Processing:** Sequential models, batch inference within each
- **Speedup:** **24x faster**

---

## File Structure

```
comparative_study/05_evaluation/AQI/
├── merge_adapters_for_vllm.py    # Step 1: Merge LoRA adapters
├── run_aqi_vllm.py                # Step 2: Run vLLM evaluation
├── run_full_aqi_evaluation.py     # Original transformers version
└── AQI_Evaluation_Results/        # Output directory
    ├── 00_baseline_results/
    ├── 01a_sft_baseline_results/
    ├── 02a_dpo_baseline_results/
    ├── 03a_cita_baseline_results/
    ├── All_Models_AQI_Comparison.csv
    ├── Overall_AQI_Comparison.png
    ├── Per_Axiom_AQI_Heatmap.png
    └── Per_Axiom_AQI_Grouped_Bars.png

outputs/merged_models_for_vllm/    # Merged models (created by Step 1)
├── llama3-8b-sft-merged/          # ~16GB
├── llama3-8b-dpo-merged/          # ~16GB
└── llama3-8b-cita-merged/         # ~16GB
```

---

## Troubleshooting

### Error: "Merged model not found"
```
Run: python comparative_study/05_evaluation/AQI/merge_adapters_for_vllm.py
```

### Error: "CUDA out of memory"
- vLLM processes ONE model at a time (sequential)
- Each model uses ~36GB GPU memory (90% of 40GB)
- If OOM, reduce `gpu_memory_utilization=0.9` to `0.8` in `run_aqi_vllm.py:300`

### Check disk space before merging
```bash
df -h outputs/
# Need ~48GB free space for 3 merged models
```

---

## Key Differences: vLLM vs Transformers

| Feature | Transformers | vLLM |
|---------|-------------|------|
| **Input** | LoRA adapters (HF repos) | Merged full models (local) |
| **Batch size** | 4 prompts | 1400 prompts (continuous batching) |
| **GPU util** | 46% | 90%+ |
| **Speed** | 1x (baseline) | 24x |
| **Model loading** | Base + adapter merge at runtime | Pre-merged models |
| **Disk usage** | Minimal (adapters ~100MB) | ~48GB (merged models) |
| **Setup time** | 0 min | 20-30 min (one-time merge) |

---

## Summary

1. **Merge adapters once** (20-30 min, 48GB disk): `merge_adapters_for_vllm.py`
2. **Run vLLM evaluation** (5-10 min): `run_aqi_vllm.py`
3. **Get 24x speedup** with 90%+ GPU utilization

Models are processed **sequentially** (one at a time) to avoid OOM on A100-40GB.
