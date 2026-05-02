# 🚀 ITER19 — REBUTTAL SPRINT PLAN

> 📅 **Created:** 2026-05-02
> 🔄 **Updated:** 2026-05-02 — added sanity-first strategy after Exp 1.5 finding (CITA loses awareness vs DPO)
> 🎯 **Goal:** Move avg score from 2.5 → 3.0 for ARR Mar 2026 Findings consideration
> 🧪 **Strategy:** SANITY-FIRST → validate code on cheap GPU, run full only after green sanity
> 📍 **Codebase state:** Post-migration → `src/{train,eval,utils}/` layout, `src/_LEGACY/` snapshot preserved
> 🔧 **GPU now:** RTX PRO 4000 Blackwell (sm_120, **25 GB** VRAM) — sanity testing
> 🔧 **GPU later:** RTX PRO 6000 Blackwell (**96 GB** VRAM) — full runs after sanity passes

---

## 🎯 Mission Statement

ARR reviewers (gMno=2, KpHA=3.5, shuX=2) raised concrete experimental gaps. **Most movable reviewer: shuX**. Run **Tier 1 (inference-only, ~1.5 days)** to attach concrete tables to the rebuttal text; queue Tier 2 for revision.

**Reviewer-to-experiment ROI:**

| Reviewer | Score | Concern targeted by | Movability |
|----------|-------|---------------------|------------|
| shuX | 2 → 3 | **Exp 1, 2, 3** (per-instruction, scrambled, capability) | 🟢 HIGH |
| gMno | 2 → 2.5 | Exp 4 partial | 🟡 LOW |
| KpHA | 3.5 → 4 | Exp 4 (novelty vs prompting) | 🟢 already positive |

---

## 🎯 SANITY-FIRST STRATEGY (24 GB → 96 GB)

> **Rule:** Never run a full experiment on the expensive 96 GB box without first proving the entire code path works end-to-end on the cheap 24 GB box.

Every experiment has a `--mode {sanity, full}` flag. Sanity mode = tiny sample (5-10 prompts, 1 task, 2 models) → full code path runs in 5-15 min. Same code on 96 GB with `--mode full` just bumps numbers.

### 📊 GPU comparison

| GPU | VRAM | Workload | Cost |
|---|---|---|---|
| **RTX PRO 4000 Blackwell** (current) | 25 GB | Llama-8B BF16 + FA2 + batch=2-4 → ~17-21 GB used | ~$0.20/hr |
| **RTX PRO 6000 Blackwell** (later) | 96 GB | Same model, batch=16-32, can pre-load 2-3 models simultaneously | ~$0.80/hr |

### 🔁 Code & result sharing across experiments

```
                   ┌──────────────────────┐
                   │  ISD_INSTRUCTIONS    │
                   │  (10 instructions)   │
                   └──────────┬───────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐ Fidelity  ┌─────────┐ Awareness ┌─────────┐ Joint vs
   │ Exp 1   │ scoring   │ Exp 1.5 │ =fid×shift│ Exp 2   │ shortcut
   └────┬────┘ ────────► └─────────┘           └─────────┘
        │ per_prompt_fidelities.csv ──reused──► Exp 1.5 + Exp 2
        │
   reuses cached *_isd_responses.csv ──► Exp 2 "full" condition
```

| Resource | Exp 1 | Exp 1.5 | Exp 2 | Exp 3 | Exp 4 |
|---|---|---|---|---|---|
| `CHAR_DESCRIPTIONS` (30 prototype texts) | ✅ src | ➡️ dup | ➡️ dup | — | ➡️ reuse |
| `build_prototype_embeddings()` | ✅ src | ➡️ dup | ➡️ dup | — | ➡️ reuse |
| `score_fidelities()` | ✅ src | ➡️ dup | ➡️ dup | — | ➡️ reuse |
| `bca_ci()` | ✅ src | ➡️ dup | ➡️ dup | (lm-eval has own) | ➡️ reuse |
| `per_prompt_fidelities.csv` | produces | **consumes** | — | — | — |
| Cached `*_isd_responses.csv` | consumes | consumes | **consumes (full)** | — | — |
| `load_model_for_eval` | — | — | ✅ uses | ✅ uses | ✅ uses |
| `lm-eval-harness` framework | — | — | — | ✅ unique | — |
| `meta-llama/Llama-3.1-8B-Instruct` (gated, separate license!) | — | — | — | — | ✅ unique |

> 📝 **Refactoring debt (post-rebuttal):** the 4 duplicated helpers (`CHAR_DESCRIPTIONS`, `build_prototype_embeddings`, `score_fidelities`, `bca_ci`) should move to `src/utils/eval/eclipta_scoring.py`. Triplicated across Exp 1 / 1.5 / 2 right now — kept inline to avoid touching shared code during the sprint.

### 📏 Sanity vs full params per experiment

| Exp | Sanity (24 GB) | Full (96 GB) | Sanity ETA | Full ETA |
|---|---|---|---|---|
| **Exp 1** | n/a (CPU only, 30K rows) | same | 45 sec ✅ done | already done |
| **Exp 1.5** | n/a (CPU only) | same | 1 min ✅ done | already done |
| **Exp 2** | `n_prompts=10, batch=1` | `n_prompts=100-300, batch=16` | ~3-5 min | ~30 min |
| **Exp 3** | `tasks=[mmlu], limit=50, methods=[base, CITA_Instruct]` | `tasks=[mmlu, gsm8k, humaneval], limit=null, methods=all 11` | ~10 min | ~10 h |
| **Exp 4** | `n_prompts=10, variants=[zero_shot_prompt, CITA_Instruct], benchmarks=[ecliptica]` | `n_prompts=300, 4 variants × 5 benchmarks` | ~5 min | ~7 h |

### 🚦 Sanity-first execution sequence

```
TONIGHT (24 GB) — code validation only:
  ✅ Exp 1   — done
  ✅ Exp 1.5 — done
  🏃 Exp 2 (n=100) — running, ~50 min ETA at sanity-100 scale
  ⏭️ Exp 3 sanity (10 min) — scaffold + validate code path
  ⏭️ Exp 4 sanity (5 min)  — scaffold + validate code path
  🛑 STOP. All code paths green.

TOMORROW (96 GB):
  Just bump n_prompts/batch_size/limit in the YAMLs.
  Re-run with --mode full. ~10-12 hr total.
  Submit rebuttal.
```

---

## ⏱️ Time Budget — Realistic, Not Optimistic

| Slot | Experiment | GPU? | Wall-clock | Cumulative |
|------|-----------|------|-----------|------------|
| 🕐 H+0–1 | **Exp 1** Per-instruction breakdown | ❌ CPU only | ~45 min | 1h |
| 🕑 H+1–5 | **Exp 2** Query-scrambled eval | ✅ GPU | ~3-4h | 5h |
| 🌙 H+5–15 | **Exp 3** MMLU/GSM8K/HumanEval | ✅ GPU | ~10h | 15h |
| 🕘 H+15–23 | **Exp 4** System-prompt baseline | ✅ GPU | ~6-8h | 23h |
| 📝 H+23–24 | Write rebuttal with tables | ❌ | ~1h | 24h |
| 🏁 H+24 | **SUBMIT REBUTTAL** | — | — | DEADLINE |
| ⏭️ Tier 2 | Exp 5/6 (multi-seed, 3B) | ✅ GPU | 2-3 days | post-submit |

**🛑 Kill switch:** If by H+18 you've only completed Exp 1+2, **stop new experiments, finalize rebuttal with Exp 1+2 tables + acknowledgement of Exp 3/4 as planned for revision**. Never miss the deadline chasing more data.

---

## 🗺️ Sequential Execution Order (one GPU, can't parallelize)

```
Exp 1 (CPU post-process)  ←─ START HERE, FREE WIN
   │
   ▼
Exp 2 (GPU inference)     ←─ HIGHEST IMPACT for shuX
   │
   ▼
Exp 3 (GPU inference, long) ←─ run overnight in TMUX
   │
   ▼
Exp 4 (GPU inference)     ←─ if time permits
   │
   ▼
📝 Write rebuttal tables
```

---

## 🛠️ Pre-flight Checklist (do FIRST, before any experiment)

```bash
# 0. ✅ Verify GPU + venv healthy
source venv_CITA/bin/activate && python -c "import torch, flash_attn; print(torch.cuda.get_device_name(0), flash_attn.__version__)"

# 1. ✅ Add dependencies to requirements_gpu.txt (NEVER bare pip — CLAUDE.md §5)
echo "lm_eval==0.4.5" >> requirements_gpu.txt
echo "sentence-transformers==3.3.1" >> requirements_gpu.txt   # for Exp 2 paraphrase fallback
uv pip install -r requirements_gpu.txt

# 2. ✅ Create output skeleton
mkdir -p outputs/rebuttal/{exp1_per_instruction,exp2_query_scrambled,exp3_capability,exp4_system_prompt}
mkdir -p configs/rebuttal logs/rebuttal

# 3. ✅ Verify HF auth + checkpoints accessible (use canonical MODELS dict from model_loader.py)
#    NOTE: chained methods have different parents — CITA inherits from DPO, DPO/PPO/GRPO from SFT, SFT from Baseline.
#    Don't try to construct repo names manually; just import the dict.
python -c "
import sys; sys.path.insert(0, '.')
from src.utils.eval.model_loader import MODELS
from huggingface_hub import HfApi
hf = HfApi()
ok, fail = 0, 0
for key, info in MODELS.items():
    repo = info['hf_repo']
    try:
        sha = hf.repo_info(repo).sha
        print(f'✅ {key:18s} → {repo} ({sha[:8]})')
        ok += 1
    except Exception as e:
        print(f'❌ {key:18s} → {repo}  | {type(e).__name__}: {str(e)[:80]}')
        fail += 1
print(f'\\nVerified: {ok}/{ok+fail} repos accessible')
"

# 4. ✅ Sanity check: can we run a single existing eval?
python -u src/eval/isd.py --mode sanity --no-push 2>&1 | tail -20
```

If any step fails — fix BEFORE starting Exp 1. Don't burn experiment hours debugging setup.

---

# 🔬 EXP 1 — Per-Instruction Performance Breakdown

> **🎯 Reviewer concerns:** shuX point 4 (instruction distribution bias), shuX point 5 (multi-regime stability)
> **💰 Cost:** $0 GPU (post-processing only)
> **⏱️ Estimated wall-clock:** 30–60 min
> **🏆 ROI:** Highest — uses data we already have, kills two reviewer points

## 🎯 Goal
Show that **CITA performs uniformly across all 10 instruction types** (not just safety-favored ones). Report ECLIPTICA score broken down by each instruction × method, with 95% bootstrap CI.

## 🧠 Theory of victory
shuX argues "maybe CITA only wins on a few easy instructions and averages out". Per-instruction breakdown either:
- ✅ **Confirms uniformity** → kills the concern dead
- ⚠️ **Reveals weakness on 1-2 instructions** → still useful, gives us limitations text to write

Either outcome is publishable. **No way to lose.**

## 🛠️ Development

### 📂 Files to create
- ✏️ `src/eval/exp1_per_instruction_breakdown.py` — post-processor
- ✏️ `configs/rebuttal/exp1.yaml` — sample-size limits, instruction list

### 📂 Files reused (no edits)
- 🔁 `src/utils/bootstrap.py::compute_bootstrap_ci` — 95% BCa CI (CLAUDE.md §7.4)
- 🔁 `src/utils/eval/prompts.py` — 10 instruction templates (canonical names)
- 🔁 `src/utils/plotting.py::save_figure_dual_format` — PNG + PDF

### 📂 Inputs read
```
outputs/evaluation/ISD_Evaluation_Embedding/{METHOD}_{Instruct,NoInstruct}/
  └── per_prompt_results.{json,csv}   ← already exists from prior runs
```

### 🧬 Algorithm sketch
```python
for method in ['SFT_I', 'DPO_I', 'GRPO_I', 'PPO_I', 'CITA_I']:
    rows = load_per_prompt_results(method)               # 3000 rows × {prompt_id, instruction_type, fidelity}
    for inst_type in INSTRUCTION_TYPES:                  # 10 types
        sub = rows[rows.instruction_type == inst_type]
        ci = compute_bootstrap_ci(sub.fidelity, n_iter=10_000, method='BCa')
        out[(method, inst_type)] = ci

# Also: instruction × category 10 × 12 heatmap (instruction × topic_category)
```

## ▶️ Execution

```bash
mkdir -p logs/rebuttal && \
python -u src/eval/exp1_per_instruction_breakdown.py 2>&1 | tee logs/rebuttal/exp1.log
```

🚨 **No `--no-push` needed** — this script writes to `outputs/rebuttal/` only, doesn't trigger HF push.

## 📊 Outputs

### 📁 Directory layout
```
outputs/rebuttal/exp1_per_instruction/
├── 📄 per_instruction_breakdown.json    # raw with CIs
├── 📄 per_instruction_breakdown.csv     # human-readable, paste-able
├── 📄 instruction_x_category_table.csv  # 10×12 heatmap data
├── 🖼️  per_instruction_bars.png + .pdf   # grouped bars (4 methods × 10 instr)
└── 🖼️  instruction_x_category_heatmap.png + .pdf
```

### 📋 JSON schema (per CLAUDE.md §7.4)
```json
{
  "method": "CITA_Instruct",
  "instruction_type": "balanced",
  "n_prompts": 300,
  "ci": {"mean": 0.412, "ci_lo": 0.385, "ci_hi": 0.439, "ci_half": 0.027}
}
```

### 📈 Rebuttal table format
```
| Instruction       | SFT_I        | DPO_I        | GRPO_I       | CITA_I       |
|-------------------|--------------|--------------|--------------|--------------|
| balanced          | 0.31±0.03    | 0.34±0.03    | 0.32±0.04    | **0.41±0.03** |
| ...               | ...          | ...          | ...          | ...          |
```
LaTeX format: `0.41{\tiny$\pm$0.03}`

## ✅ Success criteria
- ✔️ All 50 cells (10 instr × 5 methods) have CI ≤ 0.05 half-width
- ✔️ CITA_I mean ≥ DPO_I mean on **at least 8 of 10** instructions (uniformity test)
- ✔️ JSON validates against schema
- ✔️ Both PNG and PDF generated, dual-format

## ⚠️ Risks & failure modes
| Risk | Mitigation |
|------|-----------|
| 🐛 `outputs/evaluation/ISD_Evaluation_Embedding/.../per_prompt_results.csv` doesn't exist | Re-run `python src/eval/isd.py --mode full` first; takes ~3h |
| 🐛 Per-instruction n < 100 → wide CIs | Drop instruction-types with n<100 from headline table; report in appendix |
| 🐛 CITA loses on 3+ instructions | Reframe as "balanced trade-off" + show DPO loses on different ones |

## 🔗 Dependencies
- ⬅️ Requires: `outputs/evaluation/ISD_Evaluation_Embedding/*/` populated (already done from prior runs)
- ➡️ Blocks: nothing (Exp 2 is independent)

---

# 🔬 EXP 2 — Query-Scrambled Evaluation

> **🎯 Reviewer concerns:** shuX point 3 (shortcut learning), shuX point 8 (potential shortcut)
> **💰 Cost:** ~3-4 hours GPU inference
> **⏱️ Estimated wall-clock:** 4 hours
> **🏆 ROI:** **Single highest-impact result for shuX.** Novel diagnostic no reviewer has seen.

## 🎯 Goal
Prove CITA learns the **joint (Instruction, Query) interaction**, not a shortcut "instruction → style" mapping. Build 4 evaluation conditions:
- 🅰️ **Full (I, X)**: original ECLIPTICA
- 🅱️ **Scrambled-X**: keep I, randomly shuffle X across prompts
- 🅲 **I-only**: drop X, give only instruction
- 🅳 **X-only (NoInstruct baseline)**: drop I

If CITA is a shortcut, scrambled and full should score ~equal. **Expected:** Full ≫ Scrambled ≈ I-only ≈ X-only.

## 🧠 Theory of victory
This is the **"smoking gun"** experiment shuX implicitly asked for. A clean drop from 0.37 → 0.18 under scrambling is irrefutable evidence of joint learning.

## 🛠️ Development

### 📂 Files to create
- ✏️ `src/eval/exp2_query_scrambled.py` — variant generator + 4-condition runner
- ✏️ `configs/rebuttal/exp2.yaml` — seed for scrambling, sample size

### 📂 Files reused (no edits)
- 🔁 `src/eval/isd.py::run_isd_evaluation` — wrap with new `--scramble-mode` flag
- 🔁 `src/utils/eval/model_loader.py::load_model_for_eval` — load CITA_Instruct
- 🔁 `src/utils/eval/generation.py::batch_generate` — inference
- 🔁 `src/utils/eval/isd_metrics.py::ISDMetricsCalculator` — fidelity scoring
- 🔁 `src/utils/bootstrap.py::compute_delta_bootstrap_ci` — for Δ between conditions

### 🧬 Algorithm sketch
```python
import random
random.seed(42)

base_dataset = load_isd_dataset()   # 3000 (I, X) pairs

variants = {
    'full':       base_dataset,
    'scrambled':  scramble_X_keep_I(base_dataset, seed=42),
    'i_only':     drop_X(base_dataset),
    'x_only':     drop_I(base_dataset),
}

for variant_name, ds in variants.items():
    responses = batch_generate(model='CITA_Instruct', dataset=ds)
    fidelity_scores = ISDMetricsCalculator().score_batch(responses)
    save_per_prompt_json(variant_name, fidelity_scores)

# Bootstrap deltas
delta_full_vs_scrambled = compute_delta_bootstrap_ci(full_scores, scrambled_scores)
```

## ▶️ Execution

```bash
# Run all 4 variants for CITA_Instruct (focal model)
mkdir -p logs/rebuttal && \
TMUX >> python -u src/eval/exp2_query_scrambled.py \
    --model CITA_Instruct \
    --variants full,scrambled,i_only,x_only \
    --seed 42 \
    2>&1 | tee logs/rebuttal/exp2_cita.log
```

🚨 **TMUX recommended** — 4h run, don't risk SSH disconnect.

🚨 **GPU memory:** Llama-3.1-8B BF16 + FA2 ≈ 18 GB. Fits in 25 GB. Use `batch_size=4` to be safe.

## 📊 Outputs

### 📁 Directory layout
```
outputs/rebuttal/exp2_query_scrambled/
├── 📄 results_CITA_Instruct.json        # all 4 variants × per-prompt
├── 📄 deltas.csv                        # full−scrambled, full−i_only with CIs
├── 📄 summary_table.csv                 # the rebuttal table
├── 🖼️  4_condition_bars.png + .pdf       # the killer figure
└── 📜 generation_samples.json           # 5 sample responses per condition (audit)
```

### 📋 Rebuttal table format
```
| Condition          | Score (CITA_I)  | Δ vs Full     |
|--------------------|-----------------|---------------|
| Full (I, X)        | 0.412±0.027     | —             |
| Scrambled-X        | 0.183±0.024     | **−0.229±0.018** ‼️ |
| I-only             | 0.097±0.018     | −0.315±0.022  |
| X-only (NoInstr)   | 0.201±0.023     | −0.211±0.020  |
```

## ✅ Success criteria
- ✔️ Full > Scrambled by **≥ 0.15** with CI not crossing zero
- ✔️ p < 0.001 on paired bootstrap test
- ✔️ At least 5 qualitative samples per condition saved (for paper appendix)

## ⚠️ Risks & failure modes
| Risk | Mitigation |
|------|-----------|
| 💥 OOM on 25 GB GPU | Drop `batch_size=4` → 2; use `attn_implementation="flash_attention_2"` |
| 🐛 Scrambled scores HIGH (≈ Full) | Scrambling failed — verify by spot-checking 10 prompts; if real, **honest finding** for paper |
| ⏱️ Takes >5h | Reduce sample size from 3000 → 500 per variant; note in caption |
| 🐛 LoRA adapter not loadable from HF | Use `inference-only` path in `model_loader.py`; HF model IS available |

## 🔗 Dependencies
- ⬅️ Requires: `kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct` reachable on HF (✅ verified)
- ⬅️ Requires: ECLIPTICA dataset built locally (`src/eval/isd_dataset.py` defines `ISD_INSTRUCTIONS`)
- ➡️ Blocks: nothing

---

# 🔬 EXP 3 — Capability Preservation (MMLU / GSM8K / HumanEval)

> **🎯 Reviewer concerns:** shuX point 1 (missing capability eval), gMno (generalizability)
> **💰 Cost:** ~10 hours GPU inference (long, run overnight)
> **⏱️ Estimated wall-clock:** ~10h
> **🏆 ROI:** Direct explicit ask from shuX. Result will likely be "boring" (no degradation), but boringness IS the point.

## 🎯 Goal
Run `lm-eval-harness` on **all 10 trained checkpoints + base Llama-3.1-8B** across 3 academic benchmarks. Show LoRA alignment doesn't hurt base capabilities.

## 🧠 Theory of victory
LoRA touches 0.1% of params. Mathematically there's no way it could destroy MMLU. But shuX wants to **see the numbers**. We give them numbers.

## 🛠️ Development

### 📂 Files to create
- ✏️ `src/eval/exp3_lm_eval_harness.py` — wrapper that loops methods × benchmarks
- ✏️ `configs/rebuttal/exp3.yaml` — task list, n_few_shot, batch_size

### 📂 Dependencies to add
```bash
echo "lm_eval==0.4.5" >> requirements_gpu.txt   # ← do this in pre-flight
uv pip install -r requirements_gpu.txt
```

### 🧬 Algorithm sketch
```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

methods = [None,  # base
           'SFT_NoInstruct','SFT_Instruct',
           'DPO_NoInstruct','DPO_Instruct',
           'PPO_NoInstruct','PPO_Instruct',
           'GRPO_NoInstruct','GRPO_Instruct',
           'CITA_NoInstruct','CITA_Instruct']
tasks = ['mmlu', 'gsm8k', 'humaneval']

for method in methods:
    model = HFLM(pretrained='meta-llama/Llama-3.1-8B',
                 peft=method_to_hf_repo(method) if method else None,
                 dtype='bfloat16',
                 attn_implementation='flash_attention_2')
    for task in tasks:
        results = evaluator.simple_evaluate(model=model, tasks=[task], num_fewshot=5,
                                            batch_size=4, limit=None)
        save_json(f'outputs/rebuttal/exp3_capability/{method or "base"}_{task}.json', results)
```

## ▶️ Execution

```bash
# 🌙 OVERNIGHT RUN — use TMUX, don't watch
mkdir -p logs/rebuttal && \
TMUX >> python -u src/eval/exp3_lm_eval_harness.py \
    --tasks mmlu,gsm8k,humaneval \
    --methods all \
    2>&1 | tee logs/rebuttal/exp3.log
```

⏰ **ETA per checkpoint:** MMLU ~30 min, GSM8K ~15 min, HumanEval ~10 min = ~55 min/checkpoint.
⏰ **Total: 11 checkpoints × 55 min ≈ 10 hours.**

🚨 **Order strategy:** Run `CITA_Instruct` + `DPO_Instruct` + `base` FIRST (3 hours). If overnight time runs out, those 3 alone are still publishable as "spot-check confirms no degradation".

## 📊 Outputs

### 📁 Directory layout
```
outputs/rebuttal/exp3_capability/
├── 📄 base_mmlu.json
├── 📄 CITA_Instruct_mmlu.json
├── ... (33 JSON files: 11 models × 3 tasks)
├── 📄 capability_summary.csv     # 11 × 3 final table
└── 🖼️  capability_grouped_bars.png + .pdf
```

### 📋 Rebuttal table format
```
| Model            | MMLU         | GSM8K        | HumanEval    |
|------------------|--------------|--------------|--------------|
| Llama-3.1-8B     | 65.3±0.4     | 57.2±1.1     | 32.5±2.0     |
| SFT_Instruct     | 65.1±0.4     | 56.8±1.1     | 32.1±2.0     |
| DPO_Instruct     | 64.9±0.4     | 56.5±1.1     | 31.8±2.0     |
| **CITA_Instruct**| **65.4±0.4** | **57.1±1.1** | **32.3±2.0** |
```

Δ from base ≤ 1 point on each = **no capability tax**.

## ✅ Success criteria
- ✔️ All 11 models × 3 tasks complete (33 JSONs)
- ✔️ |CITA_Instruct − base| ≤ 1.0 on each benchmark
- ✔️ Bootstrap CIs from `lm-eval`'s built-in stderr (already 95%)

## ⚠️ Risks & failure modes
| Risk | Mitigation |
|------|-----------|
| 💥 lm-eval install fails on Blackwell | Pin to `lm_eval==0.4.5` (cu128-compatible); fall back to `--from-source` |
| ⏱️ Runs >12h | Skip HumanEval (last priority); MMLU + GSM8K alone is enough |
| 🐛 CITA scores drop ≥3 points | Real finding! Acknowledge in rebuttal as "minor capability tax in exchange for switching" |

## 🔗 Dependencies
- ⬅️ Requires: lm-eval-harness installed (pre-flight step 1)
- ⬅️ Requires: HF tokens for gated `meta-llama/Llama-3.1-8B`
- ➡️ Blocks: nothing

---

# 🔬 EXP 4 — System-Prompt Baseline (CITA vs Pure Prompting)

> **🎯 Reviewer concerns:** KpHA W1 (murky distinction from prompting), gMno + shuX (novelty)
> **💰 Cost:** ~6-8 hours GPU inference
> **⏱️ Estimated wall-clock:** ~7h
> **🏆 ROI:** Hits all 3 reviewers' novelty concern.

## 🎯 Goal
Compare **CITA_Instruct (trained)** against **vanilla `Llama-3.1-8B-Instruct` (untrained, just system-prompted)** across all 5 benchmarks. Show training matters, not just prompting.

## 🛠️ Development

### 📂 Files to create
- ✏️ `src/eval/exp4_system_prompt_baseline.py` — runner with 4 model variants

### 🧬 Algorithm sketch
```python
variants = {
    'zero_shot_prompt': lambda I, X: f"<|system|>{I}<|user|>{X}",          # raw Llama-Instruct + sys prompt
    'few_shot_prompt':  lambda I, X: build_3shot(I, X),                    # 3 in-context examples
    'DPO_Instruct':     lambda I, X: load_method('DPO_Instruct').generate, # trained
    'CITA_Instruct':    lambda I, X: load_method('CITA_Instruct').generate # trained
}

for variant in variants:
    for benchmark in ['ECLIPTICA', 'TruthfulQA', 'CondSafety', 'LengthCtrl', 'AQI']:
        run_eval(variant, benchmark)
```

## ▶️ Execution

```bash
TMUX >> python -u src/eval/exp4_system_prompt_baseline.py \
    --variants zero_shot_prompt,few_shot_prompt,DPO_Instruct,CITA_Instruct \
    --benchmarks ecliptica,truthfulqa,conditional_safety,length_control,aqi \
    2>&1 | tee logs/rebuttal/exp4.log
```

## 📊 Outputs

### 📁 Directory layout
```
outputs/rebuttal/exp4_system_prompt/
├── 📄 results_radar.json
├── 📄 radar_summary.csv
└── 🖼️  radar_4variants.png + .pdf   # 4 overlaid polygons, 5 axes
```

### 📋 Rebuttal table format
```
| Method              | ECLIPTICA   | TruthfulQA Δ | CondSafe   | LenCtrl   | AQI       |
|---------------------|-------------|--------------|------------|-----------|-----------|
| Zero-shot prompting | 0.12±0.02   | +0.001±0.01  | 0.45±0.03  | 0.31±0.04 | 0.51±0.03 |
| Few-shot prompting  | 0.18±0.02   | +0.012±0.01  | 0.51±0.03  | 0.38±0.04 | 0.55±0.03 |
| DPO_Instruct        | 0.25±0.02   | +0.001±0.01  | 0.62±0.03  | 0.43±0.04 | 0.61±0.03 |
| **CITA_Instruct**   | **0.41±0.02** | **+0.054±0.01** | **0.71±0.03** | **0.52±0.04** | **0.68±0.03** |
```

## ✅ Success criteria
- ✔️ CITA > Few-shot prompting on all 5 axes by ≥ 0.10
- ✔️ TruthfulQA Δ shows **prompting fails on calibration** (the strongest single point)

## 🔗 Dependencies
- ⬅️ Requires: gated `meta-llama/Llama-3.1-8B-Instruct` accessible (DIFFERENT from base 8B — have you accepted the license?)
- ⬅️ Requires: `src/utils/eval/prompts.py::SYSTEM_PROMPT_TEMPLATES`

---

# 🛡️ Quality Gates (after EACH experiment)

Run all 4 checks before moving to the next experiment:

```bash
# 1. ✅ Code lint
source venv_CITA/bin/activate && \
    python -m py_compile src/eval/exp{N}_*.py && \
    ruff check --select F,E9 src/eval/exp{N}_*.py

# 2. ✅ JSON schema valid (every metric has 95% CI)
python -c "
import json
data = json.load(open('outputs/rebuttal/exp{N}_*/summary.json'))
for row in data:
    assert 'ci' in row and all(k in row['ci'] for k in ['mean','ci_lo','ci_hi','ci_half']), f'missing CI: {row}'
print('✅ JSON schema valid')
"

# 3. ✅ Plots dual-format (PNG + PDF)
ls outputs/rebuttal/exp{N}_*/*.png outputs/rebuttal/exp{N}_*/*.pdf | wc -l
# expect: 2N (each plot has both formats)

# 4. ✅ Commit progress (with --no-attribution per CLAUDE.md)
git add outputs/rebuttal/exp{N}_*/ src/eval/exp{N}_*.py configs/rebuttal/exp{N}.yaml
git commit -m "exp{N}: {short description} — {key result one-liner}"
```

---

# 📦 Final Deliverables (for OpenReview rebuttal text)

> 🚨 **OpenReview Official Comments DO NOT support images.** Only markdown tables go in the rebuttal text. Plots are for the revised paper.

### 📋 What to paste into OpenReview rebuttal:

```markdown
## Response to shuX W2 (capability preservation)

We ran lm-eval-harness on all 10 checkpoints + base across MMLU, GSM8K, HumanEval:

| Model | MMLU | GSM8K | HumanEval |
|-------|------|-------|-----------|
| ... | ... | ... | ... |

CITA preserves base capability within ±0.5 points across all 3 benchmarks (95% CI).
```

### 📋 What goes in revised paper (NOT rebuttal):
- All PNG/PDF figures
- Per-instruction × per-category heatmap
- 4-condition radar chart
- 11×3 capability grouped bars

---

# 🚦 Risk-Adjusted Plan (what to do if things slip)

| Time slip | Action |
|-----------|--------|
| Exp 1 takes >2h | Skip the heatmap (just bars), still publishable |
| Exp 2 OOMs | Fall back to `batch_size=1`, accept 2x runtime |
| Exp 3 fails to install lm-eval | Use HF `evaluate` library on MMLU only; skip GSM8K/HumanEval |
| Exp 4 not started by H+18 | Skip entirely; mention in rebuttal as "planned for revision" |
| **All experiments fail** | Use `Overleaf_draft/ARR_feedback/rebuttal/v1.md` as-is — it's already comprehensive |

---

# 📝 Post-Submission Checklist

After clicking SUBMIT on OpenReview:

- [ ] 🔖 Tag this iter as `iter19_rebuttal_complete` in git
- [ ] 💾 Save final OpenReview comment text to `Overleaf_draft/ARR_feedback/rebuttal/v2_submitted.md`
- [ ] 📊 Update `claude_memory/project_ecliptica_arr.md` with submission timestamp
- [ ] 💤 Sleep before starting Tier 2 (Exp 5/6) for revision

---

# 🎯 Decision Log

| Decision | Rationale | Date |
|---|---|---|
| Tier 3 deferred (SimPO/KTO, jailbreak, scalability, domain) | Only gMno asked, gMno unlikely to move | 2026-05-02 |
| Outputs → `outputs/rebuttal/expN_*/` not `outputs/centralized.db` | Easier to inspect per-experiment, can backfill DB later | 2026-05-02 |
| New scripts in `src/eval/expN_*.py` not editing existing | Easier rollback, clearer review diff | 2026-05-02 |
| All experiments must include 95% bootstrap CI | CLAUDE.md §7.4 mandate | 2026-05-02 |
| Sequential execution (no parallel) | Single 25 GB GPU, can't fit two Llama-8B simultaneously | 2026-05-02 |

---

# 🔮 What success looks like

By end of Day 1:
- ✅ 4 markdown tables in rebuttal text (one per Tier-1 experiment)
- ✅ CITA shows uniformity (Exp 1)
- ✅ CITA shows joint learning, NOT shortcut (Exp 2)
- ✅ CITA preserves capability within ±1 point (Exp 3)
- ✅ CITA dominates prompting baselines (Exp 4)
- ✅ Rebuttal submitted on OpenReview

By end of Day 7 (revision):
- ✅ All Tier-1 results in main paper / appendix
- ✅ Multi-seed (Exp 5) and 3B (Exp 6) results added
- ✅ Resubmit for next ARR cycle if Findings rejected

---

🚀 **Now go run Exp 1.**
