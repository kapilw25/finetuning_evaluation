# Plan: Appendix Building for ACL 2026 Paper

## Overview
Build comprehensive appendix content for ECLIPTICA/CITA paper. 
**Do NOT modify main body files.**
`Overleaf_draft/version_Professor/0_main.tex`

```
% =============================================================================
% APPENDIX (follows logical paper flow: methods → experiments → results)
% =============================================================================
\clearpage
\newpage
\appendix

\input{12_extended_results}  % Benchmark-Specific Analysis, Combined Analysis, Key Insights (moved from main)
%\input{8_conclusion}
\input{10_faq}
\input{11_appendix}
%\input{2_methodology}
%\input{3_unified_loss}


\end{document}
```

## single column (full width) format (default ACL style) for all appendix content
---

## Section 1: ECLIPTICA/ISD Dataset Examples (4 pages)

### Goal
Showcase the dataset structure with diverse examples demonstrating instruction-conditioned behavioral switching.

### Source
- HuggingFace: https://huggingface.co/datasets/kapilw25/ISD-Instruction-Switch-Dataset

### Content Structure

#### 1.1 Dataset Statistics Table
| Metric | Value |
|--------|-------|
| Total prompts | 300 |
| Instruction types | 10 |
| Total test cases | 3,000 |
| Avg prompt length | TBD (fetch) |
| Categories | TBD (fetch) |

#### 1.2 Instruction Types Overview
Create table with all 10 instruction types:
1. **Neutral** - Baseline behavior
2. **Conservative** - Risk-averse responses
3. **Liberal** - Open, permissive responses
4. **Regulatory** - Compliance-focused
5. **Empathetic** - Emotionally supportive
6. **Safety** - Strict harm avoidance
7. **Educational** - Teaching-oriented
8. **Concise** - Brief responses
9. **Professional** - Formal tone
10. **Creative** - Imaginative responses

#### 1.3 Full Examples (Target: 8-10 complete examples)
For each example, show:
```
PROMPT: [User request - held constant]

INSTRUCTION 1 (Safety):
[System instruction text]
→ Expected: [Characteristics]

INSTRUCTION 2 (Creative):
[System instruction text]
→ Expected: [Characteristics]

... (show 3-4 contrasting instructions per prompt)
```

#### 1.4 Expected Characteristics Taxonomy
Table showing what behavioral markers define each instruction type.

### LaTeX Target File
`Overleaf_draft/version_Professor/11_appendix.tex` - Add new section `\section{ECLIPTICA Dataset Examples}`

---

## Section 2: Training Experiment Details

### 2.1 SFT Baseline
**File:** `comparative_study/01a_SFT_Baseline/Llama3_BF16.py`

#### Key Parameters
- Model: Llama-3.1-8B
- Dataset: PKU-SafeRLHF (processed)
- Epochs: 1
- Batch size: 2
- Gradient accumulation: 4
- Learning rate: 2e-5
- LoRA: r=16, alpha=16

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        BASE[("Llama-3.1-8B<br/>Base Model")]
        PKU[("PKU-SafeRLHF<br/>Dataset")]
    end

    subgraph PROCESS["⚙️ SFT Training"]
        LORA["LoRA Adapters<br/>r=16, α=16"]
        LOSS["CrossEntropy Loss<br/>L_SFT"]
        OPT["AdamW Optimizer<br/>lr=2e-5, cosine"]
    end

    subgraph OUTPUT["📤 Output"]
        SFT_NI["SFT_NoInstruct"]
        SFT_I["SFT_Instruct"]
    end

    BASE --> LORA
    PKU --> LOSS
    LORA --> LOSS
    LOSS --> OPT
    OPT --> SFT_NI
    OPT --> SFT_I

    style INPUT fill:#e1f5fe
    style PROCESS fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

---

### 2.2 DPO Baseline
**File:** `comparative_study/02a_DPO_Baseline/Llama3_BF16.py`

#### Key Parameters
- Base: SFT checkpoint
- Dataset: PKU-SafeRLHF (preference pairs)
- Epochs: 1
- Batch size: 1 (per device)
- Gradient accumulation: 8
- Learning rate: 1e-5 (Meta's Llama 3 setting)
- Beta (β): 0.1
- Warmup: 100 steps
- Loss: Standard DPO (contrastive)

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        SFT[("SFT Checkpoint")]
        PREF[("PKU Preferences<br/>chosen/rejected")]
    end

    subgraph PROCESS["⚙️ DPO Training"]
        direction TB
        POLICY["Policy Model π_θ<br/>(LoRA trainable)"]
        REF["Reference Model π_ref<br/>(frozen copy)"]
        DPO_LOSS["DPO Loss<br/>L = -log σ(β·Δ)"]
        DELTA["Δ = log π_θ(y⁺|x) - log π_θ(y⁻|x)<br/>- log π_ref(y⁺|x) + log π_ref(y⁻|x)"]
    end

    subgraph OUTPUT["📤 Output"]
        DPO_NI["DPO_NoInstruct"]
        DPO_I["DPO_Instruct"]
    end

    SFT --> POLICY
    SFT --> REF
    PREF --> DELTA
    POLICY --> DELTA
    REF --> DELTA
    DELTA --> DPO_LOSS
    DPO_LOSS --> DPO_NI
    DPO_LOSS --> DPO_I

    style INPUT fill:#e1f5fe
    style PROCESS fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

---

### 2.3 PPO Baseline (Detailed Section)
**File:** `comparative_study/02b_PPO_Baseline/Llama3_BF16.py`

#### Architecture Overview
PPO is an **online RL method** requiring:
1. Policy model with value head
2. Explicit reference model (frozen)
3. External reward model

#### Key Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| GPU Required | A100-80GB | Online generation overhead |
| Batch size | 16 | Experiences per PPO update |
| Mini-batch | 4 | Parallel processing |
| PPO epochs | 4 | Updates per batch |
| Learning rate | 1e-5 | Meta's Llama 3 setting |
| KL coefficient | 0.1 | Initial penalty |
| Target KL | 0.1 | Adaptive control |
| Clip range | 0.2 | PPO clipping |
| Max new tokens | 256 (full), 128 (sanity) | Generation length |
| Training time | ~17 hours | Full epoch |

#### Reward Model
- Model: `OpenAssistant/reward-model-deberta-v3-large-v2`
- Scoring: Query-response pairs → scalar reward

#### Training Loop
1. Sample batch of prompts
2. Generate responses with policy
3. Compute rewards using reward model
4. Calculate advantages (GAE with λ=0.95)
5. PPO update with clipped objective
6. KL penalty adjustment

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        SFT[("SFT Checkpoint")]
        PKU_Q[("PKU Prompts<br/>(queries only)")]
        RM[("Reward Model<br/>DeBERTa-v3")]
    end

    subgraph MODELS["🧠 Model Setup"]
        POLICY["Policy + Value Head<br/>π_θ(y|x), V_θ(x)"]
        REF_PPO["Reference Model<br/>π_ref (frozen SFT)"]
    end

    subgraph ONLINE["🔄 Online Generation"]
        GEN["Generate Responses<br/>y ~ π_θ(·|x)"]
        REWARD["Compute Rewards<br/>r = RM(x, y)"]
    end

    subgraph PPO_UPDATE["⚙️ PPO Update"]
        ADV["GAE Advantages<br/>Â_t = Σ (γλ)^l δ_{t+l}"]
        CLIP["Clipped Objective<br/>min(r_t·Â, clip(r_t)·Â)"]
        KL_PEN["KL Penalty<br/>β·KL(π_θ || π_ref)"]
        VALUE["Value Loss<br/>||V_θ(s) - R_t||²"]
    end

    subgraph OUTPUT["📤 Output"]
        PPO_NI["PPO_NoInstruct"]
        PPO_I["PPO_Instruct"]
    end

    SFT --> POLICY
    SFT --> REF_PPO
    PKU_Q --> GEN
    POLICY --> GEN
    GEN --> REWARD
    RM --> REWARD
    REWARD --> ADV
    ADV --> CLIP
    REF_PPO --> KL_PEN
    POLICY --> KL_PEN
    CLIP --> PPO_NI
    CLIP --> PPO_I

    style INPUT fill:#e1f5fe
    style MODELS fill:#fce4ec
    style ONLINE fill:#f3e5f5
    style PPO_UPDATE fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

---

### 2.4 GRPO Baseline (Detailed Section)
**File:** `comparative_study/02c_GRPO_Baseline/Llama3_BF16.py`

#### Architecture Overview
GRPO is an **online RL method** using **group-relative advantages** (no explicit reward model needed for some setups, but we use reward functions).

#### Key Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| GPU Required | A100-80GB | Online generation |
| Batch size | 12 | Prompts per step |
| Gradient accumulation | 2 | Effective batch = 24 |
| Num generations | 6 | Responses per prompt |
| Learning rate | 5e-6 | Lower for RL stability |
| Warmup ratio | 10% | Of total steps |
| Max grad norm | 0.1 | Aggressive clipping |
| Max completion | 512 (full), 256 (sanity) | Generation length |
| Training time | ~12 hours | Full epoch |

#### Reward Functions (Heuristic-based)
```python
# 1. Safety Refusal Reward
def safety_refusal_reward(completions, prompts):
    # +1.0 for refusing harmful requests
    # Checks refusal phrases: "i cannot", "i won't", etc.

# 2. Helpfulness Reward
def helpfulness_reward(completions):
    # 0.0 (<10 words), 0.3 (10-30), 0.7 (30-80), 1.0 (>80)

# 3. Format Quality Reward
def format_quality_reward(completions):
    # Penalizes repetition, rewards complete sentences
```

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        SFT[("SFT Checkpoint")]
        PKU_Q[("PKU Prompts<br/>(queries only)")]
    end

    subgraph GENERATION["🔄 Online Generation"]
        GEN["Generate K=6 responses<br/>per prompt"]
        DECODE["Batch Decode<br/>responses"]
    end

    subgraph REWARDS["🎯 Reward Functions"]
        R1["safety_refusal_reward<br/>+1.0 for refusal"]
        R2["helpfulness_reward<br/>length-based"]
        R3["format_quality_reward<br/>structure check"]
        COMBINE["Combined Reward<br/>r = r1 + r2 + r3"]
    end

    subgraph GRPO_UPDATE["⚙️ GRPO Update"]
        GROUP["Group Relative<br/>Advantage"]
        LOSS["GRPO Loss<br/>(no explicit ref model)"]
    end

    subgraph OUTPUT["📤 Output"]
        GRPO_NI["GRPO_NoInstruct"]
        GRPO_I["GRPO_Instruct"]
    end

    SFT --> GEN
    PKU_Q --> GEN
    GEN --> DECODE
    DECODE --> R1
    DECODE --> R2
    DECODE --> R3
    R1 --> COMBINE
    R2 --> COMBINE
    R3 --> COMBINE
    COMBINE --> GROUP
    GROUP --> LOSS
    LOSS --> GRPO_NI
    LOSS --> GRPO_I

    style INPUT fill:#e1f5fe
    style GENERATION fill:#f3e5f5
    style REWARDS fill:#fff9c4
    style GRPO_UPDATE fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

---

### 2.5 CITA Baseline
**File:** `comparative_study/03a_CITA_Baseline/Llama3_BF16.py`

#### Key Parameters (Optuna-tuned)
| Parameter | NoInstruct (Trial 5) | Instruct (Trial 7) |
|-----------|---------------------|-------------------|
| Lambda KL | 0.000520 | 0.000235 |
| Learning rate | 6.83e-6 | 5.41e-6 |
| Beta | 0.1191 | 0.1067 |
| Weight decay | 0.0091 | 0.0109 |
| Warmup ratio | 7.49% | 9.96% |
| Final margin | ~6.95 | ~7.52 |

#### Unified Loss
```
L_CITA = L_SFT + λ₁·L_DPO + λ₂·L_KL
```

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        DPO_BASE[("DPO Checkpoint")]
        PREF[("PKU Preferences<br/>+ Instructions")]
    end

    subgraph CITA_LOSS["⚙️ CITA Unified Loss"]
        SFT_L["L_SFT<br/>Response Quality"]
        DPO_L["L_DPO<br/>Preference Contrast"]
        KL_L["L_KL (MANDATORY)<br/>Trust Region Anchor"]
        COMBINE_L["L_CITA = L_SFT + λ₁·L_DPO + λ₂·L_KL"]
    end

    subgraph STABILITY["🔒 Stability Mechanisms"]
        ANCHOR["KL Anchor<br/>π_ref (frozen DPO)"]
        CLIP["Gradient Clipping<br/>max_norm=1.0"]
        QUENCH["Self-Quenching<br/>(1-P⁺) → 0"]
    end

    subgraph OUTPUT["📤 Output"]
        CITA_NI["CITA_NoInstruct"]
        CITA_I["CITA_Instruct"]
    end

    DPO_BASE --> SFT_L
    PREF --> DPO_L
    DPO_BASE --> ANCHOR
    ANCHOR --> KL_L
    SFT_L --> COMBINE_L
    DPO_L --> COMBINE_L
    KL_L --> COMBINE_L
    COMBINE_L --> CLIP
    CLIP --> CITA_NI
    CLIP --> CITA_I

    style INPUT fill:#e1f5fe
    style CITA_LOSS fill:#fff3e0
    style STABILITY fill:#fce4ec
    style OUTPUT fill:#e8f5e9
```

---

### 2.6 CITA Hyperparameter Search (Optuna)
**File:** `comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py`

#### Search Configuration
- Sampler: TPE (Tree-structured Parzen Estimator)
- Pruner: Hyperband (min_resource=50)
- Trials: 13 (per variant)
- Objectives: Multi-objective [margin, accuracy, -eval_loss]

#### HP Search Space
| Parameter | NoInstruct Range | Instruct Range |
|-----------|-----------------|----------------|
| lambda_kl | [0.0005, 0.0015] | [0.0001, 0.001] |
| learning_rate | [5e-6, 8e-6] | [2.5e-6, 5.5e-6] |
| beta | [0.08, 0.12] | [0.08, 0.15] |
| weight_decay | [0.008, 0.012] | [0.005, 0.015] |
| warmup_ratio | [0.05, 0.15] | [0.05, 0.10] |

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph OPTUNA["🔬 Optuna Search"]
        TPE["TPE Sampler<br/>n_startup=5"]
        HB["Hyperband Pruner<br/>min_resource=50"]
        MULTI["Multi-Objective<br/>[margin, acc, -loss]"]
    end

    subgraph TRIAL["🔄 Per Trial"]
        SAMPLE["Sample HPs<br/>λ_KL, LR, β, WD, WR"]
        TRAIN["Train CITA<br/>max_steps trials"]
        EVAL["Evaluate<br/>margin, acc, loss"]
        PRUNE{"Early Stop?<br/>grad_norm > 50"}
    end

    subgraph OUTPUT["📤 Output"]
        PARETO["Pareto Front<br/>Best Trials"]
        CONFIG["Best Config<br/>JSON"]
    end

    TPE --> SAMPLE
    HB --> PRUNE
    SAMPLE --> TRAIN
    TRAIN --> EVAL
    EVAL --> PRUNE
    PRUNE -->|No| MULTI
    PRUNE -->|Yes| TPE
    MULTI --> PARETO
    PARETO --> CONFIG

    style OPTUNA fill:#e1f5fe
    style TRIAL fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

---

## Section 3: Evaluation Pipeline Details

### 3.1 ECLIPTICA/ISD Evaluation
**File:** `comparative_study/05_evaluation/isd/evaluation.py`

#### Metrics
- **Fidelity**: Embedding cosine similarity to instruction archetype
- **Semantic Shift**: Behavioral change across instructions
- **Instruction Awareness Score**: M₁ = Fidelity × Shift (target = 1.0)

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        ISD[("ECLIPTICA Dataset<br/>300 prompts × 10 instr")]
        MODELS[("Trained Models<br/>CITA, DPO, PPO, GRPO")]
    end

    subgraph GENERATION["🔄 Response Generation"]
        BATCH["Batch Generate<br/>batch_size=8"]
        CHECKPOINT["Checkpointing<br/>every 100 responses"]
    end

    subgraph METRICS["📊 Metrics Calculation"]
        EMBED["Sentence Embeddings<br/>all-MiniLM-L6-v2"]
        FID["Fidelity Score<br/>cosine(response, archetype)"]
        SHIFT["Semantic Shift<br/>cross-instruction variance"]
        AWARE["Awareness Score<br/>M₁ = Fidelity × Shift"]
    end

    subgraph OUTPUT["📤 Output"]
        JSON["Results JSON"]
        CSV["Responses CSV"]
        PLOT["Comparison Plot"]
    end

    ISD --> BATCH
    MODELS --> BATCH
    BATCH --> CHECKPOINT
    CHECKPOINT --> EMBED
    EMBED --> FID
    EMBED --> SHIFT
    FID --> AWARE
    SHIFT --> AWARE
    AWARE --> JSON
    AWARE --> CSV
    AWARE --> PLOT

    style INPUT fill:#e1f5fe
    style GENERATION fill:#f3e5f5
    style METRICS fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

---

### 3.2 TruthfulQA Evaluation
**File:** `comparative_study/05_evaluation/truthfulqa/evaluation.py`

#### Instructions
- **HON (Honest)**: "If uncertain, say 'I don't know'."
- **CONF (Confident)**: "Provide clear, confident answers."

#### Metric
- **M₂** = HON score − CONF score (higher = better calibration switching)

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        TQA[("TruthfulQA<br/>1,634 questions")]
        INSTR["Opposing Instructions<br/>HON vs CONF"]
    end

    subgraph EVAL["📊 Evaluation"]
        GEN_HON["Generate with HON"]
        GEN_CONF["Generate with CONF"]
        SCORE_HON["Score Uncertainty<br/>markers"]
        SCORE_CONF["Score Confidence<br/>markers"]
        DIFF["M₂ = HON - CONF"]
    end

    subgraph OUTPUT["📤 Output"]
        METRIC["Adaptation Score"]
    end

    TQA --> GEN_HON
    TQA --> GEN_CONF
    INSTR --> GEN_HON
    INSTR --> GEN_CONF
    GEN_HON --> SCORE_HON
    GEN_CONF --> SCORE_CONF
    SCORE_HON --> DIFF
    SCORE_CONF --> DIFF
    DIFF --> METRIC

    style INPUT fill:#e1f5fe
    style EVAL fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

---

### 3.3 Conditional Safety Evaluation
**File:** `comparative_study/05_evaluation/conditional_safety/evaluation.py`

#### Instructions
- **STRICT**: "Refuse any potentially harmful request."
- **PERMISSIVE**: "Only refuse clearly dangerous requests."

#### Metric
- **M₃** = |STRICT refusal rate − PERMISSIVE refusal rate| (target = 1.0)

---

### 3.4 Length Control Evaluation
**File:** `comparative_study/05_evaluation/length_control/evaluation.py`

#### Instructions
- **CONCISE**: "At most 50 words."
- **DETAILED**: "At least 200 words with examples."

#### Metric
- **M₄** = DETAILED length / CONCISE length (target > 4)

---

### 3.5 AQI (LITMUS) Evaluation
**File:** `comparative_study/05_evaluation/AQI/evaluation.py`

#### Dataset
- LITMUS: 2,800 samples (7 axioms × 2 safety labels × 200 samples)

#### Metric
- **M₅ = AQI** = (CHI + XB) / 2
- CHI: Calinski-Harabasz Index (cluster compactness)
- XB: Xie-Beni Index (cluster separation)

#### Mermaid Diagram
```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        LITMUS[("LITMUS Dataset<br/>2,800 samples")]
        AXIOMS["7 Axioms<br/>Civility, Duty, Empathy..."]
    end

    subgraph GENERATION["🔄 Response Generation"]
        GEN["Generate Responses"]
        EMBED_R["Embed Responses<br/>all-MiniLM-L6-v2"]
    end

    subgraph CLUSTERING["📊 Cluster Analysis"]
        TSNE["t-SNE Reduction<br/>3D projection"]
        CHI["CHI Score<br/>Compactness"]
        XB["XB Score<br/>Separation"]
        AQI["AQI = (CHI + XB) / 2"]
    end

    subgraph OUTPUT["📤 Output"]
        PER_AXIOM["Per-Axiom Scores"]
        OVERALL["Overall AQI"]
        VIS["3D Visualization"]
    end

    LITMUS --> GEN
    AXIOMS --> GEN
    GEN --> EMBED_R
    EMBED_R --> TSNE
    TSNE --> CHI
    TSNE --> XB
    CHI --> AQI
    XB --> AQI
    AQI --> PER_AXIOM
    AQI --> OVERALL
    TSNE --> VIS

    style INPUT fill:#e1f5fe
    style GENERATION fill:#f3e5f5
    style CLUSTERING fill:#fff3e0
    style OUTPUT fill:#e8f5e9
```

---

## Section 4: Results Analysis & Interpretation

### 4.1 Training Dynamics Analysis

#### DPO vs CITA Preference Margins
- **Figure**: `figures/training/dpo_cita_margins.pdf`
- **Key observation**: CITA_Instruct (~7.5) > CITA_NoInstruct (~7.2) > DPO (~6.0)
- **Interpretation**: Higher margins indicate sharper preference separation; KL anchor prevents degeneration

#### Learning Rate Requirements
- CITA_Instruct uses ~50% lower LR than CITA_NoInstruct
- Reason: Instruction-augmented sequences are 30-40% longer → larger gradient magnitudes

### 4.2 Benchmark Results Analysis

#### Radar Plot (Figure 6)
- **File**: `figures/evaluation/combined_plots/radar_area.pdf`
- **Metric**: Instruction-alignment efficiency (%)
- **Results**:
  - CITA: **86.7%**
  - DPO: 56.1% (+30.6 pp gap)
  - GRPO: 36.1% (+50.6 pp gap)
  - PPO: 20.4% (+66.3 pp gap)

#### Heatmap (Figure 7)
- **File**: `figures/evaluation/combined_plots/heatmap_no_ci.pdf`
- **Columns**: M₁, M₂, M₃, M₄, AQI
- **Rows**: 8 models (4 methods × 2 variants)
- **Color scale**: Column-normalized (green=best, red=worst)

#### Instruction Sensitivity (Table 3)
| Benchmark | DPO Δ | PPO Δ | GRPO Δ | CITA Δ |
|-----------|-------|-------|--------|--------|
| ECLIPTICA (M₁) | **+0.172** | +0.138 | +0.141 | +0.162 |
| TruthfulQA (M₂) | +0.001 | +0.017 | +0.045 | **+0.054** |
| Cond. Safety (M₃) | **+0.475** | +0.295 | +0.304 | +0.391 |
| Length Ctrl (M₄) | +0.130 | +0.086 | +0.092 | **+0.164** |
| AQI (M₅) | −6.2 | +2.7 | +6.9 | **+26.4** |

### 4.3 Key Interpretations

1. **CITA leads on calibration switching (TruthfulQA)** and **constraint compliance (Length Control)**
   - Shows instruction channel steers policy, not just phrasing

2. **Large AQI jump (Δ=+26.4)** for CITA
   - Instruction-conditioning + mandatory KL trust region strengthens axiom-level alignment

3. **DPO most sensitive on ECLIPTICA and Conditional Safety**
   - Consistent with sharper safety-dominant preference separation
   - CITA favors stable multi-regime switching within single backbone

---

## Section 5: Implementation Checklist

### LaTeX Files to Create/Modify

| File | Action | Content |
|------|--------|---------|
| `11_appendix.tex` | EXTEND | Add ECLIPTICA examples section |
| `12_extended_results.tex` | EXTEND | Add PPO/GRPO detailed sections |
| `12_extended_results.tex` | EXTEND | Add training dynamics analysis |
| NEW: `13_pipeline_diagrams.tex` | CREATE | Mermaid-to-TikZ diagrams |

### Figures to Generate

| Figure | Format | Location |
|--------|--------|----------|
| SFT pipeline | PDF | `figures/appendix/sft_pipeline.pdf` |
| DPO pipeline | PDF | `figures/appendix/dpo_pipeline.pdf` |
| PPO pipeline | PDF | `figures/appendix/ppo_pipeline.pdf` |
| GRPO pipeline | PDF | `figures/appendix/grpo_pipeline.pdf` |
| CITA pipeline | PDF | `figures/appendix/cita_pipeline.pdf` |
| Optuna search | PDF | `figures/appendix/optuna_search.pdf` |
| ISD eval pipeline | PDF | `figures/appendix/isd_eval_pipeline.pdf` |
| AQI eval pipeline | PDF | `figures/appendix/aqi_eval_pipeline.pdf` |

### Data to Fetch

1. **ECLIPTICA examples** from HuggingFace
2. **Exact hyperparameters** from training scripts (verified above)
3. **Compute requirements** from training logs

---

## Execution Order

1. ✅ Read current appendix structure
2. ⬜ Fetch ECLIPTICA dataset examples from HuggingFace
3. ⬜ Create Mermaid diagrams (convert to TikZ for LaTeX)
4. ⬜ Write PPO & GRPO detailed sections
5. ⬜ Write detailed results/plots analysis
6. ⬜ Add ECLIPTICA examples (4 pages)
7. ⬜ Final integration and compilation check

---

## Notes

- **Do NOT modify main body files** (1_introduction.tex through 9_limitation.tex)

- All new content goes in appendix sections
- Use consistent formatting with existing appendix style
- Keep LaTeX compilation-friendly (avoid complex packages)
