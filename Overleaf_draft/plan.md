# CITA Paper - LaTeX Structure Plan

## Reference: VLM Adversarial Paper Structure

```
MultiModal_Adversarial_attacks/Overleaf_draft/
├── main.tex              # Master file with \input{sections/...}
├── references.bib        # BibTeX references
├── figures/              # All plots (numbered 01_, 03_, etc.)
└── sections/
    ├── abstract.tex
    ├── introduction.tex
    ├── methodology.tex
    ├── results.tex
    └── conclusion.tex
```

## Key Patterns to Adopt

| Aspect | VLM Paper | CITA Adaptation |
|--------|-----------|-----------------|
| **Abstract** | Single paragraph, clear structure | Keep same style |
| **Introduction** | Problem, Gap, Contributions, Organization | Same structure |
| **Methodology** | Equations with `\begin{equation}`, itemized lists | Use for L_unified, L_CITA |
| **Results** | `\begin{figure*}` for wide plots, numbered figures | 6 eval plots |
| **References** | Separate `.bib` file | Same |

## Proposed CITA Directory Structure

```
Overleaf_draft/
├── main.tex
├── references.bib
├── figures/
│   ├── isd_comparison.png
│   ├── toxicity_comparison.png
│   ├── truthfulqa_comparison.png
│   ├── conditional_safety_comparison.png
│   ├── style_transfer_comparison.png
│   └── aqi_comparison.png
└── sections/
    ├── abstract.tex
    ├── introduction.tex        # Problem, Gap, Contributions
    ├── related_work.tex        # SFT, DPO, RLHF, Constitutional AI
    ├── methodology.tex         # CITA loss, ISD dataset, training pipeline
    ├── experiments.tex         # 6 models, 6 evals, hardware setup
    ├── results.tex             # Results comparison table + plots
    └── conclusion.tex
```

## Main.tex Template

```latex
\documentclass[10pt,twocolumn]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}  % For professional tables

\title{CITA: Calibrated Instruction Tuning with Alignment}
\author{Anonymous ACL submission}

\begin{document}
\maketitle

\begin{abstract}
\input{sections/abstract}
\end{abstract}

\section{Introduction}
\input{sections/introduction}

\section{Related Work}
\input{sections/related_work}

\section{Methodology}
\input{sections/methodology}

\section{Experiments}
\input{sections/experiments}

\section{Results}
\input{sections/results}

\section{Conclusion}
\input{sections/conclusion}

\bibliographystyle{acl_natbib}
\bibliography{references}
\end{document}
```

## Key LaTeX Patterns

### 1. Equations (methodology.tex)

```latex
\begin{equation}
\mathcal{L}_{\text{unified}} = \mathcal{L}_{\text{SFT}} + \lambda_1 \mathcal{L}_{\text{DPO}} + \lambda_2 \mathcal{L}_{\text{KL}}
\end{equation}
```

### 2. Full-width Figures (results.tex)

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.9\textwidth]{figures/isd_comparison.png}
\caption{ISD evaluation: Instruct variants show 2x higher fidelity...}
\label{fig:isd}
\end{figure*}
```

### 3. Comparison Tables

```latex
\begin{table*}[!t]
\centering
\begin{tabular}{lcccccc}
\toprule
Model & ISD & Toxicity & TruthfulQA & Cond. Safety & Style & AQI \\
\midrule
CITA\_Instruct & \textbf{0.439} & 58.7\% & \textbf{0.111} & \textbf{0.390} & 1.14 & \textbf{66.5} \\
CITA\_NoInstruct & 0.215 & 36.5\% & $-$0.040 & 0.010 & 1.11 & 28.0 \\
DPO\_Instruct & \textbf{0.453} & \textbf{100\%} & $-$0.300 & 0.370 & \textbf{1.26} & 53.3 \\
DPO\_NoInstruct & 0.246 & 79.7\% & $-$0.260 & 0.030 & 1.02 & 21.6 \\
\bottomrule
\end{tabular}
\caption{Results comparison across 6 evaluation benchmarks. Bold indicates best per metric.}
\label{tab:results}
\end{table*}
```

### 4. Itemized Contributions (introduction.tex)

```latex
\begin{enumerate}
    \item \textbf{CITA Framework}: Unified loss combining SFT+DPO+KL with mandatory KL regularization
    \item \textbf{ISD Benchmark}: 300 prompts $\times$ 10 instruction types for instruction-alignment testing
    \item \textbf{Empirical Validation}: 4 models (2 CITA, 2 DPO) across 6 evaluation benchmarks
    \item \textbf{Key Finding}: CITA improves more than DPO when moving from NoInstruct $\rightarrow$ Instruct (4/5 evals)
\end{enumerate}
```

## Section Content Mapping

| Section | Content from Codebase |
|---------|----------------------|
| **Abstract** | Ecliptica PDF abstract + results summary |
| **Introduction** | Problem: static alignment; Gap: no instruction-aware methods; Contribution: CITA |
| **Related Work** | SFT, RLHF, DPO, Constitutional AI, instruction-tuning literature |
| **Methodology** | L\_unified equation, CITA-KL loss, ISD dataset design (from Ecliptica PDF) |
| **Experiments** | Training pipeline (SFT $\rightarrow$ DPO $\rightarrow$ CITA), 6 evals from observation.md |
| **Results** | observation.md comparison tables + plots from logs\_training/iter13/plots/ |
| **Conclusion** | CITA wins 4/5 improvement metrics, future work |

## Figures to Include (from logs\_training/iter13/plots/)

1. `isd_comparison.png` - Instruction Switch Dataset results
2. `toxicity_comparison.png` - Safe refusal rates
3. `truthfulqa_comparison.png` - Confidence adaptation scores
4. `conditional_safety_comparison.png` - STRICT vs PERMISSIVE gap
5. `style_transfer_comparison.png` - Length adaptation ratios
6. `aqi_comparison.png` - Alignment Quality Index

## Key Results to Highlight

### CITA vs DPO: Improvement from NoInstruct → Instruct

| Eval | CITA_NoInstruct | CITA_Instruct | CITA_Δ | DPO_NoInstruct | DPO_Instruct | DPO_Δ | Winner |
|------|-----------------|---------------|--------|----------------|--------------|-------|--------|
| ISD | 0.215 | 0.439 | +0.224 🥇 | 0.246 | 0.453 | +0.207 🥈 | CITA |
| TruthfulQA | -0.040 | 0.111 | +0.151 🥇 | -0.260 | -0.300 | -0.040 🥈 | CITA |
| Conditional Safety | 0.010 | 0.390 | +0.380 🥇 | 0.030 | 0.370 | +0.340 🥈 | CITA |
| Style Transfer | 1.11 | 1.14 | +0.03 🥈 | 1.02 | 1.26 | +0.24 🥇 | DPO |
| AQI | 28.0 | 66.5 | +38.5 🥇 | 21.6 | 53.3 | +31.7 🥈 | CITA |

*Note: Excluding Toxicity (only eval using LLM-as-judge)*

**Final Score: CITA 4 - DPO 1**

**Key Narrative**: CITA benefits more from instruction-awareness than DPO. Most notably on TruthfulQA where CITA improves (+0.151) while DPO actually degrades (-0.040).

## LaTeX Special Characters Reference

| Character | LaTeX Code |
|-----------|------------|
| % | `\%` |
| _ | `\_` |
| x (multiply) | `$\times$` |
| -> (arrow) | `$\rightarrow$` |
| - (negative) | `$-$` (in math mode) |
| & | `\&` |
| # | `\#` |
