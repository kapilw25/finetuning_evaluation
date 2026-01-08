"""
Generate Unified Evaluation Pipeline Diagram from Mermaid code

Usage:
    python Overleaf_draft/src/system_design_diagrams/generate_evaluation_pipeline.py

Output:
    Overleaf_draft/version_Professor/figures/pipeline/evaluation_pipeline.png
    Overleaf_draft/version_Professor/figures/pipeline/evaluation_pipeline.pdf

Requirements:
    pip install requests Pillow

Note: Uses mermaid.ink API (no local installation needed)

Evaluation Pipeline Architecture - Key Insight:
  INFERENCE is IDENTICAL for all 5 benchmarks (standard vLLM generation)
  METRIC CALCULATION is FUNDAMENTALLY DIFFERENT:

Three Computational Branches:

1. EMBEDDING-BASED (requires ML inference):
   - ECLIPTICA (M₁): SentenceTransformer → cosine_sim(Y_NI, Y_I) = Fidelity
                     → semantic_distance(shift) = Shift
                     → M₁ = Fidelity × Shift
   - LITMUS/AQI (M₅): SentenceTransformer → t-SNE reduction
                      → Calinski-Harabasz index (CHI) + Xie-Beni index (XB)
                      → AQI = (CHI + XB) / 2

2. HEURISTIC PHRASE DETECTION (no ML, text pattern matching):
   - TruthfulQA (M₂): Count 23 uncertainty markers ("maybe", "I'm not sure", etc.)
                      → honest_rate = matches / total
                      → M₂ = HON_rate - CONF_rate (calibration)
   - Cond. Safety (M₃): Count 25 refusal indicators ("I cannot", "I won't", etc.)
                        → refusal_rate = weighted_matches / total
                        → M₃ = |STRICT_rate - PERMIS_rate| (adaptation)

3. PURE COUNTING (no ML, simple arithmetic):
   - Length Ctrl (M₄): word_count = len(text.split())
                       → M₄ = DETAIL_words / CONC_words (ratio)
"""

from pathlib import Path
import sys

# Add parent to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.mermaid_utils import render_mermaid_to_png, convert_png_to_pdf

# Paths
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent.parent / "version_Professor" / "figures" / "pipeline"

# Title for the figure
TITLE = "Evaluation Pipeline: Metric Calculation Branches"

# Evaluation Pipeline Mermaid Code (Left-to-Right with explicit branching)
# Key insight: INFERENCE is identical, METRIC CALCULATION differs fundamentally
MERMAID_CODE = """
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '36px', 'fontFamily': 'Arial Black, Helvetica, sans-serif', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'arrowheadColor': '#000000'}, 'flowchart': {'subGraphTitleMargin': {'top': 25, 'bottom': 20}, 'padding': 35, 'nodeSpacing': 50, 'rankSpacing': 80, 'curve': 'linear', 'arrowMarkerAbsolute': true}}}%%
flowchart LR
    subgraph INPUT["<b>INPUT</b>"]
        direction TB
        MODELS["<b>10 Models<br/>(SFT,DPO,PPO<br/>GRPO,CITA)<br/>× (NI, I)</b>"]
    end

    subgraph INFER["<b>INFERENCE</b>"]
        direction TB
        LOAD["<b>Load Model</b>"]
        LOAD ~~~ GEN
        GEN["<b>Generate<br/>Responses</b>"]
    end

    subgraph CALC["<b>METRIC CALCULATION</b>"]
        direction TB

        subgraph EMB["<b>EMBEDDING</b>"]
            direction TB
            ST["<b>SentenceTransformer</b>"]
            M1["<b>M₁ ECLIPTICA<br/>Fidelity × Shift</b>"]
            M5["<b>M₅ LITMUS<br/>(CHI + XB)/2</b>"]
        end

        subgraph HEUR["<b>HEURISTIC</b>"]
            direction TB
            PAT["<b>Pattern Match</b>"]
            M2["<b>M₂ TruthfulQA<br/>HON − CONF</b>"]
            M3["<b>M₃ Safety<br/>|STR − PER|</b>"]
        end

        subgraph CNT["<b>COUNTING</b>"]
            direction TB
            WC["<b>Word Count</b>"]
            M4["<b>M₄ Length<br/>DET / CON</b>"]
        end
    end

    subgraph OUT["<b>OUTPUT</b>"]
        direction TB
        SCORES["<b>Scores</b>"]
        SCORES ~~~ DELTA
        DELTA["<b>Δ=I−NI</b>"]
    end

    INPUT --> INFER
    INFER --> EMB
    INFER --> HEUR
    INFER --> CNT
    EMB --> OUT
    HEUR --> OUT
    CNT --> OUT

    classDef inputStyle fill:#E6E6FA,stroke:#000000,stroke-width:5px,color:#000000,font-weight:bold
    classDef inferStyle fill:#FFE4B5,stroke:#000000,stroke-width:5px,color:#000000,font-weight:bold
    classDef calcStyle fill:#FFFFFF,stroke:#000000,stroke-width:4px,color:#000000,font-weight:bold
    classDef embStyle fill:#ADD8E6,stroke:#000000,stroke-width:5px,color:#000000,font-weight:bold
    classDef heurStyle fill:#FFB6C1,stroke:#000000,stroke-width:5px,color:#000000,font-weight:bold
    classDef cntStyle fill:#98FB98,stroke:#000000,stroke-width:5px,color:#000000,font-weight:bold
    classDef outputStyle fill:#90EE90,stroke:#000000,stroke-width:5px,color:#000000,font-weight:bold
    classDef default stroke:#000000,stroke-width:4px,color:#000000,font-weight:bold
    linkStyle default stroke:#000000,stroke-width:6px

    class INPUT inputStyle
    class INFER inferStyle
    class CALC calcStyle
    class EMB embStyle
    class HEUR heurStyle
    class CNT cntStyle
    class OUT outputStyle
"""


def main():
    print("=" * 60)
    print("GENERATE EVALUATION PIPELINE DIAGRAM")
    print("(Metric Calculation Branches)")
    print("=" * 60)
    print("Key insight: INFERENCE is identical, METRIC CALCULATION differs")
    print()
    print("3 Computational Branches:")
    print("  [EMBEDDING-BASED] (requires ML)")
    print("    - M₁ ECLIPTICA: SentenceTransformer → Fidelity × Shift")
    print("    - M₅ LITMUS: SentenceTransformer → CHI + XB clusters")
    print()
    print("  [HEURISTIC DETECTION] (pattern matching)")
    print("    - M₂ TruthfulQA: 23 uncertainty markers → HON - CONF")
    print("    - M₃ Cond. Safety: 25 refusal indicators → |STRICT - PERMIS|")
    print()
    print("  [PURE COUNTING] (arithmetic)")
    print("    - M₄ Length Ctrl: word count → DETAIL / CONC")
    print("=" * 60)

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}")

    # Output paths
    png_path = FIGURES_DIR / "evaluation_pipeline.png"
    pdf_path = FIGURES_DIR / "evaluation_pipeline.pdf"

    # Render to PNG (scale=4 for high resolution)
    print("\n[1/2] Rendering Mermaid to PNG...")
    if not render_mermaid_to_png(MERMAID_CODE, png_path, scale=4, title=TITLE):
        print("[FAILED] Could not generate PNG")
        return

    # Convert to PDF
    print("\n[2/2] Converting PNG to PDF...")
    if not convert_png_to_pdf(png_path, pdf_path):
        print("[FAILED] Could not generate PDF")
        return

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print("\nTo include in LaTeX:")
    print(r"  \includegraphics[width=\textwidth]{figures/pipeline/evaluation_pipeline.pdf}")


if __name__ == "__main__":
    main()
