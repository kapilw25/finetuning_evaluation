"""
Generate GRPO Training Pipeline Diagram from Mermaid code

Usage:
    python Overleaf_draft/src/system_design_diagrams/generate_grpo_pipeline.py

Output:
    Overleaf_draft/version_Professor/figures/pipeline/grpo_pipeline.png
    Overleaf_draft/version_Professor/figures/pipeline/grpo_pipeline.pdf

Requirements:
    pip install requests Pillow

Note: Uses mermaid.ink API (no local installation needed)

GRPO Pipeline Architecture (from deep dive of comparative_study/02c_GRPO_Baseline/):
- INPUT: SFT checkpoint (merged LoRA) + PKU-SafeRLHF prompts
- DATA: Clear contrast filter → Extract prompts only (no preference pairs)
- FORMAT: NoInstruct [user] vs Instruct [sys, user] + generation prompt
- MODEL:
  * Policy π_θ with LoRA (trainable) - NO REFERENCE MODEL (key difference from PPO)
  * Reward Functions (heuristic): safety_refusal, helpfulness, format_quality
- TRAINING (Online RL Loop):
  * Generate K=6 responses per prompt
  * Score with reward functions (combined)
  * Group-Relative Advantage: compare within batch (no reference model)
  * Policy gradient update with aggressive clipping (max_grad_norm=0.1)
- OUTPUT: GRPO_NoInstruct π(Y|X), GRPO_Instruct π(Y|I,X)
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
TITLE = "GRPO (Group Relative Policy Optimization) Pipeline"

# GRPO Pipeline Mermaid Code (Left-to-Right layout for full A4 width)
MERMAID_CODE = """
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '40px', 'fontFamily': 'Arial Black, Helvetica, sans-serif', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'arrowheadColor': '#000000'}, 'flowchart': {'subGraphTitleMargin': {'top': 30, 'bottom': 30}, 'padding': 40, 'nodeSpacing': 60, 'rankSpacing': 80, 'curve': 'linear', 'arrowMarkerAbsolute': true}}}%%
flowchart LR
    subgraph INPUT["<b>INPUT</b>"]
        direction TB
        SFT["<b>SFT Checkpoint<br/>(merged LoRA)</b>"]
        SFT ~~~ PKU
        PKU["<b>PKU-SafeRLHF<br/>12,035 prompts</b>"]
    end

    subgraph DATA["<b>DATA</b>"]
        direction TB
        FILTER["<b>Clear Contrast<br/>Filter</b>"]
        FILTER ~~~ QUERY
        QUERY["<b>Extract Queries<br/>(prompts only)</b>"]
    end

    subgraph MODEL["<b>MODEL</b>"]
        direction TB
        POLICY["<b>Policy π_θ<br/>LoRA r=16</b>"]
        POLICY ~~~ REWARD
        REWARD["<b>Reward Funcs<br/>refusal+help+fmt</b>"]
    end

    subgraph FORMAT["<b>FORMAT</b>"]
        direction TB
        FMT_I["<b>Instruct<br/>[sys, user]</b>"]
        FMT_I ~~~ FMT_NI
        FMT_NI["<b>NoInstruct<br/>[user]</b>"]
    end

    subgraph TRAIN["<b>RL LOOP</b>"]
        direction TB
        GEN["<b>Generate K=6<br/>responses</b>"]
        GEN ~~~ SCORE
        SCORE["<b>Score with<br/>reward funcs</b>"]
        SCORE ~~~ ADV
        ADV["<b>Group-Relative<br/>Advantage</b>"]
    end

    subgraph OUTPUT["<b>OUTPUT</b>"]
        direction TB
        GRPO_NI["<b>GRPO_NoInstruct<br/>π(Y|X)</b>"]
        GRPO_NI ~~~ GRPO_I
        GRPO_I["<b>GRPO_Instruct<br/>π(Y|I,X)</b>"]
    end

    INPUT --> DATA
    DATA --> MODEL
    DATA --> FORMAT
    MODEL --> TRAIN
    FORMAT --> TRAIN
    TRAIN --> OUTPUT

    classDef inputStyle fill:#E6E6FA,stroke:#000000,stroke-width:6px,color:#000000,font-weight:bold
    classDef dataStyle fill:#FFE4B5,stroke:#000000,stroke-width:6px,color:#000000,font-weight:bold
    classDef modelStyle fill:#ADD8E6,stroke:#000000,stroke-width:6px,color:#000000,font-weight:bold
    classDef formatStyle fill:#B0E0E6,stroke:#000000,stroke-width:6px,color:#000000,font-weight:bold
    classDef trainStyle fill:#FAFAD2,stroke:#000000,stroke-width:6px,color:#000000,font-weight:bold
    classDef outputStyle fill:#90EE90,stroke:#000000,stroke-width:6px,color:#000000,font-weight:bold
    classDef default stroke:#000000,stroke-width:4px,color:#000000,font-weight:bold
    linkStyle default stroke:#000000,stroke-width:8px

    class INPUT inputStyle
    class DATA dataStyle
    class MODEL modelStyle
    class FORMAT formatStyle
    class TRAIN trainStyle
    class OUTPUT outputStyle
"""


def main():
    print("=" * 60)
    print("GENERATE GRPO PIPELINE DIAGRAM")
    print("=" * 60)

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}")

    # Output paths
    png_path = FIGURES_DIR / "grpo_pipeline.png"
    pdf_path = FIGURES_DIR / "grpo_pipeline.pdf"

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
    print(r"  \includegraphics[width=\textwidth]{figures/pipeline/grpo_pipeline.pdf}")


if __name__ == "__main__":
    main()
