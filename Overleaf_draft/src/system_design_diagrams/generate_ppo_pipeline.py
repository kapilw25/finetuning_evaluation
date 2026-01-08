"""
Generate PPO Training Pipeline Diagram from Mermaid code

Usage:
    python Overleaf_draft/src/system_design_diagrams/generate_ppo_pipeline.py

Output:
    Overleaf_draft/version_Professor/figures/pipeline/ppo_pipeline.png
    Overleaf_draft/version_Professor/figures/pipeline/ppo_pipeline.pdf

Requirements:
    pip install requests Pillow

Note: Uses mermaid.ink API (no local installation needed)

PPO Pipeline Architecture (from deep dive of comparative_study/02b_PPO_Baseline/):
- INPUT: SFT checkpoint (merged LoRA) + PKU-SafeRLHF dataset
- DATA: Clear contrast filter → Extract prompts only (no preference pairs)
- FORMAT: NoInstruct [user] vs Instruct [sys, user] + generation prompt
- MODEL:
  * Policy π_θ with Value Head (trainable)
  * Reference π_ref (frozen SFT copy)
  * Reward Model (external: OpenAssistant DeBERTa-v3)
- TRAINING (Online RL Loop):
  * Generate responses from policy
  * Score with reward model
  * Compute advantage (GAE, γ=1.0, λ=0.95)
  * PPO update with clipped surrogate (ε=0.2)
  * KL penalty (init_coef=0.1, target=0.1)
- OUTPUT: PPO_NoInstruct π(Y|X), PPO_Instruct π(Y|I,X)
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
TITLE = "PPO (Proximal Policy Optimization) Pipeline"

# PPO Pipeline Mermaid Code (Left-to-Right layout for full A4 width)
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
        QUERY ~~~ TOKEN
        TOKEN["<b>Tokenize<br/>input_ids</b>"]
    end

    subgraph MODEL["<b>MODELS</b>"]
        direction TB
        POLICY["<b>Policy π_θ<br/>+ Value Head</b>"]
        POLICY ~~~ REF
        REF["<b>Reference π_ref<br/>(frozen SFT)</b>"]
        REF ~~~ REWARD
        REWARD["<b>Reward Model<br/>DeBERTa-v3</b>"]
    end

    subgraph FORMAT["<b>FORMAT</b>"]
        direction TB
        FMT_I["<b>Instruct<br/>[sys, user]</b>"]
        FMT_I ~~~ FMT_NI
        FMT_NI["<b>NoInstruct<br/>[user]</b>"]
    end

    subgraph TRAIN["<b>RL LOOP</b>"]
        direction TB
        GEN["<b>Generate Y<br/>from π_θ</b>"]
        GEN ~~~ SCORE
        SCORE["<b>R(X,Y)<br/>from Reward</b>"]
        SCORE ~~~ PPO
        PPO["<b>PPO Update<br/>clip=0.2, KL=0.1</b>"]
    end

    subgraph OUTPUT["<b>OUTPUT</b>"]
        direction TB
        PPO_NI["<b>PPO_NoInstruct<br/>π(Y|X)</b>"]
        PPO_NI ~~~ PPO_I
        PPO_I["<b>PPO_Instruct<br/>π(Y|I,X)</b>"]
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
    print("GENERATE PPO PIPELINE DIAGRAM")
    print("=" * 60)

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}")

    # Output paths
    png_path = FIGURES_DIR / "ppo_pipeline.png"
    pdf_path = FIGURES_DIR / "ppo_pipeline.pdf"

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
    print(r"  \includegraphics[width=\textwidth]{figures/pipeline/ppo_pipeline.pdf}")


if __name__ == "__main__":
    main()
