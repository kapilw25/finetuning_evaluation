"""
Generate SFT Training Pipeline Diagram from Mermaid code

Usage:
    python Overleaf_draft/src/system_design_diagrams/generate_sft_pipeline.py

Output:
    Overleaf_draft/version_Professor/figures/pipeline/sft_pipeline.png
    Overleaf_draft/version_Professor/figures/pipeline/sft_pipeline.pdf

Requirements:
    pip install requests Pillow

Note: Uses mermaid.ink API (no local installation needed)
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
TITLE = "SFT (Supervised Fine-Tuning) Pipeline"

# SFT Pipeline Mermaid Code (Left-to-Right layout for full A4 width)
MERMAID_CODE = """
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '40px', 'fontFamily': 'Arial Black, Helvetica, sans-serif', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'arrowheadColor': '#000000'}, 'flowchart': {'subGraphTitleMargin': {'top': 30, 'bottom': 30}, 'padding': 40, 'nodeSpacing': 60, 'rankSpacing': 80, 'curve': 'linear', 'arrowMarkerAbsolute': true}}}%%
flowchart LR
    subgraph INPUT["<b>INPUT</b>"]
        direction TB
        BASE["<b>Llama-3.1-8B<br/>BF16</b>"]
        BASE ~~~ PKU
        PKU["<b>PKU-SafeRLHF<br/>12,035 samples</b>"]
    end

    subgraph DATA["<b>DATA</b>"]
        direction TB
        FILTER["<b>Clear Contrast<br/>Filter</b>"]
        FILTER ~~~ EXTRACT
        EXTRACT["<b>Extract Y+<br/>harm_categories</b>"]
        EXTRACT ~~~ SYNTH
        SYNTH["<b>Synthesize<br/>Instruction I</b>"]
    end

    subgraph MODEL["<b>MODEL</b>"]
        direction TB
        LORA["<b>LoRA r=16<br/>q,k,v,o,gate,up,down</b>"]
    end

    subgraph FORMAT["<b>FORMAT</b>"]
        direction TB
        FMT_I["<b>Instruct<br/>[sys, user, asst]</b>"]
        FMT_I ~~~ FMT_NI
        FMT_NI["<b>NoInstruct<br/>[user, asst]</b>"]
    end

    subgraph TRAIN["<b>TRAINING</b>"]
        direction TB
        LOSS["<b>L_SFT = -Sum log P(yt|y_t, X)</b>"]
        LOSS ~~~ OPT
        OPT["<b>AdamW lr=2e-4<br/>Cosine LR</b>"]
    end

    subgraph OUTPUT["<b>OUTPUT</b>"]
        direction TB
        SFT_NI["<b>SFT_NoInstruct<br/>pi(Y|X)</b>"]
        SFT_NI ~~~ SFT_I
        SFT_I["<b>SFT_Instruct<br/>pi(Y|I,X)</b>"]
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
    print("GENERATE SFT PIPELINE DIAGRAM")
    print("=" * 60)

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}")

    # Output paths
    png_path = FIGURES_DIR / "sft_pipeline.png"
    pdf_path = FIGURES_DIR / "sft_pipeline.pdf"

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
    print(r"  \includegraphics[width=\textwidth]{figures/pipeline/sft_pipeline.pdf}")


if __name__ == "__main__":
    main()
