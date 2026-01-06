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

import base64
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

# Paths
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent.parent / "version_Professor" / "figures" / "pipeline"

# PPO Pipeline Mermaid Code (Left-to-Right layout for full A4 width)
# Key difference from DPO: 3 models (Policy+ValueHead, Reference, Reward), Online RL loop
MERMAID_CODE = """
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '56px', 'fontFamily': 'Arial Black, Helvetica, sans-serif', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'arrowheadColor': '#000000'}, 'flowchart': {'subGraphTitleMargin': {'top': 30, 'bottom': 30}, 'padding': 40, 'nodeSpacing': 60, 'rankSpacing': 80, 'curve': 'linear', 'arrowMarkerAbsolute': true}}}%%
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

    classDef inputStyle fill:#E6E6FA,stroke:#000000,stroke-width:4px,color:#000000
    classDef dataStyle fill:#FFE4B5,stroke:#000000,stroke-width:4px,color:#000000
    classDef modelStyle fill:#ADD8E6,stroke:#000000,stroke-width:4px,color:#000000
    classDef formatStyle fill:#B0E0E6,stroke:#000000,stroke-width:4px,color:#000000
    classDef trainStyle fill:#FAFAD2,stroke:#000000,stroke-width:4px,color:#000000
    classDef outputStyle fill:#90EE90,stroke:#000000,stroke-width:4px,color:#000000
    classDef default stroke:#000000,stroke-width:3px,color:#000000
    linkStyle default stroke:#000000,stroke-width:6px

    class INPUT inputStyle
    class DATA dataStyle
    class MODEL modelStyle
    class FORMAT formatStyle
    class TRAIN trainStyle
    class OUTPUT outputStyle
"""


def render_mermaid_to_png(mermaid_code: str, output_path: Path, scale: int = 3,
                          width_multiplier: float = 1.0, height_multiplier: float = 1.0) -> bool:
    """
    Render Mermaid diagram to PNG using mermaid.ink API

    Args:
        mermaid_code: Mermaid diagram code
        output_path: Path to save PNG
        scale: Scale factor for resolution (1-5)
        width_multiplier: Multiply final width by this factor
        height_multiplier: Multiply final height by this factor

    Returns:
        True if successful, False otherwise
    """
    # Encode mermaid code to base64
    encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')

    # Use mermaid.ink API
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=white"

    print(f"Fetching diagram from mermaid.ink...")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Load image and optionally scale up for higher resolution
        img = Image.open(BytesIO(response.content))

        if scale > 1:
            new_size = (img.width * scale, img.height * scale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Apply width/height multipliers for aspect ratio adjustment
        if width_multiplier != 1.0 or height_multiplier != 1.0:
            new_width = int(img.width * width_multiplier)
            new_height = int(img.height * height_multiplier)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"[INFO] Resized to {new_width}x{new_height}")

        # Convert to RGB if necessary (for PDF compatibility)
        if img.mode in ('RGBA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            img = background

        # Save PNG
        img.save(output_path, 'PNG', dpi=(300, 300))
        print(f"[OK] PNG saved: {output_path}")

        return True

    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch from mermaid.ink: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")
        return False


def convert_png_to_pdf(png_path: Path, pdf_path: Path) -> bool:
    """
    Convert PNG to PDF

    Args:
        png_path: Path to input PNG
        pdf_path: Path to output PDF

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(png_path)

        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            img = background

        # Save as PDF
        img.save(pdf_path, 'PDF', resolution=300.0)
        print(f"[OK] PDF saved: {pdf_path}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to convert to PDF: {e}")
        return False


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
    if not render_mermaid_to_png(MERMAID_CODE, png_path, scale=4):
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
