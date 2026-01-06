"""
Generate CITA Training Pipeline Diagram from Mermaid code

Usage:
    python Overleaf_draft/src/system_design_diagrams/generate_cita_pipeline.py

Output:
    Overleaf_draft/version_Professor/figures/pipeline/cita_pipeline.png
    Overleaf_draft/version_Professor/figures/pipeline/cita_pipeline.pdf

Requirements:
    pip install requests Pillow

Note: Uses mermaid.ink API (no local installation needed)

CITA Pipeline Architecture (comprehensive from 3 files):

FROM Llama3_BF16.py:
- INPUT: DPO checkpoint (merged LoRA) + PKU-SafeRLHF
- Uses CITATrainer with fixed Optuna-tuned HPs
- Training: gradient clipping (max_grad_norm=1.0)
- Output: CITA_NoInstruct (Trial 5 HPs), CITA_Instruct (Trial 7 HPs)

FROM Llama3_BF16_adaptive_Optuna.py:
- Multi-objective optimization: [margin, accuracy, -eval_loss]
- TPE Sampler + Hyperband Pruner for early stopping
- Conditional HP space: Instruct uses 50% lower LR (longer sequences)
- HP search: lambda_kl, learning_rate, beta, weight_decay, warmup_ratio

FROM cita_trainer.py:
- Inherits from DPOTrainer (apple-to-apple comparison)
- Unified Loss: L_CITA = λ_DPO × L_DPO + λ_KL × L_KL
- NO L_SFT! (causes catastrophic interference on DPO-tuned models)
- KL anchor: L_KL = mean(log π_θ - log π_ref) for chosen+rejected
- Gradient explosion detection (grad_norm > 50 → prune trial)
- Manual gradient clipping in training_step
"""

import base64
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

# Paths
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent.parent / "version_Professor" / "figures" / "pipeline"

# CITA Pipeline Mermaid Code (Left-to-Right layout for full A4 width)
# Comprehensive: covers training script, Optuna search, and CITATrainer internals
MERMAID_CODE = """
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '52px', 'fontFamily': 'Arial Black, Helvetica, sans-serif', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'arrowheadColor': '#000000'}, 'flowchart': {'subGraphTitleMargin': {'top': 30, 'bottom': 30}, 'padding': 40, 'nodeSpacing': 60, 'rankSpacing': 80, 'curve': 'linear', 'arrowMarkerAbsolute': true}}}%%
flowchart LR
    subgraph INPUT["<b>INPUT</b>"]
        direction TB
        DPO["<b>DPO Checkpoint<br/>(merged LoRA)</b>"]
        DPO ~~~ PKU
        PKU["<b>PKU-SafeRLHF<br/>12,035 samples</b>"]
    end

    subgraph DATA["<b>DATA</b>"]
        direction TB
        FILTER["<b>Clear Contrast<br/>safe != unsafe</b>"]
        FILTER ~~~ EXTRACT
        EXTRACT["<b>Extract Y+, Y-<br/>harm_categories</b>"]
        EXTRACT ~~~ PAIRS
        PAIRS["<b>Preference Pairs<br/>(chosen, rejected)</b>"]
    end

    subgraph MODEL["<b>MODEL</b>"]
        direction TB
        POLICY["<b>Policy π_θ<br/>LoRA r=16</b>"]
        POLICY ~~~ REF
        REF["<b>Reference π_ref<br/>(frozen DPO)</b>"]
    end

    subgraph OPTUNA["<b>HP TUNING</b>"]
        direction TB
        TPE["<b>TPE Sampler<br/>+ Hyperband</b>"]
        TPE ~~~ MULTI
        MULTI["<b>Multi-Objective<br/>margin,acc,-loss</b>"]
        MULTI ~~~ HPS
        HPS["<b>λ_KL, LR, β<br/>wd, warmup</b>"]
    end

    subgraph TRAIN["<b>UNIFIED LOSS</b>"]
        direction TB
        LOSS["<b>L = λ_DPO×L_DPO<br/>+ λ_KL×L_KL</b>"]
        LOSS ~~~ KL
        KL["<b>KL Anchor<br/>trust region</b>"]
        KL ~~~ CLIP
        CLIP["<b>Grad Clip<br/>norm ≤ 1.0</b>"]
    end

    subgraph OUTPUT["<b>OUTPUT</b>"]
        direction TB
        CITA_NI["<b>CITA_NoInstruct<br/>Trial 5 HPs</b>"]
        CITA_NI ~~~ CITA_I
        CITA_I["<b>CITA_Instruct<br/>Trial 7 HPs</b>"]
    end

    INPUT --> DATA
    DATA --> MODEL
    DATA --> OPTUNA
    MODEL --> TRAIN
    OPTUNA --> TRAIN
    TRAIN --> OUTPUT

    classDef inputStyle fill:#E6E6FA,stroke:#000000,stroke-width:4px,color:#000000
    classDef dataStyle fill:#FFE4B5,stroke:#000000,stroke-width:4px,color:#000000
    classDef modelStyle fill:#ADD8E6,stroke:#000000,stroke-width:4px,color:#000000
    classDef optunaStyle fill:#FFB6C1,stroke:#000000,stroke-width:4px,color:#000000
    classDef trainStyle fill:#FAFAD2,stroke:#000000,stroke-width:4px,color:#000000
    classDef outputStyle fill:#90EE90,stroke:#000000,stroke-width:4px,color:#000000
    classDef default stroke:#000000,stroke-width:3px,color:#000000
    linkStyle default stroke:#000000,stroke-width:6px

    class INPUT inputStyle
    class DATA dataStyle
    class MODEL modelStyle
    class OPTUNA optunaStyle
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
    print("GENERATE CITA PIPELINE DIAGRAM (COMPREHENSIVE)")
    print("=" * 60)
    print("Sources:")
    print("  - Llama3_BF16.py (training script)")
    print("  - Llama3_BF16_adaptive_Optuna.py (HP search)")
    print("  - cita_trainer.py (unified loss)")
    print("=" * 60)

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}")

    # Output paths
    png_path = FIGURES_DIR / "cita_pipeline.png"
    pdf_path = FIGURES_DIR / "cita_pipeline.pdf"

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
    print(r"  \includegraphics[width=\textwidth]{figures/pipeline/cita_pipeline.pdf}")


if __name__ == "__main__":
    main()
