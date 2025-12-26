"""
Generate Training Pipeline Diagram from Mermaid code

Usage:
    python Overleaf_draft/src/generate_pipeline_diagram.py

Output:
    Overleaf_draft/figures/pipeline/training_pipeline.png
    Overleaf_draft/figures/pipeline/training_pipeline.pdf

Requirements:
    pip install requests Pillow

Note: Uses mermaid.ink API (no local installation needed)
"""

import base64
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

# Paths
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "pipeline"

# Mermaid code (Version 5 - BOLD BLACK borders, large font)
MERMAID_CODE = """
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '24px', 'fontFamily': 'Arial Black, Helvetica, sans-serif', 'primaryTextColor': '#000000'}}}%%
    flowchart LR
        classDef base fill:#E6E6FA,stroke:#000,stroke-width:6px,color:#000,font-size:24px
        classDef sft fill:#FFE4B5,stroke:#000,stroke-width:6px,color:#000,font-size:24px
        classDef align fill:#ADD8E6,stroke:#000,stroke-width:6px,color:#000,font-size:24px
        classDef cita fill:#90EE90,stroke:#000,stroke-width:6px,color:#000,font-size:24px
        classDef online fill:#FFB6C1,stroke:#000,stroke-width:6px,color:#000,font-size:24px

        A@{ shape: rect, label: "🦙 Llama-3.1-8B<br/>(Pretrained)", w: 500 }
        B@{ shape: rect, label: "📚 SFT<br/>Loss: L_SFT<br/>Data: PKU chosen", w: 500 }
        C@{ shape: rect, label: "🎯 PPO<br/>Loss: L_PPO<br/>Online + Reward Model", w: 500 }
        D@{ shape: rect, label: "⚖️ DPO<br/>Loss: L_DPO<br/>Offline Preference Pairs", w: 500 }
        E@{ shape: rect, label: "🔄 GRPO<br/>Loss: L_GRPO<br/>Online + Reward Functions", w: 500 }
        F@{ shape: rect, label: "🌟 CITA<br/>Loss: L_DPO + λ·L_KL<br/>Instruction-Conditioned<br/>+ Mandatory KL", w: 900}

        class A base;
        class B sft;
        class C online;
        class D align;
        class E online;
        class F cita;

        A ==> B
        B ==> C
        B ==> D
        B ==> E
        D ==> F

        linkStyle 0 stroke:#000,stroke-width:6px
        linkStyle 1 stroke:#000,stroke-width:6px
        linkStyle 2 stroke:#000,stroke-width:6px
        linkStyle 3 stroke:#000,stroke-width:6px
        linkStyle 4 stroke:#000,stroke-width:6px
"""


def render_mermaid_to_png(mermaid_code: str, output_path: Path, scale: int = 3) -> bool:
    """
    Render Mermaid diagram to PNG using mermaid.ink API

    Args:
        mermaid_code: Mermaid diagram code
        output_path: Path to save PNG
        scale: Scale factor for resolution (1-5)

    Returns:
        True if successful, False otherwise
    """
    # Encode mermaid code to base64
    encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')

    # Use mermaid.ink API
    # Options: ?type=png, ?bgColor=white, ?theme=default
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=white"

    print(f"Fetching diagram from mermaid.ink...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Load image and optionally scale up for higher resolution
        img = Image.open(BytesIO(response.content))

        if scale > 1:
            new_size = (img.width * scale, img.height * scale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

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
    print("GENERATE TRAINING PIPELINE DIAGRAM")
    print("=" * 60)

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}")

    # Output paths
    png_path = FIGURES_DIR / "training_pipeline.png"
    pdf_path = FIGURES_DIR / "training_pipeline.pdf"

    # Render to PNG (scale=3 for high resolution)
    print("\n[1/2] Rendering Mermaid to PNG...")
    if not render_mermaid_to_png(MERMAID_CODE, png_path, scale=3):
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
    print(r"  \includegraphics[width=\columnwidth]{figures/pipeline/training_pipeline.pdf}")


if __name__ == "__main__":
    main()
