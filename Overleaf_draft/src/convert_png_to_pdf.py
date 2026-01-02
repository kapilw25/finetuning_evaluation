"""
Convert PNG to PDF with optional padding.

Usage:
    python Overleaf_draft/src/convert_png_to_pdf.py

Output:
    Overleaf_draft/version_Professor/figures/pipeline/training_pipeline.pdf
"""

from pathlib import Path
from PIL import Image

# Paths
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent / "version_Professor" / "figures" / "pipeline"

# Settings
PADDING = 50  # pixels on all sides
DPI = 300


def convert_png_to_pdf(png_path: Path, pdf_path: Path, padding: int = 50) -> bool:
    """
    Convert PNG to PDF with padding.

    Args:
        png_path: Path to input PNG
        pdf_path: Path to output PDF
        padding: Pixels of white padding on all sides

    Returns:
        True if successful
    """
    try:
        img = Image.open(png_path)
        print(f"Original size: {img.size}")

        # Convert to RGB if needed (PDF doesn't support RGBA)
        if img.mode in ('RGBA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                bg.paste(img, mask=img.split()[3])
            else:
                bg.paste(img)
            img = bg

        # Add padding
        if padding > 0:
            new_width = img.width + padding * 2
            new_height = img.height + padding * 2
            padded = Image.new('RGB', (new_width, new_height), (255, 255, 255))
            padded.paste(img, (padding, padding))
            img = padded
            print(f"Padded size: {img.size} (+{padding}px on all sides)")

        # Save as PDF
        img.save(pdf_path, 'PDF', resolution=DPI)
        print(f"[OK] PDF saved: {pdf_path} ({DPI} DPI)")
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    print("=" * 50)
    print("CONVERT PNG TO PDF")
    print("=" * 50)

    png_path = FIGURES_DIR / "training_pipeline.png"
    pdf_path = FIGURES_DIR / "training_pipeline.pdf"

    if not png_path.exists():
        print(f"[ERROR] PNG not found: {png_path}")
        return

    convert_png_to_pdf(png_path, pdf_path, padding=PADDING)

    print("=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
