"""
Shared utilities for Mermaid diagram generation.

Functions:
    - add_title_to_image: Add bold title above image
    - render_mermaid_to_png: Fetch from mermaid.ink API and save PNG
    - convert_png_to_pdf: Convert PNG to PDF
"""

import base64
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO


def add_title_to_image(img: Image.Image, title: str, font_size: int = 140) -> Image.Image:
    """
    Add a bold title above the image.

    Args:
        img: PIL Image
        title: Title text to add
        font_size: Font size for title (default 140 to match figure text at scale=4)

    Returns:
        New image with title added above
    """
    # Create drawing context to measure text
    draw = ImageDraw.Draw(img)

    # Try bold fonts in order of preference
    font = None
    bold_fonts = [
        # macOS bold fonts
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        # Linux bold fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # Windows bold fonts
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial Bold.ttf",
    ]

    for font_path in bold_fonts:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue

    if font is None:
        font = ImageFont.load_default()

    # Measure text size
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Create new image with space for title
    title_padding = 60
    new_height = img.height + text_height + title_padding * 2
    new_img = Image.new('RGB', (img.width, new_height), (255, 255, 255))

    # Draw title centered at top - pure black (#000000) for dark bold appearance
    title_x = (img.width - text_width) // 2
    title_y = title_padding
    title_draw = ImageDraw.Draw(new_img)
    title_draw.text((title_x, title_y), title, fill=(0, 0, 0), font=font)

    # Paste original image below title
    new_img.paste(img, (0, text_height + title_padding * 2))

    return new_img


def render_mermaid_to_png(mermaid_code: str, output_path: Path, scale: int = 3,
                          width_multiplier: float = 1.0, height_multiplier: float = 1.0,
                          title: str = None) -> bool:
    """
    Render Mermaid diagram to PNG using mermaid.ink API

    Args:
        mermaid_code: Mermaid diagram code
        output_path: Path to save PNG
        scale: Scale factor for resolution (1-5)
        width_multiplier: Multiply final width by this factor
        height_multiplier: Multiply final height by this factor
        title: Optional title to add above the image

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

        # Add title if provided
        if title:
            img = add_title_to_image(img, title)
            print(f"[INFO] Added title: {title}")

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
