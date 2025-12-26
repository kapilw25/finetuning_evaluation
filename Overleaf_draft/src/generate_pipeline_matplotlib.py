"""
Generate Training Pipeline Diagram using Matplotlib

Usage:
    python Overleaf_draft/src/generate_pipeline_matplotlib.py

Output:
    Overleaf_draft/figures/pipeline/training_pipeline_matplotlib.png
    Overleaf_draft/figures/pipeline/training_pipeline_matplotlib.pdf

Requirements:
    pip install matplotlib

Note: Pure Python - full control over fonts, colors, line widths
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "pipeline"

# Box configurations
BOXES = {
    'base': {
        'text': ['Llama-3.1-8B', '(Pretrained)'],
        'color': '#E6E6FA',  # Lavender
        'pos': (0.5, 0.92),
        'size': (0.22, 0.12)
    },
    'sft': {
        'text': ['SFT', 'Loss: L_SFT', 'Data: PKU chosen'],
        'color': '#FFE4B5',  # Moccasin
        'pos': (0.5, 0.72),
        'size': (0.24, 0.14)
    },
    'ppo': {
        'text': ['PPO', 'Loss: L_PPO', 'Online + Reward Model'],
        'color': '#FFB6C1',  # Light pink
        'pos': (0.15, 0.47),
        'size': (0.26, 0.14)
    },
    'dpo': {
        'text': ['DPO', 'Loss: L_DPO', 'Offline Preference Pairs'],
        'color': '#ADD8E6',  # Light blue
        'pos': (0.5, 0.47),
        'size': (0.26, 0.14)
    },
    'grpo': {
        'text': ['GRPO', 'Loss: L_GRPO', 'Online + Reward Functions'],
        'color': '#FFB6C1',  # Light pink
        'pos': (0.85, 0.47),
        'size': (0.26, 0.14)
    },
    'cita': {
        'text': ['CITA', 'Loss: L_DPO + λ·L_KL', 'Instruction-Conditioned', '+ Mandatory KL'],
        'color': '#90EE90',  # Light green
        'pos': (0.5, 0.18),
        'size': (0.28, 0.18)
    }
}

# Arrow connections (from, to)
ARROWS = [
    ('base', 'sft'),
    ('sft', 'ppo'),
    ('sft', 'dpo'),
    ('sft', 'grpo'),
    ('dpo', 'cita')
]


def draw_box(ax, key, config):
    """Draw a single box with text"""
    x, y = config['pos']
    w, h = config['size']

    # Create rounded rectangle with BOLD BLACK border
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=config['color'],
        edgecolor='black',
        linewidth=4.0,  # BOLD BLACK border
        zorder=2
    )
    ax.add_patch(box)

    # Add text - BOLD and larger
    lines = config['text']
    n_lines = len(lines)
    line_height = 0.028
    start_y = y + (n_lines - 1) * line_height / 2

    for i, line in enumerate(lines):
        # First line (method name) is larger and bolder
        if i == 0:
            fontsize = 18
            fontweight = 'black'
        else:
            fontsize = 13
            fontweight = 'bold'

        ax.text(x, start_y - i * line_height, line,
                ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight,
                color='black', zorder=3)


def draw_arrow(ax, from_key, to_key):
    """Draw arrow between two boxes"""
    from_box = BOXES[from_key]
    to_box = BOXES[to_key]

    from_x, from_y = from_box['pos']
    to_x, to_y = to_box['pos']
    from_h = from_box['size'][1]
    to_h = to_box['size'][1]

    # Arrow from bottom of source to top of target
    start_y = from_y - from_h/2
    end_y = to_y + to_h/2

    # Draw arrow with BOLD BLACK line
    arrow = FancyArrowPatch(
        (from_x, start_y), (to_x, end_y),
        arrowstyle='-|>',
        mutation_scale=25,
        linewidth=4.0,  # BOLD
        color='black',  # BLACK
        zorder=1
    )
    ax.add_patch(arrow)


def generate_diagram(png_path: Path, pdf_path: Path):
    """Generate the training pipeline diagram"""

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw arrows first (behind boxes)
    for from_key, to_key in ARROWS:
        draw_arrow(ax, from_key, to_key)

    # Draw boxes
    for key, config in BOXES.items():
        draw_box(ax, key, config)

    # Tight layout
    plt.tight_layout()

    # Save PNG
    fig.savefig(png_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] PNG saved: {png_path}")

    # Save PDF
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] PDF saved: {pdf_path}")

    plt.close(fig)


def main():
    print("=" * 60)
    print("GENERATE TRAINING PIPELINE DIAGRAM (Matplotlib)")
    print("=" * 60)

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}")

    # Output paths - use _matplotlib suffix
    png_path = FIGURES_DIR / "training_pipeline_matplotlib.png"
    pdf_path = FIGURES_DIR / "training_pipeline_matplotlib.pdf"

    # Generate diagram
    print("\nGenerating diagram with BOLD BLACK borders...")
    generate_diagram(png_path, pdf_path)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print("\nTo include in LaTeX:")
    print(r"  \includegraphics[width=\columnwidth]{figures/pipeline/training_pipeline_matplotlib.pdf}")


if __name__ == "__main__":
    main()
