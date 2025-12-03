"""
Generate Publication-Quality Training Plots from TensorBoard Logs

Converts TensorBoard logs into individual PDFs with white background.
Each metric (accuracy, loss, margins) saved as separate PDF.

Usage:
    python comparative_study/05_evaluation/generate_training_plots.py

Output:
    Overleaf_draft/figures/training/pdf/
    ├── fig2_accuracy.pdf
    ├── fig2_loss.pdf
    ├── fig2_margins.pdf
    ├── fig3_accuracy.pdf
    ├── fig3_loss.pdf
    ├── fig3_margins.pdf
    ├── sft_accuracy.pdf
    ├── sft_loss.pdf
    └── sft_margins.pdf
"""

import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

from tensorboard.backend.event_processing import event_accumulator

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TB_LOGS = PROJECT_ROOT / "tensorboard_logs"
OUTPUT_DIR = PROJECT_ROOT / "Overleaf_draft" / "figures" / "training" / "pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (publication friendly)
COLORS = {
    'CITA_Instruct': '#1f77b4',      # blue
    'CITA_NoInstruct': '#2ca02c',    # green
    'DPO_Instruct': '#d62728',       # red
    'DPO_NoInstruct': '#ff7f0e',     # orange
    'SFT_Instruct': '#9467bd',       # purple
    'SFT_NoInstruct': '#8c564b',     # brown
    'trial_2': '#17becf',            # cyan
    'trial_7': '#bcbd22',            # olive
}

LINESTYLES = {
    'CITA_Instruct': '-',
    'CITA_NoInstruct': '--',
    'DPO_Instruct': '-.',
    'DPO_NoInstruct': ':',
    'SFT_Instruct': '-',
    'SFT_NoInstruct': '--',
}

# Metric tag mappings (TensorBoard tags)
METRIC_TAGS = {
    'accuracy': ['eval/rewards/accuracies', 'eval/accuracy', 'train/rewards/accuracies'],
    'loss': ['eval/loss', 'train/loss'],
    'margins': ['eval/rewards/margins', 'train/rewards/margins'],
}


def load_tensorboard_data(log_dir: Path) -> dict:
    """Load scalar data from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={event_accumulator.SCALARS: 0}  # Load all
    )
    ea.Reload()

    data = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
        }
    return data


def find_metric_data(data: dict, metric_type: str) -> tuple:
    """Find the appropriate metric data from loaded TensorBoard data."""
    for tag in METRIC_TAGS.get(metric_type, []):
        if tag in data:
            return data[tag]['steps'], data[tag]['values']

    # Fallback: search for partial match
    for tag, values in data.items():
        if metric_type.lower() in tag.lower():
            return values['steps'], values['values']

    return None, None


def plot_single_metric(
    runs: dict,  # {label: (steps, values)}
    metric_name: str,
    title: str,
    output_path: Path,
    ylabel: str = None,
    figsize: tuple = (6, 4),
):
    """Generate a single publication-quality plot."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    for label, (steps, values) in runs.items():
        if steps is None or values is None:
            print(f"  ⚠️  No data for {label}")
            continue

        # Determine color and linestyle
        color = COLORS.get(label, '#333333')
        ls = LINESTYLES.get(label, '-')

        # Check for trial variants
        for key in COLORS:
            if key in label:
                color = COLORS[key]
                break

        ax.plot(steps, values, label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel(ylabel or metric_name.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved: {output_path}")


def generate_figure2_plots():
    """
    Figure 2: CITA_Instruct (best trials) vs DPO_Instruct vs CITA_NoInstruct
    """
    print("\n" + "="*60)
    print("Generating Figure 2 plots...")
    print("="*60)

    # Load runs
    runs_data = {}

    # CITA_Instruct trials
    trial_dirs = [
        ("CITA_Instruct_Adaptive_trial_2", "CITA_I (trial 2)"),
        ("CITA_Instruct_Adaptive_trial_7", "CITA_I (trial 7)"),
        ("CITA_Instruct_20251118_031257", "CITA_I (final)"),
    ]

    for dirname, label in trial_dirs:
        log_dir = TB_LOGS / dirname
        if log_dir.exists():
            runs_data[label] = load_tensorboard_data(log_dir)
            print(f"  Loaded: {dirname}")

    # DPO_Instruct
    dpo_dir = TB_LOGS / "DPO_Instruct_20251116_035213"
    if dpo_dir.exists():
        runs_data["DPO_Instruct"] = load_tensorboard_data(dpo_dir)
        print(f"  Loaded: DPO_Instruct")

    # CITA_NoInstruct
    cita_ni_dir = TB_LOGS / "CITA_NoInstruct_20251116_015238"
    if cita_ni_dir.exists():
        runs_data["CITA_NoInstruct"] = load_tensorboard_data(cita_ni_dir)
        print(f"  Loaded: CITA_NoInstruct")

    # Generate plots for each metric
    for metric in ['accuracy', 'loss', 'margins']:
        runs = {}
        for label, data in runs_data.items():
            steps, values = find_metric_data(data, metric)
            runs[label] = (steps, values)

        ylabel_map = {
            'accuracy': 'Reward Accuracy',
            'loss': 'Eval Loss',
            'margins': 'Reward Margin',
        }

        plot_single_metric(
            runs=runs,
            metric_name=metric,
            title=f"CITA Instruct Training: {metric.capitalize()}",
            output_path=OUTPUT_DIR / f"fig2_{metric}.pdf",
            ylabel=ylabel_map.get(metric, metric.capitalize()),
        )


def generate_figure3_plots():
    """
    Figure 3: DPO_NoInstruct vs CITA_NoInstruct
    """
    print("\n" + "="*60)
    print("Generating Figure 3 plots...")
    print("="*60)

    runs_data = {}

    # DPO_NoInstruct
    dpo_dir = TB_LOGS / "DPO_NoInstruct_20251115_234037"
    if dpo_dir.exists():
        runs_data["DPO_NoInstruct"] = load_tensorboard_data(dpo_dir)
        print(f"  Loaded: DPO_NoInstruct")

    # CITA_NoInstruct
    cita_dir = TB_LOGS / "CITA_NoInstruct_20251116_015238"
    if cita_dir.exists():
        runs_data["CITA_NoInstruct"] = load_tensorboard_data(cita_dir)
        print(f"  Loaded: CITA_NoInstruct")

    for metric in ['accuracy', 'loss', 'margins']:
        runs = {}
        for label, data in runs_data.items():
            steps, values = find_metric_data(data, metric)
            runs[label] = (steps, values)

        ylabel_map = {
            'accuracy': 'Reward Accuracy',
            'loss': 'Eval Loss',
            'margins': 'Reward Margin',
        }

        plot_single_metric(
            runs=runs,
            metric_name=metric,
            title=f"DPO vs CITA (NoInstruct): {metric.capitalize()}",
            output_path=OUTPUT_DIR / f"fig3_{metric}.pdf",
            ylabel=ylabel_map.get(metric, metric.capitalize()),
        )


def generate_sft_plots():
    """
    SFT comparison: SFT_NoInstruct vs SFT_Instruct
    """
    print("\n" + "="*60)
    print("Generating SFT plots...")
    print("="*60)

    runs_data = {}

    # SFT_NoInstruct
    sft_ni_dir = TB_LOGS / "SFT_NoInstruct_20251115_212216"
    if sft_ni_dir.exists():
        runs_data["SFT_NoInstruct"] = load_tensorboard_data(sft_ni_dir)
        print(f"  Loaded: SFT_NoInstruct")

    # SFT_Instruct
    sft_i_dir = TB_LOGS / "SFT_Instruct_20251115_223957"
    if sft_i_dir.exists():
        runs_data["SFT_Instruct"] = load_tensorboard_data(sft_i_dir)
        print(f"  Loaded: SFT_Instruct")

    for metric in ['accuracy', 'loss', 'margins']:
        runs = {}
        for label, data in runs_data.items():
            steps, values = find_metric_data(data, metric)
            runs[label] = (steps, values)

        ylabel_map = {
            'accuracy': 'Accuracy',
            'loss': 'Training Loss',
            'margins': 'Margin',
        }

        plot_single_metric(
            runs=runs,
            metric_name=metric,
            title=f"SFT Training: {metric.capitalize()}",
            output_path=OUTPUT_DIR / f"sft_{metric}.pdf",
            ylabel=ylabel_map.get(metric, metric.capitalize()),
        )


def main():
    print("="*60)
    print("GENERATING PUBLICATION-QUALITY TRAINING PLOTS")
    print("="*60)
    print(f"TensorBoard logs: {TB_LOGS}")
    print(f"Output directory: {OUTPUT_DIR}")

    # List available logs
    print("\nAvailable TensorBoard logs:")
    for d in sorted(TB_LOGS.iterdir()):
        if d.is_dir() and not d.name.startswith('archive'):
            print(f"  - {d.name}")

    generate_figure2_plots()
    generate_figure3_plots()
    generate_sft_plots()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    pdfs = list(OUTPUT_DIR.glob("*.pdf"))
    print(f"Generated {len(pdfs)} PDFs:")
    for pdf in sorted(pdfs):
        print(f"  - {pdf.name}")


if __name__ == "__main__":
    main()
