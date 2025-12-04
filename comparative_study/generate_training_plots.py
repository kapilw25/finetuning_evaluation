"""
Generate Publication-Quality Training Plots from TensorBoard Logs

Generates 5 PDFs total:
- DPO vs CITA (4 models): accuracy, loss, margins = 3 PDFs
- SFT (2 models): loss, mean_token_accuracy = 2 PDFs

Usage:
    python comparative_study/generate_training_plots.py

Output:
    Overleaf_draft/figures/training/
    ├── dpo_cita_accuracy.{pdf,png}
    ├── dpo_cita_loss.{pdf,png}
    ├── dpo_cita_margins.{pdf,png}
    ├── sft_loss.{pdf,png}
    └── sft_mean_token_accuracy.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

from tensorboard.backend.event_processing import event_accumulator

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TB_LOGS = PROJECT_ROOT / "tensorboard_logs"
OUTPUT_DIR = PROJECT_ROOT / "Overleaf_draft" / "figures" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (publication friendly)
COLORS = {
    'CITA_Instruct': '#1f77b4',      # blue
    'CITA_NoInstruct': '#2ca02c',    # green
    'DPO_Instruct': '#d62728',       # red
    'DPO_NoInstruct': '#ff7f0e',     # orange
    'SFT_Instruct': '#9467bd',       # purple
    'SFT_NoInstruct': '#8c564b',     # brown
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
    'mean_token_accuracy': ['eval/mean_token_accuracy', 'train/mean_token_accuracy'],
}

# Event file mappings
EVENT_FILES = {
    'CITA_Instruct': 'CITA_Instruct_Adaptive_trial_7',
    'CITA_NoInstruct': 'CITA_NoInstruct_20251116_015238',
    'DPO_Instruct': 'DPO_Instruct_20251116_035213',
    'DPO_NoInstruct': 'DPO_NoInstruct_20251115_234037',
    'SFT_Instruct': 'SFT_Instruct_20251115_223957',
    'SFT_NoInstruct': 'SFT_NoInstruct_20251115_212216',
}


def load_tensorboard_data(log_dir: Path) -> dict:
    """Load scalar data from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={event_accumulator.SCALARS: 0}
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
    return None, None


def plot_single_metric(
    runs: dict,
    metric_name: str,
    title: str,
    output_path: Path,
    ylabel: str = None,
    figsize: tuple = (6, 4),
):
    """Generate a single publication-quality plot."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    has_data = False
    for label, (steps, values) in runs.items():
        if steps is None or values is None:
            print(f"  [WARN] No data for {label}")
            continue

        color = COLORS.get(label, '#333333')
        ls = LINESTYLES.get(label, '-')
        ax.plot(steps, values, label=label, color=color, linestyle=ls, linewidth=1.5)
        has_data = True

    if not has_data:
        print(f"  [SKIP] No data for {output_path.name}")
        plt.close(fig)
        return False

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel(ylabel or metric_name.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    # Save both PDF and PNG
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {output_path.stem}.{{pdf,png}}")
    return True


def generate_dpo_cita_plots():
    """
    DPO vs CITA comparison (4 models): 3 PDFs
    Models: CITA_Instruct, CITA_NoInstruct, DPO_Instruct, DPO_NoInstruct
    Metrics: accuracy, loss, margins
    """
    print("\n" + "="*60)
    print("DPO vs CITA (4 models) - 3 PDFs")
    print("="*60)

    models = ['CITA_Instruct', 'CITA_NoInstruct', 'DPO_Instruct', 'DPO_NoInstruct']
    runs_data = {}

    for model in models:
        log_dir = TB_LOGS / EVENT_FILES[model]
        if log_dir.exists():
            runs_data[model] = load_tensorboard_data(log_dir)
            print(f"  Loaded: {model}")
        else:
            print(f"  [MISS] {log_dir}")

    metrics = {
        'accuracy': ('Reward Accuracy', 'DPO vs CITA: Reward Accuracy'),
        'loss': ('Eval Loss', 'DPO vs CITA: Eval Loss'),
        'margins': ('Reward Margin', 'DPO vs CITA: Reward Margins'),
    }

    count = 0
    for metric, (ylabel, title) in metrics.items():
        runs = {}
        for label, data in runs_data.items():
            steps, values = find_metric_data(data, metric)
            runs[label] = (steps, values)

        if plot_single_metric(
            runs=runs,
            metric_name=metric,
            title=title,
            output_path=OUTPUT_DIR / f"dpo_cita_{metric}.pdf",
            ylabel=ylabel,
        ):
            count += 1

    return count


def generate_sft_plots():
    """
    SFT comparison (2 models): 2 PDFs
    Models: SFT_Instruct, SFT_NoInstruct
    Metrics: loss, mean_token_accuracy
    """
    print("\n" + "="*60)
    print("SFT (2 models) - 2 PDFs")
    print("="*60)

    models = ['SFT_Instruct', 'SFT_NoInstruct']
    runs_data = {}

    for model in models:
        log_dir = TB_LOGS / EVENT_FILES[model]
        if log_dir.exists():
            runs_data[model] = load_tensorboard_data(log_dir)
            print(f"  Loaded: {model}")
        else:
            print(f"  [MISS] {log_dir}")

    metrics = {
        'loss': ('Training Loss', 'SFT: Training Loss'),
        'mean_token_accuracy': ('Mean Token Accuracy', 'SFT: Mean Token Accuracy'),
    }

    count = 0
    for metric, (ylabel, title) in metrics.items():
        runs = {}
        for label, data in runs_data.items():
            steps, values = find_metric_data(data, metric)
            runs[label] = (steps, values)

        if plot_single_metric(
            runs=runs,
            metric_name=metric,
            title=title,
            output_path=OUTPUT_DIR / f"sft_{metric}.pdf",
            ylabel=ylabel,
        ):
            count += 1

    return count


def main():
    print("="*60)
    print("GENERATING 5 TRAINING PLOTS (PDF + PNG)")
    print("="*60)
    print(f"TensorBoard logs: {TB_LOGS}")
    print(f"Output: {OUTPUT_DIR}")

    total = 0
    total += generate_dpo_cita_plots()
    total += generate_sft_plots()

    print("\n" + "="*60)
    print(f"DONE: {total}/5 plots generated (PDF + PNG each)")
    print("="*60)
    print("PDFs:")
    for pdf in sorted(OUTPUT_DIR.glob("*.pdf")):
        print(f"  - {pdf.name}")
    print("PNGs:")
    for png in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {png.name}")


if __name__ == "__main__":
    main()
