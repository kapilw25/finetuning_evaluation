"""
Generate Publication-Quality Training Plots from TensorBoard Logs

Model: Llama-3.1-8B (meta-llama/Llama-3.1-8B)

Generates dual versions:
- Primary (no SFT): Policy optimization methods only (DPO, CITA, GRPO, PPO)
- Secondary (_withSFT): Includes SFT for completeness

Usage:
    python comparative_study/generate_training_plots.py

Output:
    Overleaf_draft/figures/training/
    ├── combined_eval_loss.{pdf,png}        # DPO, CITA, GRPO (no SFT)
    ├── combined_eval_loss_withSFT.{pdf,png} # includes SFT
    ├── combined_accuracy.{pdf,png}          # DPO/CITA only
    ├── combined_accuracy_withSFT.{pdf,png}  # includes SFT
    ├── dpo_cita_accuracy.{pdf,png}
    ├── dpo_cita_loss.{pdf,png}
    └── dpo_cita_margins.{pdf,png}
"""

# Model identifier for figure titles
MODEL_NAME = "Llama-3.1-8B"

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'bold'  # ALL text bold globally
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

from tensorboard.backend.event_processing import event_accumulator

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TB_LOGS = PROJECT_ROOT / "tensorboard_logs"
OUTPUT_DIR = PROJECT_ROOT / "Overleaf_draft" / "figures" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (publication friendly)
# Same color per method, differentiated by line style (solid=Instruct, dashed=NoInstruct)
METHOD_COLORS = {
    'CITA': '#00008B',      # dark blue
    'DPO': '#006400',       # dark green
    'PPO': '#4B0082',       # indigo
    'GRPO': '#E07000',      # burnt orange
    'SFT': '#8B0000',       # dark red
}

def get_color(model_name: str) -> str:
    """Get color for model - same color for Instruct/NoInstruct variants."""
    for method in METHOD_COLORS:
        if model_name.startswith(method):
            return METHOD_COLORS[method]
    return '#333333'  # fallback gray

def get_linestyle(model_name: str) -> str:
    """Get line style - solid for Instruct, dashed for NoInstruct."""
    return '--' if 'NoInstruct' in model_name else '-'

# Legacy dicts for backward compatibility (maps to functions above)
COLORS = {f"{m}_{s}": METHOD_COLORS[m] for m in METHOD_COLORS for s in ['Instruct', 'NoInstruct']}
LINESTYLES = {f"{m}_Instruct": '-' for m in METHOD_COLORS}
LINESTYLES.update({f"{m}_NoInstruct": '--' for m in METHOD_COLORS})

# Metric tag mappings (TensorBoard tags)
METRIC_TAGS = {
    'accuracy': ['eval/rewards/accuracies', 'eval/accuracy', 'train/rewards/accuracies'],
    'loss': ['eval/loss', 'train/loss'],
    'margins': ['eval/rewards/margins', 'train/rewards/margins'],
    'mean_token_accuracy': ['eval/mean_token_accuracy', 'train/mean_token_accuracy'],
    # GRPO-specific metrics (eval only)
    'grpo_loss': ['eval/loss'],
    'grpo_reward': ['eval/reward'],
    'grpo_entropy': ['eval/entropy'],
}

# Event file mappings
EVENT_FILES = {
    'CITA_Instruct': 'CITA_Instruct_Adaptive_trial_7',
    'CITA_NoInstruct': 'CITA_NoInstruct_20251116_015238',
    'DPO_Instruct': 'DPO_Instruct_20251116_035213',
    'DPO_NoInstruct': 'DPO_NoInstruct_20251115_234037',
    'PPO_Instruct': 'PPO_Instruct_20251220_225226',
    'PPO_NoInstruct': 'PPO_NoInstruct_20251220_062208',
    'GRPO_Instruct': 'GRPO_Instruct_20251220_201022',
    'GRPO_NoInstruct': 'GRPO_NoInstruct_20251220_082628',
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
    figsize: tuple = (8, 6),  # Square-ish aspect ratio (scale in LaTeX with width=\columnwidth)
) -> list:
    """Generate a single publication-quality plot. Returns list of generated file paths."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    has_data = False
    for label, (steps, values) in runs.items():
        if steps is None or values is None:
            print(f"  [WARN] No data for {label}")
            continue

        color = COLORS.get(label, '#333333')
        ls = LINESTYLES.get(label, '-')
        ax.plot(steps, values, label=label, color=color, linestyle=ls, linewidth=4)  # BOLD
        has_data = True

    if not has_data:
        print(f"  [SKIP] No data for {output_path.name}")
        plt.close(fig)
        return []

    ax.set_xlabel('Training Steps', fontsize=24, fontweight='bold')
    ax.set_ylabel(ylabel or metric_name.capitalize(), fontsize=24, fontweight='bold')
    ax.set_title(title, fontsize=26, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)

    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=18)
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
    return [pdf_path, png_path]


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
        'accuracy': ('Reward Accuracy', f'DPO vs CITA: Reward Accuracy ({MODEL_NAME})'),
        'loss': ('Eval Loss', f'DPO vs CITA: Eval Loss ({MODEL_NAME})'),
        'margins': ('Reward Margin', f'DPO vs CITA: Reward Margins ({MODEL_NAME})'),
    }

    generated_files = []
    for metric, (ylabel, title) in metrics.items():
        runs = {}
        for label, data in runs_data.items():
            steps, values = find_metric_data(data, metric)
            runs[label] = (steps, values)

        files = plot_single_metric(
            runs=runs,
            metric_name=metric,
            title=title,
            output_path=OUTPUT_DIR / f"dpo_cita_{metric}",
            ylabel=ylabel,
        )
        generated_files.extend(files)

    return generated_files


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
        'loss': ('Training Loss', f'SFT: Training Loss ({MODEL_NAME})'),
        'mean_token_accuracy': ('Mean Token Accuracy', f'SFT: Mean Token Accuracy ({MODEL_NAME})'),
    }

    generated_files = []
    for metric, (ylabel, title) in metrics.items():
        runs = {}
        for label, data in runs_data.items():
            steps, values = find_metric_data(data, metric)
            runs[label] = (steps, values)

        files = plot_single_metric(
            runs=runs,
            metric_name=metric,
            title=title,
            output_path=OUTPUT_DIR / f"sft_{metric}",
            ylabel=ylabel,
        )
        generated_files.extend(files)

    return generated_files


def generate_grpo_plots():
    """
    GRPO comparison (2 models): 3 PDFs
    Models: GRPO_Instruct, GRPO_NoInstruct
    Metrics: loss, reward, entropy (eval only)
    """
    print("\n" + "="*60)
    print("GRPO (2 models) - 3 PDFs")
    print("="*60)

    models = ['GRPO_Instruct', 'GRPO_NoInstruct']
    runs_data = {}

    for model in models:
        log_dir = TB_LOGS / EVENT_FILES[model]
        if log_dir.exists():
            runs_data[model] = load_tensorboard_data(log_dir)
            print(f"  Loaded: {model}")
        else:
            print(f"  [MISS] {log_dir}")

    metrics = {
        'grpo_loss': ('Eval Loss', f'GRPO: Eval Loss ({MODEL_NAME})'),
        'grpo_reward': ('Eval Reward', f'GRPO: Eval Reward ({MODEL_NAME})'),
        'grpo_entropy': ('Eval Entropy', f'GRPO: Eval Entropy ({MODEL_NAME})'),
    }

    generated_files = []
    for metric, (ylabel, title) in metrics.items():
        runs = {}
        for label, data in runs_data.items():
            steps, values = find_metric_data(data, metric)
            runs[label] = (steps, values)

        files = plot_single_metric(
            runs=runs,
            metric_name=metric,
            title=title,
            output_path=OUTPUT_DIR / f"{metric}",
            ylabel=ylabel,
        )
        generated_files.extend(files)

    return generated_files


def generate_combined_loss_subplot():
    """
    Combined eval loss subplots - generates TWO versions:
    1. combined_eval_loss.{pdf,png} - Policy optimization methods only (DPO, CITA, GRPO) - 1x3
    2. combined_eval_loss_withSFT.{pdf,png} - Includes SFT (2x2)
    """
    print("\n" + "="*60)
    print("COMBINED EVAL LOSS (dual versions)")
    print("="*60)

    # Load all trainer data
    all_trainers = [
        'SFT_Instruct', 'SFT_NoInstruct',
        'DPO_Instruct', 'DPO_NoInstruct',
        'CITA_Instruct', 'CITA_NoInstruct',
        'GRPO_Instruct', 'GRPO_NoInstruct',
    ]

    runs_data = {}
    for model in all_trainers:
        log_dir = TB_LOGS / EVENT_FILES[model]
        if log_dir.exists():
            runs_data[model] = load_tensorboard_data(log_dir)
            print(f"  Loaded: {model}")
        else:
            print(f"  [MISS] {log_dir}")

    generated_files = []

    # =========================================================================
    # VERSION 1: WITHOUT SFT (1x3 layout) - Policy optimization methods only
    # =========================================================================
    subplot_config_no_sft = [
        {'models': ['DPO_Instruct', 'DPO_NoInstruct'], 'title': 'DPO'},
        {'models': ['CITA_Instruct', 'CITA_NoInstruct'], 'title': 'CITA'},
        {'models': ['GRPO_Instruct', 'GRPO_NoInstruct'], 'title': 'GRPO'},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Eval Loss: Policy Optimization Methods ({MODEL_NAME})', fontsize=28, fontweight='bold', y=0.98)

    for idx, config in enumerate(subplot_config_no_sft):
        ax = axes[idx]
        models = config['models']
        title = config['title']

        for model in models:
            if model not in runs_data:
                continue
            data = runs_data[model]
            steps, values = None, None
            for tag in ['eval/loss', 'train/loss']:
                if tag in data:
                    steps = data[tag]['steps']
                    values = data[tag]['values']
                    break
            if steps and values:
                color = COLORS.get(model, '#808080')
                linestyle = LINESTYLES.get(model, '-')
                label = model.replace('_', ' ')
                ax.plot(steps, values, color=color, linestyle=linestyle, linewidth=4, label=label)

        ax.set_title(title, fontsize=24, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=20, fontweight='bold')
        ax.set_ylabel('Eval Loss', fontsize=20, fontweight='bold')
        ax.legend(loc='best', fontsize=14, frameon=True, fancybox=False, edgecolor='black')
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = OUTPUT_DIR / "combined_eval_loss"
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  [OK] combined_eval_loss.{pdf,png} (no SFT)")
    generated_files.extend([pdf_path, png_path])

    # =========================================================================
    # VERSION 2: WITH SFT (2x2 layout) - All training stages
    # =========================================================================
    subplot_config_with_sft = {
        (0, 0): {'models': ['SFT_Instruct', 'SFT_NoInstruct'], 'title': 'SFT'},
        (0, 1): {'models': ['DPO_Instruct', 'DPO_NoInstruct'], 'title': 'DPO'},
        (1, 0): {'models': ['CITA_Instruct', 'CITA_NoInstruct'], 'title': 'CITA'},
        (1, 1): {'models': ['GRPO_Instruct', 'GRPO_NoInstruct'], 'title': 'GRPO'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Eval Loss: All Training Stages ({MODEL_NAME})', fontsize=32, fontweight='bold', y=0.98)

    for (row, col), config in subplot_config_with_sft.items():
        ax = axes[row, col]
        models = config['models']
        title = config['title']

        for model in models:
            if model not in runs_data:
                continue
            data = runs_data[model]
            steps, values = None, None
            for tag in ['eval/loss', 'train/loss']:
                if tag in data:
                    steps = data[tag]['steps']
                    values = data[tag]['values']
                    break
            if steps and values:
                color = COLORS.get(model, '#808080')
                linestyle = LINESTYLES.get(model, '-')
                label = model.replace('_', ' ')
                ax.plot(steps, values, color=color, linestyle=linestyle, linewidth=4, label=label)

        ax.set_title(title, fontsize=28, fontweight='bold')
        ax.set_ylabel('Eval Loss', fontsize=22, fontweight='bold')
        ax.legend(loc='best', fontsize=18, frameon=True, fancybox=False, edgecolor='black')
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[1, 0].set_xlabel('Training Steps', fontsize=22, fontweight='bold')
    axes[1, 1].set_xlabel('Training Steps', fontsize=22, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = OUTPUT_DIR / "combined_eval_loss_withSFT"
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  [OK] combined_eval_loss_withSFT.{pdf,png}")
    generated_files.extend([pdf_path, png_path])

    return generated_files


def generate_combined_accuracy_subplot():
    """
    Combined accuracy subplots - generates TWO versions:
    1. combined_accuracy.{pdf,png} - DPO/CITA reward accuracy only (single plot)
    2. combined_accuracy_withSFT.{pdf,png} - 1x2 with SFT token accuracy
    """
    print("\n" + "="*60)
    print("COMBINED ACCURACY (dual versions)")
    print("="*60)

    # Load all required data
    all_models = [
        'DPO_Instruct', 'DPO_NoInstruct',
        'CITA_Instruct', 'CITA_NoInstruct',
        'SFT_Instruct', 'SFT_NoInstruct',
    ]

    runs_data = {}
    for model in all_models:
        log_dir = TB_LOGS / EVENT_FILES[model]
        if log_dir.exists():
            runs_data[model] = load_tensorboard_data(log_dir)
            print(f"  Loaded: {model}")
        else:
            print(f"  [MISS] {log_dir}")

    generated_files = []

    # =========================================================================
    # VERSION 1: WITHOUT SFT - DPO/CITA Reward Accuracy only (single plot)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_facecolor('white')

    dpo_cita_models = ['DPO_Instruct', 'DPO_NoInstruct', 'CITA_Instruct', 'CITA_NoInstruct']
    for model in dpo_cita_models:
        if model not in runs_data:
            continue
        data = runs_data[model]
        steps, values = find_metric_data(data, 'accuracy')
        if steps and values:
            color = COLORS.get(model, '#808080')
            linestyle = LINESTYLES.get(model, '-')
            label = model.replace('_', ' ')
            ax.plot(steps, values, color=color, linestyle=linestyle, linewidth=4, label=label)

    ax.set_title(f'DPO / CITA: Reward Accuracy ({MODEL_NAME})', fontsize=26, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=22, fontweight='bold')
    ax.set_ylabel('Reward Accuracy', fontsize=22, fontweight='bold')
    ax.legend(loc='best', fontsize=16, frameon=True, fancybox=False, edgecolor='black')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "combined_accuracy"
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  [OK] combined_accuracy.{pdf,png} (no SFT)")
    generated_files.extend([pdf_path, png_path])

    # =========================================================================
    # VERSION 2: WITH SFT - 1x2 layout (DPO/CITA + SFT token accuracy)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Training Accuracy Metrics ({MODEL_NAME})', fontsize=32, fontweight='bold', y=0.98)

    # Left subplot: DPO/CITA Reward Accuracy
    ax_left = axes[0]
    ax_left.set_facecolor('white')

    for model in dpo_cita_models:
        if model not in runs_data:
            continue
        data = runs_data[model]
        steps, values = find_metric_data(data, 'accuracy')
        if steps and values:
            color = COLORS.get(model, '#808080')
            linestyle = LINESTYLES.get(model, '-')
            label = model.replace('_', ' ')
            ax_left.plot(steps, values, color=color, linestyle=linestyle, linewidth=4, label=label)

    ax_left.set_title('DPO / CITA: Reward Accuracy', fontsize=28, fontweight='bold')
    ax_left.set_xlabel('Training Steps', fontsize=22, fontweight='bold')
    ax_left.set_ylabel('Reward Accuracy', fontsize=22, fontweight='bold')
    ax_left.legend(loc='best', fontsize=16, frameon=True, fancybox=False, edgecolor='black')
    ax_left.tick_params(axis='both', labelsize=18)
    ax_left.grid(True, alpha=0.3, linestyle='--')
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)

    # Right subplot: SFT Mean Token Accuracy
    ax_right = axes[1]
    ax_right.set_facecolor('white')
    sft_models = ['SFT_Instruct', 'SFT_NoInstruct']

    for model in sft_models:
        if model not in runs_data:
            continue
        data = runs_data[model]
        steps, values = find_metric_data(data, 'mean_token_accuracy')
        if steps and values:
            color = COLORS.get(model, '#808080')
            linestyle = LINESTYLES.get(model, '-')
            label = model.replace('_', ' ')
            ax_right.plot(steps, values, color=color, linestyle=linestyle, linewidth=4, label=label)

    ax_right.set_title('SFT: Mean Token Accuracy', fontsize=28, fontweight='bold')
    ax_right.set_xlabel('Training Steps', fontsize=22, fontweight='bold')
    ax_right.set_ylabel('Mean Token Accuracy', fontsize=22, fontweight='bold')
    ax_right.legend(loc='best', fontsize=16, frameon=True, fancybox=False, edgecolor='black')
    ax_right.tick_params(axis='both', labelsize=18)
    ax_right.grid(True, alpha=0.3, linestyle='--')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = OUTPUT_DIR / "combined_accuracy_withSFT"
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  [OK] combined_accuracy_withSFT.{pdf,png}")
    generated_files.extend([pdf_path, png_path])

    return generated_files


def main():
    print("="*60)
    print("GENERATING TRAINING PLOTS (PDF + PNG)")
    print("="*60)
    print(f"TensorBoard logs: {TB_LOGS}")
    print(f"Output: {OUTPUT_DIR}")

    generated_files = []
    generated_files.extend(generate_dpo_cita_plots())
    generated_files.extend(generate_sft_plots())
    generated_files.extend(generate_grpo_plots())
    generated_files.extend(generate_combined_loss_subplot())
    generated_files.extend(generate_combined_accuracy_subplot())

    # Separate PDFs and PNGs
    pdfs = sorted([f for f in generated_files if f.suffix == '.pdf'])
    pngs = sorted([f for f in generated_files if f.suffix == '.png'])

    print("\n" + "="*60)
    print(f"DONE: {len(pdfs)} plots generated (PDF + PNG each)")
    print("="*60)
    print("PDFs:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")
    print("PNGs:")
    for png in pngs:
        print(f"  - {png.name}")


if __name__ == "__main__":
    main()
