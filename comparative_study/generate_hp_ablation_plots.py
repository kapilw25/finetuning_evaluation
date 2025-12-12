"""
Generate Hyperparameter Ablation Plots from Optuna Trial Configs

Generates ablation plots showing sensitivity to:
- beta (DPO temperature)
- lambda_kl (KL regularization strength)
- learning_rate
- weight_decay
- warmup_ratio

Similar to ALKALI paper's hyperparameter ablation figures.

Usage:
    python comparative_study/generate_hp_ablation_plots.py

Output:
    Overleaf_draft/figures/appendix/
    ├── hp_ablation_beta.{pdf,png}
    ├── hp_ablation_lambda_kl.{pdf,png}
    ├── hp_ablation_learning_rate.{pdf,png}
    ├── hp_ablation_combined.{pdf,png}  (2x2 grid)
"""

import json
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRIAL_DIR = PROJECT_ROOT / "outputs" / "CITA_Instruct_Adaptive"
OUTPUT_DIR = PROJECT_ROOT / "Overleaf_draft" / "figures" / "appendix"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for metrics
METRIC_COLORS = {
    'margin': '#1f77b4',      # blue
    'accuracy': '#2ca02c',    # green
    'eval_loss': '#d62728',   # red
}


def load_all_trial_configs():
    """Load trial configs from all trial directories."""
    trials = []

    for trial_dir in sorted(TRIAL_DIR.glob("trial_*")):
        if trial_dir.name == "best_trial":
            continue

        config_path = trial_dir / "trial_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                trials.append(config)

    return trials


def plot_hp_vs_metric(
    trials: list,
    hp_name: str,
    hp_label: str,
    output_path: Path,
    best_trial_num: int = 7,
    figsize: tuple = (6, 4),
) -> list:
    """
    Plot hyperparameter vs metrics (margin, accuracy, eval_loss).

    Args:
        trials: List of trial config dicts
        hp_name: Key name for hyperparameter in config
        hp_label: Display label for x-axis
        output_path: Output file path (without extension)
        best_trial_num: Trial number to highlight as best
        figsize: Figure size

    Returns:
        List of generated file paths
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='white')

    # Extract data
    hp_values = [t[hp_name] for t in trials]
    margins = [t['final_margin'] for t in trials]
    accuracies = [t['final_accuracy'] * 100 for t in trials]  # Convert to %
    eval_losses = [t['final_eval_loss'] for t in trials]
    trial_nums = [t['trial_number'] for t in trials]

    # Find best trial index
    best_idx = None
    for i, t in enumerate(trials):
        if t['trial_number'] == best_trial_num:
            best_idx = i
            break

    metrics = [
        (margins, 'Reward Margin', 'margin', '↑ Higher is Better'),
        (accuracies, 'Accuracy (%)', 'accuracy', '↑ Higher is Better'),
        (eval_losses, 'Eval Loss', 'eval_loss', '↓ Lower is Better'),
    ]

    for ax, (values, ylabel, metric_key, annotation) in zip(axes, metrics):
        ax.set_facecolor('white')

        # Scatter plot
        color = METRIC_COLORS[metric_key]
        ax.scatter(hp_values, values, c=color, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Highlight best trial
        if best_idx is not None:
            ax.scatter([hp_values[best_idx]], [values[best_idx]],
                      c='gold', s=150, marker='*', edgecolors='black',
                      linewidth=1, zorder=5, label=f'Best (Trial {best_trial_num})')

        # Add trial numbers as annotations (small)
        for i, (x, y, tn) in enumerate(zip(hp_values, values, trial_nums)):
            ax.annotate(str(tn), (x, y), fontsize=7, alpha=0.6,
                       xytext=(3, 3), textcoords='offset points')

        # Add trend line (polynomial fit)
        if len(hp_values) > 2:
            try:
                z = np.polyfit(hp_values, values, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(hp_values), max(hp_values), 50)
                ax.plot(x_trend, p(x_trend), '--', color=color, alpha=0.5, linewidth=1.5)
            except Exception:
                pass  # Skip trend if fitting fails

        ax.set_xlabel(hp_label, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add annotation for direction
        ax.annotate(annotation, xy=(0.98, 0.02), xycoords='axes fraction',
                   fontsize=8, ha='right', va='bottom', style='italic', alpha=0.7)

        if best_idx is not None:
            ax.legend(loc='best', fontsize=9)

    fig.suptitle(f'CITA Hyperparameter Sensitivity: {hp_label}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save both PDF and PNG
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [OK] {output_path.stem}.{{pdf,png}}")
    return [pdf_path, png_path]


def plot_combined_ablation(trials: list, output_path: Path, best_trial_num: int = 7) -> list:
    """
    Generate combined 2x2 ablation plot (most important hyperparameters).

    Shows: beta, lambda_kl, learning_rate, weight_decay
    Metric: Reward Margin (primary optimization target)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='white')
    axes = axes.flatten()

    hps = [
        ('beta', 'β (DPO Temperature)'),
        ('lambda_kl', 'λ_KL (KL Regularization)'),
        ('learning_rate', 'Learning Rate'),
        ('weight_decay', 'Weight Decay'),
    ]

    # Extract data
    margins = [t['final_margin'] for t in trials]
    trial_nums = [t['trial_number'] for t in trials]

    # Find best trial index
    best_idx = None
    for i, t in enumerate(trials):
        if t['trial_number'] == best_trial_num:
            best_idx = i
            break

    for ax, (hp_name, hp_label) in zip(axes, hps):
        ax.set_facecolor('white')

        hp_values = [t[hp_name] for t in trials]

        # Scatter plot
        ax.scatter(hp_values, margins, c=METRIC_COLORS['margin'], s=60,
                  alpha=0.7, edgecolors='black', linewidth=0.5)

        # Highlight best trial
        if best_idx is not None:
            ax.scatter([hp_values[best_idx]], [margins[best_idx]],
                      c='gold', s=150, marker='*', edgecolors='black',
                      linewidth=1, zorder=5)

        # Add trial numbers
        for x, y, tn in zip(hp_values, margins, trial_nums):
            ax.annotate(str(tn), (x, y), fontsize=7, alpha=0.6,
                       xytext=(3, 3), textcoords='offset points')

        # Trend line
        if len(hp_values) > 2:
            try:
                z = np.polyfit(hp_values, margins, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(hp_values), max(hp_values), 50)
                ax.plot(x_trend, p(x_trend), '--', color=METRIC_COLORS['margin'],
                       alpha=0.5, linewidth=1.5)
            except Exception:
                pass

        ax.set_xlabel(hp_label, fontsize=10)
        ax.set_ylabel('Reward Margin', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('CITA Hyperparameter Ablation Study\n(Metric: Reward Margin, * = Best Trial)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    # Save
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [OK] {output_path.stem}.{{pdf,png}}")
    return [pdf_path, png_path]


def plot_pareto_frontier(trials: list, output_path: Path, best_trial_num: int = 7) -> list:
    """
    Generate Pareto frontier plot: Margin vs Accuracy trade-off.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.set_facecolor('white')

    margins = [t['final_margin'] for t in trials]
    accuracies = [t['final_accuracy'] * 100 for t in trials]
    trial_nums = [t['trial_number'] for t in trials]

    # Find best trial index
    best_idx = None
    for i, t in enumerate(trials):
        if t['trial_number'] == best_trial_num:
            best_idx = i
            break

    # Scatter plot
    ax.scatter(margins, accuracies, c='#1f77b4', s=80, alpha=0.7,
              edgecolors='black', linewidth=0.5, label='Trials')

    # Highlight best trial
    if best_idx is not None:
        ax.scatter([margins[best_idx]], [accuracies[best_idx]],
                  c='gold', s=200, marker='*', edgecolors='black',
                  linewidth=1, zorder=5, label=f'Best (Trial {best_trial_num})')

    # Add trial numbers
    for x, y, tn in zip(margins, accuracies, trial_nums):
        ax.annotate(str(tn), (x, y), fontsize=9, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')

    # Compute and draw Pareto frontier
    points = list(zip(margins, accuracies))
    pareto = []
    for i, (m, a) in enumerate(points):
        dominated = False
        for j, (m2, a2) in enumerate(points):
            if i != j and m2 >= m and a2 >= a and (m2 > m or a2 > a):
                dominated = True
                break
        if not dominated:
            pareto.append((m, a))

    if pareto:
        pareto = sorted(pareto, key=lambda x: x[0])
        pareto_m, pareto_a = zip(*pareto)
        ax.plot(pareto_m, pareto_a, 'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')

    ax.set_xlabel('Reward Margin (↑ Higher is Better)', fontsize=12)
    ax.set_ylabel('Accuracy % (↑ Higher is Better)', fontsize=12)
    ax.set_title('CITA Optuna Trials: Margin-Accuracy Trade-off', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [OK] {output_path.stem}.{{pdf,png}}")
    return [pdf_path, png_path]


def print_trial_summary(trials: list):
    """Print summary table of all trials."""
    print("\n" + "="*100)
    print("TRIAL SUMMARY")
    print("="*100)
    print(f"{'Trial':<6} {'beta':<8} {'λ_KL':<12} {'LR':<12} {'Margin':<10} {'Acc%':<8} {'Loss':<8}")
    print("-"*100)

    # Sort by margin descending
    sorted_trials = sorted(trials, key=lambda x: x['final_margin'], reverse=True)

    for t in sorted_trials:
        print(f"{t['trial_number']:<6} "
              f"{t['beta']:<8.4f} "
              f"{t['lambda_kl']:<12.6f} "
              f"{t['learning_rate']:<12.2e} "
              f"{t['final_margin']:<10.4f} "
              f"{t['final_accuracy']*100:<8.1f} "
              f"{t['final_eval_loss']:<8.4f}")

    print("="*100)


def main():
    print("="*60)
    print("GENERATING HYPERPARAMETER ABLATION PLOTS")
    print("="*60)
    print(f"Trial configs: {TRIAL_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Load trials
    trials = load_all_trial_configs()
    print(f"\nLoaded {len(trials)} trial configs")

    if not trials:
        print("[ERROR] No trial configs found!")
        return

    # Print summary
    print_trial_summary(trials)

    # Generate individual HP plots
    generated_files = []

    hp_configs = [
        ('beta', 'β (DPO Temperature)'),
        ('lambda_kl', 'λ_KL (KL Regularization)'),
        ('learning_rate', 'Learning Rate'),
    ]

    print("\n" + "="*60)
    print("Individual HP Ablation Plots")
    print("="*60)

    for hp_name, hp_label in hp_configs:
        files = plot_hp_vs_metric(
            trials=trials,
            hp_name=hp_name,
            hp_label=hp_label,
            output_path=OUTPUT_DIR / f"hp_ablation_{hp_name}",
            best_trial_num=7,
        )
        generated_files.extend(files)

    # Generate combined plot
    print("\n" + "="*60)
    print("Combined Ablation Plot (2x2)")
    print("="*60)

    files = plot_combined_ablation(
        trials=trials,
        output_path=OUTPUT_DIR / "hp_ablation_combined",
        best_trial_num=7,
    )
    generated_files.extend(files)

    # Generate Pareto frontier
    print("\n" + "="*60)
    print("Pareto Frontier Plot")
    print("="*60)

    files = plot_pareto_frontier(
        trials=trials,
        output_path=OUTPUT_DIR / "hp_pareto_frontier",
        best_trial_num=7,
    )
    generated_files.extend(files)

    # Summary
    pdfs = [f for f in generated_files if str(f).endswith('.pdf')]
    pngs = [f for f in generated_files if str(f).endswith('.png')]

    print("\n" + "="*60)
    print(f"DONE: {len(pdfs)} plots generated (PDF + PNG each)")
    print("="*60)
    print("Generated files:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")


if __name__ == "__main__":
    main()
