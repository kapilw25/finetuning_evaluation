"""HP ablation plots from Optuna trial configs (beta, lambda_kl, lr, wd, warmup). CPU-only.

    python -u src/eval/hp_ablation_plots.py 2>&1 | tee logs/hp_ablation_plots.log

Model: Llama-3.1-8B (meta-llama/Llama-3.1-8B). Similar to ALKALI paper figures.

Output:
    Overleaf_draft/figures/appendix/
    ├── hp_ablation_beta.{pdf,png}
    ├── hp_ablation_lambda_kl.{pdf,png}
    ├── hp_ablation_learning_rate.{pdf,png}
    ├── hp_ablation_combined.{pdf,png}  (2x2 grid)
"""

# Model identifier for figure titles
MODEL_NAME = "Llama-3.1-8B"

from pathlib import Path
import numpy as np
import optuna
from adjustText import adjust_text

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'bold'  # ALL text bold globally
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OPTUNA_DB_INSTRUCT = PROJECT_ROOT / "outputs" / "training" / "optuna_cita_instruct.db"
OPTUNA_DB_NOINSTRUCT = PROJECT_ROOT / "outputs" / "training" / "optuna_cita.db"
OUTPUT_DIR = PROJECT_ROOT / "Overleaf_draft" / "figures" / "appendix"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for metrics
METRIC_COLORS = {
    'margin': '#1f77b4',      # blue
    'accuracy': '#2ca02c',    # green
    'eval_loss': '#d62728',   # red
}


def load_all_trial_configs(db_path: Path = None):
    """Load trial configs from Optuna database.

    Args:
        db_path: Path to Optuna SQLite database. Defaults to OPTUNA_DB_INSTRUCT.

    Returns:
        List of trial config dicts with hyperparameters and metrics.
    """
    if db_path is None:
        db_path = OPTUNA_DB_INSTRUCT

    if not db_path.exists():
        print(f"  [WARN] Database not found: {db_path}")
        return []

    # Load Optuna study
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=None, storage=storage)
    except Exception as e:
        print(f"  [ERROR] Failed to load study: {e}")
        return []

    trials = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        # Extract hyperparameters
        config = {
            'trial_number': trial.number,
            'lambda_kl': trial.params.get('lambda_kl', 0),
            'learning_rate': trial.params.get('learning_rate', 0),
            'beta': trial.params.get('beta', 0),
            'weight_decay': trial.params.get('weight_decay', 0),
            'warmup_ratio': trial.params.get('warmup_ratio', 0),
        }

        # Extract metrics (values[0]=margin, values[1]=accuracy, values[2]=-loss)
        if trial.values and len(trial.values) >= 3:
            config['final_margin'] = trial.values[0]
            config['final_accuracy'] = trial.values[1]
            config['final_eval_loss'] = abs(trial.values[2])  # Stored as negative
        else:
            # Single objective (margin only)
            config['final_margin'] = trial.value if trial.value else 0
            config['final_accuracy'] = 0
            config['final_eval_loss'] = 0

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
    # 2x2 layout with larger figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')
    axes = axes.flatten()

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
        (margins, 'Reward Margin', 'margin', '(↑ Higher is Better)'),
        (accuracies, 'Accuracy (%)', 'accuracy', '(↑ Higher is Better)'),
        (eval_losses, 'Eval Loss', 'eval_loss', '(↓ Lower is Better)'),
    ]

    # Scale small values and include multiplier in label
    if hp_name == 'lambda_kl':
        hp_values_plot = [v * 10000 for v in hp_values]  # Scale to 1-6 range
        xlabel_display = f'{hp_label} (×10⁻⁴)'
    elif hp_name == 'learning_rate':
        hp_values_plot = [v * 1000000 for v in hp_values]  # Scale to 2.6-5.4 range
        xlabel_display = f'{hp_label} (×10⁻⁶)'
    else:
        hp_values_plot = hp_values
        xlabel_display = hp_label

    for idx, (ax, (values, ylabel, metric_key, direction)) in enumerate(zip(axes[:3], metrics)):
        ax.set_facecolor('white')

        # Scatter plot
        color = METRIC_COLORS[metric_key]
        ax.scatter(hp_values_plot, values, c=color, s=200, alpha=0.7, edgecolors='black', linewidth=2)

        # Highlight best trial
        if best_idx is not None:
            ax.scatter([hp_values_plot[best_idx]], [values[best_idx]],
                      c='gold', s=500, marker='*', edgecolors='black',
                      linewidth=3, zorder=5, label=f'Best (Trial {best_trial_num})')

        # Use adjustText for annotations
        texts = []
        for i, (x, y, tn) in enumerate(zip(hp_values_plot, values, trial_nums)):
            texts.append(ax.text(x, y, str(tn), fontsize=18, fontweight='bold', alpha=0.8))

        # Auto-adjust text positions to avoid overlap
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

        # Add trend line
        if len(hp_values_plot) > 2:
            try:
                z = np.polyfit(hp_values_plot, values, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(hp_values_plot), max(hp_values_plot), 50)
                ax.plot(x_trend, p(x_trend), '--', color=color, alpha=0.5, linewidth=4)
            except Exception:
                pass  # Skip trend if fitting fails

        ax.set_xlabel(xlabel_display, fontsize=22, fontweight='bold')
        # Include direction in ylabel to free up plot area
        ax.set_ylabel(f'{ylabel} {direction}', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if best_idx is not None:
            ax.legend(loc='best', fontsize=16)

    # Hide the 4th subplot (empty in 2x2 grid with 3 metrics)
    axes[3].axis('off')

    fig.suptitle(f'CITA Hyperparameter Sensitivity: {hp_label} ({MODEL_NAME})', fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

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
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')
    axes = axes.flatten()

    hps = [
        ('beta', 'β (DPO Temperature)'),
        ('lambda_kl', 'λ_KL (KL Regularization)'),
        ('learning_rate', 'Learning Rate'),
        ('weight_decay', 'Weight Decay'),
    ]

    # Extract data - use Reward Margin
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

        # Scale small values and include multiplier in label
        if hp_name == 'lambda_kl':
            hp_values_plot = [v * 10000 for v in hp_values]
            xlabel_display = f'{hp_label} (×10⁻⁴)'
        elif hp_name == 'learning_rate':
            hp_values_plot = [v * 1000000 for v in hp_values]
            xlabel_display = f'{hp_label} (×10⁻⁶)'
        else:
            hp_values_plot = hp_values
            xlabel_display = hp_label

        # Scatter plot
        ax.scatter(hp_values_plot, margins, c=METRIC_COLORS['margin'], s=200,
                  alpha=0.7, edgecolors='black', linewidth=2)

        # Highlight best trial
        if best_idx is not None:
            ax.scatter([hp_values_plot[best_idx]], [margins[best_idx]],
                      c='gold', s=500, marker='*', edgecolors='black',
                      linewidth=3, zorder=5, label=f'Best (Trial {best_trial_num})')

        # Use adjustText for annotations
        texts = []
        for x, y, tn in zip(hp_values_plot, margins, trial_nums):
            texts.append(ax.text(x, y, str(tn), fontsize=18, fontweight='bold', alpha=0.8))

        # Auto-adjust text positions to avoid overlap
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

        # Trend line
        if len(hp_values_plot) > 2:
            try:
                z = np.polyfit(hp_values_plot, margins, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(hp_values_plot), max(hp_values_plot), 50)
                ax.plot(x_trend, p(x_trend), '--', color=METRIC_COLORS['margin'],
                       alpha=0.5, linewidth=4)
            except Exception:
                pass

        ax.set_xlabel(xlabel_display, fontsize=22, fontweight='bold')
        ax.set_ylabel('Reward Margin', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if best_idx is not None:
            ax.legend(loc='best', fontsize=14)

    fig.suptitle(f'CITA Hyperparameter Ablation Study ({MODEL_NAME})\n(Metric: Reward Margin, Best = Trial {best_trial_num})',
                fontsize=26, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

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
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
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

    # Scatter plot (DOUBLED: larger markers)
    ax.scatter(margins, accuracies, c='#1f77b4', s=320, alpha=0.7,
              edgecolors='black', linewidth=2, label='Trials')

    # Highlight best trial (DOUBLED: larger star)
    if best_idx is not None:
        ax.scatter([margins[best_idx]], [accuracies[best_idx]],
                  c='gold', s=800, marker='*', edgecolors='black',
                  linewidth=4, zorder=5, label=f'Best (Trial {best_trial_num})')

    # Use adjustText for annotations
    texts = []
    for x, y, tn in zip(margins, accuracies, trial_nums):
        texts.append(ax.text(x, y, str(tn), fontsize=28, fontweight='bold', alpha=0.8))

    # Auto-adjust text positions to avoid overlap
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

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
        ax.plot(pareto_m, pareto_a, 'r--', linewidth=8, alpha=0.7, label='Pareto Frontier')

    ax.set_xlabel('Reward Margin (↑ Higher is Better)', fontsize=36, fontweight='bold')
    ax.set_ylabel('Accuracy % (↑ Higher is Better)', fontsize=36, fontweight='bold')
    ax.set_title(f'CITA Optuna Trials: Margin-Accuracy Trade-off ({MODEL_NAME})', fontsize=40, fontweight='bold')
    ax.tick_params(axis='both', labelsize=28)
    ax.legend(loc='lower right', fontsize=28)
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
    print(f"Optuna DB (Instruct): {OPTUNA_DB_INSTRUCT}")
    print(f"Output: {OUTPUT_DIR}")

    # Load trials from Optuna database
    trials = load_all_trial_configs(OPTUNA_DB_INSTRUCT)
    print(f"\nLoaded {len(trials)} completed trials")

    if not trials:
        print("[ERROR] No trials found in database!")
        return

    # Print summary
    print_trial_summary(trials)

    # Multi-objective selection: Find Pareto-optimal trials, then pick best balanced one
    def is_pareto_dominated(trial_a, trial_b):
        """Returns True if trial_b dominates trial_a (b is better on ALL objectives)"""
        margin_better = trial_b['final_margin'] >= trial_a['final_margin']
        acc_better = trial_b['final_accuracy'] >= trial_a['final_accuracy']
        loss_better = trial_b['final_eval_loss'] <= trial_a['final_eval_loss']
        strictly_better = (
            trial_b['final_margin'] > trial_a['final_margin'] or
            trial_b['final_accuracy'] > trial_a['final_accuracy'] or
            trial_b['final_eval_loss'] < trial_a['final_eval_loss']
        )
        return margin_better and acc_better and loss_better and strictly_better

    # Find Pareto-optimal trials
    pareto_trials = []
    for t in trials:
        dominated = False
        for other in trials:
            if t['trial_number'] != other['trial_number'] and is_pareto_dominated(t, other):
                dominated = True
                break
        if not dominated:
            pareto_trials.append(t)

    print(f"\nPareto-optimal trials: {[t['trial_number'] for t in pareto_trials]}")

    # Select best from Pareto front: balanced score = margin × accuracy / loss
    def balanced_score(t):
        return t['final_margin'] * t['final_accuracy'] / (t['final_eval_loss'] + 0.001)

    best_trial = max(pareto_trials, key=balanced_score)
    best_trial_num = best_trial['trial_number']
    print(f"Best trial (balanced): #{best_trial_num} (margin={best_trial['final_margin']:.4f}, acc={best_trial['final_accuracy']*100:.1f}%, loss={best_trial['final_eval_loss']:.4f})")

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
            best_trial_num=best_trial_num,
        )
        generated_files.extend(files)

    # Generate combined plot
    print("\n" + "="*60)
    print("Combined Ablation Plot (2x2)")
    print("="*60)

    files = plot_combined_ablation(
        trials=trials,
        output_path=OUTPUT_DIR / "hp_ablation_combined",
        best_trial_num=best_trial_num,
    )
    generated_files.extend(files)

    # Generate Pareto frontier
    print("\n" + "="*60)
    print("Pareto Frontier Plot")
    print("="*60)

    files = plot_pareto_frontier(
        trials=trials,
        output_path=OUTPUT_DIR / "hp_pareto_frontier",
        best_trial_num=best_trial_num,
    )
    generated_files.extend(files)

    # Summary
    pdfs = [f for f in generated_files if str(f).endswith('.pdf')]
    pngs = [f for f in generated_files if str(f).endswith('.png')]  # noqa: F841

    print("\n" + "="*60)
    print(f"DONE: {len(pdfs)} plots generated (PDF + PNG each)")
    print("="*60)
    print("Generated files:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")


if __name__ == "__main__":
    main()
