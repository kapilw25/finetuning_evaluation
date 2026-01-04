"""
Plotting Utilities for Evaluation Scripts

Shared color mapping and legend elements for consistent plots.
Includes improvement ratio visualization for NoInstruct → Instruct comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'bold'  # ALL text bold globally
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
from matplotlib.patches import Patch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from .model_loader import MODEL_NAME


def save_figure_dual_format(fig, output_path: Path, dpi: int = 300) -> Tuple[str, str]:
    """
    Save figure in both PDF (for Overleaf) and PNG (for sharing) formats.

    Args:
        fig: Matplotlib figure
        output_path: Base output path (will save .pdf and .png)
        dpi: Resolution for PNG

    Returns:
        Tuple of (pdf_path, png_path)
    """
    output_path = Path(output_path)

    # Save PNG
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')

    # Save PDF
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight', facecolor='white')

    return str(pdf_path), str(png_path)


def get_model_color(model_name: str) -> str:
    """
    Get consistent color for model based on name

    Color scheme:
    - SFT: Red tones (dark for Instruct, light for NoInstruct)
    - DPO: Green tones
    - PPO: Purple tones
    - GRPO: Orange tones
    - CITA: Blue tones
    - Baseline: Gray

    Args:
        model_name: Model name/key

    Returns:
        Hex color string
    """
    if 'SFT' in model_name:
        return '#8B0000' if 'Instruct' in model_name else '#FF6B6B'
    elif 'DPO' in model_name:
        return '#006400' if 'Instruct' in model_name else '#90EE90'
    elif 'PPO' in model_name:
        return '#4B0082' if 'Instruct' in model_name else '#DDA0DD'  # Indigo / Plum
    elif 'GRPO' in model_name:
        return '#FF8C00' if 'Instruct' in model_name else '#FFDAB9'  # DarkOrange / PeachPuff
    elif 'CITA' in model_name:
        return '#00008B' if 'Instruct' in model_name else '#87CEEB'
    elif 'Baseline' in model_name:
        return '#808080'
    else:
        return '#FFA500'


def get_model_colors(model_names: List[str]) -> List[str]:
    """
    Get colors for list of models

    Args:
        model_names: List of model names

    Returns:
        List of hex color strings
    """
    return [get_model_color(m) for m in model_names]


def get_legend_elements(include_baseline: bool = False) -> List[Patch]:
    """
    Get standard legend elements for model comparison plots

    Args:
        include_baseline: Whether to include Baseline in legend

    Returns:
        List of Patch elements for legend
    """
    elements = [
        Patch(facecolor='#8B0000', edgecolor='black', label='SFT_Instruct'),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='SFT_NoInstruct'),
        Patch(facecolor='#006400', edgecolor='black', label='DPO_Instruct'),
        Patch(facecolor='#90EE90', edgecolor='black', label='DPO_NoInstruct'),
        Patch(facecolor='#4B0082', edgecolor='black', label='PPO_Instruct'),
        Patch(facecolor='#DDA0DD', edgecolor='black', label='PPO_NoInstruct'),
        Patch(facecolor='#FF8C00', edgecolor='black', label='GRPO_Instruct'),
        Patch(facecolor='#FFDAB9', edgecolor='black', label='GRPO_NoInstruct'),
        Patch(facecolor='#00008B', edgecolor='black', label='CITA_Instruct'),
        Patch(facecolor='#87CEEB', edgecolor='black', label='CITA_NoInstruct'),
    ]

    if include_baseline:
        elements.insert(0, Patch(facecolor='#808080', edgecolor='black', label='Baseline'))

    return elements


def add_figure_legend(fig, models: List[str], ncol: int = 4, fontsize: int = 10):
    """
    Add standard legend to figure

    Args:
        fig: Matplotlib figure
        models: List of model names (to check for Baseline)
        ncol: Number of columns in legend
        fontsize: Legend font size
    """
    include_baseline = any('Baseline' in m for m in models)
    legend_elements = get_legend_elements(include_baseline)

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05)
    )


# =============================================================================
# SHARED COMPARISON PLOT (Single bar per model)
# =============================================================================

def generate_comparison_plots(
    models: List[str],
    overall_scores: List[float],
    valid_scores: List[float] = None,  # Deprecated, kept for backward compatibility
    valid_rates: List[float] = None,   # Deprecated, kept for backward compatibility
    output_dir: Path = None,
    plot_filename: str = None,
    ylabel: str = "Score",
    title: str = "Model Comparison",
    perfect_score: float = 1.0,
    perfect_label: str = "Perfect = 1.0",
    ylim_max: Optional[float] = None,
    ylim_min: float = 0,
    reference_line: Optional[float] = None,
    reference_label: str = "Reference",
    score_format: str = ".3f",
    higher_is_better: bool = True,
    error_bars: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[str, str]:
    """
    Shared comparison plot generator for all evaluation scripts.

    Creates a clean bar chart showing overall scores only.

    Args:
        models: List of model names
        overall_scores: Overall scores for each model
        valid_scores: DEPRECATED - ignored (kept for backward compatibility)
        valid_rates: DEPRECATED - ignored (kept for backward compatibility)
        output_dir: Directory to save plot
        plot_filename: Filename without extension (e.g., "isd_comparison")
        ylabel: Y-axis label
        title: Plot title
        perfect_score: Perfect score value for annotation
        perfect_label: Label for perfect score annotation
        ylim_max: Maximum y-axis limit (auto if None)
        ylim_min: Minimum y-axis limit (default 0)
        reference_line: Y value for horizontal reference line (optional)
        reference_label: Label for reference line
        score_format: Format string for score labels (e.g., ".3f", ".2f", ".1f")
        higher_is_better: If True, sort ascending (best on right)
        error_bars: Optional dict mapping model name to (ci_lower, ci_upper) tuple
            for 95% confidence interval. If provided, error whiskers are added.

    Returns:
        Tuple of (pdf_path, png_path)
    """
    if len(models) < 2:
        print("Need at least 2 models for comparison plots")
        return None, None

    # Filter out SFT models (SFT is a training stage, not a policy optimization method)
    filtered_indices = [i for i, m in enumerate(models) if not m.startswith('SFT')]
    if len(filtered_indices) < len(models):
        print(f"  [INFO] Filtered out SFT models ({len(models) - len(filtered_indices)} removed)")
        models = [models[i] for i in filtered_indices]
        overall_scores = [overall_scores[i] for i in filtered_indices]
        if error_bars:
            error_bars = {k: v for k, v in error_bars.items() if not k.startswith('SFT')}

    if len(models) < 2:
        print("Need at least 2 models after filtering")
        return None, None

    # Sort by overall score (ascending = best on right for higher_is_better)
    sorted_indices = np.argsort(overall_scores)
    if not higher_is_better:
        sorted_indices = sorted_indices[::-1]

    models_sorted = [models[i] for i in sorted_indices]
    overall_sorted = [overall_scores[i] for i in sorted_indices]

    # Get colors using shared utility
    colors_sorted = get_model_colors(models_sorted)

    # Create figure - slightly smaller since only one bar per model
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models_sorted))
    bar_width = 0.3  # Reduced by 50%

    # Single bars for overall scores
    bars = ax.bar(x, overall_sorted, bar_width,
                  color=colors_sorted, edgecolor='black', linewidth=2.0)

    # Add error bars if provided (Bootstrap CI whiskers)
    if error_bars is not None:
        yerr_lower = []
        yerr_upper = []
        for model in models_sorted:
            if model in error_bars:
                ci_lower, ci_upper = error_bars[model]
                score = overall_scores[models.index(model)]
                # Clamp to non-negative (matplotlib requires yerr >= 0)
                yerr_lower.append(max(0, score - ci_lower))
                yerr_upper.append(max(0, ci_upper - score))
            else:
                yerr_lower.append(0)
                yerr_upper.append(0)

        # Only add error bars if we have valid values
        if any(y > 0 for y in yerr_lower + yerr_upper):
            ax.errorbar(x, overall_sorted, yerr=[yerr_lower, yerr_upper],
                        fmt='none', ecolor='black', elinewidth=2, capsize=5, capthick=2)

    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='black')
    # Remove "Overall vs Valid-Only" from title if present
    clean_title = title.replace(" - Overall vs Valid-Only", "").replace("Overall vs Valid-Only", "")
    # Auto-append MODEL_NAME to title if not already present
    if MODEL_NAME and MODEL_NAME not in clean_title:
        clean_title = f"{clean_title} ({MODEL_NAME})"
    ax.set_title(clean_title, fontsize=14, fontweight='bold', color='black', pad=15)

    # Set y-axis limits
    if ylim_max is None:
        ylim_max = max(overall_sorted) * 1.3 if overall_sorted else 1.0

    # Handle negative values
    if min(overall_sorted) < 0:
        y_margin = max(abs(min(overall_sorted)), abs(max(overall_sorted))) * 0.3
        ax.set_ylim(min(overall_sorted) - y_margin, max(overall_sorted) + y_margin)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    else:
        ax.set_ylim(ylim_min, ylim_max)

    # Add reference line if specified
    if reference_line is not None:
        ax.axhline(y=reference_line, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=reference_label)

    # Add Perfect score annotation
    ax.text(0.98, 0.98, perfect_label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', color='black', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=90, ha='center', fontsize=11, fontweight='bold', color='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, score in zip(bars, overall_sorted):
        if score >= 0:
            y_offset = bar.get_height() + (ylim_max - ylim_min) * 0.01
            va = 'bottom'
        else:
            y_offset = bar.get_height() - (ylim_max - ylim_min) * 0.01
            va = 'top'

        ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                f'{score:{score_format}}', ha='center', va=va,
                fontsize=10, fontweight='bold', color='black')

    plt.tight_layout()
    plot_path = output_dir / plot_filename
    pdf_path, png_path = save_figure_dual_format(fig, plot_path, dpi=300)
    plt.close(fig)

    print(f"Saved plot:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")

    # Print ranking
    direction = "Best to Worst" if higher_is_better else "Worst to Best"
    print(f"\nRanking ({direction}):")
    for rank, (model, score) in enumerate(zip(reversed(models_sorted), reversed(overall_sorted)), 1):
        print(f"   {rank}. {model}: {score:{score_format}}")

    return pdf_path, png_path


# =============================================================================
# BOX AND VIOLIN PLOTS (Per-sample distribution visualization)
# =============================================================================

def generate_boxviolin_chart(
    data_by_model: Dict[str, List[float]],
    output_dir: Path,
    plot_filename: str,
    ylabel: str = "Score",
    title: str = "Distribution Comparison",
    plot_type: str = "both",
    higher_is_better: bool = True,
    show_points: bool = False,
    reference_lines: Optional[List[Tuple[float, str, str]]] = None
) -> Dict[str, Tuple[str, str]]:
    """
    Generate box plot and/or violin plot for per-sample score distributions.

    Uses consistent color scheme: SFT=red, DPO=green, CITA=blue.

    Args:
        data_by_model: Dict mapping model names to lists of per-sample scores
            e.g., {'SFT_Instruct': [0.5, 0.6, ...], 'CITA_Instruct': [...]}
        output_dir: Directory to save plots
        plot_filename: Base filename (will add _box.pdf, _violin.pdf)
        ylabel: Y-axis label
        title: Plot title
        plot_type: "box", "violin", or "both"
        higher_is_better: If True, sort models by median (best on right)
        show_points: If True, overlay individual points on box/violin
        reference_lines: List of (y_value, label, color) for horizontal lines

    Returns:
        Dict mapping plot_type to (pdf_path, png_path)
    """
    import seaborn as sns

    if len(data_by_model) < 2:
        print("Need at least 2 models for box/violin plots")
        return {}

    # Build DataFrame for seaborn
    plot_data = []
    for model, scores in data_by_model.items():
        for score in scores:
            plot_data.append({'Model': model, 'Score': score})

    df = pd.DataFrame(plot_data)

    # Sort models by median score
    medians = df.groupby('Model')['Score'].median()
    if higher_is_better:
        sorted_models = medians.sort_values().index.tolist()  # Ascending (best on right)
    else:
        sorted_models = medians.sort_values(ascending=False).index.tolist()

    # Get colors using shared color scheme
    palette = {model: get_model_color(model) for model in sorted_models}

    results = {}

    # Generate Box Plot
    if plot_type in ["box", "both"]:
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(
            data=df,
            x='Model',
            y='Score',
            order=sorted_models,
            palette=palette,
            ax=ax,
            linewidth=2.0,
            fliersize=3,
            width=0.3  # Reduced by 50%
        )

        if show_points:
            sns.stripplot(
                data=df,
                x='Model',
                y='Score',
                order=sorted_models,
                color='black',
                alpha=0.3,
                size=2,
                ax=ax
            )

        # Add reference lines
        if reference_lines:
            for y_val, label, color in reference_lines:
                ax.axhline(y=y_val, color=color, linestyle='--', linewidth=1.5,
                           alpha=0.7, label=label)
            ax.legend(loc='upper right', fontsize=9, prop={'weight': 'bold'})

        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='black')
        # Auto-append MODEL_NAME to title if not already present
        box_title = f'{title} (Box Plot)'
        if MODEL_NAME and MODEL_NAME not in title:
            box_title = f'{title} ({MODEL_NAME}) (Box Plot)'
        ax.set_title(box_title, fontsize=14, fontweight='bold', color='black', pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=11, fontweight='bold', color='black')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        box_path = output_dir / f"{plot_filename}_box"
        pdf_path, png_path = save_figure_dual_format(fig, box_path, dpi=300)
        plt.close(fig)

        results['box'] = (pdf_path, png_path)
        print(f"Saved box plot:")
        print(f"  PDF: {pdf_path}")
        print(f"  PNG: {png_path}")

    # Generate Violin Plot
    if plot_type in ["violin", "both"]:
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.violinplot(
            data=df,
            x='Model',
            y='Score',
            order=sorted_models,
            palette=palette,
            ax=ax,
            linewidth=1.5,
            inner='quartile',
            cut=0
        )

        # Add reference lines
        if reference_lines:
            for y_val, label, color in reference_lines:
                ax.axhline(y=y_val, color=color, linestyle='--', linewidth=1.5,
                           alpha=0.7, label=label)
            ax.legend(loc='upper right', fontsize=9, prop={'weight': 'bold'})

        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='black')
        # Auto-append MODEL_NAME to title if not already present
        violin_title = f'{title} (Violin Plot)'
        if MODEL_NAME and MODEL_NAME not in title:
            violin_title = f'{title} ({MODEL_NAME}) (Violin Plot)'
        ax.set_title(violin_title, fontsize=14, fontweight='bold', color='black', pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=11, fontweight='bold', color='black')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        violin_path = output_dir / f"{plot_filename}_violin"
        pdf_path, png_path = save_figure_dual_format(fig, violin_path, dpi=300)
        plt.close(fig)

        results['violin'] = (pdf_path, png_path)
        print(f"Saved violin plot:")
        print(f"  PDF: {pdf_path}")
        print(f"  PNG: {png_path}")

    return results


# =============================================================================
# IMPROVEMENT RATIO PLOT (NoInstruct → Instruct)
# =============================================================================

def get_method_base_color(method: str) -> str:
    """Get base color for training method (SFT, DPO, CITA)."""
    if 'SFT' in method:
        return '#8B0000'  # Dark red
    elif 'DPO' in method:
        return '#006400'  # Dark green
    elif 'CITA' in method:
        return '#00008B'  # Dark blue
    return '#808080'


def calculate_improvement_delta(
    scores: Dict[str, float],
    methods: List[str] = ['SFT', 'DPO', 'CITA']
) -> Dict[str, Tuple[float, float, float]]:
    """
    Calculate improvement delta from NoInstruct to Instruct for each method.

    Args:
        scores: Dict mapping model names (e.g., 'SFT_Instruct') to scores
        methods: List of training methods to compare

    Returns:
        Dict mapping method to (noinstruct_score, instruct_score, delta)
    """
    improvements = {}

    for method in methods:
        no_key = f"{method}_NoInstruct"
        inst_key = f"{method}_Instruct"

        if no_key in scores and inst_key in scores:
            no_score = scores[no_key]
            inst_score = scores[inst_key]
            delta = inst_score - no_score
            improvements[method] = (no_score, inst_score, delta)

    return improvements


def generate_improvement_ratio_plot(
    scores: Dict[str, float],
    output_path: Path,
    eval_name: str,
    metric_name: str = "Score",
    higher_is_better: bool = True,
    valid_only_scores: Optional[Dict[str, float]] = None,
    valid_rates: Optional[Dict[str, float]] = None,
    methods: List[str] = ['SFT', 'DPO', 'CITA']
) -> Optional[str]:
    """
    Generate improvement ratio plot showing NoInstruct → Instruct delta.

    Shows grouped bars for each training method:
    - NoInstruct score
    - Instruct score
    - Delta (improvement) as annotation

    Args:
        scores: Dict mapping model names to overall scores
        output_path: Path to save the plot
        eval_name: Name of evaluation (e.g., "TruthfulQA")
        metric_name: Name of the metric being plotted
        higher_is_better: If True, positive delta = improvement
        valid_only_scores: Optional dict of valid-only scores (same keys)
        valid_rates: Optional dict of valid response rates (same keys)
        methods: Training methods to include

    Returns:
        Path to saved plot, or None if not enough data
    """
    # Calculate improvements
    improvements = calculate_improvement_delta(scores, methods)

    if len(improvements) < 2:
        print(f"Need at least 2 methods with both NoInstruct/Instruct for improvement plot")
        return None

    # Also calculate for valid-only if provided
    valid_improvements = None
    if valid_only_scores:
        valid_improvements = calculate_improvement_delta(valid_only_scores, methods)

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 7))

    method_list = [m for m in methods if m in improvements]
    x = np.arange(len(method_list))
    bar_width = 0.125  # Reduced by 50%

    # Colors for NoInstruct (lighter) and Instruct (darker)
    noinstruct_colors = ['#FF6B6B', '#90EE90', '#87CEEB']  # Light red, green, blue
    instruct_colors = ['#8B0000', '#006400', '#00008B']     # Dark red, green, blue
    method_to_idx = {'SFT': 0, 'DPO': 1, 'CITA': 2}

    # Plot bars
    noinstruct_scores = [improvements[m][0] for m in method_list]
    instruct_scores = [improvements[m][1] for m in method_list]
    deltas = [improvements[m][2] for m in method_list]

    colors_no = [noinstruct_colors[method_to_idx[m]] for m in method_list]
    colors_inst = [instruct_colors[method_to_idx[m]] for m in method_list]

    # NoInstruct bars
    bars_no = ax.bar(x - bar_width/2, noinstruct_scores, bar_width,
                     color=colors_no, edgecolor='black', linewidth=2.0,
                     label='NoInstruct')

    # Instruct bars
    bars_inst = ax.bar(x + bar_width/2, instruct_scores, bar_width,
                       color=colors_inst, edgecolor='black', linewidth=2.0,
                       label='Instruct')

    # Add delta annotations with arrows
    for i, (m, delta) in enumerate(zip(method_list, deltas)):
        # Determine annotation color and prefix
        if higher_is_better:
            is_improvement = delta > 0
        else:
            is_improvement = delta < 0

        color = 'green' if is_improvement else 'red'
        prefix = '+' if delta > 0 else ''

        # Position arrow between bars
        no_val = noinstruct_scores[i]
        inst_val = instruct_scores[i]

        # Draw arrow from NoInstruct to Instruct
        mid_x = x[i]
        arrow_y = max(no_val, inst_val) + abs(inst_val - no_val) * 0.3

        # Delta label
        ax.annotate(
            f'Δ = {prefix}{delta:.3f}',
            xy=(mid_x, arrow_y),
            fontsize=11,
            fontweight='bold',
            color=color,
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9)
        )

    # Add value labels on bars
    for bar, val in zip(bars_no, noinstruct_scores):
        y_pos = val + 0.01 if val >= 0 else val - 0.01
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va=va, fontsize=10, fontweight='bold', color='black')

    for bar, val in zip(bars_inst, instruct_scores):
        y_pos = val + 0.01 if val >= 0 else val - 0.01
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va=va, fontsize=10, fontweight='bold', color='black')

    # Highlight winner
    best_idx = np.argmax(deltas) if higher_is_better else np.argmin(deltas)
    winner = method_list[best_idx]
    winner_delta = deltas[best_idx]

    # Labels and styling
    direction = "Higher = Better" if higher_is_better else "Lower = Better"
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold', color='black')
    ax.set_title(f'{eval_name}: Instruction Adaptation (NoInstruct → Instruct)\n{direction}',
                 fontsize=14, fontweight='bold', color='black', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(method_list, fontsize=12, fontweight='bold', color='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Zero line if scores can be negative
    all_vals = noinstruct_scores + instruct_scores
    if min(all_vals) < 0:
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    # Dynamic y-axis
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.4)  # Extra room for delta labels

    # Legend
    legend_elements = [
        Patch(facecolor='#AAAAAA', edgecolor='black', label='NoInstruct (lighter)'),
        Patch(facecolor='#555555', edgecolor='black', label='Instruct (darker)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, prop={'weight': 'bold'})

    # Winner annotation
    ax.text(0.98, 0.98,
            f'Best Δ: {winner} ({winner_delta:+.3f})',
            transform=ax.transAxes, fontsize=11, fontweight='bold', color='black',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', alpha=0.8))

    plt.tight_layout()
    pdf_path, png_path = save_figure_dual_format(fig, output_path, dpi=300)
    plt.close()

    print(f"Saved improvement plot:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")

    # Print ranking
    sorted_methods = sorted(zip(method_list, deltas), key=lambda x: x[1], reverse=higher_is_better)
    print(f"\nImprovement Ranking ({'+Δ' if higher_is_better else '-Δ'} = Better):")
    for rank, (method, delta) in enumerate(sorted_methods, 1):
        print(f"   {rank}. {method}: {delta:+.3f}")

    return str(output_path)


# =============================================================================
# RADAR CHART - Average Radius (Mean Normalized Improvement)
# =============================================================================

def calculate_avg_radius(radii: List[float]) -> float:
    """
    Calculate simple average of radar chart radii.

    Formula: avg = Σ(rᵢ) / n

    This computes the mean of normalized radii across all benchmarks,
    NOT the geometric pentagon area.

    Args:
        radii: List of radius values (already normalized 0-1)

    Returns:
        Average radius (0-1 scale, multiply by 100 for percentage)
    """
    if not radii:
        return 0.0

    return sum(radii) / len(radii)


def generate_radar_chart_area_based(
    eval_deltas: Dict[str, Dict[str, float]],
    output_path: Path,
    methods: List[str] = ['DPO', 'PPO', 'GRPO', 'CITA'],
    normalize: bool = True,
    delta_ci: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    model_name: str = None
) -> Optional[str]:
    """
    Generate radar chart with AVERAGE RADIUS ranking (higher avg = better overall performance).

    This version computes the mean of normalized radii across all benchmarks,
    providing a holistic measure of instruction alignment efficiency.

    Args:
        eval_deltas: Dict mapping eval_name to {method: delta}
        output_path: Path to save the plot
        methods: Training methods to compare
        normalize: If True, normalize deltas to [0, 1] per eval for comparability
        delta_ci: Optional dict mapping eval_name to {method: (ci_lower, ci_upper)}
            for confidence intervals. If provided, shaded CI bands are drawn.

    Returns:
        Path to saved plot, or None if not enough data
    """
    eval_names = list(eval_deltas.keys())
    num_evals = len(eval_names)

    if num_evals < 2:
        print("Need at least 2 evals for radar chart")
        return None

    # Setup angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_evals, endpoint=False).tolist()
    angles_closed = angles + angles[:1]  # Close the polygon for plotting

    # Method colors (consistent with get_model_color for *_Instruct variants)
    method_colors = {
        'SFT': '#8B0000',    # Dark red
        'DPO': '#006400',    # Dark green
        'PPO': '#4B0082',    # Indigo (purple)
        'GRPO': '#FF8C00',   # Dark orange
        'CITA': '#00008B',   # Dark blue
    }

    # Prepare data
    method_data = {m: [] for m in methods}
    raw_deltas = {m: [] for m in methods}

    for eval_name in eval_names:
        deltas = eval_deltas[eval_name]
        for method in methods:
            raw_deltas[method].append(deltas.get(method, 0))

    # Normalize if requested (min-max per eval to [0, 1])
    if normalize:
        for i, eval_name in enumerate(eval_names):
            vals = [raw_deltas[m][i] for m in methods]
            min_val, max_val = min(vals), max(vals)
            range_val = max_val - min_val if max_val != min_val else 1

            for method in methods:
                # Normalize to [0.1, 1.0] so all methods are visible
                normalized = 0.1 + 0.9 * (raw_deltas[method][i] - min_val) / range_val
                method_data[method].append(normalized)
    else:
        method_data = {m: list(raw_deltas[m]) for m in methods}

    # Calculate average radius for each method
    method_avg = {}
    for method in methods:
        avg = calculate_avg_radius(method_data[method])
        method_avg[method] = avg

    # Convert to percentage (radii are already normalized 0-1)
    method_avg_pct = {m: avg * 100 for m, avg in method_avg.items()}

    # Rank methods by average radius
    sorted_methods = sorted(methods, key=lambda m: method_avg[m], reverse=True)

    # Close the polygon for plotting
    method_data_closed = {m: method_data[m] + method_data[m][:1] for m in methods}

    # Create figure - compact layout (radar + legend below)
    fig, ax = plt.subplots(figsize=(7, 7.5), subplot_kw=dict(polar=True))
    ax.set_position([0.08, 0.18, 0.84, 0.72])  # Maximize radar area, leave room for legend

    # Prepare CI data if provided (normalize CI bounds same as deltas)
    method_ci_lower = {m: [] for m in methods}
    method_ci_upper = {m: [] for m in methods}
    has_ci = delta_ci is not None

    if has_ci:
        for i, eval_name in enumerate(eval_names):
            ci_data = delta_ci.get(eval_name, {})
            vals = [raw_deltas[m][i] for m in methods]
            min_val, max_val = min(vals), max(vals)
            range_val = max_val - min_val if max_val != min_val else 1

            for method in methods:
                if method in ci_data:
                    ci_lo, ci_hi = ci_data[method]
                    # Normalize CI bounds using same scale as delta
                    if normalize:
                        ci_lo_norm = 0.1 + 0.9 * (ci_lo - min_val) / range_val
                        ci_hi_norm = 0.1 + 0.9 * (ci_hi - min_val) / range_val
                    else:
                        ci_lo_norm, ci_hi_norm = ci_lo, ci_hi
                    method_ci_lower[method].append(ci_lo_norm)
                    method_ci_upper[method].append(ci_hi_norm)
                else:
                    # No CI for this method/eval, use same as mean
                    method_ci_lower[method].append(method_data[method][i])
                    method_ci_upper[method].append(method_data[method][i])

        # Close CI polygons
        for method in methods:
            method_ci_lower[method] = method_ci_lower[method] + method_ci_lower[method][:1]
            method_ci_upper[method] = method_ci_upper[method] + method_ci_upper[method][:1]

    # Plot each method (sorted by avg for legend ordering)
    for method in sorted_methods:
        color = method_colors.get(method, '#808080')
        avg_pct = method_avg_pct[method]

        # Draw CI band if available (shaded region between CI bounds)
        if has_ci:
            ci_lo = method_ci_lower[method]
            ci_hi = method_ci_upper[method]
            ax.fill_between(angles_closed, ci_lo, ci_hi, alpha=0.15, color=color,
                            linewidth=0, label=None)

        ax.plot(angles_closed, method_data_closed[method], 'o-', linewidth=3,
                label=f'{method} ({avg_pct:.1f}%)', color=color, markersize=10)
        ax.fill(angles_closed, method_data_closed[method], alpha=0.20, color=color)

    # Set labels - Place OUTSIDE circle with TANGENTIAL orientation (perpendicular to radius)
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # Hide default labels

    # Manually place labels at radius endpoint, tangential (perpendicular to radius)
    label_radius = 1.22  # Outside the 100% circle with padding for delta annotations
    for i, (angle, label) in enumerate(zip(angles, eval_names)):
        angle_deg = np.degrees(angle) % 360

        # TANGENTIAL = perpendicular to radius = angle - 90 degrees
        # Adjust for readability (text should not be upside down)
        if 0 <= angle_deg <= 180:
            # Top half: text flows clockwise
            rotation = angle_deg - 90
            va = 'bottom'
        else:
            # Bottom half: text flows counter-clockwise (flip to stay readable)
            rotation = angle_deg + 90
            va = 'top'

        ax.text(angle, label_radius, label,
                fontsize=15, fontweight='bold', color='black',
                ha='center', va=va, rotation=rotation, rotation_mode='anchor')

    # Add raw delta annotations at each vertex (winner only, TANGENTIAL orientation)
    for i, eval_name in enumerate(eval_names):
        angle = angles[i]
        angle_deg = np.degrees(angle) % 360

        for method in methods:
            raw_val = raw_deltas[method][i]
            all_vals = [raw_deltas[m][i] for m in methods]

            # Only annotate the winner for each eval
            if raw_val == max(all_vals):
                method_color = method_colors.get(method, '#808080')
                # Position annotation just outside the data point
                r = method_data[method][i] + 0.15

                # TANGENTIAL rotation (perpendicular to radius)
                if 0 <= angle_deg <= 180:
                    rotation = angle_deg - 90
                else:
                    rotation = angle_deg + 90

                ax.text(angle, r, f'{raw_val:+.3f}',
                        fontsize=13, fontweight='bold', color=method_color,
                        ha='center', va='center', rotation=rotation, rotation_mode='anchor')

    # Styling - LARGER FONT SIZE
    ax.set_ylim(0, 1.45)  # Extended to fit labels outside circle
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])  # Hide default labels

    # Manually place percentage labels with TANGENTIAL orientation
    # Position them at angle between first two data points for visibility
    pct_angle = (angles[0] + angles[1]) / 2  # Between ECLIPTICA and TruthfulQA
    pct_angle_deg = np.degrees(pct_angle) % 360
    # Tangential rotation for this angle
    if 0 <= pct_angle_deg <= 180:
        pct_rotation = pct_angle_deg - 90
    else:
        pct_rotation = pct_angle_deg + 90

    for r_val, label in [(0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (1.0, '100%')]:
        ax.text(pct_angle, r_val, label,
                fontsize=12, fontweight='bold', color='#404040',
                ha='center', va='center', rotation=pct_rotation, rotation_mode='anchor')

    # Subtle grid: thin lines, slightly darker than default
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5, color='#404040')

    # Title - reduced font size so width ≤ outer circle diameter
    title = 'Instruction Alignment Efficiency\n(Average Radius = Overall Performance)'
    if model_name:
        title = f'Instruction Alignment Efficiency ({model_name})\n(Average Radius = Overall Performance)'
    ax.set_title(title, fontsize=13, fontweight='bold', color='black', pad=15)

    # =========================================================================
    # LEGEND: Single horizontal line below radar chart
    # Format: Method (Avg%): -o- CITA(95.6%), -o- DPO(70.5%), ...
    # =========================================================================
    from matplotlib.lines import Line2D

    # Create custom legend handles with line+marker
    legend_handles = []
    legend_labels = []
    for method in sorted_methods:
        color = method_colors.get(method, '#808080')
        avg_pct = method_avg_pct[method]
        handle = Line2D([0], [0], marker='o', color=color, linewidth=3,
                        markersize=10, markerfacecolor=color)
        legend_handles.append(handle)
        legend_labels.append(f'{method} ({avg_pct:.1f}%)')

    # Place legend at bottom center in 2 rows x 2 cols (more padding from circle)
    legend = fig.legend(handles=legend_handles, labels=legend_labels,
               loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=2, fontsize=14, labelcolor='black',
               prop={'weight': 'bold'}, frameon=True,
               title='Method (Avg %)', title_fontsize=13,
               columnspacing=1.5, handletextpad=0.5)
    # Make legend title bold black
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_color('black')

    # Save
    pdf_path, png_path = save_figure_dual_format(fig, output_path, dpi=300)
    plt.close()

    print(f"\nSaved AVERAGE RADIUS radar chart:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")

    # Print ranking summary
    winner = sorted_methods[0]
    winner_avg = method_avg_pct[winner]
    runner_up = sorted_methods[1] if len(sorted_methods) > 1 else None
    runner_up_avg = method_avg_pct[runner_up] if runner_up else 0
    margin = winner_avg - runner_up_avg

    print(f"\n{'='*60}")
    print(f"INSTRUCTION ALIGNMENT EFFICIENCY (Average Radius)")
    print(f"{'='*60}")
    for rank, method in enumerate(sorted_methods, 1):
        avg = method_avg[method]
        avg_pct = method_avg_pct[method]
        print(f"  #{rank} {method}: {avg_pct:.1f}% (avg={avg:.4f})")

    print(f"\n  WINNER: {winner} with {winner_avg:.1f}% coverage")
    print(f"   Margin over {runner_up}: +{margin:.1f}%")

    return str(output_path)


# =============================================================================
# HEATMAP - Combined Absolute Scores Across All Evals
# =============================================================================

def get_heatmap_figsize(n_rows: int, n_cols: int,
                        cell_width: float = 1.3, cell_height: float = 0.5) -> Tuple[float, float]:
    """
    Auto-calculate figsize based on data dimensions for compact heatmaps.

    Args:
        n_rows: Number of rows (models)
        n_cols: Number of columns (evaluations)
        cell_width: Width per cell in inches
        cell_height: Height per cell in inches

    Returns:
        Tuple of (width, height) for figsize
    """
    margin_x = 1.8  # for y-axis labels (shortened: CITA_NI, CITA_I)
    margin_y = 1.8  # for title + colorbar
    return (n_cols * cell_width + margin_x, n_rows * cell_height + margin_y)


def _shorten_model_labels(models: List[str]) -> List[str]:
    """Shorten model labels: _NoInstruct -> _NI, _Instruct -> _I"""
    return [m.replace('_NoInstruct', '_NI').replace('_Instruct', '_I') for m in models]


def _wrap_eval_labels(eval_names: List[str]) -> List[str]:
    """Format eval names for heatmap columns - use full names with line breaks."""
    # Use full names with line break for better readability
    metric_map = {
        'ECLIPTICA (M₁)': 'ECLIPTICA\n(M₁)',
        'TruthfulQA (M₂)': 'TruthfulQA\n(M₂)',
        'Cond. Safety (M₃)': 'Cond. Safety\n(M₃)',
        'Length Ctrl (M₄)': 'Length Ctrl\n(M₄)',
        'LITMUS (AQI-M₅)': 'LITMUS\n(AQI-M₅)',
    }
    return [metric_map.get(name, name) for name in eval_names]


def generate_combined_heatmap(
    eval_scores: Dict[str, Dict[str, float]],
    output_path: Path,
    models: List[str] = None,
    normalize_per_column: bool = True,
    show_raw_values: bool = True,
    cmap: str = 'RdYlGn',
    score_ci: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    model_name: str = None
) -> Optional[str]:
    """
    Generate heatmap showing absolute scores across all evals for all models.

    Args:
        eval_scores: Dict mapping eval_name to {model_name: score}
            e.g., {'ISD': {'SFT_NoInstruct': 0.22, 'SFT_Instruct': 0.44, ...}, ...}
        output_path: Path to save the plot
        models: List of model names in desired order (default: sorted)
        normalize_per_column: If True, normalize each eval column to [0,1] for color
        show_raw_values: If True, show raw values in cells (not normalized)
        cmap: Colormap name (RdYlGn = red-yellow-green, good for higher=better)
        score_ci: Optional dict mapping eval_name to {model_name: (ci_lower, ci_upper)}
            for confidence intervals. If provided, cells show "value +/- ci".

    Returns:
        Path to saved plot, or None if not enough data
    """
    import seaborn as sns

    eval_names = list(eval_scores.keys())
    if len(eval_names) < 2:
        print("Need at least 2 evals for heatmap")
        return None

    # Get all models across evals
    all_models = set()
    for scores in eval_scores.values():
        all_models.update(scores.keys())

    # Default model order: group by method, NoInstruct before Instruct
    # Note: SFT excluded - it's a training stage, not a policy optimization method
    if models is None:
        models = [
            'DPO_NoInstruct', 'DPO_Instruct',
            'PPO_NoInstruct', 'PPO_Instruct',
            'GRPO_NoInstruct', 'GRPO_Instruct',
            'CITA_NoInstruct', 'CITA_Instruct'
        ]
        models = [m for m in models if m in all_models]

    if len(models) < 2:
        print("Need at least 2 models for heatmap")
        return None

    # Build data matrix
    data = []
    raw_data = []
    for model in models:
        row = []
        raw_row = []
        for eval_name in eval_names:
            score = eval_scores[eval_name].get(model, np.nan)
            row.append(score)
            raw_row.append(score)
        data.append(row)
        raw_data.append(raw_row)

    data = np.array(data)
    raw_data = np.array(raw_data)

    # Normalize per column for color mapping
    if normalize_per_column:
        normalized_data = np.zeros_like(data)
        for j in range(data.shape[1]):
            col = data[:, j]
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() > 0:
                col_min = np.nanmin(col)
                col_max = np.nanmax(col)
                col_range = col_max - col_min if col_max != col_min else 1
                normalized_data[:, j] = (col - col_min) / col_range
            else:
                normalized_data[:, j] = 0.5
        color_data = normalized_data
    else:
        color_data = data

    # Create figure with dynamic sizing based on data dimensions
    n_rows, n_cols = len(models), len(eval_names)
    fig, ax = plt.subplots(figsize=get_heatmap_figsize(n_rows, n_cols),
                           constrained_layout=True)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Build annotation data: separate value and CI for different styling
    import pandas as pd
    value_annot = []  # Main values (bold)
    ci_annot = []     # CI values (italic, not bold)
    for i in range(len(models)):
        value_row = []
        ci_row = []
        model = models[i]
        for j in range(len(eval_names)):
            val = raw_data[i, j]
            eval_name = eval_names[j]

            if not np.isnan(val):
                # Format main value
                if abs(val) >= 10:
                    value_row.append(f'{val:.1f}')
                else:
                    value_row.append(f'{val:.3f}')

                # Check if CI is available for this cell
                ci_text = ""
                if score_ci is not None and eval_name in score_ci:
                    ci_data = score_ci[eval_name].get(model)
                    if ci_data is not None:
                        ci_lo, ci_hi = ci_data
                        ci_half = (ci_hi - ci_lo) / 2
                        if abs(val) >= 10:
                            ci_text = f'+/-{ci_half:.1f}'
                        else:
                            ci_text = f'+/-{ci_half:.2f}'
                ci_row.append(ci_text)
            else:
                value_row.append('')
                ci_row.append('')
        value_annot.append(value_row)
        ci_annot.append(ci_row)

    # Shorten labels for compact display
    display_models = _shorten_model_labels(models)
    display_evals = _wrap_eval_labels(eval_names)

    # Use seaborn heatmap WITHOUT annotation (we'll add manually for mixed styling)
    import seaborn as sns
    heatmap = sns.heatmap(
        color_data,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=False,
        cbar_kws={'shrink': 0.8},
        xticklabels=display_evals,
        yticklabels=display_models,
        linewidths=0,
        linecolor='none'
    )

    # Make colorbar text BOLD (tick labels + title)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.set_label('Normalized Score (per eval)', fontsize=13, fontweight='bold')

    # Manually add annotations with different styles
    for i in range(len(models)):
        for j in range(len(eval_names)):
            val_text = value_annot[i][j]
            ci_text = ci_annot[i][j]
            if val_text:
                # Main value: bold, larger font for visibility
                ax.text(j + 0.5, i + 0.38, val_text,
                        ha='center', va='center',
                        fontsize=16, fontweight='bold', color='black')
                # CI value: smaller italic, not bold
                if ci_text:
                    ax.text(j + 0.5, i + 0.68, ci_text,
                            ha='center', va='center',
                            fontsize=11, fontstyle='italic', fontweight='normal', color='black')

    # Style tick labels - rotate 20 degrees, centered alignment
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, fontweight='bold', color='black', rotation=20, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, fontweight='bold', color='black')

    # Title - clarify that numbers are original, colors are normalized
    title = 'Model Performance Across All Evaluations\n(Values = Original, Colors = Normalized)'
    if model_name:
        title = f'Model Performance ({model_name})\n(Values = Original, Colors = Normalized)'
    ax.set_title(title, fontsize=16, fontweight='bold', color='black', pad=15)

    # Add method separators (horizontal lines between DPO/PPO/GRPO/CITA)
    for idx in [2, 4, 6]:  # After each method group (4 methods = 8 rows)
        if idx < len(models):
            ax.axhline(y=idx, color='black', linewidth=3)

    # Note: Don't call tight_layout() - constrained_layout=True already handles this
    # and they conflict when colorbar is present
    pdf_path, png_path = save_figure_dual_format(fig, output_path, dpi=300)
    plt.close()

    print(f"Saved heatmap:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")

    # Print summary table
    print(f"\nPerformance Summary:")
    header = f"{'Model':<20}" + "".join([f"{e:<12}" for e in eval_names])
    print(header)
    print("-" * len(header))
    for i, model in enumerate(models):
        row = f"{model:<20}"
        for j, eval_name in enumerate(eval_names):
            val = raw_data[i, j]
            if not np.isnan(val):
                if abs(val) >= 10:
                    row += f"{val:<12.1f}"
                else:
                    row += f"{val:<12.3f}"
            else:
                row += f"{'N/A':<12}"
        print(row)

    return str(output_path)
