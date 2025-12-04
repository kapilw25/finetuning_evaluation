"""
Plotting Utilities for Evaluation Scripts

Shared color mapping and legend elements for consistent plots.
Includes improvement ratio visualization for NoInstruct → Instruct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import List, Dict, Optional, Tuple


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
    bar_width = 0.25

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
                     color=colors_no, edgecolor='black', linewidth=1.5,
                     label='NoInstruct')

    # Instruct bars
    bars_inst = ax.bar(x + bar_width/2, instruct_scores, bar_width,
                       color=colors_inst, edgecolor='black', linewidth=1.5,
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
                f'{val:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    for bar, val in zip(bars_inst, instruct_scores):
        y_pos = val + 0.01 if val >= 0 else val - 0.01
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    # Highlight winner
    best_idx = np.argmax(deltas) if higher_is_better else np.argmin(deltas)
    winner = method_list[best_idx]
    winner_delta = deltas[best_idx]

    # Labels and styling
    direction = "Higher = Better" if higher_is_better else "Lower = Better"
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    ax.set_title(f'{eval_name}: Instruction Adaptation (NoInstruct → Instruct)\n{direction}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(method_list, fontsize=12, fontweight='bold')
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
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Winner annotation
    ax.text(0.98, 0.98,
            f'Best Δ: {winner} ({winner_delta:+.3f})',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
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
# RADAR CHART - Combined Improvement Across All Evals
# =============================================================================

def generate_improvement_radar_chart(
    eval_deltas: Dict[str, Dict[str, float]],
    output_path: Path,
    methods: List[str] = ['SFT', 'DPO', 'CITA'],
    normalize: bool = True
) -> Optional[str]:
    """
    Generate radar chart showing improvement (NoInstruct → Instruct) across all evals.

    Args:
        eval_deltas: Dict mapping eval_name to {method: delta}
            e.g., {'ISD': {'SFT': 0.2, 'DPO': 0.21, 'CITA': 0.22}, ...}
        output_path: Path to save the plot
        methods: Training methods to compare
        normalize: If True, normalize deltas to [0, 1] per eval for comparability

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
    angles += angles[:1]  # Close the polygon

    # Method colors
    method_colors = {
        'SFT': '#8B0000',   # Dark red
        'DPO': '#006400',   # Dark green
        'CITA': '#00008B'   # Dark blue
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
        method_data = raw_deltas

    # Close the polygon for each method
    for method in methods:
        method_data[method] = method_data[method] + method_data[method][:1]

    # Create figure - wider to accommodate table on the right
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))
    # Center the radar
    ax.set_position([0.1, 0.15, 0.55, 0.75])

    # Plot each method
    for method in methods:
        color = method_colors.get(method, '#808080')
        ax.plot(angles, method_data[method], 'o-', linewidth=2.5,
                label=method, color=color, markersize=8)
        ax.fill(angles, method_data[method], alpha=0.15, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(eval_names, fontsize=12, fontweight='bold')

    # Add raw delta annotations at each vertex
    for i, eval_name in enumerate(eval_names):
        angle = angles[i]
        for method in methods:
            raw_val = raw_deltas[method][i]
            # Position text slightly outside the plot
            r = method_data[method][i] + 0.12

            # Only annotate the winner for each eval to reduce clutter
            all_vals = [raw_deltas[m][i] for m in methods]
            if raw_val == max(all_vals):
                color = method_colors.get(method, '#808080')
                ax.annotate(f'{raw_val:+.2f}',
                           xy=(angle, r),
                           fontsize=9, fontweight='bold',
                           color=color, ha='center', va='center')

    # Styling
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Title
    ax.set_title('Instruction Adaptation: NoInstruct → Instruct\n(CITA vs DPO vs SFT)',
                 fontsize=14, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=11)

    # Add winner summary
    wins = {m: 0 for m in methods}
    for eval_name in eval_names:
        deltas = eval_deltas[eval_name]
        winner = max(methods, key=lambda m: deltas.get(m, float('-inf')))
        wins[winner] += 1

    # Add delta table showing all values for all methods (right side, outside radar)
    # Build table data: rows=evals, cols=methods
    table_data = []
    for eval_name in eval_names:
        row = []
        deltas = eval_deltas[eval_name]
        for method in methods:
            val = deltas.get(method, 0)
            row.append(f'{val:+.3f}')
        table_data.append(row)

    # Add table to figure (right bottom, outside the radar)
    table_ax = fig.add_axes([0.68, 0.15, 0.30, 0.35])  # [left, bottom, width, height]
    table_ax.axis('off')

    # Create table
    cell_colors = []
    for i, eval_name in enumerate(eval_names):
        row_colors = []
        deltas = eval_deltas[eval_name]
        max_val = max(deltas.get(m, float('-inf')) for m in methods)
        for method in methods:
            val = deltas.get(method, 0)
            if val == max_val:
                row_colors.append('#90EE90')  # Light green for winner
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    tbl = table_ax.table(
        cellText=table_data,
        rowLabels=eval_names,
        colLabels=methods,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)  # Reduced width scale from 1.3 to 1.0

    # Table title
    table_ax.set_title('Δ = Instruct - NoInstruct', fontsize=11, fontweight='bold', pad=5)

    # Add winner summary at bottom with boundary padding
    summary = " | ".join([f"{m}: {wins[m]}/{num_evals}" for m in methods])
    fig.text(0.82, 0.05, f"Wins: {summary}", ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    pdf_path, png_path = save_figure_dual_format(fig, output_path, dpi=300)
    plt.close()

    print(f"Saved radar chart:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")

    # Print summary
    print(f"\nImprovement Summary (NoInstruct → Instruct):")
    print(f"{'Eval':<20} {'SFT':<12} {'DPO':<12} {'CITA':<12} {'Winner':<10}")
    print("-" * 66)
    for eval_name in eval_names:
        deltas = eval_deltas[eval_name]
        winner = max(methods, key=lambda m: deltas.get(m, float('-inf')))
        row = f"{eval_name:<20}"
        for m in methods:
            val = deltas.get(m, 0)
            row += f"{val:+.3f}{'':>5}"
        row += f" {winner}"
        print(row)

    print(f"\nFinal Score: " + " | ".join([f"{m}: {wins[m]}" for m in methods]))

    return str(output_path)


# =============================================================================
# HEATMAP - Combined Absolute Scores Across All Evals
# =============================================================================

def generate_combined_heatmap(
    eval_scores: Dict[str, Dict[str, float]],
    output_path: Path,
    models: List[str] = None,
    normalize_per_column: bool = True,
    show_raw_values: bool = True,
    cmap: str = 'RdYlGn'
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
    if models is None:
        models = [
            'SFT_NoInstruct', 'SFT_Instruct',
            'DPO_NoInstruct', 'DPO_Instruct',
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

    # Create figure with no gaps
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create annotation matrix for seaborn
    import pandas as pd
    annot_matrix = []
    for i in range(len(models)):
        row = []
        for j in range(len(eval_names)):
            val = raw_data[i, j]
            if not np.isnan(val):
                if abs(val) >= 10:
                    row.append(f'{val:.1f}')
                else:
                    row.append(f'{val:.3f}')
            else:
                row.append('')
        annot_matrix.append(row)

    # Use seaborn heatmap (no gaps between cells)
    import seaborn as sns
    sns.heatmap(
        color_data,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=np.array(annot_matrix),
        fmt='',
        annot_kws={'fontsize': 11, 'fontweight': 'bold'},
        cbar_kws={'label': 'Normalized Score (per eval)', 'shrink': 0.8},
        xticklabels=eval_names,
        yticklabels=models,
        linewidths=0,
        linecolor='none'
    )

    # Style tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

    # Title - clarify that numbers are original, colors are normalized
    ax.set_title('Model Performance Across All Evaluations\n(Values = Original Scores, Colors = Normalized per Column, Higher = Better)',
                 fontsize=14, fontweight='bold', pad=15)

    # Add method separators (horizontal lines between SFT/DPO/CITA)
    for idx in [2, 4]:  # After SFT_Instruct, after DPO_Instruct
        if idx < len(models):
            ax.axhline(y=idx, color='black', linewidth=3)

    plt.tight_layout()
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
