"""
Plotting Utilities for Evaluation Scripts

Shared color mapping and legend elements for consistent plots.
"""

from matplotlib.patches import Patch
from typing import List


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
