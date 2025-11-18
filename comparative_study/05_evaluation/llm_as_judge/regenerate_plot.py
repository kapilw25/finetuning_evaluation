"""
Regenerate toxicity comparison plot from existing summary.json files
Usage: python regenerate_plot.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

# Paths
project_root = Path(__file__).parent.parent.parent.parent
EVAL_OUTPUT_DIR = project_root / "comparative_study" / "05_evaluation" / "llm_as_judge" / "Toxicity_Evaluation_Results"

def get_color(model_name):
    """Assign colors: SFT=red, DPO=green, CITA=blue; dark=Instruct, light=NoInstruct"""
    if 'SFT' in model_name:
        return '#8B0000' if 'Instruct' in model_name else '#FF6B6B'  # Dark/Light Red
    elif 'DPO' in model_name:
        return '#006400' if 'Instruct' in model_name else '#90EE90'  # Dark/Light Green
    elif 'CITA' in model_name:
        return '#00008B' if 'Instruct' in model_name else '#87CEEB'  # Dark/Light Blue
    elif 'Baseline' in model_name:
        return '#808080'  # Gray for baseline
    else:
        return '#FFA500'  # Orange for unknown

def main():
    print("="*80)
    print("📊 Regenerating Toxicity Comparison Plot")
    print("="*80)

    # Load all summary.json files
    all_results = {}

    for model_dir in EVAL_OUTPUT_DIR.iterdir():
        if model_dir.is_dir():
            summary_path = model_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    all_results[model_dir.name] = {'summary': summary}
                    print(f"✅ Loaded: {model_dir.name}")

    if len(all_results) < 2:
        print("\n❌ Error: Need at least 2 models with summary.json files")
        print(f"   Found: {list(all_results.keys())}")
        return

    print(f"\n📊 Generating plot for {len(all_results)} models...")

    # Prepare data
    models = list(all_results.keys())
    toxicity_means = [all_results[m]['summary']['toxicity_mean'] for m in models]
    toxicity_medians = [all_results[m]['summary']['toxicity_median'] for m in models]
    toxicity_stds = [all_results[m]['summary']['toxicity_std'] for m in models]
    refusal_rates = [all_results[m]['summary']['safe_refusal_rate'] for m in models]
    safe_refusals = [all_results[m]['summary']['safe_refusals'] for m in models]
    n_evaluated = all_results[models[0]]['summary']['n_evaluated']

    # Sort by toxicity score (descending = worst to best left-to-right)
    sorted_indices = np.argsort(toxicity_means)[::-1]
    models_sorted = [models[i] for i in sorted_indices]
    toxicity_means_sorted = [toxicity_means[i] for i in sorted_indices]
    toxicity_medians_sorted = [toxicity_medians[i] for i in sorted_indices]
    toxicity_stds_sorted = [toxicity_stds[i] for i in sorted_indices]
    refusal_rates_sorted = [refusal_rates[i] for i in sorted_indices]
    safe_refusals_sorted = [safe_refusals[i] for i in sorted_indices]

    colors_sorted = [get_color(m) for m in models_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ========== LEFT PLOT: TOXICITY SCORES ==========
    bars1 = ax1.bar(range(len(models_sorted)), toxicity_means_sorted,
                   color=colors_sorted, edgecolor='black', linewidth=1.2,
                   yerr=toxicity_stds_sorted, capsize=5, alpha=0.85)

    ax1.set_ylabel('Mean Toxicity Score', fontsize=12, fontweight='bold')
    ax1.set_title('Toxicity Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 5.5)
    ax1.set_xticks(range(len(models_sorted)))
    ax1.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=10)
    ax1.axhline(y=1, color='green', linestyle='--', label='Perfect Safety (1.0)', linewidth=2, alpha=0.7)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)

    # Value labels on bars
    for i, (bar, mean, median, std) in enumerate(zip(bars1, toxicity_means_sorted,
                                                      toxicity_medians_sorted, toxicity_stds_sorted)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.2,
                f'{mean:.2f}\n(±{std:.2f})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ========== RIGHT PLOT: SAFE REFUSAL RATES ==========
    bars2 = ax2.bar(range(len(models_sorted)), refusal_rates_sorted,
                   color=colors_sorted, edgecolor='black', linewidth=1.2, alpha=0.85)

    ax2.set_ylabel('Safe Refusal Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Safe Refusal Rate (Higher is Better)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(len(models_sorted)))
    ax2.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=10)
    ax2.axhline(y=1.0, color='green', linestyle='--', label='100% Refusal', linewidth=2, alpha=0.7)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=10)

    # Value labels on bars
    for i, (bar, rate, count) in enumerate(zip(bars2, refusal_rates_sorted, safe_refusals_sorted)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{rate:.1%}\n({count}/{n_evaluated})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add color legend
    legend_elements = [
        Patch(facecolor='#8B0000', edgecolor='black', label='SFT_Instruct'),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='SFT_NoInstruct'),
        Patch(facecolor='#006400', edgecolor='black', label='DPO_Instruct'),
        Patch(facecolor='#90EE90', edgecolor='black', label='DPO_NoInstruct'),
        Patch(facecolor='#00008B', edgecolor='black', label='CITA_Instruct'),
        Patch(facecolor='#87CEEB', edgecolor='black', label='CITA_NoInstruct'),
    ]
    # Only add baseline to legend if present
    if any('Baseline' in m for m in models_sorted):
        legend_elements.insert(0, Patch(facecolor='#808080', edgecolor='black', label='Baseline'))

    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.02, 1, 1])  # Leave space for legend

    output_path = EVAL_OUTPUT_DIR / "toxicity_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot: {output_path}")

    # Print ranking
    print(f"\n📊 Toxicity Ranking (Best to Worst):")
    for rank, (model, score) in enumerate(zip(reversed(models_sorted), reversed(toxicity_means_sorted)), 1):
        print(f"   {rank}. {model}: {score:.2f}")

    print("\n" + "="*80)
    print("✅ Plot regeneration complete!")
    print("="*80)

if __name__ == "__main__":
    main()
