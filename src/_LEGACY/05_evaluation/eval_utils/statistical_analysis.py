"""
Statistical analysis and visualization for dual-metric evaluation
Includes: Pareto frontier plot, bootstrap CI, paired t-tests, per-category breakdown

Usage:
    from statistical_analysis import run_statistical_analysis

    results = {
        "SFT_Baseline": {
            "harmlessness_df": pd.DataFrame(...),
            "helpfulness_df": pd.DataFrame(...),
            "summary": {"harmlessness_mean": 7.0, "helpfulness_mean": 8.0, ...}
        },
        ...
    }

    run_statistical_analysis(results, output_dir=Path("./results"))
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'bold'  # ALL text bold globally
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval

    Args:
        data: 1D array of scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrapped_means, alpha * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha) * 100)

    return np.mean(data), lower, upper


def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """
    Paired t-test for comparing two models

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B (same length as scores_a)

    Returns:
        Dict with t_statistic, p_value, cohens_d, significant
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

    # Cohen's d (effect size)
    diff = scores_a - scores_b
    cohens_d = np.mean(diff) / np.std(diff)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05
    }


def create_pareto_plot(
    results: Dict,
    output_path: Path
):
    """
    Create 2D Pareto frontier plot (Harmlessness vs Helpfulness)

    Args:
        results: Dict of {model_key: {"summary": {...}}}
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))

    # Extract coordinates
    models = []
    helpfulness = []
    harmlessness = []

    for model_key, data in results.items():
        summary = data['summary']
        models.append(model_key)
        helpfulness.append(summary['helpfulness_mean'])
        harmlessness.append(summary['harmlessness_mean'])

    # Plot points
    colors = {'SFT_Baseline': 'blue', 'DPO_Baseline': 'green', 'CITA_Baseline': 'red'}
    for i, model in enumerate(models):
        plt.scatter(
            helpfulness[i], harmlessness[i],
            s=200, c=colors.get(model, 'gray'),
            label=model.replace('_', ' '),
            alpha=0.8, edgecolors='black', linewidths=2
        )
        # Add annotation
        plt.annotate(
            f"({helpfulness[i]:.2f}, {harmlessness[i]:.2f})",
            (helpfulness[i], harmlessness[i]),
            textcoords="offset points", xytext=(10,10),
            ha='left', fontsize=10
        )

    # Pareto frontier line (connect non-dominated points)
    # A point (x1, y1) dominates (x2, y2) if x1 >= x2 AND y1 >= y2
    pareto_points = []
    for i in range(len(models)):
        dominated = False
        for j in range(len(models)):
            if i != j and helpfulness[j] >= helpfulness[i] and harmlessness[j] >= harmlessness[i]:
                if helpfulness[j] > helpfulness[i] or harmlessness[j] > harmlessness[i]:
                    dominated = True
                    break
        if not dominated:
            pareto_points.append((helpfulness[i], harmlessness[i], models[i]))

    # Sort Pareto points by helpfulness
    pareto_points.sort(key=lambda x: x[0])
    if len(pareto_points) > 1:
        px, py, _ = zip(*pareto_points)
        plt.plot(px, py, 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')

    plt.xlabel('Helpfulness Score (0-10)', fontsize=14, fontweight='bold')
    plt.ylabel('Harmlessness Score (0-10)', fontsize=14, fontweight='bold')
    plt.title('Dual-Metric Evaluation: Pareto Frontier\n(Higher is Better for Both)',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Pareto plot: {output_path}")
    plt.close()


def create_bootstrap_ci_plot(ci_results: List[Dict], output_path: Path):
    """Create bar chart with bootstrap confidence intervals"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = [r['model'].replace('_', ' ') for r in ci_results]

    # Harmlessness
    harm_means = [r['harmlessness_mean'] for r in ci_results]
    harm_errors = [(r['harmlessness_mean'] - r['harmlessness_ci_lower'],
                    r['harmlessness_ci_upper'] - r['harmlessness_mean']) for r in ci_results]

    ax1.bar(models, harm_means, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
    ax1.errorbar(models, harm_means, yerr=np.array(harm_errors).T, fmt='none',
                 ecolor='black', capsize=5, capthick=2)
    ax1.set_ylabel('Harmlessness Score (0-10)', fontsize=12, fontweight='bold')
    ax1.set_title('Harmlessness with 95% Bootstrap CI', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3)

    # Helpfulness
    help_means = [r['helpfulness_mean'] for r in ci_results]
    help_errors = [(r['helpfulness_mean'] - r['helpfulness_ci_lower'],
                    r['helpfulness_ci_upper'] - r['helpfulness_mean']) for r in ci_results]

    ax2.bar(models, help_means, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
    ax2.errorbar(models, help_means, yerr=np.array(help_errors).T, fmt='none',
                 ecolor='black', capsize=5, capthick=2)
    ax2.set_ylabel('Helpfulness Score (0-10)', fontsize=12, fontweight='bold')
    ax2.set_title('Helpfulness with 95% Bootstrap CI', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved CI bar chart: {output_path}")
    plt.close()


def create_category_heatmap(category_df: pd.DataFrame, output_path: Path):
    """Create heatmap of per-category refusal scores"""
    # Pivot table: categories × models
    pivot = category_df.pivot_table(
        index='category', columns='model', values='mean_refusal_score', fill_value=0
    )

    # Sort by average score across models
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False).drop('avg', axis=1)

    # Rename columns
    pivot.columns = [c.replace('_', ' ') for c in pivot.columns]

    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=10,
                cbar_kws={'label': 'Mean Refusal Score'}, linewidths=0.5)
    plt.title('Per-Category Harmlessness Scores (19 PKU Categories)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Harm Category', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved category heatmap: {output_path}")
    plt.close()


def create_distribution_plots(results: Dict, output_path: Path):
    """Create violin/box plots comparing score distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    harm_data = []
    help_data = []
    labels = []

    for model_key in results.keys():
        harm_scores = results[model_key]['harmlessness_df']['refusal_score'].dropna().values
        help_scores = results[model_key]['helpfulness_df']['helpfulness_score'].dropna().values

        harm_data.append(harm_scores)
        help_data.append(help_scores)
        labels.append(model_key.replace('_', ' '))

    # Harmlessness violin plot
    parts = ax1.violinplot(harm_data, positions=range(len(labels)), showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], ['blue', 'green', 'red']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Harmlessness Score (0-10)', fontsize=12, fontweight='bold')
    ax1.set_title('Harmlessness Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylim(-1, 11)
    ax1.grid(axis='y', alpha=0.3)

    # Helpfulness violin plot
    parts = ax2.violinplot(help_data, positions=range(len(labels)), showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], ['blue', 'green', 'red']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Helpfulness Score (0-10)', fontsize=12, fontweight='bold')
    ax2.set_title('Helpfulness Score Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylim(-1, 11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved distribution plots: {output_path}")
    plt.close()


def create_radar_chart(category_df: pd.DataFrame, output_path: Path):
    """Create radar chart for top 6 harm categories"""
    # Get top 6 categories by average score
    top_cats = category_df.groupby('category')['mean_refusal_score'].mean().nlargest(6).index.tolist()

    # Filter data
    radar_data = category_df[category_df['category'].isin(top_cats)].pivot_table(
        index='category', columns='model', values='mean_refusal_score', fill_value=0
    )

    # Number of variables
    categories = radar_data.index.tolist()
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = {'SFT_Baseline': 'blue', 'DPO_Baseline': 'green', 'CITA_Baseline': 'red'}

    for model in radar_data.columns:
        values = radar_data[model].tolist()
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=model.replace('_', ' '),
                color=colors.get(model, 'gray'), markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors.get(model, 'gray'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
    plt.title('Top 6 Harm Categories: Refusal Scores', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved radar chart: {output_path}")
    plt.close()


def create_effect_size_plot(results: Dict, output_path: Path):
    """Create effect size (Cohen's d) comparison plot"""
    effect_sizes = []
    comparisons = []

    # Calculate all pairwise Cohen's d for harmlessness
    model_list = list(results.keys())
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            model_a = model_list[i]
            model_b = model_list[j]

            scores_a = results[model_a]['harmlessness_df']['refusal_score'].dropna().values
            scores_b = results[model_b]['harmlessness_df']['refusal_score'].dropna().values
            min_len = min(len(scores_a), len(scores_b))

            ttest_result = paired_t_test(scores_a[:min_len], scores_b[:min_len])
            effect_sizes.append(abs(ttest_result['cohens_d']))
            comparisons.append(f"{model_a.replace('_', ' ')}\nvs\n{model_b.replace('_', ' ')}")

    plt.figure(figsize=(10, 6))
    bars = plt.barh(comparisons, effect_sizes, color=['steelblue', 'seagreen', 'coral'],
                    edgecolor='black', linewidth=1.5)

    # Add magnitude labels (small/medium/large)
    for bar, es in zip(bars, effect_sizes):
        if es < 0.2:
            label = "negligible"
        elif es < 0.5:
            label = "small"
        elif es < 0.8:
            label = "medium"
        else:
            label = "large"
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{es:.3f} ({label})", va='center', fontsize=10)

    plt.xlabel("Effect Size (|Cohen's d|)", fontsize=12, fontweight='bold')
    plt.title("Pairwise Effect Sizes for Harmlessness", fontsize=14, fontweight='bold')
    plt.xlim(0, max(effect_sizes) * 1.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved effect size plot: {output_path}")
    plt.close()


def run_statistical_analysis(
    results: Dict,
    output_dir: Path
):
    """
    Run full statistical analysis: bootstrap CI, t-tests, per-category breakdown + 6 plots

    Args:
        results: Dict of {model_key: {"harmlessness_df": ..., "helpfulness_df": ..., "summary": ...}}
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")

    # 1. Create Pareto plot
    create_pareto_plot(results, output_dir / "pareto_frontier.png")

    # 2. Bootstrap confidence intervals
    print(f"\n--- Bootstrap 95% CI ---")
    ci_results = []
    for model_key, data in results.items():
        harm_scores = data['harmlessness_df']['refusal_score'].dropna().values
        help_scores = data['helpfulness_df']['helpfulness_score'].dropna().values

        harm_mean, harm_lower, harm_upper = bootstrap_ci(harm_scores)
        help_mean, help_lower, help_upper = bootstrap_ci(help_scores)

        ci_results.append({
            "model": model_key,
            "harmlessness_mean": harm_mean,
            "harmlessness_ci_lower": harm_lower,
            "harmlessness_ci_upper": harm_upper,
            "helpfulness_mean": help_mean,
            "helpfulness_ci_lower": help_lower,
            "helpfulness_ci_upper": help_upper
        })

        print(f"{model_key}:")
        print(f"  Harmlessness: {harm_mean:.2f} [{harm_lower:.2f}, {harm_upper:.2f}]")
        print(f"  Helpfulness:  {help_mean:.2f} [{help_lower:.2f}, {help_upper:.2f}]")

    pd.DataFrame(ci_results).to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)

    # 3. Paired t-tests (CITA vs DPO, CITA vs SFT)
    print(f"\n--- Paired T-Tests ---")
    if "CITA_Baseline" in results and "DPO_Baseline" in results:
        # Harmlessness: CITA vs DPO
        cita_harm = results["CITA_Baseline"]['harmlessness_df']['refusal_score'].dropna().values
        dpo_harm = results["DPO_Baseline"]['harmlessness_df']['refusal_score'].dropna().values
        min_len = min(len(cita_harm), len(dpo_harm))

        ttest_harm = paired_t_test(cita_harm[:min_len], dpo_harm[:min_len])
        print(f"CITA vs DPO (Harmlessness):")
        print(f"  t={ttest_harm['t_statistic']:.3f}, p={ttest_harm['p_value']:.4f}")
        print(f"  Cohen's d={ttest_harm['cohens_d']:.3f}")
        print(f"  Significant: {'YES' if ttest_harm['significant'] else 'NO'}")

    if "CITA_Baseline" in results and "SFT_Baseline" in results:
        # Harmlessness: CITA vs SFT
        cita_harm = results["CITA_Baseline"]['harmlessness_df']['refusal_score'].dropna().values
        sft_harm = results["SFT_Baseline"]['harmlessness_df']['refusal_score'].dropna().values
        min_len = min(len(cita_harm), len(sft_harm))

        ttest_harm = paired_t_test(cita_harm[:min_len], sft_harm[:min_len])
        print(f"\nCITA vs SFT (Harmlessness):")
        print(f"  t={ttest_harm['t_statistic']:.3f}, p={ttest_harm['p_value']:.4f}")
        print(f"  Cohen's d={ttest_harm['cohens_d']:.3f}")
        print(f"  Significant: {'YES' if ttest_harm['significant'] else 'NO'}")

    # 4. Per-category breakdown (19 harm categories from PKU)
    print(f"\n--- Per-Category Breakdown (19 Harm Categories) ---")
    category_results = []
    for model_key, data in results.items():
        harm_df = data['harmlessness_df']

        # Explode harm_categories list column
        exploded = harm_df.explode('harm_categories')

        for category in exploded['harm_categories'].unique():
            if pd.isna(category):
                continue
            cat_scores = exploded[exploded['harm_categories'] == category]['refusal_score'].dropna()
            if len(cat_scores) > 0:
                category_results.append({
                    "model": model_key,
                    "category": category,
                    "mean_refusal_score": cat_scores.mean(),
                    "std": cat_scores.std(),
                    "n": len(cat_scores)
                })

    category_df = pd.DataFrame(category_results)
    category_df.to_csv(output_dir / "per_category_breakdown.csv", index=False)

    # Print top/bottom categories
    for model_key in results.keys():
        model_cats = category_df[category_df['model'] == model_key].sort_values('mean_refusal_score')
        if len(model_cats) > 0:
            print(f"\n{model_key} - Bottom 3 categories (weakest refusal):")
            print(model_cats.head(3)[['category', 'mean_refusal_score', 'n']].to_string(index=False))
            print(f"\n{model_key} - Top 3 categories (strongest refusal):")
            print(model_cats.tail(3)[['category', 'mean_refusal_score', 'n']].to_string(index=False))

    # 5. Generate additional publication-quality plots
    print(f"\n--- Generating Publication-Quality Plots ---")
    create_bootstrap_ci_plot(ci_results, output_dir / "bootstrap_ci_comparison.png")
    create_category_heatmap(category_df, output_dir / "per_category_heatmap.png")
    create_distribution_plots(results, output_dir / "score_distributions.png")
    create_radar_chart(category_df, output_dir / "top_categories_radar.png")
    create_effect_size_plot(results, output_dir / "effect_sizes.png")

    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"{'='*80}")
