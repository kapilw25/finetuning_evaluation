"""
Appendix Visualizations Module

Generates additional visualizations for the paper appendix:
1. ISD Embedding t-SNE/PCA - Response embeddings by instruction type
2. AQI 3D Scatterplot - Safe vs unsafe response clusters
3. ISD Fidelity Heatmap - Instruction type × model matrix
4. Length Control Distribution - Word count violin plots

All functions are designed to be imported and called from generate_combined_plots.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'bold'  # ALL text bold globally
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from .plotting import save_figure_dual_format, get_model_color


# =============================================================================
# 1. ISD EMBEDDING VISUALIZATION (t-SNE/PCA by Instruction Type)
# =============================================================================

def generate_isd_embedding_visualization(
    outputs_dir: Path,
    output_dir: Path,
    models: List[str] = None
) -> List[str]:
    """
    Generate t-SNE/PCA plot of ISD response embeddings colored by instruction type.

    Args:
        outputs_dir: Root outputs directory (contains ISD_Evaluation_Embedding/)
        output_dir: Where to save plots
        models: List of models to visualize (default: CITA_Instruct, DPO_Instruct)

    Returns:
        List of generated file paths
    """
    from sklearn.manifold import TSNE

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [SKIP] sentence-transformers not installed for ISD embedding viz")
        return []

    if models is None:
        models = ['CITA_Instruct', 'DPO_Instruct']

    generated_files = []
    isd_dir = outputs_dir / "evaluation" / "ISD_Evaluation_Embedding"

    if not isd_dir.exists():
        print(f"  [SKIP] ISD directory not found: {isd_dir}")
        return []

    # Load embedding model once
    print("  Loading SentenceTransformer model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    for model_key in models:
        responses_csv = isd_dir / model_key / f"{model_key}_isd_responses.csv"

        if not responses_csv.exists():
            print(f"  [SKIP] {model_key}: responses CSV not found")
            continue

        print(f"  Processing {model_key}...")
        df = pd.read_csv(responses_csv)

        # Check if instruction_type column exists
        if 'instruction_type' not in df.columns:
            print(f"  [SKIP] {model_key}: no instruction_type column")
            continue

        # Filter valid responses
        if 'is_valid' in df.columns:
            df = df[df['is_valid'] == True].copy()

        if len(df) < 50:
            print(f"  [SKIP] {model_key}: too few valid responses ({len(df)})")
            continue

        # Generate embeddings
        print(f"    Encoding {len(df)} responses...")
        embeddings = embed_model.encode(df['response'].tolist(), show_progress_bar=False)

        # t-SNE reduction to 2D
        print(f"    Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)//4))
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot colored by instruction_type
        fig, ax = plt.subplots(figsize=(12, 10))

        instruction_types = sorted(df['instruction_type'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(instruction_types)))

        for i, inst_type in enumerate(instruction_types):
            mask = df['instruction_type'] == inst_type
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=inst_type,
                alpha=0.6,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )

        ax.legend(loc='upper right', fontsize=9, frameon=True,
                 fancybox=False, edgecolor='black')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f'{model_key}: Response Embeddings by Instruction Type\n(t-SNE, n={len(df)})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plot_path = output_dir / f"isd_tsne_{model_key}"
        pdf_path, png_path = save_figure_dual_format(fig, plot_path, dpi=300)
        plt.close(fig)

        generated_files.extend([pdf_path, png_path])
        print(f"    Saved: {plot_path.name}.{{pdf,png}}")

    return generated_files


# =============================================================================
# 2. AQI 3D SCATTERPLOT (Safe vs Unsafe clusters)
# =============================================================================

def generate_aqi_3d_visualization(
    outputs_dir: Path,
    output_dir: Path,
    models: List[str] = None
) -> List[str]:
    """
    Generate 3D scatterplot of AQI embeddings colored by safety label.

    Args:
        outputs_dir: Root outputs directory (contains AQI_Evaluation/)
        output_dir: Where to save plots
        models: List of models to visualize

    Returns:
        List of generated file paths
    """
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D

    if models is None:
        models = ['CITA_Instruct', 'DPO_Instruct']

    generated_files = []
    aqi_dir = outputs_dir / "evaluation" / "AQI_Evaluation"

    if not aqi_dir.exists():
        print(f"  [SKIP] AQI directory not found: {aqi_dir}")
        return []

    for model_key in models:
        embeddings_file = aqi_dir / model_key / "embeddings.pkl"

        if not embeddings_file.exists():
            print(f"  [SKIP] {model_key}: embeddings.pkl not found")
            continue

        print(f"  Processing {model_key}...")
        df = pd.read_pickle(embeddings_file)

        # Check required columns
        if 'embedding' not in df.columns or 'safety_label_binary' not in df.columns:
            print(f"  [SKIP] {model_key}: missing required columns")
            continue

        # Extract embeddings
        embeddings = np.array(df['embedding'].tolist())

        if len(embeddings) < 50:
            print(f"  [SKIP] {model_key}: too few samples ({len(embeddings)})")
            continue

        # t-SNE to 3D
        print(f"    Running 3D t-SNE on {len(embeddings)} samples...")
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)//4))
        embeddings_3d = tsne.fit_transform(embeddings)

        # Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        safe_mask = df['safety_label_binary'] == 1

        ax.scatter(
            embeddings_3d[safe_mask, 0],
            embeddings_3d[safe_mask, 1],
            embeddings_3d[safe_mask, 2],
            c='#2ca02c', label=f'Safe (n={safe_mask.sum()})',
            alpha=0.6, s=20
        )
        ax.scatter(
            embeddings_3d[~safe_mask, 0],
            embeddings_3d[~safe_mask, 1],
            embeddings_3d[~safe_mask, 2],
            c='#d62728', label=f'Unsafe (n={(~safe_mask).sum()})',
            alpha=0.6, s=20
        )

        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.set_zlabel('t-SNE Dim 3', fontsize=10)
        ax.set_title(f'{model_key}: Response Embeddings\n(Safe vs Unsafe, 3D t-SNE)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plot_path = output_dir / f"aqi_3d_{model_key}"
        pdf_path, png_path = save_figure_dual_format(fig, plot_path, dpi=300)
        plt.close(fig)

        generated_files.extend([pdf_path, png_path])
        print(f"    Saved: {plot_path.name}.{{pdf,png}}")

    return generated_files


# =============================================================================
# 3. INSTRUCTION FIDELITY HEATMAP (10 instruction types × 6 models)
# =============================================================================

def generate_instruction_fidelity_heatmap(
    outputs_dir: Path,
    output_dir: Path,
    models: List[str] = None,
    instruction_types: List[str] = None
) -> List[str]:
    """
    Heatmap: Instruction Type × Model showing fidelity scores.

    Args:
        outputs_dir: Root outputs directory
        output_dir: Where to save plots
        models: List of models (default: all 6)
        instruction_types: List of instruction types (default: all 10)

    Returns:
        List of generated file paths
    """
    import seaborn as sns

    if models is None:
        models = ['SFT_NoInstruct', 'SFT_Instruct', 'DPO_NoInstruct', 'DPO_Instruct',
                  'CITA_NoInstruct', 'CITA_Instruct']

    if instruction_types is None:
        instruction_types = ['neutral', 'conservative', 'liberal', 'regulatory', 'empathetic',
                            'safety_first', 'educational', 'concise', 'professional', 'creative']

    isd_dir = outputs_dir / "evaluation" / "ISD_Evaluation_Embedding"

    if not isd_dir.exists():
        print(f"  [SKIP] ISD directory not found: {isd_dir}")
        return []

    # Build matrix
    data = np.zeros((len(instruction_types), len(models)))
    valid_models = []

    for j, model in enumerate(models):
        metrics_file = isd_dir / model / f"{model}_isd_metrics.json"

        if not metrics_file.exists():
            print(f"  [SKIP] {model}: metrics file not found")
            data[:, j] = np.nan
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        fidelity_by_inst = metrics.get('fidelity_by_instruction', {})

        for i, inst in enumerate(instruction_types):
            data[i, j] = fidelity_by_inst.get(inst, 0)

        valid_models.append(model)

    if len(valid_models) < 2:
        print(f"  [SKIP] Not enough models with fidelity data")
        return []

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create display labels (capitalize instruction types)
    display_instructions = [inst.replace('_', ' ').title() for inst in instruction_types]

    sns.heatmap(
        data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        xticklabels=models,
        yticklabels=display_instructions,
        ax=ax,
        cbar_kws={'label': 'Fidelity Score [0-1]', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Instruction Type', fontsize=12, fontweight='bold')
    ax.set_title('ISD: Per-Instruction Fidelity Scores\n(Higher = Better instruction following)',
                fontsize=14, fontweight='bold', pad=15)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plot_path = output_dir / "isd_fidelity_heatmap"
    pdf_path, png_path = save_figure_dual_format(fig, plot_path, dpi=300)
    plt.close(fig)

    print(f"  Saved: {plot_path.name}.{{pdf,png}}")
    return [pdf_path, png_path]


# =============================================================================
# 4. WORD COUNT DISTRIBUTION (Box/Violin plot)
# =============================================================================

def generate_length_control_distribution(
    outputs_dir: Path,
    output_dir: Path,
    models: List[str] = None
) -> List[str]:
    """
    Box/Violin plot of word counts for CONCISE vs DETAILED.

    Args:
        outputs_dir: Root outputs directory
        output_dir: Where to save plots
        models: List of models to include

    Returns:
        List of generated file paths
    """
    import seaborn as sns

    if models is None:
        models = ['CITA_Instruct', 'DPO_Instruct', 'SFT_Instruct']

    lc_dir = outputs_dir / "evaluation" / "Length_Control_Evaluation"

    if not lc_dir.exists():
        print(f"  [SKIP] Length Control directory not found: {lc_dir}")
        return []

    # Collect data from all models
    all_data = []
    valid_models = []

    for model in models:
        concise_path = lc_dir / model / "concise_responses.csv"
        detailed_path = lc_dir / model / "detailed_responses.csv"

        if not concise_path.exists() or not detailed_path.exists():
            print(f"  [SKIP] {model}: response files not found")
            continue

        concise_df = pd.read_csv(concise_path)
        detailed_df = pd.read_csv(detailed_path)

        # Add to combined data
        for wc in concise_df['word_count']:
            all_data.append({'Model': model, 'Variant': 'CONCISE', 'Word Count': wc})
        for wc in detailed_df['word_count']:
            all_data.append({'Model': model, 'Variant': 'DETAILED', 'Word Count': wc})

        valid_models.append(model)

    if len(valid_models) < 1:
        print(f"  [SKIP] No valid models for length control viz")
        return []

    combined_df = pd.DataFrame(all_data)

    # Create figure with subplots for each model
    n_models = len(valid_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)

    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, valid_models):
        model_data = combined_df[combined_df['Model'] == model]

        # Get model color
        color = get_model_color(model)

        sns.violinplot(
            data=model_data,
            x='Variant',
            y='Word Count',
            ax=ax,
            palette={'CONCISE': '#87CEEB', 'DETAILED': '#2ca02c'},
            inner='quartile'
        )

        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=200, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

        # Add target labels
        ax.text(1.02, 50, 'Target: 50', transform=ax.get_yaxis_transform(),
               fontsize=8, color='red', va='center')
        ax.text(1.02, 200, 'Target: 200', transform=ax.get_yaxis_transform(),
               fontsize=8, color='green', va='center')

        ax.set_xlabel('')
        if ax == axes[0]:
            ax.set_ylabel('Word Count', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')

    fig.suptitle('Length Control: Word Count Distribution\n(CONCISE vs DETAILED instructions)',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plot_path = output_dir / "length_control_distribution"
    pdf_path, png_path = save_figure_dual_format(fig, plot_path, dpi=300)
    plt.close(fig)

    print(f"  Saved: {plot_path.name}.{{pdf,png}}")
    return [pdf_path, png_path]


# =============================================================================
# 5. ISD FIDELITY RADAR (10-axis radar per model)
# =============================================================================

def generate_isd_fidelity_radar(
    outputs_dir: Path,
    output_dir: Path,
    models: List[str] = None,
    instruction_types: List[str] = None
) -> List[str]:
    """
    Generate 10-axis radar chart showing fidelity scores per instruction type.

    One radar plot with all models overlaid for easy comparison.

    Args:
        outputs_dir: Root outputs directory
        output_dir: Where to save plots
        models: List of models (default: all 6)
        instruction_types: List of instruction types (default: all 10)

    Returns:
        List of generated file paths
    """
    if models is None:
        models = ['SFT_NoInstruct', 'SFT_Instruct', 'DPO_NoInstruct', 'DPO_Instruct',
                  'CITA_NoInstruct', 'CITA_Instruct']

    if instruction_types is None:
        instruction_types = ['neutral', 'conservative', 'liberal', 'regulatory', 'empathetic',
                            'safety_first', 'educational', 'concise', 'professional', 'creative']

    isd_dir = outputs_dir / "evaluation" / "ISD_Evaluation_Embedding"

    if not isd_dir.exists():
        print(f"  [SKIP] ISD directory not found: {isd_dir}")
        return []

    # Collect fidelity data for all models
    model_data = {}

    for model in models:
        metrics_file = isd_dir / model / f"{model}_isd_metrics.json"

        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        fidelity_by_inst = metrics.get('fidelity_by_instruction', {})
        values = [fidelity_by_inst.get(inst, 0) for inst in instruction_types]
        model_data[model] = values

    if len(model_data) < 2:
        print(f"  [SKIP] Not enough models with fidelity data")
        return []

    # Setup radar chart
    num_vars = len(instruction_types)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Color scheme matching eval_utils/plotting.py
    model_colors = {
        'SFT_NoInstruct': '#FF6B6B', 'SFT_Instruct': '#8B0000',
        'DPO_NoInstruct': '#90EE90', 'DPO_Instruct': '#006400',
        'PPO_NoInstruct': '#DDA0DD', 'PPO_Instruct': '#4B0082',
        'GRPO_NoInstruct': '#FFDAB9', 'GRPO_Instruct': '#FF8C00',
        'CITA_NoInstruct': '#87CEEB', 'CITA_Instruct': '#00008B'
    }

    model_linestyles = {
        'SFT_NoInstruct': '--', 'SFT_Instruct': '-',
        'DPO_NoInstruct': '--', 'DPO_Instruct': '-',
        'PPO_NoInstruct': '--', 'PPO_Instruct': '-',
        'GRPO_NoInstruct': '--', 'GRPO_Instruct': '-',
        'CITA_NoInstruct': '--', 'CITA_Instruct': '-'
    }

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    # Plot each model
    for model, values in model_data.items():
        values_closed = values + values[:1]  # Close the polygon
        color = model_colors.get(model, '#808080')
        ls = model_linestyles.get(model, '-')

        ax.plot(angles, values_closed, 'o-', linewidth=2, label=model,
               color=color, linestyle=ls, markersize=5)
        ax.fill(angles, values_closed, alpha=0.1, color=color)

    # Set labels
    display_labels = [inst.replace('_', ' ').title() for inst in instruction_types]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=10)

    # Set radial limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_title('ISD: Per-Instruction Fidelity Scores\n(Higher = Better, Outer = 1.0)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    plt.tight_layout()
    plot_path = output_dir / "isd_fidelity_radar"
    pdf_path, png_path = save_figure_dual_format(fig, plot_path, dpi=300)
    plt.close(fig)

    print(f"  Saved: {plot_path.name}.{{pdf,png}}")
    return [pdf_path, png_path]


# =============================================================================
# 6. TRUTHFULQA PER-CATEGORY HEATMAP (38 categories × models)
# =============================================================================

def generate_truthfulqa_category_heatmap(
    outputs_dir: Path,
    output_dir: Path,
    models: List[str] = None
) -> List[str]:
    """
    Generate heatmap showing TruthfulQA adaptation scores per category.

    Args:
        outputs_dir: Root outputs directory
        output_dir: Where to save plots
        models: List of models to include

    Returns:
        List of generated file paths
    """
    import seaborn as sns

    if models is None:
        models = ['SFT_NoInstruct', 'SFT_Instruct', 'DPO_NoInstruct', 'DPO_Instruct',
                  'CITA_NoInstruct', 'CITA_Instruct']

    tqa_dir = outputs_dir / "evaluation" / "TruthfulQA_Evaluation"

    if not tqa_dir.exists():
        print(f"  [SKIP] TruthfulQA directory not found: {tqa_dir}")
        return []

    # Collect per-category data
    all_categories = set()
    model_category_data = {}

    for model in models:
        # Try to load honest responses (has category column)
        honest_path = tqa_dir / model / "honest_responses.csv"
        confident_path = tqa_dir / model / "confident_responses.csv"

        if not honest_path.exists() or not confident_path.exists():
            print(f"  [SKIP] {model}: response files not found")
            continue

        honest_df = pd.read_csv(honest_path)
        confident_df = pd.read_csv(confident_path)

        # Check if category column exists
        if 'category' not in honest_df.columns:
            print(f"  [SKIP] {model}: no category column")
            continue

        # Calculate per-category adaptation scores
        # Adaptation = HONEST_uncertainty - CONFIDENT_uncertainty
        category_scores = {}

        for cat in honest_df['category'].unique():
            all_categories.add(cat)

            hon_cat = honest_df[honest_df['category'] == cat]
            conf_cat = confident_df[confident_df['category'] == cat]

            if len(hon_cat) > 0 and len(conf_cat) > 0:
                # Get uncertainty markers (if available) or use a simple metric
                if 'uncertainty_score' in hon_cat.columns:
                    hon_score = hon_cat['uncertainty_score'].mean()
                    conf_score = conf_cat['uncertainty_score'].mean()
                else:
                    # Count uncertainty markers as proxy (case-insensitive via .str.lower())
                    hon_score = hon_cat['response'].str.lower().str.count(r'\b(maybe|perhaps|possibly|uncertain)\b').mean()
                    conf_score = conf_cat['response'].str.lower().str.count(r'\b(maybe|perhaps|possibly|uncertain)\b').mean()

                category_scores[cat] = hon_score - conf_score
            else:
                category_scores[cat] = 0.0

        model_category_data[model] = category_scores

    if len(model_category_data) < 2:
        print(f"  [SKIP] Not enough models with category data")
        return []

    # Build matrix
    categories = sorted(all_categories)
    valid_models = list(model_category_data.keys())

    data = np.zeros((len(categories), len(valid_models)))

    for j, model in enumerate(valid_models):
        for i, cat in enumerate(categories):
            data[i, j] = model_category_data[model].get(cat, 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(12, len(categories) * 0.4)))

    # Diverging colormap centered at 0
    vmax = max(abs(data.min()), abs(data.max()))

    sns.heatmap(
        data,
        annot=True,
        fmt='.2f',
        cmap='RdBu',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=valid_models,
        yticklabels=categories,
        ax=ax,
        cbar_kws={'label': 'Adaptation Score (HON - CONF)', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('TruthfulQA Category', fontsize=12, fontweight='bold')
    ax.set_title('TruthfulQA: Per-Category Adaptation Scores\n(Blue = Better HON→CONF adaptation, Red = Reverse)',
                fontsize=14, fontweight='bold', pad=15)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_path = output_dir / "truthfulqa_category_heatmap"
    pdf_path, png_path = save_figure_dual_format(fig, plot_path, dpi=300)
    plt.close(fig)

    print(f"  Saved: {plot_path.name}.{{pdf,png}}")
    return [pdf_path, png_path]


# =============================================================================
# MAIN GENERATOR FUNCTION (Called from generate_combined_plots.py)
# =============================================================================

def generate_all_appendix_visualizations(
    outputs_dir: Path,
    output_dir: Path,
    skip_embeddings: bool = False
) -> Dict[str, List[str]]:
    """
    Generate all appendix visualizations.

    Args:
        outputs_dir: Root outputs directory
        output_dir: Where to save plots (e.g., outputs/combined_plots/appendix/)
        skip_embeddings: If True, skip embedding visualizations (ISD t-SNE, AQI 3D)

    Returns:
        Dict mapping visualization type to list of generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("\n" + "=" * 60)
    print("GENERATING APPENDIX VISUALIZATIONS")
    print("=" * 60)

    # 1. ISD Fidelity Heatmap (no embeddings needed)
    print("\n1. ISD Fidelity Heatmap...")
    results['isd_fidelity_heatmap'] = generate_instruction_fidelity_heatmap(
        outputs_dir, output_dir
    )

    # 2. ISD Fidelity Radar (10-axis radar per model)
    print("\n2. ISD Fidelity Radar (10-axis)...")
    results['isd_fidelity_radar'] = generate_isd_fidelity_radar(
        outputs_dir, output_dir
    )

    # 3. Length Control Distribution (no embeddings needed)
    print("\n3. Length Control Distribution...")
    results['length_control_distribution'] = generate_length_control_distribution(
        outputs_dir, output_dir
    )

    # 4. TruthfulQA Per-Category Heatmap
    print("\n4. TruthfulQA Category Heatmap...")
    results['truthfulqa_category_heatmap'] = generate_truthfulqa_category_heatmap(
        outputs_dir, output_dir
    )

    if not skip_embeddings:
        # 5. ISD Embedding Visualization (requires sentence-transformers)
        print("\n5. ISD Embedding t-SNE...")
        results['isd_embedding_tsne'] = generate_isd_embedding_visualization(
            outputs_dir, output_dir
        )

        # 6. AQI 3D Scatterplot
        print("\n6. AQI 3D Scatterplot...")
        results['aqi_3d_scatterplot'] = generate_aqi_3d_visualization(
            outputs_dir, output_dir
        )
    else:
        print("\n[SKIP] Embedding visualizations (--skip-embeddings)")
        results['isd_embedding_tsne'] = []
        results['aqi_3d_scatterplot'] = []

    # Note: Final summary with total count is handled by generate_combined_plots.py
    # (includes HP ablation plots which are generated separately)
    return results
