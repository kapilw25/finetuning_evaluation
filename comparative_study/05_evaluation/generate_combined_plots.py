"""
Generate Combined Plots for All Evaluations

Single entry point for ALL visualization generation:

1. MAIN PAPER PLOTS (--main):
   - Radar chart: Instruction adaptation (NoInstruct → Instruct) improvement
   - Heatmap: Absolute scores across all models and evals

2. APPENDIX PLOTS (--appendix):
   - ISD Fidelity Heatmap: Instruction type × model matrix
   - Length Control Distribution: Word count violin plots
   - ISD Embedding t-SNE: Response clusters by instruction type
   - AQI 3D Scatterplot: Safe vs unsafe response clusters
   - HP Ablation: Hyperparameter sensitivity from Optuna trials

Both PDF (for Overleaf) and PNG (for sharing) formats are generated.

Usage:
    # Generate all plots (main + appendix)
    python comparative_study/05_evaluation/generate_combined_plots.py

    # Generate only main paper plots
    python comparative_study/05_evaluation/generate_combined_plots.py --main

    # Generate only appendix plots
    python comparative_study/05_evaluation/generate_combined_plots.py --appendix

    # Skip embedding visualizations (faster, no sentence-transformers needed)
    python comparative_study/05_evaluation/generate_combined_plots.py --appendix --skip-embeddings

Output:
    outputs/combined_plots/
    ├── radar.{pdf,png}                    # Main paper
    ├── heatmap.{pdf,png}                  # Main paper
    ├── appendix/
    │   ├── isd_fidelity_heatmap.{pdf,png}
    │   ├── length_control_distribution.{pdf,png}
    │   ├── isd_tsne_CITA_Instruct.{pdf,png}
    │   ├── isd_tsne_DPO_Instruct.{pdf,png}
    │   ├── aqi_3d_CITA_Instruct.{pdf,png}
    │   ├── aqi_3d_DPO_Instruct.{pdf,png}
    │   └── hp_ablation_*.{pdf,png}        # From Optuna trials
"""

import sys
import json
import csv
import argparse
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "05_evaluation"))

from eval_utils.plotting import generate_improvement_radar_chart, generate_combined_heatmap
from eval_utils.appendix_visualizations import generate_all_appendix_visualizations

# Import HP ablation generator (separate script due to different data source)
sys.path.insert(0, str(project_root / "comparative_study"))
from generate_hp_ablation_plots import (
    load_all_trial_configs,
    plot_hp_vs_metric,
    plot_combined_ablation,
    plot_pareto_frontier,
)

# Output directories
OUTPUTS_DIR = project_root / "outputs"
COMBINED_PLOTS_DIR = OUTPUTS_DIR / "combined_plots"
COMBINED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input directories for each evaluation
EVAL_DIRS = {
    "ISD": OUTPUTS_DIR / "ISD_Evaluation_Embedding",  # Embedding-based ISD
    "TruthfulQA": OUTPUTS_DIR / "TruthfulQA_Evaluation",
    "Cond. Safety": OUTPUTS_DIR / "Conditional_Safety_Evaluation",
    "Length Ctrl": OUTPUTS_DIR / "Length_Control_Evaluation",
    "AQI": OUTPUTS_DIR / "AQI_Evaluation",
}

# Metric keys for each evaluation
METRIC_KEYS = {
    "ISD": "instruction_awareness_score",
    "TruthfulQA": "adaptation_score",
    "Cond. Safety": "adaptation_score",
    "Length Ctrl": "adaptation_score",
    "AQI": "aqi_score",
}

METHODS = ['SFT', 'DPO', 'CITA']


def load_eval_metrics(eval_name: str, eval_dir: Path, metric_key: str) -> dict:
    """Load metrics from evaluation output directory.

    Handles multiple formats:
    1. Per-model directories: outputs/Eval/ModelName/metrics.json
    2. Combined file: outputs/Eval/metrics.json
    3. AQI format: outputs/Eval/ModelName/ModelName_metrics_summary.csv
    """
    scores = {}

    if not eval_dir.exists():
        print(f"  ⚠️  {eval_name}: Directory not found at {eval_dir}")
        return scores

    # Try per-model directory structure first
    for model_dir in eval_dir.iterdir():
        if model_dir.is_dir():
            # Try metrics.json first
            metrics_file = model_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                score = data.get(metric_key)
                if score is not None:
                    scores[model_dir.name] = score
                continue

            # Try ISD format: ModelName_isd_metrics.json
            isd_file = model_dir / f"{model_dir.name}_isd_metrics.json"
            if isd_file.exists():
                with open(isd_file, 'r') as f:
                    data = json.load(f)
                score = data.get(metric_key)
                if score is not None:
                    scores[model_dir.name] = score
                continue

            # Try AQI CSV format: ModelName_metrics_summary.csv
            csv_file = model_dir / f"{model_dir.name}_metrics_summary.csv"
            if csv_file.exists():
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('Category', '').lower() == 'overall':
                            # AQI score is in "AQI [0-100] (↑)" column
                            aqi_score = row.get('AQI [0-100] (↑)')
                            if aqi_score:
                                scores[model_dir.name] = float(aqi_score)
                            break

    # If no per-model metrics found, try combined file
    if not scores:
        combined_file = eval_dir / "metrics.json"
        if combined_file.exists():
            with open(combined_file, 'r') as f:
                data = json.load(f)
            for model_name, model_data in data.items():
                if isinstance(model_data, dict):
                    if 'metrics' in model_data:
                        score = model_data['metrics'].get(metric_key)
                    else:
                        score = model_data.get(metric_key)
                    if score is not None:
                        scores[model_name] = score

    if not scores:
        print(f"  ⚠️  {eval_name}: No metrics found at {eval_dir}")

    return scores


def calculate_deltas(scores: dict) -> dict:
    """Calculate improvement delta (Instruct - NoInstruct) for each method."""
    deltas = {}
    for method in METHODS:
        no_key = f"{method}_NoInstruct"
        inst_key = f"{method}_Instruct"

        if no_key in scores and inst_key in scores:
            deltas[method] = scores[inst_key] - scores[no_key]

    return deltas


def generate_hp_ablation_plots(output_dir: Path) -> dict:
    """Generate hyperparameter ablation plots from Optuna trial configs."""
    print("=" * 80)
    print("GENERATING HP ABLATION PLOTS")
    print("=" * 80)

    trial_config_dir = project_root / "outputs" / "CITA_Instruct_Adaptive"
    results = {}

    trials = load_all_trial_configs()

    if not trials:
        print(f"  No trial configs found at {trial_config_dir}")
        return results

    print(f"  Loaded {len(trials)} trial configs")

    # Generate combined ablation plot (2x2)
    combined_path = output_dir / "hp_ablation_combined"
    files = plot_combined_ablation(trials, combined_path, best_trial_num=7)
    if files:
        results['hp_ablation_combined'] = files
        print(f"  [OK] hp_ablation_combined.{{pdf,png}}")

    # Generate Pareto frontier
    pareto_path = output_dir / "hp_pareto_frontier"
    files = plot_pareto_frontier(trials, pareto_path, best_trial_num=7)
    if files:
        results['hp_pareto_frontier'] = files
        print(f"  [OK] hp_pareto_frontier.{{pdf,png}}")

    return results


def generate_main_plots():
    """Generate main paper plots (radar chart + heatmap)."""
    print("=" * 80)
    print("GENERATING MAIN PAPER PLOTS")
    print("=" * 80)

    eval_deltas = {}      # For radar chart (improvement)
    eval_scores = {}      # For heatmap (absolute scores)

    for eval_name, eval_dir in EVAL_DIRS.items():
        metric_key = METRIC_KEYS[eval_name]
        print(f"\nLoading {eval_name}...")

        scores = load_eval_metrics(eval_name, eval_dir, metric_key)

        if not scores:
            print(f"  ❌ No scores found for {eval_name}")
            continue

        # Store absolute scores for heatmap
        eval_scores[eval_name] = scores
        print(f"  ✅ Loaded {len(scores)} models: {list(scores.keys())}")

        # Calculate deltas for radar chart
        deltas = calculate_deltas(scores)

        if len(deltas) >= 2:
            eval_deltas[eval_name] = deltas
            print(f"  ✅ Deltas: {deltas}")
        else:
            print(f"  ⚠️  Not enough methods with both variants for radar chart")

    generated = {'heatmap': False, 'radar': False}

    # =========================================================================
    # Generate Heatmap (absolute scores)
    # =========================================================================
    if len(eval_scores) >= 2:
        heatmap_path = COMBINED_PLOTS_DIR / "heatmap"
        print(f"\n{'=' * 80}")
        print("Generating Heatmap (Absolute Scores)...")
        print(f"{'=' * 80}")

        generate_combined_heatmap(
            eval_scores=eval_scores,
            output_path=heatmap_path,
            normalize_per_column=True,
            show_raw_values=True
        )
        generated['heatmap'] = True
        print(f"\n✅ Heatmap saved to: {heatmap_path}")
    else:
        print(f"\n⚠️  Need at least 2 evals for heatmap, got {len(eval_scores)}")

    # =========================================================================
    # Generate Radar Chart (improvement deltas)
    # =========================================================================
    if len(eval_deltas) >= 2:
        radar_path = COMBINED_PLOTS_DIR / "radar"
        print(f"\n{'=' * 80}")
        print("Generating Radar Chart (Improvement Deltas)...")
        print(f"{'=' * 80}")

        generate_improvement_radar_chart(
            eval_deltas=eval_deltas,
            output_path=radar_path,
            methods=METHODS,
            normalize=True
        )
        generated['radar'] = True
        print(f"\n✅ Radar chart saved to: {radar_path}")
    else:
        print(f"\n⚠️  Need at least 2 evals for radar chart, got {len(eval_deltas)}")

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined plots for all evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots (main + appendix)
  python generate_combined_plots.py

  # Generate only main paper plots
  python generate_combined_plots.py --main

  # Generate only appendix plots
  python generate_combined_plots.py --appendix

  # Skip embedding visualizations (faster)
  python generate_combined_plots.py --appendix --skip-embeddings
        """
    )
    parser.add_argument('--main', action='store_true',
                       help='Generate only main paper plots (radar, heatmap)')
    parser.add_argument('--appendix', action='store_true',
                       help='Generate only appendix plots')
    parser.add_argument('--skip-embeddings', action='store_true',
                       help='Skip embedding visualizations (ISD t-SNE, AQI 3D)')

    args = parser.parse_args()

    # Default: generate both if neither specified
    generate_main = args.main or (not args.main and not args.appendix)
    generate_appendix = args.appendix or (not args.main and not args.appendix)

    COMBINED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMBINED PLOT GENERATION")
    print("=" * 80)
    print(f"Output directory: {COMBINED_PLOTS_DIR}")
    print(f"Generate main plots: {generate_main}")
    print(f"Generate appendix plots: {generate_appendix}")
    print(f"Skip embeddings: {args.skip_embeddings}")

    main_results = {}
    appendix_results = {}

    # =========================================================================
    # Generate Main Paper Plots
    # =========================================================================
    if generate_main:
        main_results = generate_main_plots()

    # =========================================================================
    # Generate Appendix Plots
    # =========================================================================
    if generate_appendix:
        appendix_output_dir = COMBINED_PLOTS_DIR / "appendix"
        appendix_results = generate_all_appendix_visualizations(
            outputs_dir=OUTPUTS_DIR,
            output_dir=appendix_output_dir,
            skip_embeddings=args.skip_embeddings
        )

        # Generate HP ablation plots (from Optuna trial configs)
        hp_ablation_results = generate_hp_ablation_plots(appendix_output_dir)
        appendix_results.update(hp_ablation_results)

    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Output directory: {COMBINED_PLOTS_DIR}")

    if generate_main:
        print(f"\nMain Paper Plots:")
        if main_results.get('heatmap'):
            print(f"  ✅ heatmap.{{pdf,png}}")
        else:
            print(f"  ⚠️  heatmap: skipped")
        if main_results.get('radar'):
            print(f"  ✅ radar.{{pdf,png}}")
        else:
            print(f"  ⚠️  radar: skipped")

    if generate_appendix:
        print(f"\nAppendix Plots (in appendix/):")
        for viz_type, files in appendix_results.items():
            if files:
                print(f"  ✅ {viz_type}")
            else:
                print(f"  ⚠️  {viz_type}: skipped")


if __name__ == "__main__":
    main()
