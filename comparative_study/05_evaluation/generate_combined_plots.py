"""
Generate Combined Evaluation Plots

Focused on EVALUATION visualization only:
- Radar chart: Mean normalized radius for instruction alignment efficiency
- Heatmap: Absolute scores across all models and evaluations

NOTE: HP ablation plots (hyperparameter sensitivity) are generated separately by:
    python comparative_study/generate_hp_ablation_plots.py

Usage:
    python comparative_study/05_evaluation/generate_combined_plots.py

Output:
    outputs/evaluation/combined_plots/
    ├── radar_area.{pdf,png}    # Instruction alignment efficiency (average radius)
    └── heatmap.{pdf,png}       # Absolute scores heatmap
"""

import sys
import json
import csv
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "05_evaluation"))

from eval_utils.plotting import (
    generate_radar_chart_area_based,
    generate_combined_heatmap
)

# Output directories
OUTPUTS_DIR = project_root / "outputs"
EVAL_OUTPUTS_DIR = OUTPUTS_DIR / "evaluation"
COMBINED_PLOTS_DIR = EVAL_OUTPUTS_DIR / "combined_plots"
COMBINED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input directories for each evaluation
EVAL_DIRS = {
    "ISD": EVAL_OUTPUTS_DIR / "ISD_Evaluation_Embedding",  # Embedding-based ISD
    "TruthfulQA": EVAL_OUTPUTS_DIR / "TruthfulQA_Evaluation",
    "Cond. Safety": EVAL_OUTPUTS_DIR / "Conditional_Safety_Evaluation",
    "Length Ctrl": EVAL_OUTPUTS_DIR / "Length_Control_Evaluation",
    "AQI": EVAL_OUTPUTS_DIR / "AQI_Evaluation",
}

# Metric keys for each evaluation
METRIC_KEYS = {
    "ISD": "instruction_awareness_score",
    "TruthfulQA": "adaptation_score",
    "Cond. Safety": "adaptation_score",
    "Length Ctrl": "adaptation_score",
    "AQI": "aqi_score",
}

METHODS = ['SFT', 'DPO', 'PPO', 'GRPO', 'CITA']


def load_eval_metrics(eval_name: str, eval_dir: Path, metric_key: str) -> dict:
    """Load metrics from evaluation output directory.

    Handles multiple formats:
    1. Per-model directories: outputs/Eval/ModelName/metrics.json
    2. Combined file: outputs/Eval/metrics.json
    3. AQI format: outputs/Eval/ModelName/ModelName_metrics_summary.csv
    """
    scores = {}

    if not eval_dir.exists():
        print(f"  [WARN] {eval_name}: Directory not found at {eval_dir}")
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
        print(f"  [WARN] {eval_name}: No metrics found at {eval_dir}")

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


def main():
    print("=" * 70)
    print("GENERATE COMBINED EVALUATION PLOTS")
    print("=" * 70)
    print(f"Output directory: {COMBINED_PLOTS_DIR}")
    print()

    COMBINED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_deltas = {}      # For radar chart (improvement)
    eval_scores = {}      # For heatmap (absolute scores)

    for eval_name, eval_dir in EVAL_DIRS.items():
        metric_key = METRIC_KEYS[eval_name]
        print(f"Loading {eval_name}...")

        scores = load_eval_metrics(eval_name, eval_dir, metric_key)

        if not scores:
            print(f"  [SKIP] No scores found for {eval_name}")
            continue

        # Store absolute scores for heatmap
        eval_scores[eval_name] = scores
        print(f"  [OK] Loaded {len(scores)} models: {list(scores.keys())}")

        # Calculate deltas for radar chart
        deltas = calculate_deltas(scores)

        if len(deltas) >= 2:
            eval_deltas[eval_name] = deltas
            print(f"  [OK] Deltas: {deltas}")
        else:
            print(f"  [WARN] Not enough methods with both variants for radar chart")

    generated = {'heatmap': False, 'radar_area': False}

    # =========================================================================
    # Generate Heatmap (absolute scores)
    # =========================================================================
    if len(eval_scores) >= 2:
        heatmap_path = COMBINED_PLOTS_DIR / "heatmap"
        print(f"\n{'=' * 70}")
        print("Generating Heatmap (Absolute Scores)...")
        print(f"{'=' * 70}")

        generate_combined_heatmap(
            eval_scores=eval_scores,
            output_path=heatmap_path,
            normalize_per_column=True,
            show_raw_values=True
        )
        generated['heatmap'] = True
        print(f"  [OK] heatmap.{{pdf,png}}")
    else:
        print(f"\n[WARN] Need at least 2 evals for heatmap, got {len(eval_scores)}")

    # =========================================================================
    # Generate Radar Chart (Average Radius - instruction alignment efficiency)
    # =========================================================================
    if len(eval_deltas) >= 2:
        radar_area_path = COMBINED_PLOTS_DIR / "radar_area"
        print(f"\n{'=' * 70}")
        print("Generating Radar Chart (Average Radius - Instruction Alignment)...")
        print(f"{'=' * 70}")

        generate_radar_chart_area_based(
            eval_deltas=eval_deltas,
            output_path=radar_area_path,
            methods=METHODS,
            normalize=True
        )
        generated['radar_area'] = True
        print(f"  [OK] radar_area.{{pdf,png}}")
    else:
        print(f"\n[WARN] Need at least 2 evals for radar chart, got {len(eval_deltas)}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    plot_count = sum(1 for v in generated.values() if v)
    print(f"Generated: {plot_count} plots ({plot_count * 2} files: PDF + PNG each)")

    if generated['heatmap']:
        print(f"  - heatmap.{{pdf,png}}")
    if generated['radar_area']:
        print(f"  - radar_area.{{pdf,png}}")

    print(f"\nNote: HP ablation plots are generated separately by:")
    print(f"  python comparative_study/generate_hp_ablation_plots.py")


if __name__ == "__main__":
    main()
