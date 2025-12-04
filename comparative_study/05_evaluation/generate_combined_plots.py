"""
Generate Combined Plots for All Evaluations

Reads metrics from all 5 evaluation outputs and generates:
1. Radar chart: Instruction adaptation (NoInstruct → Instruct) improvement
2. Heatmap: Absolute scores across all models and evals

Both PDF (for Overleaf) and PNG (for sharing) formats are generated.

Usage:
    python comparative_study/05_evaluation/generate_combined_plots.py

Output:
    outputs/combined_plots/
    ├── radar.png      (for sharing)
    ├── radar.pdf      (for Overleaf)
    ├── heatmap.png
    └── heatmap.pdf
"""

import sys
import json
import csv
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "05_evaluation"))

from eval_utils.plotting import generate_improvement_radar_chart, generate_combined_heatmap

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


def main():
    print("=" * 80)
    print("GENERATING COMBINED PLOTS FOR ALL EVALUATIONS")
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

        print(f"\n✅ Radar chart saved to: {radar_path}")
    else:
        print(f"\n⚠️  Need at least 2 evals for radar chart, got {len(eval_deltas)}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Evals loaded: {len(eval_scores)}/5")
    print(f"Evals with deltas: {len(eval_deltas)}/5")
    print(f"\nOutput directory: {COMBINED_PLOTS_DIR}")
    if len(eval_scores) >= 2:
        print(f"  ✅ heatmap.png (for sharing)")
        print(f"  ✅ heatmap.pdf (for Overleaf)")
    if len(eval_deltas) >= 2:
        print(f"  ✅ radar.png   (for sharing)")
        print(f"  ✅ radar.pdf   (for Overleaf)")


if __name__ == "__main__":
    main()
