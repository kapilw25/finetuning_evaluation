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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "05_evaluation"))

from eval_utils.plotting import (
    generate_radar_chart_area_based,
    generate_combined_heatmap
)
from eval_utils.bootstrap import (
    compute_bootstrap_ci,
    compute_delta_bootstrap_ci,
    BootstrapResult
)

# Output directories
OUTPUTS_DIR = project_root / "outputs"
EVAL_OUTPUTS_DIR = OUTPUTS_DIR / "evaluation"
COMBINED_PLOTS_DIR = EVAL_OUTPUTS_DIR / "combined_plots"
COMBINED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input directories for each evaluation
EVAL_DIRS = {
    "ECLIPTICA (M₁)": EVAL_OUTPUTS_DIR / "ISD_Evaluation_Embedding",
    "TruthfulQA (M₂)": EVAL_OUTPUTS_DIR / "TruthfulQA_Evaluation",
    "Cond. Safety (M₃)": EVAL_OUTPUTS_DIR / "Conditional_Safety_Evaluation",
    "Length Ctrl (M₄)": EVAL_OUTPUTS_DIR / "Length_Control_Evaluation",
    "LITMUS (AQI-M₅)": EVAL_OUTPUTS_DIR / "AQI_Evaluation",
}

# Metric keys for each evaluation
METRIC_KEYS = {
    "ECLIPTICA (M₁)": "instruction_awareness_score",
    "TruthfulQA (M₂)": "adaptation_score",
    "Cond. Safety (M₃)": "adaptation_score",
    "Length Ctrl (M₄)": "adaptation_score",
    "LITMUS (AQI-M₅)": "aqi_score",
}

# Model name for figure titles
MODEL_NAME = "Llama-3.1-8B"

# Policy optimization methods only (SFT is a training stage, not a method)
METHODS = ['DPO', 'PPO', 'GRPO', 'CITA']

# Per-sample score keys to look for
PER_SAMPLE_KEYS = ['per_sample_scores', 'per_sample', 'scores', 'sample_scores']


def _load_truthfulqa_per_sample(model_dir: Path) -> Optional[List[float]]:
    """Load per-sample adaptation scores from TruthfulQA CSVs.

    Formula: per_sample = honest_uncertainty - confident_uncertainty
    """
    honest_csv = model_dir / "honest_responses.csv"
    confident_csv = model_dir / "confident_responses.csv"

    if not honest_csv.exists() or not confident_csv.exists():
        return None

    try:
        honest_df = pd.read_csv(honest_csv)
        confident_df = pd.read_csv(confident_csv)

        if 'uncertainty_total' not in honest_df.columns or 'uncertainty_total' not in confident_df.columns:
            return None

        # Align by question index (same order assumed)
        n_samples = min(len(honest_df), len(confident_df))
        per_sample = (
            honest_df['uncertainty_total'].values[:n_samples] -
            confident_df['uncertainty_total'].values[:n_samples]
        )
        return per_sample.tolist()
    except Exception:
        return None


def _load_cond_safety_per_sample(model_dir: Path) -> Optional[List[float]]:
    """Load per-sample adaptation scores from Conditional Safety CSVs.

    Formula: per_sample = strict_refusal_conf - permissive_refusal_conf
    """
    strict_csv = model_dir / "strict_responses.csv"
    permissive_csv = model_dir / "permissive_responses.csv"

    if not strict_csv.exists() or not permissive_csv.exists():
        return None

    try:
        strict_df = pd.read_csv(strict_csv)
        permissive_df = pd.read_csv(permissive_csv)

        if 'refusal_confidence' not in strict_df.columns or 'refusal_confidence' not in permissive_df.columns:
            return None

        n_samples = min(len(strict_df), len(permissive_df))
        per_sample = (
            strict_df['refusal_confidence'].values[:n_samples] -
            permissive_df['refusal_confidence'].values[:n_samples]
        )
        return per_sample.tolist()
    except Exception:
        return None


def _load_length_ctrl_per_sample(model_dir: Path) -> Optional[List[float]]:
    """Load per-sample adaptation scores from Length Control CSVs.

    Formula: per_sample = detailed_words / max(concise_words, 1)
    """
    concise_csv = model_dir / "concise_responses.csv"
    detailed_csv = model_dir / "detailed_responses.csv"

    if not concise_csv.exists() or not detailed_csv.exists():
        return None

    try:
        concise_df = pd.read_csv(concise_csv)
        detailed_df = pd.read_csv(detailed_csv)

        if 'word_count' not in concise_df.columns or 'word_count' not in detailed_df.columns:
            return None

        n_samples = min(len(concise_df), len(detailed_df))
        concise_words = concise_df['word_count'].values[:n_samples]
        detailed_words = detailed_df['word_count'].values[:n_samples]

        # Avoid division by zero
        per_sample = detailed_words / np.maximum(concise_words, 1)
        return per_sample.tolist()
    except Exception:
        return None


def _load_isd_per_sample(model_dir: Path, model_name: str) -> Optional[List[float]]:
    """Load per-sample fidelity scores from ISD metrics.json.

    ISD saves per_sample_fidelity in metrics.json (after re-run with updated script).
    """
    metrics_file = model_dir / f"{model_name}_isd_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        # Look for per_sample_fidelity key
        if 'per_sample_fidelity' in data and isinstance(data['per_sample_fidelity'], list):
            return data['per_sample_fidelity']

        return None
    except Exception:
        return None


def load_eval_metrics_with_samples(
    eval_name: str,
    eval_dir: Path,
    metric_key: str
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Load metrics AND per-sample scores from evaluation output directory.

    Reads per-sample data from CSV files (Option B):
    - TruthfulQA: honest_responses.csv + confident_responses.csv
    - Cond. Safety: strict_responses.csv + permissive_responses.csv
    - Length Control: concise_responses.csv + detailed_responses.csv
    - ISD: per_sample_fidelity from metrics.json (after re-run)
    - AQI: No per-sample data (cluster-based metric)

    Returns:
        Tuple of (aggregate_scores, per_sample_scores)
        - aggregate_scores: {model_name: score}
        - per_sample_scores: {model_name: [score1, score2, ...]} or empty if not available
    """
    scores = {}
    per_sample = {}

    if not eval_dir.exists():
        print(f"  [WARN] {eval_name}: Directory not found at {eval_dir}")
        return scores, per_sample

    # Try per-model directory structure
    for model_dir in eval_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name

            # ================================================================
            # Load aggregate score from metrics.json or ISD/AQI formats
            # ================================================================

            # Try metrics.json first
            metrics_file = model_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                score = data.get(metric_key)
                if score is not None:
                    scores[model_name] = score

                # Check if per-sample already in metrics.json
                for key in PER_SAMPLE_KEYS:
                    if key in data and isinstance(data[key], list):
                        per_sample[model_name] = data[key]
                        break

            # Try ISD format: ModelName_isd_metrics.json
            isd_file = model_dir / f"{model_name}_isd_metrics.json"
            if isd_file.exists() and model_name not in scores:
                with open(isd_file, 'r') as f:
                    data = json.load(f)
                score = data.get(metric_key)
                if score is not None:
                    scores[model_name] = score
                # ISD: No per-sample fidelity saved (embedding-based)

            # Try AQI CSV format
            csv_file = model_dir / f"{model_name}_metrics_summary.csv"
            if csv_file.exists() and model_name not in scores:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('Category', '').lower() == 'overall':
                            aqi_score = row.get('AQI [0-100] (↑)')
                            if aqi_score:
                                scores[model_name] = float(aqi_score)
                            break
                # AQI: No per-sample (cluster-based)

            # ================================================================
            # Load per-sample from CSV/JSON files (if not already loaded)
            # ================================================================
            if model_name not in per_sample:
                if "TruthfulQA" in eval_name:
                    csv_per_sample = _load_truthfulqa_per_sample(model_dir)
                    if csv_per_sample:
                        per_sample[model_name] = csv_per_sample

                elif "Cond. Safety" in eval_name:
                    csv_per_sample = _load_cond_safety_per_sample(model_dir)
                    if csv_per_sample:
                        per_sample[model_name] = csv_per_sample

                elif "Length Ctrl" in eval_name:
                    csv_per_sample = _load_length_ctrl_per_sample(model_dir)
                    if csv_per_sample:
                        per_sample[model_name] = csv_per_sample

                elif "ECLIPTICA" in eval_name:
                    # ISD/ECLIPTICA: Load per_sample_fidelity from metrics.json
                    isd_per_sample = _load_isd_per_sample(model_dir, model_name)
                    if isd_per_sample:
                        per_sample[model_name] = isd_per_sample

                # AQI/LITMUS: Skip (cluster-based metric, no per-sample scores)

    return scores, per_sample


def calculate_deltas(scores: dict) -> dict:
    """Calculate improvement delta (Instruct - NoInstruct) for each method."""
    deltas = {}
    for method in METHODS:
        no_key = f"{method}_NoInstruct"
        inst_key = f"{method}_Instruct"

        if no_key in scores and inst_key in scores:
            deltas[method] = scores[inst_key] - scores[no_key]

    return deltas


def calculate_deltas_with_ci(
    scores: Dict[str, float],
    per_sample: Dict[str, List[float]],
    n_bootstrap: int = 1000
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Calculate improvement delta AND bootstrap CI for each method.

    Returns:
        Tuple of (deltas, delta_ci)
        - deltas: {method: delta_value}
        - delta_ci: {method: (ci_lower, ci_upper)} or empty if no per-sample data
    """
    deltas = {}
    delta_ci = {}

    for method in METHODS:
        no_key = f"{method}_NoInstruct"
        inst_key = f"{method}_Instruct"

        if no_key in scores and inst_key in scores:
            deltas[method] = scores[inst_key] - scores[no_key]

            # Compute bootstrap CI if per-sample data is available
            if no_key in per_sample and inst_key in per_sample:
                result = compute_delta_bootstrap_ci(
                    scores_instruct=per_sample[inst_key],
                    scores_noinstruct=per_sample[no_key],
                    n_bootstrap=n_bootstrap
                )
                delta_ci[method] = (result.ci_lower, result.ci_upper)

    return deltas, delta_ci


def calculate_score_ci(
    per_sample: Dict[str, List[float]],
    n_bootstrap: int = 1000
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate bootstrap CI for each model's aggregate score.

    Returns:
        Dict mapping model_name to (ci_lower, ci_upper)
    """
    score_ci = {}
    for model_name, samples in per_sample.items():
        if samples and len(samples) > 1:
            result = compute_bootstrap_ci(samples, n_bootstrap=n_bootstrap)
            score_ci[model_name] = (result.ci_lower, result.ci_upper)
    return score_ci


def main():
    print("=" * 70)
    print("GENERATE COMBINED EVALUATION PLOTS")
    print("=" * 70)
    print(f"Output directory: {COMBINED_PLOTS_DIR}")
    print()

    COMBINED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_deltas = {}      # For radar chart (improvement)
    eval_scores = {}      # For heatmap (absolute scores)
    eval_delta_ci = {}    # For radar chart CI bands
    eval_score_ci = {}    # For heatmap CI values
    has_any_ci = False    # Track if any per-sample data was found

    for eval_name, eval_dir in EVAL_DIRS.items():
        metric_key = METRIC_KEYS[eval_name]
        print(f"Loading {eval_name}...")

        # Load both aggregate and per-sample scores
        scores, per_sample = load_eval_metrics_with_samples(eval_name, eval_dir, metric_key)

        if not scores:
            print(f"  [SKIP] No scores found for {eval_name}")
            continue

        # Store absolute scores for heatmap
        eval_scores[eval_name] = scores
        print(f"  [OK] Loaded {len(scores)} models: {list(scores.keys())}")

        # Check for per-sample data
        if per_sample:
            print(f"  [CI] Per-sample data found for {len(per_sample)} models")
            has_any_ci = True

            # Compute score CI for heatmap
            score_ci = calculate_score_ci(per_sample)
            if score_ci:
                eval_score_ci[eval_name] = score_ci

        # Calculate deltas (with CI if per-sample available)
        if per_sample:
            deltas, delta_ci = calculate_deltas_with_ci(scores, per_sample)
            if delta_ci:
                eval_delta_ci[eval_name] = delta_ci
                print(f"  [CI] Delta CI computed for {len(delta_ci)} methods")
        else:
            deltas = calculate_deltas(scores)

        if len(deltas) >= 2:
            eval_deltas[eval_name] = deltas
            print(f"  [OK] Deltas: {deltas}")
        else:
            print(f"  [WARN] Not enough methods with both variants for radar chart")

    # Summary of CI availability
    if has_any_ci:
        print(f"\n{'=' * 70}")
        print("BOOTSTRAP CI STATUS")
        print(f"{'=' * 70}")
        print(f"  Evals with delta CI: {list(eval_delta_ci.keys())}")
        print(f"  Evals with score CI: {list(eval_score_ci.keys())}")
    else:
        print(f"\n[INFO] No per-sample data found. Plots will not include error bars.")
        print(f"       To enable CI, re-run evaluations with per_sample_scores in metrics.json")

    generated = {'heatmap': False, 'heatmap_no_ci': False, 'radar_area': False}

    # =========================================================================
    # Generate Heatmaps (absolute scores) - WITH and WITHOUT CI
    # =========================================================================
    if len(eval_scores) >= 2:
        # Heatmap WITH CI values
        heatmap_path = COMBINED_PLOTS_DIR / "heatmap"
        print(f"\n{'=' * 70}")
        print("Generating Heatmap WITH CI (Absolute Scores)...")
        if eval_score_ci:
            print("  [CI] Including confidence intervals in heatmap")
        print(f"{'=' * 70}")

        generate_combined_heatmap(
            eval_scores=eval_scores,
            output_path=heatmap_path,
            normalize_per_column=True,
            show_raw_values=True,
            score_ci=eval_score_ci if eval_score_ci else None,
            model_name=MODEL_NAME
        )
        generated['heatmap'] = True
        print(f"  [OK] heatmap.{{pdf,png}}")

        # Heatmap WITHOUT CI values
        heatmap_no_ci_path = COMBINED_PLOTS_DIR / "heatmap_no_ci"
        print(f"\n{'=' * 70}")
        print("Generating Heatmap WITHOUT CI (Absolute Scores)...")
        print(f"{'=' * 70}")

        generate_combined_heatmap(
            eval_scores=eval_scores,
            output_path=heatmap_no_ci_path,
            normalize_per_column=True,
            show_raw_values=True,
            score_ci=None,  # No CI values
            model_name=MODEL_NAME
        )
        generated['heatmap_no_ci'] = True
        print(f"  [OK] heatmap_no_ci.{{pdf,png}}")
    else:
        print(f"\n[WARN] Need at least 2 evals for heatmap, got {len(eval_scores)}")

    # =========================================================================
    # Generate Radar Chart (Average Radius - instruction alignment efficiency)
    # NOTE: CI bands disabled - radar already cluttered with 4 methods x 5 evals
    # =========================================================================
    if len(eval_deltas) >= 2:
        radar_area_path = COMBINED_PLOTS_DIR / "radar_area"
        print(f"\n{'=' * 70}")
        print("Generating Radar Chart (Average Radius - Instruction Alignment)...")
        print("  [INFO] CI bands disabled for radar (already cluttered)")
        print(f"{'=' * 70}")

        generate_radar_chart_area_based(
            eval_deltas=eval_deltas,
            output_path=radar_area_path,
            methods=METHODS,
            normalize=True,
            delta_ci=None,  # Skip CI bands - radar too cluttered
            model_name=MODEL_NAME
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
        print(f"  - heatmap.{{pdf,png}} (with CI)")
    if generated['heatmap_no_ci']:
        print(f"  - heatmap_no_ci.{{pdf,png}} (without CI)")
    if generated['radar_area']:
        print(f"  - radar_area.{{pdf,png}}")

    print(f"\nNote: HP ablation plots are generated separately by:")
    print(f"  python comparative_study/generate_hp_ablation_plots.py")


if __name__ == "__main__":
    main()
