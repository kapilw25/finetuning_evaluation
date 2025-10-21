"""
Regenerate all publication-quality plots from cached evaluation results

Usage:
    python3 comparative_study/05_evaluation/llm_as_judge/regenerate_plots.py
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add utils to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "05_evaluation" / "llm_as_judge" / "utils"))

from statistical_analysis import run_statistical_analysis

def main():
    results_dir = Path(__file__).parent / "DualMetric_Evaluation_Results"

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    print("="*80)
    print("REGENERATING PLOTS FROM CACHED RESULTS")
    print("="*80)

    # Load results from saved CSVs
    results = {}
    models = ['SFT_Baseline', 'DPO_Baseline', 'CITA_Baseline']

    for model in models:
        model_dir = results_dir / model
        harm_csv = model_dir / "harmlessness_results.csv"
        help_csv = model_dir / "helpfulness_results.csv"

        if not harm_csv.exists() or not help_csv.exists():
            print(f"⚠️  Skipping {model} - results not found")
            continue

        # Load dataframes
        harm_df = pd.read_csv(harm_csv)
        help_df = pd.read_csv(help_csv)

        # Parse harm_categories from string to list
        harm_df['harm_categories'] = harm_df['harm_categories'].apply(eval)

        results[model] = {
            'harmlessness_df': harm_df,
            'helpfulness_df': help_df,
            'summary': {
                'harmlessness_mean': harm_df['refusal_score'].mean(),
                'helpfulness_mean': help_df['helpfulness_score'].mean()
            }
        }

        print(f"✅ Loaded {model}: {len(harm_df)} harmlessness + {len(help_df)} helpfulness samples")

    if not results:
        print("\n❌ No results found to regenerate plots from")
        return

    # Run statistical analysis (will generate all plots)
    run_statistical_analysis(results, results_dir)

    print("\n" + "="*80)
    print("✅ ALL PLOTS REGENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nPlots saved to: {results_dir}")

if __name__ == "__main__":
    main()
