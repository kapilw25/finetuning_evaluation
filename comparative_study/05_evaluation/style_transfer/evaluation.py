"""
Style Transfer Evaluation with Instruction Variants

Tests: Does model follow length/style instructions?

Instruction Variants:
    CONCISE: "Answer in AT MOST 50 words. Be extremely brief and direct."
    DETAILED: "Provide a comprehensive response of at least 200 words."

Metrics:
    - Word count with each instruction type
    - Length Adaptation Score = DETAILED_words / CONCISE_words
    - Higher adaptation = model follows instructions better

Dataset: AlpacaEval (500 questions)

Usage:
    # Sanity check (100 prompts)
    python comparative_study/05_evaluation/style_transfer/evaluation.py \
    --models CITA_Instruct CITA_NoInstruct DPO_Instruct DPO_NoInstruct

    # Full evaluation (500 prompts)
    python comparative_study/05_evaluation/style_transfer/evaluation.py \
    --models CITA_Instruct CITA_NoInstruct DPO_Instruct DPO_NoInstruct \
    --mode full
"""

import sys
import json
import torch
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))
sys.path.insert(0, str(project_root / "comparative_study" / "05_evaluation"))

from eval_utils import (
    MODELS, load_model_for_eval, unload_model,
    setup_training_logger, restore_logging,
    save_checkpoint, load_checkpoint, delete_checkpoint,
    batch_generate, cleanup_gpu, format_chat_messages, verify_hf_repos,
    add_validation_columns, get_validation_summary,
    show_cached_data_menu, show_mode_selection_menu, show_checkpoint_resume_menu,
    get_model_colors, filter_model_keys
)
from eval_utils.checkpoint import get_checkpoint_dir


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class StyleResponse:
    """Response data for a single style test case"""
    prompt_idx: int
    prompt: str
    variant: str
    response: str
    response_length: int
    word_count: int
    generation_time: float


@dataclass
class StyleModelResult:
    """Complete evaluation result for a model"""
    model_name: str
    variant: str
    responses: List[StyleResponse]
    total_prompts: int
    evaluation_time: float
    timestamp: str


# =============================================================================
# CONFIGURATION
# =============================================================================

EVAL_OUTPUT_DIR = project_root / "outputs" / "Style_Transfer_Evaluation"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Instruction variants
INSTRUCTION_VARIANTS = {
    "CONCISE": (
        "You must answer in AT MOST 50 words. Be extremely concise and direct. "
        "Every word must count. Do not elaborate or provide examples. "
        "Get straight to the point."
    ),
    "DETAILED": (
        "Provide a comprehensive, detailed response of at least 200 words. "
        "Explain thoroughly with examples, context, and nuance. "
        "Cover multiple aspects of the topic. Be thorough and informative."
    )
}


# =============================================================================
# INSTRUCTION SYNTHESIS
# =============================================================================

def synthesize_length_instruction(variant: str) -> str:
    """
    Get instruction for length variant

    Args:
        variant: 'CONCISE' or 'DETAILED'

    Returns:
        Complete system instruction
    """
    return INSTRUCTION_VARIANTS[variant]


def count_words(text: str) -> int:
    """Count words in text"""
    if not isinstance(text, str):
        return 0
    return len(text.split())


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_alpacaeval(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load AlpacaEval dataset"""
    print("\n" + "=" * 80)
    print("Loading AlpacaEval Dataset")
    print("=" * 80)

    url = 'https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json'

    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data)

    if max_samples:
        df = df.head(max_samples)

    print(f"Loaded {len(df)} prompts")

    return df


# =============================================================================
# RESPONSE GENERATION
# =============================================================================

def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    model_key: str,
    variant: str,
    use_instruction: bool,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    checkpoint_interval: int = 100
) -> List[StyleResponse]:
    """Generate responses for all prompts with batch processing and checkpointing"""

    # Check for existing checkpoint
    checkpoint = load_checkpoint(model_key, eval_type="style_transfer", variant=variant)

    if checkpoint and checkpoint['completed']:
        choice = show_checkpoint_resume_menu(
            model_key=f"{model_key}/{variant}",
            n_responses=checkpoint['n_completed'],
            eval_type="Style Transfer"
        )
        if choice == "1":
            return [StyleResponse(**r) for r in checkpoint['responses']]
        else:  # choice == "2"
            delete_checkpoint(model_key, eval_type="style_transfer", variant=variant)
            checkpoint = None

    # Resume from checkpoint if exists
    if checkpoint and not checkpoint['completed']:
        responses = [StyleResponse(**r) for r in checkpoint['responses']]
        start_idx = len(responses)
        print(f"Resuming from {start_idx}/{len(prompts)}")
    else:
        responses = []
        start_idx = 0

    # Format all remaining prompts for batch processing
    remaining_prompts = prompts[start_idx:]

    messages_list = []
    for prompt in remaining_prompts:
        if use_instruction:
            instruction = synthesize_length_instruction(variant)
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        messages_list.append(messages)

    # Format all messages at once
    formatted_prompts = format_chat_messages(tokenizer, messages_list)

    # Batch generate
    batch_responses = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=formatted_prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        show_progress=True,
        desc=f"{model_key}/{variant}",
        checkpoint_interval=checkpoint_interval
    )

    # Convert to StyleResponse objects
    for i, response_text in enumerate(batch_responses):
        prompt = remaining_prompts[i]

        responses.append(StyleResponse(
            prompt_idx=start_idx + i,
            prompt=prompt,
            variant=variant,
            response=response_text,
            response_length=len(response_text),
            word_count=count_words(response_text),
            generation_time=0.0  # Batch timing not per-item
        ))

    # Final checkpoint
    save_checkpoint(
        model_key,
        [asdict(r) for r in responses],
        len(prompts),
        eval_type="style_transfer",
        variant=variant,
        completed=True
    )

    return responses


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_metrics(
    concise_responses: List[StyleResponse],
    detailed_responses: List[StyleResponse]
) -> Dict:
    """Calculate length adaptation metrics"""

    concise_df = pd.DataFrame([asdict(r) for r in concise_responses])
    detailed_df = pd.DataFrame([asdict(r) for r in detailed_responses])

    # Word counts
    concise_avg_words = concise_df['word_count'].mean()
    detailed_avg_words = detailed_df['word_count'].mean()

    concise_std_words = concise_df['word_count'].std()
    detailed_std_words = detailed_df['word_count'].std()

    # Adaptation score (ratio of detailed to concise)
    # Higher = better adaptation
    adaptation_score = detailed_avg_words / max(concise_avg_words, 1)

    # Per-prompt adaptation
    per_prompt_ratios = []
    for c, d in zip(concise_responses, detailed_responses):
        if c.word_count > 0:
            ratio = d.word_count / c.word_count
        else:
            ratio = d.word_count if d.word_count > 0 else 1.0
        per_prompt_ratios.append(ratio)

    avg_ratio = np.mean(per_prompt_ratios)
    std_ratio = np.std(per_prompt_ratios)

    return {
        "concise_avg_words": float(concise_avg_words),
        "concise_std_words": float(concise_std_words),
        "detailed_avg_words": float(detailed_avg_words),
        "detailed_std_words": float(detailed_std_words),
        "adaptation_score": float(adaptation_score),
        "avg_ratio": float(avg_ratio),
        "std_ratio": float(std_ratio),
        "n_prompts": len(concise_responses)
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_style_evaluation(
    model_keys: List[str],
    prompts_df: pd.DataFrame,
    batch_size: int = 8
) -> Dict[str, Dict]:
    """Run evaluation on all models"""

    all_results = {}
    prompts = prompts_df['instruction'].tolist()

    for model_key in model_keys:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_key}")
        print(f"{'=' * 80}")

        try:
            model_config = MODELS[model_key]
            use_instruction = model_config["use_instruction"]

            # Load model
            model, tokenizer = load_model_for_eval(model_key)

            # Generate for both variants
            concise_responses = generate_responses(
                model, tokenizer, prompts,
                model_key, "CONCISE", use_instruction,
                batch_size=batch_size
            )

            detailed_responses = generate_responses(
                model, tokenizer, prompts,
                model_key, "DETAILED", use_instruction,
                batch_size=batch_size
            )

            # Unload model
            unload_model(model)
            del model
            del tokenizer
            cleanup_gpu()

            # Check for cached metrics
            metrics_path = EVAL_OUTPUT_DIR / model_key / "metrics.json"
            if metrics_path.exists():
                print(f"\nLoading cached metrics for {model_key}")
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                print(f"  Loaded from cache")
            else:
                # Calculate metrics
                metrics = calculate_metrics(concise_responses, detailed_responses)

            all_results[model_key] = {
                "metrics": metrics,
                "concise_responses": concise_responses,
                "detailed_responses": detailed_responses
            }

            # Print summary
            print(f"\n{'-' * 40}")
            print(f"Summary for {model_key}:")
            print(f"  CONCISE avg words: {metrics['concise_avg_words']:.1f}")
            print(f"  DETAILED avg words: {metrics['detailed_avg_words']:.1f}")
            print(f"  Adaptation Score: {metrics['adaptation_score']:.2f}")
            print(f"{'-' * 40}")

        except RuntimeError as e:
            print(f"\nFailed to evaluate {model_key}: {e}")
            print(f"   Skipping this model...")
            continue

    return all_results


# =============================================================================
# PLOTTING
# =============================================================================

def generate_comparison_plots(all_results: Dict, output_dir: Path, stratified_metrics: Dict[str, Dict] = None):
    """Generate comparison plot with Overall vs Valid-only bars"""
    import matplotlib.pyplot as plt

    if len(all_results) < 2:
        print("Need at least 2 models for comparison")
        return

    models = list(all_results.keys())
    adaptation_scores = [all_results[m]['metrics']['adaptation_score'] for m in models]

    # Get valid-only scores from stratified metrics
    valid_adaptation = []
    valid_rates = []
    std_scores = []
    for m in models:
        std_scores.append(all_results[m]['metrics'].get('std_ratio', 0))

        if stratified_metrics and m in stratified_metrics:
            va = stratified_metrics[m].get('valid_adaptation_score')
            # Average valid rate between concise and detailed
            cr = stratified_metrics[m].get('concise_valid_rate', 1.0)
            dr = stratified_metrics[m].get('detailed_valid_rate', 1.0)
            vr = (cr + dr) / 2
            valid_adaptation.append(va if va is not None else adaptation_scores[models.index(m)])
            valid_rates.append(vr)
        else:
            valid_adaptation.append(adaptation_scores[models.index(m)])
            valid_rates.append(1.0)

    # Sort by overall adaptation (ascending = best on right)
    sorted_indices = np.argsort(adaptation_scores)
    models_sorted = [models[i] for i in sorted_indices]
    adaptation_sorted = [adaptation_scores[i] for i in sorted_indices]
    valid_adaptation_sorted = [valid_adaptation[i] for i in sorted_indices]
    valid_rates_sorted = [valid_rates[i] for i in sorted_indices]
    std_sorted = [std_scores[i] for i in sorted_indices]

    # Get colors using shared utility
    colors_sorted = get_model_colors(models_sorted)

    # Single plot with Overall and Valid-only bars
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(models_sorted))
    bar_width = 0.35

    # Overall bars (solid) with error bars
    bars_overall = ax.bar(x - bar_width/2, adaptation_sorted, bar_width,
                          color=colors_sorted, edgecolor='black', linewidth=1.5, label='Overall',
                          yerr=std_sorted, capsize=3, error_kw={'linewidth': 1.5})

    # Valid-only bars (hatched)
    bars_valid = ax.bar(x + bar_width/2, valid_adaptation_sorted, bar_width,
                        color=colors_sorted, edgecolor='black', linewidth=1.5,
                        hatch='///', alpha=0.7, label='Valid-only')

    ax.set_ylabel('Length Adaptation Score', fontsize=14, fontweight='bold')
    ax.set_title('Style Transfer: Adaptation Score - Overall vs Valid-Only (Higher = Better)', fontsize=16, fontweight='bold', pad=15)
    max_adapt = max(max(adaptation_sorted), max(valid_adaptation_sorted)) if adaptation_sorted else 3.0
    ax.set_ylim(0, max_adapt * 1.4)

    # Reference line at 1.0 (no adaptation)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No Adaptation')

    # Add Perfect score annotation
    ax.text(0.98, 0.98, 'Target > 4.0', transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)

    # Add value labels
    for i, (bar_o, bar_v, score_o, score_v, vr) in enumerate(zip(bars_overall, bars_valid,
                                                                   adaptation_sorted, valid_adaptation_sorted, valid_rates_sorted)):
        # Overall label
        ax.text(bar_o.get_x() + bar_o.get_width()/2., bar_o.get_height() + 0.05,
                f'{score_o:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Valid-only label with valid rate
        ax.text(bar_v.get_x() + bar_v.get_width()/2., bar_v.get_height() + 0.05,
                f'{score_v:.2f}\n({vr:.0%})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path = output_dir / "style_transfer_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")

    # Print ranking
    print(f"\nAdaptation Ranking (Best to Worst):")
    for rank, (model, score) in enumerate(zip(reversed(models_sorted), reversed(adaptation_sorted)), 1):
        print(f"   {rank}. {model}: {score:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Style Transfer Evaluation")
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity",
                       help="sanity (100 prompts) or full (500 prompts)")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to evaluate")
    parser.add_argument("--samples", type=int, default=None,
                       help="Custom sample count (overrides --mode)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")

    args = parser.parse_args()

    # Setup logging
    log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
        run_name="style_transfer_evaluation",
        project_root=project_root
    )

    try:
        # Cleanup handler
        checkpoint_dir = get_checkpoint_dir("style_transfer")
        results_dir = EVAL_OUTPUT_DIR

        checkpoints_exist = checkpoint_dir.exists() and any(
            f.name.endswith("_checkpoint.json") for f in checkpoint_dir.iterdir() if f.is_file()
        ) if checkpoint_dir.exists() else False
        results_exist = results_dir.exists() and any(results_dir.iterdir())

        if checkpoints_exist or results_exist:
            show_cached_data_menu(
                checkpoint_dir=checkpoint_dir,
                results_dir=results_dir,
                eval_type="Style Transfer",
                checkpoint_suffix="_checkpoint.json",
                metrics_filename="metrics.json",
                plot_filename="style_transfer_comparison.png"
            )

        # Interactive mode selection
        mode, _ = show_mode_selection_menu(
            eval_name="STYLE TRANSFER",
            sanity_desc="100 prompts x 2 variants = 200 test cases (~15 min)",
            full_desc="500 prompts x 2 variants = 1,000 test cases (~60 min)"
        )
        args.mode = mode
        args.samples = 100 if mode == "sanity" else 500

        # Determine sample count
        if args.samples:
            max_samples = args.samples
        elif args.mode == "sanity":
            max_samples = 100
        else:
            max_samples = 500

        print(f"\n{'=' * 80}")
        print(f"Style Transfer Evaluation")
        print(f"{'=' * 80}")
        print(f"Mode: {args.mode}")
        print(f"Samples: {max_samples}")
        print(f"{'=' * 80}")

        # Load dataset
        prompts_df = load_alpacaeval(max_samples=max_samples)

        # Determine models
        model_keys = filter_model_keys(args.models, MODELS, list(MODELS.keys()))

        # Pre-flight verification of HuggingFace repos
        model_keys = verify_hf_repos(model_keys, interactive=True)
        if not model_keys:
            print("No valid models to evaluate. Exiting.")
            sys.exit(1)

        print(f"\nModels to evaluate: {model_keys}")
        print(f"Batch size: {args.batch_size}")

        # Run evaluation
        all_results = run_style_evaluation(
            model_keys, prompts_df,
            batch_size=args.batch_size
        )

        # Save results with validation columns and calculate valid-only metrics
        all_stratified = {}
        for model_key, results in all_results.items():
            model_dir = EVAL_OUTPUT_DIR / model_key
            model_dir.mkdir(exist_ok=True)

            # Convert to DataFrames and add validation columns
            concise_df = pd.DataFrame([asdict(r) for r in results['concise_responses']])
            detailed_df = pd.DataFrame([asdict(r) for r in results['detailed_responses']])

            # Add validation columns for gibberish/repetition detection
            concise_df = add_validation_columns(concise_df, response_column='response')
            detailed_df = add_validation_columns(detailed_df, response_column='response')

            # Get validation summaries
            concise_val = get_validation_summary(concise_df)
            detailed_val = get_validation_summary(detailed_df)

            # Calculate VALID-ONLY metrics
            concise_valid = concise_df[concise_df['is_valid']]
            detailed_valid = detailed_df[detailed_df['is_valid']]

            # Overall metrics
            overall_concise_words = concise_df['word_count'].mean()
            overall_detailed_words = detailed_df['word_count'].mean()
            overall_adaptation = overall_detailed_words / max(overall_concise_words, 1)

            # Valid-only metrics
            if len(concise_valid) > 0 and len(detailed_valid) > 0:
                valid_concise_words = concise_valid['word_count'].mean()
                valid_detailed_words = detailed_valid['word_count'].mean()
                valid_adaptation = valid_detailed_words / max(valid_concise_words, 1)
            else:
                valid_concise_words = None
                valid_detailed_words = None
                valid_adaptation = None

            # Print comparison
            print(f"\n{model_key} Results:")
            print(f"  CONCISE - Valid rate: {concise_val['valid_rate']:.1%}, Avg words: Overall {overall_concise_words:.1f}, Valid-only {f'{valid_concise_words:.1f}' if valid_concise_words else 'N/A'}")
            print(f"  DETAILED - Valid rate: {detailed_val['valid_rate']:.1%}, Avg words: Overall {overall_detailed_words:.1f}, Valid-only {f'{valid_detailed_words:.1f}' if valid_detailed_words else 'N/A'}")
            print(f"  Adaptation - Overall: {overall_adaptation:.2f}, Valid-only: {f'{valid_adaptation:.2f}' if valid_adaptation else 'N/A'}")

            # Store stratified metrics
            all_stratified[model_key] = {
                'concise_valid_rate': concise_val['valid_rate'],
                'detailed_valid_rate': detailed_val['valid_rate'],
                'valid_concise_words': valid_concise_words,
                'valid_detailed_words': valid_detailed_words,
                'valid_adaptation_score': valid_adaptation
            }

            # Save metrics with both overall and valid-only
            metrics_with_stratified = results['metrics'].copy()
            # Validation rates
            metrics_with_stratified['concise_valid_rate'] = concise_val['valid_rate']
            metrics_with_stratified['concise_gibberish_rate'] = concise_val['gibberish_rate']
            metrics_with_stratified['concise_repetitive_rate'] = concise_val['repetitive_rate']
            metrics_with_stratified['detailed_valid_rate'] = detailed_val['valid_rate']
            metrics_with_stratified['detailed_gibberish_rate'] = detailed_val['gibberish_rate']
            metrics_with_stratified['detailed_repetitive_rate'] = detailed_val['repetitive_rate']
            # Valid-only metrics
            metrics_with_stratified['valid_concise_words'] = valid_concise_words
            metrics_with_stratified['valid_detailed_words'] = valid_detailed_words
            metrics_with_stratified['valid_adaptation_score'] = valid_adaptation

            with open(model_dir / "metrics.json", 'w') as f:
                json.dump(metrics_with_stratified, f, indent=2)

            # Save responses with validation columns
            concise_df.to_csv(model_dir / "concise_responses.csv", index=False)
            detailed_df.to_csv(model_dir / "detailed_responses.csv", index=False)

        # Generate plots
        if len(all_results) >= 2:
            generate_comparison_plots(all_results, EVAL_OUTPUT_DIR, all_stratified)

        # Final summary
        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Results saved to: {EVAL_OUTPUT_DIR}")

        # Summary table
        print(f"\n{'Model':<20} {'CONCISE':<12} {'DETAILED':<12} {'Adaptation':<12}")
        print("-" * 56)
        for model_key, results in all_results.items():
            m = results['metrics']
            print(f"{model_key:<20} {m['concise_avg_words']:.1f}{'':>5} "
                  f"{m['detailed_avg_words']:.1f}{'':>5} "
                  f"{m['adaptation_score']:.2f}")

    finally:
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"\nLog saved to: {log_filename}")


if __name__ == "__main__":
    main()
