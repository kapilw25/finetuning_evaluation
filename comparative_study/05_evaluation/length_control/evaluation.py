"""
Length Control Evaluation with Instruction Variants

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
    python comparative_study/05_evaluation/length_control/evaluation.py \
    --models SFT_Instruct SFT_NoInstruct DPO_Instruct DPO_NoInstruct \
             PPO_Instruct PPO_NoInstruct GRPO_Instruct GRPO_NoInstruct \
             CITA_Instruct CITA_NoInstruct \
    --mode sanity

    # Full evaluation (500 prompts)
    python comparative_study/05_evaluation/length_control/evaluation.py \
    --models SFT_Instruct SFT_NoInstruct DPO_Instruct DPO_NoInstruct \
             PPO_Instruct PPO_NoInstruct GRPO_Instruct GRPO_NoInstruct \
             CITA_Instruct CITA_NoInstruct \
    --mode full

Available models: SFT_NoInstruct, SFT_Instruct, DPO_NoInstruct, DPO_Instruct,
                  PPO_NoInstruct, PPO_Instruct, GRPO_NoInstruct, GRPO_Instruct,
                  CITA_NoInstruct, CITA_Instruct
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
    filter_model_keys,
    get_length_control_max_samples,
    generate_comparison_plots as _generate_comparison_plots,  # shared plotting function
    generate_boxviolin_chart as _generate_boxviolin_chart
)
from eval_utils.bootstrap import compute_bootstrap_ci
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

EVAL_OUTPUT_DIR = project_root / "outputs" / "evaluation" / "Length_Control_Evaluation"
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
    checkpoint = load_checkpoint(model_key, eval_type="length_control", variant=variant)

    if checkpoint and checkpoint['completed']:
        choice = show_checkpoint_resume_menu(
            model_key=f"{model_key}/{variant}",
            n_responses=checkpoint['n_completed'],
            eval_type="Length Control"
        )
        if choice == "1":
            return [StyleResponse(**r) for r in checkpoint['responses']]
        else:  # choice == "2"
            delete_checkpoint(model_key, eval_type="length_control", variant=variant)
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

    # Create checkpoint callback for intermediate saves
    def checkpoint_cb(batch_responses_so_far):
        temp_responses = responses.copy()
        for i, response_text in enumerate(batch_responses_so_far):
            prompt = remaining_prompts[i]
            temp_responses.append(StyleResponse(
                prompt_idx=start_idx + i,
                prompt=prompt,
                variant=variant,
                response=response_text,
                response_length=len(response_text),
                word_count=count_words(response_text),
                generation_time=0.0
            ))
        save_checkpoint(
            model_key,
            [asdict(r) for r in temp_responses],
            len(prompts),
            eval_type="length_control",
            variant=variant,
            completed=False
        )

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
        checkpoint_callback=checkpoint_cb,
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
        eval_type="length_control",
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
    """Generate Length Control comparison plot using shared plotting function with Bootstrap CI"""
    if len(all_results) < 2:
        print("Need at least 2 models for comparison")
        return

    models = list(all_results.keys())
    adaptation_scores = [all_results[m]['metrics']['adaptation_score'] for m in models]

    # Get valid-only scores from stratified metrics
    valid_adaptation = []
    valid_rates = []
    for m in models:
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

    # Compute Bootstrap CI for ratio of means (matches adaptation_score)
    # NOTE: Bar shows mean(detailed)/mean(concise), so bootstrap the ratio of means
    error_bars = {}
    for model in models:
        model_dir = output_dir / model
        concise_csv = model_dir / "concise_responses.csv"
        detailed_csv = model_dir / "detailed_responses.csv"

        if concise_csv.exists() and detailed_csv.exists():
            concise_df = pd.read_csv(concise_csv)
            detailed_df = pd.read_csv(detailed_csv)

            if 'word_count' in concise_df.columns and 'word_count' in detailed_df.columns:
                concise_words = concise_df['word_count'].values
                detailed_words = detailed_df['word_count'].values
                n_samples = len(concise_words)

                if n_samples > 1:
                    # Bootstrap the ratio of means (not mean of ratios)
                    rng = np.random.RandomState(42)
                    bootstrap_ratios = []
                    for _ in range(1000):
                        idx = rng.choice(n_samples, size=n_samples, replace=True)
                        ratio = np.mean(detailed_words[idx]) / max(np.mean(concise_words[idx]), 1)
                        bootstrap_ratios.append(ratio)
                    ci_lower = np.percentile(bootstrap_ratios, 2.5)
                    ci_upper = np.percentile(bootstrap_ratios, 97.5)
                    error_bars[model] = (ci_lower, ci_upper)

    if error_bars:
        print(f"  [CI] Bootstrap CI computed for {len(error_bars)} models")

    # Use shared plotting function - Bar chart with error bars
    _generate_comparison_plots(
        models=models,
        overall_scores=adaptation_scores,
        valid_scores=valid_adaptation,
        valid_rates=valid_rates,
        output_dir=output_dir,
        plot_filename="length_control_comparison",
        ylabel="Length Adaptation Score",
        title="Length Control: Adaptation Score (Higher = Better)",
        perfect_score=None,  # No perfect line, just text annotation
        perfect_label="Target > 4.0",
        ylim_max=None,  # Auto-scale
        ylim_min=0,
        reference_line=1.0,  # "No Adaptation" reference line
        reference_label="No Adaptation",
        score_format=".2f",
        higher_is_better=True,
        error_bars=None  # CI disabled - overlaps with values, kept for combined plots
    )

    # Generate box/violin plots for per-sample ADAPTATION distribution
    # Adaptation = DETAILED_words / CONCISE_words (per prompt)
    adaptation_data = {}

    for model in models:
        model_dir = output_dir / model
        concise_csv = model_dir / "concise_responses.csv"
        detailed_csv = model_dir / "detailed_responses.csv"

        if concise_csv.exists() and detailed_csv.exists():
            concise_df = pd.read_csv(concise_csv)
            detailed_df = pd.read_csv(detailed_csv)

            if 'word_count' in concise_df.columns and 'word_count' in detailed_df.columns:
                # Per-sample adaptation ratio: DETAILED / CONCISE (higher = better)
                concise_words = concise_df['word_count'].values
                detailed_words = detailed_df['word_count'].values
                # Avoid division by zero
                per_sample_ratio = detailed_words / np.maximum(concise_words, 1)
                adaptation_data[model] = per_sample_ratio.tolist()

    # Generate box/violin for adaptation ratio distribution
    if len(adaptation_data) >= 2:
        _generate_boxviolin_chart(
            data_by_model=adaptation_data,
            output_dir=output_dir,
            plot_filename="length_control_adaptation_distribution",
            ylabel="Per-Sample Ratio (DETAILED / CONCISE)",
            title="Length Control: Adaptation Ratio Distribution",
            plot_type="both",
            higher_is_better=True,  # Higher ratio = better adaptation
            reference_lines=[(1.0, "No Adaptation", "gray"), (4.0, "Target: 4x", "green")]
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Length Control Evaluation")
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
        run_name="length_control_evaluation",
        project_root=project_root
    )

    try:
        # Cleanup handler
        checkpoint_dir = get_checkpoint_dir("length_control")
        results_dir = EVAL_OUTPUT_DIR

        checkpoints_exist = checkpoint_dir.exists() and any(
            f.name.endswith("_checkpoint.json") for f in checkpoint_dir.iterdir() if f.is_file()
        ) if checkpoint_dir.exists() else False
        results_exist = results_dir.exists() and any(results_dir.iterdir())

        cache_choice = None
        if checkpoints_exist or results_exist:
            cache_choice = show_cached_data_menu(
                checkpoint_dir=checkpoint_dir,
                results_dir=results_dir,
                eval_type="Length Control",
                checkpoint_suffix="_checkpoint.json",
                metrics_filename="metrics.json",
                plot_filename="length_control_comparison.png"
            )

        # Option 1: Regenerate plots only from cached metrics (skip model loading)
        if cache_choice == "1":
            print("\n" + "=" * 80)
            print("LOADING CACHED METRICS (Skipping inference)")
            print("=" * 80)

            all_results = {}
            all_stratified = {}

            for model_dir in EVAL_OUTPUT_DIR.iterdir():
                if model_dir.is_dir():
                    model_key = model_dir.name
                    metrics_path = model_dir / "metrics.json"

                    if metrics_path.exists():
                        with open(metrics_path, 'r') as f:
                            cached = json.load(f)

                        # Build result structure expected by generate_comparison_plots
                        all_results[model_key] = {
                            'metrics': {
                                'adaptation_score': cached.get('adaptation_score', 0),
                                'concise_avg_words': cached.get('concise_avg_words', 0),
                                'detailed_avg_words': cached.get('detailed_avg_words', 0)
                            }
                        }

                        all_stratified[model_key] = {
                            'concise_valid_rate': cached.get('concise_valid_rate', 1.0),
                            'detailed_valid_rate': cached.get('detailed_valid_rate', 1.0),
                            'valid_concise_avg_words': cached.get('valid_concise_avg_words'),
                            'valid_detailed_avg_words': cached.get('valid_detailed_avg_words'),
                            'valid_adaptation_score': cached.get('valid_adaptation_score')
                        }
                        print(f"  Loaded: {model_key} (Adaptation: {cached.get('adaptation_score', 0):.3f})")

            if len(all_results) < 2:
                print("\nError: Need at least 2 models with cached metrics. Run full evaluation first.")
                sys.exit(1)

            # Generate plots from cached data
            print("\n" + "=" * 80)
            print("Generating Comparison Plots")
            print("=" * 80)
            generate_comparison_plots(all_results, EVAL_OUTPUT_DIR, all_stratified)

            print("\n" + "=" * 80)
            print("PLOTS REGENERATED FROM CACHED METRICS")
            print("=" * 80)
            print(f"Results saved to: {EVAL_OUTPUT_DIR}")
            return

        # Interactive mode selection (defer HF fetch to avoid M1 mutex lock)
        mode, _ = show_mode_selection_menu(
            eval_name="LENGTH CONTROL",
            sanity_desc="100 prompts x 2 variants = 200 test cases (~15 min)",
            full_desc="500 prompts x 2 variants = 1,000 test cases (~60 min)",
            max_desc="100% of dataset (fetches from HF)"
        )
        args.mode = mode
        if mode == "sanity":
            args.samples = 100
        elif mode == "full":
            args.samples = 500
        else:  # max - fetch dynamically
            max_prompts, max_cases, source = get_length_control_max_samples()
            print(f"   Max Available: {max_prompts} prompts x 2 = {max_cases:,} [{source}]")
            args.samples = max_prompts

        # Determine sample count
        if args.samples:
            max_samples = args.samples
        elif args.mode == "sanity":
            max_samples = 100
        else:
            max_samples = 500

        print(f"\n{'=' * 80}")
        print(f"Length Control Evaluation")
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
