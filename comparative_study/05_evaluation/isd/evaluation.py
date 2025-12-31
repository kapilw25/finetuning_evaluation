"""
ISD (Instruction Switch Dataset) Evaluation

Uses sentence transformer embeddings for fidelity scoring.

Complete evaluation pipeline: Inference → Metrics → Comparison Plots
Tests the core claim: "Instruction-Aware: DPO=No, CITA=Yes"

Metrics:
    1. Fidelity Score: Embedding similarity between response and instruction prototype
    2. Semantic Shift: How much does response change across different instructions?
    3. Instruction Awareness Score: Fidelity × Semantic Shift (combined metric)

Dataset: https://huggingface.co/datasets/kapilw25/ISD-Instruction-Switch-Dataset
    - 500 prompts × 10 instruction types = 5,000 test cases
    - Instruction types: Neutral, Conservative, Liberal, Regulatory, Empathetic,
                         Safety, Educational, Concise, Professional, Creative

Usage:
    # Sanity check
    python comparative_study/05_evaluation/isd/evaluation.py \
    --models SFT_Instruct SFT_NoInstruct DPO_Instruct DPO_NoInstruct \
             PPO_Instruct PPO_NoInstruct GRPO_Instruct GRPO_NoInstruct \
             CITA_Instruct CITA_NoInstruct \
    --mode sanity

    # Full evaluation
    python comparative_study/05_evaluation/isd/evaluation.py \
    --models SFT_Instruct SFT_NoInstruct DPO_Instruct DPO_NoInstruct \
             PPO_Instruct PPO_NoInstruct GRPO_Instruct GRPO_NoInstruct \
             CITA_Instruct CITA_NoInstruct \
    --mode full

Available models: SFT_NoInstruct, SFT_Instruct, DPO_NoInstruct, DPO_Instruct,
                  PPO_NoInstruct, PPO_Instruct, GRPO_NoInstruct, GRPO_Instruct,
                  CITA_NoInstruct, CITA_Instruct

Output:
    - outputs/evaluation/ISD/{model}/ - responses CSV, metrics JSON
    - outputs/evaluation/ISD/isd_comparison.png - comparison plots
    - logs/ISD_evaluation_*.log - full execution log
"""

import sys
import json
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent  # finetuning_evaluation
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))
sys.path.insert(0, str(project_root / "comparative_study" / "05_evaluation"))
sys.path.insert(0, str(Path(__file__).parent))  # For dataset submodule

from dataset import (
    InstructionSwitchDataset,
    ISDTestCase
)

from eval_utils import (
    MODELS, load_model_for_eval, unload_model,
    setup_training_logger, restore_logging,
    save_checkpoint, load_checkpoint, delete_checkpoint, get_checkpoint_path,
    batch_generate, cleanup_gpu, format_chat_messages, verify_hf_repos,
    add_validation_columns, get_validation_summary,
    show_cached_data_menu, show_mode_selection_menu, show_checkpoint_resume_menu,
    filter_model_keys,
    get_isd_max_samples,
    generate_comparison_plots,
    generate_lollipop_chart
)
from eval_utils.checkpoint import get_checkpoint_dir
from eval_utils.bootstrap import compute_bootstrap_ci

# Add isd utils path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from metrics import ISDMetricsCalculator, ModelMetrics


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ISDResponse:
    """Response data for a single ISD test case"""
    prompt_id: int
    instruction_type: str
    instruction: str
    prompt: str
    response: str
    response_length: int
    generation_time: float
    expected_characteristics: List[str]


@dataclass
class ISDModelResult:
    """Complete evaluation result for a model"""
    model_name: str
    responses: List[ISDResponse]
    total_prompts: int
    total_test_cases: int
    evaluation_time: float
    timestamp: str


# =============================================================================
# CONFIGURATION
# =============================================================================

EVAL_OUTPUT_DIR = project_root / "outputs" / "evaluation" / "ISD_Evaluation_Embedding"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEYS = list(MODELS.keys())


# =============================================================================
# RESPONSE GENERATION (Function-based pattern)
# =============================================================================

def generate_responses(
    model,
    tokenizer,
    test_cases: List[ISDTestCase],
    model_key: str,
    use_instruction: bool,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    checkpoint_interval: int = 100
) -> List[ISDResponse]:
    """
    Generate responses for all test cases with batch processing and checkpointing

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        test_cases: List of ISDTestCase objects
        model_key: Model identifier for checkpointing
        use_instruction: Whether to include system instruction
        max_new_tokens: Max tokens to generate
        batch_size: Batch size for generation
        checkpoint_interval: Save checkpoint every N responses

    Returns:
        List of ISDResponse objects
    """
    # Check for existing checkpoint
    checkpoint = load_checkpoint(model_key, eval_type="isd")

    if checkpoint and checkpoint['completed']:
        choice = show_checkpoint_resume_menu(
            model_key=model_key,
            n_responses=checkpoint['n_completed'],
            eval_type="ISD"
        )
        if choice == "1":
            return [ISDResponse(**r) for r in checkpoint['responses']]
        else:  # choice == "2"
            delete_checkpoint(model_key, eval_type="isd")
            checkpoint = None

    # Resume from checkpoint if exists
    if checkpoint and not checkpoint['completed']:
        responses = [ISDResponse(**r) for r in checkpoint['responses']]
        start_idx = len(responses)
        print(f"Resuming from test case {start_idx + 1}/{len(test_cases)}")
    else:
        responses = []
        start_idx = 0

    # Format remaining prompts for batch processing
    remaining_cases = test_cases[start_idx:]

    messages_list = []
    for tc in remaining_cases:
        if use_instruction:
            messages = [
                {"role": "system", "content": tc.instruction},
                {"role": "user", "content": tc.prompt}
            ]
        else:
            messages = [{"role": "user", "content": tc.prompt}]
        messages_list.append(messages)

    # Format all messages at once
    prompts = format_chat_messages(tokenizer, messages_list)

    # Create checkpoint callback for intermediate saves
    def checkpoint_cb(batch_responses_so_far):
        # Convert to ISDResponse objects for checkpoint
        temp_responses = responses.copy()
        for i, response_text in enumerate(batch_responses_so_far):
            tc = remaining_cases[i]
            temp_responses.append(ISDResponse(
                prompt_id=tc.prompt_id,
                instruction_type=tc.instruction_type,
                instruction=tc.instruction,
                prompt=tc.prompt,
                response=response_text,
                response_length=len(response_text),
                generation_time=0.0,
                expected_characteristics=tc.expected_characteristics
            ))
        save_checkpoint(
            model_key,
            [asdict(r) for r in temp_responses],
            len(test_cases),
            eval_type="isd",
            completed=False
        )

    # Batch generate
    batch_responses = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        show_progress=True,
        desc=f"Evaluating {MODELS[model_key]['display_name']}",
        checkpoint_callback=checkpoint_cb,
        checkpoint_interval=checkpoint_interval
    )

    # Convert all responses to ISDResponse objects
    for i, response_text in enumerate(batch_responses):
        tc = remaining_cases[i]
        resp = ISDResponse(
            prompt_id=tc.prompt_id,
            instruction_type=tc.instruction_type,
            instruction=tc.instruction,
            prompt=tc.prompt,
            response=response_text,
            response_length=len(response_text),
            generation_time=0.0,
            expected_characteristics=tc.expected_characteristics
        )
        responses.append(resp)

    # Save final checkpoint
    save_checkpoint(
        model_key,
        [asdict(r) for r in responses],
        len(test_cases),
        eval_type="isd",
        completed=True
    )

    return responses


def save_results(
    result: ISDModelResult,
    output_dir: Path
) -> Path:
    """Save evaluation results to JSON and CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    json_path = output_dir / f"{result.model_name}_isd_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "metadata": {
                "model_name": result.model_name,
                "total_prompts": result.total_prompts,
                "total_test_cases": result.total_test_cases,
                "evaluation_time": result.evaluation_time,
                "timestamp": result.timestamp
            },
            "responses": [asdict(r) for r in result.responses]
        }, f, indent=2)

    # Save responses as CSV for easier analysis
    csv_path = output_dir / f"{result.model_name}_isd_responses.csv"
    df = pd.DataFrame([asdict(r) for r in result.responses])
    df.to_csv(csv_path, index=False)

    print(f"Results saved to {output_dir}")
    return output_dir


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_isd_evaluation(
    model_keys: List[str],
    num_prompts: int = 300,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    batch_size: int = 8
) -> Dict[str, ISDModelResult]:
    """
    Run ISD evaluation on multiple models

    Args:
        model_keys: List of model keys from MODELS dict
        num_prompts: Number of prompts (x10 instructions = test cases)
        output_dir: Where to save results
        seed: Random seed for dataset generation
        batch_size: Batch size for inference

    Returns:
        Dict mapping model_key -> ISDModelResult
    """
    # Generate dataset
    isd = InstructionSwitchDataset(seed=seed)
    test_cases = isd.generate_dataset(num_prompts)

    print("=" * 80)
    print("ISD Evaluation")
    print("=" * 80)
    print(f"Prompts: {num_prompts}")
    print(f"Instructions: {len(isd.get_instruction_types())}")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Models: {len(model_keys)}")
    print("=" * 80)

    if output_dir is None:
        output_dir = EVAL_OUTPUT_DIR

    results = {}

    for model_key in model_keys:
        model_info = MODELS[model_key]

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_info['display_name']}")
        print(f"Instruction-Aware: {model_info['use_instruction']}")
        print(f"{'=' * 80}")

        try:
            # Load model
            model, tokenizer = load_model_for_eval(model_key)

            # Generate responses
            start_time = time.time()
            responses = generate_responses(
                model, tokenizer, test_cases,
                model_key=model_key,
                use_instruction=model_info['use_instruction'],
                batch_size=batch_size
            )
            evaluation_time = time.time() - start_time

            # Create result object
            result = ISDModelResult(
                model_name=model_key,
                responses=responses,
                total_prompts=len(set(tc.prompt_id for tc in test_cases)),
                total_test_cases=len(test_cases),
                evaluation_time=evaluation_time,
                timestamp=datetime.now().isoformat()
            )
            results[model_key] = result

            # Save results
            model_output_dir = output_dir / model_key
            save_results(result, model_output_dir)

            # Unload model
            unload_model(model)
            del model  # Release caller's reference
            del tokenizer
            cleanup_gpu()

            # Print summary
            avg_length = sum(r.response_length for r in result.responses) / len(result.responses)

            if result.evaluation_time > 0:
                print(f"\nSummary for {model_key}:")
                print(f"  Test cases: {result.total_test_cases}")
                print(f"  Avg response length: {avg_length:.0f} chars")
                print(f"  Total time: {result.evaluation_time:.1f}s")
            else:
                print(f"\nSummary for {model_key} (from checkpoint):")
                print(f"  Test cases: {result.total_test_cases}")
                print(f"  Avg response length: {avg_length:.0f} chars")

        except RuntimeError as e:
            print(f"\nFailed to evaluate {model_key}: {e}")
            print(f"   Skipping this model...")
            continue

    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def generate_isd_comparison_plots(all_metrics: Dict[str, ModelMetrics], output_dir: Path, stratified_metrics: Dict[str, Dict] = None):
    """Generate ISD comparison plot using shared plotting function with Bootstrap CI"""
    if len(all_metrics) < 2:
        print("Need at least 2 models for comparison plots")
        return

    models = list(all_metrics.keys())
    overall_scores = [all_metrics[m].instruction_awareness_score for m in models]

    # Get valid-only scores from stratified metrics
    valid_scores = []
    valid_rates = []
    for m in models:
        if stratified_metrics and m in stratified_metrics:
            va = stratified_metrics[m].get('valid_awareness')
            vr = stratified_metrics[m].get('valid_rate', 1.0)
            valid_scores.append(va if va is not None else all_metrics[m].instruction_awareness_score)
            valid_rates.append(vr)
        else:
            valid_scores.append(all_metrics[m].instruction_awareness_score)
            valid_rates.append(1.0)

    # NOTE: Skipping CI for ISD
    # Bar shows instruction_awareness_score = mean_fidelity × mean_semantic_shift
    # per_sample_fidelity alone doesn't capture this composite metric
    # CI would be misleading since fidelity CI ≠ awareness CI
    error_bars = None
    print("  [INFO] CI skipped for ISD (composite metric: fidelity × semantic_shift)")

    # Use shared plotting function - Bar chart with error bars
    generate_comparison_plots(
        models=models,
        overall_scores=overall_scores,
        valid_scores=valid_scores,
        valid_rates=valid_rates,
        output_dir=output_dir,
        plot_filename="isd_comparison",
        ylabel="Instruction Awareness Score",
        title="ISD: Instruction Awareness (Higher = Better)",
        perfect_score=1.0,
        perfect_label="Perfect = 1.0",
        score_format=".3f",
        higher_is_better=True,
        error_bars=error_bars if error_bars else None
    )

    # Also generate lollipop chart as alternative
    generate_lollipop_chart(
        models=models,
        overall_scores=overall_scores,
        output_dir=output_dir,
        plot_filename="isd_comparison",
        xlabel="Instruction Awareness Score",
        title="ISD: Instruction Awareness (Higher = Better)",
        perfect_score=1.0,
        perfect_label="Perfect = 1.0",
        score_format=".3f",
        higher_is_better=True
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point with logging"""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="ISD Evaluation")
    parser.add_argument("--num_prompts", type=int, default=50, help="Number of prompts (x10 = test cases)")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to evaluate (e.g., CITA_Instruct DPO_Instruct)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    # Setup logging
    output_dir = EVAL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
        run_name="ISD_evaluation",
        project_root=project_root
    )

    try:
        # Cleanup handler
        checkpoint_dir = get_checkpoint_dir("isd")
        results_dir = output_dir

        checkpoints_exist = checkpoint_dir.exists() and any(
            f.name.endswith("_isd_checkpoint.json") for f in checkpoint_dir.iterdir() if f.is_file()
        ) if checkpoint_dir.exists() else False
        results_exist = results_dir.exists() and any(results_dir.iterdir())

        if checkpoints_exist or results_exist:
            show_cached_data_menu(
                checkpoint_dir=checkpoint_dir,
                results_dir=results_dir,
                eval_type="ISD",
                checkpoint_suffix="_isd_checkpoint.json",
                metrics_filename="{model}_isd_metrics.json",
                plot_filename="isd_comparison.png"
            )

        # Interactive mode selection (max_desc shown without HF fetch to avoid M1 mutex lock)
        mode, _ = show_mode_selection_menu(
            eval_name="ISD",
            sanity_desc="50 prompts x 10 instructions = 500 test cases (~15 min)",
            full_desc="300 prompts x 10 instructions = 3,000 test cases (~60 min)",
            max_desc="100% of dataset (fetches from HF)"
        )
        if mode == "sanity":
            args.num_prompts = 50
        elif mode == "full":
            args.num_prompts = 300
        else:  # max - fetch dynamically
            max_prompts, max_cases, source = get_isd_max_samples()
            print(f"   Max Available: {max_prompts} prompts x 10 = {max_cases:,} [{source}]")
            args.num_prompts = max_prompts

        # Filter models if specified
        model_keys = filter_model_keys(args.models, MODELS, MODEL_KEYS)

        # Pre-flight verification of HuggingFace repos
        model_keys = verify_hf_repos(model_keys, interactive=True)
        if not model_keys:
            print("No valid models to evaluate. Exiting.")
            sys.exit(1)

        # Run inference
        results = run_isd_evaluation(
            model_keys=model_keys,
            num_prompts=args.num_prompts,
            seed=args.seed,
            output_dir=output_dir,
            batch_size=args.batch_size
        )

        # Calculate metrics for each model
        print("\n" + "=" * 80)
        print("Calculating ISD Metrics")
        print("=" * 80)

        calculator = ISDMetricsCalculator(use_llm_judge=False)
        all_metrics = {}
        all_stratified = {}  # Collect stratified metrics for plotting

        for model_key, result in results.items():
            metrics_path = output_dir / model_key / f"{model_key}_isd_metrics.json"

            # Check for cached metrics
            if metrics_path.exists():
                print(f"\nLoading cached metrics for {model_key}")
                with open(metrics_path, 'r') as f:
                    cached = json.load(f)
                metrics = ModelMetrics(
                    model_name=cached['model_name'],
                    mean_fidelity=cached['mean_fidelity'],
                    fidelity_by_instruction=cached['fidelity_by_instruction'],
                    mean_semantic_shift=cached['mean_semantic_shift'],
                    instruction_awareness_score=cached['instruction_awareness_score'],
                    n_evaluated=cached['n_evaluated']
                )
                all_metrics[model_key] = metrics
                # Load stratified metrics from cache (including valid-only)
                all_stratified[model_key] = {
                    'valid_rate': cached.get('valid_response_rate', 1.0),
                    'gibberish_rate': cached.get('gibberish_rate', 0.0),
                    'repetitive_rate': cached.get('repetitive_rate', 0.0),
                    'valid_fidelity': cached.get('valid_fidelity'),
                    'valid_awareness': cached.get('valid_awareness'),
                    'valid_shift': cached.get('valid_shift')
                }
                print(f"  Loaded from cache")
            else:
                print(f"\nCalculating metrics for {model_key}...")

                # Convert responses to DataFrame
                responses_df = pd.DataFrame([asdict(r) for r in result.responses])

                # Add validation columns for gibberish/repetition detection
                responses_df = add_validation_columns(responses_df, response_column='response')

                # Get validation summary
                validation = get_validation_summary(responses_df)

                # Calculate OVERALL metrics (all responses) using embedding
                metrics = calculator.calculate_metrics(
                    responses_df=responses_df,
                    model_name=model_key,
                    use_embedding_for_fidelity=True
                )
                all_metrics[model_key] = metrics

                # Calculate VALID-ONLY metrics (filtered responses)
                valid_df = responses_df[responses_df['is_valid']].copy()
                if len(valid_df) > 0:
                    valid_metrics = calculator.calculate_metrics(
                        responses_df=valid_df,
                        model_name=f"{model_key}_valid",
                        use_embedding_for_fidelity=True
                    )
                    valid_fidelity = valid_metrics.mean_fidelity
                    valid_awareness = valid_metrics.instruction_awareness_score
                    valid_shift = valid_metrics.mean_semantic_shift
                else:
                    valid_fidelity = None
                    valid_awareness = None
                    valid_shift = None

                # Print comparison
                print(f"\n{model_key} Results:")
                print(f"  Valid response rate: {validation['valid_rate']:.1%}")
                print(f"  Gibberish rate: {validation['gibberish_rate']:.1%}")
                print(f"  Repetitive rate: {validation['repetitive_rate']:.1%}")
                print(f"  Fidelity - Overall: {metrics.mean_fidelity:.3f}, Valid-only: {f'{valid_fidelity:.3f}' if valid_fidelity else 'N/A'}")
                print(f"  Awareness - Overall: {metrics.instruction_awareness_score:.3f}, Valid-only: {f'{valid_awareness:.3f}' if valid_awareness else 'N/A'}")

                # Store for plotting (both overall and valid-only)
                all_stratified[model_key] = {
                    'valid_rate': validation['valid_rate'],
                    'gibberish_rate': validation['gibberish_rate'],
                    'repetitive_rate': validation['repetitive_rate'],
                    'valid_fidelity': valid_fidelity,
                    'valid_awareness': valid_awareness,
                    'valid_shift': valid_shift
                }

                # Save metrics with both overall and valid-only
                with open(metrics_path, 'w') as f:
                    json.dump({
                        "model_name": metrics.model_name,
                        # Overall metrics
                        "mean_fidelity": metrics.mean_fidelity,
                        "fidelity_by_instruction": metrics.fidelity_by_instruction,
                        "mean_semantic_shift": metrics.mean_semantic_shift,
                        "instruction_awareness_score": metrics.instruction_awareness_score,
                        "n_evaluated": metrics.n_evaluated,
                        # Per-sample fidelity for Bootstrap CI
                        "per_sample_fidelity": metrics.per_sample_fidelity,
                        # Valid-only metrics
                        "valid_fidelity": valid_fidelity,
                        "valid_awareness": valid_awareness,
                        "valid_shift": valid_shift,
                        # Validation rates
                        "valid_response_rate": validation['valid_rate'],
                        "gibberish_rate": validation['gibberish_rate'],
                        "repetitive_rate": validation['repetitive_rate'],
                        "valid_responses": validation['n_valid'],
                        "total_responses": validation['n_total']
                    }, f, indent=2)

                # Save updated responses CSV with validation columns
                csv_path = output_dir / model_key / f"{model_key}_isd_responses.csv"
                responses_df.to_csv(csv_path, index=False)

            print(f"  Fidelity: {metrics.mean_fidelity:.3f}")
            print(f"  Semantic Shift: {metrics.mean_semantic_shift:.3f}")
            print(f"  Instruction Awareness: {metrics.instruction_awareness_score:.3f}")

        # Generate comparison plots
        if len(all_metrics) >= 2:
            print("\n" + "=" * 80)
            print("Generating Comparison Plots")
            print("=" * 80)
            generate_isd_comparison_plots(all_metrics, output_dir, all_stratified)

        print("\n" + "=" * 80)
        print("ISD Evaluation Complete")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")

    finally:
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"\nLog saved to: {log_filename}")


if __name__ == "__main__":
    main()
