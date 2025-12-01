"""
CLI Menu Utilities for Evaluation Scripts

Shared interactive menus for cached data options and mode selection.
"""

import shutil
from pathlib import Path
from typing import List, Tuple, Optional


def show_cached_data_menu(
    checkpoint_dir: Path,
    results_dir: Path,
    eval_type: str,
    checkpoint_suffix: str,
    metrics_filename: str = "metrics.json",
    plot_filename: str = "comparison.png"
) -> str:
    """
    Show cached data options menu

    Args:
        checkpoint_dir: Directory containing checkpoints
        results_dir: Directory containing results
        eval_type: Type of evaluation (for display, e.g., "ISD", "TruthfulQA")
        checkpoint_suffix: Suffix for checkpoint files (e.g., "_isd_checkpoint.json")
        metrics_filename: Name of metrics file to check
        plot_filename: Name of plot file to check

    Returns:
        Choice: "1" (use cached), "2" (recalculate metrics), "3" (delete all)
    """
    # Detect what exists
    n_checkpoints = 0
    if checkpoint_dir.exists():
        n_checkpoints = len([f for f in checkpoint_dir.iterdir()
                            if f.name.endswith(checkpoint_suffix)])

    n_metrics = 0
    if results_dir.exists():
        for d in results_dir.iterdir():
            if d.is_dir():
                # Handle different metrics filename patterns
                if metrics_filename.startswith("{model}"):
                    metrics_file = d / metrics_filename.replace("{model}", d.name)
                else:
                    metrics_file = d / metrics_filename
                if metrics_file.exists():
                    n_metrics += 1

    plot_exists = (results_dir / plot_filename).exists() if results_dir.exists() else False

    print("\n" + "=" * 80)
    print("🔧 CACHED DATA OPTIONS")
    print("=" * 80)

    print(f"\nDetected:")
    print(f"  {'✅' if n_checkpoints > 0 else '❌'} Inference checkpoints ({n_checkpoints} files)")
    print(f"  {'✅' if n_metrics > 0 else '❌'} Metrics JSONs ({n_metrics} models)")
    print(f"  {'✅' if plot_exists else '❌'} Comparison plot")

    print("\nWhat do you want to re-run?")
    print("\n  1) Regenerate plots only (keep inference + metrics)")
    print("  2) Re-calculate metrics + plots (keep inference, re-run LLM-as-judge)")
    print("  3) Re-run everything (delete all, start fresh)")
    print("=" * 80)

    while True:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()

        if choice == "1":
            print("\n✅ Will regenerate plots using cached metrics")
            return choice
        elif choice == "2":
            print("\n🗑️  Deleting metrics (keeping inference checkpoints)")
            # Delete only metrics files
            if results_dir.exists():
                for model_dir in results_dir.iterdir():
                    if model_dir.is_dir():
                        if metrics_filename.startswith("{model}"):
                            metrics_file = model_dir / metrics_filename.replace("{model}", model_dir.name)
                        else:
                            metrics_file = model_dir / metrics_filename
                        if metrics_file.exists():
                            metrics_file.unlink()
                            print(f"   ✅ Deleted: {metrics_file.name}")
            # Delete plot
            plot_file = results_dir / plot_filename
            if plot_file.exists():
                plot_file.unlink()
                print(f"   ✅ Deleted: {plot_filename}")
            return choice
        elif choice == "3":
            print("\n🗑️  Deleting all cached data")
            # Delete checkpoints
            if checkpoint_dir.exists():
                for f in checkpoint_dir.iterdir():
                    if f.name.endswith(checkpoint_suffix):
                        f.unlink()
                        print(f"   ✅ Deleted checkpoint: {f.name}")
            # Delete results
            if results_dir.exists() and any(results_dir.iterdir()):
                shutil.rmtree(results_dir)
                print(f"   ✅ Deleted: {results_dir.name}/")
            results_dir.mkdir(exist_ok=True)
            return choice
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def show_mode_selection_menu(
    eval_name: str,
    sanity_desc: str,
    full_desc: str,
    max_desc: Optional[str] = None
) -> Tuple[str, int]:
    """
    Show evaluation mode selection menu

    Args:
        eval_name: Name of evaluation (e.g., "ISD", "TRUTHFULQA")
        sanity_desc: Description for sanity check option
        full_desc: Description for full evaluation option
        max_desc: Description for max available option (optional)

    Returns:
        Tuple of (mode, num_samples) - mode is "sanity", "full", or "max"
    """
    print("\n" + "=" * 80)
    print(f"🎯 {eval_name} EVALUATION: Mode Selection")
    print("=" * 80)
    print("\nChoose evaluation mode:")
    print(f"  1) Sanity Check    - {sanity_desc}")
    print(f"  2) Full Evaluation - {full_desc}")
    if max_desc:
        print(f"  3) Max Available   - {max_desc}")
    print("=" * 80)

    valid_choices = ["1", "2", "3"] if max_desc else ["1", "2"]
    prompt = "\nEnter choice (1, 2, or 3): " if max_desc else "\nEnter choice (1 or 2): "

    while True:
        choice = input(prompt).strip()
        if choice == "1":
            print("\n✅ Running Sanity Check")
            return "sanity", None  # Caller sets actual number
        elif choice == "2":
            print("\n✅ Running Full Evaluation")
            return "full", None
        elif choice == "3" and max_desc:
            print("\n✅ Running Max Available")
            return "max", None
        else:
            print(f"❌ Invalid choice. Please enter {', '.join(valid_choices)}.")


def show_checkpoint_resume_menu(
    model_key: str,
    n_responses: int,
    eval_type: str
) -> str:
    """
    Show checkpoint resume options

    Args:
        model_key: Model identifier
        n_responses: Number of cached responses
        eval_type: Type of evaluation

    Returns:
        Choice: "1" (use cached) or "2" (re-run)
    """
    print(f"\n{'=' * 80}")
    print(f"✅ INFERENCE ALREADY COMPLETED for {model_key}")
    print(f"{'=' * 80}")
    print(f"  Responses: {n_responses}")
    print(f"{'=' * 80}")
    print("\nChoose action:")
    print("  1) Use cached responses (skip inference)")
    print("  2) Re-run inference (overwrite checkpoint)")
    print("=" * 80)

    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            if choice == "2":
                print("\n🔄 Re-running inference from scratch...")
            return choice
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def filter_model_keys(
    requested_models: Optional[List[str]],
    available_models: dict,
    default_keys: List[str]
) -> List[str]:
    """
    Filter and validate model keys

    Args:
        requested_models: List of requested model keys (or None for all)
        available_models: Dict of available models (MODELS dict)
        default_keys: Default list of model keys

    Returns:
        List of valid model keys
    """
    import sys

    if requested_models:
        invalid = [m for m in requested_models if m not in available_models]
        if invalid:
            print(f"Invalid model keys: {invalid}")
            print(f"Available models: {list(available_models.keys())}")
            sys.exit(1)
        return requested_models
    else:
        return default_keys
