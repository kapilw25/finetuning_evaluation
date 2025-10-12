"""
Generic PBT (Population-Based Training) Utilities
Can be used with any finetuning method: CITA, DPO, LoRA, etc.

Key Features:
- Configurable PBT scheduler
- Automatic exploit/explore
- Best hyperparameter tracking
- Compatible with HuggingFace Transformers
"""

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining


def create_pbt_scheduler(
    hyperparam_mutations,
    mutation_interval=100,
    metric="eval_loss",
    mode="min"
):
    """
    Create a generic PBT scheduler for any hyperparameters

    Args:
        hyperparam_mutations: Dict of hyperparameters to tune
            Example: {
                "learning_rate": tune.uniform(1e-5, 5e-5),
                "weight_decay": tune.uniform(0.001, 0.05),
            }
        mutation_interval: Steps between exploit/explore (default: 100)
        metric: Metric to optimize (default: "eval_loss")
        mode: "min" or "max" (default: "min")

    Returns:
        PopulationBasedTraining scheduler

    How it works:
    - Every mutation_interval steps: compare all workers
    - Bottom 50% copy weights from top 50%
    - Then mutate hyperparameters ±20%
    """
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",  # HuggingFace reports training_iteration, not step
        metric=metric,
        mode=mode,
        perturbation_interval=1,  # 1 iteration = save_steps (e.g., 100 steps)
        hyperparam_mutations=hyperparam_mutations,
        quantile_fraction=0.5,  # Top 50% vs bottom 50%
    )
    return scheduler


def run_pbt_training(
    trainable,
    hp_space,
    scheduler,
    num_workers=3,
    max_iterations=10,
    output_dir="./outputs/ray_results",
    name="pbt_training"
):
    """
    Run PBT training with specified configuration

    Args:
        trainable: Training function (receives config dict)
        hp_space: Hyperparameter search space (dict or function)
        scheduler: PBT scheduler (from create_pbt_scheduler)
        num_workers: Number of parallel workers (default: 3)
        max_iterations: Number of checkpoints (default: 10)
        output_dir: Directory for Ray Tune results
        name: Experiment name

    Returns:
        analysis: Ray Tune analysis object with best hyperparameters
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Run hyperparameter search with PBT
    # IMPORTANT: Must explicitly set stop condition to prevent training beyond max_iterations
    # Without stop condition, Trainer.train() continues past max_steps causing waste
    analysis = tune.run(
        trainable,
        name=name,
        config=hp_space,
        scheduler=scheduler,
        num_samples=num_workers,
        resources_per_trial={"gpu": 1},  # Each worker uses 1 GPU
        stop={"training_iteration": max_iterations},  # ✅ FIX: Stop after max_iterations
        keep_checkpoints_num=2,
        storage_path=output_dir,  # Updated from local_dir (Ray 2.x API)
        verbose=1,
    )

    return analysis


def print_best_hyperparameters(analysis, metric="eval_loss", mode="min"):
    """
    Print best hyperparameters found by PBT

    Args:
        analysis: Ray Tune analysis object
        metric: Metric used for optimization (default: "eval_loss")
        mode: "min" or "max" (default: "min")

    Returns:
        best_trial: Best trial object
    """
    best_trial = analysis.get_best_trial(metric, mode, "last")

    print(f"\n{'='*80}")
    print(f"🏆 Best Hyperparameters Found by PBT:")
    print(f"{'='*80}")

    # Print all config parameters
    for key, value in sorted(best_trial.config.items()):
        # Format floats nicely
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n  Final {metric}: {best_trial.last_result.get(metric, 'N/A')}")
    print(f"  Best checkpoint: {best_trial.checkpoint.dir_or_data}")
    print(f"{'='*80}\n")

    return best_trial


def save_best_config(best_trial, output_path="./outputs/best_pbt_config.json"):
    """
    Save best hyperparameters to JSON file

    Args:
        best_trial: Best trial from print_best_hyperparameters()
        output_path: Path to save JSON file

    Returns:
        output_path: Path where config was saved
    """
    import json
    from pathlib import Path

    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path, "w") as f:
        json.dump(best_trial.config, f, indent=2)

    print(f"✅ Best hyperparameters saved to: {output_path}")
    return output_path
