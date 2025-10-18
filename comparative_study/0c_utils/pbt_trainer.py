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
from ray.tune import Stopper


class AllWorkersSafetyStopper(Stopper):
    """
    Custom Ray Tune Stopper that aborts experiment if ALL workers fail safety checks

    Features:
    - Checks every checkpoint (every 50 steps) if ALL workers failed
    - Failure = (negative margin) OR (gibberish detected) - OR logic
    - Aborts experiment immediately when all workers fail (prevents GPU waste)
    - Does NOT stop individual workers (PBT rescues failed workers)

    How it works:
    1. __call__(trial_id, result): Stores latest result for each worker
    2. stop_all(): Checks if ALL workers failed → Abort if True

    GPU Efficiency:
    - OLD: If all workers fail at step 250, train until step 1000 (waste: 2,250 GPU-steps)
    - NEW: If all workers fail at step 250, abort at step 250 (waste: 0 GPU-steps)
    """

    def __init__(self, margin_tolerance=0.0):
        """
        Args:
            margin_tolerance: Minimum acceptable margin (default: 0.0 = must be positive)
        """
        self.trial_results = {}  # {trial_id: latest_result}
        self.margin_tolerance = margin_tolerance
        self.all_workers_failed = False  # Flag for logging (print abort message once)

    def __call__(self, trial_id: str, result: dict) -> bool:
        """
        Called after each trial reports results (every 50 steps)

        Args:
            trial_id: Unique trial identifier
            result: Training metrics from this trial

        Returns:
            False: Never stop individual trials (PBT rescues them)
        """
        # Store latest result for this trial
        self.trial_results[trial_id] = result

        # Never stop individual trials - let PBT handle recovery
        return False

    def stop_all(self) -> bool:
        """
        Called periodically to check if entire experiment should be aborted

        Checks if ALL workers failed using OR logic:
        - Worker failed if: (margin <= 0) OR (gibberish detected) OR (unsafe behavior)

        Returns:
            True if ALL workers failed (experiment aborted)
            False otherwise (experiment continues)
        """
        # Need at least 1 trial to check
        if not self.trial_results:
            return False

        # Check if ALL workers failed (OR logic for failure conditions)
        all_failed = all(
            self._is_worker_failed(result)
            for result in self.trial_results.values()
        )

        if all_failed and not self.all_workers_failed:
            # First time detecting all-worker failure - log it once
            self.all_workers_failed = True
            self._print_abort_message()

        return all_failed

    def _is_worker_failed(self, result: dict) -> bool:
        """
        Check if a single worker failed (OR logic)

        A worker is considered failed if ANY of these conditions are true:
        1. Negative margin (margin <= tolerance)
        2. Gibberish detected (gibberish_detected == True)
        3. Unsafe behavior detected (unsafe_behavior_detected == True)

        Args:
            result: Trial result dict with metrics

        Returns:
            True if worker failed (any condition met)
            False if worker is healthy (all conditions OK)
        """
        # Condition 1: Negative margin (model prefers unsafe responses)
        margin = result.get('cita/margin', 1)  # Default 1 = assume healthy if not reported
        has_negative_margin = margin <= self.margin_tolerance

        # Condition 2: Gibberish detected (mode collapse)
        gibberish_detected = result.get('gibberish_detected', False)

        # Condition 3: Unsafe behavior detected (alternative to margin check)
        unsafe_behavior = result.get('unsafe_behavior_detected', False)

        # OR logic: Failed if ANY condition is true
        return has_negative_margin or gibberish_detected or unsafe_behavior

    def _print_abort_message(self):
        """Print detailed abort message with failure reasons for each worker"""
        print(f"\n{'!'*80}")
        print(f"🛑 GLOBAL ABORT: ALL {len(self.trial_results)} WORKERS FAILED SAFETY CHECKS")
        print(f"{'!'*80}")
        print(f"Aborting experiment immediately to prevent GPU waste")
        print(f"")
        print(f"Worker failure status:")

        for trial_id, result in self.trial_results.items():
            margin = result.get('cita/margin', 'N/A')
            gibberish = result.get('gibberish_detected', False)
            unsafe = result.get('unsafe_behavior_detected', False)
            iteration = result.get('training_iteration', 0)

            # Determine failure reasons
            reasons = []
            if isinstance(margin, (int, float)) and margin <= self.margin_tolerance:
                reasons.append(f"margin={margin:.2f}")
            if gibberish:
                reasons.append("gibberish")
            if unsafe:
                reasons.append("unsafe")

            status = ", ".join(reasons) if reasons else "unknown"
            print(f"  - {trial_id}: FAILED ({status}) at iteration {iteration}")

        print(f"")
        print(f"Possible causes:")
        print(f"  - Hyperparameter search space too wide (learning rate too high)")
        print(f"  - Lambda_kl too low (KL divergence control insufficient)")
        print(f"  - Training instability (mode collapse from iteration 0)")
        print(f"  - Batch size too large (gradient instability)")
        print(f"")
        print(f"Recommended fixes:")
        print(f"  1. Constrain hyperparameter ranges (±10-20% of working baseline)")
        print(f"  2. Increase lambda_kl minimum (e.g., [0.001, 0.0015])")
        print(f"  3. Reduce learning rate range (e.g., [1.8e-5, 2.2e-5])")
        print(f"  4. Increase warmup_steps minimum (e.g., [100, 120])")
        print(f"  5. Reduce batch size (e.g., 4 instead of 8)")
        print(f"{'!'*80}\n")


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
        PopulationBasedTraining scheduler (synchronous mode)

    How it works (synchronous PBT with synch=True):
    - Every mutation_interval iterations: ALL workers pause and wait
    - Once all workers sync at same iteration: compare all workers fairly
    - Bottom 50% copy weights from top 50% (exploit)
    - Then mutate hyperparameters ±20% (explore)

    Why synchronous (synch=True) vs asynchronous (synch=False, default):
    - synch=True: Fair comparison (all workers at same iteration)
    - synch=False: Faster but unfair (worker A at iter 5 vs worker B at iter 4)
    - Trade-off: Slower but scientifically correct (prevents apples-to-oranges comparison)
    """
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",  # HuggingFace reports training_iteration, not step
        metric=metric,
        mode=mode,
        perturbation_interval=1,  # 1 iteration = save_steps (e.g., 100 steps)
        hyperparam_mutations=hyperparam_mutations,
        quantile_fraction=0.5,  # Top 50% vs bottom 50%
        synch=True,  # ✅ SYNCHRONOUS: All workers sync at same iteration before comparison (fair exploit/explore)
    )
    return scheduler


def run_pbt_training(
    trainable,
    hp_space,
    scheduler,
    num_workers=3,
    max_iterations=10,
    output_dir="./outputs/ray_results",
    name="pbt_training",
    gpu_fraction=1.0
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
        gpu_fraction: GPU fraction per worker (default: 1.0)
            Examples:
            - 1.0 = 1 worker uses 1 full GPU (sequential)
            - 0.5 = 2 workers share 1 GPU (parallel)
            - 0.167 = 6 workers share 1 GPU (parallel, for A100-80GB)

    Returns:
        analysis: Ray Tune analysis object with best hyperparameters
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Extract max_steps from hp_space for secondary stop condition
    max_steps = hp_space.get("max_steps", 1000) if isinstance(hp_space, dict) else 1000

    # Create safety stopper (aborts if ALL workers fail)
    safety_stopper = AllWorkersSafetyStopper(margin_tolerance=0.0)

    # Run hyperparameter search with PBT
    # IMPORTANT: Multiple stop conditions to prevent off-by-one bug AND GPU waste
    # 1. Ray Tune stop dict: Prevents off-by-one bug (dual condition)
    # 2. AllWorkersSafetyStopper: Aborts if ALL workers fail (prevents GPU waste)
    analysis = tune.run(
        trainable,
        name=name,
        config=hp_space,
        scheduler=scheduler,
        num_samples=num_workers,
        resources_per_trial={"gpu": gpu_fraction},  # ✅ Configurable GPU allocation per worker
        stop={
            "training_iteration": max_iterations,  # Primary: Stop after N checkpoints
            "timesteps_total": max_steps,  # Secondary: Stop after N training steps (safety net)
        },
        # ✅ GPU-EFFICIENT: Abort if ALL workers fail (OR logic: negative margin OR gibberish)
        # Prevents wasting GPU time training failed models until max_steps
        stopper=safety_stopper,
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
