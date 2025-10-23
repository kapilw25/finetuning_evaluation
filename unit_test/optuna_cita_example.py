#!/usr/bin/env python3
"""
Optuna integration for CITA training (replaces PBT)
Truly adaptive hyperparameter search with early stopping
"""

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

# Your existing CITA training imports
# from comparative_study.03a_CITA_Baseline.Llama3_BF16_PBT import train_cita_with_pbt


def objective(trial):
    """
    Optuna objective function for CITA training

    Each trial:
    1. Suggest hyperparameters (Optuna decides based on TPE)
    2. Train CITA model
    3. Report harmlessness score
    4. Prune if gibberish detected

    Args:
        trial: Optuna trial object

    Returns:
        float: Harmlessness score (higher = better)
    """

    # ===== HYPERPARAMETER SAMPLING (Dynamic - TPE decides) =====
    config = {
        # Same ranges as PBT, but Optuna dynamically focuses sampling
        "lambda_kl": trial.suggest_float("lambda_kl", 0.001, 0.0015, log=False),
        "learning_rate": trial.suggest_float("learning_rate", 8e-6, 1.2e-5, log=True),
        "beta": trial.suggest_float("beta", 0.08, 0.12),
        "weight_decay": trial.suggest_float("weight_decay", 0.008, 0.012),
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 120),

        # Fixed params
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "save_steps": 50,  # Safety check every 50 steps
        "max_steps": 200,  # SANITY mode
    }

    # ===== TRAIN CITA MODEL =====
    # Replace this with your actual training function
    # result = train_cita_with_pbt(config)
    # harmlessness_score = result["final_harmlessness"]
    # gibberish_detected = result["gibberish_detected"]

    # Mock training (replace with actual)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from unit_test.adaptive_optuna_demo import mock_cita_training
    harmlessness_score, gibberish_detected = mock_cita_training(
        config["lambda_kl"],
        config["learning_rate"],
        config["beta"],
        config["weight_decay"],
        config["warmup_steps"]
    )

    # ===== EARLY STOPPING (Gibberish Detection) =====
    if gibberish_detected:
        raise optuna.TrialPruned(f"Gibberish detected at step 50 - negative margin")

    # ===== INTERMEDIATE PRUNING (Optional - for step-by-step monitoring) =====
    # If you want to prune bad trials at step 50, 100, 150:
    # for step in [50, 100, 150]:
    #     step_harmlessness = get_harmlessness_at_step(step)
    #     trial.report(step_harmlessness, step)
    #
    #     if trial.should_prune():
    #         raise optuna.TrialPruned(f"Pruned at step {step} (Hyperband)")

    return harmlessness_score


def run_optuna_cita_search(n_trials=27, timeout_hours=15):
    """
    Run Optuna hyperparameter search for CITA

    Features:
    - TPE sampler: Dynamically focuses on promising regions
    - Hyperband pruner: Early stopping for gibberish trials
    - TensorBoard logging: Visualize optimization progress
    - Checkpoint resumption: Resume interrupted searches

    Args:
        n_trials: Number of trials (default: 27)
        timeout_hours: Max time to run (default: 15 hours)

    Returns:
        best_params: Best hyperparameters found
    """

    # ===== CREATE STUDY =====
    study = optuna.create_study(
        study_name="cita_hyperparameter_search",
        direction="maximize",  # Maximize harmlessness score

        # ✅ ADAPTIVE: TPE dynamically models P(good|params) and P(bad|params)
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,  # Random exploration first (build model)
            multivariate=True,  # Model correlations between params
        ),

        # ✅ EARLY STOPPING: Hyperband kills bad trials at 50/100/150 steps
        pruner=HyperbandPruner(
            min_resource=50,  # Min steps before pruning
            max_resource=200,  # Max steps
            reduction_factor=2,  # Aggressiveness
        ),

        # ✅ RESUMPTION: Load from DB if exists
        storage="sqlite:///outputs/optuna_cita.db",
        load_if_exists=True,
    )

    # ===== RUN OPTIMIZATION =====
    print("="*80)
    print("🔬 OPTUNA ADAPTIVE SEARCH FOR CITA")
    print("="*80)
    print(f"Trials: {n_trials}, Timeout: {timeout_hours}h")
    print(f"Sampler: TPE (adaptive)")
    print(f"Pruner: Hyperband (early stopping)")
    print("="*80 + "\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_hours * 3600,
        show_progress_bar=True,

        # ✅ TENSORBOARD: Optional - Install optuna-integration[tensorboard]
        # callbacks=[TensorBoardCallback("outputs/optuna_logs", metric_name="harmlessness")],
    )

    # ===== RESULTS =====
    print("\n" + "="*80)
    print("🏆 BEST HYPERPARAMETERS FOUND")
    print("="*80)

    best_params = study.best_params
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n  Best harmlessness: {study.best_value:.2f}/10")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print("="*80 + "\n")

    # ===== SAVE BEST CONFIG =====
    import json
    with open("outputs/best_optuna_config.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("✅ Best config saved to: outputs/best_optuna_config.json\n")

    return best_params


if __name__ == "__main__":
    # Run Optuna search (27 trials, ~1.5 hours for SANITY mode)
    best_params = run_optuna_cita_search(n_trials=27, timeout_hours=2)

    print("🎯 Next Steps:")
    print("1. Train full model with best hyperparameters")
    print("2. Evaluate on test set")
    print("3. Compare vs DPO baseline (4.18/10 harmlessness)")
