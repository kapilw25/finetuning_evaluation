"""
Adaptive CITA Training Script (Optuna-based)
Truly adaptive hyperparameter search using Optuna TPE + Hyperband

Early Stopping Strategy:
- Checks at steps: 50, 100, 150, 200 (NOT arbitrary steps)
- Stops immediately on: gibberish OR negative margin OR high KL
- Research-backed: 80% of harmful outputs detected within first 30%

Time Estimates (DPO baseline: 34.55 min/200 steps = 0.173 min/step):
- MVP mode (5 × 100 steps): ~87 min = 1.5 hours (with pruning: ~1.2 hours)
- Sanity mode (27 × 200 steps): ~933 min = 15.5 hours (with pruning: ~13 hours)
- Full mode (27 × 1000 steps): ~4664 min = 78 hours (with pruning: ~66 hours)

Usage:
    # MVP (5 trials, 100 steps, ~1.5 hours) - VALIDATES OPTUNA + EARLY STOPPING
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive.py --mode mvp

    # Sanity check (27 trials, 200 steps, ~15.5 hours)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive.py --mode sanity

    # Full training (27 trials, 1000 steps, ~78 hours = 3.25 days)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive.py --mode full

Outputs:
    - Optuna study database: ./outputs/optuna_cita.db
    - Best hyperparameters: ./outputs/best_optuna_config.json
    - Best model checkpoint: ./outputs/CITA_Adaptive/best_trial/
    - Training log: ./logs/CITA_Adaptive_training_<timestamp>.log
"""

import sys
from pathlib import Path
import os
import argparse
from datetime import datetime
import json

# ===== FIX CUDA OOM: Enable expandable segments for memory fragmentation =====
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

# Add utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# ===================================================================
# Advanced Logging Setup (Tee System)
# ===================================================================
from logging_utils import setup_training_logger, restore_logging

# Setup logging to capture ALL terminal output (stdout + stderr)
log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
    run_name="CITA_Adaptive",
    project_root=project_root
)

# ===================================================================
# HuggingFace Configuration
# ===================================================================
from model_utils import load_hf_token, get_model_repo_name

# Load HuggingFace token and authenticate
HF_TOKEN = load_hf_token(project_root)

# Get HuggingFace repository name
RUN_NAME = "CITA_Adaptive"
HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

print(f"📦 Model will be pushed to: {HF_REPO}")
print("="*80 + "\n")


# ===================================================================
# Optuna Objective Function (CITA Training)
# ===================================================================

def train_cita_trial(trial, max_steps=200, base_model=None):
    """
    Single Optuna trial - trains CITA model with suggested hyperparameters

    Args:
        trial: Optuna trial object
        max_steps: Maximum training steps (default: 200 for sanity, 1000 for full)
        base_model: HuggingFace model ID to load LoRA adapters from (for stacking)

    Returns:
        float: Final margin (higher = better, model prefers safe responses)

    Raises:
        optuna.TrialPruned: If gibberish detected or negative margin
    """

    # ===== HYPERPARAMETER SAMPLING (Optuna TPE decides dynamically) =====
    lambda_kl = trial.suggest_float("lambda_kl", 0.001, 0.0015, log=False)
    learning_rate = trial.suggest_float("learning_rate", 8e-6, 1.2e-5, log=True)
    beta = trial.suggest_float("beta", 0.08, 0.12)
    weight_decay = trial.suggest_float("weight_decay", 0.008, 0.012)
    warmup_steps = trial.suggest_int("warmup_steps", 100, 120)

    print(f"\n{'='*80}")
    print(f"🔬 Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*80}")
    print(f"  lambda_kl:      {lambda_kl:.6f}")
    print(f"  learning_rate:  {learning_rate:.6e}")
    print(f"  beta:           {beta:.4f}")
    print(f"  weight_decay:   {weight_decay:.4f}")
    print(f"  warmup_steps:   {warmup_steps}")
    print(f"")
    print(f"  Training config:")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Batch size: 1 (per device)")
    print(f"  - Gradient accumulation: 8 (effective batch = 8)")
    print(f"  - Safety checks: Every 50 steps")
    print(f"  - Early stopping: Enabled (gibberish/negative margin/high KL)")
    print(f"{'='*80}\n")

    # ===== LOAD MODEL & TOKENIZER =====
    from model_utils import load_model_bf16

    model, tokenizer = load_model_bf16(
        model_id="meta-llama/Llama-3.1-8B",
        max_seq_length=2048,
        use_flash_attention=True
    )

    # ===== LOAD BASE MODEL LORA (IF STACKING) =====
    if base_model:
        print(f"\n🔗 Loading LoRA adapters from HuggingFace: {base_model}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, base_model, token=HF_TOKEN)
        print("✅ LoRA adapters loaded")

        # Merge adapters into base model
        print("🔄 Merging LoRA adapters into base model...")
        merged_model = model.merge_and_unload()

        # Clear PEFT config to avoid warnings
        try:
            delattr(merged_model, 'peft_config')
        except AttributeError:
            pass
        try:
            delattr(merged_model, '_hf_peft_config_loaded')
        except AttributeError:
            pass

        model = merged_model
        print("✅ LoRA adapters merged\n")

    # ===== APPLY LORA ADAPTERS =====
    from model_utils import setup_lora

    model = setup_lora(
        model,
        r=16,
        lora_alpha=16,
        use_gradient_checkpointing=True
    )

    # ===== TORCH.COMPILE() OPTIMIZATION =====
    from model_utils import apply_torch_compile
    model = apply_torch_compile(model)

    # ===== LOAD DATASET =====
    from data_prep import load_pku_filtered, format_dataset

    # Note: Both use split="train" but return_val flag controls the actual split
    # return_val=False → 90% of train split (for training)
    # return_val=True  → 10% of train split (for validation)
    dataset_raw_train = load_pku_filtered(
        split="train",
        max_samples=None,
        return_val=False  # 90% of train split
    )

    dataset_raw_val = load_pku_filtered(
        split="train",
        max_samples=None,
        return_val=True   # 10% of train split
    )

    # Format dataset for CITA (DPO format)
    train_dataset = format_dataset(dataset_raw_train, method="cita")
    val_dataset = format_dataset(dataset_raw_val, method="cita")

    # ===== CREATE TRAINING ARGS =====
    trial_output_dir = project_root / "outputs" / "CITA_Adaptive" / f"trial_{trial.number}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard logging for this trial
    tensorboard_dir = project_root / "tensorboard_logs" / f"CITA_Adaptive_trial_{trial.number}"

    training_args = DPOConfig(
        output_dir=str(trial_output_dir),
        per_device_train_batch_size=1,  # Reduced for DPO ref model
        gradient_accumulation_steps=8,  # Effective batch=8
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_torch",
        # optim="adamw_torch_fused",  # FIXED: Fused version handles BF16 correctly (adamw_torch has dtype bugs in PyTorch 2.5.1)
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        seed=3407,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=str(tensorboard_dir),
        logging_first_step=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        # Validation
        eval_strategy="steps",
        eval_steps=50,
        per_device_eval_batch_size=1,
        # DPO-specific parameters
        beta=beta,
        max_length=2048,
        max_prompt_length=1024,
    )

    # ===== CALLBACKS: SAFETY MONITORING + TRAINING SUMMARY =====
    from model_utils import get_test_prompts
    from monitoring_callback import GibberishDetectionCallback, TrainingSummaryCallback

    test_prompts = get_test_prompts()

    # ✅ AGGRESSIVE EARLY STOPPING
    # Checks at steps: 50, 100, 150, 200 (check_every_n_steps=50)
    # Stops immediately on: gibberish OR negative margin OR high KL
    safety_callback = GibberishDetectionCallback(
        test_prompts=test_prompts,
        check_every_n_steps=50,
        repetition_threshold=0.5,
        diversity_threshold=15,
        stop_on_gibberish=True,  # ✅ STOP IMMEDIATELY
        use_alpaca_format=True,
        stop_on_negative_margin=True,  # ✅ STOP IMMEDIATELY
        margin_tolerance=0.0,
        stop_on_high_kl=True,  # ✅ STOP IMMEDIATELY
        kl_threshold=0.5
    )

    # ✅ TRAINING SUMMARY (prints every 50 steps)
    summary_callback = TrainingSummaryCallback(
        training_method="cita",
        check_every_n_steps=50,
        window_size=50  # Last 50 logs for statistics
    )

    # ===== CREATE CITA TRAINER =====
    from cita_trainer import CITATrainer

    trainer = CITATrainer(
        model=model,
        tokenizer=tokenizer,  # CITATrainer expects 'tokenizer', not 'processing_class'
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        lambda_kl=lambda_kl,
        callbacks=[safety_callback, summary_callback],
    )

    # ===== SHOW GPU MEMORY (BEFORE TRAINING) =====
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved before training.\n")

    # ===== TRAIN =====
    print(f"\n{'='*80}")
    print(f"🏋️  Training CITA (Trial {trial.number})...")
    print(f"{'='*80}\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print(f"\n⚠️  Trial {trial.number} interrupted by user")
        del model
        del trainer
        torch.cuda.empty_cache()
        raise optuna.TrialPruned("User interrupted training")

    # ===== CHECK IF STOPPED EARLY =====
    current_step = trainer.state.global_step

    if current_step < max_steps:
        # Stopped early - check why
        prune_reasons = []

        if hasattr(safety_callback, 'negative_margin_violations') and safety_callback.negative_margin_violations > 0:
            prune_reasons.append(f"negative margin (×{safety_callback.negative_margin_violations})")

        if hasattr(safety_callback, 'kl_violations') and safety_callback.kl_violations > 0:
            prune_reasons.append(f"high KL (×{safety_callback.kl_violations})")

        if not prune_reasons:
            prune_reasons.append("gibberish detected")

        prune_message = f"Early stop at step {current_step}: {', '.join(prune_reasons)}"
        print(f"\n⚠️  {prune_message}")

        del model
        del trainer
        torch.cuda.empty_cache()

        raise optuna.TrialPruned(prune_message)

    # ===== EXTRACT FINAL MARGIN =====
    final_margin = None

    if hasattr(trainer.state, 'log_history') and len(trainer.state.log_history) > 0:
        for log_entry in reversed(trainer.state.log_history):
            if 'cita/margin' in log_entry:
                final_margin = log_entry['cita/margin']
                break
            elif 'eval_cita/margin' in log_entry:
                final_margin = log_entry['eval_cita/margin']
                break

    # ===== CHECK FINAL MARGIN =====
    if final_margin is not None:
        if final_margin <= 0:
            print(f"\n❌ Trial {trial.number}: Final margin = {final_margin:.4f} (≤ 0)")
            del model
            del trainer
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"Negative final margin: {final_margin:.4f}")
    else:
        # No margin - use eval_loss
        print(f"\n⚠️  No margin found, falling back to eval_loss")
        for log_entry in reversed(trainer.state.log_history):
            if 'eval_loss' in log_entry:
                final_eval_loss = log_entry['eval_loss']
                print(f"   Final eval_loss: {final_eval_loss:.4f}")
                del model
                del trainer
                torch.cuda.empty_cache()
                return -final_eval_loss

        # No metrics
        print(f"\n❌ Trial {trial.number}: No metrics found")
        del model
        del trainer
        torch.cuda.empty_cache()
        raise optuna.TrialPruned("No metrics found")

    # ===== SHOW FINAL MEMORY =====
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_training = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)

    print(f"\n{'='*80}")
    print(f"✅ Training complete!")
    print(f"Peak reserved memory = {used_memory} GB")
    print(f"Peak reserved memory for training = {used_memory_for_training} GB")
    print(f"Peak reserved memory % of max memory = {used_percentage}%")
    print(f"{'='*80}\n")

    # ===== SAVE TRIAL CHECKPOINT =====
    lora_output_dir = trial_output_dir / "lora_adapters"
    lora_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_output_dir))
    tokenizer.save_pretrained(str(lora_output_dir))

    trial_config = {
        "trial_number": trial.number,
        "lambda_kl": lambda_kl,
        "learning_rate": learning_rate,
        "beta": beta,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "final_margin": final_margin,
        "completed_steps": current_step,
    }

    with open(trial_output_dir / "trial_config.json", 'w') as f:
        json.dump(trial_config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Trial {trial.number} complete!")
    print(f"   Final margin: {final_margin:.4f}")
    print(f"   Checkpoint saved: {lora_output_dir}")
    print(f"{'='*80}\n")

    # Clean up
    del model
    del trainer
    torch.cuda.empty_cache()

    return final_margin


# ===================================================================
# Optuna Study Runner
# ===================================================================

def run_optuna_cita_search(
    n_trials=27,
    max_steps=200,
    base_model=None,
    timeout_hours=15
):
    """
    Run Optuna adaptive hyperparameter search for CITA

    Args:
        n_trials: Number of trials
        max_steps: Training steps per trial
        base_model: HuggingFace model ID to load LoRA adapters from
        timeout_hours: Max time to run

    Returns:
        best_trial: Best Optuna trial with hyperparameters and checkpoint
    """

    # ===== CREATE OPTUNA STUDY =====
    study_name = f"cita_adaptive_{max_steps}steps"
    storage_path = str(project_root / "outputs" / "optuna_cita.db")

    # Ensure outputs directory exists (SQLite needs parent directory)
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"🔬 OPTUNA ADAPTIVE SEARCH FOR CITA")
    print(f"{'='*80}")
    print(f"Study name: {study_name}")
    print(f"Storage: {storage_path}")
    print(f"Trials: {n_trials}")
    print(f"Steps per trial: {max_steps}")
    print(f"Timeout: {timeout_hours}h")
    print(f"Sampler: TPE (truly adaptive)")
    print(f"Pruner: Hyperband (early stopping)")
    print(f"Safety checks: Steps 50, 100, 150, 200")
    print(f"{'='*80}\n")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",

        sampler=TPESampler(
            seed=42,
            n_startup_trials=min(10, n_trials // 3),  # 10 warmup or 1/3 of trials
            multivariate=True,
        ),

        pruner=HyperbandPruner(
            min_resource=50,
            max_resource=max_steps,
            reduction_factor=2,
        ),

        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    # ===== RUN OPTIMIZATION =====
    def objective(trial):
        return train_cita_trial(trial, max_steps=max_steps, base_model=base_model)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_hours * 3600,
        show_progress_bar=True,
    )

    # ===== RESULTS =====
    print(f"\n{'='*80}")
    print(f"🏆 BEST HYPERPARAMETERS FOUND")
    print(f"{'='*80}")

    best_trial = study.best_trial
    best_params = best_trial.params

    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n  Best margin: {best_trial.value:.4f}")
    print(f"  Best trial: {best_trial.number}")
    print(f"  Total trials: {len(study.trials)}")

    # Count pruned trials
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)} ({100*len(pruned_trials)/len(study.trials):.1f}%)")
    print(f"{'='*80}\n")

    # ===== SAVE BEST CONFIG =====
    config_path = project_root / "outputs" / "best_optuna_config.json"
    config_path.parent.mkdir(exist_ok=True)

    best_config = {
        "method": "CITA_Adaptive",
        "max_steps": max_steps,
        "best_trial": best_trial.number,
        "best_margin": best_trial.value,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
        **best_params
    }

    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"✅ Best config saved to: {config_path}\n")

    # ===== COPY BEST CHECKPOINT =====
    best_trial_dir = project_root / "outputs" / "CITA_Adaptive" / f"trial_{best_trial.number}"
    best_checkpoint_dir = project_root / "outputs" / "CITA_Adaptive" / "best_trial"

    if best_trial_dir.exists():
        import shutil
        if best_checkpoint_dir.exists():
            shutil.rmtree(best_checkpoint_dir)
        shutil.copytree(best_trial_dir, best_checkpoint_dir)
        print(f"✅ Best checkpoint copied to: {best_checkpoint_dir}\n")

    return best_trial


# ===================================================================
# Main Execution
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CITA Adaptive Training - Optuna-based hyperparameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MVP (5 trials × 100 steps, ~1.5 hours) - VALIDATES OPTUNA WORKS
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive.py --mode mvp

  # Sanity (27 trials × 200 steps, ~15.5 hours)
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive.py --mode sanity

  # Full (27 trials × 1000 steps, ~78 hours)
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive.py --mode full
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["mvp", "sanity", "full"],
        default="mvp",
        help="Training mode: 'mvp' (5×100), 'sanity' (27×200), 'full' (27×1000)"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides --mode)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Training steps per trial (overrides --mode)"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="HuggingFace model ID to load LoRA adapters from"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Timeout in hours (default: 15)"
    )

    args = parser.parse_args()

    # Determine configuration
    if args.trials is not None and args.steps is not None:
        n_trials = args.trials
        max_steps = args.steps
        print(f"✅ Custom: {n_trials} trials × {max_steps} steps")
    elif args.mode == "mvp":
        n_trials = 5
        max_steps = 100
        print(f"✅ MVP mode: {n_trials} trials × {max_steps} steps (~1.5 hours)")
    elif args.mode == "sanity":
        n_trials = 27
        max_steps = 200
        print(f"✅ Sanity mode: {n_trials} trials × {max_steps} steps (~15.5 hours)")
    else:  # full
        n_trials = 27
        max_steps = 1000
        print(f"✅ Full mode: {n_trials} trials × {max_steps} steps (~78 hours)")

    # Time estimate
    time_per_step = 0.173  # minutes (from DPO baseline: 34.55/200)
    time_per_trial = max_steps * time_per_step
    total_time = n_trials * time_per_trial
    total_time_pruned = total_time * 0.85  # 15% savings from early stopping

    print(f"\n{'='*80}")
    print(f"⏱️  TIME ESTIMATE")
    print(f"{'='*80}")
    print(f"Time per trial: {time_per_trial:.1f} min")
    print(f"Total (no pruning): {total_time:.1f} min ({total_time/60:.1f} hours)")
    print(f"Total (with early stopping): {total_time_pruned:.1f} min ({total_time_pruned/60:.1f} hours)")
    print(f"")
    print(f"Early stopping:")
    print(f"  - Checks at steps: 50, 100, 150, 200")
    print(f"  - Stops immediately on: gibberish OR negative margin OR high KL")
    print(f"  - Expected pruning: 15-20% of trials")
    print(f"{'='*80}\n")

    # Confirm
    proceed = input(f"Proceed with {n_trials} trials? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Aborted")
        sys.exit(0)

    # Run
    try:
        best_trial = run_optuna_cita_search(
            n_trials=n_trials,
            max_steps=max_steps,
            base_model=args.base_model,
            timeout_hours=args.timeout
        )

        print(f"\n🏁 CITA Adaptive Training Complete!")
        print(f"📝 Log file: {log_filename}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"📝 Complete log saved: {log_filename}")
