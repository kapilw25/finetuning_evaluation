"""
Adaptive CITA Training Script (Optuna-based) - Updated for 1354-step FULL training
Truly adaptive hyperparameter search using Optuna TPE + Hyperband

Features:
- Conditional HP spaces: CITA_Instruct uses 50-70% lower LR than CITA_NoInstruct
- Prevents gradient explosions from longer instruction sequences
- Multi-objective optimization: [margin, accuracy, -eval_loss]

Time Estimates (10 trials × 1354 steps, with pruning):
- CITA_NoInstruct: ~20-24 hours
- CITA_Instruct: ~20-24 hours

Usage:
    # CITA_NoInstruct - Full search (10 trials × 1354 steps, ~20-24 hours)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py --mode full \
        --use-instruction false --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct

    # CITA_Instruct - Full search (conservative HP space for longer sequences)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py --mode full \
        --use-instruction true --base_model kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct

    # MVP (5 trials × 100 steps, ~1.5 hours) - VALIDATES OPTUNA + EARLY STOPPING
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py --mode mvp \
        --use-instruction false

    # Sanity check (27 trials × 200 steps, ~15.5 hours)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py --mode sanity \
        --use-instruction true

Outputs:
    - Optuna study database: ./outputs/optuna_cita_noinstruct.db or ./outputs/optuna_cita_instruct.db
    - Best hyperparameters: ./outputs/best_optuna_config.json
    - Best model checkpoint: ./outputs/CITA_NoInstruct_Adaptive/best_trial/ or ./outputs/CITA_Instruct_Adaptive/best_trial/
    - Training log: ./logs/CITA_NoInstruct_Adaptive_training_<timestamp>.log or ./logs/CITA_Instruct_Adaptive_training_<timestamp>.log
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

# ===== FIX torch.compile() CUDAGraph bug: Disable CUDAGraphs for dynamic shapes =====
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

# Add utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# ===================================================================
# Logging Setup (will be done AFTER argparse to use instruction-aware RUN_NAME)
# ===================================================================
from logging_utils import setup_training_logger, restore_logging

# Note: Logging setup moved to main execution block (after args parsing)

# ===================================================================
# HuggingFace Configuration (will be done AFTER argparse to use instruction-aware RUN_NAME)
# ===================================================================
from model_utils import load_hf_token

# Note: HF_TOKEN will be set in main execution block (after args parsing)
# Optuna trials are intermediate runs and NOT pushed to HuggingFace


# ===================================================================
# Optuna Objective Function (CITA Training)
# ===================================================================

def train_cita_trial(trial, max_steps=200, base_model=None, use_instruction=False, run_name="CITA_Adaptive"):
    """
    Single Optuna trial - trains CITA model with suggested hyperparameters

    Args:
        trial: Optuna trial object
        max_steps: Maximum training steps (200 for sanity, 1354 for full)
        base_model: HuggingFace model ID to load LoRA adapters from (for stacking)
        use_instruction: Whether to use instruction conditioning (affects HP space)
        run_name: Run name for output directories (e.g., "CITA_NoInstruct" or "CITA_Instruct")

    Returns:
        float: Final margin (higher = better, model prefers safe responses)

    Raises:
        optuna.TrialPruned: If gibberish detected or negative margin
    """

    # ===== HYPERPARAMETER SAMPLING (Optuna TPE decides dynamically) =====
    # HP space differs for NoInstruct vs Instruct (iter12 finding: Instruct needs lower LR)
    # Evidence: CITA_Instruct exploded with LR=6.83e-06 (grad_norm 103), NoInstruct succeeded
    # Root cause: System instructions add ~50-100 tokens → longer sequences → larger gradients

    if use_instruction:
        # CITA_Instruct: Ultra-conservative HP space for longer sequences (system instructions add ~50-100 tokens)
        # After iter12: 5/5 trials exploded with LR [2e-6, 5e-6] → lowered to [8e-7, 2.5e-6]
        # Uses gradient clipping (max_grad_norm=1.0) to match CITA_NoInstruct iter11 methodology
        lambda_kl = trial.suggest_float("lambda_kl", 0.0001, 0.001, log=False)
        learning_rate = trial.suggest_float("learning_rate", 8e-7, 2.5e-6, log=True)  # Much lower for stability
        beta = trial.suggest_float("beta", 0.08, 0.15)
        weight_decay = trial.suggest_float("weight_decay", 0.005, 0.015)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.10)  # Match SFT/DPO/CITA_NoInstruct (~100 steps)
    else:
        # CITA_NoInstruct: Original HP space from Trial 5 (proven stable)
        lambda_kl = trial.suggest_float("lambda_kl", 0.0005, 0.0015, log=False)
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 8e-6, log=True)
        beta = trial.suggest_float("beta", 0.08, 0.12)
        weight_decay = trial.suggest_float("weight_decay", 0.008, 0.012)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15)  # Fair: matches SFT/DPO

    print(f"\n{'='*80}")
    print(f"🔬 Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*80}")
    print(f"  lambda_kl:      {lambda_kl:.6f}")
    print(f"  learning_rate:  {learning_rate:.6e}")
    print(f"  beta:           {beta:.4f}")
    print(f"  weight_decay:   {weight_decay:.4f}")
    print(f"  warmup_ratio:   {warmup_ratio:.4f}  (auto-scales to {int(warmup_ratio * max_steps)} steps for {max_steps} total)")
    # Get training config values
    per_device_batch = 1
    grad_accum = 8

    print(f"")
    print(f"  Training config:")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Batch size: {per_device_batch} (per device)")
    print(f"  - Gradient accumulation: {grad_accum} (effective batch = {per_device_batch * grad_accum})")
    print(f"  - Callbacks: TrainingSummaryCallback only (matches DPO baseline)")
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

    # Cast to BF16 AFTER applying new LoRA adapters (to ensure all params including new LoRA weights are BF16)
    model = model.to(torch.bfloat16)
    print("✅ Model cast to BF16 (all params including LoRA)")

    # ===== TORCH.COMPILE() OPTIMIZATION =====
    # DISABLED: Causes KeyError: 'hidden_states' during gradient checkpointing
    # torch.compile() + gradient_checkpointing has compatibility issues with CITA (DPO-based)
    # Error occurs in transformers/utils/generic.py wrapped_forward output collection
    # Trade-off: ~10% slower training (59h → ~71h) vs no crash
    print("\n⚠️  torch.compile() disabled (prevents gradient checkpointing bug for CITA)")
    # from model_utils import apply_torch_compile
    # model = apply_torch_compile(model)

    # ===== LOAD DATASET =====
    from data_prep.loader_pku import load_pku_combined_clear_contrast
    from data_prep.formatters import format_pku_for_cita_Instruct, format_pku_for_cita_NoInstruct

    # Load combined dataset (12,035 samples) and split 90/10
    dataset_split = load_pku_combined_clear_contrast(val_split=0.1)

    # Format for CITA (conditional: WITH or WITHOUT instructions)
    formatter = format_pku_for_cita_Instruct if use_instruction else format_pku_for_cita_NoInstruct
    instruction_status = "WITH" if use_instruction else "NO"

    train_dataset = dataset_split['train'].map(
        formatter,
        remove_columns=dataset_split['train'].column_names,
        desc=f"Formatting PKU for CITA ({instruction_status} instructions)"
    )

    val_dataset = dataset_split['test'].map(
        formatter,
        remove_columns=dataset_split['test'].column_names,
        desc=f"Formatting PKU validation for CITA ({instruction_status} instructions)"
    )

    # ===== CREATE TRAINING ARGS =====
    trial_output_dir = project_root / "outputs" / run_name / f"trial_{trial.number}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard logging for this trial
    tensorboard_dir = project_root / "tensorboard_logs" / f"{run_name}_trial_{trial.number}"

    # ===== CALCULATE CHECKPOINT INTERVAL (20% of training, matches production) =====
    checkpoint_interval = int(max_steps * 0.2)  # 1354 steps → 270, 200 steps → 40
    print(f"\n📊 Checkpoint/Eval interval: {checkpoint_interval} steps (20% of {max_steps} total steps)")

    training_args = DPOConfig(
        output_dir=str(trial_output_dir),
        per_device_train_batch_size=1,  # Reduced for DPO ref model
        gradient_accumulation_steps=8,  # Effective batch=8
        warmup_ratio=warmup_ratio,  # Epoch-agnostic warmup (iter7 fix)
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_torch",  # Merged model cast to BF16 for dtype consistency
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        seed=3407,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,  # Enable clipping to match CITA_NoInstruct iter11 (fair comparison)
        # Note: CITA_NoInstruct used clipping, so CITA_Instruct should too for apples-to-apples comparison
        # Future work (Option B): Rerun both NoInstruct & Instruct without clipping for truly stable HPs
        save_steps=checkpoint_interval,  # Dynamic: 20% of training (matches production)
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=str(tensorboard_dir),
        logging_first_step=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        # Validation
        eval_strategy="steps",
        eval_steps=checkpoint_interval,  # Dynamic: 20% of training (matches production)
        per_device_eval_batch_size=1,
        # DPO-specific parameters
        beta=beta,
        max_length=2048,
        max_prompt_length=1024,
    )

    # ===== CALLBACKS: TRAINING SUMMARY ONLY (MATCHES DPO BASELINE) =====
    from monitoring_callback import TrainingSummaryCallback

    # ✅ TRAINING SUMMARY (prints every 50 steps) - MATCHES DPO BASELINE
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
        callbacks=[summary_callback],  # Only summary, no safety (matches DPO baseline)
    )

    # ===== VERIFY GRADIENT CLIPPING IS CONFIGURED =====
    max_grad_norm = getattr(trainer.args, 'max_grad_norm', None)
    if max_grad_norm is None or max_grad_norm == 0:
        print(f"\n⚠️  Gradient clipping DISABLED (intentional for Optuna HP search)")
        print(f"   Finding stable hyperparameters WITHOUT clipping band-aid")
        print(f"   Explosion detector will prune unstable trials (grad_norm > 50.0)")
        print(f"   This finds HPs that are truly stable, not just clipping-masked\n")
    else:
        print(f"\n✅ Gradient clipping ENABLED: max_grad_norm={max_grad_norm}")
        print(f"   WARNING: Clipping may mask unstable HPs!")
        print(f"   Monitor 'cita/grad_norm_clipped' and 'cita/clipping_active' in logs\n")

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
    except ValueError as e:
        # Catch explosion detector from cita_trainer.py (grad_norm > 50.0)
        if "exploded" in str(e).lower() or "grad_norm" in str(e).lower():
            print(f"\n🚨 Trial {trial.number} EXPLOSION DETECTED")
            print(f"   Error: {e}")
            print(f"   Hyperparameters caused training instability")

            # Try to report last eval metric for TPE learning
            # ❌ DISABLED: trial.report() not supported for multi-objective optimization
            # Optuna will learn from final metrics of completed trials instead
            reported = False
            if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
                for log in reversed(trainer.state.log_history):
                    if 'eval_rewards/margins' in log:
                        margin = log['eval_rewards/margins']
                        step = log.get('step', trainer.state.global_step)
                        # trial.report(margin, step)  # ❌ Causes NotImplementedError for multi-objective
                        print(f"   Last stable eval: margin={margin:.4f} at step {step}")
                        print(f"   (Intermediate reporting disabled for multi-objective optimization)")
                        reported = True
                        break

            if not reported:
                print(f"   No eval metrics found (explosion before first eval)")

            print(f"   Cleaning up and marking trial as pruned\n")

            # Clean up GPU memory
            del model
            del trainer
            torch.cuda.empty_cache()

            # Mark as pruned (Optuna continues with next trial)
            raise optuna.TrialPruned(f"Training exploded: {e}")
        else:
            # Other ValueError (not explosion-related) - re-raise to crash study
            raise

    # ===== EXTRACT FINAL METRICS (MULTI-OBJECTIVE) =====
    # Extract metrics FIRST (before checking early stop) so we always have values to return
    current_step = trainer.state.global_step
    final_margin = None
    final_accuracy = None
    final_eval_loss = None

    if hasattr(trainer.state, 'log_history') and len(trainer.state.log_history) > 0:
        # Get most recent eval metrics
        for log_entry in reversed(trainer.state.log_history):
            if 'eval_rewards/margins' in log_entry:
                final_margin = log_entry['eval_rewards/margins']
                final_accuracy = log_entry.get('eval_rewards/accuracies', None)
                final_eval_loss = log_entry.get('eval_loss', None)
                break

    # ===== CHECK IF STOPPED EARLY =====
    if current_step < max_steps:
        # Stopped early (Optuna Hyperband pruning, no safety callbacks)
        prune_message = f"Early stop at step {current_step}: Hyperband pruning"
        print(f"\n⚠️  {prune_message}")
        margin_str = f"{final_margin:.4f}" if final_margin is not None else "N/A"
        accuracy_str = f"{final_accuracy:.4f}" if final_accuracy is not None else "N/A"
        print(f"   Partial metrics: margin={margin_str}, accuracy={accuracy_str}")

        # DON'T raise TrialPruned - return partial metrics instead
        # This allows Optuna multi-objective to use early-stopped trials
        trial.set_user_attr("early_stopped", True)
        trial.set_user_attr("stop_reason", prune_message[:255])  # Truncate to prevent DB overflow

    # ===== CHECK IF NO METRICS =====
    if final_margin is None:
        # No metrics at all - cannot proceed
        print(f"\n❌ Trial {trial.number}: No eval metrics found")
        del model
        del trainer
        torch.cuda.empty_cache()
        raise optuna.TrialPruned("No eval metrics found")

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

    # ===== SAVE TRIAL CHECKPOINT (non-critical, wrapped in try-except) =====
    try:
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
            "warmup_ratio": warmup_ratio,
            "max_steps": max_steps,
            "final_margin": final_margin,
            "final_accuracy": final_accuracy,
            "final_eval_loss": final_eval_loss,
            "completed_steps": current_step,
        }

        with open(trial_output_dir / "trial_config.json", 'w') as f:
            json.dump(trial_config, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✅ Trial {trial.number} complete!")
        print(f"   Final metrics:")
        print(f"     - Margin: {final_margin:.4f}")
        print(f"     - Accuracy: {final_accuracy:.4f}" if final_accuracy is not None else "     - Accuracy: N/A")
        print(f"     - Eval Loss: {final_eval_loss:.4f}" if final_eval_loss is not None else "     - Eval Loss: N/A")
        print(f"   Checkpoint saved: {lora_output_dir}")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to save checkpoint for trial {trial.number}: {e}")
        print(f"   Metrics still recorded - Optuna will continue\n")

    # Clean up (wrapped to ensure Optuna continues even if cleanup fails)
    try:
        del model
        del trainer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️  Warning: Cleanup failed: {e}")

    # Return multi-objective metrics: (margin, accuracy, -eval_loss)
    # Note: negate eval_loss because lower loss = better, but Optuna maximizes
    return (
        final_margin,                                           # Objective 1: maximize margin
        final_accuracy if final_accuracy is not None else 0.0, # Objective 2: maximize accuracy
        -final_eval_loss if final_eval_loss is not None else -999.9  # Objective 3: minimize eval_loss (penalize missing metrics)
    )


# ===================================================================
# Optuna Study Runner
# ===================================================================

def run_optuna_cita_search(
    n_trials=27,
    max_steps=200,
    base_model=None,
    use_instruction=False,
    timeout_hours=15,
    run_name="CITA_Adaptive"
):
    """
    Run Optuna adaptive hyperparameter search for CITA

    Args:
        n_trials: Number of trials
        max_steps: Training steps per trial
        base_model: HuggingFace model ID to load LoRA adapters from
        use_instruction: Whether to use instruction conditioning (affects HP space)
        timeout_hours: Max time to run
        run_name: Run name for output directories (e.g., "CITA_NoInstruct" or "CITA_Instruct")

    Returns:
        best_trial: Best Optuna trial with hyperparameters and checkpoint
    """

    # ===== CREATE OPTUNA STUDY =====
    instruction_suffix = "_instruct" if use_instruction else "_noinstruct"
    study_name = f"cita_adaptive_{max_steps}steps{instruction_suffix}"
    storage_path = str(project_root / "outputs" / f"optuna_cita{instruction_suffix}.db")

    # Ensure outputs directory exists (SQLite needs parent directory)
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

    # ===== ASK: FRESH RUN OR CONTINUE? =====
    db_exists = Path(storage_path).exists()

    if db_exists:
        print(f"\n{'='*80}")
        print(f"📊 EXISTING OPTUNA DATABASE FOUND")
        print(f"{'='*80}")
        print(f"Location: {storage_path}")
        print(f"\nOptions:")
        print(f"  1) Fresh run (delete previous trials, start from scratch)")
        print(f"  2) Continue from previous run (load existing trials)")
        print(f"{'='*80}\n")

        choice = input("Select option (1 or 2): ").strip()

        if choice == "1":
            print(f"\n🗑️  Deleting previous database: {storage_path}")
            Path(storage_path).unlink()
            print(f"✅ Database deleted. Starting fresh run.\n")
        elif choice == "2":
            print(f"\n♻️  Continuing from previous run.\n")
        else:
            print(f"\n⚠️  Invalid choice '{choice}'. Defaulting to continue (option 2).\n")
    else:
        print(f"\n✅ No existing database found. Starting fresh run.\n")

    print(f"\n{'='*80}")
    print(f"🔬 OPTUNA ADAPTIVE SEARCH FOR CITA")
    print(f"{'='*80}")
    print(f"Study name: {study_name}")
    print(f"Storage: {storage_path}")
    print(f"Trials: {n_trials}")
    print(f"Steps per trial: {max_steps}")
    print(f"Timeout: {timeout_hours}h")
    print(f"Sampler: TPE (truly adaptive)")
    print(f"Pruner: Hyperband (min_resource=50, early stopping)")
    print(f"Monitoring: TrainingSummaryCallback every 50 steps")
    print(f"{'='*80}\n")

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "maximize", "maximize"],  # Multi-objective: [margin, accuracy, -eval_loss]

        sampler=TPESampler(
            seed=42,
            n_startup_trials=5,  # Reduced from 10 - TPE learns faster (avoids repeating explosions)
            multivariate=True,  # Keeps correlations (lr↑ needs warmup↑)
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
        return train_cita_trial(trial, max_steps=max_steps, base_model=base_model, use_instruction=use_instruction, run_name=run_name)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_hours * 3600,
        show_progress_bar=True,
    )

    # ===== RESULTS (MULTI-OBJECTIVE) =====
    print(f"\n{'='*80}")
    print(f"🏆 PARETO-OPTIMAL TRIALS FOUND")
    print(f"{'='*80}")

    # For multi-objective, there's no single "best" trial - get Pareto front
    best_trials = study.best_trials  # Returns list of Pareto-optimal trials

    if len(best_trials) == 0:
        print("⚠️  No completed trials found")
        return None

    # Show all Pareto-optimal trials
    print(f"\nFound {len(best_trials)} Pareto-optimal trial(s):\n")
    for i, trial in enumerate(best_trials):
        print(f"  Trial {trial.number} (Pareto solution {i+1}/{len(best_trials)}):")
        for key, value in trial.params.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        print(f"    Objectives: margin={trial.values[0]:.4f}, accuracy={trial.values[1]:.4f}, -eval_loss={trial.values[2]:.4f}")
        print()

    # Pick the trial with highest margin as "best" for saving
    best_trial = max(best_trials, key=lambda t: t.values[0])  # Max margin
    best_params = best_trial.params

    print(f"✅ Selected trial {best_trial.number} (highest margin) for best config")
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
        "best_margin": best_trial.values[0],
        "best_accuracy": best_trial.values[1],
        "best_neg_eval_loss": best_trial.values[2],
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
        "pareto_optimal_trials": len(best_trials),
        # Optuna-optimized hyperparameters
        **best_params,
        # Fixed hyperparameters (not tuned by Optuna)
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "max_seq_length": 2048,
        "max_prompt_length": 1024,
    }

    try:
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"✅ Best config saved to: {config_path}\n")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save best config: {e}\n")

    # ===== COPY BEST CHECKPOINT (non-critical) =====
    try:
        best_trial_dir = project_root / "outputs" / run_name / f"trial_{best_trial.number}"
        best_checkpoint_dir = project_root / "outputs" / run_name / "best_trial"

        if best_trial_dir.exists():
            import shutil
            if best_checkpoint_dir.exists():
                shutil.rmtree(best_checkpoint_dir)
            shutil.copytree(best_trial_dir, best_checkpoint_dir)
            print(f"✅ Best checkpoint copied to: {best_checkpoint_dir}\n")
        else:
            print(f"⚠️  Warning: Best trial dir not found: {best_trial_dir}\n")
    except Exception as e:
        print(f"⚠️  Warning: Failed to copy best checkpoint: {e}\n")

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
  # CITA_NoInstruct - Full (10 trials × 1354 steps, ~20-24 hours)
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py --mode full \
      --use-instruction false --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct

  # CITA_Instruct - Full (10 trials × 1354 steps, ~20-24 hours)
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py --mode full \
      --use-instruction true --base_model kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct

  # MVP (5 trials × 100 steps, ~1.5 hours) - VALIDATES OPTUNA WORKS
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py --mode mvp \
      --use-instruction false

  # Sanity (27 trials × 200 steps, ~15.5 hours)
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py --mode sanity \
      --use-instruction true
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["mvp", "sanity", "full"],
        default="mvp",
        help="Training mode: 'mvp' (5×100), 'sanity' (27×200), 'full' (27×1354)"
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
        "--use-instruction",
        type=str,
        required=True,
        choices=["true", "false"],
        help="REQUIRED: Use instruction conditioning (true=CITA_Instruct, false=CITA_NoInstruct). Affects HP search space."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Timeout in hours (default: 15)"
    )

    args = parser.parse_args()

    # ===================================================================
    # Set USE_INSTRUCTION and RUN_NAME from command-line argument
    # ===================================================================
    use_instruction = args.use_instruction.lower() == "true"
    RUN_NAME = "CITA_Instruct_Adaptive" if use_instruction else "CITA_NoInstruct_Adaptive"

    print(f"✅ Instruction mode: {'ENABLED' if use_instruction else 'DISABLED'} ({RUN_NAME})")

    # ===================================================================
    # Setup Logging (now that we have RUN_NAME)
    # ===================================================================
    log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
        run_name=RUN_NAME,
        project_root=project_root
    )

    # ===================================================================
    # HuggingFace Authentication (now that we have RUN_NAME)
    # ===================================================================
    HF_TOKEN = load_hf_token(project_root)

    # Note: Optuna trials are NOT pushed to HuggingFace (intermediate search runs)
    # HF_REPO would only be used if we decided to push the best trial
    print("="*80 + "\n")

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
        max_steps = 1354  # Match current Llama3_BF16.py FULL epoch
        print(f"✅ Full mode: {n_trials} trials × {max_steps} steps (~59 hours with pruning)")

    # Time estimate (based on actual CITA runs with eval margin stopping)
    # Actual data from logs/CITA_Adaptive_training_20251023_194316.log:
    # - Trials 11-14: avg 9.75 min for 50 steps (early stop at eval check)
    # - Estimated: 19.5 min for 100 steps (no early stop)
    # - torch.compile() disabled: +15% overhead (0.195 → 0.224 min/step)
    time_per_step = 0.224  # minutes (CITA without torch.compile: 22.4 min / 100 steps)
    time_per_trial = max_steps * time_per_step
    total_time = n_trials * time_per_trial

    # Early stopping is very effective with eval margin checks (stops at 50 steps)
    # Actual savings: ~50% (9.75 min vs 19.5 min for 100 steps)
    total_time_pruned = total_time * 0.5  # 50% time with early stopping

    print(f"\n{'='*80}")
    print(f"⏱️  TIME ESTIMATE")
    print(f"{'='*80}")
    print(f"Time per trial: {time_per_trial:.1f} min (no early stop)")
    print(f"Total (no early stopping): {total_time:.1f} min ({total_time/60:.1f} hours)")
    print(f"Total (with early stopping): {total_time_pruned:.1f} min ({total_time_pruned/60:.1f} hours)")
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
            use_instruction=use_instruction,
            timeout_hours=args.timeout,
            run_name=RUN_NAME
        )

        print(f"\n🏁 {RUN_NAME} Optuna Search Complete!")
        print(f"📝 Log file: {log_filename}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"📝 Complete log saved: {log_filename}")
