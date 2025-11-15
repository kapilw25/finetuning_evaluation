"""
CITA Baseline Training Script (BF16 precision)
Calibrated Instruction Tuning with Alignment - Fixed Hyperparameters

Configuration:
- Model: Llama-3.1-8B (BF16 precision)
- Method: CITA (Calibrated Instruction Tuning with Alignment)
- Loss: L_CITA = L_DPO + λ_KL × L_KL (combines DPO + KL regularization)
- Dataset: PKU-SafeRLHF (10,813 samples, clear safety contrast)
- Training: Fixed hyperparameters from best Optuna trial
- Precision: BF16 + Flash Attention 2
- LoRA: r=16, alpha=16
- Warmup: Uses warmup_ratio=0.253 (FIXED: iter7 explosion bug)
- Gradient Clipping: max_grad_norm=1.0 (iter8 - prevents explosion)
- Expected time: ~120 minutes on A100-40GB (1.0 epoch)

Best Hyperparameters (from Optuna Trial 2, tuned for warmup_ratio=25.3%):
- Learning rate: 1.185448e-05
- Beta: 0.1133
- Lambda KL: 0.001010
- Weight decay: 0.008849
- Warmup ratio: 0.253 (epoch-agnostic, enables hyperparameter transfer)

Usage:
    # SANITY: 0.3 epochs (steps auto-calculated, ~36 minutes, ~$0.90)
    # Stack on DPO (REQUIRED):
    python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode sanity \
        --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct

    # FULL: 1.0 epoch (steps auto-calculated, ~120 minutes, ~$3.00)
    # Stack on DPO (REQUIRED):
python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full \
    --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct

Outputs:
    - Model checkpoints: ./outputs/CITA_Baseline/checkpoint-*/
    - LoRA adapters: ./outputs/CITA_Baseline/lora_model_CITA_Baseline/
    - TensorBoard logs: ./tensorboard_logs/CITA_Baseline_<timestamp>/
    - Training log: ./logs/CITA_Baseline_training_<timestamp>.log
    - HuggingFace: kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct
"""

import sys
from pathlib import Path
import os
import argparse
from datetime import datetime

# ===== FIX CUDA OOM: Enable expandable segments for memory fragmentation =====
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

# ===== FIX torch.compile() CUDAGraph bug: Disable CUDAGraphs for dynamic shapes =====
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

# Add utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# ===================================================================
# Import Shared Utilities
# ===================================================================
from model_utils import (
    load_hf_token,
    load_model_bf16,
    setup_lora,
    apply_torch_compile,
    get_test_prompts,
    get_model_repo_name,
    get_latest_checkpoint,
    is_training_complete,
    log_gpu_memory_start,
    log_gpu_memory_end
)
from data_prep.loader_pku import load_pku_combined_clear_contrast
from data_prep.formatters import format_pku_for_cita_Instruct, format_pku_for_cita_NoInstruct
from push_automation import PushAutomation
from logging_utils import setup_training_logger, restore_logging

# ===================================================================
# INSTRUCTION MODE TOGGLE
# ===================================================================
USE_INSTRUCTION = False  # False: CITA_NoInstruct, True: CITA_Instruct

RUN_NAME = "CITA_Instruct" if USE_INSTRUCTION else "CITA_NoInstruct"

# ===================================================================
# Advanced Logging Setup (Tee System)
# ===================================================================
# Setup logging to capture ALL terminal output (stdout + stderr)
log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
    run_name=RUN_NAME,
    project_root=project_root
)

# ===================================================================
# HuggingFace Authentication
# ===================================================================
HF_TOKEN = load_hf_token(project_root)

# Get HuggingFace repository name
HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

print(f"📦 Model will be pushed to: {HF_REPO}")
print("="*80 + "\n")

# ===================================================================
# Main Training Function
# ===================================================================

def train_cita_baseline(num_epochs=1.0, output_dir=None, base_model=None, force_skip=False):
    """
    Train CITA baseline with epoch-based training

    Args:
        num_epochs: Number of training epochs (default: 1.0 for full, 0.1 for sanity)
        output_dir: Directory to save checkpoints
        base_model: HuggingFace model ID to load LoRA adapters from (DPO model)
        force_skip: Skip training and use HuggingFace model directly

    Returns:
        trainer: CITATrainer instance
        training_skipped: Whether training was skipped
    """

    # ===== BEST HYPERPARAMETERS FROM OPTUNA =====
    # Trial 5: 1354 steps, eval_loss=0.2791, margin=6.95, accuracy=89.5% (BEST)
    LAMBDA_KL = 0.000520
    LEARNING_RATE = 6.827978e-06
    BETA = 0.1191
    WEIGHT_DECAY = 0.0091
    WARMUP_RATIO = 0.0749  # Auto-scales: SANITY=101 steps, FULL=101 steps

    # Set output_dir dynamically based on RUN_NAME if not provided
    if output_dir is None:
        output_dir = f"./outputs/{RUN_NAME}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard directory (unique per run)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_run_dir = project_root / "tensorboard_logs" / f"{RUN_NAME}_{timestamp}"
    tensorboard_run_dir.mkdir(parents=True, exist_ok=True)

    # ===== CHECK FOR EXISTING CHECKPOINT =====
    print("="*80)
    print("🔍 Checking for existing checkpoints...")
    print("="*80 + "\n")

    latest_checkpoint = get_latest_checkpoint(output_dir)
    training_skipped = False

    # Note: Completion check will happen after loading dataset (need total_steps)
    if latest_checkpoint:
        print(f"📂 Found checkpoint: {latest_checkpoint}")
        print(f"   Will check completion status after loading dataset...\n")
    else:
        print(f"🆕 No checkpoint found")
        print(f"   Will start fresh training (even if HF repo exists)...")

    if force_skip:
        print(f"\n🔄 Force skip enabled: Will skip training and use HuggingFace model")
        training_skipped = True

    print()

    # ===== SKIP TRAINING IF REQUESTED =====
    if not training_skipped:

        # ===== LOAD BASE MODEL =====
        print("Loading model...")
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
            print("✅ LoRA adapters merged (ready for new training stage)")

        # ===== APPLY LORA ADAPTERS =====
        print("\nApplying LoRA adapters...")
        model = setup_lora(
            model,
            r=16,
            lora_alpha=16,
            use_gradient_checkpointing=True
        )

        # Cast to BF16 AFTER applying new LoRA adapters (fixes FlashAttention dtype error)
        model = model.to(torch.bfloat16)
        print("✅ Model cast to BF16 (all params including LoRA)")

        # ===== TORCH.COMPILE() OPTIMIZATION =====
        # DISABLED: Causes KeyError: 'hidden_states' during gradient checkpointing
        # torch.compile() + gradient_checkpointing has compatibility issues with CITA (DPO-based)
        # Error occurs in transformers/utils/generic.py wrapped_forward output collection
        # Trade-off: ~10% slower training vs no crash
        print("\n⚠️  torch.compile() disabled (prevents gradient checkpointing bug for CITA)")
        # model = apply_torch_compile(model)

        # ===== LOAD DATASET =====
        print("\nLoading PKU-SafeRLHF dataset (combined train+test clear contrast)...")

        # Load combined dataset (12,035 samples) and split 90/10
        dataset_split = load_pku_combined_clear_contrast(val_split=0.1)

        # Format for CITA (conditional: WITH or WITHOUT instructions based on USE_INSTRUCTION toggle)
        formatter = format_pku_for_cita_Instruct if USE_INSTRUCTION else format_pku_for_cita_NoInstruct
        format_desc = f"Formatting PKU for CITA ({'WITH' if USE_INSTRUCTION else 'NO'} instructions)"

        train_dataset = dataset_split['train'].map(
            formatter,
            remove_columns=dataset_split['train'].column_names,
            desc=format_desc
        )

        val_dataset = dataset_split['test'].map(
            formatter,
            remove_columns=dataset_split['test'].column_names,
            desc=f"Formatting PKU validation for CITA ({'WITH' if USE_INSTRUCTION else 'NO'} instructions)"
        )

        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples: {len(val_dataset):,}")

        # ===== CALCULATE TRAINING STEPS (for checkpoint intervals) =====
        effective_batch_size = 1 * 8  # per_device=1, grad_accum=8
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_steps = int(steps_per_epoch * num_epochs)

        # Checkpoint interval: 20% of training (dynamic scaling)
        checkpoint_interval = int(total_steps * 0.2)  # SANITY=80, FULL=270
        # REMOVED iter9 experiment: Fixed eval_steps=80 (was too frequent for production)

        print(f"\n📊 Training Configuration:")
        print(f"   Dataset size: {len(train_dataset):,} samples")
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Total steps: {total_steps:,} ({num_epochs} epochs)")
        print(f"   Checkpoint interval: {checkpoint_interval} steps (20% of training)")

        # ===== SCALE VALIDATION SET FOR SANITY MODE =====
        if num_epochs < 1.0:
            val_size_scaled = int(len(val_dataset) * num_epochs)
            val_dataset = val_dataset.select(range(val_size_scaled))
            print(f"\n⚡ SANITY mode: Scaled validation set to {num_epochs:.1f}x ({len(val_dataset):,} samples)")

        # ===== CHECK TRAINING COMPLETION (now that we have total_steps) =====
        if latest_checkpoint and is_training_complete(latest_checkpoint, total_steps):
            print(f"\n✅ Training already completed at: {latest_checkpoint}")
            print(f"   Total steps: {total_steps} ({num_epochs} epochs)")
            training_skipped = True

        print(f"\n📊 TensorBoard logs: {tensorboard_run_dir}\n")

        # ===== CREATE TRAINING ARGS =====
        training_args = DPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=num_epochs,  # ← CHANGED: Epoch-based training
            warmup_ratio=WARMUP_RATIO,  # ← FIXED: 25.3% warmup (auto-scales with total steps)
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="cosine",
            seed=3407,
            bf16=True,
            gradient_checkpointing=True,
            max_grad_norm=1.0,  # Phase 1: Gradient clipping to prevent explosion
            save_steps=checkpoint_interval,  # ← CHANGED: Dynamic interval (20% of training)
            save_total_limit=5,
            report_to="tensorboard",
            logging_dir=str(tensorboard_run_dir),
            logging_first_step=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            # Validation
            eval_strategy="steps",
            eval_steps=checkpoint_interval,  # ← CHANGED: Dynamic interval (aligned with checkpoints)
            per_device_eval_batch_size=1,
            # CITA-specific parameters (DPOConfig compatible)
            beta=BETA,
            max_length=2048,
            max_prompt_length=1024,
        )

        # ===== CREATE CITA TRAINER =====
        print("\nInitializing CITATrainer...")
        from cita_trainer import CITATrainer
        from monitoring_callback import TrainingSummaryCallback

        summary_callback = TrainingSummaryCallback(
            check_every_n_steps=50,
            training_method="cita"
        )

        trainer = CITATrainer(
            model=model,
            tokenizer=tokenizer,  # CITA uses 'tokenizer' not 'processing_class'
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            lambda_kl=LAMBDA_KL,  # CITA-specific: KL regularization weight
            callbacks=[summary_callback]
        )

        # ===== VERIFY GRADIENT CLIPPING IS CONFIGURED =====
        max_grad_norm = getattr(trainer.args, 'max_grad_norm', None)
        if max_grad_norm is not None and max_grad_norm > 0:
            print(f"\n✅ Gradient clipping ENABLED: max_grad_norm={max_grad_norm}")
            print(f"   Gradients will be clipped to prevent explosion")
            print(f"   Monitor 'cita/grad_norm_clipped' and 'cita/clipping_active' in logs\n")
        else:
            print(f"\n⚠️  WARNING: Gradient clipping NOT configured (max_grad_norm={max_grad_norm})")
            print(f"   Training may be unstable without clipping!\n")

        # ===== SHOW GPU MEMORY =====
        start_gpu_memory = log_gpu_memory_start()
    else:
        # Training skipped - initialize empty vars
        trainer = None

    # ===== TRAIN =====
    if not training_skipped:
        print("\n" + "="*80)
        print("🏋️  Training CITA Baseline...")
        print("="*80 + "\n")

        trainer.train(resume_from_checkpoint=latest_checkpoint)

        # ===== SAVE LORA ADAPTERS =====
        print("\n" + "="*80)
        print("💾 Saving LoRA adapters...")
        print("="*80 + "\n")

        lora_output_dir = output_dir / f"lora_model_{RUN_NAME}"
        lora_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(lora_output_dir))
        tokenizer.save_pretrained(str(lora_output_dir))

        print(f"✅ LoRA adapters saved to: {lora_output_dir}\n")

        # ===== SHOW FINAL MEMORY =====
        log_gpu_memory_end(start_gpu_memory)

    # ===== EXTRACT FINAL METRICS =====
    final_margin = 'N/A'
    if training_skipped:
        # Training was skipped - load from existing checkpoint
        if latest_checkpoint:
            try:
                import json
                trainer_state_path = Path(latest_checkpoint) / "trainer_state.json"
                if trainer_state_path.exists():
                    with open(trainer_state_path, 'r') as f:
                        state = json.load(f)
                    # Find last eval_rewards/margins
                    for log_entry in reversed(state['log_history']):
                        if 'eval_rewards/margins' in log_entry:
                            final_margin = log_entry['eval_rewards/margins']
                            break
                    print(f"📊 Loaded final margin from checkpoint: {final_margin:.4f}")
            except Exception as e:
                print(f"⚠️  Could not load metrics from checkpoint: {e}")
    else:
        # Training completed - extract from trainer state
        if hasattr(trainer.state, 'log_history'):
            for log_entry in reversed(trainer.state.log_history):
                if 'eval_rewards/margins' in log_entry:
                    final_margin = log_entry['eval_rewards/margins']
                    break

    # ===== SAVE TRAINING CONFIG =====
    training_config = {
        "method": "CITA",
        "num_epochs": num_epochs,
        "lambda_kl": LAMBDA_KL,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "max_seq_length": 2048,
        "max_prompt_length": 1024,
        "final_margin": final_margin if final_margin != 'N/A' else None,
    }

    # Note: Config will be saved by PushAutomation.prepare_baseline_push()
    # with correct name: {RUN_NAME}_config.json

    return trainer, training_skipped, training_config


# ===================================================================
# Main Execution
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CITA Baseline Training - Fixed hyperparameters from Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["sanity", "full"],
        default="sanity",
        help="Training mode: 'sanity' (0.1 epochs) or 'full' (1.0 epochs)"
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Number of training epochs (overrides --mode)"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="HuggingFace model ID to load LoRA adapters from (e.g., DPO model)"
    )

    args = parser.parse_args()

    # Determine configuration
    if args.epochs is not None:
        num_epochs = args.epochs
        print(f"✅ Custom configuration: {num_epochs} epochs")
    elif args.mode == "sanity":
        num_epochs = 0.3  # 30% of data (~3,249 samples, ~405 steps, ~36 min)
        print(f"✅ Sanity check mode: {num_epochs} epochs (~36 minutes)")
    else:
        num_epochs = 1.0  # Full epoch (~10,831 samples, ~1,353 steps, ~120 min)
        print(f"✅ Full training mode: {num_epochs} epochs (~120 minutes)")

    print()
    print("="*80)
    print("🔄 Training Mode Selection")
    print("="*80)

    # Check if HuggingFace model exists
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    try:
        api.model_info(HF_REPO)
        hf_model_exists = True
        print(f"✅ Found existing model on HuggingFace: {HF_REPO}")
    except Exception:
        hf_model_exists = False
        print(f"❌ No existing model on HuggingFace: {HF_REPO}")
        print(f"   This will be the first training run")

    print("="*80)
    print(f"Training will take approximately: {'~36 minutes' if num_epochs == 0.3 else '~120 minutes'}\n")

    # Show options
    print("Options:")
    print("  1) Inference only from HF_repo")
    print("  2) Train and push to HuggingFace")
    print("="*80)

    mode_choice = input("Enter choice (1 or 2): ").strip()
    print("="*80 + "\n")

    # Validate option 1 requires HF repo
    if mode_choice == "1":
        if not hf_model_exists:
            print("❌ Error: Option 1 requires existing HuggingFace model")
            print("   Please choose option 2 to train first")
            sys.exit(1)
        force_skip = True
        load_from_hf = True
        print("✅ Inference mode selected")
        print("   Will use existing HuggingFace model\n")
    elif mode_choice == "2":
        force_skip = False
        load_from_hf = False
        if hf_model_exists:
            print("✅ Training mode selected")
            print("   Will train and compare with HuggingFace model\n")
        else:
            print("✅ Training mode selected")
            print("   Will train and push to HuggingFace (first time)\n")
    else:
        print(f"❌ Invalid choice: {mode_choice}")
        sys.exit(1)

    # Show configuration
    print("="*80)
    print(f"🚀 Starting {RUN_NAME} Training")
    print("="*80)
    print("Configuration:")
    print("  - Model: Llama-3.1-8B (BF16)")
    print("  - Method: CITA (Calibrated Instruction Tuning with Alignment)")
    print("  - Loss: L_CITA = L_DPO + λ_KL × L_KL")
    print(f"  - Training epochs: {num_epochs}")
    print("  - Batch size: 1 (per device)")
    print("  - Gradient accumulation: 8 (effective batch=8)")
    print("  - Learning rate: 1.185e-05 (Optuna Trial 2)")
    print("  - Beta: 0.1133 (Optuna Trial 2)")
    print("  - Lambda KL: 0.001010 (Optuna Trial 2)")
    print("  - Weight decay: 0.00885 (Optuna Trial 2)")
    print("  - Warmup ratio: 25.3% (Optuna Trial 2 - auto-scales with training length)")
    print("  - LR scheduler: cosine")
    print("  - Optimizer: adamw_torch")
    print("  - Gradient clipping: max_grad_norm=1.0 (iter8 - prevents explosion)")
    print("  - Precision: BF16 + Flash Attention 2")
    print("="*80 + "\n\n")

    # Train
    try:
        trainer, training_skipped, training_config = train_cita_baseline(
            num_epochs=num_epochs,
            base_model=args.base_model,
            force_skip=force_skip
        )

        # ===== PUSH TO HUGGINGFACE =====
        # Use unified push utility (extracts metrics, saves config, pushes to HF/GitHub)
        # Remove final_margin (will be re-extracted from checkpoint by utility)
        training_config.pop('final_margin', None)

        PushAutomation.prepare_baseline_push(
            method="CITA",
            output_dir=f"outputs/{RUN_NAME}",
            training_config=training_config,
            training_skipped=training_skipped,
            hf_token=HF_TOKEN,
            hf_repo=HF_REPO,
            run_name=RUN_NAME,
            metric_names=["eval_rewards/margins", "rewards/margins"],
            metric_mode="max",
            project_root=project_root
        )

        print("\n" + "="*80)
        print("🏁 CITA Baseline Training Complete!")
        print("="*80)
        print(f"📝 Log file: {log_filename}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"📝 Complete log saved: {log_filename}")
