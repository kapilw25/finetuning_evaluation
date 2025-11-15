"""
SFT Baseline Training Script (BF16 precision)
Standard Supervised Fine-Tuning without PBT

Configuration:
- Model: Llama-3.1-8B (BF16 precision)
- Method: Standard SFT (supervised learning on chosen responses only)
- Loss: L_SFT only (no L_DPO or L_KL)
- Dataset: PKU-SafeRLHF (10,813 samples, chosen responses only)
- Training: Fixed hyperparameters (no PBT)
- Precision: BF16 + Flash Attention 2
- LoRA: r=16, alpha=16
- Warmup: Uses warmup_ratio (epoch-agnostic, hyperparameters transfer across training lengths)
- Expected time: ~43 minutes on A100-40GB (1.0 epoch)
- Expected cost: ~$1.08 (43 min × $1.5/hr)

Usage:
    # SANITY: 0.3 epochs (steps auto-calculated, ~13 minutes, ~$0.32)
    python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity

    # FULL: 1.0 epoch (steps auto-calculated, ~43 minutes, ~$1.08)
    python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full

Outputs:
    - Model checkpoints: ./outputs/SFT_Baseline/checkpoint-*/
    - LoRA adapters: ./outputs/SFT_Baseline/lora_model_SFT_Baseline/
    - TensorBoard logs: ./tensorboard_logs/SFT_Baseline_<timestamp>/
    - Training log: ./logs/SFT_Baseline_training_<timestamp>.log
    - HuggingFace: kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct
"""

import sys
from pathlib import Path
import torch
import os
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

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
from data_prep.formatters import format_pku_for_sft, format_pku_for_sft_Instruct
from push_automation import PushAutomation
from logging_utils import setup_training_logger, restore_logging

# ===================================================================
# INSTRUCTION MODE TOGGLE
# ===================================================================
USE_INSTRUCTION = False  # False: SFT_NoInstruct, True: SFT_Instruct

RUN_NAME = "SFT_Instruct" if USE_INSTRUCTION else "SFT_NoInstruct"

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

def train_sft_baseline(num_epochs=1.0, output_dir=None, base_model=None, force_skip=False):
    """
    Train SFT baseline with epoch-based training

    Args:
        num_epochs: Number of training epochs (default: 1.0 for full, 0.1 for sanity)
        output_dir: Output directory for checkpoints
        base_model: HuggingFace model ID to load LoRA adapters from (for stacking)
        force_skip: If True, skip training and only run inference (user selected option 1)

    Returns:
        trainer: Trained SFTTrainer instance
    """
    # Set output_dir dynamically based on RUN_NAME if not provided
    if output_dir is None:
        output_dir = f"./outputs/{RUN_NAME}"

    print("\n" + "="*80)
    print(f"🚀 Starting {RUN_NAME} Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Model: Llama-3.1-8B (BF16)")
    print(f"  - Method: Standard SFT (supervised learning)")
    print(f"  - Loss: L_SFT only")
    print(f"  - Training epochs: {num_epochs}")
    print(f"  - Batch size: 2 (per device)")
    print(f"  - Gradient accumulation: 4 (effective batch=8)")
    print(f"  - Learning rate: 2e-4 (QLoRA recommendation)")
    print(f"  - Warmup steps: 100")
    print(f"  - LR scheduler: cosine")
    print(f"  - Optimizer: adamw_torch")
    print(f"  - Precision: BF16 + Flash Attention 2")
    print("="*80 + "\n")

    # ===== CHECKPOINT DETECTION (BEFORE LOADING MODEL) =====
    print("\n" + "="*80)
    print("🔍 Checking for existing checkpoints...")
    print("="*80 + "\n")

    training_skipped = False
    latest_checkpoint = None
    load_from_hf = False  # NEW: Track if we should load from HF (option 1 only)

    # Force skip if user selected inference-only mode
    if force_skip:
        print("🚫 User selected inference-only mode")
        print("   Skipping training, will load model from HuggingFace for inference...\n")
        training_skipped = True
        load_from_hf = True  # Option 1: Load from HF
    else:
        # Priority 1: Check local checkpoints (CHANGED ORDER - local first, HF second)
        # This allows retraining even if HF repo exists (user chose option 2)
        print("1️⃣ Checking local checkpoints...")
        latest_checkpoint = get_latest_checkpoint(output_dir)

        # Note: Completion check will happen after loading dataset (need total_steps)
        if latest_checkpoint:
            print(f"📂 Found checkpoint: {latest_checkpoint}")
            print(f"   Will check completion status after loading dataset...\n")
        else:
            print(f"🆕 No checkpoint found")
            print(f"   Will start fresh training (even if HF repo exists)...\n")
            latest_checkpoint = None

    # ===== LOAD MODEL & TOKENIZER =====
    # Skip model loading if training already complete on HF (will download for inference later)
    if not training_skipped:
        print("Loading model...")
        model, tokenizer = load_model_bf16(
            model_id="meta-llama/Llama-3.1-8B",
            max_seq_length=2048,  # Match DPO baseline for fair comparison
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
            model = model.merge_and_unload()
            print("✅ LoRA adapters merged (ready for new training stage)")

        # ===== APPLY LORA ADAPTERS =====
        print("\nApplying LoRA adapters...")
        model = setup_lora(
            model,
            r=16,
            lora_alpha=16,
            use_gradient_checkpointing=True
        )

        # ===== TORCH.COMPILE() OPTIMIZATION =====
        # DISABLED: Causes OOM during evaluation due to compilation overhead
        # torch.compile() triggers recompilation with different shapes during eval
        # Combined with Accelerate's BF16→FP32 conversion, causes 312MB OOM
        # Trade-off: ~10% slower training vs no crash
        # Note: DPO/CITA keep torch.compile() enabled (they have expandable_segments)
        print("\n⚠️  torch.compile() disabled (prevents eval OOM for SFT)")
        # model = apply_torch_compile(model)

        # ===== LOAD DATASET =====
        print("\nLoading PKU-SafeRLHF dataset (combined train+test clear contrast)...")

        # Load combined dataset (12,035 samples) and split 90/10
        dataset_split = load_pku_combined_clear_contrast(val_split=0.1)

        # Format for SFT (conditional: WITH or WITHOUT instruction)
        formatter = format_pku_for_sft_Instruct if USE_INSTRUCTION else format_pku_for_sft
        train_dataset = dataset_split['train'].map(
            formatter,
            remove_columns=dataset_split['train'].column_names,
            desc=f"Formatting PKU for SFT ({'WITH' if USE_INSTRUCTION else 'NO'} instruction)"
        )

        val_dataset = dataset_split['test'].map(
            formatter,
            remove_columns=dataset_split['test'].column_names,
            desc=f"Formatting PKU validation for SFT ({'WITH' if USE_INSTRUCTION else 'NO'} instruction)"
        )

        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples: {len(val_dataset):,}")

        # ===== CALCULATE TRAINING STEPS (for checkpoint intervals) =====
        effective_batch_size = 2 * 4  # per_device=2, grad_accum=4
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_steps = int(steps_per_epoch * num_epochs)
        checkpoint_interval = int(total_steps * 0.2)  # Save/eval every 20%

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
            print(f"   Skipping training, will load from local checkpoint for inference...\n")
            training_skipped = True
            load_from_hf = False
    else:
        # Training skipped - initialize empty vars (will be loaded for inference if needed)
        model = None
        tokenizer = None
        train_dataset = None
        val_dataset = None

    # Skip trainer setup if training already complete
    if not training_skipped:
        # ===== TENSORBOARD SETUP =====
        tensorboard_base_dir = project_root / "tensorboard_logs"
        tensorboard_base_dir.mkdir(exist_ok=True)

        # Generate timestamp for unique TensorBoard run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_run_dir = tensorboard_base_dir / f"{RUN_NAME}_{timestamp}"

        print(f"📊 TensorBoard logs: {tensorboard_run_dir}")

        # ===== CREATE TRAINING ARGS =====
        # Epoch-based training with dynamic checkpoints
        training_args = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=num_epochs,  # ← CHANGED: Epoch-based training
            warmup_steps=100,  # ← FIXED: 100 steps (from iter4 successful run)
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            bf16=True,
            gradient_checkpointing=True,
            save_strategy="steps",
            save_steps=checkpoint_interval,  # ← CHANGED: Dynamic interval (20% of training)
            save_total_limit=5,
            report_to="tensorboard",
            logging_dir=str(tensorboard_run_dir),
            logging_first_step=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            # SFT-specific parameters
            max_length=2048,
            packing=False,
            # Validation to detect overfitting
            eval_strategy="steps",
            eval_steps=checkpoint_interval,  # ← CHANGED: Dynamic interval (aligned with checkpoints)
            per_device_eval_batch_size=2,
        )

        # ===== CREATE SFT TRAINER =====
        print("\nInitializing SFTTrainer...")
        # TRL 0.22.2 SFTTrainer only accepts: model, processing_class, args, train_dataset, eval_dataset
        # Unlike Unsloth's SFTTrainer, it does NOT accept: max_seq_length, packing, dataset_text_field
        # These are handled automatically via the "messages" field formatting
        # ===== TRAINING SUMMARY CALLBACK =====
        from monitoring_callback import TrainingSummaryCallback

        summary_callback = TrainingSummaryCallback(
            check_every_n_steps=50,
            training_method="sft"
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,  # TRL 0.22.2 parameter name
            args=training_args,
            train_dataset=train_dataset,  # Already formatted with "messages" field
            eval_dataset=val_dataset,
            callbacks=[summary_callback]
        )

        # ===== SHOW GPU MEMORY =====
        start_gpu_memory = log_gpu_memory_start()
    else:
        # Training skipped - initialize empty vars
        trainer = None

    # ===== TRAIN =====
    if not training_skipped:
        print("\n" + "="*80)
        print("🏋️  Training SFT Baseline...")
        print("="*80 + "\n")

        trainer.train(resume_from_checkpoint=latest_checkpoint)

        # Set best_metric for push_automation.py (uses min eval_loss)
        eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
        if eval_losses:
            trainer.state.best_metric = min(eval_losses)

    # ===== SHOW FINAL MEMORY =====
    if not training_skipped:
        log_gpu_memory_end(start_gpu_memory)
        print(f"✅ Training complete!")

    # ===== SAVE LORA ADAPTERS =====
    lora_output_dir = Path(output_dir) / f"lora_model_{RUN_NAME}"

    if not training_skipped:
        # Training just completed - save LoRA adapters
        lora_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving LoRA adapters to: {lora_output_dir}")
        model.save_pretrained(str(lora_output_dir))
        tokenizer.save_pretrained(str(lora_output_dir))
        print(f"✅ LoRA adapters saved!")
    elif load_from_hf:
        # Option 1 (inference-only mode) - download model from HF for inference
        print(f"📥 Downloading model from HuggingFace for inference: {HF_REPO}")
        model, tokenizer = load_model_bf16(
            model_id="meta-llama/Llama-3.1-8B",
            max_seq_length=2048,
            use_flash_attention=True
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, HF_REPO, token=HF_TOKEN)
        print(f"✅ Model downloaded from HuggingFace")
    else:
        # Option 2 with local checkpoint - load from local checkpoint for inference
        print(f"📂 Loading model from local checkpoint for inference: {latest_checkpoint}")
        model, tokenizer = load_model_bf16(
            model_id="meta-llama/Llama-3.1-8B",
            max_seq_length=2048,
            use_flash_attention=True
        )
        from peft import PeftModel
        # Load from local checkpoint, not HF
        checkpoint_lora_path = Path(latest_checkpoint)
        model = PeftModel.from_pretrained(model, str(checkpoint_lora_path))
        print(f"✅ Model loaded from local checkpoint")

    # ===== INFERENCE TEST =====
    print("\n" + "="*80)
    print("🧪 Running inference tests...")
    print("="*80 + "\n")

    from transformers import TextStreamer

    # Prepare model for inference
    model.eval()

    # Ensure model is in bf16 for Flash Attention compatibility
    # (Training with gradient_checkpointing can cause dtype issues)
    model = model.to(torch.bfloat16)

    # Disable gradient checkpointing for inference (not needed, can cause issues)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    test_prompts = get_test_prompts()

    # Test on 3 prompts (1 helpful, 2 harmful)
    test_cases = [
        (test_prompts[0], "Helpful instruction following"),
        (test_prompts[1], "Refusing harmful request (hacking)"),
        (test_prompts[6], "Helpful instruction following (exercise)"),
    ]

    for prompt, description in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {description}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt[:70]}...")

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)

        with torch.no_grad():
            _ = model.generate(
                input_ids,
                streamer=text_streamer,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9,
            )

    print(f"\n{'='*80}")
    print(f"✅ Inference tests completed")
    print(f"{'='*80}\n")

    return trainer, training_skipped


# ===================================================================
# Main Execution
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT Baseline Training - Standard Supervised Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sanity check (0.3 epochs, ~13 minutes)
  python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity

  # Full training (1.0 epoch, ~43 minutes)
  python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full

  # Custom epochs
  python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --epochs 0.5
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["sanity", "full"],
        default="full",
        help="Training mode: 'sanity' (0.1 epoch) or 'full' (1.0 epoch)"
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
        help="HuggingFace model ID to load LoRA adapters from (for stacking SFT→DPO→CITA)"
    )

    args = parser.parse_args()

    # Determine configuration
    if args.epochs is not None:
        num_epochs = args.epochs
        print(f"✅ Custom configuration: {num_epochs} epochs")
    elif args.mode == "sanity":
        num_epochs = 0.3  # 30% of data (~3,249 samples, ~405 steps, ~13 min)
        print(f"✅ Sanity check mode: {num_epochs} epochs (~13 minutes)")
    else:
        num_epochs = 1.0  # Full epoch (~10,831 samples, ~1,353 steps, ~43 min)
        print(f"✅ Full training mode: {num_epochs} epochs (~43 minutes)")

    # ===================================================================
    # Ask about retraining BEFORE starting (check HF repo first)
    # ===================================================================
    print(f"\n{'='*80}")
    print("🔄 Training Mode Selection")
    print(f"{'='*80}")

    # Check if HF repo exists
    hf_model_exists = False
    previous_metric = None
    try:
        from huggingface_hub import repo_exists
        if repo_exists(HF_REPO, token=HF_TOKEN, repo_type="model"):
            hf_model_exists = True
            print(f"✅ Found existing model on HuggingFace: {HF_REPO}")

            # Try to get previous metric
            from push_automation import PushAutomation
            pusher_temp = PushAutomation(hf_token=HF_TOKEN, project_root=project_root)
            previous_metric = pusher_temp._get_previous_best_margin(HF_REPO)

            if previous_metric:
                print(f"   Previous performance: loss={previous_metric:.4f}")
        else:
            print(f"❌ No existing model on HuggingFace: {HF_REPO}")
            print(f"   This will be the first training run")
    except Exception as e:
        print(f"⚠️  Could not check HuggingFace: {type(e).__name__}")

    print(f"{'='*80}")
    print(f"Training will take approximately: {'~13 minutes' if num_epochs == 0.3 else '~43 minutes'}")
    print(f"\nOptions:")
    if hf_model_exists:
        print(f"  1) Inference only from HF_repo (use existing HF model)")
        print(f"  2) Retrain and replace HF model (only if performance improves)")
    else:
        print(f"  1) Inference only from HF_repo")
        print(f"  2) Train and push to HuggingFace")
    print(f"{'='*80}")

    mode_choice = input("Enter choice (1 or 2): ").strip()
    print(f"{'='*80}\n")

    force_skip = False  # Flag to override checkpoint detection
    if mode_choice == "1":
        # Option 1: Inference-only mode (requires HF repo)
        if not hf_model_exists:
            print("❌ Error: Option 1 requires existing HuggingFace model")
            print("   HuggingFace repo does not exist yet")
            print("   Please choose option 2 to train and create the model first")
            sys.exit(1)
        print("✅ Inference-only mode selected")
        print("   Will load model from HuggingFace for inference tests")
        force_skip = True  # Will skip training regardless of checkpoint status
    elif mode_choice == "2":
        # Option 2: Training mode (comparison happens in push_automation.py)
        print("✅ Training mode selected")
        if hf_model_exists:
            print("   Will compare local vs HF metrics and push ONLY if performance improves")
        else:
            print("   Will train and push to HuggingFace (first time)")
        force_skip = False
    else:
        print("⚠️  Invalid choice, defaulting to training mode")
        force_skip = False

    # Run training
    try:
        trainer, training_skipped = train_sft_baseline(num_epochs=num_epochs, base_model=args.base_model, force_skip=force_skip)
        print(f"\n🏁 SFT Baseline Training Complete!")
        print(f"📝 Log file: {log_filename}")

        # ===================================================================
        # Automated Push to HuggingFace & GitHub + Auto-Shutdown
        # ===================================================================

        # Use unified push utility (extracts metrics, saves config, pushes to HF/GitHub)
        training_config = {
            "method": "SFT",
            "num_epochs": num_epochs,
            "learning_rate": 2e-4,  # QLoRA recommendation for small models
            "warmup_steps": 100,  # Fixed (from iter4 successful run)
            "optimizer": "adamw_torch",
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",  # Cosine for smoother convergence
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
        }

        PushAutomation.prepare_baseline_push(
            method="SFT",
            output_dir=f"outputs/{RUN_NAME}",
            training_config=training_config,
            training_skipped=training_skipped,
            hf_token=HF_TOKEN,
            hf_repo=HF_REPO,
            run_name=RUN_NAME,
            metric_names=["eval_loss"],
            metric_mode="min",
            project_root=project_root
        )

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Restore original stdout/stderr and close log file
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"📝 Complete log file saved: {log_filename}")
