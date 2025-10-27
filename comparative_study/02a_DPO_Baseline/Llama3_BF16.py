"""
DPO Baseline Training Script (BF16 precision)
Standard Direct Preference Optimization without PBT

Configuration:
- Model: Llama-3.1-8B (BF16 precision)
- Method: Standard DPO (Rafailov et al. 2023)
- Loss: L_DPO only (no L_SFT or L_KL)
- Dataset: Vaibhaav (50,001 samples, 90/10 split)
- Training: Epoch-based, fixed hyperparameters (no PBT)
- Precision: BF16 + Flash Attention 2
- LoRA: r=16, alpha=16
- Expected time: ~130 minutes on A100-40GB (1.0 epoch)

Usage:
    # Sanity check (0.1 epoch, ~17 minutes)
    python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode sanity

    # Full training (1.0 epoch, ~130 minutes)
    python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full

Outputs:
    - Model checkpoint: ./outputs/DPO_Baseline/checkpoint-<step>/
    - LoRA adapters: ./outputs/DPO_Baseline/lora_model_DPO_Baseline/
    - TensorBoard logs: ./tensorboard_logs/DPO_Baseline_<timestamp>/
    - Training log: ./logs/DPO_Baseline_training_<timestamp>.log
"""

import sys
from pathlib import Path
import os
import argparse
from datetime import datetime

# ===== FIX CUDA OOM: Enable expandable segments for memory fragmentation =====
# DPO requires both trainable model + reference model, causing fragmentation
# MUST be set BEFORE importing torch/transformers
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

# ===== FIX torch.compile() CUDAGraph bug: Disable CUDAGraphs for dynamic shapes =====
# Fixes: "Expected curr_block->next == nullptr" error during eval with torch.compile()
# Warning showed 51 distinct input sizes → CUDAGraph memory allocator bug
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

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
    log_gpu_memory_start,
    log_gpu_memory_end
)
from push_automation import PushAutomation
from logging_utils import setup_training_logger, restore_logging

# ===================================================================
# Advanced Logging Setup (Tee System)
# ===================================================================
# Setup logging to capture ALL terminal output (stdout + stderr)
log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
    run_name="DPO_Baseline",
    project_root=project_root
)

# ===================================================================
# HuggingFace Authentication
# ===================================================================
HF_TOKEN = load_hf_token(project_root)

# Get HuggingFace repository name
RUN_NAME = "DPO_Baseline"
HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

print(f"📦 Model will be pushed to: {HF_REPO}")
print("="*80 + "\n")


# ===================================================================
# Main Training Function
# ===================================================================

def train_dpo_baseline(num_epochs=1.0, output_dir="./outputs/DPO_Baseline", base_model=None, force_skip=False):
    """
    Train DPO baseline with fixed hyperparameters

    Args:
        num_epochs: Number of training epochs (default: 1.0 for full, 0.1 for sanity)
        output_dir: Output directory for checkpoints
        base_model: HuggingFace model ID to load LoRA adapters from (for stacking)
        force_skip: If True, skip training and only run inference (user selected option 1)

    Returns:
        trainer: Trained DPOTrainer instance
    """
    print("\n" + "="*80)
    print(f"🚀 Starting DPO Baseline Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Model: Llama-3.1-8B (BF16)")
    print(f"  - Method: Standard DPO (Rafailov 2023)")
    print(f"  - Loss: L_DPO only")
    print(f"  - Training epochs: {num_epochs}")
    print(f"  - Dataset: Vaibhaav (50K samples, 90/10 split)")
    print(f"  - Batch size: 1 (per device, reduced for DPO ref model)")
    print(f"  - Gradient accumulation: 8 (effective batch=8)")
    print(f"  - Learning rate: 1e-5 (Meta's Llama 3 DPO setting)")
    print(f"  - Beta: 0.1 (Meta's Llama 3 DPO setting)")
    print(f"  - Warmup ratio: 0.03 (3% of training)")
    print(f"  - Eval frequency: Every 20% (5 checkpoints per epoch)")
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

        # Check if training is complete (simplified: check if final checkpoint exists)
        if latest_checkpoint:
            print(f"✅ Training checkpoint found at: {latest_checkpoint}")
            print(f"   Epochs: {num_epochs}")
            print(f"   Skipping training, will load from local checkpoint for inference...\n")
            training_skipped = True
            load_from_hf = False  # Option 2 with local checkpoint: Load from local
        else:
            print(f"🆕 No checkpoint found")
            print(f"   Will start fresh training (even if HF repo exists)...\n")
            latest_checkpoint = None

    # ===== LOAD MODEL & TOKENIZER =====
    # Skip model loading if training already complete (will load local checkpoint for inference later)
    if not training_skipped:
        print("Loading model...")
        model, tokenizer = load_model_bf16(
            model_id="meta-llama/Llama-3.1-8B",
            max_seq_length=2048,  # Match SFT baseline for fair comparison
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

            # Clear PEFT config to avoid "Already found peft_config" warning
            # merge_and_unload() leaves peft_config and _hf_peft_config_loaded attributes
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

        # Cast to BF16 AFTER applying new LoRA adapters (to ensure all params including new LoRA weights are BF16)
        model = model.to(torch.bfloat16)
        print("✅ Model cast to BF16 (all params including LoRA)")

        # ===== TORCH.COMPILE() OPTIMIZATION =====
        # DISABLED: Causing AttributeError: 'float' object has no attribute 'meta'
        # print("\nApplying torch.compile()...")
        # model = apply_torch_compile(model)

        # ===== LOAD DATASET =====
        print("\nLoading Vaibhaav/alignment-instructions dataset...")
        from data_prep.loader_vaibhaav import load_vaibhaav_alignment, format_vaibhaav_for_dpo

        # Load raw dataset (50,001 samples)
        dataset_raw = load_vaibhaav_alignment(split="train")

        # Format for DPO (NO instruction - baseline)
        dataset_formatted = dataset_raw.map(
            format_vaibhaav_for_dpo,
            remove_columns=dataset_raw.column_names,
            desc="Formatting Vaibhaav for DPO (NO instruction)"
        )

        # Split train/val (90/10)
        dataset_split = dataset_formatted.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split["train"]
        val_dataset = dataset_split["test"]

        # Scale validation set for SANITY mode (faster evaluation)
        if num_epochs < 1.0:
            val_size_scaled = int(len(val_dataset) * num_epochs)
            val_dataset = val_dataset.select(range(val_size_scaled))
            print(f"⚡ SANITY mode: Scaled validation set to {num_epochs:.1f}x ({len(val_dataset):,} samples)")

        # Calculate steps for percentage-based checkpointing
        steps_per_epoch = len(train_dataset) // 8  # effective_batch_size=8
        total_steps = int(steps_per_epoch * num_epochs)  # Scale to actual training duration
        checkpoint_interval = int(total_steps * 0.2)  # Save/eval every 20% of ACTUAL training

        print(f"✅ Dataset loaded: {len(train_dataset):,} train / {len(val_dataset):,} val")
        print(f"   Steps per epoch: {steps_per_epoch:,} (batch_size=8)")
        print(f"   Total training steps: {total_steps:,} ({num_epochs} epoch)")
        print(f"   Checkpoint interval: {checkpoint_interval:,} steps (20% of training)")
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
        # Match hyperparameters from SFT baseline for fair comparison
        training_args = DPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=1,  # ✅ FIXED: Reduced from 2 to avoid OOM (DPO needs ref model copy)
            gradient_accumulation_steps=8,  # ✅ FIXED: Doubled to maintain effective batch=8
            num_train_epochs=num_epochs,  # Epoch-based training
            warmup_ratio=0.03,  # 3% of training (auto-scales)
            learning_rate=1e-5,  # ✅ FIXED: Meta's official Llama 3 DPO setting (was 2e-4, 20× too high)
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # ✅ FIXED: Cosine for smoother convergence
            seed=3407,
            bf16=True,  # BF16 precision
            gradient_checkpointing=True,
            save_strategy="steps",
            save_steps=checkpoint_interval,  # 20% of epoch
            save_total_limit=5,
            report_to="tensorboard",
            logging_dir=str(tensorboard_run_dir),
            logging_first_step=True,
            dataloader_num_workers=2,  # Parallel data loading
            dataloader_pin_memory=True,  # Faster CPU→GPU transfer
            # ✅ Evaluation every 20% of epoch
            eval_strategy="steps",
            eval_steps=checkpoint_interval,  # 20% of epoch (5 evals per full epoch)
            per_device_eval_batch_size=1,  # ✅ FIXED: Match training batch size
            # ✅ DPO-specific parameters (TRL 0.22.2+)
            beta=0.1,  # Contrastive temperature (standard DPO value)
            max_length=2048,  # Match max_seq_length
            max_prompt_length=1024,  # Half of max_length
        )

        # ===== CREATE DPO TRAINER =====
        print("\nInitializing DPOTrainer...")
        # TRL 0.22.2: DPO params go in DPOConfig, only base params in DPOTrainer
        # ===== TRAINING SUMMARY CALLBACK =====
        from monitoring_callback import TrainingSummaryCallback

        summary_callback = TrainingSummaryCallback(
            check_every_n_steps=50,
            training_method="dpo"
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # DPOTrainer creates reference model automatically
            processing_class=tokenizer,  # TRL 0.22.2 parameter name
            args=training_args,
            train_dataset=train_dataset,
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
        print("🏋️  Training DPO Baseline...")
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
    lora_output_dir = Path(output_dir) / "lora_model_DPO_Baseline"

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
        description="DPO Baseline Training - Standard Direct Preference Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sanity check (0.1 epoch, ~17 minutes)
  python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode sanity

  # Full training (1.0 epoch, ~130 minutes)
  python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full

  # Custom epochs
  python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --epochs 0.5
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
        num_epochs = 0.1  # 10% of data (~4,500 samples, ~17 min)
        print(f"✅ Sanity check mode: {num_epochs} epochs (~17 minutes)")
    else:
        num_epochs = 1.0  # Full epoch (~45,000 samples, ~130 min)
        print(f"✅ Full training mode: {num_epochs} epochs (~130 minutes)")

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
                print(f"   Previous performance: margin={previous_metric:.4f}")
        else:
            print(f"❌ No existing model on HuggingFace: {HF_REPO}")
            print(f"   This will be the first training run")
    except Exception as e:
        print(f"⚠️  Could not check HuggingFace: {type(e).__name__}")

    print(f"{'='*80}")
    print(f"Training will take approximately: {'~17 minutes' if num_epochs == 0.1 else '~130 minutes'}")
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
        trainer, training_skipped = train_dpo_baseline(num_epochs=num_epochs, base_model=args.base_model, force_skip=force_skip)
        print(f"\n🏁 DPO Baseline Training Complete!")
        print(f"📝 Log file: {log_filename}")

        # ===================================================================
        # Automated Push to HuggingFace & GitHub + Auto-Shutdown
        # ===================================================================

        # Use unified push utility (extracts metrics, saves config, pushes to HF/GitHub)
        training_config = {
            "method": "DPO",
            "num_epochs": num_epochs,
            "learning_rate": 1e-5,  # Meta's official Llama 3 DPO setting
            "warmup_ratio": 0.03,  # 3% of training (auto-scales)
            "optimizer": "adamw_torch",
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",  # Cosine for smoother convergence
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
            "max_prompt_length": 1024,  # DPO-specific (prompts truncated to fit chosen+rejected)
            "beta": 0.1,  # Meta's official Llama 3 DPO setting
        }

        PushAutomation.prepare_baseline_push(
            method="DPO",
            output_dir="outputs/DPO_Baseline",
            training_config=training_config,
            training_skipped=training_skipped,
            hf_token=HF_TOKEN,
            hf_repo=HF_REPO,
            run_name=RUN_NAME,
            metric_names=["eval_rewards/margins", "rewards/margins"],
            metric_mode="max",
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
