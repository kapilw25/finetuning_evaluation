"""
DPO Baseline Training Script (BF16 precision)
Standard Direct Preference Optimization without PBT

Configuration:
- Model: Llama-3.1-8B (BF16 precision)
- Method: Standard DPO (Rafailov et al. 2023)
- Loss: L_DPO only (no L_SFT or L_KL)
- Dataset: PKU-SafeRLHF (10,813 samples, clear safety contrast)
- Training: Fixed hyperparameters (no PBT)
- Precision: BF16 + Flash Attention 2
- LoRA: r=16, alpha=16
- Expected time: ~40 minutes on A100-40GB (1000 steps)
- Expected cost: ~$1.00 (40 min × $1.5/hr)

Usage:
    # Sanity check (200 steps, ~8 minutes)
    python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode sanity

    # Full training (1000 steps, ~40 minutes)
    python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full

Outputs:
    - Model checkpoint: ./outputs/DPO_Baseline/checkpoint-1000/
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
    load_training_dataset,
    get_test_prompts,
    get_model_repo_name,
    get_latest_checkpoint,
    is_training_complete
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

def train_dpo_baseline(max_steps=300, output_dir="./outputs/DPO_Baseline", base_model=None, force_skip=False):
    """
    Train DPO baseline with fixed hyperparameters

    Args:
        max_steps: Maximum training steps (default: 300 for full, 100 for sanity)
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
    print(f"  - Training steps: {max_steps}")
    print(f"  - Batch size: 1 (per device, reduced for DPO ref model)")
    print(f"  - Gradient accumulation: 8 (effective batch=8)")
    print(f"  - Learning rate: 1e-5 (Meta's Llama 3 DPO setting)")
    print(f"  - Beta: 0.1 (Meta's Llama 3 DPO setting)")
    print(f"  - Warmup steps: 100 (10% of total steps)")
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

    # Force skip if user selected inference-only mode
    if force_skip:
        print("🚫 User selected inference-only mode")
        print("   Skipping training, will load model from HuggingFace for inference...\n")
        training_skipped = True
    else:
        # Priority 1: Check local checkpoints (CHANGED ORDER - local first, HF second)
        # This allows retraining even if HF repo exists (user chose option 2)
        print("1️⃣ Checking local checkpoints...")
        latest_checkpoint = get_latest_checkpoint(output_dir)

        if latest_checkpoint and is_training_complete(latest_checkpoint, max_steps):
            print(f"✅ Training already completed at: {latest_checkpoint}")
            print(f"   Max steps: {max_steps}")
            print(f"   Skipping training, loading final model...\n")
            training_skipped = True
        elif latest_checkpoint:
            print(f"📂 Found checkpoint: {latest_checkpoint}")
            print(f"   Resuming training from this checkpoint...\n")
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

        # ===== TORCH.COMPILE() OPTIMIZATION =====
        print("\nApplying torch.compile()...")
        model = apply_torch_compile(model)

        # ===== LOAD DATASET =====
        print("\nLoading dataset...")
        train_dataset = load_training_dataset(
            split="train",
            max_samples=None,  # Use all samples
            method="dpo",  # DPO format (prompt, chosen, rejected)
            return_val=False  # Training split
        )

        val_dataset = load_training_dataset(
            split="train",
            max_samples=None,
            method="dpo",
            return_val=True  # Validation split (10% by default)
        )
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
        # Match hyperparameters from 4bit notebook for consistency
        training_args = DPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=1,  # ✅ FIXED: Reduced from 2 to avoid OOM (DPO needs ref model copy)
            gradient_accumulation_steps=8,  # ✅ FIXED: Doubled to maintain effective batch=8
            warmup_steps=100,  # ✅ FIXED: 10% warmup (2024 best practice for DPO)
            max_steps=max_steps,
            learning_rate=1e-5,  # ✅ FIXED: Meta's official Llama 3 DPO setting (was 2e-4, 20× too high)
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # ✅ FIXED: Cosine for smoother convergence
            seed=3407,
            bf16=True,  # BF16 precision
            gradient_checkpointing=True,
            save_steps=50,
            save_total_limit=5,
            report_to="tensorboard",
            logging_dir=str(tensorboard_run_dir),
            logging_first_step=True,
            dataloader_num_workers=2,  # Parallel data loading
            dataloader_pin_memory=True,  # Faster CPU→GPU transfer
            # ✅ ADDED: Validation to detect overfitting
            eval_strategy="steps",
            eval_steps=50,  # Aligned with save_steps
            per_device_eval_batch_size=1,  # ✅ FIXED: Match training batch size
            # ✅ DPO-specific parameters (TRL 0.22.2+)
            beta=0.1,  # Contrastive temperature (standard DPO value)
            max_length=2048,  # Match max_seq_length
            max_prompt_length=1024,  # Half of max_length
        )

        # ===== CREATE DPO TRAINER =====
        print("\nInitializing DPOTrainer...")
        # TRL 0.22.2: DPO params go in DPOConfig, only base params in DPOTrainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # DPOTrainer creates reference model automatically
            processing_class=tokenizer,  # TRL 0.22.2 parameter name
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # ===== SHOW GPU MEMORY =====
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved before training.")
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
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)

        print(f"\n{'='*80}")
        print(f"✅ Training complete!")
        print(f"Peak reserved memory = {used_memory} GB")
        print(f"Peak reserved memory for training = {used_memory_for_training} GB")
        print(f"Peak reserved memory % of max memory = {used_percentage}%")
        print(f"{'='*80}\n")

    # ===== SAVE LORA ADAPTERS =====
    lora_output_dir = Path(output_dir) / "lora_model_DPO_Baseline"

    if not training_skipped:
        lora_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving LoRA adapters to: {lora_output_dir}")
        model.save_pretrained(str(lora_output_dir))
        tokenizer.save_pretrained(str(lora_output_dir))
        print(f"✅ LoRA adapters saved!")
    else:
        # Training skipped - try HF first, fallback to local checkpoint
        model, tokenizer = load_model_bf16(
            model_id="meta-llama/Llama-3.1-8B",
            max_seq_length=2048,
            use_flash_attention=True
        )
        from peft import PeftModel

        # Try downloading from HuggingFace first
        try:
            print(f"📥 Downloading model from HuggingFace for inference: {HF_REPO}")
            model = PeftModel.from_pretrained(model, HF_REPO, token=HF_TOKEN)
            print(f"✅ Model downloaded from HuggingFace")
        except Exception as e:
            # HF repo not available, load from local checkpoint
            print(f"❌ HuggingFace download failed: {type(e).__name__}")
            print(f"📥 Loading model from local checkpoint: {lora_output_dir}")
            model = PeftModel.from_pretrained(model, str(lora_output_dir))
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
  # Sanity check (200 steps, ~8 minutes)
  python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode sanity

  # Full training (1000 steps, ~40 minutes)
  python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full

  # Custom steps
  python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --steps 500
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["sanity", "full"],
        default="full",
        help="Training mode: 'sanity' (100 steps) or 'full' (1000 steps)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides --mode)"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="HuggingFace model ID to load LoRA adapters from (for stacking SFT→DPO→CITA)"
    )

    args = parser.parse_args()

    # Determine configuration
    if args.steps is not None:
        max_steps = args.steps
        print(f"✅ Custom configuration: {max_steps} steps")
    elif args.mode == "sanity":
        max_steps = 200  # 100 warmup + 100 training (see actual LR schedule)
        print(f"✅ Sanity check mode: {max_steps} steps (~8 minutes)")
    else:
        max_steps = 1000
        print(f"✅ Full training mode: {max_steps} steps (~40 minutes)")

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
    print(f"Training will take approximately: {'~12 minutes' if max_steps == 200 else '~62 minutes'}")
    print(f"\nOptions:")
    if hf_model_exists:
        print(f"  1) Run inference only (use existing HF model)")
        print(f"  2) Retrain and replace HF model (only if performance improves)")
    else:
        print(f"  1) Skip training")
        print(f"  2) Train and push to HuggingFace")
    print(f"{'='*80}")

    mode_choice = input("Enter choice (1 or 2): ").strip()
    print(f"{'='*80}\n")

    force_skip = False  # Flag to override checkpoint detection
    if mode_choice == "1":
        print("✅ Inference-only mode selected")
        force_skip = True  # Will skip training regardless of checkpoint status
    elif mode_choice == "2":
        print("✅ Training mode selected")
        if hf_model_exists:
            print("   Will retrain and push ONLY if performance improves")
        else:
            print("   Will train and push to HuggingFace")
        force_skip = False
    else:
        print("⚠️  Invalid choice, defaulting to training mode")
        force_skip = False

    # Run training
    try:
        trainer, training_skipped = train_dpo_baseline(max_steps=max_steps, base_model=args.base_model, force_skip=force_skip)
        print(f"\n🏁 DPO Baseline Training Complete!")
        print(f"📝 Log file: {log_filename}")

        # ===================================================================
        # Automated Push to HuggingFace & GitHub + Auto-Shutdown
        # ===================================================================

        # Extract final rewards/margin from training or checkpoint
        # DPO logs: rewards/chosen, rewards/rejected, rewards/margin
        # NOTE: trainer.state.log_history[-1] might be train_runtime (no metrics)
        #       Better to read from checkpoint's trainer_state.json (saved before train_runtime added)
        if not training_skipped:
            # Load metric from checkpoint's trainer_state.json (most reliable)
            import json
            from model_utils import get_latest_checkpoint
            latest_checkpoint = get_latest_checkpoint(str(project_root / "outputs" / "DPO_Baseline"))
            if latest_checkpoint:
                trainer_state_path = Path(latest_checkpoint) / "trainer_state.json"
                if trainer_state_path.exists():
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                        if trainer_state.get('log_history'):
                            # Get last entry (checkpoint saved before train_runtime added)
                            last_entry = trainer_state['log_history'][-1]
                            # Note: DPO uses 'margins' (plural) not 'margin'
                            final_margin = last_entry.get('eval_rewards/margins',
                                                         last_entry.get('rewards/margins',
                                                                      last_entry.get('eval_loss',
                                                                                   last_entry.get('loss', 'N/A'))))
                        else:
                            final_margin = 'N/A'
                else:
                    final_margin = 'N/A'
            else:
                final_margin = 'N/A'
        else:
            # Training was skipped - load metric from checkpoint's trainer_state.json
            import json
            from model_utils import get_latest_checkpoint
            latest_checkpoint = get_latest_checkpoint(str(project_root / "outputs" / "DPO_Baseline"))
            if latest_checkpoint:
                trainer_state_path = Path(latest_checkpoint) / "trainer_state.json"
                if trainer_state_path.exists():
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                        if trainer_state.get('log_history'):
                            # Get last entry, prefer eval metrics over training metrics
                            last_entry = trainer_state['log_history'][-1]
                            # Note: DPO uses 'margins' (plural) not 'margin'
                            final_margin = last_entry.get('eval_rewards/margins',
                                                         last_entry.get('rewards/margins',
                                                                      last_entry.get('eval_loss',
                                                                                   last_entry.get('loss', 'N/A'))))
                        else:
                            final_margin = 'N/A'
                else:
                    final_margin = 'N/A'
            else:
                final_margin = 'N/A'

        # Save training config
        import json
        config_dir = project_root / "outputs"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "dpo_baseline_config.json"

        training_config = {
            "method": "DPO",
            "max_steps": max_steps,
            "learning_rate": 1e-5,  # Meta's official Llama 3 DPO setting
            "warmup_steps": 100,  # 10% warmup (2024 best practice)
            "optimizer": "adamw_torch",
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",  # Cosine for smoother convergence
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "beta": 0.1,  # Meta's official Llama 3 DPO setting
            "final_margin": final_margin if final_margin != 'N/A' else None,
        }

        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)

        print(f"📊 Saved training config: {config_path}")

        # Create simple namespace to mimic best_trial interface
        from types import SimpleNamespace
        pseudo_trial = SimpleNamespace(
            final_metric=final_margin
        )

        # Get best checkpoint path (last checkpoint saved)
        lora_checkpoint = str(project_root / "outputs" / "DPO_Baseline" / "lora_model_DPO_Baseline")

        # Initialize push automation
        pusher = PushAutomation(
            hf_token=HF_TOKEN,
            github_email="kapilw25@gmail.com",
            github_username="kapilw25",
            project_root=project_root
        )

        # Push to HF (conditional) + GitHub (always)
        pusher.push_all(
            best_trial=pseudo_trial,
            best_checkpoint=lora_checkpoint,
            hf_repo=HF_REPO,
            config_path=str(config_path),
            run_name=RUN_NAME,
            metric_name="rewards/margin",
            metric_mode="max",  # Higher margin is better
            skip_local_backup=training_skipped  # Skip if inference-only mode
        )

        # Summary
        print(f"\n{'='*80}")
        print("✅ All results saved!")
        print(f"{'='*80}")
        print("Results saved to:")
        print(f"  - Local: {lora_checkpoint}")
        print(f"  - HuggingFace: {HF_REPO} (only if performance improved)")
        print(f"  - GitHub: Logs and code pushed")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Restore original stdout/stderr and close log file
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"📝 Complete log file saved: {log_filename}")
