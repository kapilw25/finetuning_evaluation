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
- Expected time: ~62 minutes on A100-40GB (1000 steps)

Best Hyperparameters (from Optuna Trial 2, 400 steps, margin=4.34, acc=89.4%):
- Learning rate: 1.185448e-05
- Beta: 0.1133
- Lambda KL: 0.001010
- Weight decay: 0.008849
- Warmup steps: 103

Usage:
    # Sanity check (200 steps, ~12 minutes)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode sanity

    # Full training (1000 steps, ~62 minutes)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full

Outputs:
    - Model checkpoint: ./outputs/CITA_Baseline/checkpoint-1000/
    - LoRA adapters: ./outputs/CITA_Baseline/lora_model_CITA_Baseline/
    - TensorBoard logs: ./tensorboard_logs/CITA_Baseline_<timestamp>/
    - Training log: ./logs/CITA_Baseline_training_<timestamp>.log
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
    load_training_dataset,
    get_test_prompts,
    get_model_repo_name,
    get_latest_checkpoint,
    is_training_complete,
    log_gpu_memory_start,
    log_gpu_memory_end
)
from push_automation import PushAutomation
from logging_utils import setup_training_logger, restore_logging

# ===================================================================
# Advanced Logging Setup (Tee System)
# ===================================================================
log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
    run_name="CITA_Baseline",
    project_root=project_root
)

# ===================================================================
# HuggingFace Authentication
# ===================================================================
HF_TOKEN = load_hf_token(project_root)

# Get HuggingFace repository name
RUN_NAME = "CITA_Baseline"
HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

print(f"📦 Model will be pushed to: {HF_REPO}")
print("="*80 + "\n")


# ===================================================================
# Main Training Function
# ===================================================================

def train_cita_baseline(max_steps=300, output_dir="./outputs/CITA_Baseline", base_model=None, force_skip=False):
    """
    Train CITA baseline with fixed hyperparameters from best Optuna trial

    Args:
        max_steps: Number of training steps (200 for sanity, 1000 for full)
        output_dir: Directory to save checkpoints
        base_model: HuggingFace model ID to load LoRA adapters from (DPO model)
        force_skip: Skip training and use HuggingFace model directly

    Returns:
        trainer: CITATrainer instance
        training_skipped: Whether training was skipped
    """

    # ===== BEST HYPERPARAMETERS FROM OPTUNA =====
    # Trial 2: 400 steps, margin=4.34, accuracy=89.4% (best at 200 steps: +20.2% vs DPO)
    LAMBDA_KL = 0.0010102922471479012
    LEARNING_RATE = 1.1854483432291239e-05
    BETA = 0.11329770563201687
    WEIGHT_DECAY = 0.008849356442713105
    WARMUP_STEPS = 103

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard directory (unique per run)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_run_dir = project_root / "tensorboard_logs" / f"CITA_Baseline_{timestamp}"
    tensorboard_run_dir.mkdir(parents=True, exist_ok=True)

    # ===== CHECK FOR EXISTING CHECKPOINT =====
    print("="*80)
    print("🔍 Checking for existing checkpoints...")
    print("="*80 + "\n")

    latest_checkpoint = get_latest_checkpoint(output_dir)
    training_skipped = False

    if latest_checkpoint:
        print(f"1️⃣ Checking local checkpoints...")
        if is_training_complete(latest_checkpoint, max_steps):
            print(f"✅ Found complete training checkpoint: {latest_checkpoint.name}")
            print(f"   Training already finished at {max_steps} steps")
            training_skipped = True
        else:
            completed_steps = int(latest_checkpoint.name.split("-")[-1])
            print(f"⏸️  Found incomplete checkpoint: {latest_checkpoint.name}")
            print(f"   Will resume from step {completed_steps}")
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
        print("\nApplying torch.compile()...")
        model = apply_torch_compile(model)

        # ===== LOAD DATASET =====
        print("\nLoading dataset...")
        from data_prep import load_pku_filtered, format_dataset

        # Training set (90% of train split)
        dataset_raw_train = load_pku_filtered(
            split="train",
            max_samples=None,
            return_val=False
        )
        train_dataset = format_dataset(dataset_raw_train, method="dpo")

        # Validation set (10% of train split)
        dataset_raw_val = load_pku_filtered(
            split="train",
            max_samples=None,
            return_val=True
        )
        val_dataset = format_dataset(dataset_raw_val, method="dpo")

        print(f"📊 TensorBoard logs: {tensorboard_run_dir}\n")

        # ===== CREATE TRAINING ARGS =====
        training_args = DPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=WARMUP_STEPS,
            max_steps=max_steps,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="cosine",
            seed=3407,
            bf16=True,
            gradient_checkpointing=True,
            save_steps=50,
            save_total_limit=5,
            report_to="tensorboard",
            logging_dir=str(tensorboard_run_dir),
            logging_first_step=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            # Validation
            eval_strategy="steps",
            eval_steps=50,
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

        lora_output_dir = output_dir / "lora_model_CITA_Baseline"
        lora_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(lora_output_dir))
        tokenizer.save_pretrained(str(lora_output_dir))

        print(f"✅ LoRA adapters saved to: {lora_output_dir}\n")

        # ===== SHOW FINAL MEMORY =====
        log_gpu_memory_end(start_gpu_memory)

    # ===== EXTRACT FINAL METRICS =====
    final_margin = 'N/A'
    if not training_skipped and hasattr(trainer.state, 'log_history'):
        for log_entry in reversed(trainer.state.log_history):
            if 'eval_rewards/margins' in log_entry:
                final_margin = log_entry['eval_rewards/margins']
                break

    # ===== SAVE TRAINING CONFIG =====
    training_config = {
        "method": "CITA",
        "max_steps": max_steps,
        "lambda_kl": LAMBDA_KL,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "max_seq_length": 2048,
        "max_prompt_length": 1024,
        "final_margin": final_margin if final_margin != 'N/A' else None,
    }

    import json
    config_path = project_root / "outputs" / "cita_baseline_config.json"
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)

    print(f"✅ Training config saved to: {config_path}\n")

    return trainer, training_skipped


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
        help="Training mode: 'sanity' (200 steps) or 'full' (1000 steps)"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="HuggingFace model ID to load LoRA adapters from (e.g., DPO model)"
    )

    args = parser.parse_args()

    # Determine max_steps based on mode
    if args.mode == "sanity":
        max_steps = 200
        print(f"✅ Sanity mode: {max_steps} steps (~12 minutes)\n")
    else:  # full
        max_steps = 1000
        print(f"✅ Full training mode: {max_steps} steps (~62 minutes)\n")

    # Time estimate
    time_per_step = 0.062  # minutes (CITA: ~62 min / 1000 steps)
    estimated_time = max_steps * time_per_step

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
    print(f"Training will take approximately: ~{int(estimated_time)} minutes\n")

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
    print("🚀 Starting CITA Baseline Training")
    print("="*80)
    print("Configuration:")
    print("  - Model: Llama-3.1-8B (BF16)")
    print("  - Method: CITA (Calibrated Instruction Tuning with Alignment)")
    print("  - Loss: L_CITA = L_DPO + λ_KL × L_KL")
    print(f"  - Training steps: {max_steps}")
    print("  - Batch size: 1 (per device)")
    print("  - Gradient accumulation: 8 (effective batch=8)")
    print("  - Learning rate: 1.185e-05 (Optuna Trial 2)")
    print("  - Beta: 0.1133 (Optuna Trial 2)")
    print("  - Lambda KL: 0.001010 (Optuna Trial 2)")
    print("  - Weight decay: 0.00885 (Optuna Trial 2)")
    print("  - Warmup steps: 103 (Optuna Trial 2)")
    print("  - LR scheduler: cosine")
    print("  - Optimizer: adamw_torch")
    print("  - Precision: BF16 + Flash Attention 2")
    print("="*80 + "\n\n")

    # Train
    try:
        trainer, training_skipped = train_cita_baseline(
            max_steps=max_steps,
            base_model=args.base_model,
            force_skip=force_skip
        )

        # ===== PUSH TO HUGGINGFACE =====
        if not force_skip or (mode_choice == "2" and hf_model_exists):
            print("\n" + "="*80)
            print("📤 Pushing to HuggingFace")
            print("="*80 + "\n")

            lora_output_dir = project_root / "outputs" / "CITA_Baseline" / "lora_model_CITA_Baseline"

            # Load training config
            import json
            config_path = project_root / "outputs" / "cita_baseline_config.json"
            with open(config_path, 'r') as f:
                training_config = json.load(f)

            # Initialize push automation
            push_automation = PushAutomation(project_root=project_root)

            # If Option 2 with existing HF model, compare metrics
            should_push = True
            if mode_choice == "2" and hf_model_exists and not training_skipped:
                print("\n📊 Comparing local model with HuggingFace model...")
                local_margin = training_config.get('final_margin')

                try:
                    # Try to get HF model's README to extract margin
                    from huggingface_hub import hf_hub_download
                    readme_path = hf_hub_download(
                        repo_id=HF_REPO,
                        filename="README.md",
                        token=HF_TOKEN
                    )
                    with open(readme_path, 'r') as f:
                        readme_content = f.read()

                    # Extract final margin from README (simple parsing)
                    hf_margin = None
                    for line in readme_content.split('\n'):
                        if 'Final Margin:' in line:
                            try:
                                hf_margin = float(line.split(':')[1].strip())
                            except:
                                pass

                    if local_margin is not None and hf_margin is not None:
                        print(f"\nLocal margin: {local_margin:.4f}")
                        print(f"HF margin: {hf_margin:.4f}")

                        if local_margin > hf_margin:
                            print(f"✅ Local model is BETTER (Δ = +{local_margin - hf_margin:.4f})")
                            should_push = True
                        else:
                            print(f"⚠️  Local model is WORSE (Δ = {local_margin - hf_margin:.4f})")
                            print(f"   Skipping push to preserve better HF model")
                            should_push = False
                    else:
                        print("⚠️  Could not extract margin from HF model, will push anyway")
                        should_push = True

                except Exception as e:
                    print(f"⚠️  Could not fetch HF model README: {e}")
                    print("   Will push anyway")
                    should_push = True

            if should_push:
                success, message = push_automation.push_to_huggingface(
                    lora_dir=lora_output_dir,
                    hf_repo=HF_REPO,
                    method="CITA",
                    base_model="meta-llama/Llama-3.1-8B",
                    dataset="PKU-Alignment/PKU-SafeRLHF",
                    config=training_config,
                    final_metric=training_config.get('final_margin', 0),
                    metric_name="margin"
                )

                if success:
                    print(f"\n✅ {message}")
                    print(f"🔗 Model available at: https://huggingface.co/{HF_REPO}\n")
                else:
                    print(f"\n❌ {message}\n")
            else:
                print("\n⏭️  Skipped HuggingFace push (local model not better)\n")

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
