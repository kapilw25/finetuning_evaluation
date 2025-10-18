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
- Expected time: ~40 minutes on A100-40GB (1000 steps)
- Expected cost: ~$1.00 (40 min × $1.5/hr)

Usage:
    # Sanity check (100 steps, ~4 minutes)
    python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity

    # Full training (1000 steps, ~40 minutes)
    python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full

Outputs:
    - Model checkpoint: ./outputs/SFT_Baseline/checkpoint-1000/
    - LoRA adapters: ./outputs/SFT_Baseline/lora_model_SFT_Baseline/
    - TensorBoard logs: ./tensorboard_logs/SFT_Baseline_<timestamp>/
    - Training log: ./logs/SFT_Baseline_training_<timestamp>.log
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
    load_training_dataset,
    get_test_prompts,
    get_model_repo_name
)
from push_automation import PushAutomation

# ===================================================================
# Logging Setup
# ===================================================================
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = logs_dir / f"SFT_Baseline_training_{timestamp}.log"

# Simple logging (no Tee class needed for baseline)
print(f"📝 Logging to: {log_filename}")

# ===================================================================
# HuggingFace Authentication
# ===================================================================
HF_TOKEN = load_hf_token(project_root)

# Get HuggingFace repository name
RUN_NAME = "SFT_Baseline"
HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

print(f"📦 Model will be pushed to: {HF_REPO}")
print("="*80 + "\n")


# ===================================================================
# Main Training Function
# ===================================================================

def train_sft_baseline(max_steps=300, output_dir="./outputs/SFT_Baseline"):
    """
    Train SFT baseline with fixed hyperparameters

    Args:
        max_steps: Maximum training steps (default: 300 for full, 100 for sanity)
        output_dir: Output directory for checkpoints

    Returns:
        trainer: Trained SFTTrainer instance
    """
    print("\n" + "="*80)
    print(f"🚀 Starting SFT Baseline Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Model: Llama-3.1-8B (BF16)")
    print(f"  - Method: Standard SFT (supervised learning)")
    print(f"  - Loss: L_SFT only")
    print(f"  - Training steps: {max_steps}")
    print(f"  - Batch size: 2 (per device)")
    print(f"  - Gradient accumulation: 4 (effective batch=8)")
    print(f"  - Learning rate: 2e-4")
    print(f"  - Warmup steps: 5")
    print(f"  - Optimizer: adamw_torch")
    print(f"  - Precision: BF16 + Flash Attention 2")
    print("="*80 + "\n")

    # ===== LOAD MODEL & TOKENIZER =====
    print("Loading model...")
    model, tokenizer = load_model_bf16(
        model_id="meta-llama/Llama-3.1-8B",
        max_seq_length=2048,  # Match DPO baseline for fair comparison
        use_flash_attention=True
    )

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
    dataset = load_training_dataset(
        split="train",
        max_samples=None,  # Use all samples
        method="sft"  # SFT format (chosen responses only)
    )

    # ===== TENSORBOARD SETUP =====
    tensorboard_base_dir = project_root / "tensorboard_logs"
    tensorboard_base_dir.mkdir(exist_ok=True)

    tensorboard_run_dir = tensorboard_base_dir / f"{RUN_NAME}_{timestamp}"

    print(f"📊 TensorBoard logs: {tensorboard_run_dir}")

    # ===== CREATE TRAINING ARGS =====
    # Match hyperparameters from DPO baseline for fair comparison
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=max_steps,
        learning_rate=2e-4,  # Match DPO baseline
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
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
        dataset_text_field="text",  # SFT expects text field
        max_seq_length=2048,  # Match model's max_seq_length
        packing=False,  # Disable packing for alignment training
    )

    # ===== CREATE SFT TRAINER =====
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    # ===== SHOW GPU MEMORY =====
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved before training.")

    # ===== TRAIN =====
    print("\n" + "="*80)
    print("🏋️  Training SFT Baseline...")
    print("="*80 + "\n")

    trainer.train()

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

    # ===== SAVE LORA ADAPTERS =====
    lora_output_dir = Path(output_dir) / "lora_model_SFT_Baseline"
    lora_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving LoRA adapters to: {lora_output_dir}")
    model.save_pretrained(str(lora_output_dir))
    tokenizer.save_pretrained(str(lora_output_dir))
    print(f"✅ LoRA adapters saved!")

    # ===== INFERENCE TEST =====
    print("\n" + "="*80)
    print("🧪 Running inference tests...")
    print("="*80 + "\n")

    from transformers import TextStreamer

    # Enable inference mode
    model.eval()

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

    return trainer


# ===================================================================
# Main Execution
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT Baseline Training - Standard Supervised Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sanity check (100 steps, ~4 minutes)
  python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode sanity

  # Full training (1000 steps, ~40 minutes)
  python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full

  # Custom steps
  python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --steps 500
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

    args = parser.parse_args()

    # Determine configuration
    if args.steps is not None:
        max_steps = args.steps
        print(f"✅ Custom configuration: {max_steps} steps")
    elif args.mode == "sanity":
        max_steps = 100
        print(f"✅ Sanity check mode: {max_steps} steps (~4 minutes)")
    else:
        max_steps = 1000
        print(f"✅ Full training mode: {max_steps} steps (~40 minutes)")

    # Run training
    try:
        trainer = train_sft_baseline(max_steps=max_steps)
        print(f"\n🏁 SFT Baseline Training Complete!")
        print(f"📝 Log file: {log_filename}")

        # ===================================================================
        # Automated Push to HuggingFace & GitHub + Auto-Shutdown
        # ===================================================================

        # Extract final loss from training
        final_loss = trainer.state.log_history[-1].get('loss', 'N/A')

        # Save training config
        import json
        config_dir = project_root / "outputs"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "sft_baseline_config.json"

        training_config = {
            "method": "SFT",
            "max_steps": max_steps,
            "learning_rate": 2e-4,
            "warmup_steps": 5,
            "optimizer": "adamw_torch",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "final_loss": final_loss if final_loss != 'N/A' else None,
        }

        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)

        print(f"📊 Saved training config: {config_path}")

        # Create simple namespace to mimic best_trial interface
        from types import SimpleNamespace
        pseudo_trial = SimpleNamespace(
            final_metric=final_loss
        )

        # Get best checkpoint path (last checkpoint saved)
        lora_checkpoint = str(project_root / "outputs" / "SFT_Baseline" / "lora_model_SFT_Baseline")

        # Initialize push automation
        pusher = PushAutomation(
            hf_token=HF_TOKEN,
            github_email="kapilw25@gmail.com",
            github_username="kapilw25",
            project_root=project_root
        )

        # Push to HF (conditional) + GitHub (always) + auto-shutdown
        pusher.push_all(
            best_trial=pseudo_trial,
            best_checkpoint=lora_checkpoint,
            hf_repo=HF_REPO,
            config_path=str(config_path),
            run_name=RUN_NAME,
            metric_name="loss",
            metric_mode="min"  # Lower loss is better
        )

        # Auto-shutdown GPU instance (cost savings)
        print(f"\n{'='*80}")
        print("💰 Auto-Shutdown: Stopping GPU instance to save costs")
        print(f"{'='*80}")
        print("All results saved to:")
        print(f"  - Local: {lora_checkpoint}")
        print(f"  - HuggingFace: {HF_REPO} (if performance improved)")
        print(f"  - GitHub: Logs and code pushed")
        print(f"{'='*80}\n")

        os.system("sudo shutdown -h now")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
