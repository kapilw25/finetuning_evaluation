"""
Usage (6 commands = 3 modes × 2 instruction settings):

    ### 2c. GRPO (venv_GRPO - requires TRL 0.22.2)
    source venv_GRPO/bin/activate

    # MICRO: 0.05 epochs (~35 min)
    TMUX >> python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode micro --use-instruction false
    TMUX >> python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode micro --use-instruction true

    # SANITY: 0.3 epochs (~4 hours)
    TMUX >> python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction false
    TMUX >> python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction true

    # FULL: 1.0 epoch (~12 hours) - use TMUX
    TMUX >> python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode full --use-instruction false
    TMUX >> python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode full --use-instruction true

  Projections (batch_size=12, grad_accum=2, effective_batch=24):
  | Mode                | Steps | ETA        |
  |---------------------|-------|------------|
  | Micro (0.05 epochs) | ~22   | ~35 min    |
  | Sanity (0.3 epochs) | ~135  | ~4 hours   |
  | Full (1.0 epoch)    | ~451  | ~12 hours  |
"""

import sys
from pathlib import Path
import os
import argparse
from datetime import datetime
# ===== FIX CUDA OOM: Enable expandable segments for memory fragmentation =====
# GRPO generates multiple responses per prompt, causing memory spikes
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

# ===== FIX torch.compile() CUDAGraph bug: Disable CUDAGraphs for dynamic shapes =====
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer

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
    get_test_prompts,
    get_model_repo_name,
    get_latest_checkpoint,
    is_training_complete,
    log_gpu_memory_start,
    log_gpu_memory_end,
    BASE_MODEL_MAP
)
from data_prep.loader_pku import load_pku_combined_clear_contrast, get_safe_unsafe_responses, synthesize_system_instruction
from push_automation import PushAutomation
from logging_utils import setup_training_logger, restore_logging
from monitoring_callback import TrainingSummaryCallback

# ===================================================================
# INSTRUCTION MODE TOGGLE (set via command-line argument --use-instruction)
# ===================================================================
# USE_INSTRUCTION will be set from command-line args (no default value)
# Script will fail if --use-instruction is not provided


# ===================================================================
# Reward Functions for Safety Alignment
# ===================================================================
# Note: These are APPROXIMATIONS of human preferences
# DPO uses actual human labels, which is why DPO > GRPO for safety tasks

def safety_refusal_reward(completions, prompts, **kwargs) -> list[float]:
    """
    Reward for refusing harmful requests.
    Checks for refusal phrases in response.

    This is a crude heuristic - human labels (DPO) capture nuance better.
    """
    refusal_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i will not", "i won't", "it's not appropriate",
        "it is not appropriate", "i must refuse", "sorry, but",
        "i'm not able", "i am not able", "cannot assist",
        "cannot help", "not comfortable", "against my guidelines",
        "would be harmful", "could be dangerous", "not safe"
    ]

    rewards = []
    for completion in completions:
        response = completion[0]['content'].lower() if completion else ""
        has_refusal = any(phrase in response for phrase in refusal_phrases)
        rewards.append(1.0 if has_refusal else 0.0)

    return rewards


def helpfulness_reward(completions, **kwargs) -> list[float]:
    """
    Reward for helpful, substantive responses.
    Penalizes very short or empty responses.
    """
    rewards = []
    for completion in completions:
        response = completion[0]['content'] if completion else ""
        length = len(response.split())

        if length < 10:
            rewards.append(0.0)  # Too short
        elif length < 30:
            rewards.append(0.3)  # Brief
        elif length < 80:
            rewards.append(0.7)  # Moderate
        else:
            rewards.append(1.0)  # Substantive

    return rewards


def format_quality_reward(completions, **kwargs) -> list[float]:
    """
    Reward for well-formatted responses.
    Checks for proper structure (no repetition, complete sentences).
    """
    rewards = []
    for completion in completions:
        response = completion[0]['content'] if completion else ""

        if not response.strip():
            rewards.append(0.0)
            continue

        score = 0.5  # Base score

        # Penalize repetition
        words = response.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Very high repetition
                score -= 0.4
            elif unique_ratio < 0.5:  # High repetition
                score -= 0.2

        # Reward complete sentences
        if response.strip().endswith(('.', '!', '?')):
            score += 0.2

        # Penalize gibberish (excessive special chars)
        if len(response) > 0:
            special_ratio = sum(1 for c in response if not c.isalnum() and c not in ' .,!?\'"-:;()') / len(response)
            if special_ratio > 0.3:
                score -= 0.3

        # Penalize responses that just repeat the prompt
        # (common failure mode in RL)
        if len(words) > 3 and len(set(words[:5])) < 3:
            score -= 0.2

        rewards.append(max(0.0, min(1.0, score)))

    return rewards


# ===================================================================
# Dataset Preparation for GRPO
# ===================================================================

def prepare_grpo_dataset(use_instruction: bool = False):
    """
    Prepare PKU-SafeRLHF dataset for GRPO training.

    GRPO only needs prompts - responses are generated during training.
    Unlike DPO, we don't use the chosen/rejected labels.

    FAIR COMPARISON: Uses same synthesize_system_instruction() as DPO
    to get CUSTOMIZED instructions per example based on harm categories.

    Returns:
        tuple: (train_dataset, val_dataset) with 'prompt' field (list of chat messages)
    """
    # Load PKU dataset (90/10 split)
    dataset_split = load_pku_combined_clear_contrast(val_split=0.1)
    train_data = dataset_split['train']
    val_data = dataset_split['test']

    def format_prompt(example):
        prompt = example.get('prompt', '')

        if use_instruction:
            # FAIR: Use SAME customized instruction as DPO!
            # Extract harm categories from example (same as DPO)
            _, _, harmful_categories = get_safe_unsafe_responses(example)

            # Synthesize instruction based on harm type (same as DPO)
            instruction = synthesize_system_instruction(harmful_categories)

            # GRPO expects 'prompt' as list of messages with system role
            return {
                "prompt": [
                    {"role": "system", "content": instruction},  # ← CUSTOMIZED per example!
                    {"role": "user", "content": prompt}
                ]
            }
        else:
            # NoInstruct variant: no system message (same as DPO_NoInstruct)
            return {
                "prompt": [
                    {"role": "user", "content": prompt}
                ]
            }

    # Format train dataset
    formatted_train = train_data.map(
        format_prompt,
        remove_columns=train_data.column_names,
        desc=f"Formatting PKU train for GRPO ({'WITH CUSTOMIZED' if use_instruction else 'NO'} instruction)"
    )

    # Format validation dataset
    formatted_val = val_data.map(
        format_prompt,
        remove_columns=val_data.column_names,
        desc=f"Formatting PKU val for GRPO ({'WITH CUSTOMIZED' if use_instruction else 'NO'} instruction)"
    )

    print(f"✅ Prepared {len(formatted_train):,} train / {len(formatted_val):,} val prompts for GRPO")
    if use_instruction:
        print(f"   Using CUSTOMIZED instructions per example (fair comparison with DPO)")
    print(f"   Note: GRPO generates responses online (no chosen/rejected pairs used)")

    return formatted_train, formatted_val


# ===================================================================
# Main Training Function
# ===================================================================

def train_grpo_baseline(num_epochs=1.0, output_dir=None, base_model=None, force_skip=False, training_mode="full"):
    """
    Train GRPO baseline with online generation.

    GRPO generates multiple responses per prompt and uses group-relative
    advantage for policy updates (no reference model needed).

    Args:
        num_epochs: Number of training epochs
        output_dir: Output directory for checkpoints
        base_model: HuggingFace model ID to load LoRA adapters from (for stacking)
        force_skip: If True, skip training and only run inference
        training_mode: "micro" (256 tokens), "sanity" (256 tokens), "full" (512 tokens)

    Returns:
        trainer: Trained GRPOTrainer instance
        training_skipped: Whether training was skipped
        training_config: Dict of training hyperparameters for HF metadata
    """
    # Set output_dir dynamically based on RUN_NAME if not provided
    if output_dir is None:
        output_dir = f"./outputs/training/{RUN_NAME}"

    # ===== GRPO HYPERPARAMETERS (A100-80GB MAXED) =====
    # Define at start to use in prints and avoid NameError in training_config
    # CONSTRAINT: (batch * grad_accum) must be divisible by num_generations
    # Solution: 12 * 2 = 24, 24 % 6 = 0 ✓
    num_generations = 6
    per_device_batch_size = 12  # 12 prompts × 6 gen = 72 responses/step
    gradient_accum_steps = 2

    print("\n" + "="*80)
    print(f"🚀 Starting {RUN_NAME} Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Model: Llama-3.1-8B (BF16)")
    print(f"  - Method: GRPO (Group Relative Policy Optimization)")
    print(f"  - Training: Online generation + reward functions")
    print(f"  - Training epochs: {num_epochs}")
    print(f"  - Batch size: {per_device_batch_size} (per device, optimized for 80GB GPU)")
    print(f"  - Gradient accumulation: {gradient_accum_steps} (effective batch={per_device_batch_size * gradient_accum_steps} prompts)")
    print(f"  - Num generations: {num_generations} (responses per prompt)")
    print(f"  - Learning rate: 5e-6 (lower for RL stability)")
    print(f"  - Warmup ratio: 10%")
    print(f"  - LR scheduler: cosine")
    print(f"  - Optimizer: adamw_torch")
    print(f"  - Precision: BF16 + Flash Attention 2")
    print("="*80 + "\n")

    # ===== CHECKPOINT DETECTION =====
    print("\n" + "="*80)
    print("🔍 Checking for existing checkpoints...")
    print("="*80 + "\n")

    training_skipped = False
    latest_checkpoint = None
    load_from_hf = False

    if force_skip:
        print("🚫 User selected inference-only mode")
        print("   Skipping training, will load model from HuggingFace for inference...\n")
        training_skipped = True
        load_from_hf = True
    else:
        print("1️⃣ Checking local checkpoints...")
        latest_checkpoint = get_latest_checkpoint(output_dir)

        if latest_checkpoint:
            print(f"📂 Found checkpoint: {latest_checkpoint}")
            print(f"   Will check completion status after loading dataset...\n")
        else:
            print(f"🆕 No checkpoint found")
            print(f"   Will start fresh training...\n")

    # ===== LOAD MODEL & TOKENIZER =====
    if not training_skipped:
        print("Loading model...")
        model, tokenizer = load_model_bf16(
            model_id="meta-llama/Llama-3.1-8B",
            max_seq_length=2048,
            use_flash_attention=True
        )

        # Set padding side for generation
        tokenizer.padding_side = "left"

        # ===== LOAD BASE MODEL LORA (IF STACKING) =====
        if base_model:
            print(f"\n🔗 Loading LoRA adapters from HuggingFace: {base_model}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, base_model, token=HF_TOKEN)
            print("✅ LoRA adapters loaded")

            # Merge adapters into base model
            print("🔄 Merging LoRA adapters into base model...")
            merged_model = model.merge_and_unload()

            # Clear PEFT config
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

        # Cast to BF16
        model = model.to(torch.bfloat16)
        print("✅ Model cast to BF16 (all params including LoRA)")

        # torch.compile() disabled for RL stability
        print("\n⚠️  torch.compile() disabled (prevents RL training instability)")

        # ===== LOAD DATASET =====
        print("\nLoading PKU-SafeRLHF dataset (prompts only for GRPO)...")
        train_dataset, val_dataset = prepare_grpo_dataset(use_instruction=USE_INSTRUCTION)

        # ===== LIMIT VAL DATASET (cap for reasonable eval time) =====
        # Full mode: 200 samples max. Micro/sanity: proportionally smaller.
        MAX_VAL_SAMPLES = 200  # Base cap for full mode
        val_fraction = num_epochs  # 0.05 for micro, 0.3 for sanity, 1.0 for full
        val_samples = max(1, int(MAX_VAL_SAMPLES * val_fraction))
        val_dataset = val_dataset.select(range(val_samples))
        est_eval_min = (val_samples * 13 * 5) // 60  # 5 evals during training
        print(f"📊 Validation: {val_samples} samples ({val_fraction:.0%} of {MAX_VAL_SAMPLES} cap, ~{est_eval_min} min total eval)")

        # ===== CALCULATE TRAINING STEPS =====
        # GRPO: effective batch = batch_size * gradient_accum (hyperparameters defined above)
        effective_batch_size = per_device_batch_size * gradient_accum_steps  # 16 prompts per step
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_steps = int(steps_per_epoch * num_epochs)
        checkpoint_interval = max(1, int(total_steps * 0.2))  # 20% intervals

        # Dynamic generation config based on training_mode
        max_completion_length = 512 if training_mode == "full" else 256

        print(f"\n📊 Training Configuration:")
        print(f"   Dataset size: {len(train_dataset):,} prompts")
        print(f"   Per-device batch size: {per_device_batch_size}")
        print(f"   Gradient accumulation: {gradient_accum_steps}")
        print(f"   Num generations: {num_generations}")
        print(f"   Effective batch size: {effective_batch_size} prompts ({effective_batch_size * num_generations} responses/step)")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Total steps: {total_steps:,} ({num_epochs} epochs)")
        print(f"   Checkpoint interval: {checkpoint_interval} steps (20% of training)")
        print(f"📦 Generation: max_completion_length={max_completion_length} (mode={training_mode})")

        # ===== CHECK TRAINING COMPLETION =====
        if latest_checkpoint and is_training_complete(latest_checkpoint, total_steps):
            print(f"\n✅ Training already completed at: {latest_checkpoint}")
            print(f"   Total steps: {total_steps} ({num_epochs} epochs)")
            print(f"   Skipping training, will load from local checkpoint for inference...\n")
            training_skipped = True
            load_from_hf = False
    else:
        model = None
        tokenizer = None
        train_dataset = None
        val_dataset = None

    # ===== SETUP TRAINER =====
    if not training_skipped:
        # TensorBoard setup
        tensorboard_base_dir = project_root / "tensorboard_logs"
        tensorboard_base_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_run_dir = tensorboard_base_dir / f"{RUN_NAME}_{timestamp}"

        print(f"📊 TensorBoard logs: {tensorboard_run_dir}")

        # GRPO Config
        training_args = GRPOConfig(
            output_dir=str(output_dir),

            # Training (optimized for 80GB GPU)
            learning_rate=5e-6,  # Lower LR for RL stability
            per_device_train_batch_size=per_device_batch_size,  # 4 prompts per device
            gradient_accumulation_steps=gradient_accum_steps,   # 4 accumulation steps
            max_steps=total_steps,

            # GRPO-specific (dynamic based on training_mode)
            num_generations=num_generations,
            max_prompt_length=512,
            max_completion_length=max_completion_length,  # Dynamic: 512 for full, 256 for micro/sanity

            # Optimization
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            weight_decay=0.1,
            max_grad_norm=0.1,  # Aggressive clipping for RL stability

            # Logging
            logging_steps=1,
            report_to="tensorboard",
            logging_dir=str(tensorboard_run_dir),
            logging_first_step=True,

            # Checkpointing
            save_steps=checkpoint_interval,
            save_total_limit=5,

            # Evaluation (detect overfitting during training)
            eval_strategy="steps",
            eval_steps=checkpoint_interval,  # Aligned with checkpoints (20% intervals)
            # FIX: eval_batch_size must be divisible by num_generations (GRPOConfig constraint)
            per_device_eval_batch_size=num_generations,  # 6 (divisible by 6)

            # Dataloader optimization
            dataloader_num_workers=2,  # Parallel data loading
            dataloader_pin_memory=True,  # Faster CPU→GPU transfer

            # Precision
            bf16=True,

            seed=3407,
        )

        # Create GRPO Trainer
        print("\nInitializing GRPOTrainer...")
        print("   Reward functions: safety_refusal, helpfulness, format_quality")

        # Training summary callback (consistent with DPO/CITA)
        summary_callback = TrainingSummaryCallback(
            check_every_n_steps=50,
            training_method="grpo"
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                safety_refusal_reward,    # +1.0 for refusing harmful
                helpfulness_reward,       # +1.0 for substantive response
                format_quality_reward,    # +0.5 for good format
            ],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # FIX: Added validation for overfitting detection
            callbacks=[summary_callback],
        )

        # GPU Memory
        start_gpu_memory = log_gpu_memory_start()
    else:
        trainer = None

    # ===== TRAIN =====
    if not training_skipped:
        print("\n" + "="*80)
        print("🏋️  Training GRPO Baseline...")
        print("   Note: GRPO generates responses online (slower than DPO)")
        print("="*80 + "\n")

        trainer.train(resume_from_checkpoint=latest_checkpoint)

        # Extract best metric (reward)
        if hasattr(trainer.state, 'log_history'):
            rewards = [log.get('reward', 0) for log in trainer.state.log_history if 'reward' in log]
            if rewards:
                trainer.state.best_metric = max(rewards)

    # ===== SHOW FINAL MEMORY =====
    if not training_skipped:
        log_gpu_memory_end(start_gpu_memory)
        print(f"✅ Training complete!")

    # ===== SAVE LORA ADAPTERS =====
    lora_output_dir = Path(output_dir) / f"lora_model_{RUN_NAME}"

    if not training_skipped:
        lora_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving LoRA adapters to: {lora_output_dir}")
        model.save_pretrained(str(lora_output_dir))
        tokenizer.save_pretrained(str(lora_output_dir))
        print(f"✅ LoRA adapters saved!")
    elif load_from_hf:
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
        print(f"📂 Loading model from local checkpoint for inference: {latest_checkpoint}")
        model, tokenizer = load_model_bf16(
            model_id="meta-llama/Llama-3.1-8B",
            max_seq_length=2048,
            use_flash_attention=True
        )
        from peft import PeftModel
        checkpoint_lora_path = Path(latest_checkpoint)
        model = PeftModel.from_pretrained(model, str(checkpoint_lora_path))
        print(f"✅ Model loaded from local checkpoint")

    # ===== INFERENCE TEST =====
    print("\n" + "="*80)
    print("🧪 Running inference tests...")
    print("="*80 + "\n")

    model.eval()
    model = model.to(torch.bfloat16)

    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    test_prompts = get_test_prompts()

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

    # Build training_config from actual variables used (avoid hardcoding)
    max_completion_length = 512 if training_mode == "full" else 256  # Match GRPOConfig
    training_config = {
        "method": "GRPO",
        "num_epochs": num_epochs,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.1,
        "optimizer": "adamw_torch",
        "weight_decay": 0.1,
        "lr_scheduler_type": "cosine",
        "batch_size": per_device_batch_size,  # From variable, not hardcoded
        "gradient_accumulation_steps": gradient_accum_steps,  # From variable
        "effective_batch_size": per_device_batch_size * gradient_accum_steps,  # Calculated
        "num_generations": num_generations,  # From variable
        "max_prompt_length": 512,
        "max_completion_length": max_completion_length,  # Dynamic based on mode
        "max_grad_norm": 0.1,
        "reward_functions": ["safety_refusal", "helpfulness", "format_quality"],
    }

    return trainer, training_skipped, training_config


# ===================================================================
# Main Execution
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO Baseline Training - Group Relative Policy Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NoInstruct variant (sanity check, 0.3 epochs, ~2 hours)
  python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction false

  # Instruct variant (full training, 1.0 epoch, ~8 hours)
  python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --mode full --use-instruction true

  # Custom epochs
  python comparative_study/02c_GRPO_Baseline/Llama3_BF16.py --epochs 0.5 --use-instruction false
        """
    )

    parser.add_argument(
        "--use-instruction",
        type=str,
        required=True,
        choices=["true", "false"],
        help="REQUIRED: Use instruction conditioning (true=GRPO_Instruct, false=GRPO_NoInstruct)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["micro", "sanity", "full"],
        default="full",
        help="Training mode: 'micro' (0.05 epochs, ~35 min), 'sanity' (0.3 epochs, ~4 hours), or 'full' (1.0 epochs, ~12 hours)"
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
        help="Override base model (auto-derived from BASE_MODEL_MAP if not provided)"
    )

    args = parser.parse_args()

    # ===================================================================
    # Set USE_INSTRUCTION from command-line argument
    # ===================================================================
    USE_INSTRUCTION = args.use_instruction.lower() == "true"
    RUN_NAME = "GRPO_Instruct" if USE_INSTRUCTION else "GRPO_NoInstruct"

    # Auto-derive base_model from BASE_MODEL_MAP if not provided
    # GRPO stacks on SFT (same as DPO/PPO)
    if args.base_model is None:
        args.base_model = BASE_MODEL_MAP.get(RUN_NAME)

    print(f"✅ Instruction mode: {'ENABLED' if USE_INSTRUCTION else 'DISABLED'} ({RUN_NAME})")
    print(f"✅ Base model: {args.base_model}")

    # ===================================================================
    # Logging Setup
    # ===================================================================
    log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
        run_name=RUN_NAME,
        project_root=project_root
    )

    # ===================================================================
    # HuggingFace Authentication
    # ===================================================================
    HF_TOKEN = load_hf_token(project_root)
    HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

    print(f"📦 Model will be pushed to: {HF_REPO}")
    print("="*80 + "\n")

    # Determine configuration
    if args.epochs is not None:
        num_epochs = args.epochs
        print(f"✅ Custom configuration: {num_epochs} epochs")
    elif args.mode == "micro":
        num_epochs = 0.05
        print(f"✅ Micro test mode: {num_epochs} epochs (~35 min)")
    elif args.mode == "sanity":
        num_epochs = 0.3
        print(f"✅ Sanity check mode: {num_epochs} epochs (~4 hours)")
    else:
        num_epochs = 1.0
        print(f"✅ Full training mode: {num_epochs} epochs (~12 hours)")

    # ===================================================================
    # Training Mode Selection
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

            pusher_temp = PushAutomation(hf_token=HF_TOKEN, project_root=project_root)
            previous_metric = pusher_temp._get_previous_best_margin(HF_REPO, metric_name="reward")

            if previous_metric:
                print(f"   Previous performance: reward={previous_metric:.4f}")
        else:
            print(f"❌ No existing model on HuggingFace: {HF_REPO}")
            print(f"   This will be the first training run")
    except Exception as e:
        print(f"⚠️  Could not check HuggingFace: {type(e).__name__}")

    print(f"{'='*80}")
    time_estimates = {"micro": "~35 min", "sanity": "~4 hours", "full": "~12 hours"}
    print(f"Training will take approximately: {time_estimates.get(args.mode, 'varies')}")
    print(f"\nOptions:")
    if hf_model_exists:
        print(f"  1) Inference only from HF_repo (use existing HF model)")
        print(f"  2) Retrain and replace HF model (regardless of performance)")
    else:
        print(f"  1) Inference only from HF_repo (UNAVAILABLE - no model)")
        print(f"  2) Train and push to HuggingFace")
    print(f"{'='*80}")

    mode_choice = input("Enter choice (1 or 2): ").strip()
    print(f"{'='*80}\n")

    force_skip = False
    if mode_choice == "1":
        if not hf_model_exists:
            print("❌ Error: Option 1 requires existing HuggingFace model")
            print("   Please choose option 2 to train first")
            sys.exit(1)
        print("✅ Inference-only mode selected")
        force_skip = True
    elif mode_choice == "2":
        print("="*80)
        print("\n✅ Training mode selected")

        # Check for existing checkpoints and offer to delete (matching PPO behavior)
        checkpoint_dir = Path(f"./outputs/training/{RUN_NAME}")
        existing_checkpoints = list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []

        if existing_checkpoints:
            print(f"\n⚠️  Found {len(existing_checkpoints)} existing checkpoint(s) in {checkpoint_dir}:")
            for ckpt in sorted(existing_checkpoints)[:5]:  # Show first 5
                print(f"   - {ckpt.name}")
            if len(existing_checkpoints) > 5:
                print(f"   ... and {len(existing_checkpoints) - 5} more")

            print("\n🔄 Options:")
            print("   [R] Resume from latest checkpoint (default)")
            print("   [D] Delete ALL checkpoints and start fresh")
            delete_choice = input("\nEnter choice [R/D]: ").strip().upper()

            if delete_choice == "D":
                import shutil
                print(f"\n🗑️  Deleting all checkpoints in {checkpoint_dir}...")
                for ckpt in existing_checkpoints:
                    shutil.rmtree(ckpt)
                    print(f"   ✅ Deleted {ckpt.name}")
                print("✅ All checkpoints deleted. Starting fresh training.")
            else:
                print("✅ Will resume from latest checkpoint.")

        if hf_model_exists:
            print("   Will compare local vs HF metrics and push ONLY if performance improves")
        else:
            print("   Will train and push to HuggingFace (first time)")
        force_skip = False
    else:
        print("⚠️  Invalid choice, defaulting to training mode")
        force_skip = False

    # Determine training_mode for dynamic config
    if args.epochs is not None:
        training_mode = "custom"
    else:
        training_mode = args.mode

    # Run training
    try:
        trainer, training_skipped, training_config = train_grpo_baseline(
            num_epochs=num_epochs,
            base_model=args.base_model,
            force_skip=force_skip,
            training_mode=training_mode
        )
        print(f"\n🏁 GRPO Baseline Training Complete!")
        print(f"📝 Log file: {log_filename}")

        # ===================================================================
        # Push to HuggingFace (training_config returned from function)
        # ===================================================================

        PushAutomation.prepare_baseline_push(
            method="GRPO",
            output_dir=f"outputs/training/{RUN_NAME}",
            training_config=training_config,
            training_skipped=training_skipped,
            hf_token=HF_TOKEN,
            hf_repo=HF_REPO,
            run_name=RUN_NAME,
            metric_names=["reward", "rewards/mean"],
            metric_mode="max",
            project_root=project_root
        )

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"📝 Complete log file saved: {log_filename}")
