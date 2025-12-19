"""
PPO is known to be unstable (NaN losses common), which is why DPO was invented.
This baseline exists to show CITA beats traditional PPO.

Usage (6 commands = 3 modes × 2 instruction settings):

    # MICRO: 0.05 epochs (~30 min) - quick pipeline validation
    python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode micro --use-instruction false
    python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode micro --use-instruction true

    # SANITY: 0.3 epochs (~9 hours) - validate checkpoints/HF push
    python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction false
    python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction true

    # FULL: 1.0 epoch (~34 hours) - production training
    python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode full --use-instruction false
    python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode full --use-instruction true
    
  Projections:                                                                                                      
  | Mode                | Steps | ETA        |                                                                      
  |---------------------|-------|------------|                                                                      
  | Micro (0.05 epochs) | ~68   | ~1.7 hours |                                                                      
  | Sanity (0.3 epochs) | 405   | ~10 hours  |                                                                      
  | Full (1.0 epoch)    | 1353  | ~33 hours  |
"""

import sys
from pathlib import Path
import os
import argparse
from datetime import datetime
import gc

# ===== FIX CUDA OOM: Enable expandable segments for memory fragmentation =====
# PPO requires policy + value head + ref model + reward model → high memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

# ===== FIX torch.compile() CUDAGraph bug: Disable CUDAGraphs for dynamic shapes =====
# Fixes: "Expected curr_block->next == nullptr" error during eval with torch.compile()
# Warning showed 51 distinct input sizes → CUDAGraph memory allocator bug
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextStreamer

# ===== TRL PPO API DEPRECATION NOTICE =====
# Current import path: from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
# Will be DEPRECATED in TRL 0.29.0 and moved to:
#   from trl.experimental.ppo import PPOConfig, PPOTrainer
# Key parameter renames in NEW API:
#   - ppo_epochs → num_ppo_epochs
#   - init_kl_coef → kl_coef
#   - max_new_tokens → response_length
#   - mini_batch_size → num_mini_batches
# NEW API also supports: eval_dataset, missing_eos_penalty, stop_token, whiten_rewards
# See: https://huggingface.co/docs/trl/main/ppo_trainer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from peft import LoraConfig, PeftModel

# Add utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# ===================================================================
# Import Shared Utilities
# ===================================================================
from model_utils import (
    load_hf_token,
    get_model_repo_name,
    get_test_prompts,
    get_latest_checkpoint,
    is_training_complete,
    log_gpu_memory_start,
    log_gpu_memory_end,
    MODEL_NAME_MAP,
    BASE_MODEL_MAP
)
from data_prep.loader_pku import load_pku_combined_clear_contrast
from push_automation import PushAutomation
from logging_utils import setup_training_logger, restore_logging

# ===================================================================
# PPO-Specific Model Loading
# ===================================================================

def load_policy_model_with_value_head(
    model_id: str = "meta-llama/Llama-3.1-8B",
    max_seq_length: int = 2048,  # Match DPO for fair comparison
    use_flash_attention: bool = True
):
    """
    Load policy model with value head for PPO training

    Returns:
        Tuple of (policy_model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # PPO requires left padding for generation
    tokenizer.model_max_length = max_seq_length

    # Set Llama-3.1 chat template if not present
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and message['role'] != 'system' %}"
            "{{ '<|begin_of_text|>' }}"
            "{% endif %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )

    # LoRA config for PPO (same as DPO for fair comparison)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load model with value head
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "peft_config": lora_config,
    }

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"✅ Flash Attention 2 enabled")

    # AutoModelForCausalLMWithValueHead adds a value head on top of the LM
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        **model_kwargs
    )

    print(f"✅ Policy model loaded with value head: {model_id}")
    print(f"   - Precision: BF16")
    print(f"   - Max sequence length: {max_seq_length}")
    print(f"   - LoRA: r=16, alpha=16")

    return model, tokenizer


def load_reward_model(reward_model_id: str = "OpenAssistant/reward-model-deberta-v3-large-v2"):
    """
    Load off-the-shelf reward model for scoring responses

    Note: Using OpenAssistant reward model for simplicity.
    For production, you'd train a custom reward model on PKU-SafeRLHF.

    Returns:
        Tuple of (reward_model, reward_tokenizer)
    """
    print(f"\n📦 Loading reward model: {reward_model_id}")

    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"✅ Reward model loaded")

    return reward_model, reward_tokenizer


# ===================================================================
# Dataset Preparation for PPO
# ===================================================================

def prepare_ppo_dataset(tokenizer, use_instruction: bool = False, max_samples: int = None):
    """
    Prepare PKU-SafeRLHF dataset for PPO training

    PPO only needs queries (prompts), not preference pairs.
    The reward model scores the generated responses.

    TRL 0.11.4 requires dataset to have 'input_ids' column with tokenized sequences.
    See: https://huggingface.co/docs/trl/v0.11.4/en/ppo_trainer

    Returns:
        Dataset with 'input_ids' field containing tokenized prompts
    """
    from datasets import Dataset

    # Load PKU dataset
    dataset_split = load_pku_combined_clear_contrast(val_split=0.1)
    train_data = dataset_split['train']

    # Limit samples if specified
    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))

    # Extract and tokenize prompts
    # TRL 0.11.4 requires 'input_ids' column, not just text
    def extract_and_tokenize_prompt(example):
        prompt = example.get('prompt', '')

        # Add instruction prefix if enabled
        if use_instruction:
            instruction = (
                "You are a helpful, harmless, and honest AI assistant. "
                "If a request is harmful or unethical, politely refuse and explain why. "
                "Always prioritize safety while being as helpful as possible.\n\n"
            )
            prompt = instruction + prompt

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Tokenize the formatted prompt (TRL 0.11.4 requires input_ids)
        # Only return input_ids - text is decoded later for reward computation
        tokenized = tokenizer.encode(formatted)

        return {"input_ids": tokenized}

    # Format and tokenize dataset
    formatted_data = train_data.map(
        extract_and_tokenize_prompt,
        remove_columns=train_data.column_names,
        desc=f"Tokenizing PKU for PPO ({'WITH' if use_instruction else 'NO'} instruction)"
    )

    print(f"✅ Prepared {len(formatted_data):,} samples for PPO training")

    return formatted_data


# ===================================================================
# Reward Function
# ===================================================================

def create_reward_function(reward_model, reward_tokenizer, device):
    """
    Create a reward function that scores query-response pairs

    Returns:
        Function that takes (queries, responses) and returns rewards
    """
    def compute_rewards(queries, responses):
        """
        Compute rewards for query-response pairs

        Args:
            queries: List of query strings
            responses: List of response strings

        Returns:
            List of reward tensors
        """
        rewards = []

        for query, response in zip(queries, responses):
            # Combine query and response
            text = f"Query: {query}\nResponse: {response}"

            # Tokenize
            inputs = reward_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            # Get reward score
            with torch.no_grad():
                outputs = reward_model(**inputs)
                # Reward model outputs logits, take the positive class score
                reward = outputs.logits[0, 0].float()  # Single scalar reward

            rewards.append(reward)

        return rewards

    return compute_rewards


# ===================================================================
# Main Training Function
# ===================================================================

def train_ppo_baseline(
    num_epochs: float = 1.0,
    output_dir: str = None,
    base_model: str = None,
    force_skip: bool = False,
    training_mode: str = "full"
):
    """
    Train PPO baseline with epoch-based training (matching DPO for fair comparison)

    Args:
        num_epochs: Number of training epochs (default: 1.0 for full, 0.3 for sanity)
        output_dir: Output directory for checkpoints
        base_model: HuggingFace model ID to load LoRA adapters from (for stacking)
        force_skip: If True, skip training and only run inference
        training_mode: "micro" (128 tokens), "sanity" (128 tokens), "full" (256 tokens)

    Returns:
        trainer, training_skipped
    """
    if output_dir is None:
        output_dir = f"./outputs/{RUN_NAME}"

    print("\n" + "="*80)
    print(f"🚀 Starting {RUN_NAME} Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Model: Llama-3.1-8B (BF16)")
    print(f"  - Method: Proximal Policy Optimization (Schulman 2017)")
    print(f"  - Training epochs: {num_epochs}")
    print(f"  - Learning rate: 1e-5 (Meta's Llama 3 setting)")
    print(f"  - Precision: BF16 + Flash Attention 2")
    print("="*80 + "\n")

    # ===== CHECKPOINT DETECTION (BEFORE LOADING MODEL) =====
    print("\n" + "="*80)
    print("🔍 Checking for existing checkpoints...")
    print("="*80 + "\n")

    training_skipped = False
    latest_checkpoint = None
    load_from_hf = False  # Track if we should load from HF (option 1 only)

    if force_skip:
        print("🚫 User selected inference-only mode")
        print("   Skipping training, will load model from HuggingFace for inference...\n")
        training_skipped = True
        load_from_hf = True  # Option 1: Load from HF
        model = None
        tokenizer = None
    else:
        # Priority 1: Check local checkpoints
        print("1️⃣ Checking local checkpoints...")
        latest_checkpoint = get_latest_checkpoint(output_dir)
        resume_step = 0  # Track step to resume from

        if latest_checkpoint:
            print(f"📂 Found checkpoint: {latest_checkpoint}")
            # Note: Completion check will happen after loading dataset (need total_steps)
            # Extract step number from checkpoint path (e.g., checkpoint-500 → 500)
            import re
            match = re.search(r'checkpoint-(\d+)', str(latest_checkpoint))
            if match:
                resume_step = int(match.group(1))
            print(f"   Found checkpoint at step {resume_step}, will check completion after loading dataset...\n")
        else:
            print(f"🆕 No checkpoint found")
            print(f"   Will start fresh training (even if HF repo exists)...\n")
            latest_checkpoint = None

    if not training_skipped:
        # ===== LOAD POLICY MODEL =====
        print("\n📥 Loading policy model with value head...")
        model, tokenizer = load_policy_model_with_value_head(
            model_id="meta-llama/Llama-3.1-8B",
            max_seq_length=2048,  # Match DPO for fair comparison
            use_flash_attention=True
        )

        # ===== LOAD BASE MODEL LORA (IF STACKING) =====
        if base_model:
            print(f"\n🔗 Loading SFT LoRA from HuggingFace: {base_model}")
            # MERGE APPROACH with EXPLICIT REFERENCE MODEL (Option A)
            # 1. Load SFT LoRA and merge into base
            # 2. Create explicit copy of merged model for reference
            # 3. Add PPO LoRA to policy model only
            # 4. Use explicit ref_model instead of disable_adapter()
            from peft import PeftModel

            # The pretrained_model is the base LM inside AutoModelForCausalLMWithValueHead
            model.pretrained_model = PeftModel.from_pretrained(
                model.pretrained_model, base_model, token=HF_TOKEN
            )
            print("✅ SFT LoRA adapters loaded")

            # Merge SFT adapters into base model
            print("🔄 Merging SFT LoRA adapters into base model...")
            merged_pretrained = model.pretrained_model.merge_and_unload()

            # Clear PEFT config to avoid warnings
            try:
                delattr(merged_pretrained, 'peft_config')
            except AttributeError:
                pass
            try:
                delattr(merged_pretrained, '_hf_peft_config_loaded')
            except AttributeError:
                pass

            model.pretrained_model = merged_pretrained
            print("✅ SFT LoRA merged into base model")

            # Create reference model using TRL's official function
            # MUST be done AFTER SFT merge but BEFORE PPO LoRA
            # See: https://huggingface.co/docs/trl/en/models#trl.create_reference_model
            print("\n📋 Creating reference model (TRL create_reference_model)...")
            from trl import create_reference_model
            ref_model_for_ppo = create_reference_model(model)
            print("✅ Reference model created (frozen copy, shares layers for memory efficiency)")

            # Apply NEW LoRA adapters for PPO training (only to policy model)
            print("\n🔧 Applying new LoRA adapters for PPO training (policy model only)...")
            ppo_lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            from peft import get_peft_model
            model.pretrained_model = get_peft_model(model.pretrained_model, ppo_lora_config)
            print("✅ PPO LoRA adapters applied to policy model")

            # Set is_peft_model flag (for PPOTrainer to recognize PEFT)
            model.is_peft_model = True
            print("✅ Merge approach complete: Policy=SFT+PPO, Reference=SFT(frozen)")

        # ===== LOAD REWARD MODEL =====
        print("\n📥 Loading reward model...")
        reward_model, reward_tokenizer = load_reward_model()
        device = next(reward_model.parameters()).device

        # ===== PREPARE DATASET =====
        print("\n📊 Preparing PPO dataset...")
        ppo_dataset = prepare_ppo_dataset(
            tokenizer,
            use_instruction=USE_INSTRUCTION,
            max_samples=None  # Use full dataset (epoch-based training)
        )

        # ===== CALCULATE TRAINING STEPS (epoch-based, matching DPO) =====
        effective_batch_size = 1 * 8  # per_device=1, grad_accum=8 (match DPO)
        steps_per_epoch = len(ppo_dataset) // effective_batch_size
        total_steps = int(steps_per_epoch * num_epochs)
        checkpoint_interval = max(1, total_steps // 5)  # Save/eval every 20%

        print(f"\n📊 Training Configuration:")
        print(f"   Dataset size: {len(ppo_dataset):,} samples")
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Total steps: {total_steps:,} ({num_epochs} epochs)")
        print(f"   Checkpoint interval: {checkpoint_interval} steps (20% of training)")

        # ===== CHECK TRAINING COMPLETION (now that we have total_steps) =====
        if latest_checkpoint and is_training_complete(latest_checkpoint, total_steps):
            print(f"\n✅ Training already completed at: {latest_checkpoint}")
            print(f"   Total steps: {total_steps} ({num_epochs} epochs)")
            print(f"   Skipping training, will load from local checkpoint for inference...\n")
            training_skipped = True
            load_from_hf = False

    # Skip rest of training setup if training already complete
    if not training_skipped:
        # ===== TENSORBOARD SETUP =====
        tensorboard_base_dir = project_root / "tensorboard_logs"
        tensorboard_base_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_run_dir = tensorboard_base_dir / f"{RUN_NAME}_{timestamp}"

        print(f"📊 TensorBoard logs: {tensorboard_run_dir}")

        # ===== PPO CONFIG =====
        # Note: PPOConfig inherits from TrainingArguments
        # Unlike DPOTrainer, PPOTrainer doesn't support native eval_dataset
        # Validation is done by monitoring reward metrics during training
        # TRL 0.11.4 PPOConfig - only supports PPO-specific parameters
        # (not full TrainingArguments like newer versions)
        ppo_config = PPOConfig(
            # Training - TRL 0.11.4 constraint: batch_size >= mini_batch_size * gradient_accumulation_steps
            # EXPLICIT REF_MODEL CONFIG: Dual-model memory (policy + reference)
            learning_rate=1e-5,  # Meta's official Llama 3 setting
            # --- A100-40GB settings (commented out) ---
            # batch_size=8,  # Total experiences before PPO update (16→8 for dual-model memory)
            # mini_batch_size=2,  # Process 2 samples in parallel (4→2 for dual-model memory)
            # gradient_accumulation_steps=4,  # Accumulate gradients (8 >= 2*4 ✓)
            # --- A100-80GB settings (active) ---
            batch_size=16,  # Total experiences before PPO update (2x memory headroom)
            mini_batch_size=4,  # Process 4 samples in parallel (2x from 40GB)
            gradient_accumulation_steps=4,  # Accumulate gradients (16 >= 4*4 ✓)
            ppo_epochs=4,  # PPO epochs per batch

            # PPO-specific
            init_kl_coef=0.1,  # Initial KL penalty coefficient
            target_kl=0.1,  # Target KL divergence
            kl_penalty="full",  # FIX: 'full' KL prevents negative values (GitHub issue #1017)
            cliprange=0.2,  # PPO clipping parameter
            cliprange_value=0.2,  # Value function clipping
            gamma=1.0,  # Discount factor (1.0 for single-turn)
            lam=0.95,  # GAE lambda

            # Logging
            log_with="tensorboard",
            project_kwargs={"logging_dir": str(tensorboard_run_dir)},

            # Memory optimization
            gradient_checkpointing=True,  # CRITICAL: Reduce memory usage
            max_grad_norm=1.0,  # Gradient clipping (match CITA, prevent policy divergence)

            # Seed
            seed=3407,
        )

        # Print actual PPO config values (dynamic, not hardcoded)
        print(f"📦 PPO Config: batch_size={ppo_config.batch_size}, mini_batch_size={ppo_config.mini_batch_size}, "
              f"grad_accum={ppo_config.gradient_accumulation_steps}, ppo_epochs={ppo_config.ppo_epochs}")

        # ===== GENERATION KWARGS (separate from PPOConfig) =====
        # TRL recommended settings to prevent negative KL divergence:
        # - top_p=1.0 (not 0.9) - truncated sampling causes KL issues
        # - top_k=0.0 - disable top-k sampling
        # - min_length=-1 - don't force minimum length
        # See: https://github.com/huggingface/trl/issues/235
        # max_new_tokens: 128 for micro/sanity (faster), 256 for full (quality)
        max_new_tokens = 256 if training_mode == "full" else 128
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_length": -1,  # TRL recommended: don't force min length
            "top_k": 0.0,  # TRL recommended: disable top-k
            "top_p": 1.0,  # TRL recommended: full distribution (not 0.9!)
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        print(f"📦 Generation: max_new_tokens={max_new_tokens} (mode={training_mode})")

        # ===== TORCH.COMPILE() OPTIMIZATION =====
        # DISABLED: Same as DPO - causes issues with gradient checkpointing
        # torch.compile() + gradient_checkpointing has compatibility issues
        # Trade-off: ~10% slower training vs no crash
        print("\n⚠️  torch.compile() disabled (prevents gradient checkpointing bug)")

        # ===== CREATE REWARD FUNCTION =====
        compute_rewards = create_reward_function(reward_model, reward_tokenizer, device)

        # ===== DATA COLLATOR FOR VARIABLE-LENGTH SEQUENCES =====
        # TRL 0.11.4 needs a collator to pad variable-length input_ids in each batch
        from transformers import DataCollatorWithPadding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        # ===== CREATE PPO TRAINER =====
        print("\n🏗️  Initializing PPOTrainer with explicit reference model...")
        # Using explicit ref_model (frozen copy of merged SFT) instead of disable_adapter()
        # Policy model = SFT (merged) + PPO LoRA
        # Reference model = SFT (merged, frozen)
        # This approach works even though SFT is merged (not PEFT adapter)
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model_for_ppo,  # Explicit frozen reference (merged SFT)
            tokenizer=tokenizer,
            dataset=ppo_dataset,
            data_collator=data_collator,  # Handle variable-length sequences
        )

        # ===== GPU MEMORY =====
        start_gpu_memory = log_gpu_memory_start()

        # ===== RESUME FROM CHECKPOINT (if applicable) =====
        if latest_checkpoint and resume_step > 0:
            print(f"\n📂 Resuming from checkpoint: {latest_checkpoint}")
            print(f"   Skipping first {resume_step} steps...")
            # Load saved model state from checkpoint
            try:
                from peft import PeftModel
                model.pretrained_model = PeftModel.from_pretrained(
                    model.pretrained_model, str(latest_checkpoint)
                )
                print(f"✅ Loaded model weights from checkpoint")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint weights: {e}")
                print(f"   Starting fresh training...")
                resume_step = 0

        # ===== TRAINING LOOP =====
        print("\n" + "="*80)
        print("🏋️  Training PPO Baseline...")
        if resume_step > 0:
            print(f"   Resuming from step {resume_step}")
        print("="*80 + "\n")

        # Track metrics for summary (equivalent to TrainingSummaryCallback)
        reward_history = []
        kl_history = []
        policy_loss_history = []
        value_loss_history = []
        best_reward = float('-inf')  # Track best reward for push comparison

        # Persistent log_history for trainer_state.json (load from previous checkpoint if resuming)
        import json
        log_history = []
        if latest_checkpoint:
            prev_state_path = Path(latest_checkpoint) / "trainer_state.json"
            if prev_state_path.exists():
                with open(prev_state_path, 'r') as f:
                    prev_state = json.load(f)
                    log_history = prev_state.get('log_history', [])
                    best_reward = prev_state.get('best_metric', float('-inf'))
                print(f"📂 Loaded {len(log_history)} log entries from previous checkpoint")

        # PPO training loop (epoch-based, matching DPO)
        for step, batch in enumerate(ppo_trainer.dataloader):
            # Skip steps if resuming
            if step < resume_step:
                continue

            if step >= total_steps:
                break

            # TRL 0.11.4 requires list of 1D tensors, not 2D batch tensor
            query_tensors = [q for q in batch["input_ids"]]

            # Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs
            )

            # Decode for reward computation
            queries = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
            responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute rewards
            rewards = compute_rewards(queries, responses)

            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            # Track metrics
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            reward_history.append(mean_reward)
            kl_history.append(stats.get("objective/kl", 0))
            policy_loss_history.append(stats.get("ppo/loss/policy", 0))
            value_loss_history.append(stats.get("ppo/loss/value", 0))

            # Track best reward for push comparison (equivalent to trainer.state.best_metric)
            if mean_reward > best_reward:
                best_reward = mean_reward

            # Regular logging (every 10 steps)
            if step % 10 == 0:
                kl = stats.get("objective/kl", 0)
                print(f"Step {step}/{total_steps} | Reward: {mean_reward:.3f} | KL: {kl:.4f}")

                # Append to persistent log_history
                log_history.append({
                    "step": step,
                    "objective/scores": mean_reward,
                    "objective/kl": kl,
                    "ppo/mean_scores": mean_reward,
                })

            # ===== TRAINING SUMMARY (every 50 steps) =====
            # Inline equivalent of TrainingSummaryCallback for PPO
            if step > 0 and step % 50 == 0:
                print(f"\n{'='*80}")
                print(f"📊 PPO TRAINING SUMMARY - Step {step}")
                print(f"{'='*80}")

                # Recent window (last 50 batches)
                window = min(50, len(reward_history))
                recent_rewards = reward_history[-window:]
                recent_kl = kl_history[-window:]
                recent_policy_loss = policy_loss_history[-window:]
                recent_value_loss = value_loss_history[-window:]

                # Reward trajectory
                print(f"\nREWARD trajectory (last {window} batches):")
                print(f"  Current: {recent_rewards[-1]:.4f}")
                print(f"  Average: {sum(recent_rewards)/len(recent_rewards):.4f}")
                print(f"  Min: {min(recent_rewards):.4f}")
                print(f"  Max: {max(recent_rewards):.4f}")

                # Trend analysis
                if len(recent_rewards) > 10:
                    first_half = sum(recent_rewards[:len(recent_rewards)//2]) / (len(recent_rewards)//2)
                    second_half = sum(recent_rewards[len(recent_rewards)//2:]) / (len(recent_rewards) - len(recent_rewards)//2)
                    trend = "↑ INCREASING" if second_half > first_half else "↓ DECREASING"
                    print(f"  Trend: {trend} (first half: {first_half:.4f}, second half: {second_half:.4f})")

                # KL divergence
                avg_kl = sum(recent_kl) / len(recent_kl)
                print(f"\nKL DIVERGENCE (policy vs reference):")
                print(f"  Current: {recent_kl[-1]:.4f}")
                print(f"  Average: {avg_kl:.4f}")
                if avg_kl > 0.3:
                    print(f"  ⚠️  WARNING: High KL = policy drifting from reference!")

                # Policy loss
                if recent_policy_loss:
                    print(f"\nPOLICY LOSS:")
                    print(f"  Current: {recent_policy_loss[-1]:.4f}")
                    print(f"  Average: {sum(recent_policy_loss)/len(recent_policy_loss):.4f}")

                # Value loss
                if recent_value_loss:
                    print(f"\nVALUE LOSS:")
                    print(f"  Current: {recent_value_loss[-1]:.4f}")
                    print(f"  Average: {sum(recent_value_loss)/len(recent_value_loss):.4f}")

                print(f"{'='*80}\n")

            # Save checkpoint periodically (every 20% of training)
            if (step + 1) % checkpoint_interval == 0 or (step + 1) == total_steps:
                checkpoint_dir = Path(output_dir) / f"checkpoint-{step + 1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n💾 Saving checkpoint at step {step + 1}/{total_steps}...")
                ppo_trainer.save_pretrained(str(checkpoint_dir))
                tokenizer.save_pretrained(str(checkpoint_dir))

                # Create trainer_state.json for push_automation compatibility
                # PPOTrainer doesn't create this, but push_automation needs it
                trainer_state = {
                    "best_metric": best_reward,
                    "global_step": step + 1,
                    "log_history": log_history  # Use accumulated log_history
                }
                with open(checkpoint_dir / "trainer_state.json", "w") as f:
                    json.dump(trainer_state, f, indent=2)

                print(f"✅ Checkpoint saved to: {checkpoint_dir}\n")

            # Clear cache periodically
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # ===== FINAL MEMORY =====
        log_gpu_memory_end(start_gpu_memory)

        # ===== TRAINING SUMMARY =====
        print(f"\n✅ PPO training complete!")
        print(f"   Total steps: {total_steps} ({num_epochs} epochs)")
        print(f"   Best reward: {best_reward:.4f}")
        print(f"   Final reward (avg last 50): {sum(reward_history[-50:])/min(50, len(reward_history)):.4f}")

        # Store best_metric for push_automation (equivalent to trainer.state.best_metric)
        # PPO uses reward (higher is better), unlike DPO which uses margin
        ppo_trainer.best_metric = best_reward

    # ===== SAVE MODEL =====
    lora_output_dir = Path(output_dir) / f"lora_model_{RUN_NAME}"

    if not training_skipped:
        lora_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving model to: {lora_output_dir}")
        ppo_trainer.save_pretrained(str(lora_output_dir))
        tokenizer.save_pretrained(str(lora_output_dir))
        print(f"✅ Model saved!")

    # ===== INFERENCE TEST =====
    print("\n" + "="*80)
    print("🧪 Running inference tests...")
    print("="*80 + "\n")

    if training_skipped:
        if load_from_hf:
            # Option 1 (inference-only mode) - download model from HF for inference
            print(f"📥 Downloading model from HuggingFace for inference: {HF_REPO}")
            model, tokenizer = load_policy_model_with_value_head(
                model_id="meta-llama/Llama-3.1-8B",
                max_seq_length=2048,  # Match DPO
                use_flash_attention=True
            )
            # Load LoRA adapters from HF
            try:
                from peft import PeftModel
                # Note: AutoModelForCausalLMWithValueHead may need special handling
                print(f"🔧 Loading LoRA adapters from HF: {HF_REPO}")
                model.pretrained_model = PeftModel.from_pretrained(
                    model.pretrained_model, HF_REPO, token=HF_TOKEN
                )
                print(f"✅ Model downloaded from HuggingFace")
            except Exception as e:
                print(f"⚠️  Could not load LoRA from HF: {e}")
                print(f"   Using base model for inference")
        else:
            # Option 2 with local checkpoint - load from local checkpoint for inference
            print(f"📂 Loading model from local checkpoint for inference: {latest_checkpoint}")
            model, tokenizer = load_policy_model_with_value_head(
                model_id="meta-llama/Llama-3.1-8B",
                max_seq_length=2048,  # Match DPO
                use_flash_attention=True
            )
            # Load LoRA adapters from local checkpoint
            try:
                checkpoint_lora_path = Path(latest_checkpoint)
                model.pretrained_model = PeftModel.from_pretrained(
                    model.pretrained_model, str(checkpoint_lora_path)
                )
                print(f"✅ Model loaded from local checkpoint")
            except Exception as e:
                print(f"⚠️  Could not load LoRA from checkpoint: {e}")
                print(f"   Using base model for inference")

    # Prepare model for inference
    model.eval()

    # Ensure model is in bf16 for Flash Attention compatibility
    model = model.to(torch.bfloat16)

    # Disable gradient checkpointing for inference (not needed, can cause issues)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    elif hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'gradient_checkpointing_disable'):
        model.pretrained_model.gradient_checkpointing_disable()

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

    return ppo_trainer if not training_skipped else None, training_skipped


# ===================================================================
# Main Execution
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO Baseline Training - Proximal Policy Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NoInstruct variant (sanity check, 0.3 epochs, ~31 minutes)
  python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode sanity --use-instruction false

  # Instruct variant (full training, 1.0 epoch, ~103 minutes)
  python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --mode full --use-instruction true

  # Custom epochs
  python comparative_study/02b_PPO_Baseline/Llama3_BF16.py --epochs 0.5 --use-instruction false
        """
    )

    parser.add_argument(
        "--use-instruction",
        type=str,
        required=True,
        choices=["true", "false"],
        help="REQUIRED: Use instruction conditioning (true=PPO_Instruct, false=PPO_NoInstruct)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["micro", "sanity", "full"],
        default="full",
        help="Training mode: 'micro' (0.05 epochs, ~30min validation), 'sanity' (0.3 epochs), or 'full' (1.0 epochs)"
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
    RUN_NAME = "PPO_Instruct" if USE_INSTRUCTION else "PPO_NoInstruct"

    # Auto-derive base_model from BASE_MODEL_MAP if not provided
    if args.base_model is None:
        args.base_model = BASE_MODEL_MAP[RUN_NAME]
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

    # Get HuggingFace repository name (PPO now in MODEL_NAME_MAP)
    HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

    print(f"📦 Model will be pushed to: {HF_REPO}")
    print("="*80 + "\n")

    # Determine configuration (epoch-based, matching DPO)
    if args.epochs is not None:
        num_epochs = args.epochs
        training_mode = "custom"
        print(f"✅ Custom configuration: {num_epochs} epochs")
    elif args.mode == "micro":
        num_epochs = 0.05  # ~50 steps, ~30 min - quick validation of pipeline
        training_mode = "micro"
        print(f"✅ Micro validation mode: {num_epochs} epochs (~30 min, ~50 steps)")
    elif args.mode == "sanity":
        num_epochs = 0.3  # 30% of data (~405 steps)
        training_mode = "sanity"
        print(f"✅ Sanity check mode: {num_epochs} epochs (~9 hours)")
    else:
        num_epochs = 1.0  # Full epoch (~1353 steps)
        training_mode = "full"
        print(f"✅ Full training mode: {num_epochs} epochs (~34 hours)")

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

            # Try to get previous metric
            pusher_temp = PushAutomation(hf_token=HF_TOKEN, project_root=project_root)
            previous_metric = pusher_temp._get_previous_best_margin(HF_REPO, metric_name="objective/scores")

            if previous_metric:
                print(f"   Previous performance: reward={previous_metric:.4f}")
        else:
            print(f"❌ No existing model on HuggingFace: {HF_REPO}")
            print(f"   This will be the first training run")
    except Exception as e:
        print(f"⚠️  Could not check HuggingFace: {type(e).__name__}")

    print(f"{'='*80}")
    time_estimates = {"micro": "~30 min", "sanity": "~9 hours", "full": "~34 hours", "custom": "varies"}
    print(f"Training will take approximately: {time_estimates.get(training_mode, 'varies')}")
    print(f"\nOptions:")
    if hf_model_exists:
        print(f"  1) Inference only from HF_repo (use existing HF model)")
        print(f"  2) Retrain and replace HF model (only if performance improves)")
    else:
        print(f"  1) Inference only from HF_repo (UNAVAILABLE - no model)")
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
        print("="*80)
        print("\n✅ Training mode selected")

        # Check for existing checkpoints and offer to delete
        checkpoint_dir = Path(f"./outputs/{RUN_NAME}")
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

    # Run training
    try:
        trainer, training_skipped = train_ppo_baseline(
            num_epochs=num_epochs,
            base_model=args.base_model,
            force_skip=force_skip,
            training_mode=training_mode
        )
        print(f"\n🏁 PPO Baseline Training Complete!")
        print(f"📝 Log file: {log_filename}")

        # ===================================================================
        # Push to HuggingFace
        # ===================================================================
        training_config = {
            "method": "PPO",
            "num_epochs": num_epochs,  # Epoch-based (matching DPO)
            "learning_rate": 1e-5,  # Meta's official Llama 3 setting
            "warmup_steps": 100,  # Same as DPO (from iter4 successful run)
            "optimizer": "adamw_torch",  # Match DPO
            "weight_decay": 0.01,  # Same as DPO
            "lr_scheduler_type": "cosine",  # Same as DPO (cosine for smoother convergence)
            "batch_size": 8,  # Experiences per PPO update (effective batch=8, match DPO)
            "gradient_accumulation_steps": 1,  # No accumulation (batch_size handles it)
            "mini_batch_size": 1,  # Memory efficient
            "ppo_epochs": 4,
            "init_kl_coef": 0.1,
            "target_kl": 0.1,
            "cliprange": 0.2,
            "max_seq_length": 2048,  # Match DPO for fair comparison
            "max_new_tokens": 128,
            "reward_model": "OpenAssistant/reward-model-deberta-v3-large-v2",
        }

        # Note: PPO uses reward instead of margins
        PushAutomation.prepare_baseline_push(
            method="PPO",
            output_dir=f"outputs/{RUN_NAME}",
            training_config=training_config,
            training_skipped=training_skipped,
            hf_token=HF_TOKEN,
            hf_repo=HF_REPO,
            run_name=RUN_NAME,
            metric_names=["objective/scores", "ppo/mean_scores"],  # PPO reward metrics
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
