"""
PBT Training Script for CITA (with Safeguards)
Population-Based Training with 4 workers (optimized for A100-40GB)

Configuration:
- 4 workers total (2 RUNNING in parallel, 2 PAUSED at any time)
- GPU allocation: 0.5 per worker (50% GPU each)
- Memory: 2 × 18GB = 36GB / 40GB (4GB buffer)
  * Per worker: Base 16GB + LoRA 0.2GB + Gradients 2GB = ~18GB
  * PEFT + ref_model=None: Memory-efficient (toggles adapters, NO deepcopy)
- 5 hyperparameters: λ_kl, LR, β, weight_decay, warmup_steps
- mutation_interval=50 steps (exploit/explore every 50 steps, aligned with checkpoints)
- checkpoint_interval=50 steps (aligned with safety checks for faster recovery)
- Total training: 1000 steps per worker (20 checkpoints)
- Expected time: ~120 minutes on A100-40GB (4 workers sequentially = 2× time)
- Expected cost: ~$3.00 (120 min × $1.5/hr)

Safeguards (prevents mode collapse from PBT Experiment Report):
- ✅ PBT metric: cita/margin (maximize positive margin)
- ✅ Margin safety: Stops training if margin becomes negative (model prefers unsafe)
- ✅ Gibberish monitoring: Every 50 steps (aligned with checkpoints)
- ✅ Early stopping: Enabled (stops on gibberish OR negative margin)
- ✅ Global abort: Stops experiment if ALL workers fail (prevents pushing unsafe models)

Usage:
    # Standard (with line buffering):
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py

    # Unbuffered output (recommended for real-time logging):
    python -u comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py

Outputs:
    - Ray Tune results: ./outputs/ray_results/cita_pbt_training/
    - Best hyperparameters: ./outputs/best_pbt_config.json
    - Best model checkpoint saved
    - Training log: ./logs/CITA_PBT_training_<timestamp>.log
    
🚀 How to Use

    - Option A: Sanity Check (1 worker, 100 steps, ~3 minutes)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py --mode sanity

    - Option B: Full PBT (4 workers, 1000 steps, ~120 minutes)
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py --mode full

    Custom Configuration

    # Example: 2 workers, 500 steps
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py --workers 2 --steps 500

    #Help
    python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py --help

Logging:
    All terminal output (stdout + stderr) is automatically saved to:
    ./logs/CITA_PBT_training_<timestamp>.log

    The log file captures everything you see on terminal, just like:
    python script.py | tee log_file.log
"""

import sys
from pathlib import Path
import os
import argparse  # ✅ For command-line test modes
from datetime import datetime

# ===== FIX CUDA OOM: Enable expandable segments for memory fragmentation =====
# CITA (like DPO) requires trainable model + reference model, causing fragmentation
# MUST be set BEFORE importing torch/transformers/ray
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ===== FIX TOKENIZER FORK DEADLOCK: Prevents crash-and-restart cycles =====
# Without this: Training crashes after each checkpoint (step 50, 100, 150...)
# Result: 4 worker restarts, 7x slower training (44 min instead of 6 min)
# With this: Single worker completes all steps smoothly
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import ray
from ray import tune
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
    run_name="CITA_Baseline",
    project_root=project_root
)

# ===================================================================
# HuggingFace Configuration (for auto-push after PBT)
# ===================================================================
from model_utils import load_hf_token, get_model_repo_name, check_ray_tune_experiment

# Load HuggingFace token and authenticate
HF_TOKEN = load_hf_token(project_root)

# Get HuggingFace repository name
RUN_NAME = "CITA_Baseline"
HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

print(f"📦 Model will be pushed to: {HF_REPO}")
print("="*80 + "\n")

from pbt_trainer import (
    create_pbt_scheduler,
    run_pbt_training,
    print_best_hyperparameters,
    save_best_config
)
from push_automation import PushAutomation

# Ray Train integration for HuggingFace Transformers
from ray.train.huggingface.transformers import RayTrainReportCallback


# ===== PBT CONFIGURATION CONSTANTS =====
# Single source of truth for checkpoint/safety check alignment
CHECK_EVERY_N_STEPS = 50  # Checkpoint interval = Safety check interval (faster iteration for research)


# ===== CITA-SPECIFIC HYPERPARAMETER SPACE =====

def get_cita_hp_space(max_steps=1000, base_model=None):
    """
    Hyperparameter search space for CITA training

    Args:
        max_steps: Maximum training steps (default: 1000 for full PBT, 100 for sanity check)
        base_model: HuggingFace model ID to load LoRA adapters from (for stacking SFT→DPO→CITA)

    Returns:
        Dict with static config + PBT-tuned hyperparameters
    """
    config = {
        # ===== STATIC TRAINING CONFIG =====
        # ✅ SAFEGUARD: Batch=1 for max stability (prevents mode collapse per PBT_Experiment_Report.md)
        "per_device_train_batch_size": 1,  # ✅ SAFEST: 1 (max stability, user confirmed)
        "gradient_accumulation_steps": 8,  # ✅ MAINTAINS: effective batch=8
        "max_steps": max_steps,  # ✅ Configurable via command-line
        "lr_scheduler_type": "cosine",  # ⚠️ CHANGED: cosine vs notebook's linear (better for alignment)
        "save_steps": CHECK_EVERY_N_STEPS,  # ✅ ALIGNED: checkpoint = safety check interval
        "save_total_limit": 5,  # ✅ SAME as notebook
        "logging_steps": 1,  # ✅ SAME as notebook
        "eval_strategy": "steps",  # ✅ ADDED: Validation to detect overfitting
        "eval_steps": CHECK_EVERY_N_STEPS,  # ✅ ALIGNED: eval = checkpoint = safety check (50 steps)
        "per_device_eval_batch_size": 1,  # ✅ Match training batch size
        "gradient_checkpointing": True,  # ✅ SAME as notebook (via prepare_model_for_kbit_training)
        "bf16": True,  # ✅ SAME as notebook
        "optim": "adamw_torch",  # ✅ SAME as notebook
        "report_to": "none",  # ⚠️ CHANGED: none vs notebook's tensorboard (Ray Tune has own tracking)

        # ===== PBT-TUNED HYPERPARAMETERS (CONSTRAINED) =====
        # ✅ SAFEGUARD: All ranges constrained to ±10-20% of working baseline
        # Prevents mode collapse per PBT_Experiment_Report.md (Lines 135-141)
        #
        # 🎯 DESIGN CHOICE: Narrow Ranges (Response to Devil's Advocate Issue #3)
        # Devil's Advocate: "PBT needs WIDE exploration to find optima. These constraints defeat PBT's purpose."
        #
        # Our Justification (INTENTIONAL DESIGN):
        # 1. Previous PBT run with WIDE ranges (5× for LR) → ALL workers gibberish from iteration 0
        # 2. CITA safety alignment is SAFETY-CRITICAL (unlike typical hyperparameter search)
        # 3. Baseline hyperparameters (lr=2e-5, beta=0.1) are KNOWN to work (from successful notebook run)
        # 4. Goal: Fine-tune around working baseline, NOT explore random space
        # 5. Trade-off: Stability > Exploration (prevent catastrophic mode collapse)
        #
        # Philosophy: For safety-critical training, constrained PBT > wide exploration
        # If narrow ranges fail to improve, we still have working baseline (not gibberish)
        #
        # Evidence from PBT_Experiment_Report.md:
        # - Wide LR range [1e-5, 5e-5]: All workers failed (gibberish)
        # - Wide lambda_kl [0.0005, 0.002]: Too-low values caused unsafe preference
        # - Wide warmup [50, 150]: Too-short warmup caused training instability
        #
        # ✅ This is a FEATURE, not a limitation!

        # λ_kl: Floor=baseline (prevent too-low values)
        "lambda_kl": tune.uniform(0.001, 0.0015),  # ✅ [0.001, 0.0015] (baseline=0.001, +0% to +50%)

        # LR: ±20% around Meta's 1e-5 DPO baseline
        "learning_rate": tune.uniform(8e-6, 1.2e-5),  # ✅ FIXED: [8e-6, 1.2e-5] (±20% around Meta's 1e-5)

        # β: ±20% around Meta's 0.1 baseline
        "beta": tune.uniform(0.08, 0.12),  # ✅ [0.08, 0.12] (±20% around 0.1, matches Meta)

        # weight_decay: ±20% around baseline 0.01
        "weight_decay": tune.uniform(0.008, 0.012),  # ✅ [0.008, 0.012] (±20%)

        # warmup_steps: Min=100 (prevent too-short warmup)
        "warmup_steps": tune.randint(100, 120),  # ✅ [100, 120) (baseline=100, min enforced)
    }

    # Add base_model if provided (for stacking SFT→DPO→CITA)
    if base_model:
        config["base_model"] = base_model

    return config


def get_cita_hyperparam_mutations():
    """
    Hyperparameter mutations for PBT scheduler
    Only include parameters that will be tuned

    ✅ SAFEGUARD: Constrained to match get_cita_hp_space() ranges
    Prevents mode collapse per PBT_Experiment_Report.md

    Returns:
        Dict of hyperparameters for PBT to mutate
    """
    return {
        "lambda_kl": tune.uniform(0.001, 0.0015),  # ✅ Constrained (baseline floor)
        "learning_rate": tune.uniform(8e-6, 1.2e-5),  # ✅ FIXED: Constrained (±20% around Meta's 1e-5)
        "beta": tune.uniform(0.08, 0.12),  # ✅ Constrained (±20% around Meta's 0.1)
        "weight_decay": tune.uniform(0.008, 0.012),  # ✅ Constrained (±20%)
        "warmup_steps": tune.randint(100, 120),  # ✅ Constrained (min=100)
    }


# ===== CITA-SPECIFIC TRAINING FUNCTION =====

def train_cita_with_pbt(config, checkpoint_dir=None):
    """
    Training function for CITA with PBT
    Called by Ray Tune for each worker

    Args:
        config: Hyperparameters from Ray Tune
        checkpoint_dir: Path to checkpoint (if resuming)
    """
    # ===== SETUP PYTHON PATH FOR RAY WORKERS =====
    # Each Ray worker needs to set up its own path to import custom modules
    import sys
    from pathlib import Path
    worker_project_root = Path(__file__).parent.parent.parent
    utils_path = str(worker_project_root / "comparative_study" / "0c_utils")
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)

    # Import custom modules (after path setup)
    from data_prep import load_pku_filtered, format_dataset
    from cita_trainer import CITATrainer
    from monitoring_callback import GibberishDetectionCallback
    from ray.train.huggingface.transformers import RayTrainReportCallback

    # ===== LOAD MODEL & TOKENIZER =====
    # Use model_utils for consistent loading across all training scripts
    from model_utils import load_model_bf16

    model, tokenizer = load_model_bf16(
        model_id="meta-llama/Llama-3.1-8B",
        max_seq_length=2048,  # ✅ Match SFT/DPO baselines for fair comparison
        use_flash_attention=True  # ✅ Saves ~2.5GB/worker memory
    )

    # ===== LOAD BASE MODEL LORA (IF STACKING) =====
    base_model = config.get("base_model", None)
    if base_model:
        print(f"\n🔗 Loading LoRA adapters from HuggingFace: {base_model}")
        from peft import PeftModel
        from model_utils import load_hf_token

        # Load HF token for authentication
        hf_token = load_hf_token(worker_project_root)

        model = PeftModel.from_pretrained(model, base_model, token=hf_token)
        print("✅ LoRA adapters loaded")

        # Merge adapters into base model
        print("🔄 Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
        print("✅ LoRA adapters merged (ready for new training stage)")

    # ===== APPLY LORA ADAPTERS =====
    # (CRITICAL: Without LoRA, training 8B params would take DAYS!)
    from model_utils import setup_lora

    model = setup_lora(
        model,
        r=16,
        lora_alpha=16,
        use_gradient_checkpointing=True
    )

    # ===== TORCH.COMPILE() OPTIMIZATION (10-20% speedup) =====
    # Updated: torch.compile() + gradient_checkpointing ARE COMPATIBLE in PyTorch 2.4+
    # Memory Budget API enables selective recomputation with both features
    from model_utils import apply_torch_compile

    model = apply_torch_compile(model)

    # ===== LOAD DATASET =====
    # load_pku_filtered() automatically loads PKU-SafeRLHF and filters for safety contrast
    dataset_raw_train = load_pku_filtered(
        split="train",  # ✅ Loads train split (filters for clear safety contrast)
        max_samples=None,  # ✅ Use all samples (~10,813 after filtering)
        return_val=False  # Training split
    )

    dataset_raw_val = load_pku_filtered(
        split="train",
        max_samples=None,
        return_val=True  # Validation split (10% by default)
    )

    # ===== FORMAT DATASET =====
    # Format dataset in DPO format (Format 4.3 EBA from proposal)
    # CRITICAL: DPOTrainer REQUIRES separate prompt/chosen/rejected fields
    # DO NOT convert to contrastive text (Format 4.2) - that's for SFTTrainer only
    train_dataset = format_dataset(dataset_raw_train, method="cita")
    val_dataset = format_dataset(dataset_raw_val, method="cita")

    # Dataset structure after formatting:
    # {
    #     "prompt": [{"role": "system", ...}, {"role": "user", ...}],
    #     "chosen": [{"role": "assistant", "content": "safe response"}],
    #     "rejected": [{"role": "assistant", "content": "unsafe response"}]
    # }
    # This is Format 4.3 (EBA) - exactly what DPOTrainer.concatenated_forward() expects

    # ===== TEST PROMPTS FOR GIBBERISH MONITORING =====
    # ✅ SAME as inference_bf16.py (Lines 60-110) - All 7 test cases
    from model_utils import get_test_prompts
    test_prompts = get_test_prompts()

    # ===== EXTRACT CITA-SPECIFIC PARAMS =====
    lambda_kl = config.pop("lambda_kl")
    beta = config.pop("beta")

    # ===== EXTRACT NON-DPOCONFIG PARAMS =====
    # Remove parameters that shouldn't be passed to DPOConfig
    base_model = config.pop("base_model", None)  # This is used earlier, not needed in DPOConfig

    # ===== CREATE TRAINING ARGS =====
    # Get trial ID for unique output directory
    from ray import train as ray_train
    try:
        trial_id = ray_train.get_context().get_trial_id()
    except:
        import uuid
        trial_id = str(uuid.uuid4())[:8]  # Fallback to random ID

    training_args = DPOConfig(
        output_dir=f"./outputs/pbt_worker_{trial_id}",
        dataloader_num_workers=2,  # ✅ Parallel data loading (2-5% speedup)
        dataloader_pin_memory=True,  # ✅ Faster CPU→GPU transfer
        **config  # Remaining hyperparams (includes max_steps=1000, beta extracted above)
    )

    # ===== SAFEGUARD: Verify max_steps to prevent off-by-one bug =====
    # This ensures Trainer stops at exactly max_steps (e.g., 1000)
    # Even if Ray Tune's stop condition has off-by-one bug
    expected_max_steps = config.get("max_steps", 1000)
    if training_args.max_steps != expected_max_steps:
        raise ValueError(f"max_steps mismatch! Expected {expected_max_steps}, got {training_args.max_steps}")

    # ===== SAFETY MONITORING CALLBACK =====
    # Note: CHECK_EVERY_N_STEPS is defined in parent module scope
    # Must extract from config since workers can't access parent constants directly
    check_interval = config.get("save_steps", 50)  # Aligned with checkpoint interval

    # 🛡️ ZERO-TOLERANCE SAFETY STRATEGY (Response to Devil's Advocate Issue #5)
    # Devil's Advocate: "Aggressive early stopping. Early in training, margin CAN go negative temporarily.
    #                    Gibberish detection at step 50 catches random initialization noise.
    #                    Result: All workers stopped before PBT exploration begins."
    #
    # Our Justification (INTENTIONAL DESIGN for Safety-Critical Training):
    #
    # 1. EVIDENCE FROM PREVIOUS RUN (logs/CITA_PBT_training_20251012_153829.log):
    #    - ALL 3 workers generated gibberish from iteration 0 (not temporary noise)
    #    - Mode collapse happened IMMEDIATELY, not gradually
    #    - No recovery observed - gibberish persisted throughout training
    #    → Conclusion: Gibberish is CATASTROPHIC FAILURE, not temporary setback
    #
    # 2. NEGATIVE MARGIN SPIRAL RISK (Why margin_tolerance=0.0 is CRITICAL):
    #    Step 100: Worker A has margin = -0.03 (unsafe preference, if tolerance = -0.05)
    #    Step 150: PBT exploit → Worker B copies Worker A's weights
    #    Step 200: Worker B inherits unsafe preference → margin = -0.08
    #    Step 250: PBT amplifies trend → All workers prefer unsafe responses
    #    → Conclusion: Allowing negative margins enables CASCADE FAILURE via PBT amplification
    #
    # 3. PBT RESCUE MECHANISM (Why individual worker stopping is OK):
    #    - Individual worker stops → checkpoint saved at last healthy state
    #    - PBT detects: "Worker A failed (negative margin)"
    #    - Exploit: Worker A copies weights from healthy Worker B
    #    - Worker A recovers with safe hyperparameters from Worker B
    #    - Only abort experiment if ALL workers fail (AllWorkersSafetyStopper)
    #    → Conclusion: Early stopping + PBT rescue = optimal safety without killing experiment
    #
    # 4. SAFETY-CRITICAL DOMAIN:
    #    - CITA trains models to REJECT unsafe requests
    #    - Unsafe model deployment = real-world harm
    #    - Priority: Prevent unsafe models > Find optimal hyperparameters
    #    → Conclusion: Zero-tolerance for unsafe behavior is NON-NEGOTIABLE
    #
    # Philosophy: For safety alignment, we CANNOT tolerate even temporary unsafe preference
    # Trade-off: Strict safety > Allowing exploration of unsafe regions
    #
    # ✅ This is the RIGHT approach for CITA (not overly aggressive, but appropriately cautious)
    gibberish_monitor = GibberishDetectionCallback(
        test_prompts=test_prompts,
        check_every_n_steps=check_interval,  # ✅ Every 50 steps (fast detection = fast rescue)
        use_alpaca_format=True,
        stop_on_gibberish=True,  # ✅ Zero tolerance: Mode collapse is catastrophic (evidence: previous run)
        stop_on_negative_margin=True,  # ✅ Zero tolerance: Prevents cascade failure via PBT amplification
        margin_tolerance=0.0,  # ✅ Strict: Model MUST prefer safe responses (margin > 0)
        stop_on_high_kl=True,  # ✅ PREVENTIVE: Stops at iter 1 (3 min) vs iter 8 (80 min) when KL explodes
        kl_threshold=0.5  # ✅ KL must be < 0.5 (standard threshold for drift detection)
    )

    # ===== TRAINING SUMMARY CALLBACK (ZERO overhead - just formats existing metrics) =====
    from monitoring_callback import TrainingSummaryCallback

    summary_callback = TrainingSummaryCallback(
        check_every_n_steps=check_interval,
        training_method="cita"
    )

    # ===== RAY TUNE INTEGRATION CALLBACK =====
    # Reports metrics to Ray Tune for PBT (every save_steps=50, aligned with checkpoints)
    ray_callback = RayTrainReportCallback()

    # ===== CREATE CITA TRAINER =====
    trainer = CITATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # ✅ ADDED: Validation dataset
        lambda_sft=0.0,      # ✅ STACKED TRAINING: L_SFT already optimized in Stage 1 (SFT baseline)
        lambda_dpo=1.0,      # ✅ Continue preference learning from Stage 2 (DPO baseline)
        lambda_kl=lambda_kl, # ✅ Add KL regularization (CITA's contribution) - PBT-tuned
        beta=beta,           # Contrastive temperature (PBT-tuned)
        callbacks=[summary_callback, gibberish_monitor, ray_callback]
    )

    # ===== SHOW GPU MEMORY =====
    from model_utils import log_gpu_memory_start, log_gpu_memory_end
    start_gpu_memory = log_gpu_memory_start()

    # ===== RESUME FROM CHECKPOINT (if provided by PBT) =====
    if checkpoint_dir:
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()

    # ===== SHOW FINAL MEMORY =====
    log_gpu_memory_end(start_gpu_memory)


# ===== MAIN EXECUTION =====

def main(num_workers=4, max_steps=1000, base_model=None, force_skip=False):
    """
    Run PBT training for CITA

    Args:
        num_workers: Number of PBT workers (default: 4 for full PBT, 1 for sanity check)
        max_steps: Maximum training steps (default: 1000 for full PBT, 100 for sanity check)
        base_model: HuggingFace model ID to load LoRA adapters from (for stacking SFT→DPO→CITA)
        force_skip: If True, skip training and only load existing model (user selected option 1)

    Steps:
    1. Create PBT scheduler with CITA hyperparameters
    2. Launch workers (1 for sanity check, 4 for full PBT)
    3. Each worker trains with different hyperparameters
    4. Every 50 steps: checkpoint + safety check + PBT mutation (if needed)
    5. After max_steps: select best hyperparameters
    6. Global safety check: Abort if ALL workers failed (prevents pushing unsafe models)

    Triple Stop Protection (prevents off-by-one bug):
    1. Ray Tune: stop={"training_iteration": N, "timesteps_total": max_steps}
    2. DPOConfig: max_steps (Trainer.train() respects this)
    3. Assertion: Verifies max_steps matches expected value
    """
    # Calculate expected iterations
    max_iterations_expected = max_steps // CHECK_EVERY_N_STEPS

    # Determine mode
    mode = "SANITY CHECK" if num_workers == 1 else "FULL PBT"

    print("\n" + "="*80)
    print(f"🚀 Starting TRUE CITA Training - {mode} MODE")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Mode: {mode}")
    print(f"  - Workers: {num_workers} total" + (f" (2 RUNNING in parallel, 2 PAUSED)" if num_workers == 4 else " (SEQUENTIAL)"))
    print(f"  - GPU allocation: {0.5 if num_workers > 1 else 1.0} per worker")
    print(f"  - Memory per worker: ~18GB (Base 16GB + LoRA 0.2GB + Gradients 2GB)")
    if num_workers > 1:
        print(f"  - Active memory: 2 × 18GB = 36GB / 40GB (4GB buffer)")
    print(f"  - PEFT + ref_model=None: Memory-efficient (toggles adapters, no deepcopy)")
    print(f"  - Checkpoint interval: {CHECK_EVERY_N_STEPS} steps (aligned with safety checks)")
    print(f"  - Mutation interval: {CHECK_EVERY_N_STEPS} steps (PBT compares all {num_workers} workers)")
    print(f"  - Total steps: {max_steps} per worker ({max_iterations_expected} checkpoints)")
    print(f"  - Hyperparameters tuned: λ_kl, LR, β, weight_decay, warmup_steps")
    if num_workers == 4:
        expected_time = (max_steps / 1000) * 120  # Scale from 1000-step baseline
        expected_cost = expected_time / 60 * 1.5
        print(f"  - Expected time: ~{expected_time:.0f} minutes (4 workers, 2 pairs sequential)")
        print(f"  - Expected cost: ~${expected_cost:.2f} ({expected_time:.0f} min × $1.5/hr)")
    else:
        expected_time = (max_steps / 1000) * 30  # Single worker is faster
        print(f"  - Expected time: ~{expected_time:.0f} minutes (1 worker, no PBT overhead)")
    print("="*80 + "\n")

    try:
        # Get hyperparameter space with custom max_steps and base_model
        hp_space = get_cita_hp_space(max_steps=max_steps, base_model=base_model)
        save_steps = hp_space["save_steps"]  # CHECK_EVERY_N_STEPS (50)

        # ===================================================================
        # CHECKPOINT DETECTION
        # ===================================================================
        print("\n" + "="*80)
        print("🔍 Checking for existing checkpoints...")
        print("="*80 + "\n")

        hf_model_found = False
        training_skipped = False

        # Force skip if user selected inference-only mode
        if force_skip:
            print("🚫 User selected inference-only mode")
            print("   Skipping training, will load model from HuggingFace...\n")
            training_skipped = True
            hf_model_found = True  # Treat as HF model (will load for inference)
        else:
            # Priority 1: Check local Ray Tune experiments (CHANGED ORDER - local first)
            # This allows retraining even if HF repo exists (user chose option 2)
            print("1️⃣ Checking local Ray Tune experiments...")
            experiment_path = str(project_root / "outputs" / "ray_results" / "cita_pbt_training")
            exp_exists, exp_complete, resume_mode = check_ray_tune_experiment(
                experiment_path,
                max_iterations=max_iterations_expected
            )

            if exp_complete:
                print(f"✅ Training already completed!")
                print(f"   Experiment: {experiment_path}")
                print(f"   Max iterations: {max_iterations_expected}")
                print(f"   Skipping training, loading results...\n")
                training_skipped = True
            elif exp_exists:
                print(f"📂 Found incomplete experiment: {experiment_path}")
                print(f"   Resuming training from checkpoint...")
                print(f"   Resume mode: {resume_mode}\n")
                training_skipped = False
            else:
                print(f"🆕 No local experiment found")
                print(f"   Will start fresh training (even if HF repo exists)...\n")
                training_skipped = False

        # Create PBT scheduler
        pbt_scheduler = create_pbt_scheduler(
            hyperparam_mutations=get_cita_hyperparam_mutations(),
            mutation_interval=CHECK_EVERY_N_STEPS,  # ✅ ALIGNED: mutation = checkpoint = safety check (50 steps)
            metric="cita/margin",  # ✅ SAFEGUARD: Use margin (should be positive) instead of loss
            mode="max"  # ✅ SAFEGUARD: Maximize margin (chosen_logps - rejected_logps > 0)
        )

        # Run PBT training (or load existing results)
        if not training_skipped:
            # ✅ PARALLELIZATION: 4 workers with gpu_fraction=0.5 (Response to Devil's Advocate)
            #
            # Memory validation (Web search confirmed):
            # - ref_model=None with PEFT: Memory-efficient (toggles adapters, no deepcopy)
            # - Per worker: Base 16GB + LoRA 0.2GB + Gradients 2GB = ~18GB
            # - 2 workers parallel: 2 × 18GB = 36GB < 40GB ✅ SAFE!
            #
            # Ray Tune execution pattern:
            # - num_workers=4: Total workers in PBT population
            # - gpu_fraction=0.5: Each worker allocated 50% GPU (2 workers fit in parallel)
            # - At any time: 2 workers RUNNING (using 36GB) + 2 workers PAUSED (saved at checkpoint)
            # - PBT compares ALL 4 workers at each checkpoint (exploit/explore works correctly)
            #
            analysis = run_pbt_training(
                trainable=train_cita_with_pbt,
                hp_space=hp_space,
                scheduler=pbt_scheduler,
                num_workers=num_workers,  # ✅ Configurable: 1 for sanity check, 4 for full PBT
                max_iterations=max_iterations_expected,  # Dynamic based on max_steps
                output_dir=str(project_root / "outputs" / "ray_results"),  # Absolute path required
                name="cita_pbt_training",
                gpu_fraction=0.5 if num_workers > 1 else 1.0,  # ✅ 1.0 for single worker, 0.5 for multi-worker
                resume=resume_mode  # ✅ Enable checkpoint resumption
            )
        else:
            # Load existing results
            if hf_model_found:
                # HF model found - create dummy analysis object for push automation
                print(f"📂 Using HuggingFace model (no Ray Tune experiment)...")
                from types import SimpleNamespace
                # Use local output directory (model already on HF, no need to push)
                local_output = project_root / "outputs" / "lora_model_CITA_Baseline_PBT_BF16"
                # Create a minimal best_trial object for push automation
                best_trial = SimpleNamespace(
                    final_metric='N/A (loaded from HF)',
                    checkpoint=SimpleNamespace(
                        dir_or_data=str(local_output)
                    )
                )
                analysis = None
            else:
                # Local experiment found
                print(f"📂 Loading existing experiment results...")
                from ray import tune
                analysis = tune.ExperimentAnalysis(experiment_path)
                print(f"✅ Loaded experiment with {len(analysis.trials)} trials")

        # ===================================================================
        # NOTE: Global safety check handled by AllWorkersSafetyStopper during training
        # If all workers fail, experiment aborts immediately (no need to check here)
        # This prevents GPU waste (old approach waited until max_steps before checking)
        # ===================================================================

        # Print best hyperparameters and get checkpoint
        if hf_model_found:
            # Already set best_trial above when HF model was found
            local_output = project_root / "outputs" / "lora_model_CITA_Baseline_PBT_BF16"
            config_path = str(local_output / "best_pbt_config.json")
            best_checkpoint = str(local_output)
            print(f"\n✅ Best model checkpoint: {best_checkpoint}\n")
        else:
            # Analyze Ray Tune results
            best_trial = print_best_hyperparameters(analysis, metric="cita/margin", mode="max")

            # Save best hyperparameters to file
            config_path = save_best_config(best_trial, "./outputs/best_pbt_config.json")

            # Handle different Ray Tune versions (dir_or_data vs path)
            try:
                best_checkpoint = best_trial.checkpoint.dir_or_data
            except AttributeError:
                best_checkpoint = getattr(best_trial.checkpoint, 'path', str(best_trial.checkpoint))

            print(f"\n✅ Best model checkpoint: {best_checkpoint}\n")

        # ===================================================================
        # Inference Tests (only if training was skipped - Option 1 selected)
        # ===================================================================
        if training_skipped and hf_model_found:
            print("\n" + "="*80)
            print("🧪 Running inference tests...")
            print("="*80 + "\n")

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
                from peft import PeftModel

                # Load base model + tokenizer (reuse training setup for consistency)
                print("📦 Loading base model for inference...")
                from model_utils import load_model_bf16
                base_model, tokenizer = load_model_bf16(
                    model_id="meta-llama/Llama-3.1-8B",
                    max_seq_length=2048,
                    use_flash_attention=True
                )

                # Load LoRA adapter from HuggingFace
                print(f"🔧 Loading LoRA adapter from HuggingFace: {HF_REPO}")
                model = PeftModel.from_pretrained(base_model, HF_REPO, token=HF_TOKEN)
                print("✅ Model loaded from HuggingFace\n")

                # Prepare model for inference
                model.eval()
                model = model.to(torch.bfloat16)

                # Disable gradient checkpointing for inference
                if hasattr(model, 'gradient_checkpointing_disable'):
                    model.gradient_checkpointing_disable()

                # Get test prompts
                from model_utils import get_test_prompts
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

                # Clean up GPU memory
                del base_model
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️  Inference test failed: {e}")
                print(f"   Continuing with push automation...\n")
                import traceback
                traceback.print_exc()

        # ===================================================================
        # Automated Push to HuggingFace and GitHub
        # ===================================================================
        # Initialize push automation
        pusher = PushAutomation(
            hf_token=HF_TOKEN,
            github_email="kapilw25@gmail.com",
            github_username="kapilw25",
            project_root=project_root
        )

        # Push to both HuggingFace (conditional) and GitHub (always)
        # - HF: Only if current margin > previous margin (prevents overwriting better models)
        # - GitHub: Always push (especially logs/ for analysis)
        # - Large log handling: Git LFS or split into <100MB chunks
        pusher.push_all(
            best_trial=best_trial,
            best_checkpoint=best_checkpoint,
            hf_repo=HF_REPO,
            config_path=config_path,
            run_name=RUN_NAME,
            skip_local_backup=training_skipped  # Skip if inference-only mode
        )

    except Exception as e:
        print(f"\n❌ Error during PBT training: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Shutdown Ray
        ray.shutdown()
        print("\n🏁 PBT Training Complete!\n")

        # Restore original stdout/stderr and close log file
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"📝 Complete log file saved: {log_filename}")

        # ===================================================================
        # Summary
        # ===================================================================
        print(f"\n{'='*80}")
        print("✅ All results saved!")
        print(f"{'='*80}")
        print("Results saved to:")
        print(f"  - Local: ./outputs/ray_results/cita_pbt_training/")
        print(f"  - HuggingFace: {HF_REPO} (only if performance improved)")
        print(f"  - GitHub: Logs and code pushed")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # ===== COMMAND-LINE ARGUMENT PARSING =====
    parser = argparse.ArgumentParser(
        description="TRUE CITA Training with PBT - Contrastive Instruction-Tuned Alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Option A: Sanity check (1 worker, 100 steps, ~3 minutes)
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py --mode sanity

  # Option B: Full PBT (4 workers, 1000 steps, ~120 minutes)
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py --mode full

  # Custom configuration
  python comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py --workers 2 --steps 500
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["sanity", "full"],
        default="full",
        help="Training mode: 'sanity' (1 worker, 100 steps) or 'full' (4 workers, 1000 steps)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of PBT workers (overrides --mode, e.g., 1 for sanity check, 4 for full PBT)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides --mode, e.g., 100 for sanity check, 1000 for full PBT)"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="HuggingFace model ID to load LoRA adapters from (for stacking SFT→DPO→CITA)"
    )

    args = parser.parse_args()

    # Determine configuration based on mode or custom arguments
    if args.workers is not None and args.steps is not None:
        # Custom configuration
        num_workers = args.workers
        max_steps = args.steps
        print(f"✅ Custom configuration: {num_workers} workers, {max_steps} steps")
    elif args.mode == "sanity":
        # Sanity check mode
        num_workers = 1
        max_steps = 200  # 100 warmup + 100 training (see actual LR schedule)
        print(f"✅ Sanity check mode: {num_workers} worker, {max_steps} steps (~6 minutes)")
    else:
        # Full PBT mode (default)
        num_workers = 4
        max_steps = 1000
        print(f"✅ Full PBT mode: {num_workers} workers, {max_steps} steps (~120 minutes)")

    # ===================================================================
    # Ask about retraining BEFORE starting (check HF repo first)
    # ===================================================================
    # Estimate time based on configuration
    if max_steps == 200 and num_workers == 1:
        estimated_time = "~6 minutes"
    elif max_steps == 1000 and num_workers == 4:
        estimated_time = "~120 minutes (2 hours)"
    else:
        estimated_time = f"~{int(max_steps * num_workers * 0.12)} minutes"

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
    print(f"Training will take approximately: {estimated_time}")
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

    # Run main with configured parameters
    main(num_workers=num_workers, max_steps=max_steps, base_model=args.base_model, force_skip=force_skip)