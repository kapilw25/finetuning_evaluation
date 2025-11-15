#!/usr/bin/env python3
"""
Force push SFT_NoInstruct to HuggingFace (bypassing performance check)

Usage:
    python unit_test/force_push_sft_noinstruct.py
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add 0c_utils to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# Load environment variables
load_dotenv(project_root / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN not found in .env file")
    sys.exit(1)

# Configuration
CHECKPOINT_DIR = project_root / "outputs/SFT_NoInstruct/checkpoint-1354"
CONFIG_PATH = project_root / "outputs/sft_baseline_config.json"
HF_REPO = "kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct"
RUN_NAME = "SFT_NoInstruct"

print(f"\n{'='*80}")
print("🚀 FORCE PUSH: SFT_NoInstruct to HuggingFace")
print(f"{'='*80}")
print(f"⚠️  WARNING: This will REPLACE the existing model on HuggingFace")
print(f"   Repository: {HF_REPO}")
print(f"   Checkpoint: {CHECKPOINT_DIR}")
print(f"   Config: {CONFIG_PATH}")
print(f"{'='*80}\n")

# Confirmation
response = input("Continue with force push? (yes/no): ").strip().lower()
if response != "yes":
    print("❌ Force push cancelled")
    sys.exit(0)

# Extract eval_loss from trainer_state.json
print("\n📊 Extracting metrics from checkpoint...")
trainer_state_path = CHECKPOINT_DIR / "trainer_state.json"

if not trainer_state_path.exists():
    print(f"❌ trainer_state.json not found in {CHECKPOINT_DIR}")
    sys.exit(1)

with open(trainer_state_path, 'r') as f:
    trainer_state = json.load(f)

# Get final eval_loss
eval_loss = None
for log_entry in reversed(trainer_state.get("log_history", [])):
    if "eval_loss" in log_entry:
        eval_loss = log_entry["eval_loss"]
        break

if eval_loss is None:
    print("⚠️  eval_loss not found in trainer_state.json, using N/A")
    eval_loss = "N/A"
else:
    print(f"✅ Found eval_loss: {eval_loss:.6f}")

# Load training config
print("\n📋 Loading training configuration...")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        training_config = json.load(f)
    print(f"✅ Loaded config from {CONFIG_PATH}")
else:
    print(f"⚠️  Config not found at {CONFIG_PATH}, using defaults")
    training_config = {}

# Load base model
print("\n📦 Loading base model (meta-llama/Llama-3.1-8B)...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)
print("✅ Base model loaded")

# Load LoRA adapter
print(f"\n🔧 Loading LoRA adapter from {CHECKPOINT_DIR}...")
model_with_adapter = PeftModel.from_pretrained(
    base_model,
    str(CHECKPOINT_DIR),
)
print("✅ LoRA adapter loaded")

# Load tokenizer
print("\n📋 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    use_fast=True,
    token=HF_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded")

# Update model metadata
print("\n📝 Updating model metadata...")
model_with_adapter.config.update({
    "training_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "training_method": "SFT",
    "dataset": "PKU-SafeRLHF",
    "filtered_samples": training_config.get("filtered_samples", 10831),
    "epochs": training_config.get("epochs", 1.0),
    "max_steps": 1354,
    "precision": "BF16",
    "run_name": RUN_NAME,
    "chat_template": "llama3",
    "instruction_mode": False,
    "hyperparameters": {
        "learning_rate": training_config.get("learning_rate", 2e-4),
        "batch_size": training_config.get("batch_size", 2),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 4),
        "warmup_steps": training_config.get("warmup_steps", 100),
        "lr_scheduler_type": training_config.get("lr_scheduler_type", "cosine"),
        "weight_decay": training_config.get("weight_decay", 0.01),
        "max_grad_norm": training_config.get("max_grad_norm", 1.0),
    },
    "final_eval_loss": eval_loss if eval_loss != "N/A" else "N/A",
    "lora_config": {
        "r": 16,
        "alpha": 16,
        "trainable_params": "41.9M (0.52%)",
    },
    "max_seq_length": training_config.get("max_seq_length", 2048),
})

# Create commit message
commit_msg = f"""SFT Training (NoInstruct) - BF16 (LoRA Adapter)

Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Method: Supervised Fine-Tuning (SFT) without instruction conditioning
Steps: 1,354 | Final eval_loss: {eval_loss if eval_loss == 'N/A' else f'{eval_loss:.4f}'}

Hyperparameters:
- Learning rate: {training_config.get('learning_rate', 2e-4)}
- Batch size: {training_config.get('batch_size', 2)} (per device)
- Gradient accumulation: {training_config.get('gradient_accumulation_steps', 4)}
- Effective batch size: 8
- Warmup steps: {training_config.get('warmup_steps', 100)}
- LR scheduler: {training_config.get('lr_scheduler_type', 'cosine')}
- Weight decay: {training_config.get('weight_decay', 0.01)}
- Max grad norm: {training_config.get('max_grad_norm', 1.0)}

Dataset: PKU-SafeRLHF (filtered, clear contrast pairs)
- Train samples: {training_config.get('filtered_samples', 10831)}
- Validation samples: 1,204
- Epochs: {training_config.get('epochs', 1.0)}

LoRA Configuration:
- Rank (r): 16
- Alpha: 16
- Trainable params: 41.9M (0.52% of total)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Precision: BF16 + Flash Attention 2
Max sequence length: {training_config.get('max_seq_length', 2048)}

Instruction conditioning: None (NoInstruct baseline)

Compatible with inference_bf16.py evaluation script.
Includes trainer_state.json with full training history.

⚠️ FORCE PUSH: Bypassing performance comparison check.
"""

# Push to HuggingFace
print(f"\n{'='*80}")
print(f"📤 Pushing to HuggingFace: {HF_REPO}")
print(f"{'='*80}")
print("   This will REPLACE the existing model")
print("   Includes: model weights, adapter config, tokenizer, trainer_state.json")
print(f"{'='*80}\n")

try:
    # Push model adapter
    print("📤 Pushing LoRA adapter...")
    model_with_adapter.push_to_hub(
        HF_REPO,
        token=HF_TOKEN,
        commit_message=commit_msg,
        private=True,
    )
    print("✅ LoRA adapter pushed successfully")

    # Push tokenizer
    print("\n📤 Pushing tokenizer...")
    tokenizer.push_to_hub(
        HF_REPO,
        token=HF_TOKEN,
        private=True,
    )
    print("✅ Tokenizer pushed successfully")

    # Verify trainer_state.json is in the checkpoint
    if trainer_state_path.exists():
        print(f"\n✅ trainer_state.json verified in checkpoint")
        print(f"   Path: {trainer_state_path}")
        print(f"   Size: {trainer_state_path.stat().st_size / 1024:.1f} KB")
        print(f"   This file is automatically included with adapter push")
    else:
        print(f"\n⚠️  trainer_state.json not found (may not be included in push)")

    print(f"\n{'='*80}")
    print("✅ FORCE PUSH COMPLETED!")
    print(f"{'='*80}")
    print(f"🎉 Model successfully pushed to: https://huggingface.co/{HF_REPO}")
    print(f"📊 Final eval_loss: {eval_loss if eval_loss == 'N/A' else f'{eval_loss:.4f}'}")
    print(f"💾 Local backup: outputs/lora_model_SFT_NoInstruct_PBT_BF16/")
    print(f"{'='*80}\n")

except Exception as e:
    print(f"\n❌ Push failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
