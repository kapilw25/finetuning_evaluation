"""
Push existing SFT/DPO models to new HuggingFace repos with clearer names

Uses existing push_automation.py infrastructure
"""

import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

from model_utils import load_hf_token
from push_automation import PushAutomation

# Load HF token
HF_TOKEN = load_hf_token(project_root)

print("=" * 80)
print("📦 Pushing Models to Renamed HuggingFace Repos")
print("=" * 80)
print()

# ================================================================
# 1. SFT_Baseline → kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct
# ================================================================

print("1️⃣ Pushing SFT_Baseline to new repo...")
print()

SFT_CONFIG = {
    "method": "SFT",
    "num_epochs": 1.0,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "optimizer": "adamw_torch",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
}

PushAutomation.prepare_baseline_push(
    method="SFT",
    output_dir="outputs/SFT_Baseline",
    training_config=SFT_CONFIG,
    training_skipped=False,  # We have local checkpoint
    hf_token=HF_TOKEN,
    hf_repo="kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct",  # NEW NAME
    run_name="SFT_Baseline",
    metric_names=["eval_loss"],
    metric_mode="min",
    project_root=project_root
)

print()
print("=" * 80)
print()

# ================================================================
# 2. DPO_Baseline → kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct
# ================================================================

print("2️⃣ Pushing DPO_Baseline to new repo...")
print()

DPO_CONFIG = {
    "method": "DPO",
    "num_epochs": 1.0,
    "learning_rate": 1e-5,
    "warmup_steps": 100,
    "optimizer": "adamw_torch",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 2048,
    "max_prompt_length": 1024,
    "beta": 0.1,
}

PushAutomation.prepare_baseline_push(
    method="DPO",
    output_dir="outputs/DPO_Baseline",
    training_config=DPO_CONFIG,
    training_skipped=False,  # We have local checkpoint
    hf_token=HF_TOKEN,
    hf_repo="kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct",  # NEW NAME
    run_name="DPO_Baseline",
    metric_names=["eval_rewards/margins", "rewards/margins"],
    metric_mode="max",
    project_root=project_root
)

print()
print("=" * 80)
print("✅ Both models pushed to new HuggingFace repos!")
print("=" * 80)
print()
print("New repos:")
print("  - https://huggingface.co/kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct")
print("  - https://huggingface.co/kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct")
print()
print("Old repos still exist:")
print("  - kapilw25/llama3-8b-pku-sft-baseline-bf16")
print("  - kapilw25/llama3-8b-pku-dpo-sft-bf16")
print()
print("You can delete old repos manually using:")
print("  huggingface-cli delete-repo <repo_id>")
print("=" * 80)
