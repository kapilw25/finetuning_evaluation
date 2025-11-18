"""
Push Trial 7 (CITA_Instruct) to HuggingFace
Uses exact push code from Llama3_BF16.py (lines 589-600)
"""

import sys
import json
from pathlib import Path

# Add utils to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

from model_utils import load_hf_token, get_model_repo_name
from push_automation import PushAutomation

# ===== CONFIGURATION =====
RUN_NAME = "CITA_Instruct"
TRIAL_7_DIR = project_root / "outputs" / "CITA_Instruct_Adaptive" / "trial_7"

# Load HF token
HF_TOKEN = load_hf_token(project_root)
HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")

print("=" * 80)
print("🚀 Pushing Trial 7 (CITA_Instruct) to HuggingFace")
print("=" * 80)
print(f"Source: {TRIAL_7_DIR}")
print(f"Target: {HF_REPO}")
print("=" * 80 + "\n")

# ===== LOAD TRIAL CONFIG =====
trial_config_path = TRIAL_7_DIR / "trial_config.json"

if not trial_config_path.exists():
    print(f"❌ Error: {trial_config_path} not found")
    sys.exit(1)

with open(trial_config_path, 'r') as f:
    trial_data = json.load(f)

print(f"✅ Loaded Trial 7 config")
print(f"   Final margin: {trial_data['final_margin']:.4f}")
print(f"   Final accuracy: {trial_data['final_accuracy']:.4f}")
print(f"   Final eval_loss: {trial_data['final_eval_loss']:.4f}\n")

# ===== BUILD TRAINING CONFIG (same structure as Llama3_BF16.py lines 378-393) =====
training_config = {
    "method": "CITA",
    "num_epochs": 1.0,
    "lambda_kl": trial_data["lambda_kl"],
    "learning_rate": trial_data["learning_rate"],
    "beta": trial_data["beta"],
    "weight_decay": trial_data["weight_decay"],
    "warmup_ratio": trial_data["warmup_ratio"],
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "optimizer": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "max_seq_length": 2048,
    "max_prompt_length": 1024,
    # Note: final_margin will be re-extracted from checkpoint by PushAutomation
}

# ===== PUSH TO HUGGINGFACE (exact code from Llama3_BF16.py lines 589-600) =====
PushAutomation.prepare_baseline_push(
    method="CITA",
    output_dir=str(TRIAL_7_DIR),
    training_config=training_config,
    training_skipped=False,
    hf_token=HF_TOKEN,
    hf_repo=HF_REPO,
    run_name=RUN_NAME,
    metric_names=["eval_rewards/margins", "rewards/margins"],
    metric_mode="max",
    project_root=project_root
)

print("\n" + "=" * 80)
print("✅ Trial 7 Push Complete!")
print("=" * 80)
print(f"HuggingFace: https://huggingface.co/{HF_REPO}")
print("=" * 80)
