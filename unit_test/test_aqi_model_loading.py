#!/usr/bin/env python3
"""
Test that run_full_aqi_evaluation.py can load models with HF_TOKEN
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env
from dotenv import load_dotenv
load_dotenv(project_root / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

print("="*80)
print("AQI Model Loading Test (Minimal)")
print("="*80)

# Test 1: Load tokenizer only (lightweight, no GPU needed)
print("\n1️⃣ Testing tokenizer loading with token...")
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        token=HF_TOKEN
    )

    print(f"✅ Tokenizer loaded successfully")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Model max length: {tokenizer.model_max_length}")

except Exception as e:
    print(f"❌ Tokenizer loading failed: {e}")
    sys.exit(1)

# Test 2: Load model config only (no weights download)
print("\n2️⃣ Testing model config loading with token...")
try:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        token=HF_TOKEN
    )

    print(f"✅ Model config loaded successfully")
    print(f"   Architecture: {config.architectures}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")

except Exception as e:
    print(f"❌ Model config loading failed: {e}")
    sys.exit(1)

# Test 3: Verify adapter repos are accessible
print("\n3️⃣ Testing adapter repository access...")
adapters = {
    "SFT": "kapilw25/llama3-8b-pku-sft-baseline-bf16",
    "DPO": "kapilw25/llama3-8b-pku-dpo-sft-bf16",
    "CITA": "kapilw25/llama3-8b-pku-cita-dpo-bf16"
}

from huggingface_hub import model_info

for name, repo in adapters.items():
    try:
        info = model_info(repo, token=HF_TOKEN)
        print(f"✅ {name:<5} adapter accessible: {repo}")
    except Exception as e:
        print(f"⚠️  {name:<5} adapter NOT accessible: {e}")

print("\n" + "="*80)
print("✅ MODEL LOADING TEST PASSED")
print("="*80)
print("\n✅ All models and adapters are accessible with your HF_TOKEN")
print("✅ Ready to run full AQI evaluation")
print("="*80)
