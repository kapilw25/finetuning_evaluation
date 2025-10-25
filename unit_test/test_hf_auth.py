#!/usr/bin/env python3
"""
Test HuggingFace Authentication and Llama-3.1-8B Access
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("HuggingFace Authentication Test")
print("="*80)

# Step 1: Load HF_TOKEN from .env
print("\n1️⃣ Loading HF_TOKEN from .env...")
from dotenv import load_dotenv

env_path = project_root / ".env"
if not env_path.exists():
    print(f"❌ .env file not found at: {env_path}")
    sys.exit(1)

load_dotenv(env_path)
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN not found in .env file")
    sys.exit(1)

print(f"✅ HF_TOKEN loaded: {HF_TOKEN[:10]}...{HF_TOKEN[-5:]}")

# Step 2: Test HF API authentication
print("\n2️⃣ Testing HuggingFace API authentication...")
try:
    from huggingface_hub import HfApi, whoami

    api = HfApi(token=HF_TOKEN)
    user_info = whoami(token=HF_TOKEN)

    print(f"✅ Authenticated as: {user_info['name']}")
    print(f"   Type: {user_info['type']}")
    print(f"   Full name: {user_info.get('fullname', 'N/A')}")

except Exception as e:
    print(f"❌ Authentication failed: {e}")
    sys.exit(1)

# Step 3: Test access to Llama-3.1-8B (without downloading)
print("\n3️⃣ Testing access to meta-llama/Llama-3.1-8B...")
try:
    from huggingface_hub import model_info

    info = model_info("meta-llama/Llama-3.1-8B", token=HF_TOKEN)

    print(f"✅ Access granted to: {info.modelId}")
    print(f"   Model type: {info.pipeline_tag}")
    print(f"   Gated: {info.gated}")
    print(f"   Downloads: {info.downloads}")

except Exception as e:
    print(f"❌ Access denied: {e}")
    print("\n⚠️  You need to:")
    print("   1. Visit: https://huggingface.co/meta-llama/Llama-3.1-8B")
    print("   2. Accept the license agreement")
    print("   3. Wait for approval (usually instant)")
    sys.exit(1)

# Step 4: Test loading tokenizer (lightweight test)
print("\n4️⃣ Testing tokenizer loading...")
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

# Step 5: Test loading model config (no weights download)
print("\n5️⃣ Testing model config loading...")
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
    print(f"   Num attention heads: {config.num_attention_heads}")

except Exception as e:
    print(f"❌ Model config loading failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED")
print("="*80)
print("\n✅ Your HF_TOKEN is valid and you have access to Llama-3.1-8B")
print("✅ Ready to run AQI evaluation")
print("="*80)
