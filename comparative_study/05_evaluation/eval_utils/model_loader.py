"""
Shared model loading utilities for evaluation scripts

Usage:
    from eval_utils import MODELS, load_model_for_eval, unload_model

    model, tokenizer = load_model_for_eval("CITA_Instruct")
    # ... run evaluation ...
    unload_model(model)
"""

import os
import sys
import gc
import time
import torch
from pathlib import Path
from typing import Tuple, List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

from model_utils import load_hf_token

# ===================================================================
# Shared Configuration
# ===================================================================

BASE_MODEL = "meta-llama/Llama-3.1-8B"
HF_TOKEN = load_hf_token(project_root)

# Standardized model configurations (used by ALL evaluation scripts)
MODELS = {
    "Baseline": {
        "hf_repo": None,  # No adapter - just base model
        "display_name": "Baseline (Unaligned)",
        "use_instruction": False,
    },
    "SFT_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct",
        "display_name": "SFT NoInstruct",
        "use_instruction": False,
    },
    "SFT_Instruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct",
        "display_name": "SFT Instruct",
        "use_instruction": True,
    },
    "DPO_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct",
        "display_name": "DPO NoInstruct",
        "use_instruction": False,
    },
    "DPO_Instruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct",
        "display_name": "DPO Instruct",
        "use_instruction": True,
    },
    "CITA_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-CITA-NoInstruct-DPO-NoInstruct",
        "display_name": "CITA NoInstruct",
        "use_instruction": False,
    },
    "CITA_Instruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct",
        "display_name": "CITA Instruct",
        "use_instruction": True,
    },
}


# ===================================================================
# Model Loading
# ===================================================================

def load_model_for_eval(model_key: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model from HuggingFace in BF16

    Args:
        model_key: Key from MODELS dict (e.g., "CITA_Instruct", "DPO_NoInstruct")

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n{'='*80}")
    print(f"Loading {MODELS[model_key]['display_name']} for evaluation")
    print(f"{'='*80}")

    model_info = MODELS[model_key]
    hf_repo = model_info["hf_repo"]

    if hf_repo:
        print(f"Loading from HuggingFace: {hf_repo}")
    else:
        print(f"Loading base model (no adapter)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models in batch generation

    # Set Llama-3.1 chat template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and message['role'] != 'system' %}"
            "{{ '<|begin_of_text|>' }}"
            "{% endif %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{% endif %}"
        )
        print("Set Llama-3.1 chat template")

    # Load base model in BF16
    print(f"Loading base model in BF16: {BASE_MODEL}")

    # Force GPU-only (no CPU offloading) to prevent Flash Attention failures
    max_memory = {0: "22GiB", "cpu": "0GiB"}  # Explicitly disallow CPU offloading

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="flash_attention_2",
            token=HF_TOKEN
        )
        print("Using Flash Attention 2")
    except Exception as e:
        print(f"Flash Attention 2 unavailable, using eager mode")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="eager",
            token=HF_TOKEN
        )

    # Load LoRA adapter if specified
    if hf_repo is not None:
        print(f"Downloading adapter from HuggingFace: {hf_repo}")
        try:
            model = PeftModel.from_pretrained(model, hf_repo, token=HF_TOKEN)
            print("Merging adapter weights...")
            model = model.merge_and_unload()
            print("Adapter merged")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load adapter from HuggingFace: {hf_repo}\n"
                f"Error: {e}\n\n"
                f"Possible causes:\n"
                f"  1. Model not yet trained and pushed to HuggingFace\n"
                f"  2. Training performance did not improve (push_automation skipped push)\n"
                f"  3. HuggingFace authentication issue (check HF_TOKEN in .env)\n"
            )

    model.eval()
    print(f"{MODELS[model_key]['display_name']} loaded successfully")

    return model, tokenizer


def unload_model(model):
    """Free GPU memory with proper cleanup"""
    if model is not None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force inter-process memory collection
            torch.cuda.ipc_collect()
        print("Model unloaded, GPU memory freed")
        time.sleep(5)  # Give GPU time to fully release memory


def verify_hf_repos(model_keys: List[str], interactive: bool = True) -> List[str]:
    """
    Pre-flight verification of HuggingFace repositories

    Args:
        model_keys: List of model keys to verify
        interactive: If True, prompt user on missing repos

    Returns:
        List of valid model keys (missing ones removed)
    """
    print("\n" + "=" * 80)
    print("Verifying HuggingFace repositories...")
    print("=" * 80)

    hf_api = HfApi()
    missing_repos = []
    valid_keys = []

    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"  ❌ {model_key}: Unknown model key")
            continue

        hf_repo = MODELS[model_key]['hf_repo']
        if hf_repo:
            try:
                hf_api.repo_info(repo_id=hf_repo, repo_type="model", token=HF_TOKEN)
                print(f"  ✅ {model_key}: {hf_repo}")
                valid_keys.append(model_key)
            except Exception as e:
                print(f"  ❌ {model_key}: {hf_repo} (NOT FOUND)")
                missing_repos.append(model_key)
        else:
            print(f"  ✅ {model_key}: (base model only, no adapter)")
            valid_keys.append(model_key)

    if missing_repos and interactive:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: Some models are missing from HuggingFace")
        print("=" * 80)
        print(f"Missing models: {', '.join(missing_repos)}")
        print("\nOptions:")
        print("  1) Continue evaluation (skip missing models)")
        print("  2) Abort evaluation")
        print("=" * 80)

        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == "1":
                print("\n✅ Continuing evaluation (will skip missing models)")
                break
            elif choice == "2":
                print("\n❌ Aborting evaluation")
                return []
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")

    return valid_keys
