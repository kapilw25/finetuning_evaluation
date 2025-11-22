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
import torch
from pathlib import Path
from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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
    """Free GPU memory"""
    if model is not None:
        del model
        torch.cuda.empty_cache()
        print("Model unloaded, GPU memory freed")
