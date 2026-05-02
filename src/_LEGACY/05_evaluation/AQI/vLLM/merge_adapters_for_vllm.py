#!/usr/bin/env python3
"""
Merge LoRA Adapters to Full Models for vLLM

vLLM cannot load LoRA adapters directly - it needs merged full models.
This script downloads adapters from HuggingFace and merges them with the base model.
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent.parent.parent  # finetuning_evaluation directory (up 3 levels)

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
OUTPUT_BASE_DIR = BASE_DIR / "outputs" / "merged_models_for_vllm"  # Relative path for portability

# Models to merge (using HuggingFace repos)
ADAPTERS = {
    "sft_baseline": {
        "hf_repo": "kapilw25/llama3-8b-pku-sft-baseline-bf16",
        "output_dir": "llama3-8b-sft-merged"
    },
    "dpo_baseline": {
        "hf_repo": "kapilw25/llama3-8b-pku-dpo-sft-bf16",
        "output_dir": "llama3-8b-dpo-merged"
    },
    "cita_baseline": {
        "hf_repo": "kapilw25/llama3-8b-pku-cita-dpo-bf16",
        "output_dir": "llama3-8b-cita-merged"
    }
}


def merge_adapter(adapter_name, adapter_info):
    """Merge LoRA adapter with base model and save"""

    print(f"\n{'='*80}")
    print(f"Merging {adapter_name}")
    print(f"{'='*80}")

    output_dir = OUTPUT_BASE_DIR / adapter_info["output_dir"]

    # Check if already merged
    if output_dir.exists() and (output_dir / "config.json").exists():
        print(f"✅ Already merged: {output_dir}")
        print(f"   Skipping merge (delete directory to re-merge)")
        return str(output_dir)

    print(f"📥 Loading base model: {BASE_MODEL_NAME} (BF16)")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print(f"📥 Loading adapter from HuggingFace: {adapter_info['hf_repo']}")
    model = PeftModel.from_pretrained(base_model, adapter_info["hf_repo"])

    print("🔄 Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    print(f"💾 Saving merged model to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    # Also save tokenizer
    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Merged model saved: {output_dir}")
    print(f"   Files: config.json, model.safetensors, tokenizer files")

    # Cleanup
    del merged_model
    del model
    del base_model
    torch.cuda.empty_cache()

    return str(output_dir)


def main():
    print(f"\n{'='*80}")
    print("MERGE LORA ADAPTERS FOR VLLM")
    print(f"{'='*80}")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Adapters to merge: {len(ADAPTERS)}")
    print(f"\nThis will:")
    print(f"  1. Download adapters from HuggingFace")
    print(f"  2. Merge with base model (BF16)")
    print(f"  3. Save full merged models (~16GB each)")
    print(f"  4. Total disk usage: ~48GB for 3 models")
    print(f"  5. Estimated time: 20-30 minutes")

    merged_paths = {}

    for adapter_name, adapter_info in ADAPTERS.items():
        try:
            merged_path = merge_adapter(adapter_name, adapter_info)
            merged_paths[adapter_name] = merged_path
        except Exception as e:
            print(f"\n❌ Error merging {adapter_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("MERGE COMPLETE")
    print(f"{'='*80}")
    print(f"\nSuccessfully merged {len(merged_paths)}/{len(ADAPTERS)} models:")
    for name, path in merged_paths.items():
        print(f"  ✅ {name}: {path}")

    if len(merged_paths) < len(ADAPTERS):
        print(f"\n⚠️  {len(ADAPTERS) - len(merged_paths)} models failed to merge")

    print(f"\n{'='*80}")
    print("NEXT STEP: Update run_aqi_vllm.py")
    print(f"{'='*80}")
    print("Copy these paths to MODELS dict in run_aqi_vllm.py:")
    print()
    for name, path in merged_paths.items():
        model_key = name.replace('_baseline', '').upper() + '_Baseline'
        print(f'    "{model_key}": {{')
        print(f'        "local_path": "{path}",')
        print(f'        "hf_repo": None,  # Using local merged model')
        print(f'        ...')
        print(f'    }},')
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
