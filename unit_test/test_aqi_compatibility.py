#!/usr/bin/env python3
"""
Quick test to validate LoRA adapter compatibility with AQI evaluation

Run this BEFORE building remaining notebooks to catch issues early.
"""

import sys
import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add AQI eval to path
sys.path.insert(0, "/lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study/0a_AQI_EVAL_utils/src")
from aqi.aqi_dealign_chi_sil import *

print("="*80)
print("  AQI COMPATIBILITY TEST - Critical for ACL Deadline")
print("="*80)

# Configuration
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
ADAPTERS_TO_TEST = {
    "SFT_Baseline": "/lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study/01a_SFT_Baseline/lora_model_SFT_Baseline",
    "SFT_GRIT": "/lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study/01b_SFT_GRIT/lora_model_SFT_GRIT",
}

# Load tiny test dataset (10 samples only for speed)
print("\n1. Loading test dataset (10 samples)...")
try:
    from datasets import load_dataset
    test_df = load_dataset("hasnat79/litmus", split="train[:10]").to_pandas()

    # Add required columns
    if 'axiom' not in test_df.columns:
        test_df['axiom'] = 'overall'
    if 'prompt' in test_df.columns and 'input' not in test_df.columns:
        test_df = test_df.rename(columns={'prompt': 'input'})

    print(f"✅ Loaded {len(test_df)} test samples")
    print(f"   Columns: {test_df.columns.tolist()}")
except Exception as e:
    print(f"❌ FAILED to load dataset: {e}")
    sys.exit(1)

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Test each adapter
results = {}

for adapter_name, adapter_path in ADAPTERS_TO_TEST.items():
    print(f"\n{'='*80}")
    print(f"  Testing: {adapter_name}")
    print(f"  Path: {adapter_path}")
    print(f"{'='*80}")

    try:
        # 1. Check if adapter exists
        if not os.path.exists(adapter_path):
            print(f"❌ ERROR: Adapter path does not exist!")
            results[adapter_name] = "MISSING"
            continue

        print(f"\n2. Loading base model: {BASE_MODEL_NAME}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✅ Base model loaded")

        # 2. Load adapter
        print(f"\n3. Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"✅ Adapter loaded successfully")

        # 3. Merge and unload (AQI eval expects merged model)
        print(f"\n4. Merging adapter weights...")
        model = model.merge_and_unload()
        model.eval()
        print(f"✅ Adapter merged")

        # 4. Test inference (critical step)
        print(f"\n5. Testing inference...")
        test_prompt = "What is 2+2?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Prompt: {test_prompt}")
        print(f"   Response: {response}")
        print(f"✅ Inference works")

        # 5. Test AQI embedding extraction (most critical)
        print(f"\n6. Testing AQI embedding extraction...")
        try:
            # This is the actual function from baseEvalAqi.ipynb
            processed_df = process_model_data(
                model,
                tokenizer,
                test_df,
                model_name=adapter_name,
                cache_file=None  # No cache for test - will always recompute
            )
            print(f"✅ AQI embedding extraction works!")
            print(f"   Processed {len(processed_df)} samples")
            print(f"   Columns: {processed_df.columns.tolist()}")

            # Check if embeddings exist
            if 'embedding' in processed_df.columns:
                emb_shape = processed_df['embedding'].iloc[0].shape
                print(f"   Embedding shape: {emb_shape}")
            else:
                print(f"⚠️  WARNING: No 'embedding' column found!")

            results[adapter_name] = "✅ PASS"

        except Exception as e:
            print(f"❌ FAILED at AQI embedding extraction: {e}")
            import traceback
            traceback.print_exc()
            results[adapter_name] = f"❌ FAIL: {str(e)[:50]}"

        # Cleanup
        del base_model, model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n✅ Cleanup complete")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[adapter_name] = f"❌ ERROR: {str(e)[:50]}"

# Final Report
print("\n" + "="*80)
print("  FINAL COMPATIBILITY REPORT")
print("="*80)

all_pass = True
for adapter_name, result in results.items():
    print(f"{adapter_name:20s}: {result}")
    if "PASS" not in result:
        all_pass = False

print("="*80)

if all_pass:
    print("\n✅✅✅ SUCCESS: All adapters compatible with AQI evaluation!")
    print("      → SAFE to proceed with building remaining notebooks")
    print("      → Pipeline is validated and working")
    sys.exit(0)
else:
    print("\n❌❌❌ FAILURE: Compatibility issues detected!")
    print("      → DO NOT build remaining notebooks yet")
    print("      → FIX these issues first, then apply fixes to all notebooks")
    print("      → Check error messages above for details")
    sys.exit(1)
