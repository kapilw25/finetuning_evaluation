#!/usr/bin/env python3
"""
Quick file structure validation (no GPU needed)
Checks if LoRA adapters have correct structure for AQI eval
"""

import os
import json

print("="*80)
print("  ADAPTER FILE STRUCTURE CHECK (No GPU needed)")
print("="*80)

ADAPTERS_TO_CHECK = {
    "SFT_Baseline": "/lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study/01a_SFT_Baseline/lora_model_SFT_Baseline",
    "SFT_GRIT": "/lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study/01b_SFT_GRIT/lora_model_SFT_GRIT",
}

REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.bin",  # or adapter_model.safetensors
]

all_good = True

for adapter_name, adapter_path in ADAPTERS_TO_CHECK.items():
    print(f"\n{'='*80}")
    print(f"Checking: {adapter_name}")
    print(f"Path: {adapter_path}")
    print(f"{'='*80}")

    # 1. Check if directory exists
    if not os.path.exists(adapter_path):
        print(f"❌ FAIL: Directory does not exist!")
        all_good = False
        continue

    if not os.path.isdir(adapter_path):
        print(f"❌ FAIL: Path is not a directory!")
        all_good = False
        continue

    print(f"✅ Directory exists")

    # 2. List all files
    files = os.listdir(adapter_path)
    print(f"\nFiles found ({len(files)}):")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(adapter_path, f))
        print(f"  - {f:40s} ({size/1024/1024:.2f} MB)")

    # 3. Check required files
    print(f"\nRequired files check:")
    for req_file in REQUIRED_FILES:
        # Check for both .bin and .safetensors
        if req_file == "adapter_model.bin":
            has_bin = "adapter_model.bin" in files
            has_safetensors = "adapter_model.safetensors" in files
            if has_bin or has_safetensors:
                found = "adapter_model.bin" if has_bin else "adapter_model.safetensors"
                print(f"  ✅ {found}")
            else:
                print(f"  ❌ Missing: {req_file} (or .safetensors)")
                all_good = False
        else:
            if req_file in files:
                print(f"  ✅ {req_file}")
            else:
                print(f"  ❌ Missing: {req_file}")
                all_good = False

    # 4. Validate adapter_config.json
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            print(f"\nAdapter config validation:")
            print(f"  - peft_type: {config.get('peft_type', 'MISSING')}")
            print(f"  - r (rank): {config.get('r', 'MISSING')}")
            print(f"  - lora_alpha: {config.get('lora_alpha', 'MISSING')}")
            print(f"  - target_modules: {config.get('target_modules', 'MISSING')}")

            # Check for expected values
            if config.get('peft_type') != 'LORA':
                print(f"  ⚠️  WARNING: peft_type is not 'LORA'")
                all_good = False

            if config.get('r') is None:
                print(f"  ⚠️  WARNING: LoRA rank (r) not found")
                all_good = False

        except Exception as e:
            print(f"  ❌ FAIL: Could not parse adapter_config.json: {e}")
            all_good = False

print(f"\n{'='*80}")
print(f"  FINAL RESULT")
print(f"{'='*80}")

if all_good:
    print("\n✅ All adapter files look good!")
    print("   → File structure is correct")
    print("   → Still need to test GPU loading (run full test after training)")
else:
    print("\n❌ Issues detected in adapter files!")
    print("   → Fix these before proceeding")

print(f"{'='*80}")
