#!/usr/bin/env python3
"""
Test Import Verification for PBT Script
Tests all required imports without running training

Usage:
    python3 unit_test/test_imports_pbt.py

This test is SAFE to run on M1 MacBook (16GB RAM)
"""

import sys
from pathlib import Path

def test_imports():
    """Test all imports required by PBT script"""
    print("\n" + "="*80)
    print("🧪 Testing PBT Script Imports (Safe for M1 MacBook)")
    print("="*80)

    success_count = 0
    total_tests = 9

    # Test 1: transformers
    print("\n[1/9] Testing transformers...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✅ transformers imports OK")
        print(f"   - AutoModelForCausalLM: {AutoModelForCausalLM.__module__}")
        print(f"   - AutoTokenizer: {AutoTokenizer.__module__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False

    # Test 2: trl
    print("\n[2/9] Testing trl (DPOConfig)...")
    try:
        from trl import DPOConfig
        print("✅ trl.DPOConfig import OK")
        print(f"   - DPOConfig: {DPOConfig.__module__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ trl.DPOConfig import failed: {e}")
        print(f"   💡 Hint: Install with 'pip install trl'")
        return False

    # Test 3: peft
    print("\n[3/9] Testing peft (LoRA)...")
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        print("✅ peft imports OK")
        print(f"   - LoraConfig: {LoraConfig.__module__}")
        print(f"   - get_peft_model: Available")
        print(f"   - prepare_model_for_kbit_training: Available")
        success_count += 1
    except ImportError as e:
        print(f"❌ peft import failed: {e}")
        print(f"   💡 Hint: Install with 'pip install peft'")
        return False

    # Test 4: ray
    print("\n[4/9] Testing ray (PBT framework)...")
    try:
        import ray
        from ray import tune
        from ray.tune.schedulers import PopulationBasedTraining
        print("✅ ray imports OK")
        print(f"   - ray version: {ray.__version__}")
        print(f"   - tune: {tune.__name__}")
        print(f"   - PopulationBasedTraining: Available")
        success_count += 1
    except ImportError as e:
        print(f"❌ ray import failed: {e}")
        print(f"   💡 Hint: Install with 'pip install ray[tune]'")
        return False

    # Test 5: torch
    print("\n[5/9] Testing torch...")
    try:
        import torch
        print("✅ torch import OK")
        print(f"   - torch version: {torch.__version__}")
        print(f"   - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   - CUDA version: {torch.version.cuda}")
            print(f"   - GPU count: {torch.cuda.device_count()}")
        success_count += 1
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False

    # Test 6: data_prep
    print("\n[6/9] Testing project imports (data_prep)...")
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

    try:
        from data_prep import load_pku_filtered, format_dataset
        print("✅ data_prep imports OK")
        print(f"   - load_pku_filtered: Available")
        print(f"   - format_dataset: Available")
        success_count += 1
    except ImportError as e:
        print(f"❌ data_prep import failed: {e}")
        print(f"   💡 Hint: Check that 0c_utils/data_prep/ exists")
        return False

    # Test 7: cita_trainer
    print("\n[7/9] Testing project imports (cita_trainer)...")
    try:
        from cita_trainer import CITATrainer
        print("✅ cita_trainer import OK")
        print(f"   - CITATrainer: {CITATrainer.__module__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ cita_trainer import failed: {e}")
        print(f"   💡 Hint: Check that 0c_utils/cita_trainer.py exists")
        return False

    # Test 8: monitoring_callback
    print("\n[8/9] Testing project imports (monitoring_callback)...")
    try:
        from monitoring_callback import GibberishDetectionCallback
        print("✅ monitoring_callback import OK")
        print(f"   - GibberishDetectionCallback: {GibberishDetectionCallback.__module__}")
        success_count += 1
    except ImportError as e:
        print(f"❌ monitoring_callback import failed: {e}")
        print(f"   💡 Hint: Check that 0c_utils/monitoring_callback.py exists")
        return False

    # Test 9: pbt_trainer
    print("\n[9/9] Testing project imports (pbt_trainer)...")
    try:
        from pbt_trainer import (
            create_pbt_scheduler,
            run_pbt_training,
            print_best_hyperparameters,
            save_best_config
        )
        print("✅ pbt_trainer imports OK")
        print(f"   - create_pbt_scheduler: Available")
        print(f"   - run_pbt_training: Available")
        print(f"   - print_best_hyperparameters: Available")
        print(f"   - save_best_config: Available")
        success_count += 1
    except ImportError as e:
        print(f"❌ pbt_trainer import failed: {e}")
        print(f"   💡 Hint: Check that 0c_utils/pbt_trainer.py exists")
        return False

    # Final summary
    print("\n" + "="*80)
    print(f"📊 Import Test Results: {success_count}/{total_tests} passed")
    print("="*80)

    if success_count == total_tests:
        print("🎉 All imports successful! PBT script is ready for GH200.")
        print("\n✅ Next steps:")
        print("   1. Run syntax check: python3 unit_test/test_syntax_pbt.py")
        print("   2. Run full test suite: python3 unit_test/test_all_pbt.py")
        print("   3. Deploy to GH200 and run PBT training")
        return True
    else:
        print("❌ Some imports failed. Fix errors before deploying to GH200.")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
