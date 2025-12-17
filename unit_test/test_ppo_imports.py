"""
PPO Baseline Import Test Script
Run this on A100-40GB to verify all imports work correctly.

Usage:
    python unit_test/test_ppo_imports.py
"""

import sys
from pathlib import Path

# Add utils to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

def test_imports():
    """Test all imports required by PPO baseline script"""
    results = []

    print("="*80)
    print("PPO Baseline Import Test")
    print("="*80)

    # 1. Standard library
    print("\n1. Standard library imports...")
    try:
        from pathlib import Path
        from datetime import datetime
        import gc
        import argparse
        import os
        results.append(("Standard library", True, None))
        print("   ✅ OK")
    except Exception as e:
        results.append(("Standard library", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 2. PyTorch
    print("\n2. PyTorch imports...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        results.append(("PyTorch", True, f"CUDA: {cuda_available}, Device: {device_name}"))
        print(f"   ✅ OK - CUDA: {cuda_available}, Device: {device_name}")

        # Test CUDAGraph config
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        print("   ✅ CUDAGraph config set successfully")
    except Exception as e:
        results.append(("PyTorch", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 3. Transformers
    print("\n3. Transformers imports...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import transformers
        results.append(("Transformers", True, f"v{transformers.__version__}"))
        print(f"   ✅ OK - v{transformers.__version__}")
    except Exception as e:
        results.append(("Transformers", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 4. TRL
    print("\n4. TRL imports...")
    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
        import trl
        results.append(("TRL", True, f"v{trl.__version__}"))
        print(f"   ✅ OK - v{trl.__version__}")

        # Check PPOConfig supports our parameters
        import inspect
        sig = inspect.signature(PPOConfig)
        params = list(sig.parameters.keys())
        needed = ['learning_rate', 'warmup_steps', 'weight_decay', 'lr_scheduler_type']
        missing = [p for p in needed if p not in params]
        if missing:
            print(f"   ⚠️  Missing PPOConfig params: {missing}")
        else:
            print(f"   ✅ All required PPOConfig params available")
    except Exception as e:
        results.append(("TRL", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 5. PEFT
    print("\n5. PEFT imports...")
    try:
        from peft import LoraConfig, PeftModel
        import peft
        results.append(("PEFT", True, f"v{peft.__version__}"))
        print(f"   ✅ OK - v{peft.__version__}")
    except Exception as e:
        results.append(("PEFT", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 6. model_utils
    print("\n6. model_utils imports...")
    try:
        from model_utils import (
            load_hf_token,
            get_model_repo_name,
            get_test_prompts,
            get_latest_checkpoint,
            is_training_complete,
            log_gpu_memory_start,
            log_gpu_memory_end,
            MODEL_NAME_MAP
        )
        # Check PPO is in MODEL_NAME_MAP
        ppo_keys = [k for k in MODEL_NAME_MAP.keys() if 'PPO' in k]
        results.append(("model_utils", True, f"PPO keys: {ppo_keys}"))
        print(f"   ✅ OK - PPO keys in MODEL_NAME_MAP: {ppo_keys}")
    except Exception as e:
        results.append(("model_utils", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 7. data_prep
    print("\n7. data_prep imports...")
    try:
        from data_prep.loader_pku import load_pku_combined_clear_contrast
        results.append(("data_prep", True, None))
        print("   ✅ OK")
    except Exception as e:
        results.append(("data_prep", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 8. push_automation
    print("\n8. push_automation imports...")
    try:
        from push_automation import PushAutomation
        results.append(("push_automation", True, None))
        print("   ✅ OK")
    except Exception as e:
        results.append(("push_automation", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 9. logging_utils
    print("\n9. logging_utils imports...")
    try:
        from logging_utils import setup_training_logger, restore_logging
        results.append(("logging_utils", True, None))
        print("   ✅ OK")
    except Exception as e:
        results.append(("logging_utils", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # 10. HuggingFace Hub
    print("\n10. HuggingFace Hub imports...")
    try:
        from huggingface_hub import repo_exists
        import huggingface_hub
        results.append(("huggingface_hub", True, f"v{huggingface_hub.__version__}"))
        print(f"   ✅ OK - v{huggingface_hub.__version__}")
    except Exception as e:
        results.append(("huggingface_hub", False, str(e)))
        print(f"   ❌ FAILED: {e}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, status, _ in results if status)
    failed = sum(1 for _, status, _ in results if not status)

    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed > 0:
        print("\nFailed imports:")
        for name, status, error in results:
            if not status:
                print(f"  ❌ {name}: {error}")

    print("\n" + "="*80)

    return failed == 0


def test_ppo_script_syntax():
    """Verify PPO script syntax"""
    print("\n" + "="*80)
    print("PPO Script Syntax Test")
    print("="*80)

    ppo_script = project_root / "comparative_study" / "02b_PPO_Baseline" / "Llama3_BF16.py"

    # 1. py_compile
    print("\n1. py_compile check...")
    try:
        import py_compile
        py_compile.compile(str(ppo_script), doraise=True)
        print("   ✅ py_compile passed")
    except py_compile.PyCompileError as e:
        print(f"   ❌ py_compile FAILED: {e}")
        return False

    # 2. AST parsing
    print("\n2. AST parsing...")
    try:
        import ast
        with open(ppo_script, 'r') as f:
            code = f.read()
        tree = ast.parse(code)

        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        print(f"   ✅ AST parsing passed - {len(functions)} functions defined")
        print(f"   Functions: {functions}")
    except SyntaxError as e:
        print(f"   ❌ AST parsing FAILED: {e}")
        return False

    print("\n" + "="*80)
    return True


def test_gpu_memory():
    """Test GPU memory availability"""
    print("\n" + "="*80)
    print("GPU Memory Test")
    print("="*80)

    try:
        import torch
        if not torch.cuda.is_available():
            print("   ⚠️  CUDA not available")
            return True

        device = torch.device("cuda:0")
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        free_memory = (torch.cuda.get_device_properties(device).total_memory -
                      torch.cuda.memory_allocated(device)) / (1024**3)

        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Total memory: {total_memory:.1f} GB")
        print(f"   Free memory: {free_memory:.1f} GB")

        # PPO requires ~30GB for policy + value head + ref model + reward model
        if total_memory >= 40:
            print("   ✅ A100-40GB detected - sufficient for PPO training")
        elif total_memory >= 24:
            print("   ⚠️  24GB GPU - may need reduced batch size")
        else:
            print("   ❌ Insufficient GPU memory for PPO training")

        print("\n" + "="*80)
        return True
    except Exception as e:
        print(f"   ❌ GPU test FAILED: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PPO BASELINE FULL IMPORT TEST")
    print("Run on A100-40GB GPU")
    print("="*80)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test syntax
    if not test_ppo_script_syntax():
        all_passed = False

    # Test GPU
    if not test_gpu_memory():
        all_passed = False

    # Final result
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - PPO baseline ready for training")
    else:
        print("❌ SOME TESTS FAILED - Fix issues before training")
    print("="*80 + "\n")

    sys.exit(0 if all_passed else 1)
