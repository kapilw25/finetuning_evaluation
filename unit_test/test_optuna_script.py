"""
Comprehensive tests for Llama3_BF16_adaptive_Optuna.py
Tests: py_compile, imports, AST syntax, function calls, dependencies

Run: python unit_test/test_optuna_script.py
"""

import sys
import ast
import py_compile
import importlib
import tempfile
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

OPTUNA_SCRIPT = project_root / "comparative_study/03a_CITA_Baseline/Llama3_BF16_adaptive_Optuna.py"

# ===================================================================
# Test 1: py_compile (Syntax Check)
# ===================================================================

def test_py_compile():
    """Test if script compiles without syntax errors"""
    print("\n" + "="*80)
    print("TEST 1: py_compile (Syntax Check)")
    print("="*80)

    try:
        with tempfile.NamedTemporaryFile(suffix='.pyc', delete=True) as f:
            py_compile.compile(str(OPTUNA_SCRIPT), cfile=f.name, doraise=True)
        print("✅ PASS: Script compiles without syntax errors")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ FAIL: Syntax error in script")
        print(f"   Error: {e}")
        return False


# ===================================================================
# Test 2: AST Parse (Advanced Syntax Check)
# ===================================================================

def test_ast_parse():
    """Test if script parses correctly with AST"""
    print("\n" + "="*80)
    print("TEST 2: AST Parse (Advanced Syntax Check)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(OPTUNA_SCRIPT))
        print(f"✅ PASS: AST parsed successfully")
        print(f"   Functions defined: {len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])}")
        print(f"   Classes defined: {len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])}")
        print(f"   Imports: {len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])}")
        return True
    except SyntaxError as e:
        print(f"❌ FAIL: AST parsing failed")
        print(f"   Error: {e}")
        return False


# ===================================================================
# Test 3: Import Dependencies
# ===================================================================

def test_imports():
    """Test if all required dependencies can be imported"""
    print("\n" + "="*80)
    print("TEST 3: Import Dependencies")
    print("="*80)

    required_imports = [
        ("torch", "PyTorch"),
        ("optuna", "Optuna"),
        ("transformers", "HuggingFace Transformers"),
        ("trl", "TRL (DPOConfig)"),
        ("peft", "PEFT (LoRA)"),
        ("datasets", "HuggingFace Datasets"),
    ]

    failed = []
    for module_name, display_name in required_imports:
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {display_name} ({module_name})")
        except ImportError as e:
            print(f"   ❌ {display_name} ({module_name}) - MISSING")
            failed.append(module_name)

    if failed:
        print(f"\n❌ FAIL: Missing dependencies: {', '.join(failed)}")
        return False
    else:
        print(f"\n✅ PASS: All dependencies available")
        return True


# ===================================================================
# Test 4: Custom Utils Imports
# ===================================================================

def test_custom_imports():
    """Test if custom utilities can be imported"""
    print("\n" + "="*80)
    print("TEST 4: Custom Utils Imports")
    print("="*80)

    custom_imports = [
        ("logging_utils", ["setup_training_logger", "restore_logging"]),
        ("model_utils", ["load_hf_token", "get_model_repo_name", "load_model_bf16", "setup_lora", "apply_torch_compile"]),
        ("data_prep.loader_pku", ["load_pku_combined_clear_contrast"]),
        ("data_prep.formatters", ["format_pku_for_cita_no_instruct"]),
        ("monitoring_callback", ["TrainingSummaryCallback"]),
        ("cita_trainer", ["CITATrainer"]),
    ]

    failed = []
    for module_name, functions in custom_imports:
        try:
            module = importlib.import_module(module_name)
            for func in functions:
                if not hasattr(module, func):
                    print(f"   ❌ {module_name}.{func} - NOT FOUND")
                    failed.append(f"{module_name}.{func}")
                else:
                    print(f"   ✅ {module_name}.{func}")
        except ImportError as e:
            print(f"   ❌ {module_name} - IMPORT FAILED: {e}")
            failed.append(module_name)

    if failed:
        print(f"\n❌ FAIL: Missing custom imports: {', '.join(failed)}")
        return False
    else:
        print(f"\n✅ PASS: All custom imports available")
        return True


# ===================================================================
# Test 5: Function Signatures
# ===================================================================

def test_function_signatures():
    """Test if key functions have correct signatures"""
    print("\n" + "="*80)
    print("TEST 5: Function Signatures")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Extract function definitions
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = [arg.arg for arg in node.args.args]

        # Check key functions
        expected = {
            "train_cita_trial": ["trial", "max_steps", "base_model"],
            "run_optuna_cita_search": ["n_trials", "max_steps", "base_model", "timeout_hours"],
        }

        for func_name, expected_args in expected.items():
            if func_name not in functions:
                print(f"   ❌ Function '{func_name}' not found")
                return False

            actual_args = functions[func_name]
            # Check if expected args are subset of actual args (allow defaults)
            missing = [arg for arg in expected_args if arg not in actual_args]

            if missing:
                print(f"   ❌ {func_name}({', '.join(actual_args)}) - missing: {missing}")
                return False
            else:
                print(f"   ✅ {func_name}({', '.join(actual_args)})")

        print(f"\n✅ PASS: All function signatures valid")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 6: Hyperparameter Ranges
# ===================================================================

def test_hyperparameter_ranges():
    """Test if hyperparameter ranges are correctly defined"""
    print("\n" + "="*80)
    print("TEST 6: Hyperparameter Ranges (iter9 compatibility)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        # Check lambda_kl range (should be lowered for 1354-step training)
        if 'lambda_kl = trial.suggest_float("lambda_kl", 0.0005, 0.0015' in source:
            print("   ✅ lambda_kl: [0.0005, 0.0015] (iter9 fix: lower range for longer training)")
        else:
            print("   ❌ lambda_kl: Range not matching iter9 spec")
            return False

        # Check warmup_ratio (should be epoch-agnostic)
        if 'warmup_ratio = trial.suggest_float("warmup_ratio", 0.20, 0.35' in source:
            print("   ✅ warmup_ratio: [0.20, 0.35] (iter7 fix: epoch-agnostic)")
        else:
            print("   ❌ warmup_ratio: Range not matching iter7 spec")
            return False

        # Check learning_rate (log scale)
        if 'learning_rate = trial.suggest_float("learning_rate", 8e-6, 1.2e-5, log=True' in source:
            print("   ✅ learning_rate: [8e-6, 1.2e-5] (log scale)")
        else:
            print("   ⚠️  learning_rate: Check log scale parameter")

        # Check beta
        if 'beta = trial.suggest_float("beta", 0.08, 0.12' in source:
            print("   ✅ beta: [0.08, 0.12] (DPO temperature)")
        else:
            print("   ❌ beta: Range not found")
            return False

        print(f"\n✅ PASS: Hyperparameter ranges match iter9 spec")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 7: Explosion Handler
# ===================================================================

def test_explosion_handler():
    """Test if explosion handler is implemented (iter9 requirement)"""
    print("\n" + "="*80)
    print("TEST 7: Explosion Handler (iter9 safety)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        checks = [
            ('except ValueError as e:', "ValueError exception handler"),
            ('"exploded" in str(e).lower()', "Explosion detection"),
            ('trial.report(margin, step)', "Report to Optuna TPE"),
            ('optuna.TrialPruned', "Mark trial as pruned"),
        ]

        for pattern, description in checks:
            if pattern in source:
                print(f"   ✅ {description}: {pattern}")
            else:
                print(f"   ❌ {description}: NOT FOUND")
                return False

        print(f"\n✅ PASS: Explosion handler implemented correctly")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 8: Eval Frequency
# ===================================================================

def test_eval_frequency():
    """Test if eval_steps is set to dynamic 20% (matches production)"""
    print("\n" + "="*80)
    print("TEST 8: Eval Frequency (dynamic 20%, matches production)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        # Check for checkpoint_interval calculation
        if 'checkpoint_interval = int(max_steps * 0.2)' in source:
            print("   ✅ checkpoint_interval = int(max_steps * 0.2) (dynamic 20%)")
        else:
            print("   ❌ checkpoint_interval calculation not found")
            return False

        # Check for eval_steps=checkpoint_interval
        if 'eval_steps=checkpoint_interval' in source:
            print("   ✅ eval_steps=checkpoint_interval (dynamic, matches production)")
        else:
            print("   ❌ eval_steps not set to checkpoint_interval")
            return False

        # Check for save_steps=checkpoint_interval
        if 'save_steps=checkpoint_interval' in source:
            print("   ✅ save_steps=checkpoint_interval (aligned with eval)")
        else:
            print("   ⚠️  save_steps not aligned with eval_steps")

        print(f"\n✅ PASS: Eval frequency matches production (20% dynamic)")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 9: Multi-Objective Return
# ===================================================================

def test_multi_objective_return():
    """Test if train_cita_trial returns 3 objectives"""
    print("\n" + "="*80)
    print("TEST 9: Multi-Objective Return")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        # Check return statement
        if 'return (\n        final_margin,' in source and \
           'final_accuracy if final_accuracy else 0.0,' in source and \
           '-final_chosen if final_chosen else 0.0' in source:
            print("   ✅ Returns 3 objectives: (margin, accuracy, -chosen)")
        else:
            print("   ❌ Multi-objective return not found")
            return False

        # Check study creation
        if 'directions=["maximize", "maximize", "maximize"]' in source:
            print("   ✅ Study configured for 3 objectives (all maximize)")
        else:
            print("   ❌ Study directions not configured correctly")
            return False

        print(f"\n✅ PASS: Multi-objective optimization configured correctly")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 10: No Gradient Clipping
# ===================================================================

def test_no_gradient_clipping():
    """Test that max_grad_norm is NOT set (iter9 design: find stable hyperparams)"""
    print("\n" + "="*80)
    print("TEST 10: No Gradient Clipping (iter9 design)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        # Check DPOConfig in train_cita_trial function
        dpo_config_start = source.find('training_args = DPOConfig(')
        dpo_config_end = source.find(')', dpo_config_start)
        dpo_config_section = source[dpo_config_start:dpo_config_end]

        if 'max_grad_norm' in dpo_config_section:
            print("   ❌ max_grad_norm found in DPOConfig")
            print("   ⚠️  iter9 design: Let Optuna find stable hyperparams (no clipping)")
            return False
        else:
            print("   ✅ max_grad_norm NOT set (correct - let Optuna find stable hyperparams)")

        print(f"\n✅ PASS: Gradient clipping intentionally omitted (iter9 design)")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 11: CUDAGraph Fix
# ===================================================================

def test_cudagraph_fix():
    """Test that CUDAGraph fix is present (prevents OOM with dynamic shapes)"""
    print("\n" + "="*80)
    print("TEST 11: CUDAGraph Fix (dynamic shapes OOM prevention)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        # Check if CUDAGraph fix is present
        if 'torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True' in source:
            print("   ✅ CUDAGraph fix present")
        else:
            print("   ❌ CUDAGraph fix MISSING")
            print("   ⚠️  This can cause OOM during eval with dynamic batch sizes")
            return False

        print(f"\n✅ PASS: CUDAGraph fix implemented")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 12: torch.compile() Disabled
# ===================================================================

def test_torch_compile_disabled():
    """Test that torch.compile() is DISABLED (prevents gradient checkpointing bug)"""
    print("\n" + "="*80)
    print("TEST 12: torch.compile() Disabled (gradient checkpointing bug fix)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        # Check if apply_torch_compile is commented out
        if '# model = apply_torch_compile(model)' in source:
            print("   ✅ apply_torch_compile() is commented out")
        else:
            print("   ❌ apply_torch_compile() NOT commented out")
            print("   ⚠️  This will cause KeyError: 'hidden_states'")
            return False

        # Check if warning message is present
        if 'torch.compile() disabled (prevents gradient checkpointing bug' in source:
            print("   ✅ Warning message present")
        else:
            print("   ⚠️  Warning message not found")

        print(f"\n✅ PASS: torch.compile() disabled (prevents crash)")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 13: Gradient Clipping Verification
# ===================================================================

def test_gradient_clipping_message():
    """Test that gradient clipping verification message is present"""
    print("\n" + "="*80)
    print("TEST 13: Gradient Clipping Verification Message")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        # Check if verification code is present
        if 'max_grad_norm = getattr(trainer.args' in source:
            print("   ✅ Gradient clipping verification present")
        else:
            print("   ❌ Gradient clipping verification MISSING")
            return False

        # Check for informational messages
        if 'Gradient clipping DISABLED (intentional for Optuna)' in source:
            print("   ✅ Informational message for disabled clipping")
        else:
            print("   ⚠️  Informational message not found")

        print(f"\n✅ PASS: Gradient clipping verification implemented")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 14: Model Stacking
# ===================================================================

def test_model_stacking():
    """Test if model stacking (DPO→CITA) is implemented"""
    print("\n" + "="*80)
    print("TEST 14: Model Stacking (DPO→CITA)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            source = f.read()

        checks = [
            ('if base_model:', "Conditional stacking"),
            ('PeftModel.from_pretrained(model, base_model', "Load DPO LoRA"),
            ('merge_and_unload()', "Merge adapters"),
            ('delattr(merged_model, \'peft_config\')', "Clear PEFT config"),
        ]

        for pattern, description in checks:
            if pattern in source:
                print(f"   ✅ {description}: {pattern}")
            else:
                print(f"   ❌ {description}: NOT FOUND")
                return False

        print(f"\n✅ PASS: Model stacking implemented correctly")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


# ===================================================================
# Test 15: No trial.report() in Explosion Handler
# ===================================================================

def test_no_trial_report_in_explosion():
    """Test that explosion handler doesn't call trial.report() (causes NotImplementedError for multi-objective)"""
    print("\n" + "="*80)
    print("TEST 15: No trial.report() in Explosion Handler (Multi-Objective Safe)")
    print("="*80)

    try:
        with open(OPTUNA_SCRIPT, 'r') as f:
            content = f.read()

        # Find the explosion handler section
        explosion_section_start = content.find('except ValueError as e:')
        explosion_section_end = content.find('raise optuna.TrialPruned', explosion_section_start)

        if explosion_section_start == -1 or explosion_section_end == -1:
            print("   ❌ Could not find explosion handler section")
            return False

        explosion_handler = content[explosion_section_start:explosion_section_end + 100]

        # Check that trial.report() is commented out or removed
        if 'trial.report(margin, step)' in explosion_handler and '# trial.report' not in explosion_handler:
            print("   ❌ trial.report() is NOT commented out in explosion handler")
            print("      This causes NotImplementedError for multi-objective optimization!")
            return False

        if '# trial.report(margin, step)' in explosion_handler:
            print("   ✅ trial.report() is properly commented out")
        else:
            print("   ✅ trial.report() not found (safe)")

        # Check for the warning comment
        if 'not supported for multi-objective' in explosion_handler:
            print("   ✅ Explanation comment present")

        print("\n✅ PASS: Explosion handler safe for multi-objective")
        return True

    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        return False


# ===================================================================
# Main Test Runner
# ===================================================================

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE TESTS FOR OPTUNA SCRIPT")
    print("="*80)
    print(f"Script: {OPTUNA_SCRIPT}")
    print(f"Size: {OPTUNA_SCRIPT.stat().st_size} bytes")
    print("="*80)

    tests = [
        test_py_compile,
        test_ast_parse,
        test_imports,
        test_custom_imports,
        test_function_signatures,
        test_hyperparameter_ranges,
        test_explosion_handler,
        test_eval_frequency,
        test_multi_objective_return,
        test_no_gradient_clipping,
        test_cudagraph_fix,
        test_torch_compile_disabled,
        test_gradient_clipping_message,
        test_model_stacking,
        test_no_trial_report_in_explosion,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("="*80)
    print(f"TOTAL: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED - Script ready for training")
        return 0
    else:
        print(f"❌ {total - passed} TEST(S) FAILED - Fix issues before training")
        return 1


if __name__ == "__main__":
    exit(main())
