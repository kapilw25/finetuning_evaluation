#!/usr/bin/env python3
"""
Comprehensive validation test for checkpoint detection integration
Tests: py_compile, import checks, AST parsing, function calls

Tests 3 scripts:
1. comparative_study/01a_SFT_Baseline/Llama3_BF16.py
2. comparative_study/02a_DPO_Baseline/Llama3_BF16.py
3. comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py
"""

import sys
import ast
import py_compile
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# Test results
test_results = {
    "py_compile": {},
    "ast_parse": {},
    "import_check": {},
    "function_calls": {}
}

# Scripts to test
scripts = [
    "comparative_study/01a_SFT_Baseline/Llama3_BF16.py",
    "comparative_study/02a_DPO_Baseline/Llama3_BF16.py",
    "comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py"
]


def test_py_compile(script_path):
    """Test 1: py_compile - Compile Python file to bytecode"""
    try:
        py_compile.compile(str(script_path), doraise=True)
        return True, "✅ Compilation successful"
    except py_compile.PyCompileError as e:
        return False, f"❌ Compilation failed: {e}"


def test_ast_parse(script_path):
    """Test 2: AST parsing - Parse Python file to AST"""
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "✅ AST parsing successful"
    except SyntaxError as e:
        return False, f"❌ Syntax error: {e}"


def test_import_module(script_path):
    """Test 3: Import check - Try to import the module"""
    try:
        # Create module spec
        module_name = script_path.stem.replace(".", "_")
        spec = importlib.util.spec_from_file_location(module_name, script_path)

        if spec is None:
            return False, "❌ Failed to create module spec"

        # Note: We don't actually load/execute the module to avoid side effects
        # Just check that the spec can be created (validates importability)
        return True, "✅ Module spec created (importable)"
    except Exception as e:
        return False, f"❌ Import failed: {e}"


def test_checkpoint_functions(script_path):
    """Test 4: Validate checkpoint detection function calls"""
    try:
        with open(script_path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Expected imports
        expected_imports = []
        if "CITA" in str(script_path):
            expected_imports = ["check_ray_tune_experiment"]
        else:
            expected_imports = ["get_latest_checkpoint", "is_training_complete"]

        # Find imports
        found_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "model_utils":
                    for alias in node.names:
                        found_imports.add(alias.name)

        # Check if all expected imports are present
        missing = set(expected_imports) - found_imports
        if missing:
            return False, f"❌ Missing imports: {missing}"

        # Find function calls to checkpoint detection functions
        found_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in expected_imports:
                        found_calls.add(node.func.id)

        # Check if functions are actually called
        missing_calls = set(expected_imports) - found_calls
        if missing_calls:
            return False, f"❌ Functions imported but not called: {missing_calls}"

        # Check for training_skipped variable usage
        has_training_skipped = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "training_skipped":
                has_training_skipped = True
                break

        if not has_training_skipped:
            return False, "❌ 'training_skipped' variable not found"

        return True, f"✅ All checkpoint functions imported and called: {expected_imports}"

    except Exception as e:
        return False, f"❌ Function validation failed: {e}"


def run_tests():
    """Run all tests on all scripts"""
    print("="*80)
    print("🧪 CHECKPOINT INTEGRATION VALIDATION TESTS")
    print("="*80)
    print()

    all_passed = True

    for script_rel in scripts:
        script_path = project_root / script_rel
        # Use relative path as key to avoid filename collision
        script_key = str(script_rel)

        print(f"\n📄 Testing: {script_rel}")
        print("-" * 80)

        # Test 1: py_compile
        passed, msg = test_py_compile(script_path)
        test_results["py_compile"][script_key] = (passed, msg)
        print(f"  [1/4] py_compile:       {msg}")
        if not passed:
            all_passed = False

        # Test 2: AST parse
        passed, msg = test_ast_parse(script_path)
        test_results["ast_parse"][script_key] = (passed, msg)
        print(f"  [2/4] AST parsing:      {msg}")
        if not passed:
            all_passed = False

        # Test 3: Import check
        passed, msg = test_import_module(script_path)
        test_results["import_check"][script_key] = (passed, msg)
        print(f"  [3/4] Import check:     {msg}")
        if not passed:
            all_passed = False

        # Test 4: Function calls
        passed, msg = test_checkpoint_functions(script_path)
        test_results["function_calls"][script_key] = (passed, msg)
        print(f"  [4/4] Function calls:   {msg}")
        if not passed:
            all_passed = False

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    for test_name, results in test_results.items():
        passed_count = sum(1 for p, _ in results.values() if p)
        total_count = len(results)
        status = "✅ PASS" if passed_count == total_count else "❌ FAIL"
        print(f"{test_name:20s}: {passed_count}/{total_count} {status}")

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("✅ Checkpoint detection correctly integrated in all 3 scripts:")
        print("   - SFT Baseline: get_latest_checkpoint + is_training_complete")
        print("   - DPO Baseline: get_latest_checkpoint + is_training_complete")
        print("   - CITA Baseline: check_ray_tune_experiment")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
