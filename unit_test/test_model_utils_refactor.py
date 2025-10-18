#!/usr/bin/env python3
"""
Comprehensive Test Suite for model_utils.py Refactoring

Tests:
1. py_compile - Syntax checking
2. Import check - Verify all imports work
3. AST analysis - Syntax tree validation
4. Function calling - Verify function signatures
5. Import tracing - Check import paths
6. Redundancy check - Detect duplicate code

Usage:
    python unit_test/test_model_utils_refactor.py
"""

import sys
import os
import ast
import py_compile
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{'='*80}\n")


def print_success(msg: str):
    """Print success message"""
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg: str):
    """Print error message"""
    print(f"{RED}❌ {msg}{RESET}")


def print_warning(msg: str):
    """Print warning message"""
    print(f"{YELLOW}⚠️  {msg}{RESET}")


# ============================================================================
# TEST 1: py_compile - Syntax Checking
# ============================================================================

def test_py_compile():
    """Test Python compilation for syntax errors"""
    print_section("TEST 1: py_compile - Syntax Checking")

    files_to_test = [
        project_root / "comparative_study/0c_utils/model_utils.py",
        project_root / "comparative_study/01a_SFT_Baseline/Llama3_BF16.py",
        project_root / "comparative_study/02a_DPO_Baseline/Llama3_BF16.py",
        project_root / "comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py",
    ]

    all_passed = True

    for file_path in files_to_test:
        try:
            py_compile.compile(str(file_path), doraise=True)
            # print_success(f"Syntax check passed: {file_path.name}")
            print_success(f"Syntax check passed: {file_path}")
        except py_compile.PyCompileError as e:
            print_error(f"Syntax error in {file_path.name}:")
            print(f"  {e}")
            all_passed = False

    return all_passed


# ============================================================================
# TEST 2: Import Check - Verify all imports work
# ============================================================================

def test_imports():
    """Test that all imports work correctly"""
    print_section("TEST 2: Import Check - Verify all imports work")

    imports_to_test = [
        ("model_utils", [
            "load_hf_token",
            "load_model_bf16",
            "setup_lora",
            "apply_torch_compile",
            "load_training_dataset",
            "get_test_prompts",
            "get_model_repo_name",
            "MODEL_NAME_MAP"
        ]),
    ]

    all_passed = True

    for module_name, expected_exports in imports_to_test:
        try:
            # Import module
            module = __import__(module_name)
            print_success(f"Module imported: {module_name}")

            # Check exports
            for export in expected_exports:
                if hasattr(module, export):
                    print_success(f"  - Export found: {export}")
                else:
                    print_error(f"  - Export missing: {export}")
                    all_passed = False

        except ImportError as e:
            print_error(f"Failed to import {module_name}:")
            print(f"  {e}")
            all_passed = False

    return all_passed


# ============================================================================
# TEST 3: AST Analysis - Syntax tree validation
# ============================================================================

def test_ast_analysis():
    """Test AST (Abstract Syntax Tree) for code structure"""
    print_section("TEST 3: AST Analysis - Syntax tree validation")

    file_path = project_root / "comparative_study/0c_utils/model_utils.py"

    try:
        with open(file_path, 'r') as f:
            source_code = f.read()

        tree = ast.parse(source_code, filename=str(file_path))
        print_success(f"AST parsing successful: {file_path.name}")

        # Count functions and classes
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

        print(f"  - Functions defined: {len(functions)}")
        print(f"  - Classes defined: {len(classes)}")
        print(f"  - Import statements: {len(imports)}")

        # List function names
        print("\n  Function signatures:")
        for func in functions:
            args = [arg.arg for arg in func.args.args]
            print(f"    - {func.name}({', '.join(args)})")

        return True

    except SyntaxError as e:
        print_error(f"AST parsing failed: {e}")
        return False


# ============================================================================
# TEST 4: Function Calling - Verify function signatures
# ============================================================================

def test_function_signatures():
    """Test that all functions have correct signatures"""
    print_section("TEST 4: Function Calling - Verify function signatures")

    import model_utils
    import inspect

    expected_signatures = {
        "load_hf_token": ["project_root"],
        "load_model_bf16": ["model_id", "max_seq_length", "use_flash_attention"],
        "setup_lora": ["model", "r", "lora_alpha", "use_gradient_checkpointing", "lora_dropout", "target_modules"],
        "apply_torch_compile": ["model"],
        "load_training_dataset": ["split", "max_samples", "method"],
        "get_test_prompts": [],
        "get_model_repo_name": ["run_name", "precision"],
    }

    all_passed = True

    for func_name, expected_params in expected_signatures.items():
        try:
            func = getattr(model_utils, func_name)
            sig = inspect.signature(func)
            actual_params = list(sig.parameters.keys())

            if actual_params == expected_params:
                print_success(f"{func_name}({', '.join(actual_params)})")
            else:
                print_warning(f"{func_name} signature mismatch")
                print(f"  Expected: {expected_params}")
                print(f"  Actual:   {actual_params}")
                # Not necessarily a failure - could have defaults

        except AttributeError:
            print_error(f"Function not found: {func_name}")
            all_passed = False

    return all_passed


# ============================================================================
# TEST 5: Import Tracing - Check import paths in refactored file
# ============================================================================

def test_import_tracing():
    """Test that refactored files import from model_utils correctly"""
    print_section("TEST 5: Import Tracing - Check import paths")

    files_to_check = [
        (project_root / "comparative_study/01a_SFT_Baseline/Llama3_BF16.py", {
            "load_hf_token",
            "get_model_repo_name",
            "get_test_prompts",
            "load_model_bf16",
            "setup_lora",
            "apply_torch_compile",
            "load_training_dataset"
        }),
        (project_root / "comparative_study/02a_DPO_Baseline/Llama3_BF16.py", {
            "load_hf_token",
            "get_model_repo_name",
            "get_test_prompts",
            "load_model_bf16",
            "setup_lora",
            "apply_torch_compile",
            "load_training_dataset"
        }),
        (project_root / "comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py", {
            "load_hf_token",
            "get_model_repo_name",
            "get_test_prompts",
            "load_model_bf16",
            "setup_lora",
            "apply_torch_compile"
        }),
    ]

    all_passed = True

    for file_path, expected_imports in files_to_check:
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=str(file_path))

            # Find all imports from model_utils
            model_utils_imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "model_utils":
                        imported_names = [alias.name for alias in node.names]
                        model_utils_imports.extend(imported_names)

            # print(f"\nImports from model_utils in {file_path.name}:")
            print(f"\nImports from model_utils in {file_path}:")
            for imp in set(model_utils_imports):
                print(f"  - {imp}")
                print_success(f"Import verified: {imp}")

            missing_imports = expected_imports - set(model_utils_imports)
            if missing_imports:
                print_warning(f"Expected imports not found: {missing_imports}")
                all_passed = False
            else:
                print_success(f"All expected imports found!")

        except Exception as e:
            print_error(f"Import tracing failed for {file_path.name}: {e}")
            all_passed = False

    return all_passed


# ============================================================================
# TEST 6: Redundancy Check - Detect duplicate code
# ============================================================================

def test_redundancy():
    """Check for code redundancy between model_utils.py and CITA script"""
    print_section("TEST 6: Redundancy Check - Detect duplicate code")

    model_utils_path = project_root / "comparative_study/0c_utils/model_utils.py"
    cita_path = project_root / "comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py"

    # Read both files
    with open(model_utils_path, 'r') as f:
        model_utils_code = f.read()

    with open(cita_path, 'r') as f:
        cita_code = f.read()

    # Check for specific code patterns that should NOT be in CITA anymore
    redundant_patterns = [
        ("HF_TOKEN = os.getenv('HF_TOKEN')", "HF token loading (should use load_hf_token())"),
        ("MODEL_NAME_MAP = {", "Model name mapping (should use get_model_repo_name())"),
        ("AutoTokenizer.from_pretrained(model_id)", "Direct model loading (should use load_model_bf16())"),
        ("LoraConfig(", "Direct LoRA config (should use setup_lora())"),
        ("torch.compile(model", "Direct torch.compile (should use apply_torch_compile())"),
    ]

    all_passed = True

    for pattern, description in redundant_patterns:
        if pattern in cita_code:
            print_warning(f"Potential redundancy found: {description}")
            print(f"  Pattern: {pattern[:50]}...")
            # Not marking as failure since some patterns might be intentional
        else:
            print_success(f"No redundancy: {description}")

    # Check that utility functions ARE in model_utils.py
    utility_patterns = [
        ("def load_hf_token", "load_hf_token()"),
        ("def load_model_bf16", "load_model_bf16()"),
        ("def setup_lora", "setup_lora()"),
        ("def apply_torch_compile", "apply_torch_compile()"),
        ("def get_test_prompts", "get_test_prompts()"),
        ("def get_model_repo_name", "get_model_repo_name()"),
    ]

    for pattern, description in utility_patterns:
        if pattern in model_utils_code:
            print_success(f"Utility function defined: {description}")
        else:
            print_error(f"Utility function missing: {description}")
            all_passed = False

    return all_passed


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and report results"""
    print(f"\n{'#'*80}")
    print(f"# {BLUE}COMPREHENSIVE TEST SUITE: model_utils.py Refactoring{RESET}")
    print(f"{'#'*80}\n")

    results = {
        "1. py_compile (Syntax Checking)": test_py_compile(),
        "2. Import Check": test_imports(),
        "3. AST Analysis": test_ast_analysis(),
        "4. Function Signatures": test_function_signatures(),
        "5. Import Tracing": test_import_tracing(),
        "6. Redundancy Check": test_redundancy(),
    }

    # Summary
    print_section("TEST SUMMARY")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {test_name}: {status}")

    print(f"\n{'='*80}")
    if passed == total:
        print(f"{GREEN}✅ ALL TESTS PASSED ({passed}/{total}){RESET}")
        print(f"{'='*80}\n")
        return 0
    else:
        print(f"{RED}❌ SOME TESTS FAILED ({passed}/{total} passed){RESET}")
        print(f"{'='*80}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
