#!/usr/bin/env python3
"""
Test imports after refactoring formatters and data_prep module.
Verifies new function names work and old ones are removed.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

def test_imports():
    """Test all imports after refactoring."""

    print("Testing imports after refactoring...")
    print("=" * 50)

    # Track results
    success = []
    failures = []

    # Test 1: Import module
    try:
        from data_prep import formatters
        success.append("✓ Module import: formatters")
    except Exception as e:
        failures.append(f"✗ Module import failed: {e}")

    # Test 2: Import new core formatter functions
    new_formatters = [
        'format_sft_NoInstruct',
        'format_sft_Instruct',
        'format_dpo_NoInstruct',
        'format_dpo_Instruct',
        'format_cita_Instruct',
        'format_dataset'
    ]

    for formatter in new_formatters:
        try:
            exec(f"from data_prep import {formatter}")
            success.append(f"✓ New formatter: {formatter}")
        except ImportError:
            failures.append(f"✗ New formatter missing: {formatter}")

    # Test 3: Import new aliases
    new_aliases = [
        'format_pku_for_sft',  # Exception: no suffix
        'format_pku_for_sft_NoInstruct',
        'format_pku_for_sft_Instruct',
        'format_pku_for_dpo_NoInstruct',
        'format_pku_for_dpo_Instruct',
        'format_pku_for_cita_NoInstruct',
        'format_pku_for_cita_Instruct'
    ]

    for alias in new_aliases:
        try:
            exec(f"from data_prep import {alias}")
            success.append(f"✓ Alias: {alias}")
        except ImportError:
            failures.append(f"✗ Alias missing: {alias}")

    # Test 4: Verify old names are removed (should fail)
    old_names = ['format_sft', 'format_dpo', 'format_cita']

    for old_name in old_names:
        try:
            exec(f"from data_prep import {old_name}")
            failures.append(f"✗ Old name still exists (should be removed): {old_name}")
        except ImportError:
            success.append(f"✓ Old name removed: {old_name}")

    # Test 5: Import loader functions
    loader_functions = [
        'load_pku_filtered',
        'get_safe_unsafe_responses',
        'synthesize_system_instruction'
    ]

    for func in loader_functions:
        try:
            exec(f"from data_prep import {func}")
            success.append(f"✓ Loader function: {func}")
        except ImportError:
            failures.append(f"✗ Loader function missing: {func}")

    # Print results
    print("\nSuccessful imports:")
    for s in success:
        print(f"  {s}")

    if failures:
        print("\nFailed imports:")
        for f in failures:
            print(f"  {f}")

    # Summary
    print("\n" + "=" * 50)
    print(f"Results: {len(success)} passed, {len(failures)} failed")

    if not failures:
        print("✓ All import tests passed successfully!")
        return 0
    else:
        print("✗ Some imports failed. Please review the failures above.")
        return 1

if __name__ == "__main__":
    exit_code = test_imports()
    sys.exit(exit_code)