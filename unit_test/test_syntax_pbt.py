#!/usr/bin/env python3
"""
Test Syntax Verification for PBT and Inference Scripts
Checks Python syntax without executing the scripts

Usage:
    python3 unit_test/test_syntax_pbt.py

This test is SAFE to run on M1 MacBook (16GB RAM)
"""

import sys
import py_compile
from pathlib import Path

def test_syntax():
    """Test Python syntax for PBT and related scripts"""
    print("\n" + "="*80)
    print("🧪 Testing Python Syntax (Safe for M1 MacBook)")
    print("="*80)

    project_root = Path.cwd()

    # Scripts to test
    scripts_to_test = [
        {
            "name": "PBT Training Script",
            "path": project_root / "comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py",
            "critical": True
        },
        {
            "name": "Inference Script",
            "path": project_root / "comparative_study/05_evaluation/inference_bf16.py",
            "critical": True
        },
        {
            "name": "PBT Trainer Utils",
            "path": project_root / "comparative_study/0c_utils/pbt_trainer.py",
            "critical": True
        },
        {
            "name": "CITA Trainer",
            "path": project_root / "comparative_study/0c_utils/cita_trainer.py",
            "critical": False
        },
        {
            "name": "Monitoring Callback",
            "path": project_root / "comparative_study/0c_utils/monitoring_callback.py",
            "critical": False
        },
    ]

    success_count = 0
    total_tests = len(scripts_to_test)
    errors = []

    for i, script in enumerate(scripts_to_test, 1):
        print(f"\n[{i}/{total_tests}] Testing: {script['name']}")
        print(f"   Path: {script['path']}")

        if not script['path'].exists():
            print(f"   ⚠️  File not found (skipping)")
            if script['critical']:
                errors.append(f"{script['name']}: File not found")
            continue

        try:
            # Compile Python file to check syntax
            py_compile.compile(str(script['path']), doraise=True)
            print(f"   ✅ Syntax OK")
            success_count += 1
        except py_compile.PyCompileError as e:
            print(f"   ❌ Syntax Error:")
            print(f"      {e}")
            if script['critical']:
                errors.append(f"{script['name']}: {e}")

    # Final summary
    print("\n" + "="*80)
    print(f"📊 Syntax Test Results: {success_count}/{total_tests} passed")
    print("="*80)

    if errors:
        print("\n❌ Critical Syntax Errors Found:")
        for error in errors:
            print(f"   - {error}")
        print("\nFix these errors before deploying to GH200.")
        return False
    else:
        print("🎉 All syntax checks passed! Scripts are syntactically correct.")
        print("\n✅ Next steps:")
        print("   1. Run import test: python3 unit_test/test_imports_pbt.py")
        print("   2. Run full test suite: python3 unit_test/test_all_pbt.py")
        print("   3. Deploy to GH200 and run PBT training")
        return True


if __name__ == "__main__":
    success = test_syntax()
    sys.exit(0 if success else 1)
