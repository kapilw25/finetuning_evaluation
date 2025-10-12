#!/usr/bin/env python3
"""
Comprehensive Test Suite for PBT Script
Runs all pre-deployment verification tests

Usage:
    python3 unit_test/test_all_pbt.py

This test suite is SAFE to run on M1 MacBook (16GB RAM)
It verifies the PBT script is ready for GH200 deployment without executing training.

Test Coverage:
    1. Python Syntax Verification (py_compile)
    2. Import Verification (all dependencies)
    3. Project Structure Validation
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_subheader(title):
    """Print formatted subsection header"""
    print("\n" + "-"*80)
    print(f"  {title}")
    print("-"*80)


def run_test_script(script_name, description):
    """
    Run a test script and return success status

    Args:
        script_name: Name of the test script (e.g., 'test_syntax_pbt.py')
        description: Human-readable description of the test

    Returns:
        bool: True if test passed, False otherwise
    """
    print_subheader(f"Running: {description}")

    project_root = Path.cwd()
    script_path = project_root / "unit_test" / script_name

    if not script_path.exists():
        print(f"❌ Test script not found: {script_path}")
        return False

    try:
        # Run the test script and capture output
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Print the output from the test script
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Check exit code
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ {description} - TIMEOUT (exceeded 60 seconds)")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def validate_project_structure():
    """
    Validate that all required project files exist

    Returns:
        bool: True if all required files exist, False otherwise
    """
    print_subheader("Validating Project Structure")

    project_root = Path.cwd()

    required_files = [
        {
            "path": project_root / "comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py",
            "name": "PBT Training Script",
            "critical": True
        },
        {
            "path": project_root / "comparative_study/05_evaluation/inference_bf16.py",
            "name": "Inference Script",
            "critical": True
        },
        {
            "path": project_root / "comparative_study/0c_utils/pbt_trainer.py",
            "name": "PBT Trainer Utils",
            "critical": True
        },
        {
            "path": project_root / "comparative_study/0c_utils/cita_trainer.py",
            "name": "CITA Trainer",
            "critical": True
        },
        {
            "path": project_root / "comparative_study/0c_utils/monitoring_callback.py",
            "name": "Monitoring Callback",
            "critical": True
        },
        {
            "path": project_root / "comparative_study/0c_utils/data_prep",
            "name": "Data Prep Module",
            "critical": True
        },
    ]

    required_dirs = [
        {
            "path": project_root / "outputs",
            "name": "Outputs Directory",
            "critical": False  # Will be created by script if missing
        },
        {
            "path": project_root / "logs",
            "name": "Logs Directory",
            "critical": False  # Will be created by script if missing
        },
    ]

    all_valid = True
    missing_critical = []

    # Check required files
    print("\nChecking Required Files:")
    for item in required_files:
        if item["path"].exists():
            print(f"  ✅ {item['name']}: {item['path']}")
        else:
            print(f"  ❌ {item['name']}: {item['path']} (NOT FOUND)")
            if item["critical"]:
                all_valid = False
                missing_critical.append(item["name"])

    # Check required directories
    print("\nChecking Required Directories:")
    for item in required_dirs:
        if item["path"].exists():
            print(f"  ✅ {item['name']}: {item['path']}")
        else:
            print(f"  ⚠️  {item['name']}: {item['path']} (will be created by script)")

    if missing_critical:
        print(f"\n❌ Missing critical files:")
        for name in missing_critical:
            print(f"   - {name}")
        return False
    else:
        print(f"\n✅ Project Structure - VALID")
        return True


def main():
    """Run all pre-deployment tests"""
    start_time = datetime.now()

    print_header("🧪 PBT Pre-Deployment Test Suite")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: M1 MacBook (16GB RAM)")
    print(f"Purpose: Verify PBT script is ready for GH200 deployment")

    # Track test results
    tests = []

    # Test 1: Project Structure
    print_header("Test 1/3: Project Structure Validation")
    structure_ok = validate_project_structure()
    tests.append(("Project Structure", structure_ok))

    # Test 2: Python Syntax
    print_header("Test 2/3: Python Syntax Verification")
    syntax_ok = run_test_script("test_syntax_pbt.py", "Syntax Check")
    tests.append(("Python Syntax", syntax_ok))

    # Test 3: Import Verification
    print_header("Test 3/3: Import Verification")
    imports_ok = run_test_script("test_imports_pbt.py", "Import Check")
    tests.append(("Import Check", imports_ok))

    # Final Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("📊 FINAL TEST SUMMARY")
    print(f"\nTest Results:")
    print("-" * 80)

    passed_count = 0
    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:40s} {status}")
        if passed:
            passed_count += 1

    print("-" * 80)
    print(f"\nTotal: {passed_count}/{len(tests)} tests passed")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Deployment readiness
    print("\n" + "="*80)
    if passed_count == len(tests):
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ PBT Script is ready for GH200 deployment")
        print("\n📋 Deployment Checklist:")
        print("   1. ✅ Python syntax validated")
        print("   2. ✅ All imports verified")
        print("   3. ✅ Project structure validated")
        print("\n🚀 Next Steps:")
        print("   1. Deploy code to GH200 instance")
        print("   2. Activate conda environment on GH200")
        print("   3. Run PBT training:")
        print("      python -u comparative_study/03a_CITA_Baseline/Llama3_BF16_PBT.py")
        print("   4. Monitor logs in ./logs/CITA_PBT_training_<timestamp>.log")
        print("\n💡 Tip: Use 'python -u' flag for unbuffered real-time logging")
        print("="*80)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\n⚠️  PBT Script is NOT ready for GH200 deployment")
        print("\n🔧 Required Actions:")
        print("   1. Review failed test output above")
        print("   2. Fix all reported errors")
        print("   3. Re-run this test suite: python3 unit_test/test_all_pbt.py")
        print("   4. Ensure all tests pass before deploying to GH200")
        print("\n💡 Tip: Run individual tests for debugging:")
        print("   - python3 unit_test/test_syntax_pbt.py")
        print("   - python3 unit_test/test_imports_pbt.py")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
