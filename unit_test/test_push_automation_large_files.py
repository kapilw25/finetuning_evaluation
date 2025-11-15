#!/usr/bin/env python3
"""
Test push_automation.py large file handling modifications

Usage:
    python unit_test/test_push_automation_large_files.py
"""

import sys
from pathlib import Path

# Add 0c_utils to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# Test imports
print("=" * 80)
print("Testing push_automation.py Large File Handling")
print("=" * 80)

test_count = 0
pass_count = 0

# Test 1: Import PushAutomation
test_count += 1
try:
    from push_automation import PushAutomation
    print(f"✅ Test {test_count}: Import PushAutomation")
    pass_count += 1
except Exception as e:
    print(f"❌ Test {test_count}: Import PushAutomation failed: {e}")

# Test 2: Check _check_large_files method exists
test_count += 1
try:
    assert hasattr(PushAutomation, '_check_large_files')
    print(f"✅ Test {test_count}: _check_large_files method exists")
    pass_count += 1
except AssertionError:
    print(f"❌ Test {test_count}: _check_large_files method not found")

# Test 3: Check _add_to_gitignore method exists
test_count += 1
try:
    assert hasattr(PushAutomation, '_add_to_gitignore')
    print(f"✅ Test {test_count}: _add_to_gitignore method exists")
    pass_count += 1
except AssertionError:
    print(f"❌ Test {test_count}: _add_to_gitignore method not found")

# Test 4: Check _handle_large_files method exists (new name)
test_count += 1
try:
    assert hasattr(PushAutomation, '_handle_large_files')
    print(f"✅ Test {test_count}: _handle_large_files method exists")
    pass_count += 1
except AssertionError:
    print(f"❌ Test {test_count}: _handle_large_files method not found")

# Test 5: Verify old _handle_large_log_files method is removed
test_count += 1
try:
    assert not hasattr(PushAutomation, '_handle_large_log_files')
    print(f"✅ Test {test_count}: Old _handle_large_log_files method removed")
    pass_count += 1
except AssertionError:
    print(f"❌ Test {test_count}: Old _handle_large_log_files method still exists")

# Test 6: Verify old _split_large_file method is removed
test_count += 1
try:
    assert not hasattr(PushAutomation, '_split_large_file')
    print(f"✅ Test {test_count}: Old _split_large_file method removed")
    pass_count += 1
except AssertionError:
    print(f"❌ Test {test_count}: Old _split_large_file method still exists")

# Test 7: Check push_to_github method exists
test_count += 1
try:
    assert hasattr(PushAutomation, 'push_to_github')
    print(f"✅ Test {test_count}: push_to_github method exists")
    pass_count += 1
except AssertionError:
    print(f"❌ Test {test_count}: push_to_github method not found")

# Test 8: Verify method signatures (using inspect)
test_count += 1
try:
    import inspect

    # Check _check_large_files returns list
    sig = inspect.signature(PushAutomation._check_large_files)
    assert sig.return_annotation == list or 'list' in str(sig)

    # Check _add_to_gitignore accepts file_paths parameter
    sig = inspect.signature(PushAutomation._add_to_gitignore)
    assert 'file_paths' in sig.parameters

    print(f"✅ Test {test_count}: Method signatures correct")
    pass_count += 1
except Exception as e:
    print(f"❌ Test {test_count}: Method signature check failed: {e}")

# Summary
print("=" * 80)
print(f"RESULTS: {pass_count}/{test_count} tests passed")
print("=" * 80)

if pass_count == test_count:
    print("✅ All tests passed!")
    sys.exit(0)
else:
    print(f"❌ {test_count - pass_count} tests failed")
    sys.exit(1)
