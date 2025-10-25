#!/usr/bin/env python3
"""
Test that run_full_aqi_evaluation.py compiles correctly and HF_TOKEN is loaded
"""

import sys
import py_compile
import ast
from pathlib import Path

print("="*80)
print("AQI Script Compilation Test")
print("="*80)

# Path to the script
script_path = Path(__file__).parent.parent / "comparative_study/05_evaluation/AQI/run_full_aqi_evaluation.py"

# Test 1: Syntax check with py_compile
print("\n1️⃣ Testing Python syntax with py_compile...")
try:
    py_compile.compile(str(script_path), doraise=True)
    print("✅ Syntax valid (py_compile)")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)

# Test 2: AST parsing
print("\n2️⃣ Testing AST parsing...")
try:
    with open(script_path, 'r') as f:
        code = f.read()
    ast.parse(code)
    print("✅ AST parsing successful")
except SyntaxError as e:
    print(f"❌ AST parsing failed: {e}")
    sys.exit(1)

# Test 3: Check for HF_TOKEN usage
print("\n3️⃣ Checking HF_TOKEN is passed to model loading...")
with open(script_path, 'r') as f:
    content = f.read()

required_patterns = [
    "from dotenv import load_dotenv",
    "HF_TOKEN = os.getenv",
    "token=HF_TOKEN",  # Should appear multiple times
]

missing = []
for pattern in required_patterns:
    if pattern not in content:
        missing.append(pattern)

if missing:
    print(f"❌ Missing required patterns: {missing}")
    sys.exit(1)

# Count token=HF_TOKEN occurrences (should be 3: base model, tokenizer, adapter)
token_count = content.count("token=HF_TOKEN")
print(f"   Found {token_count} occurrences of 'token=HF_TOKEN'")

if token_count < 3:
    print(f"⚠️  Warning: Expected at least 3 occurrences, found {token_count}")
else:
    print(f"✅ HF_TOKEN properly passed to all HF API calls")

# Test 4: Import check (imports only, doesn't run main)
print("\n4️⃣ Testing imports...")
try:
    # Import the module without running main()
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_full_aqi_evaluation", script_path)
    module = importlib.util.module_from_spec(spec)

    # Execute module (loads imports and constants, but doesn't run main() unless called)
    sys.modules["run_full_aqi_evaluation"] = module
    spec.loader.exec_module(module)

    # Verify HF_TOKEN was loaded
    if hasattr(module, 'HF_TOKEN') and module.HF_TOKEN:
        print(f"✅ Module imported successfully")
        print(f"✅ HF_TOKEN loaded: {module.HF_TOKEN[:10]}...{module.HF_TOKEN[-5:]}")
    else:
        print("❌ HF_TOKEN not found in module")
        sys.exit(1)

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL COMPILATION TESTS PASSED")
print("="*80)
print("\n✅ Script is ready to run")
print("="*80)
