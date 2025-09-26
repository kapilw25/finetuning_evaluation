#!/usr/bin/env python3
"""
HuggingFace Authentication Integration Test
Purpose: Verify HF authentication works with actual virtual environment

Usage:
    source venv_FntngEval/bin/activate
    python unit_test/test_hf_auth_integration.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_hf_auth_standalone():
    """Test HF authentication directly"""
    print("=" * 60)
    print("HF AUTHENTICATION STANDALONE TEST")
    print("=" * 60)

    try:
        from m01_environment_setup import EnvironmentSetup

        print("✅ Successfully imported EnvironmentSetup")

        # Create environment setup instance
        env_setup = EnvironmentSetup()
        print("✅ EnvironmentSetup instance created")

        # Test HF authentication
        print("\n🔐 Testing HuggingFace Authentication...")
        hf_auth = env_setup.setup_huggingface_auth()

        print(f"📊 Authentication Results:")
        print(f"  - Token Found: {'✅' if hf_auth.get('token_found') else '❌'}")
        print(f"  - Token Valid: {'✅' if hf_auth.get('token_valid') else '❌'}")
        print(f"  - Login Successful: {'✅' if hf_auth.get('login_successful') else '❌'}")

        if hf_auth.get('user_info'):
            user_info = hf_auth['user_info']
            print(f"  - User: {user_info.get('name', 'Unknown')}")
            print(f"  - Type: {user_info.get('type', 'Unknown')}")

        if hf_auth.get('error'):
            print(f"  - Error: {hf_auth['error']}")

        return hf_auth.get('login_successful', False)

    except Exception as e:
        print(f"❌ Error in HF auth test: {e}")
        return False

def test_gated_model_access():
    """Test access to gated model"""
    print("\n" + "=" * 60)
    print("GATED MODEL ACCESS TEST")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer

        print("🔍 Testing access to Meta-Llama-3-8B-Instruct...")

        # Try to load just the tokenizer (lightweight test)
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            trust_remote_code=True
        )

        print("✅ Successfully accessed Meta-Llama-3-8B-Instruct tokenizer")
        print(f"✅ Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"✅ Model max length: {tokenizer.model_max_length}")

        return True

    except Exception as e:
        print(f"❌ Failed to access gated model: {e}")
        return False

def test_m05_integration():
    """Test m05 integration with HF auth"""
    print("\n" + "=" * 60)
    print("M05 INTEGRATION TEST")
    print("=" * 60)

    try:
        print("🔍 Testing m05 imports...")
        from m05_inference_backends import MVPInferenceBackend
        print("✅ Successfully imported MVPInferenceBackend")

        print("🔍 Testing m05 instantiation with HF auth...")
        # This should trigger HF authentication in __init__
        backend = MVPInferenceBackend()
        print("✅ MVPInferenceBackend created successfully with HF auth")

        print("🔍 Verifying backend configuration...")
        print(f"  - Base model: {backend.base_model_name}")
        print(f"  - QLoRA adapter path: {backend.qlora_adapter_path}")
        print(f"  - QLoRA adapter exists: {'✅' if backend.qlora_adapter_path.exists() else '❌'}")

        return True

    except Exception as e:
        print(f"❌ M05 integration test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE HF AUTHENTICATION & INTEGRATION TESTS")
    print("📍 Running in virtual environment (not sandbox)")
    print(f"📍 Python executable: {sys.executable}")
    print(f"📍 Working directory: {os.getcwd()}")

    results = {
        "hf_auth": False,
        "gated_access": False,
        "m05_integration": False
    }

    # Test 1: HF Authentication
    results["hf_auth"] = test_hf_auth_standalone()

    # Test 2: Gated Model Access (only if auth successful)
    if results["hf_auth"]:
        results["gated_access"] = test_gated_model_access()
    else:
        print("\n⏭️ Skipping gated model test - authentication failed")

    # Test 3: M05 Integration
    results["m05_integration"] = test_m05_integration()

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")

    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED - m05 ready for deployment!")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED - review errors above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)