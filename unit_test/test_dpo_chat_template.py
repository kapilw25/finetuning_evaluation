"""
Test DPO Chat Template Fix
Tests the chat template setup for Llama-3 tokenizer to resolve ValueError
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_chat_template_setup():
    """Test that chat template can be set and used for inference"""
    print("="*80)
    print("TEST: Chat Template Setup for DPO Inference")
    print("="*80)

    try:
        # Import required modules
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        import torch

        print("\n✅ Step 1: Imports successful")

        # Load model and tokenizer (small test)
        print("\n📦 Step 2: Loading tokenizer...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/llama-3-8b-bnb-4bit",
            max_seq_length = 512,  # Small for testing
            dtype = None,
            load_in_4bit = True,
        )
        print("✅ Step 2: Model and tokenizer loaded")

        # Check initial state
        print(f"\n🔍 Step 3: Initial chat_template: {tokenizer.chat_template}")

        # Apply chat template fix
        print("\n🔧 Step 4: Applying chat template...")
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3",
        )
        print(f"✅ Step 4: Chat template set: {tokenizer.chat_template is not None}")

        # Test inference setup
        print("\n⚙️ Step 5: Testing tokenizer.apply_chat_template()...")
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        print(f"✅ Step 5: Successfully created input_ids with shape {input_ids.shape}")

        # Decode to verify format
        decoded = tokenizer.decode(input_ids[0])
        print(f"\n📝 Decoded prompt preview:\n{decoded[:200]}...")

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - Chat template fix verified")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chat_template_setup()
    sys.exit(0 if success else 1)
