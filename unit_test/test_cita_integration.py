"""
Integration test for CITA Trainer with actual TRL DPOTrainer
Tests that CITA trainer can:
1. Import correctly
2. Call parent's get_batch_loss_metrics
3. Compute loss without KeyError
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'comparative_study', '03a_CITA_Baseline'))

import torch
from datasets import Dataset


def test_cita_trainer_integration():
    """Full integration test with actual CITA trainer"""
    print("="*80)
    print("CITA Trainer Integration Test")
    print("="*80)

    # Import CITA trainer
    print("\n1. Importing CITA trainer...")
    try:
        from cita_trainer import CITATrainer
        print("   ✓ CITATrainer imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import: {e}")
        return False

    # Create minimal components
    print("\n2. Setting up minimal training environment...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import DPOConfig

        # Use tiny model for testing
        model_name = "hf-internal-testing/tiny-random-gpt2"
        print(f"   Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Set a simple chat template for testing
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

        model = AutoModelForCausalLM.from_pretrained(model_name)

        print("   ✓ Model and tokenizer loaded")
        print("   ✓ Chat template set")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    # Create minimal dataset
    print("\n3. Creating test dataset...")
    try:
        dataset = Dataset.from_dict({
            "chosen": [
                [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            ],
            "rejected": [
                [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Go away"}
                ]
            ]
        })
        print(f"   ✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Failed to create dataset: {e}")
        return False

    # Create CITA trainer
    print("\n4. Initializing CITA trainer...")
    try:
        config = DPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            max_steps=1,
            logging_steps=1,
            remove_unused_columns=False,
            beta=0.1,  # DPO temperature parameter
        )

        trainer = CITATrainer(
            model=model,
            ref_model=None,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            lambda_kl=0.01,  # CITA-specific KL weight
        )
        print("   ✓ CITATrainer initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize trainer: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test get_batch_loss_metrics exists
    print("\n5. Checking parent method access...")
    try:
        assert hasattr(trainer, 'get_batch_loss_metrics'), \
            "Trainer missing get_batch_loss_metrics method"
        print("   ✓ get_batch_loss_metrics method accessible")
    except Exception as e:
        print(f"   ✗ Method check failed: {e}")
        return False

    # Test compute_loss with mock inputs
    print("\n6. Testing compute_loss with mock batch...")
    try:
        # Create a mock batch similar to what DPOTrainer produces
        batch_size = 1
        seq_len = 10
        vocab_size = tokenizer.vocab_size
        device = model.device

        mock_batch = {
            "chosen_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).to(device),
            "chosen_attention_mask": torch.ones(batch_size, seq_len).to(device),
            "rejected_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).to(device),
            "rejected_attention_mask": torch.ones(batch_size, seq_len).to(device),
            "prompt_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len // 2)).to(device),
            "prompt_attention_mask": torch.ones(batch_size, seq_len // 2).to(device),
        }
        print(f"   Mock batch created on device: {device}")

        # Try to compute loss
        loss = trainer.compute_loss(model, mock_batch, return_outputs=False)

        print(f"   ✓ compute_loss executed successfully")
        print(f"   Loss value: {loss.item():.6f}")

        # Verify loss is valid
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be inf"

        print("   ✓ Loss is valid (scalar, no NaN/inf)")

    except KeyError as e:
        print(f"   ✗ KeyError during compute_loss: {e}")
        print(f"   Missing key: {str(e)}")
        print(f"   Available keys: {list(mock_batch.keys())}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error during compute_loss: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with return_outputs=True
    print("\n7. Testing compute_loss with return_outputs=True...")
    try:
        loss, outputs = trainer.compute_loss(model, mock_batch, return_outputs=True)

        assert "loss" in outputs, "Outputs should contain 'loss' key"
        print(f"   ✓ compute_loss with outputs works")
        print(f"   Output keys: {list(outputs.keys())}")

    except Exception as e:
        print(f"   ✗ Failed with return_outputs=True: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("🎉 ALL INTEGRATION TESTS PASSED!")
    print("="*80)
    print("\nCITA Trainer is ready to use. No KeyError expected.")
    return True


if __name__ == "__main__":
    success = test_cita_trainer_integration()

    if not success:
        print("\n❌ Integration test failed - DO NOT use this trainer yet!")
        sys.exit(1)
    else:
        print("\n✅ Integration test passed - Safe to proceed with training!")
        sys.exit(0)
