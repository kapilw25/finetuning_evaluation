"""
Unit test for CITA Trainer
Tests if the trainer can call parent DPOTrainer methods correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'comparative_study', '03a_CITA_Baseline'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig


def test_dpo_trainer_get_batch_loss_metrics():
    """Test if DPOTrainer has get_batch_loss_metrics method"""
    from trl import DPOTrainer

    print("✓ Checking if DPOTrainer has get_batch_loss_metrics method...")

    # Check if method exists
    assert hasattr(DPOTrainer, 'get_batch_loss_metrics'), \
        "DPOTrainer does not have get_batch_loss_metrics method!"

    print("✓ DPOTrainer.get_batch_loss_metrics exists")

    # Check method signature
    import inspect
    sig = inspect.signature(DPOTrainer.get_batch_loss_metrics)
    params = list(sig.parameters.keys())
    print(f"  Method signature: {params}")

    assert 'model' in params, "Missing 'model' parameter"
    assert 'batch' in params or 'inputs' in params, "Missing batch/inputs parameter"

    print("✓ Method signature looks correct")
    return True


def test_cita_trainer_imports():
    """Test if CITA trainer can be imported"""
    print("\n✓ Testing CITA trainer import...")

    try:
        from cita_trainer import CITATrainer
        print("✓ CITATrainer imported successfully")

        # Check if it extends DPOTrainer
        from trl import DPOTrainer
        assert issubclass(CITATrainer, DPOTrainer), \
            "CITATrainer does not extend DPOTrainer!"

        print("✓ CITATrainer correctly extends DPOTrainer")
        return True

    except Exception as e:
        print(f"✗ Failed to import CITATrainer: {e}")
        return False


def test_mock_batch_structure():
    """Test the structure of a DPO batch to understand input keys"""
    print("\n✓ Testing DPO batch structure...")

    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Create minimal tokenizer and model
    print("  Loading tiny model for testing...")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")

    # Create minimal dataset
    dataset = Dataset.from_dict({
        "prompt": ["Hello"],
        "chosen": ["World"],
        "rejected": ["Bye"]
    })

    # Create trainer
    config = DPOConfig(
        output_dir="./test_output",
        per_device_train_batch_size=1,
        max_steps=1,
        remove_unused_columns=False,
    )

    try:
        trainer = DPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        print("✓ DPOTrainer created successfully")

        # Get a batch from dataloader
        dataloader = trainer.get_train_dataloader()
        batch = next(iter(dataloader))

        print(f"  Batch keys: {list(batch.keys())}")

        # Check for expected keys
        expected_keys = ["chosen_input_ids", "rejected_input_ids",
                        "chosen_attention_mask", "rejected_attention_mask"]

        for key in expected_keys:
            if key in batch:
                print(f"  ✓ Found key: {key}")
            else:
                print(f"  ✗ Missing key: {key}")

        return True

    except Exception as e:
        print(f"  ✗ Error during batch test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("CITA Trainer Unit Tests")
    print("="*80)

    tests = [
        ("DPOTrainer has get_batch_loss_metrics", test_dpo_trainer_get_batch_loss_metrics),
        ("CITA Trainer imports correctly", test_cita_trainer_imports),
        ("DPO batch structure", test_mock_batch_structure),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*80)
    print("Test Results:")
    print("="*80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
