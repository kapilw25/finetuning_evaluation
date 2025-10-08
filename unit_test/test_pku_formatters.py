#!/usr/bin/env python3
"""
Test script for PKU-SafeRLHF formatters
Validates SFT, DPO, and CITA formatting before notebook integration
"""

import sys
sys.path.insert(0, '/lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study/0c_DATA_PREP_utils/src')

from data_prep import load_pku_filtered, format_dataset

def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}\n")


def test_sft_format():
    """Test SFT formatting - should match 01a_SFT_Baseline notebook"""
    print_section("TEST 1: SFT Format")

    # Load and format
    dataset = load_pku_filtered(split="train", max_samples=1)
    formatted = format_dataset(dataset, method="sft")

    sample = formatted[0]

    print("✅ Expected structure: {'conversations': [{'from': 'human', 'value': ...}, {'from': 'gpt', 'value': ...}]}")
    print(f"✅ Actual structure: {list(sample.keys())}")
    print(f"\n📋 Sample conversations: {sample['conversations']}")

    # Validate
    assert 'conversations' in sample, "Missing 'conversations' key"
    assert len(sample['conversations']) == 2, f"Expected 2 turns, got {len(sample['conversations'])}"
    assert sample['conversations'][0]['from'] == 'human', "First turn should be 'human'"
    assert sample['conversations'][1]['from'] == 'gpt', "Second turn should be 'gpt'"

    print("\n✅ SFT format validation PASSED")
    return True


def test_dpo_format():
    """Test DPO formatting - should match 02a_DPO_Baseline notebook"""
    print_section("TEST 2: DPO Format")

    # Load and format
    dataset = load_pku_filtered(split="train", max_samples=1)
    formatted = format_dataset(dataset, method="dpo")

    sample = formatted[0]

    print("✅ Expected structure: {'prompt': str, 'chosen': str, 'rejected': str}")
    print(f"✅ Actual structure: {list(sample.keys())}")
    print(f"\n📋 Prompt (first 200 chars): {sample['prompt'][:200]}...")
    print(f"\n✅ Chosen (first 100 chars): {sample['chosen'][:100]}...")
    print(f"\n❌ Rejected (first 100 chars): {sample['rejected'][:100]}...")

    # Validate
    assert 'prompt' in sample, "Missing 'prompt' key"
    assert 'chosen' in sample, "Missing 'chosen' key"
    assert 'rejected' in sample, "Missing 'rejected' key"
    assert isinstance(sample['prompt'], str), "Prompt should be string"
    assert isinstance(sample['chosen'], str), "Chosen should be string"
    assert isinstance(sample['rejected'], str), "Rejected should be string"

    print("\n✅ DPO format validation PASSED")
    return True


def test_cita_format():
    """Test CITA formatting - should match 03a_CITA_Baseline notebook"""
    print_section("TEST 3: CITA Format")

    # Load and format
    dataset = load_pku_filtered(split="train", max_samples=1)
    formatted = format_dataset(dataset, method="cita")

    sample = formatted[0]

    print("✅ Expected structure: {'chosen': [msgs], 'rejected': [msgs]}")
    print(f"✅ Actual structure: {list(sample.keys())}")
    print(f"\n📋 Chosen messages ({len(sample['chosen'])} turns):")
    for i, msg in enumerate(sample['chosen']):
        print(f"   {i+1}. {msg['role']}: {msg['content'][:80]}...")

    print(f"\n📋 Rejected messages ({len(sample['rejected'])} turns):")
    for i, msg in enumerate(sample['rejected']):
        print(f"   {i+1}. {msg['role']}: {msg['content'][:80]}...")

    # Validate
    assert 'chosen' in sample, "Missing 'chosen' key"
    assert 'rejected' in sample, "Missing 'rejected' key"
    assert isinstance(sample['chosen'], list), "Chosen should be list"
    assert isinstance(sample['rejected'], list), "Rejected should be list"
    assert len(sample['chosen']) == 3, f"Expected 3 messages, got {len(sample['chosen'])}"
    assert sample['chosen'][0]['role'] == 'system', "First message should be system"
    assert sample['chosen'][1]['role'] == 'user', "Second message should be user"
    assert sample['chosen'][2]['role'] == 'assistant', "Third message should be assistant"

    print("\n✅ CITA format validation PASSED")
    return True


def main():
    print_section("PKU-SafeRLHF Formatters Testing")

    try:
        # Run all tests
        sft_pass = test_sft_format()
        dpo_pass = test_dpo_format()
        cita_pass = test_cita_format()

        # Summary
        print_section("TEST SUMMARY")
        print(f"SFT Format:  {'✅ PASS' if sft_pass else '❌ FAIL'}")
        print(f"DPO Format:  {'✅ PASS' if dpo_pass else '❌ FAIL'}")
        print(f"CITA Format: {'✅ PASS' if cita_pass else '❌ FAIL'}")

        if all([sft_pass, dpo_pass, cita_pass]):
            print("\n🎉 All formatters are working correctly!")
            print("✅ Ready for notebook integration")
            return 0
        else:
            print("\n❌ Some tests failed. Fix issues before integration.")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
