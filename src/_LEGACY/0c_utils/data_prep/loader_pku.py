"""
PKU-SafeRLHF Dataset Loader

Loads and filters PKU-Alignment/PKU-SafeRLHF dataset for safety alignment training.
Filters for clear safety contrast (one safe, one unsafe response).
"""

from datasets import load_dataset, concatenate_datasets
from typing import Dict, Optional


def load_pku_combined_clear_contrast(val_split: float = 0.1):
    """
    Load clear contrast samples from BOTH train and test splits, then create 90/10 split.

    Combines:
    - Train split clear contrast: 10,813 samples
    - Test split clear contrast: 1,222 samples
    - Total: 12,035 samples (+11% more data)

    Args:
        val_split: Fraction for validation (default: 0.1)

    Returns:
        DatasetDict with 'train' and 'test' keys
        - train: 10,831 samples (90%)
        - test: 1,204 samples (10%)
    """
    print("Loading PKU-SafeRLHF: Combining train+test clear contrast...")

    # Load train split clear contrast
    train_data = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    train_clear = train_data.filter(lambda x: x['is_response_0_safe'] != x['is_response_1_safe'])
    print(f"  Train clear contrast: {len(train_clear):,} samples")

    # Load test split clear contrast
    test_data = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
    test_clear = test_data.filter(lambda x: x['is_response_0_safe'] != x['is_response_1_safe'])
    print(f"  Test clear contrast: {len(test_clear):,} samples")

    # Combine
    combined = concatenate_datasets([train_clear, test_clear])
    print(f"  Combined: {len(combined):,} samples")

    # 90/10 split
    split_dataset = combined.train_test_split(test_size=val_split, seed=3407)
    print(f"  Final train: {len(split_dataset['train']):,} | val: {len(split_dataset['test']):,}")

    return split_dataset


def load_pku_filtered(split: str = "train", max_samples: Optional[int] = None,
                      val_split: float = 0.1, return_val: bool = False) -> Dict:
    """
    Load PKU-SafeRLHF dataset and filter for clear safety contrast.

    Args:
        split: Dataset split to load ('train' or 'test')
        max_samples: Optional limit on number of samples to return
        val_split: Fraction of training data to use for validation (default: 0.1)
        return_val: If True, return validation split instead of train split

    Returns:
        Filtered dataset with clear safe vs unsafe contrast
        If return_val=True, returns validation split (90/10 by default)
    """
    print(f"Loading PKU-Alignment/PKU-SafeRLHF (split={split})...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)

    print(f"Original dataset size: {len(dataset):,}")

    # Filter for clear safety contrast
    filtered_dataset = dataset.filter(
        lambda x: x['is_response_0_safe'] != x['is_response_1_safe']
    )

    print(f"Filtered dataset size: {len(filtered_dataset):,} (clear safety contrast)")

    # Split into train/val if needed
    if split == "train" and val_split > 0:
        # Use train_test_split from datasets
        split_dataset = filtered_dataset.train_test_split(
            test_size=val_split,
            seed=3407  # Match training seed for reproducibility
        )

        if return_val:
            filtered_dataset = split_dataset['test']  # Validation split
            print(f"Validation split size: {len(filtered_dataset):,} ({val_split*100:.0f}% of training data)")
        else:
            filtered_dataset = split_dataset['train']  # Training split
            print(f"Training split size: {len(filtered_dataset):,} ({(1-val_split)*100:.0f}% of training data)")

    if max_samples:
        filtered_dataset = filtered_dataset.select(range(min(max_samples, len(filtered_dataset))))
        print(f"Limited to: {len(filtered_dataset):,} samples")

    return filtered_dataset


def get_safe_unsafe_responses(example: Dict) -> tuple:
    """
    Extract safe and unsafe responses from PKU dataset example.

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        (safe_response, unsafe_response, harmful_categories)
    """
    if example['is_response_0_safe']:
        safe_response = example['response_0']
        unsafe_response = example['response_1']
        unsafe_harm_cat = example['response_1_harm_category']
    else:
        safe_response = example['response_1']
        unsafe_response = example['response_0']
        unsafe_harm_cat = example['response_0_harm_category']

    # Extract harm categories
    harmful_categories = [k for k, v in unsafe_harm_cat.items() if v == True]

    return safe_response, unsafe_response, harmful_categories


def synthesize_system_instruction(harmful_categories: list) -> str:
    """
    Synthesize system instruction based on harm categories.

    Args:
        harmful_categories: List of harm category names

    Returns:
        Synthesized system instruction string
    """
    if harmful_categories:
        harm_list = ', '.join(harmful_categories).lower()
        instruction = (
            f"You are a helpful AI assistant. You must REFUSE any requests "
            f"involving {harm_list}. Always prioritize user safety over helpfulness."
        )
    else:
        instruction = "You are a helpful and safe AI assistant. Always prioritize user safety."

    return instruction
