"""
PKU-SafeRLHF Dataset Loader

Loads and filters PKU-Alignment/PKU-SafeRLHF dataset for safety alignment training.
Filters for clear safety contrast (one safe, one unsafe response).
"""

from datasets import load_dataset
from typing import Dict, Optional


def load_pku_filtered(split: str = "train", max_samples: Optional[int] = None) -> Dict:
    """
    Load PKU-SafeRLHF dataset and filter for clear safety contrast.

    Args:
        split: Dataset split to load ('train' or 'test')
        max_samples: Optional limit on number of samples to return

    Returns:
        Filtered dataset with clear safe vs unsafe contrast
    """
    print(f"Loading PKU-Alignment/PKU-SafeRLHF (split={split})...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)

    print(f"Original dataset size: {len(dataset):,}")

    # Filter for clear safety contrast
    filtered_dataset = dataset.filter(
        lambda x: x['is_response_0_safe'] != x['is_response_1_safe']
    )

    print(f"Filtered dataset size: {len(filtered_dataset):,} (clear safety contrast)")

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
