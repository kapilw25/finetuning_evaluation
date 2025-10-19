"""
Dataset Formatters for SFT, DPO, and CITA Training

Formats PKU-SafeRLHF dataset to match the exact format expected by working notebooks.
"""

from typing import Dict
from .pku_loader import get_safe_unsafe_responses, synthesize_system_instruction


def format_sft(example: Dict) -> Dict:
    """
    Format PKU-SafeRLHF example for SFT training.

    Matches format from comparative_study/01a_SFT_Baseline/Llama3_alignmentDB.ipynb

    Output: Llama-3 messages with explicit system role (fair comparison with DPO/CITA)
    {"messages": [{"role": "system", "content": ...}, {"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'messages' field in Llama-3 format (SFTTrainer auto-detects)
    """
    safe_response, _, harmful_categories = get_safe_unsafe_responses(example)

    # Synthesize instruction from harm categories
    instruction = synthesize_system_instruction(harmful_categories)

    # ✅ Llama-3 format with explicit system role (matches DPO/CITA for fair comparison)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": safe_response}
    ]

    return {"messages": messages}  # SFTTrainer auto-detects "messages" field


def format_dpo(example: Dict) -> Dict:
    """
    Format PKU-SafeRLHF example for DPO training.

    ✅ UPDATED: Now uses Llama-3 message format (fair comparison with SFT/CITA)

    Output: Message lists (DPOTrainer supports this since TRL 0.8+)
    {"prompt": [messages], "chosen": [messages], "rejected": [messages]}

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'prompt', 'chosen', 'rejected' as message lists
    """
    safe_response, unsafe_response, harmful_categories = get_safe_unsafe_responses(example)

    # Synthesize instruction from harm categories
    instruction = synthesize_system_instruction(harmful_categories)

    # ✅ Llama-3 format with explicit system role (matches SFT/CITA)
    prompt_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": example['prompt']}
    ]

    chosen_messages = [{"role": "assistant", "content": safe_response}]
    rejected_messages = [{"role": "assistant", "content": unsafe_response}]

    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }


def format_cita(example: Dict) -> Dict:
    """
    Format PKU-SafeRLHF example for CITA training.

    Matches format from comparative_study/03a_CITA_Baseline/Llama3_alignmentDB.ipynb

    Output: Message lists with explicit system role
    {"chosen": [msgs], "rejected": [msgs]}

    KEY INNOVATION: Explicit separation of alignment directive from user query

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'chosen' and 'rejected' message lists
    """
    safe_response, unsafe_response, harmful_categories = get_safe_unsafe_responses(example)

    # Synthesize explicit system instruction from harm categories
    instruction = synthesize_system_instruction(harmful_categories)

    # CITA format: Explicit instruction conditioning
    chosen_messages = [
        {"role": "system", "content": instruction},      # ← I (explicit instruction)
        {"role": "user", "content": example['prompt']},  # ← X (user request)
        {"role": "assistant", "content": safe_response}  # ← Y+ (safe response)
    ]

    rejected_messages = [
        {"role": "system", "content": instruction},        # ← Same I
        {"role": "user", "content": example['prompt']},    # ← Same X
        {"role": "assistant", "content": unsafe_response}  # ← Y- (unsafe response)
    ]

    return {
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }


def format_dataset(dataset, method: str):
    """
    Format entire dataset for specified training method.

    Args:
        dataset: PKU-SafeRLHF dataset (filtered)
        method: One of 'sft', 'dpo', 'cita'

    Returns:
        Formatted dataset ready for training
    """
    if method.lower() == 'sft':
        formatter = format_sft
    elif method.lower() == 'dpo':
        formatter = format_dpo
    elif method.lower() == 'cita':
        formatter = format_cita
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'sft', 'dpo', or 'cita'")

    formatted_dataset = dataset.map(
        formatter,
        remove_columns=dataset.column_names,
        desc=f"Formatting PKU-SafeRLHF for {method.upper()}"
    )

    return formatted_dataset
