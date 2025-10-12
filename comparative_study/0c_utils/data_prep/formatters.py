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

    Output: ShareGPT format with NO system role
    {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'conversations' field in ShareGPT format
    """
    safe_response, _, harmful_categories = get_safe_unsafe_responses(example)

    # Synthesize instruction from harm categories
    instruction = synthesize_system_instruction(harmful_categories)

    # Combine instruction + user prompt (NO system role, merge into user message)
    combined_user = f"{instruction}\n\n{example['prompt']}"

    # ShareGPT format
    conversations = [
        {"from": "human", "value": combined_user},
        {"from": "gpt", "value": safe_response}
    ]

    return {"conversations": conversations}


def format_dpo(example: Dict) -> Dict:
    """
    Format PKU-SafeRLHF example for DPO training.

    Matches format from comparative_study/02a_DPO_Baseline/Llama3_alignmentDB.ipynb

    Output: Flat structure with string values
    {"prompt": str, "chosen": str, "rejected": str}

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'prompt', 'chosen', 'rejected' as strings
    """
    safe_response, unsafe_response, harmful_categories = get_safe_unsafe_responses(example)

    # Synthesize instruction from harm categories
    instruction = synthesize_system_instruction(harmful_categories)

    # Create formatted prompt (matches notebook template)
    prompt = f"""Below are some instructions that describe some tasks. Write responses that appropriately complete
each request.

### Instruction:
{instruction}

{example['prompt']}

### Response:
"""

    return {
        "prompt": prompt,
        "chosen": safe_response,
        "rejected": unsafe_response,
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
