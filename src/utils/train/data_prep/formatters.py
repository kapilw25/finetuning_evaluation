"""
Dataset Formatters for SFT, DPO, and CITA Training

Formats PKU-SafeRLHF dataset to match the exact format expected by working notebooks.
"""

from typing import Dict
from .loader_pku import get_safe_unsafe_responses, synthesize_system_instruction


def format_sft_NoInstruct(example: Dict) -> Dict:
    """
    Format PKU-SafeRLHF example for SFT training.

    Matches proposal: SFT trains on (x, y) pairs WITHOUT instruction conditioning

    Output: Llama-3 messages WITHOUT system instruction (baseline)
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'messages' field in Llama-3 format (SFTTrainer auto-detects)
    """
    safe_response, _, _ = get_safe_unsafe_responses(example)

    # SFT Baseline: NO instruction (only prompt + response)
    messages = [
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": safe_response}
    ]

    return {"messages": messages}  # SFTTrainer auto-detects "messages" field


def format_dpo_NoInstruct(example: Dict) -> Dict:
    """
    Format PKU-SafeRLHF example for DPO training.

    Matches proposal: DPO trains on preference pairs WITHOUT instruction conditioning

    Output: Message lists WITHOUT system instruction (baseline)
    {"prompt": [messages], "chosen": [messages], "rejected": [messages]}

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'prompt', 'chosen', 'rejected' as message lists
    """
    safe_response, unsafe_response, _ = get_safe_unsafe_responses(example)

    # DPO Baseline: NO instruction (only prompt + responses)
    prompt_messages = [
        {"role": "user", "content": example['prompt']}
    ]

    chosen_messages = [{"role": "assistant", "content": safe_response}]
    rejected_messages = [{"role": "assistant", "content": unsafe_response}]

    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }


def format_cita_Instruct(example: Dict) -> Dict:
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


def format_sft_Instruct(example: Dict) -> Dict:
    """
    Format PKU-SafeRLHF example for SFT training WITH instruction conditioning.

    This variant adds system instruction for fair comparison with CITA.

    Output: Llama-3 messages WITH system instruction
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'messages' field including system instruction
    """
    safe_response, _, harmful_categories = get_safe_unsafe_responses(example)

    # Synthesize system instruction from harm categories (CITA-style)
    instruction = synthesize_system_instruction(harmful_categories)

    # SFT with Instruction: Include system role
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": safe_response}
    ]

    return {"messages": messages}


def format_dpo_Instruct(example: Dict) -> Dict:
    """
    Format PKU-SafeRLHF example for DPO training WITH instruction conditioning.

    This variant adds system instruction for fair comparison with CITA.

    Output: Message lists WITH system instruction
    {"prompt": [messages], "chosen": [messages], "rejected": [messages]}

    Args:
        example: Single example from PKU-SafeRLHF dataset

    Returns:
        Dictionary with 'prompt', 'chosen', 'rejected' including system instruction
    """
    safe_response, unsafe_response, harmful_categories = get_safe_unsafe_responses(example)

    # Synthesize system instruction from harm categories (CITA-style)
    instruction = synthesize_system_instruction(harmful_categories)

    # DPO with Instruction: Include system role in prompt
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


# ===================================================================
# Aliases for consistent naming in training scripts
# ===================================================================
# Original alias for Llama baseline before SFT (no suffix - exception per user requirement)
format_pku_for_sft = format_sft_NoInstruct  # Baseline model exception

# Aliases with proper suffixes for ablation study
format_pku_for_sft_NoInstruct = format_sft_NoInstruct  # SFT without instruction
format_pku_for_sft_Instruct = format_sft_Instruct  # SFT with instruction
format_pku_for_dpo_NoInstruct = format_dpo_NoInstruct  # DPO without instruction
format_pku_for_dpo_Instruct = format_dpo_Instruct  # DPO with instruction

# CITA formatters: Reuse DPO format when possible (same underlying DPOTrainer)
format_pku_for_cita_NoInstruct = format_dpo_NoInstruct  # CITA NoInstruct = DPO NoInstruct (same format)
format_pku_for_cita_Instruct = format_cita_Instruct  # CITA Instruct uses its own special format


def format_dataset(dataset, method: str):
    """
    Format entire dataset for specified training method.

    Args:
        dataset: PKU-SafeRLHF dataset (filtered)
        method: One of the supported training methods with instruction variants

    Supported methods:
        - 'sft_NoInstruct' or 'sft': SFT without instruction
        - 'sft_Instruct': SFT with instruction
        - 'dpo_NoInstruct' or 'dpo': DPO without instruction
        - 'dpo_Instruct': DPO with instruction
        - 'cita_NoInstruct': CITA without instruction
        - 'cita_Instruct' or 'cita': CITA with instruction

    Returns:
        Formatted dataset ready for training
    """
    method_lower = method.lower()

    # Map method names to formatter functions
    if method_lower in ['sft_noinstruct', 'sft']:
        formatter = format_sft_NoInstruct
    elif method_lower == 'sft_instruct':
        formatter = format_sft_Instruct
    elif method_lower in ['dpo_noinstruct', 'dpo']:
        formatter = format_dpo_NoInstruct
    elif method_lower == 'dpo_instruct':
        formatter = format_dpo_Instruct
    elif method_lower == 'cita_noinstruct':
        formatter = format_dpo_NoInstruct  # CITA NoInstruct uses same format as DPO NoInstruct
    elif method_lower in ['cita_instruct', 'cita']:
        formatter = format_cita_Instruct
    else:
        raise ValueError(
            f"Unknown method: {method}. Must be one of: "
            "'sft_NoInstruct', 'sft_Instruct', 'dpo_NoInstruct', 'dpo_Instruct', "
            "'cita_NoInstruct', 'cita_Instruct', or their short forms 'sft', 'dpo', 'cita'"
        )

    formatted_dataset = dataset.map(
        formatter,
        remove_columns=dataset.column_names,
        desc=f"Formatting PKU-SafeRLHF for {method.upper()}"
    )

    return formatted_dataset
