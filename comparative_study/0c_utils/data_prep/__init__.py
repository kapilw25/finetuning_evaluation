"""
Data Prep Utils for PKU-SafeRLHF Dataset

Provides dataset loading, filtering, and formatting for SFT, DPO, and CITA training.
"""

from .loader_pku import (
    load_pku_filtered,
    get_safe_unsafe_responses,
    synthesize_system_instruction
)

from .formatters import (
    format_sft_NoInstruct,
    format_sft_Instruct,
    format_dpo_NoInstruct,
    format_dpo_Instruct,
    format_cita_Instruct,
    format_dataset,
    # Aliases
    format_pku_for_sft,
    format_pku_for_sft_NoInstruct,
    format_pku_for_sft_Instruct,
    format_pku_for_dpo_NoInstruct,
    format_pku_for_dpo_Instruct,
    format_pku_for_cita_NoInstruct,
    format_pku_for_cita_Instruct,
)

__all__ = [
    # Loaders
    'load_pku_filtered',
    'get_safe_unsafe_responses',
    'synthesize_system_instruction',
    # Core formatters
    'format_sft_NoInstruct',
    'format_sft_Instruct',
    'format_dpo_NoInstruct',
    'format_dpo_Instruct',
    'format_cita_Instruct',
    'format_dataset',
    # Aliases
    'format_pku_for_sft',
    'format_pku_for_sft_NoInstruct',
    'format_pku_for_sft_Instruct',
    'format_pku_for_dpo_NoInstruct',
    'format_pku_for_dpo_Instruct',
    'format_pku_for_cita_NoInstruct',
    'format_pku_for_cita_Instruct',
]
