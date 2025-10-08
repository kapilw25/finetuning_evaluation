"""
Data Prep Utils for PKU-SafeRLHF Dataset

Provides dataset loading, filtering, and formatting for SFT, DPO, and CITA training.
"""

from .pku_loader import (
    load_pku_filtered,
    get_safe_unsafe_responses,
    synthesize_system_instruction
)

from .formatters import (
    format_sft,
    format_dpo,
    format_cita,
    format_dataset
)

__all__ = [
    # Loaders
    'load_pku_filtered',
    'get_safe_unsafe_responses',
    'synthesize_system_instruction',
    # Formatters
    'format_sft',
    'format_dpo',
    'format_cita',
    'format_dataset',
]
