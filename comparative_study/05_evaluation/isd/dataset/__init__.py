"""
ISD Dataset Generator

Contains:
- instruction_switch_dataset.py: Generator for test cases
- push_isd_to_hf.py: Push dataset to HuggingFace
"""

from .instruction_switch_dataset import (
    InstructionSwitchDataset,
    ISDTestCase,
    ISD_INSTRUCTIONS,
    PROMPT_CATEGORIES,
    get_prompt_instruction_pair
)

__all__ = [
    'InstructionSwitchDataset',
    'ISDTestCase',
    'ISD_INSTRUCTIONS',
    'PROMPT_CATEGORIES',
    'get_prompt_instruction_pair'
]
