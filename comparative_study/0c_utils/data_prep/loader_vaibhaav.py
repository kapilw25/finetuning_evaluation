#!/usr/bin/env python3
"""
Vaibhaav/alignment-instructions Dataset Loader
Provides formatters for SFT, DPO, and CITA training methods

Dataset: 50,001 samples with custom natural language instructions per prompt
- SFT: NO instruction (baseline)
- DPO: NO instruction (baseline)
- CITA: WITH natural language instruction (proposal innovation)
"""

from datasets import load_dataset
from typing import Dict, Optional


def load_vaibhaav_alignment(split: str = "train", max_samples: Optional[int] = None):
    """
    Load Vaibhaav/alignment-instructions dataset

    Args:
        split: Dataset split ("train" or "test")
        max_samples: Optional limit on number of samples

    Returns:
        HuggingFace Dataset object
    """
    dataset = load_dataset("Vaibhaav/alignment-instructions", split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"✅ Loaded {len(dataset):,} samples from Vaibhaav/alignment-instructions ({split})")
    return dataset


def format_vaibhaav_for_sft(example: Dict) -> Dict:
    """
    SFT Baseline: NO instruction (only prompt + accepted response)

    Matches proposal: SFT trains on (x, y) pairs WITHOUT instruction conditioning

    Args:
        example: Dataset sample with keys:
            - "Prompt": Metadata (NOT USED - just context for AI teacher)
            - "Accepted Response": Full dialogue with H:/A: markers (safe version)
            - "Rejected Response": Full dialogue (unsafe version, NOT USED)
            - "Instruction generated": Custom natural language instruction (NOT USED)

    Returns:
        Dict with "messages" key (chat format for SFTTrainer)
    """
    # Accepted Response contains full conversation in Human:/Assistant: format
    # Extract first human message as prompt, last assistant message as response
    conversation = example["Accepted Response"].strip()

    # Parse Human: and Assistant: turns
    turns = []
    for line in conversation.split("\n\n"):
        line = line.strip()
        if line.startswith("Human:"):
            turns.append(("user", line.replace("Human:", "").strip()))
        elif line.startswith("Assistant:"):
            turns.append(("assistant", line.replace("Assistant:", "").strip()))

    # Take first human turn and last assistant turn
    if len(turns) >= 2:
        user_msg = turns[0][1] if turns[0][0] == "user" else ""
        # Find last assistant turn
        assistant_turns = [t[1] for t in turns if t[0] == "assistant"]
        assistant_msg = assistant_turns[-1] if assistant_turns else conversation
    else:
        # Fallback: use entire conversation as response
        user_msg = "Continue the conversation"
        assistant_msg = conversation

    # Format: User prompt -> Assistant response (NO system instruction)
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ]

    return {"messages": messages}


def format_vaibhaav_for_dpo(example: Dict) -> Dict:
    """
    DPO Baseline: NO instruction (only prompt + chosen/rejected responses)

    Matches proposal: DPO trains on preference pairs WITHOUT instruction conditioning

    Args:
        example: Dataset sample (same keys as format_vaibhaav_for_sft)

    Returns:
        Dict with keys:
            - "prompt": User message (NO instruction)
            - "chosen": Accepted response
            - "rejected": Rejected response
    """
    # Parse accepted conversation to get first user message
    conversation_accepted = example["Accepted Response"].strip()
    turns_accepted = []
    for line in conversation_accepted.split("\n\n"):
        line = line.strip()
        if line.startswith("Human:"):
            turns_accepted.append(("user", line.replace("Human:", "").strip()))
        elif line.startswith("Assistant:"):
            turns_accepted.append(("assistant", line.replace("Assistant:", "").strip()))

    # Get first user message
    user_msg = turns_accepted[0][1] if turns_accepted and turns_accepted[0][0] == "user" else "Continue"

    # Get last assistant responses from both accepted and rejected
    turns_rejected = []
    conversation_rejected = example["Rejected Response"].strip()
    for line in conversation_rejected.split("\n\n"):
        line = line.strip()
        if line.startswith("Assistant:"):
            turns_rejected.append(line.replace("Assistant:", "").strip())

    assistant_turns_accepted = [t[1] for t in turns_accepted if t[0] == "assistant"]
    accepted_response = assistant_turns_accepted[-1] if assistant_turns_accepted else conversation_accepted
    rejected_response = turns_rejected[-1] if turns_rejected else conversation_rejected

    # DPO format: prompt (NO instruction), chosen, rejected
    return {
        "prompt": [{"role": "user", "content": user_msg}],
        "chosen": [{"role": "assistant", "content": accepted_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}]
    }


def format_vaibhaav_for_cita(example: Dict) -> Dict:
    """
    CITA: WITH natural language instruction (proposal's core innovation!)

    Matches proposal: CITA trains on (I, x, y) triplets WITH instruction conditioning

    Key difference from baselines:
    - Uses custom per-prompt natural language instructions
    - NOT generic category-based instructions
    - Example: "When responding to illegal activities, focus on discussing
               ethical considerations, legal consequences, and avoiding methods"

    Args:
        example: Dataset sample (same keys as above)

    Returns:
        Dict with keys:
            - "chosen": [system instruction, user prompt, accepted response]
            - "rejected": [system instruction, user prompt, rejected response]
    """
    # Extract instruction (CITA's key innovation!)
    instruction = example["Instruction generated"].strip()

    # Parse conversations (same as DPO)
    conversation_accepted = example["Accepted Response"].strip()
    turns_accepted = []
    for line in conversation_accepted.split("\n\n"):
        line = line.strip()
        if line.startswith("Human:"):
            turns_accepted.append(("user", line.replace("Human:", "").strip()))
        elif line.startswith("Assistant:"):
            turns_accepted.append(("assistant", line.replace("Assistant:", "").strip()))

    conversation_rejected = example["Rejected Response"].strip()
    turns_rejected = []
    for line in conversation_rejected.split("\n\n"):
        line = line.strip()
        if line.startswith("Assistant:"):
            turns_rejected.append(line.replace("Assistant:", "").strip())

    # Get first user message and last assistant responses
    user_msg = turns_accepted[0][1] if turns_accepted and turns_accepted[0][0] == "user" else "Continue"
    assistant_turns_accepted = [t[1] for t in turns_accepted if t[0] == "assistant"]
    accepted_response = assistant_turns_accepted[-1] if assistant_turns_accepted else conversation_accepted
    rejected_response = turns_rejected[-1] if turns_rejected else conversation_rejected

    # CITA format: system instruction + user prompt + response
    # Both chosen and rejected get SAME instruction (tests instruction following)
    return {
        "chosen": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": accepted_response}
        ],
        "rejected": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": rejected_response}
        ]
    }


# Test formatters
if __name__ == "__main__":
    print("="*80)
    print("TESTING VAIBHAAV FORMATTERS")
    print("="*80)

    # Load small sample
    dataset = load_vaibhaav_alignment(split="train", max_samples=5)

    print("\n" + "="*80)
    print("1. SFT Formatter (NO instruction)")
    print("="*80)
    sft_sample = format_vaibhaav_for_sft(dataset[0])
    print(f"Messages: {len(sft_sample['messages'])} turns")
    for msg in sft_sample['messages']:
        print(f"  {msg['role']}: {msg['content'][:100]}...")

    print("\n" + "="*80)
    print("2. DPO Formatter (NO instruction)")
    print("="*80)
    dpo_sample = format_vaibhaav_for_dpo(dataset[0])
    print(f"Prompt: {dpo_sample['prompt'][0]['content'][:100]}...")
    print(f"Chosen: {dpo_sample['chosen'][0]['content'][:100]}...")
    print(f"Rejected: {dpo_sample['rejected'][0]['content'][:100]}...")

    print("\n" + "="*80)
    print("3. CITA Formatter (WITH natural language instruction)")
    print("="*80)
    cita_sample = format_vaibhaav_for_cita(dataset[0])
    print(f"Chosen messages: {len(cita_sample['chosen'])} turns")
    for msg in cita_sample['chosen']:
        print(f"  {msg['role']}: {msg['content'][:100]}...")

    print("\n" + "="*80)
    print("✅ All formatters working correctly!")
    print("="*80)
    print("\nKey differences:")
    print("  - SFT/DPO: NO system instruction (baseline)")
    print("  - CITA: Custom natural language instruction per prompt")
    print("  - Dataset: 50,001 samples (vs 10,813 in PKU)")
