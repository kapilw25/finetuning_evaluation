"""
Push ISD Dataset to HuggingFace
"""

import sys
from pathlib import Path
from datasets import Dataset

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

from model_utils import load_hf_token
from instruction_switch_dataset import InstructionSwitchDataset

# Config
HF_TOKEN = load_hf_token(project_root)
HF_REPO = "kapilw25/ISD-Instruction-Switch-Dataset"
NUM_PROMPTS = 500

# Generate dataset
print("Generating ISD dataset...")
isd = InstructionSwitchDataset(seed=42)
test_cases = isd.generate_dataset(num_prompts=NUM_PROMPTS)

# Convert to HuggingFace format
data = {
    "prompt_id": [tc.prompt_id for tc in test_cases],
    "prompt": [tc.prompt for tc in test_cases],
    "instruction_type": [tc.instruction_type for tc in test_cases],
    "instruction": [tc.instruction for tc in test_cases],
    "expected_characteristics": [tc.expected_characteristics for tc in test_cases],
    "source": [tc.source for tc in test_cases]
}

dataset = Dataset.from_dict(data)

print(f"\nDataset: {len(dataset)} test cases")
print(f"Pushing to: {HF_REPO}")

dataset.push_to_hub(
    HF_REPO,
    token=HF_TOKEN,
    private=False
)

print(f"\n✅ Pushed to https://huggingface.co/datasets/{HF_REPO}")
