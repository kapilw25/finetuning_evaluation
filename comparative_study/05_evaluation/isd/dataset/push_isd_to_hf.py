"""
Push ISD Dataset to HuggingFace

Dataset: 300 unique prompts × 10 instruction types = 3,000 test cases

Usage:
  cd /Users/kapilwanaskar/Downloads/research_projects/finetuning_evaluation
  python3 comparative_study/05_evaluation/isd/dataset/push_isd_to_hf.py

  # To only update README/license without regenerating data:
  python3 comparative_study/05_evaluation/isd/dataset/push_isd_to_hf.py --readme-only
"""

import sys
import argparse
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

from model_utils import load_hf_token

# Config
HF_TOKEN = load_hf_token(project_root)
HF_REPO = "anonymousML123/ISD-Instruction-Switch-Dataset"
NUM_PROMPTS = 300  # All unique prompts (12 categories × 25 prompts)

DATASET_README = """\
---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-generation
tags:
  - alignment
  - instruction-following
  - preference-optimization
  - safety
  - RLHF
  - DPO
  - CITA
  - ECLIPTICA
pretty_name: "ECLIPTICA: Instruction Switch Dataset (ISD)"
size_categories:
  - 1K<n<10K
---

# ECLIPTICA: Instruction Switch Dataset (ISD)

**ECLIPTICA** (**E**valuating **C**ontrollable **L**anguage **I**nstruction **P**olicy **T**ransfer via **I**nstruction-**C**onditioned **A**lignment) is a controlled benchmark for evaluating instruction-conditioned behavioral switching in LLMs.

## Key Design Principle

**Hold the user prompt fixed, vary only the alignment instruction.** This isolates policy switching from standard instruction following.

## Dataset Statistics

| Property | Value |
|---|---|
| Unique prompts | 300 (12 categories x 25) |
| Instruction types | 10 |
| Total test cases | 3,000 (300 x 10) |
| Language | English |

## Instruction Types

1. `default` - Standard helpful assistant
2. `concise` - Brief, to-the-point responses
3. `detailed` - Comprehensive, thorough responses
4. `professional` - Formal business tone
5. `educational` - Teaching-oriented explanations
6. `strict_safety` - Conservative refusal boundary
7. `permissive_safety` - Permissive safe guidance
8. `creative` - Imaginative, expressive style
9. `analytical` - Data-driven, logical analysis
10. `empathetic` - Emotionally supportive tone

## Fields

- `prompt_id`: Unique identifier for each prompt
- `prompt`: The user query (held fixed across instruction types)
- `instruction_type`: One of 10 alignment instruction types
- `instruction`: Full alignment instruction text
- `expected_characteristics`: Expected behavioral traits for evaluation
- `source`: Prompt source category

## Usage

```python
from datasets import load_dataset
ds = load_dataset("anonymousML123/ISD-Instruction-Switch-Dataset")
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{ecliptica2025,
  title={ECLIPTICA: Instruction-Conditioned Alignment via Contrastive Instruction-Tuned Alignment (CITA)},
  year={2025}
}
```

## License

This dataset is released under the [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) license.
"""


def push_readme(token: str, repo: str):
    """Upload README.md with CC-BY-4.0 license to HuggingFace dataset repo."""
    api = HfApi()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(DATASET_README)
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        token=token,
        commit_message="Add dataset card with CC-BY-4.0 license",
    )
    print(f"Uploaded README.md with CC-BY-4.0 license to {repo}")
    Path(tmp_path).unlink()


def push_data(token: str, repo: str):
    """Generate and push the full dataset."""
    from datasets import Dataset
    from instruction_switch_dataset import InstructionSwitchDataset

    print("Generating ISD dataset...")
    isd = InstructionSwitchDataset(seed=42)
    test_cases = isd.generate_dataset(num_prompts=NUM_PROMPTS)

    data = {
        "prompt_id": [tc.prompt_id for tc in test_cases],
        "prompt": [tc.prompt for tc in test_cases],
        "instruction_type": [tc.instruction_type for tc in test_cases],
        "instruction": [tc.instruction for tc in test_cases],
        "expected_characteristics": [tc.expected_characteristics for tc in test_cases],
        "source": [tc.source for tc in test_cases],
    }

    dataset = Dataset.from_dict(data)
    print(f"\nDataset: {len(dataset)} test cases")
    print(f"Pushing to: {repo}")

    dataset.push_to_hub(repo, token=token, private=False)
    print(f"\nPushed dataset to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push ISD dataset to HuggingFace")
    parser.add_argument("--readme-only", action="store_true", help="Only update README.md (skip data push)")
    args = parser.parse_args()

    if args.readme_only:
        push_readme(HF_TOKEN, HF_REPO)
    else:
        push_data(HF_TOKEN, HF_REPO)
        push_readme(HF_TOKEN, HF_REPO)

    print(f"\nhttps://huggingface.co/datasets/{HF_REPO}")
