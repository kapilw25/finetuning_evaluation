---
library_name: transformers
tags:
- alignment
- safety
- sft
- llama-3
base_model: meta-llama/Llama-3.1-8B
datasets:
- PKU-Alignment/PKU-SafeRLHF
license: llama3.1
---

# llama3-8b-pku-sft-baseline-bf16

Fine-tuned [Llama-3.1-8B](meta-llama/Llama-3.1-8B) using **SFT** (Supervised Fine-Tuning on chosen responses only) on the PKU-SafeRLHF dataset for improved safety alignment.

## Model Details

- **Base Model**: [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- **Fine-tuning Method**: SFT
- **Dataset**: [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) (10,813 samples)
- **Training Date**: 2025-10-24
- **Precision**: BF16 (bfloat16)
- **Adapter Type**: LoRA (r=16, alpha=16, ~168MB)

## Training Hyperparameters

- **Learning Rate**: 0.0002
- **Batch Size** (per device): 2
- **Gradient Accumulation Steps**: 4 (effective batch size: 8)
- **Warmup Steps**: 100
- **Max Steps**: 1000
- **Weight Decay**: 0.01
- **LR Scheduler**: cosine
- **Optimizer**: adamw_torch
- **Max Sequence Length**: 2048

## Evaluation Results

- **Final Loss**: 1.5136

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "kapilw25/llama3-8b-pku-sft-baseline-bf16")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("kapilw25/llama3-8b-pku-sft-baseline-bf16")

# Generate
messages = [{"role": "user", "content": "Explain quantum computing"}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=128, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

- **Primary Use**: Safety-aligned conversational AI
- **Recommended**: Instruction following with harm refusal capabilities
- **Not Recommended**: Medical/legal advice, factual knowledge (use base Llama-3.1 for general tasks)

## Limitations

- Fine-tuned on English-only safety dataset (PKU-SafeRLHF)
- May refuse benign requests if phrased similarly to harmful prompts
- LoRA adapter only - requires base Llama-3.1-8B for inference

## License

Llama 3.1 Community License (same as base model)

## Citation

```bibtex
@misc{llama3_8b_pku_sft_baseline_bf16_2024,
  author = {User},
  title = {llama3-8b-pku-sft-baseline-bf16},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/kapilw25/llama3-8b-pku-sft-baseline-bf16}}
}
```

## Framework Versions

- **Transformers**: 4.46.3
- **PyTorch**: 2.5.1
- **TRL**: 0.12.1
- **PEFT**: 0.13.2
