---
license: apache-2.0
library_name: peft
tags:
- base_model:adapter:meta-llama/Llama-3.1-8B
- lora
- transformers
pipeline_tag: text-generation
base_model: meta-llama/Llama-3.1-8B
---

# TRL Model

This is a [TRL language model](https://github.com/huggingface/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="kapilw25/outputs/training/PPO_Instruct/lora_model_PPO_Instruct")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("kapilw25/outputs/training/PPO_Instruct/lora_model_PPO_Instruct")
model = AutoModelForCausalLMWithValueHead.from_pretrained("kapilw25/outputs/training/PPO_Instruct/lora_model_PPO_Instruct")

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
### Framework versions

- PEFT 0.17.1