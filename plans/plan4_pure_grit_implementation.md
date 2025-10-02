# Phase 4: Pure GRIT Implementation Guide

**Status:** Ready for execution
**Created:** 2025-10-02
**Objective:** Implement standalone Pure GRIT training without Unsloth interference to determine if GRIT provides benefits

---

## 📋 Manual Modifications for `unsloth_3_Pure_Grit/Llama3_8B_GRIT.ipynb`

### **CELL 5 - Installation (REPLACE)**

```python
%%capture
# Pure GRIT - NO Unsloth dependencies!
!pip install torch transformers datasets peft accelerate bitsandbytes
!pip install tensorboard tensorboardX
```

---

### **CELL 6 - GRIT Path (KEEP, but modify)**

```python
import sys
sys.path.insert(0, './grit')

# Verify GRIT modules are accessible
from grit.config import GRITConfig
from grit.manager import GRITManager
from grit.trainer import GritTrainer  # ✅ Use GritTrainer, not SFTTrainer
print("✅ GRIT modules imported successfully")
```

---

### **CELL 8 - Model Loading (REPLACE - Remove Unsloth)**

```python
# Pure GRIT - Use standard HuggingFace loading
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

max_seq_length = 2048
dtype = torch.bfloat16
load_in_4bit = True

# BitsAndBytes 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",  # Use base model, not Unsloth's quantized version
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=dtype,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("✅ Base model loaded successfully")
```

---

### **CELL 10 - LoRA Configuration (REPLACE - Use PEFT directly)**

```python
# Pure GRIT - Use PEFT's native LoRA, not Unsloth's wrapper
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,  # Note: train_alpaca.py uses alpha=32, but we match baseline
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.config.use_cache = False

model.print_trainable_parameters()
print("✅ LoRA adapters added successfully")
```

---

### **CELL 11 - GRIT Manager (MODIFY - Conservative settings)**

```python
from grit.config import GRITConfig
from grit.manager import GRITManager

# Initialize config
grit_config = GRITConfig()

# Model settings - match LoRA config
grit_config.lora_rank = 16
grit_config.lora_alpha = 16
grit_config.precision = "bf16"

# CONSERVATIVE SETTINGS for 200-step run
grit_config.kfac_update_freq = 10         # ✅ Moderate frequency
grit_config.reprojection_freq = 50        # ✅ Less frequent than hybrid
grit_config.kfac_damping = 1e-5           # ✅ Low damping for visible effect
grit_config.lambda_kfac = 1e-6            # ✅ Minimal K-FAC regularization
grit_config.lambda_reproj = 1e-5          # ✅ Minimal reprojection regularization
grit_config.kfac_min_samples = 16         # ✅ Lower than default (64), higher than aggressive (4)

# Warmup settings
grit_config.reprojection_warmup_steps = 20  # ✅ Skip early reprojection
grit_config.ng_warmup_steps = 0             # No warmup for natural gradient
grit_config.regularizer_warmup_steps = 0    # No warmup for regularizers

# Rank adaptation - DISABLED for fair comparison
grit_config.enable_rank_adaptation = False
grit_config.use_two_sided_reprojection = True

# Logging
grit_config.log_fisher_spectrum = False     # Disable to reduce overhead
grit_config.log_top_eigs = 0
grit_config.log_eig_heatmaps = False

print("🎯 Initializing GRIT Manager...")
grit_manager = GRITManager(
    model=model,
    config=grit_config,
    device="cuda",
)
print("✅ GRIT Manager initialized successfully!")
```

---

### **CELL 13-25 - Data Prep (REPLACE - Remove Unsloth functions)**

```python
from datasets import load_dataset

dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

# Tokenize function
def tokenize_function(example):
    instr = example['instruction'].strip()
    inp = example.get('input', "").strip()
    out = example['output'].strip()

    # Create prompt
    if inp:
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n{out}"

    # Tokenize
    tokenized = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"][:]

    return tokenized

# Apply tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset"
)

print(f"✅ Dataset tokenized: {len(tokenized_dataset)} samples")
```

---

### **CELL 28 - TensorBoard (MODIFY)**

```python
import os
from datetime import datetime

tensorboard_base_dir = "/home/ubuntu/DiskUsEast1/finetuning_evaluation/tensorboard_logs"
os.makedirs(tensorboard_base_dir, exist_ok=True)

run_name = "pure_grit"  # ✅ Changed from "hybrid_grit"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_run_dir = os.path.join(tensorboard_base_dir, f"{run_name}_{timestamp}")

print(f"📊 TensorBoard logs: {tensorboard_run_dir}")
```

---

### **CELL 30 - DELETE Verification Callback**

```python
# DELETE THIS CELL - Not needed for Pure GRIT
# GritTrainer handles preconditioning internally
```

---

### **CELL 31 - Trainer (REPLACE - Use GritTrainer)**

```python
from transformers import Seq2SeqTrainingArguments
from grit.trainer import GritTrainer

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=200,  # ✅ 200 steps for comparison
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    logging_first_step=True,

    # Optimizer settings
    optim="adamw_torch",  # ✅ Standard AdamW (not 8bit)
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=5,

    # Gradient settings
    max_grad_norm=1.5,  # ✅ Trust-region clipping for natural gradient
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # TensorBoard
    report_to="tensorboard",
    logging_dir=tensorboard_run_dir,

    # Misc
    seed=3407,
    remove_unused_columns=False,
    save_strategy="no",  # Don't save checkpoints during training
    dataloader_num_workers=0,
)

# Create GritTrainer (NOT SFTTrainer!)
trainer = GritTrainer(
    grit_manager=grit_manager,
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=None,  # Will use default
)

print("✅ GritTrainer initialized successfully!")
```

---

### **CELL 33 - Training (KEEP as-is)**

```python
trainer_stats = trainer.train()
```

---

### **CELLS 38-40 - Inference (REPLACE - Remove Unsloth)**

```python
# Set model to eval mode
model.eval()

# Prepare prompt
messages = [
    {"role": "user", "content": "Continue the fibonacci sequence! Your input is 1, 1, 2, 3, 5, 8,"}
]

# Format prompt (manual since we don't have chat template)
prompt = f"### Instruction:\n{messages[0]['content']}\n\n### Response:\n"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Generate
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        streamer=text_streamer,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
```

---

### **CELL 42 - Saving (MODIFY)**

```python
# Save LoRA adapters
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("✅ LoRA adapters saved to lora_model/")
```

---

### **CELLS 48-58 - Ollama (SKIP for now)**

```python
# SKIP Ollama deployment for initial testing
# Focus on comparing loss curves first
```

---

## 🎯 Summary of Changes

| Component | Hybrid (unsloth_2) | Pure GRIT (unsloth_3) |
|-----------|-------------------|----------------------|
| **Model Loading** | `FastLanguageModel.from_pretrained()` | `AutoModelForCausalLM.from_pretrained()` |
| **LoRA** | `FastLanguageModel.get_peft_model()` | `get_peft_model()` from PEFT |
| **Dataset** | `to_sharegpt()`, `apply_chat_template()` | Manual tokenization function |
| **Trainer** | `SFTTrainer` + callback | `GritTrainer` (built-in preconditioning) |
| **Preconditioning** | `GritVerificationCallback.on_optimizer_step()` | `GritTrainer.optimizer_step()` (internal) |
| **Optimizer** | `adamw_8bit` | `adamw_torch` (standard) |
| **Settings** | Aggressive (kfac_freq=5, min_samples=4) | Conservative (kfac_freq=10, min_samples=16) |

---

## ✅ Expected Outcomes

After these modifications:

1. **If loss curves DIVERGE:** Proves Unsloth was interfering with Hybrid GRIT
2. **If loss curves REMAIN IDENTICAL:** Proves GRIT doesn't help for this task/configuration

This will give us the definitive answer! 🔬

---

## 📊 Experimental Design

### Comparison Groups

| Model | Optimizer | Trainer | Preconditioning | TensorBoard Run |
|-------|-----------|---------|-----------------|----------------|
| Baseline (unsloth_ft) | AdamW 8-bit | SFTTrainer | None | `unsloth_ft_*` |
| Pure GRIT | AdamW standard | GritTrainer | Built-in (Fisher-based) | `pure_grit_*` |

### Key Metrics to Compare

1. **Training Loss Curves** - Primary indicator of GRIT effectiveness
2. **Training Time** - Computational overhead
3. **Memory Usage** - VRAM requirements
4. **Final Loss** - Convergence quality

### Success Criteria

- **GRIT Works:** Loss curves show clear divergence, GRIT converges faster or to lower loss
- **GRIT Fails:** Loss curves remain nearly identical, indicating no benefit from Fisher preconditioning
- **GRIT Hurts:** GRIT curve shows instability or slower convergence

---

## 🚀 Execution Steps

1. Manually modify `unsloth_3_Pure_Grit/Llama3_8B_GRIT.ipynb` following guide above
2. Delete old TensorBoard logs: `rm -rf /home/ubuntu/DiskUsEast1/finetuning_evaluation/tensorboard_logs/pure_grit_*`
3. Run the notebook (expected time: ~16 minutes for 200 steps)
4. Compare TensorBoard logs: `tensorboard --logdir=/home/ubuntu/DiskUsEast1/finetuning_evaluation/tensorboard_logs --port=6006`
5. Analyze results and document findings

---

## 📝 Notes

- **No Unsloth dependencies** - ensures no gradient management interference
- **Conservative GRIT settings** - balance between visibility and stability
- **Same seed (3407)** - ensure reproducibility with baseline
- **Same dataset** - vicgalle/alpaca-gpt4 for consistency
- **Focus on loss curves** - primary indicator of GRIT effectiveness before worrying about deployment
