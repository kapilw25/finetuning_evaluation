# Plan 7: Comprehensive Alignment Methods Comparison for Ecliptica Paper Submission

## Research Objective

Implement and evaluate **7 alignment methods** on the Vaibhaav/alignment-instructions dataset to validate the effectiveness of:
1. **Contrastive Instruction-Tuned Alignment (CITA)** - Novel method from Ecliptica paper
2. **GRIT optimization** - Natural gradient descent for improved convergence
3. **CITA+GRIT combination** - Novel contribution combining both approaches

---

## Paper Context: Ecliptica Framework

**Paper:** "Ecliptica: A Novel Framework for Instruction-Driven Alignment in LLM" (Anonymous ACL Submission)

**Key Innovation:** CITA-KL Loss (Page 7, lines 155-156)
```
L_CITA-KL = -log[exp(β·log π_θ(Y+|I,X)) / (exp(β·log π_θ(Y+|I,X)) + exp(β·log π_θ(Y-|I,X)))]
            + λ·KL[π_θ(·|I,X) || π_0(·|I,X)]
```

**Components:**
- **I** (Instruction): Alignment directive (e.g., "Refuse illegal activities")
- **X** (User Prompt): Potentially harmful request
- **Y+** (Accepted): Safe, aligned response
- **Y-** (Rejected): Harmful, misaligned response
- **β**: Contrastive temperature (default: 0.1)
- **λ**: KL regularization weight (default: 0.01)

**Hypothesis:** Explicit instruction conditioning + contrastive learning improves alignment over standard DPO.

---

## Experimental Design

### Dataset: Vaibhaav/alignment-instructions
- **Size:** 50,001 training samples
- **Format:** (Instruction, Prompt, Accepted Response, Rejected Response)
- **Domain:** Safety alignment (illegal activities, hate speech, self-harm, etc.)
- **Split:** 90% train (45,000), 10% validation (5,000)

### Evaluation: AQI (Alignment Quality Index)
- **Metric:** Embedding-based clustering quality
- **Dataset:** hasnat79/litmus (1000 safe + unsafe prompts)
- **Score Range:** 0.0 (worst) to 1.0 (perfect alignment)
- **Components:** Silhouette score + Chi-squared separation test

### Base Model: Llama-3-8B
- **Quantization:** 4-bit NF4
- **LoRA:** Rank=16, Alpha=16, Target=7 projection layers
- **Trainable Params:** 41.9M (0.52% of total)

---

## Method Comparison Table

| # | Method | Dataset Usage | Loss Function | Optimizer | Implementation Status |
|---|--------|---------------|---------------|-----------|----------------------|
| 1 | **Baseline** | None (pretrained) | N/A | N/A | ✅ Available |
| 2 | **SFT** | Accepted only | L_SFT = -log P(Y+\|I,X) | AdamW 8-bit | ✅ Trained |
| 3 | **SFT+GRIT** | Accepted only | L_SFT + GRIT preconditioning | AdamW + K-FAC | ✅ Trained |
| 4 | **DPO** | Accepted + Rejected | L_DPO = contrastive loss | AdamW 8-bit | 🔨 **To Train** |
| 5 | **DPO+GRIT** | Accepted + Rejected | L_DPO + GRIT preconditioning | AdamW + K-FAC | 🔨 **To Implement** |
| 6 | **CITA** | Accepted + Rejected + Instruction | L_CITA-KL | AdamW 8-bit | 🔨 **To Implement** |
| 7 | **CITA+GRIT** | Accepted + Rejected + Instruction | L_CITA-KL + GRIT preconditioning | AdamW + K-FAC | 🔨 **To Implement** ⭐ |

---

## Implementation Roadmap

### Phase 1: Train DPO Baseline (Week 1)
**File:** `comparative_study/04_DPO/Llama3_DPO_Vaibhaav.ipynb`

**Steps:**
1. Adapt `Zephyr_(7B)_DPO.ipynb` to Llama-3-8B
2. Replace UltraFeedback dataset with Vaibhaav/alignment-instructions
3. Format data for DPO (prompt, chosen, rejected)
4. Train for 300 steps (same as SFT for fair comparison)
5. Save LoRA adapters to `lora_model_DPO_Baseline/`

**Key Code Changes:**
```python
# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Dataset preparation
dataset = load_dataset("Vaibhaav/alignment-instructions", split="train")

def format_dpo(example):
    """Format Vaibhaav dataset for DPO training"""
    # Combine Prompt (system) + Instruction (user request)
    prompt = f"{example['Prompt']}\n\n{example['Instruction generated']}"

    return {
        "prompt": prompt,
        "chosen": example['Accepted Response'],
        "rejected": example['Rejected Response'],
    }

dataset = dataset.map(format_dpo)

# DPO Training
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,  # Will create reference model internally
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 300,  # Match SFT
        learning_rate = 5e-6,
        optim = "adamw_8bit",
        beta = 0.1,  # Contrastive temperature
        output_dir = "outputs",
        logging_dir = "tensorboard_logs/DPO_Baseline_<timestamp>",
        report_to = "tensorboard",
    ),
    train_dataset = dataset,
    tokenizer = tokenizer,
    max_length = 2048,
    max_prompt_length = 1024,
)
```

**Expected Outputs:**
- Training time: ~45 minutes (300 steps)
- Peak VRAM: ~10 GB (stores reference model)
- LoRA adapters: `lora_model_DPO_Baseline/`
- TensorBoard logs: `tensorboard_logs/DPO_Baseline_<timestamp>/`

---

### Phase 2: Implement DPO+GRIT (Week 2)
**File:** `comparative_study/04_DPO/Llama3_DPO_GRIT_Vaibhaav.ipynb`

**Challenge:** DPOTrainer is not compatible with GritTrainer (different base classes)

**Solution:** Create custom CITAGritTrainer that:
1. Extends GritTrainer base class
2. Implements DPO loss computation
3. Applies GRIT preconditioning to DPO gradients

**Implementation:**
```python
# File: comparative_study/04_DPO/grit_dpo_trainer.py

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainingArguments
from comparative_study.02_SFT_GRIT.grit.trainer import GritTrainer

class DPOGritTrainer(GritTrainer):
    """
    Custom trainer combining DPO loss with GRIT optimization
    """
    def __init__(
        self,
        model,
        ref_model,
        grit_manager,
        args,
        train_dataset,
        tokenizer,
        beta=0.1,
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            grit_manager=grit_manager,
            **kwargs
        )
        self.ref_model = ref_model
        self.beta = beta

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute DPO loss: -log σ(β * (log π(y+) - log π(y-)))
        """
        # Forward pass on policy model
        chosen_logps = self._get_batch_logps(
            model,
            inputs["input_ids_chosen"],
            inputs["attention_mask_chosen"],
            inputs["labels_chosen"]
        )
        rejected_logps = self._get_batch_logps(
            model,
            inputs["input_ids_rejected"],
            inputs["attention_mask_rejected"],
            inputs["labels_rejected"]
        )

        # Forward pass on reference model
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logps(
                self.ref_model,
                inputs["input_ids_chosen"],
                inputs["attention_mask_chosen"],
                inputs["labels_chosen"]
            )
            ref_rejected_logps = self._get_batch_logps(
                self.ref_model,
                inputs["input_ids_rejected"],
                inputs["attention_mask_rejected"],
                inputs["labels_rejected"]
            )

        # DPO loss
        logits = self.beta * (
            (chosen_logps - ref_chosen_logps) -
            (rejected_logps - ref_rejected_logps)
        )
        loss = -F.logsigmoid(logits).mean()

        return (loss, {"logits": logits}) if return_outputs else loss

    def _get_batch_logps(self, model, input_ids, attention_mask, labels):
        """Compute log probabilities for a batch"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits

        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for target tokens
        labels = labels[:, 1:].clone()  # Shift labels
        log_probs = log_probs[:, :-1, :]  # Shift logits

        # Get log prob of each target token
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)

        # Sum over sequence length
        return per_token_logps.sum(-1)
```

**Usage:**
```python
# Initialize GRIT manager
from grit.config import GRITConfig
from grit.manager import GRITManager

grit_config = GRITConfig()
grit_config.lora_rank = 16
grit_config.kfac_update_freq = 10
grit_config.reprojection_freq = 50

grit_manager = GRITManager(
    model=model,
    config=grit_config,
    device="cuda",
)

# Create reference model (copy of base model)
ref_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Initialize DPO+GRIT trainer
trainer = DPOGritTrainer(
    model=model,
    ref_model=ref_model,
    grit_manager=grit_manager,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,
)

trainer.train()
```

**Expected Outputs:**
- Training time: ~2.5 hours (300 steps, 3.5x slower)
- Peak VRAM: ~20 GB (reference model + GRIT covariances)
- LoRA adapters: `lora_model_DPO_GRIT/`

---

### Phase 3: Implement CITA Baseline (Week 3)
**File:** `comparative_study/03b_De_Alignment_main/Llama3_CITA_Vaibhaav.ipynb`

**Key Difference from DPO:** Explicit instruction conditioning in prompt format

**CITA Loss Implementation:**
```python
# File: comparative_study/03b_De_Alignment_main/cita_trainer.py

import torch
import torch.nn.functional as F
from trl import DPOTrainer

class CITATrainer(DPOTrainer):
    """
    CITA Trainer - Contrastive Instruction-Tuned Alignment

    Extends DPO with explicit instruction conditioning as per Ecliptica paper
    """
    def __init__(
        self,
        model,
        ref_model,
        args,
        train_dataset,
        tokenizer,
        beta=0.1,
        lambda_kl=0.01,
        **kwargs
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=beta,
            **kwargs
        )
        self.lambda_kl = lambda_kl

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        CITA-KL Loss from Ecliptica paper (Page 7, line 156):

        L = -log[exp(β·log π(Y+|I,X)) / (exp(β·log π(Y+|I,X)) + exp(β·log π(Y-|I,X)))]
            + λ·KL[π_θ || π_0]
        """
        # Get log probabilities from policy model
        chosen_logps = self._get_batch_logps(
            model,
            inputs["input_ids_chosen"],
            inputs["attention_mask_chosen"],
            inputs["labels_chosen"]
        )
        rejected_logps = self._get_batch_logps(
            model,
            inputs["input_ids_rejected"],
            inputs["attention_mask_rejected"],
            inputs["labels_rejected"]
        )

        # Get log probabilities from reference model
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logps(
                self.ref_model,
                inputs["input_ids_chosen"],
                inputs["attention_mask_chosen"],
                inputs["labels_chosen"]
            )
            ref_rejected_logps = self._get_batch_logps(
                self.ref_model,
                inputs["input_ids_rejected"],
                inputs["attention_mask_rejected"],
                inputs["labels_rejected"]
            )

        # Contrastive term (Ecliptica equation)
        # Note: DPO uses (chosen - ref_chosen) - (rejected - ref_rejected)
        # CITA uses raw logps with different formulation
        logits_chosen = self.beta * chosen_logps
        logits_rejected = self.beta * rejected_logps

        # Softmax over [chosen, rejected]
        logits_concat = torch.stack([logits_chosen, logits_rejected], dim=1)
        probs = F.softmax(logits_concat, dim=1)

        # Loss is -log P(chosen)
        loss_contrastive = -torch.log(probs[:, 0] + 1e-10).mean()

        # KL regularization (prevent drift from reference)
        loss_kl = (
            (chosen_logps - ref_chosen_logps).pow(2).mean() +
            (rejected_logps - ref_rejected_logps).pow(2).mean()
        ) / 2

        # Total CITA loss
        loss = loss_contrastive + self.lambda_kl * loss_kl

        # Logging
        metrics = {
            "loss_contrastive": loss_contrastive.item(),
            "loss_kl": loss_kl.item(),
            "chosen_logps": chosen_logps.mean().item(),
            "rejected_logps": rejected_logps.mean().item(),
        }

        return (loss, metrics) if return_outputs else loss
```

**Dataset Formatting (Key Difference):**
```python
def format_cita(example):
    """
    CITA format explicitly separates instruction from user prompt

    Structure:
    <|system|>
    {Alignment Instruction}
    </s>
    <|user|>
    {User Request}
    </s>
    <|assistant|>
    {Response}
    </s>
    """
    # Ecliptica-style formatting (Table 2, page 2)
    instruction = example['Instruction generated'].strip()
    user_prompt = example['Prompt'].strip()

    # Create messages with explicit instruction separation
    chosen_messages = [
        {"role": "system", "content": instruction},  # Alignment directive
        {"role": "user", "content": user_prompt},    # User request
        {"role": "assistant", "content": example['Accepted Response'].strip()}
    ]

    rejected_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example['Rejected Response'].strip()}
    ]

    return {
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }

dataset = dataset.map(format_cita)
```

**Training Code:**
```python
from cita_trainer import CITATrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Add LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Create reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Initialize CITA trainer
cita_trainer = CITATrainer(
    model=model,
    ref_model=ref_model,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=300,
        learning_rate=5e-6,
        optim="adamw_8bit",
        output_dir="outputs/CITA_Baseline",
        logging_dir="tensorboard_logs/CITA_Baseline_<timestamp>",
        report_to="tensorboard",
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,      # Contrastive temperature (Ecliptica default)
    lambda_kl=0.01, # KL regularization weight
)

cita_trainer.train()
model.save_pretrained("lora_model_CITA_Baseline")
```

**Expected Outputs:**
- Training time: ~50 minutes (300 steps)
- Peak VRAM: ~10 GB
- LoRA adapters: `lora_model_CITA_Baseline/`

---

### Phase 4: Implement CITA+GRIT (Week 4) ⭐ **Novel Contribution**
**File:** `comparative_study/03b_De_Alignment_main/Llama3_CITA_GRIT_Vaibhaav.ipynb`

**Innovation:** Combine Ecliptica's CITA loss with GRIT's natural gradient optimization

**Implementation:**
```python
# File: comparative_study/03b_De_Alignment_main/cita_grit_trainer.py

import torch
import torch.nn.functional as F
from comparative_study.02_SFT_GRIT.grit.trainer import GritTrainer

class CITAGritTrainer(GritTrainer):
    """
    CITA+GRIT Trainer - Novel combination

    Combines:
    1. CITA loss (instruction-conditioned contrastive learning)
    2. GRIT optimization (natural gradient via K-FAC)
    """
    def __init__(
        self,
        model,
        ref_model,
        grit_manager,
        args,
        train_dataset,
        tokenizer,
        data_collator,
        beta=0.1,
        lambda_kl=0.01,
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            grit_manager=grit_manager,
            **kwargs
        )
        self.ref_model = ref_model
        self.beta = beta
        self.lambda_kl = lambda_kl

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        CITA-KL Loss with GRIT preconditioning

        Loss computation same as CITA, but gradients are preconditioned
        by GRIT manager during backward pass
        """
        # Get log probabilities from policy model
        chosen_logps = self._get_batch_logps(
            model,
            inputs["input_ids_chosen"],
            inputs["attention_mask_chosen"],
            inputs["labels_chosen"]
        )
        rejected_logps = self._get_batch_logps(
            model,
            inputs["input_ids_rejected"],
            inputs["attention_mask_rejected"],
            inputs["labels_rejected"]
        )

        # Get log probabilities from reference model
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logps(
                self.ref_model,
                inputs["input_ids_chosen"],
                inputs["attention_mask_chosen"],
                inputs["labels_chosen"]
            )
            ref_rejected_logps = self._get_batch_logps(
                self.ref_model,
                inputs["input_ids_rejected"],
                inputs["attention_mask_rejected"],
                inputs["labels_rejected"]
            )

        # Contrastive loss (Ecliptica formulation)
        logits_chosen = self.beta * chosen_logps
        logits_rejected = self.beta * rejected_logps
        logits_concat = torch.stack([logits_chosen, logits_rejected], dim=1)
        probs = F.softmax(logits_concat, dim=1)
        loss_contrastive = -torch.log(probs[:, 0] + 1e-10).mean()

        # KL regularization
        loss_kl = (
            (chosen_logps - ref_chosen_logps).pow(2).mean() +
            (rejected_logps - ref_rejected_logps).pow(2).mean()
        ) / 2

        # Total loss (GRIT will precondition gradients automatically)
        loss = loss_contrastive + self.lambda_kl * loss_kl

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "cita/loss_contrastive": loss_contrastive.item(),
                "cita/loss_kl": loss_kl.item(),
                "cita/margin": (chosen_logps - rejected_logps).mean().item(),
                "cita/chosen_logps": chosen_logps.mean().item(),
                "cita/rejected_logps": rejected_logps.mean().item(),
            })

        return loss

    def _get_batch_logps(self, model, input_ids, attention_mask, labels):
        """Compute log probabilities for a batch"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits

        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)

        # Shift for autoregressive modeling
        labels = labels[:, 1:].clone()
        log_probs = log_probs[:, :-1, :]

        # Gather log prob of target tokens
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)

        # Mask padding tokens
        mask = (labels != -100).float()
        per_token_logps = per_token_logps * mask

        # Sum over sequence
        return per_token_logps.sum(-1)
```

**Data Collator for CITA:**
```python
from dataclasses import dataclass
from typing import Dict, List
import torch

@dataclass
class CITADataCollator:
    """
    Collates data for CITA training (chosen + rejected pairs)
    """
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Batch collation for CITA

        Expected input format:
        features = [
            {
                "chosen": [{"role": "system", "content": "..."}, ...],
                "rejected": [{"role": "system", "content": "..."}, ...],
            },
            ...
        ]
        """
        batch_chosen = []
        batch_rejected = []

        for feature in features:
            # Apply chat template
            chosen_text = self.tokenizer.apply_chat_template(
                feature["chosen"],
                tokenize=False,
                add_generation_prompt=False,
            )
            rejected_text = self.tokenizer.apply_chat_template(
                feature["rejected"],
                tokenize=False,
                add_generation_prompt=False,
            )

            batch_chosen.append(chosen_text)
            batch_rejected.append(rejected_text)

        # Tokenize
        chosen_tokens = self.tokenizer(
            batch_chosen,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        rejected_tokens = self.tokenizer(
            batch_rejected,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (copy input_ids, mask padding)
        chosen_labels = chosen_tokens["input_ids"].clone()
        chosen_labels[chosen_labels == self.tokenizer.pad_token_id] = -100

        rejected_labels = rejected_tokens["input_ids"].clone()
        rejected_labels[rejected_labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "labels_chosen": chosen_labels,
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"],
            "labels_rejected": rejected_labels,
        }
```

**Full Training Script:**
```python
# Notebook: comparative_study/03b_De_Alignment_main/Llama3_CITA_GRIT_Vaibhaav.ipynb

# 1. Load base model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare model for k-bit training
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.config.use_cache = False

# 3. Initialize GRIT manager
import sys
sys.path.insert(0, './comparative_study/02_SFT_GRIT/grit')

from grit.config import GRITConfig
from grit.manager import GRITManager

grit_config = GRITConfig()
grit_config.lora_rank = 16
grit_config.lora_alpha = 16
grit_config.kfac_update_freq = 10
grit_config.reprojection_freq = 50
grit_config.kfac_damping = 1e-5
grit_config.lambda_kfac = 1e-6
grit_config.lambda_reproj = 1e-5

grit_manager = GRITManager(
    model=model,
    config=grit_config,
    device="cuda",
)

# 4. Load dataset
from datasets import load_dataset

dataset = load_dataset("Vaibhaav/alignment-instructions", split="train")

def format_cita(example):
    instruction = example['Instruction generated'].strip()
    user_prompt = example['Prompt'].strip()

    chosen_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example['Accepted Response'].strip()}
    ]

    rejected_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example['Rejected Response'].strip()}
    ]

    return {
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }

dataset = dataset.map(format_cita, remove_columns=dataset.column_names)

# 5. Create reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 6. Initialize data collator
data_collator = CITADataCollator(tokenizer=tokenizer, max_length=2048)

# 7. Initialize CITA+GRIT trainer
from transformers import Seq2SeqTrainingArguments
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_dir = f"tensorboard_logs/CITA_GRIT_{timestamp}"

training_args = Seq2SeqTrainingArguments(
    output_dir="outputs/CITA_GRIT",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=300,
    learning_rate=5e-6,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_steps=1,
    logging_dir=tensorboard_dir,
    report_to="tensorboard",
    bf16=True,
    seed=3407,
)

sys.path.insert(0, './comparative_study/03b_De_Alignment_main')
from cita_grit_trainer import CITAGritTrainer

trainer = CITAGritTrainer(
    model=model,
    ref_model=ref_model,
    grit_manager=grit_manager,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    beta=0.1,
    lambda_kl=0.01,
)

# 8. Train
trainer.train()

# 9. Save
model.save_pretrained("lora_model_CITA_GRIT")
tokenizer.save_pretrained("lora_model_CITA_GRIT")

print("✅ CITA+GRIT training complete!")
```

**Expected Outputs:**
- Training time: ~2.5 hours (300 steps)
- Peak VRAM: ~20 GB (reference model + GRIT)
- LoRA adapters: `lora_model_CITA_GRIT/`
- Novel metrics: `cita/margin`, `grit/kfac_updates`, `grit/reprojection_events`

---

## Phase 5: Evaluation Pipeline (Week 5)

### Evaluation Setup
**File:** `comparative_study/06_Evaluation/eval_all_methods_vaibhaav.ipynb`

**Adapted from:** `Legacy_code/baseEvalAqi.ipynb`

**Models to Evaluate:**
```python
MODELS_TO_EVALUATE = {
    "Baseline": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter": None,
    },
    "SFT": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter": "comparative_study/01_QLoRA_Baseline/lora_model_SFT_Baseline",
    },
    "SFT+GRIT": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter": "comparative_study/02_SFT_GRIT/lora_model_SFT_GRIT",
    },
    "DPO": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter": "comparative_study/04_DPO/lora_model_DPO_Baseline",
    },
    "DPO+GRIT": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter": "comparative_study/04_DPO/lora_model_DPO_GRIT",
    },
    "CITA": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter": "comparative_study/03b_De_Alignment_main/lora_model_CITA_Baseline",
    },
    "CITA+GRIT": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter": "comparative_study/03b_De_Alignment_main/lora_model_CITA_GRIT",
    },
}
```

**Evaluation Code:**
```python
# Setup AQI evaluation
import sys
sys.path.insert(0, "comparative_study/03_AQI_EVAL/src")
from aqi.aqi_dealign_chi_sil import *

import torch, gc, os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# Configuration
OUTPUT_DIR = "comparative_study/06_Evaluation/results_all_methods"
DATASET_NAME = "hasnat79/litmus"
SAMPLES_PER_CATEGORY = 500
GAMMA = 0.5
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(RANDOM_SEED)

# Load evaluation dataset
balanced_eval_df = load_and_balance_dataset(
    dataset_name=DATASET_NAME,
    samples_per_category=SAMPLES_PER_CATEGORY,
    split='train'
)

if 'axiom' not in balanced_eval_df.columns:
    balanced_eval_df['axiom'] = 'overall'
if 'prompt' in balanced_eval_df.columns and 'input' not in balanced_eval_df.columns:
    balanced_eval_df = balanced_eval_df.rename(columns={'prompt': 'input'})

# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Evaluation function
def run_full_evaluation(model, tokenizer, model_display_name, output_sub_dir, balanced_df):
    model_output_dir = os.path.join(OUTPUT_DIR, output_sub_dir)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n--- Extracting Embeddings: {model_display_name} ---")
    cache_file = os.path.join(model_output_dir, "embeddings.pkl")
    processed_df = process_model_data(
        model, tokenizer, balanced_df,
        model_name=model_display_name,
        cache_file=cache_file,
        force_recompute=True
    )

    print(f"\n--- Calculating AQI: {model_display_name} ---")
    results, embeddings_3d, _, _ = analyze_by_axiom(
        processed_df,
        model_name=model_display_name,
        gamma=GAMMA,
        dim_reduction_method='tsne'
    )

    create_metrics_summary(results, model_display_name, output_dir=model_output_dir)

    if 'overall' in embeddings_3d and embeddings_3d['overall'] is not None:
        visualize_clusters_3d(
            embeddings_3d['overall'],
            processed_df['safety_label_binary'].values,
            results['overall'],
            axiom='overall',
            title=f"{model_display_name} - Overall Clusters",
            output_dir=model_output_dir
        )

    return results.get('overall', {}).get('AQI', 'N/A')

# Evaluate all models
aqi_scores = {}

for model_name, config in MODELS_TO_EVALUATE.items():
    print(f"\n{'='*80}")
    print(f"  EVALUATING: {model_name}")
    print(f"{'='*80}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quant_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load adapter if exists
    if config["adapter"] is not None:
        print(f"Loading adapter from {config['adapter']}...")
        model = PeftModel.from_pretrained(base_model, config["adapter"])
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()

    # Evaluate
    aqi_score = run_full_evaluation(
        model, tokenizer,
        model_name,
        f"{model_name.lower().replace('+', '_')}_results",
        balanced_eval_df
    )

    aqi_scores[model_name] = aqi_score

    # Cleanup
    del base_model, model
    gc.collect()
    torch.cuda.empty_cache()

# Final report
print("\n" + "="*80)
print("              FINAL COMPARATIVE REPORT - ALL METHODS")
print("="*80)

for model_name, aqi_score in aqi_scores.items():
    print(f"{model_name:20s}: AQI = {aqi_score:.4f}")

print("-" * 80)

# Calculate improvements
baseline_aqi = aqi_scores["Baseline"]

print("\nImprovements over Baseline:")
for model_name, aqi_score in aqi_scores.items():
    if model_name == "Baseline":
        continue
    delta = aqi_score - baseline_aqi
    pct = (delta / abs(baseline_aqi)) * 100 if abs(baseline_aqi) > 1e-6 else 0
    print(f"{model_name:20s}: ΔAQI = {delta:+.4f} ({pct:+.2f}%)")

print("\nKey Comparisons:")
print(f"GRIT Effect on SFT:  {aqi_scores['SFT+GRIT'] - aqi_scores['SFT']:+.4f}")
print(f"GRIT Effect on DPO:  {aqi_scores['DPO+GRIT'] - aqi_scores['DPO']:+.4f}")
print(f"GRIT Effect on CITA: {aqi_scores['CITA+GRIT'] - aqi_scores['CITA']:+.4f}")
print(f"CITA vs DPO:         {aqi_scores['CITA'] - aqi_scores['DPO']:+.4f}")
print(f"Best Method:         {max(aqi_scores, key=aqi_scores.get)}")

print("="*80)
```

**Expected Output Structure:**
```
comparative_study/06_Evaluation/results_all_methods/
├── baseline_results/
│   ├── embeddings.pkl
│   ├── metrics_summary.csv
│   └── overall_clusters_3d.png
├── sft_results/
│   ├── embeddings.pkl
│   ├── metrics_summary.csv
│   └── overall_clusters_3d.png
├── sft_grit_results/
│   ├── embeddings.pkl
│   ├── metrics_summary.csv
│   └── overall_clusters_3d.png
├── dpo_results/
│   ├── embeddings.pkl
│   ├── metrics_summary.csv
│   └── overall_clusters_3d.png
├── dpo_grit_results/
│   ├── embeddings.pkl
│   ├── metrics_summary.csv
│   └── overall_clusters_3d.png
├── cita_results/
│   ├── embeddings.pkl
│   ├── metrics_summary.csv
│   └── overall_clusters_3d.png
├── cita_grit_results/
│   ├── embeddings.pkl
│   ├── metrics_summary.csv
│   └── overall_clusters_3d.png
└── comparison_plot.png
```

---

## Phase 6: Results Analysis & Paper Figures (Week 6)

### Create Comparison Visualizations

**File:** `comparative_study/06_Evaluation/create_paper_figures.ipynb`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load AQI scores (from evaluation results)
aqi_scores = {
    "Baseline": 0.4523,  # Example values
    "SFT": 0.5234,
    "SFT+GRIT": 0.5512,
    "DPO": 0.5678,
    "DPO+GRIT": 0.5891,
    "CITA": 0.6123,
    "CITA+GRIT": 0.6445,  # Expected best
}

# Figure 1: AQI Score Comparison (Bar Chart)
plt.figure(figsize=(10, 6))
methods = list(aqi_scores.keys())
scores = list(aqi_scores.values())
colors = ['gray', 'lightblue', 'blue', 'lightgreen', 'green', 'lightcoral', 'red']

bars = plt.bar(methods, scores, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('AQI Score', fontsize=14)
plt.xlabel('Method', fontsize=14)
plt.title('Alignment Quality Index (AQI) Comparison', fontsize=16, fontweight='bold')
plt.ylim(0, 1.0)
plt.axhline(y=aqi_scores['Baseline'], color='gray', linestyle='--', label='Baseline')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('comparative_study/06_Evaluation/figure1_aqi_comparison.png', dpi=300)
plt.show()

# Figure 2: Improvement over Baseline (Delta Plot)
baseline = aqi_scores['Baseline']
deltas = {k: v - baseline for k, v in aqi_scores.items() if k != 'Baseline'}

plt.figure(figsize=(10, 6))
methods = list(deltas.keys())
improvements = list(deltas.values())
colors_delta = ['lightblue', 'blue', 'lightgreen', 'green', 'lightcoral', 'red']

bars = plt.barh(methods, improvements, color=colors_delta, edgecolor='black', linewidth=1.5)
plt.xlabel('ΔAQI (Improvement over Baseline)', fontsize=14)
plt.ylabel('Method', fontsize=14)
plt.title('Alignment Improvement over Baseline', fontsize=16, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.savefig('comparative_study/06_Evaluation/figure2_delta_aqi.png', dpi=300)
plt.show()

# Figure 3: GRIT Effect Analysis
grit_effects = {
    "SFT → SFT+GRIT": aqi_scores['SFT+GRIT'] - aqi_scores['SFT'],
    "DPO → DPO+GRIT": aqi_scores['DPO+GRIT'] - aqi_scores['DPO'],
    "CITA → CITA+GRIT": aqi_scores['CITA+GRIT'] - aqi_scores['CITA'],
}

plt.figure(figsize=(8, 5))
transitions = list(grit_effects.keys())
effects = list(grit_effects.values())

plt.bar(transitions, effects, color=['blue', 'green', 'red'],
        edgecolor='black', linewidth=1.5, alpha=0.7)
plt.ylabel('AQI Improvement from GRIT', fontsize=14)
plt.xlabel('Method Transition', fontsize=14)
plt.title('Effect of GRIT Optimization on Different Methods', fontsize=16, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('comparative_study/06_Evaluation/figure3_grit_effect.png', dpi=300)
plt.show()

# Figure 4: Method Pairing Comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# SFT vs SFT+GRIT
ax[0].bar(['SFT', 'SFT+GRIT'],
          [aqi_scores['SFT'], aqi_scores['SFT+GRIT']],
          color=['lightblue', 'blue'], edgecolor='black')
ax[0].set_ylabel('AQI Score')
ax[0].set_title('SFT Methods')
ax[0].set_ylim(0, 1.0)

# DPO vs DPO+GRIT
ax[1].bar(['DPO', 'DPO+GRIT'],
          [aqi_scores['DPO'], aqi_scores['DPO+GRIT']],
          color=['lightgreen', 'green'], edgecolor='black')
ax[1].set_ylabel('AQI Score')
ax[1].set_title('DPO Methods')
ax[1].set_ylim(0, 1.0)

# CITA vs CITA+GRIT
ax[2].bar(['CITA', 'CITA+GRIT'],
          [aqi_scores['CITA'], aqi_scores['CITA+GRIT']],
          color=['lightcoral', 'red'], edgecolor='black')
ax[2].set_ylabel('AQI Score')
ax[2].set_title('CITA Methods (Ours)')
ax[2].set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('comparative_study/06_Evaluation/figure4_method_pairs.png', dpi=300)
plt.show()

# Figure 5: Training Cost vs Performance
training_times = {
    "SFT": 39,  # minutes
    "SFT+GRIT": 137,
    "DPO": 45,
    "DPO+GRIT": 150,
    "CITA": 50,
    "CITA+GRIT": 155,
}

vram_usage = {
    "SFT": 8.7,  # GB
    "SFT+GRIT": 18.8,
    "DPO": 10,
    "DPO+GRIT": 20,
    "CITA": 10,
    "CITA+GRIT": 20,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Training time vs AQI
methods_subset = ['SFT', 'SFT+GRIT', 'DPO', 'DPO+GRIT', 'CITA', 'CITA+GRIT']
times = [training_times[m] for m in methods_subset]
scores_subset = [aqi_scores[m] for m in methods_subset]

ax1.scatter(times, scores_subset, s=200, c=['lightblue', 'blue', 'lightgreen', 'green', 'lightcoral', 'red'],
           edgecolor='black', linewidth=2, alpha=0.7)
for i, method in enumerate(methods_subset):
    ax1.annotate(method, (times[i], scores_subset[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
ax1.set_xlabel('Training Time (minutes)', fontsize=12)
ax1.set_ylabel('AQI Score', fontsize=12)
ax1.set_title('Training Efficiency', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# VRAM vs AQI
vram_values = [vram_usage[m] for m in methods_subset]
ax2.scatter(vram_values, scores_subset, s=200, c=['lightblue', 'blue', 'lightgreen', 'green', 'lightcoral', 'red'],
           edgecolor='black', linewidth=2, alpha=0.7)
for i, method in enumerate(methods_subset):
    ax2.annotate(method, (vram_values[i], scores_subset[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
ax2.set_xlabel('Peak VRAM (GB)', fontsize=12)
ax2.set_ylabel('AQI Score', fontsize=12)
ax2.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('comparative_study/06_Evaluation/figure5_efficiency_analysis.png', dpi=300)
plt.show()

print("✅ All figures generated successfully!")
```

---

## Expected Results Summary

### Hypothesis: Performance Ranking (Best to Worst)

1. **CITA+GRIT** - Novel contribution (instruction conditioning + natural gradient)
2. **CITA** - Ecliptica's method (instruction conditioning)
3. **DPO+GRIT** - Contrastive learning + optimization
4. **DPO** - Standard contrastive learning
5. **SFT+GRIT** - Simple SFT + optimization
6. **SFT** - Standard supervised fine-tuning
7. **Baseline** - No alignment training

### Key Research Questions to Answer

| Question | Metric | Hypothesis |
|----------|--------|-----------|
| **Q1:** Does instruction conditioning improve alignment? | CITA > DPO | Yes (Ecliptica's claim) |
| **Q2:** Does GRIT improve convergence? | All +GRIT > Non-GRIT | Yes (natural gradient helps) |
| **Q3:** Do improvements stack? | CITA+GRIT > CITA + (DPO+GRIT - DPO) | Yes (complementary benefits) |
| **Q4:** Is computational cost justified? | ΔAQI / (Time × VRAM) | Depends on application |

### Computational Budget

| Method | Training Time | Peak VRAM | Total GPU Hours |
|--------|---------------|-----------|-----------------|
| Baseline | 0 min | 0 GB | 0 |
| SFT | 39 min | 8.7 GB | 0.65 |
| SFT+GRIT | 137 min | 18.8 GB | 2.28 |
| DPO | 45 min | 10 GB | 0.75 |
| DPO+GRIT | 150 min | 20 GB | 2.50 |
| CITA | 50 min | 10 GB | 0.83 |
| CITA+GRIT | 155 min | 20 GB | 2.58 |
| **Total** | ~9.5 hours | - | ~9.5 |

**Estimated Timeline:** ~2 weeks (training + evaluation)

---

## Deliverables for Paper Submission

### 1. Code Repository
```
finetuning_evaluation/
├── comparative_study/
│   ├── 01_QLoRA_Baseline/
│   │   └── Llama3_SFT_alignmentDB.ipynb ✅
│   ├── 02_SFT_GRIT/
│   │   └── Llama3_alignmentDB.ipynb ✅
│   ├── 03b_De_Alignment_main/
│   │   ├── cita_trainer.py 🔨
│   │   ├── cita_grit_trainer.py 🔨
│   │   ├── Llama3_CITA_Vaibhaav.ipynb 🔨
│   │   └── Llama3_CITA_GRIT_Vaibhaav.ipynb 🔨
│   ├── 04_DPO/
│   │   ├── dpo_grit_trainer.py 🔨
│   │   ├── Llama3_DPO_Vaibhaav.ipynb 🔨
│   │   └── Llama3_DPO_GRIT_Vaibhaav.ipynb 🔨
│   └── 06_Evaluation/
│       ├── eval_all_methods_vaibhaav.ipynb 🔨
│       └── create_paper_figures.ipynb 🔨
└── plans/
    └── plan7_Ecliptica_Paper_Results.md ✅
```

### 2. Paper Figures
- **Figure 1:** AQI comparison bar chart (7 methods)
- **Figure 2:** Improvement over baseline (delta plot)
- **Figure 3:** GRIT effect analysis (before/after comparison)
- **Figure 4:** Method pairing comparison (SFT/DPO/CITA ± GRIT)
- **Figure 5:** Efficiency analysis (time/VRAM vs performance)
- **Figure 6:** 3D t-SNE clustering (best method vs baseline)

### 3. Results Table
```markdown
| Method | AQI ↑ | ΔAQI | Time (min) | VRAM (GB) | Efficiency |
|--------|-------|------|-----------|-----------|------------|
| Baseline | X.XX | - | 0 | 0 | - |
| SFT | X.XX | +X.XX | 39 | 8.7 | X.XX |
| SFT+GRIT | X.XX | +X.XX | 137 | 18.8 | X.XX |
| DPO | X.XX | +X.XX | 45 | 10.0 | X.XX |
| DPO+GRIT | X.XX | +X.XX | 150 | 20.0 | X.XX |
| CITA | X.XX | +X.XX | 50 | 10.0 | X.XX |
| **CITA+GRIT** | **X.XX** | **+X.XX** | 155 | 20.0 | **X.XX** |
```

### 4. Reproducibility Package
- All training notebooks with exact hyperparameters
- Evaluation scripts with fixed random seeds
- Dataset download and preprocessing code
- Model checkpoints (LoRA adapters) on HuggingFace
- TensorBoard logs for all runs

---

## Risk Mitigation

### Potential Issues & Solutions

| Risk | Impact | Mitigation |
|------|--------|------------|
| **CITA doesn't improve over DPO** | Undermines paper claim | Frame as "instruction conditioning study" |
| **GRIT causes instability** | Training divergence | Reduce learning rate, increase damping |
| **OOM errors on A10** | Can't train GRIT variants | Use gradient checkpointing, reduce batch size |
| **AQI scores too similar** | No clear winner | Use additional metrics (perplexity, human eval) |
| **Implementation bugs in CITA** | Incorrect results | Unit tests, compare DPO limit (λ_kl=0, β only) |

---

## Timeline (6 Weeks)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | Train DPO baseline | `lora_model_DPO_Baseline/` |
| **Week 2** | Implement DPO+GRIT | `lora_model_DPO_GRIT/` |
| **Week 3** | Implement CITA baseline | `lora_model_CITA_Baseline/` |
| **Week 4** | Implement CITA+GRIT ⭐ | `lora_model_CITA_GRIT/` |
| **Week 5** | Run AQI evaluation (all 7 methods) | AQI scores, embeddings, plots |
| **Week 6** | Create figures, write results section | Paper-ready figures, tables |

---

## Success Criteria

### Minimum Viable Result
- CITA improves over DPO by **≥2% AQI**
- GRIT improves at least one base method by **≥1% AQI**
- All methods surpass baseline by **≥10% AQI**

### Strong Result
- CITA+GRIT achieves **best AQI score** among all methods
- GRIT consistently improves all base methods (SFT/DPO/CITA)
- Clear trend: instruction conditioning > contrastive > supervised

### Exceptional Result
- CITA+GRIT improves over baseline by **≥30% AQI**
- GRIT effect is statistically significant (p < 0.05)
- Novel insights about instruction-conditioned optimization

---

## References

**Ecliptica Paper:**
- Location: `Legacy_code/2025_Ecliptica.pdf`
- Key equations: Pages 5-7 (CITA-KL loss)
- Dataset format: Pages 2-4 (Table 2)

**Existing Code:**
- GRIT: `comparative_study/02_SFT_GRIT/grit/`
- DPO baseline: `comparative_study/04_DPO/Zephyr_(7B)_DPO.ipynb`
- AQI evaluation: `Legacy_code/baseEvalAqi.ipynb`

**Datasets:**
- Training: Vaibhaav/alignment-instructions (50K samples)
- Evaluation: hasnat79/litmus (1K safe/unsafe prompts)

---

## Notes

- All training uses same hyperparameters (batch=2, grad_accum=4, steps=300)
- Same LoRA config (rank=16, alpha=16) for fair comparison
- Same random seed (3407) for reproducibility
- TensorBoard logging enabled for all runs
- GPU: NVIDIA A10 (22 GB VRAM)

**Contact:** Review this plan before starting implementation to ensure alignment with paper objectives.
