# Plan 5: AQI Evaluation Pipeline

## Objective
Create streamlined training pipeline without visualization overhead, integrate TensorBoard logging, and implement AQI evaluation using LITMUS dataset.

---

## Step 1: Create Copy of Notebook

```bash
# In terminal
cp comparative_study/03_AQI_EVAL/scripts/lora_with_visualisation.ipynb \
   comparative_study/03_AQI_EVAL/scripts/lora_training_only.ipynb
```

---

## Step 2: Replace t-SNE Visualization with Lightweight Cluster Metrics

**In new notebook `lora_training_only.ipynb`, replace Cell 12 with:**

```python
import os
import torch
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from transformers import Trainer
from torch.utils.data import DataLoader
import gc


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.step_count += 1
        outputs = model(**inputs)
        loss = outputs.loss

        # Compute cluster metrics every N steps (without visualization)
        if self.step_count % 50 == 0:
            print(f"\n>>> Step {self.step_count}: Computing cluster metrics...")
            metrics = self.compute_cluster_metrics(model)

            # Log to TensorBoard
            if self.args.report_to == "tensorboard":
                self.log(metrics)

            print(f"[Step {self.step_count}] Loss: {loss.item():.4f}")
            if "DBS" in metrics:
                print(f"  DBS: {metrics['DBS']:.4f} (lower is better)")
                print(f"  Silhouette: {metrics['Silhouette']:.4f} (higher is better)")

        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset=None):
        """Lightweight dataloader for metric computation"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        return DataLoader(
            eval_dataset,
            batch_size=2,  # Tiny batch size
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,  # No multiprocessing to save memory
            pin_memory=False,
        )

    def compute_cluster_metrics(self, model):
        """Compute DBS and Silhouette WITHOUT t-SNE visualization"""
        torch.cuda.empty_cache()
        gc.collect()

        model.eval()
        embeddings, labels = [], []
        sample_count = 0
        MAX_SAMPLES = 50  # Small subset

        dataloader = self.get_eval_dataloader()

        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= MAX_SAMPLES:
                    break

                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Extract last token embedding
                last_hidden_state = outputs.hidden_states[-1]
                embedding = last_hidden_state[:, -1, :].detach().cpu().numpy()

                safety_labels = batch["safety_labels"].cpu().numpy()
                embeddings.append(embedding)
                labels.extend(safety_labels)
                sample_count += len(safety_labels)

                # Clear memory immediately
                del input_ids, attention_mask, outputs, last_hidden_state

        torch.cuda.empty_cache()

        X = np.vstack(embeddings)
        labels = np.array(labels)

        metrics = {}
        if len(np.unique(labels)) >= 2:
            metrics["DBS"] = davies_bouldin_score(X, labels)
            metrics["Silhouette"] = silhouette_score(X, labels)

        # Cleanup
        del X, embeddings, labels
        torch.cuda.empty_cache()
        gc.collect()

        model.train()
        return metrics
```

---

## Step 3: Add TensorBoard + Train 200-300 Steps

### Add new cell AFTER Cell 11 (custom_collator):

```python
import os
from datetime import datetime

# TensorBoard setup - matching QLoRA_Baseline format
tensorboard_base_dir = "/home/ubuntu/DiskUsEast1/finetuning_evaluation/tensorboard_logs"
os.makedirs(tensorboard_base_dir, exist_ok=True)

# Create run-specific directory with timestamp
run_name = "SafeLoRA_Training"  # Descriptive name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_run_dir = os.path.join(tensorboard_base_dir, f"{run_name}_{timestamp}")

print(f"📊 TensorBoard logs will be saved to: {tensorboard_run_dir}")
```

### Replace Cell 16 (training_args) with:

```python
from trl import SFTTrainer, SFTConfig

# Training arguments with TensorBoard
training_args = SFTConfig(
    output_dir="./llama3_sft",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch = 32
    warmup_steps=10,
    max_steps=250,  # 200-300 range
    learning_rate=2e-4,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    save_steps=250,  # Save at end
    save_total_limit=1,
    bf16=True,
    fp16=False,

    # TensorBoard configuration
    report_to="tensorboard",
    logging_dir=tensorboard_run_dir,
    logging_first_step=True,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Keep for cluster metrics computation
    args=training_args,
    data_collator=custom_collator,
    dataset_text_field="text",  # May need to adjust based on dataset format
    max_seq_length=max_seq_length,
    packing=False,
)
```

### Modify Cell 17 (Train):

```python
# Show GPU stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train
trainer_stats = trainer.train()

# Show stats after training
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
```

### Add new cell to save model:

```python
# Save LoRA adapters
save_dir = "lora_model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Model saved to {save_dir}/")

# Also save to GGUF for Ollama (optional)
if True:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q8_0")
    print(f"✅ GGUF model saved to model/")
```

### TensorBoard Metrics:

**Scalars visible in TensorBoard:**
- `train/loss` (every 10 steps)
- `train/DBS` (every 50 steps)
- `train/Silhouette` (every 50 steps)
- `train/learning_rate`
- `train/grad_norm`

**Example output:**
```
Step 50:  Loss=2.45, DBS=13.45, Silhouette=-0.018
Step 100: Loss=2.12, DBS=11.92, Silhouette=-0.016
Step 150: Loss=1.89, DBS=10.31, Silhouette=0.042
Step 200: Loss=1.67, DBS=8.75, Silhouette=0.128
```

---

## Step 4: AQI Evaluation Script

### Create new file: `comparative_study/03_AQI_EVAL/scripts/run_aqi_eval.py`

```python
#!/usr/bin/env python3
"""
AQI Evaluation Script for SafeLoRA finetuned model
Based on comparative_study/03_AQI_EVAL/README.md
"""

import os
import sys
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run AQI evaluation on SafeLoRA model')
    parser.add_argument('--model', type=str, default='lora_model',
                       help='Path to LoRA model directory')
    parser.add_argument('--dataset', type=str, default='hasnat79/ACCD',
                       help='Dataset name from HuggingFace')
    parser.add_argument('--gamma', type=float, default=0.3,
                       help='Gamma parameter for AQI calculation')
    parser.add_argument('--samples', type=int, default=939,
                       help='Number of samples to evaluate')
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"comparative_study/03_AQI_EVAL/results/aqi_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 Running AQI Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Samples: {args.samples}")
    print(f"  Output: {output_dir}")

    # Run AQI evaluation script
    cmd = f"""python src/aqi/aqi_dealign_chi_sil.py \
      --model "{args.model}" \
      --dataset "{args.dataset}" \
      --output-dir "{output_dir}" \
      --gamma {args.gamma} \
      --samples {args.samples}"""

    print(f"\n📊 Executing: {cmd}\n")
    os.system(cmd)

    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
```

### Make it executable:

```bash
chmod +x comparative_study/03_AQI_EVAL/scripts/run_aqi_eval.py
```

### Run evaluation:

```bash
python comparative_study/03_AQI_EVAL/scripts/run_aqi_eval.py \
  --model "lora_model" \
  --dataset "hasnat79/ACCD" \
  --gamma 0.3 \
  --samples 939
```

---

## Step 5: View TensorBoard

### Add cell to notebook:

```python
# View TensorBoard
print("🎯 To view TensorBoard:")
print(f"Run in terminal: tensorboard --logdir={tensorboard_base_dir} --port=6006")
print(f"Then open: http://localhost:6006")
print("")
print("📊 All training runs:")
!ls -lh {tensorboard_base_dir}
```

---

## Summary of Changes

1. ✅ **Removed**: t-SNE 3D visualization plots
2. ✅ **Added**: Lightweight cluster metrics (DBS, Silhouette) logged to TensorBoard
3. ✅ **Added**: TensorBoard logging (same format as QLoRA_Baseline)
4. ✅ **Changed**: Training to 250 steps with SFTTrainer
5. ✅ **Added**: Model saving to `lora_model/`
6. ✅ **Created**: AQI evaluation script

**Memory savings:** ~3-4 GB (no t-SNE 3D rendering, lightweight metrics on 50 samples)
