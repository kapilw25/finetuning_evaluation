import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from grit.config import GRITConfig
from grit.manager import GRITManager
from grit.trainer import GritTrainer, GritCallback


def main():
	config = GRITConfig()
	# Stability knobs to mirror Dolly setup
	config.kfac_min_samples = 256
	config.kfac_update_freq = 150
	config.kfac_damping = 0.005
	config.reprojection_warmup_steps = 500
	config.rank_adaptation_start_step = 500
	config.reprojection_freq = 300
	config.use_two_sided_reprojection = True
	
	tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "left"
	model = AutoModelForCausalLM.from_pretrained(
		config.model_id,
		device_map="auto",
		dtype=torch.bfloat16.
	)
	
	lora_config = LoraConfig(
		r=config.lora_rank,
		lora_alpha=config.lora_alpha,
		target_modules=config.lora_target_modules,
		lora_dropout=config.lora_dropout,
		bias="none",
		task_type="CAUSAL_LM",
		inference_mode=False,
		use_rslora=True,
	)
	model = get_peft_model(model, lora_config)
	model.enable_input_require_grads()
	model.config.use_cache = False

	full_dataset = load_dataset(config.dataset_name, split="train")
	dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
	train_dataset = dataset["train"]
	val_dataset = dataset["test"]

	def tokenize(example):
		instr = example['instruction'].strip()
		inp = example.get('input', "").strip()
		out = example['output'].strip()
		prompt = f"### Instruction:\n{instr}\n\n### Response:\n{out}" if not inp else f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
		tok = tokenizer(prompt, truncation=True, padding="max_length", max_length=config.max_length, return_tensors=None)
		tok["labels"] = tok["input_ids"][:]
		return tok

	tokenized_train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
	tokenized_val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

	train_len = len(tokenized_train_dataset)
	eff_batch = config.batch_size * config.gradient_accumulation_steps
	total_steps = math.ceil(train_len / eff_batch) * config.num_epochs
	print(f"Total training steps: {total_steps}")

	training_args = Seq2SeqTrainingArguments(
		output_dir="./results",
		per_device_train_batch_size=config.batch_size,
		gradient_accumulation_steps=config.gradient_accumulation_steps,
		num_train_epochs=config.num_epochs,
		learning_rate=config.learning_rate,
		bf16=config.precision == "bf16",
		# Trust-region clipping for NG
		max_grad_norm=1.5,
		gradient_checkpointing=True,
		gradient_checkpointing_kwargs={"use_reentrant": False},
		dataloader_num_workers=config.num_workers,
		dataloader_pin_memory=config.pin_memory,
		remove_unused_columns=False,
		report_to="none",
		save_strategy="epoch",
		save_total_limit=2,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Initializing GRITManager on device:", device)
	grit_manager = GRITManager(model, config, device)
	trainer = GritTrainer(
		grit_manager=grit_manager,
		model=model,
		args=training_args,
		train_dataset=tokenized_train_dataset,
		eval_dataset=tokenized_val_dataset,
		tokenizer=tokenizer,
		callbacks=[GritCallback(grit_manager)],
	)
	trainer.train()


if __name__ == "__main__":
	main()


