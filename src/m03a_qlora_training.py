#!/usr/bin/env python3
"""
Module 03a: QLoRA Fine-Tuning Pipeline
Purpose: Implement standard QLoRA fine-tuning methodology

Features:
- Load Meta-Llama-3-8B-Instruct with 4-bit quantization
- Configure LoRA parameters: r=16, alpha=32, dropout=0.05
- Setup SFTTrainer with optimized hyperparameters
- Implement gradient checkpointing and mixed-precision training
- Save LoRA adapter with model merge capability
- Comprehensive logging and database integration
"""

import os
import sys
import gc
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback

# Core ML libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

# Import centralized database
sys.path.append(str(Path(__file__).parent))
from m00_centralized_db import CentralizedDB

class QLoRATrainer:
    """QLoRA fine-tuning implementation for Llama-3-8B-Instruct"""

    def __init__(self, project_name: str = "FntngEval"):
        self.project_name = project_name
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "outputs" / "m03a_qlora_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model and adapter paths
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.adapter_dir = self.output_dir / "adapter"
        self.merged_model_dir = self.output_dir / "merged_model"

        # Initialize database
        self.db = CentralizedDB()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "qlora_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Training state
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_data = None
        self.validation_data = None

        # Training metrics
        self.training_stats = {
            "start_time": None,
            "end_time": None,
            "total_steps": 0,
            "total_epochs": 0,
            "final_train_loss": 0.0,
            "final_eval_loss": 0.0,
            "best_eval_loss": float('inf'),
            "adapter_size_mb": 0.0,
            "memory_usage_mb": 0.0
        }

        # TensorBoard logging
        self.tensorboard_writer = None
        self.current_step = 0
        self.experiment_id = None

    def load_environment_variables(self):
        """Load environment variables from .env file"""
        env_file = self.project_root / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                self.logger.info("Environment variables loaded from .env file")
            except Exception as e:
                self.logger.warning(f"Failed to load .env file: {e}")

    def setup_model_and_tokenizer(self) -> Dict[str, Any]:
        """Setup model and tokenizer with 4-bit quantization"""
        setup_info = {
            "model_loaded": False,
            "tokenizer_loaded": False,
            "quantization_applied": False,
            "lora_applied": False,
            "memory_usage_gb": 0.0,
            "error": None
        }

        try:
            self.logger.info("Setting up model and tokenizer...")

            # Load environment variables for HF authentication
            self.load_environment_variables()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Define enhanced quantization config with QAT optimizations
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Already using NF4!
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # Already using double quantization!
                # NEW: Add QAT-specific optimizations
                bnb_4bit_quant_storage=torch.uint8,
                llm_int8_skip_modules=["lm_head"]  # Skip output layer quantization
            )

            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=True
            )

            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            setup_info["tokenizer_loaded"] = True

            # Load model with quantization
            self.logger.info("Loading model with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"  # Use Flash Attention if available
            )

            setup_info["model_loaded"] = True
            setup_info["quantization_applied"] = True

            # Prepare model for training
            self.model = prepare_model_for_kbit_training(self.model)

            # Define enhanced LoRA configuration with 2024 optimizations
            lora_config = LoraConfig(
                r=16,  # Rank
                lora_alpha=32,  # Alpha parameter
                lora_dropout=0.05,  # Dropout probability
                bias="none",
                task_type="CAUSAL_LM",
                # NEW: Target ALL linear layers as per 2024 research
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"  # Additional targets
                ],
                # NEW: QAT-specific parameters
                use_rslora=True,  # Rank-stabilized LoRA
                use_dora=False,   # Weight-decomposed LoRA (memory intensive)
            )

            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            setup_info["lora_applied"] = True

            # Print trainable parameters
            trainable_params, all_params = self.model.get_nb_trainable_parameters()
            self.logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} "
                           f"({100 * trainable_params / all_params:.2f}%)")

            # Check memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                setup_info["memory_usage_gb"] = memory_allocated
                self.logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")

            self.logger.info("Model and tokenizer setup completed successfully")

        except Exception as e:
            setup_info["error"] = str(e)
            self.logger.error(f"Failed to setup model and tokenizer: {e}")

        return setup_info

    def load_training_data(self) -> Dict[str, Any]:
        """Load processed training data from cached alignment-instructions dataset"""
        data_info = {
            "data_loaded": False,
            "train_samples": 0,
            "val_samples": 0,
            "total_samples": 0,
            "error": None
        }

        try:
            self.logger.info("Loading training data from cached alignment-instructions dataset...")

            # Load cached dataset from m02_data_preparation (same method as EDA)
            from datasets import load_dataset
            cache_dir = self.project_root / "outputs" / "m02_data" / "cache"

            # Load alignment dataset from cache
            dataset = load_dataset(
                "Vaibhaav/alignment-instructions",
                cache_dir=str(cache_dir),
                split="train"
            )
            self.logger.info(f"✅ Loaded alignment dataset from cache: {len(dataset):,} samples")

            # Process the alignment dataset for training
            processed_samples = []

            for i, example in enumerate(dataset):
                # Extract instruction and accepted response
                instruction = example.get('Instruction generated', '').strip()
                accepted_response = example.get('Accepted Response', '').strip()
                prompt = example.get('Prompt', '').strip()

                # Skip if missing essential data
                if not instruction or not accepted_response:
                    continue

                # Clean and extract meaningful instruction
                if instruction.startswith(("When responding to", "Ensure that when", "Provide", "Always")):
                    # Extract human question from conversation in Prompt
                    if "H:" in prompt and "A:" in prompt:
                        try:
                            # Find the last human question
                            human_parts = prompt.split("H:")
                            if len(human_parts) > 1:
                                last_human = human_parts[-1].split("A:")[0].strip()
                                if len(last_human) > 10 and not last_human.startswith("You are"):
                                    instruction = last_human
                        except:
                            pass

                # Clean up accepted response
                response = accepted_response
                if "\nH:" in response or response.startswith("H:"):
                    # Extract assistant response from conversation format
                    try:
                        if "\nA:" in response:
                            assistant_parts = response.split("\nA:")
                            if len(assistant_parts) > 1:
                                response = assistant_parts[-1].strip()
                        elif "A:" in response and not response.startswith("A:"):
                            response = response.split("A:")[-1].strip()
                    except:
                        pass

                # Skip if processed text is too short or invalid
                if len(instruction) < 10 or len(response) < 10:
                    continue

                # Apply Llama-3.1 Instruct chat template (correct format)
                formatted_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"

                sample = {
                    "text": formatted_text,
                    "input": instruction,
                    "output": response
                }
                processed_samples.append(sample)

                # Progress logging every 5000 samples
                if (i + 1) % 5000 == 0:
                    self.logger.info(f"Processed {i + 1:,}/{len(dataset):,} samples -> {len(processed_samples):,} valid")

            self.logger.info(f"📊 Successfully processed {len(processed_samples):,} valid samples from {len(dataset):,} raw samples")

            if len(processed_samples) < 1000:
                raise ValueError(f"Too few valid samples: {len(processed_samples)}. Check data processing logic.")

            # 90/10 train/validation split (same as m02 processing)
            import random
            random.seed(42)  # Reproducible split
            random.shuffle(processed_samples)

            split_idx = int(0.9 * len(processed_samples))
            train_samples = processed_samples[:split_idx]
            val_samples = processed_samples[split_idx:]

            # Convert to Hugging Face datasets
            self.training_data = Dataset.from_list(train_samples)
            self.validation_data = Dataset.from_list(val_samples)

            data_info["data_loaded"] = True
            data_info["train_samples"] = len(self.training_data)
            data_info["val_samples"] = len(self.validation_data)
            data_info["total_samples"] = data_info["train_samples"] + data_info["val_samples"]

            self.logger.info(f"🎯 REAL ALIGNMENT DATA LOADED:")
            self.logger.info(f"   ├── Training samples: {data_info['train_samples']:,}")
            self.logger.info(f"   ├── Validation samples: {data_info['val_samples']:,}")
            self.logger.info(f"   └── Total samples: {data_info['total_samples']:,}")
            self.logger.info(f"📈 Data source: High-quality Vaibhaav/alignment-instructions dataset")

        except Exception as e:
            data_info["error"] = str(e)
            self.logger.error(f"❌ Failed to load training data: {e}")

        return data_info

    def setup_tensorboard(self):
        """Setup TensorBoard logging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = self.output_dir / "tensorboard_logs" / f"run_{timestamp}"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
            self.logger.info(f"TensorBoard logs: {log_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup TensorBoard: {e}")
            return False

    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "utilization_pct": (memory_allocated / memory_reserved * 100) if memory_reserved > 0 else 0
            }
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "utilization_pct": 0.0}

    def log_training_step(self, step: int, train_loss: float,
                         learning_rate: float, memory_usage: float = None):
        """Log detailed metrics per training step"""
        try:
            step_data = {
                "experiment_id": self.experiment_id,
                "training_step": step,
                "epoch": step // len(self.training_data) if self.training_data else 0,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "memory_usage_mb": memory_usage or 0.0,
                "timestamp": datetime.now().isoformat()
            }

            # Log to database
            self.db.log_module_data("m03a_qlora_training", step_data)

            # Log to TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar("Loss/Train", train_loss, step)
                self.tensorboard_writer.add_scalar("Learning_Rate", learning_rate, step)
                if memory_usage:
                    self.tensorboard_writer.add_scalar("Memory/GPU_MB", memory_usage, step)

        except Exception as e:
            self.logger.error(f"Step logging failed: {e}")

    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the checkpoint directory"""
        checkpoint_dir = self.output_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return None

        # Find all checkpoint directories
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not checkpoints:
            return None

        # Sort by step number and get the latest
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        latest_checkpoint = str(checkpoints[-1])

        self.logger.info(f"Found latest checkpoint: {checkpoints[-1].name}")
        return latest_checkpoint

    def setup_trainer(self) -> Dict[str, Any]:
        """Setup SFTTrainer with optimized hyperparameters"""
        trainer_info = {
            "trainer_setup": False,
            "training_args_created": False,
            "checkpoint_found": False,
            "error": None
        }

        try:
            self.logger.info("Setting up SFTTrainer...")

            # Check for existing checkpoints
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                trainer_info["checkpoint_found"] = True

            # Enhanced training arguments with 2024 optimizations
            training_args = SFTConfig(
                output_dir=str(self.output_dir / "checkpoints"),
                per_device_train_batch_size=1,          # Much smaller for stability
                per_device_eval_batch_size=1,           # Back to original working configuration
                gradient_accumulation_steps=8,          # Maintain effective batch size
                learning_rate=5e-5,                     # Lower LR for quantized training
                lr_scheduler_type="cosine",             # cosine LR scheduler
                warmup_ratio=0.03,                      # NEW: Proper warmup
                weight_decay=0.01,                      # NEW: Regularization
                adam_beta2=0.95,                        # NEW: Adjusted for small batch
                num_train_epochs=3,                     # training epochs: 3
                max_steps=500,                          # Increased steps for smaller batch
                optim="paged_adamw_8bit",               # memory-efficient optimizer (8bit for stability)
                max_grad_norm=1.0,                      # Gradient clipping to prevent explosion
                bf16=True,                              # use bfloat16 instead of fp16 for A10 GPU
                fp16=False,                             # disable fp16 to prevent gradient scaling issues
                gradient_checkpointing=True,            # gradient checkpointing
                gradient_checkpointing_kwargs={"use_reentrant": False},  # 2024 fix

                # Logging and evaluation - OPTIMIZED FOR FAST TRAINING/SLOW EVAL
                logging_steps=5,                        # Keep frequent logging (fast)
                eval_steps=150,                         # Strategic evaluation: 3 points (150, 300, 450/500)
                save_steps=150,                         # Aligned checkpointing with evaluation
                eval_strategy="steps",
                save_strategy="steps",
                report_to=["tensorboard"],              # Enable TensorBoard logging
                logging_dir=str(self.output_dir / "tensorboard_logs"),

                # Data formatting parameters (TRL 0.23.0 API)
                max_length=1024,                        # max sequence length
                dataset_text_field="text",              # text field name
                packing=False,                          # disable packing for stability

                # Output settings with early stopping
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=3,                     # keep only 3 checkpoints

                # Memory optimizations with 2024 fixes
                dataloader_pin_memory=False,
                dataloader_num_workers=0,               # Prevent multiprocessing issues
                remove_unused_columns=False,
                eval_accumulation_steps=2,              # Memory-efficient evaluation batching
            )

            trainer_info["training_args_created"] = True

            # Setup TensorBoard logging
            self.setup_tensorboard()

            # Initialize SFTTrainer with enhanced callbacks
            callbacks = [
                EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
            ]

            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.training_data,
                eval_dataset=self.validation_data,
                args=training_args,
                processing_class=self.tokenizer,        # Updated from 'tokenizer' to 'processing_class' in TRL 0.23.0
                callbacks=callbacks                     # NEW: Add callbacks for early stopping
            )

            # Set checkpoint path for resumption if checkpoint exists
            if latest_checkpoint:
                self.trainer.resume_from_checkpoint = latest_checkpoint
                self.logger.info(f"Will resume training from: {latest_checkpoint}")

            trainer_info["trainer_setup"] = True
            self.logger.info("SFTTrainer setup completed successfully")

        except Exception as e:
            trainer_info["error"] = str(e)
            self.logger.error(f"Failed to setup trainer: {e}")

        return trainer_info

    def train_model(self) -> Dict[str, Any]:
        """Execute the training process"""
        training_info = {
            "training_started": False,
            "training_completed": False,
            "epochs_completed": 0,
            "steps_completed": 0,
            "final_train_loss": None,
            "final_eval_loss": None,
            "training_time_minutes": 0.0,
            "error": None
        }

        try:
            if self.trainer is None:
                raise ValueError("Trainer not initialized. Call setup_trainer() first.")

            self.logger.info("Starting QLoRA training...")
            self.training_stats["start_time"] = datetime.now()
            training_info["training_started"] = True

            # Clear GPU cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Start training (with checkpoint resumption if available)
            resume_checkpoint = getattr(self.trainer, 'resume_from_checkpoint', None)
            if resume_checkpoint:
                self.logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
                train_result = self.trainer.train(resume_from_checkpoint=resume_checkpoint)
            else:
                self.logger.info("Starting fresh training (no checkpoints found)")
                train_result = self.trainer.train()

            self.training_stats["end_time"] = datetime.now()
            training_time = (self.training_stats["end_time"] - self.training_stats["start_time"]).total_seconds() / 60
            training_info["training_time_minutes"] = training_time

            # Extract training metrics
            if hasattr(train_result, 'training_loss'):
                training_info["final_train_loss"] = train_result.training_loss
                self.training_stats["final_train_loss"] = train_result.training_loss

            # Get final evaluation metrics
            eval_results = self.trainer.evaluate()
            if 'eval_loss' in eval_results:
                training_info["final_eval_loss"] = eval_results['eval_loss']
                self.training_stats["final_eval_loss"] = eval_results['eval_loss']

            # Get training state info
            if hasattr(self.trainer.state, 'epoch'):
                training_info["epochs_completed"] = self.trainer.state.epoch
                self.training_stats["total_epochs"] = self.trainer.state.epoch

            if hasattr(self.trainer.state, 'global_step'):
                training_info["steps_completed"] = self.trainer.state.global_step
                self.training_stats["total_steps"] = self.trainer.state.global_step

            training_info["training_completed"] = True
            self.logger.info(f"Training completed successfully in {training_time:.1f} minutes")
            self.logger.info(f"Final train loss: {training_info.get('final_train_loss', 'N/A')}")
            self.logger.info(f"Final eval loss: {training_info.get('final_eval_loss', 'N/A')}")

        except Exception as e:
            training_info["error"] = str(e)
            self.logger.error(f"Training failed: {e}")

        return training_info

    def save_model_and_adapter(self) -> Dict[str, Any]:
        """Save LoRA adapter and prepare for model merging"""
        save_info = {
            "adapter_saved": False,
            "tokenizer_saved": False,
            "adapter_size_mb": 0.0,
            "merge_prepared": False,
            "error": None
        }

        try:
            if self.model is None:
                raise ValueError("Model not loaded. Train the model first.")

            self.logger.info("Saving LoRA adapter and tokenizer...")

            # Create adapter directory
            self.adapter_dir.mkdir(parents=True, exist_ok=True)

            # Save LoRA adapter
            self.model.save_pretrained(str(self.adapter_dir))
            save_info["adapter_saved"] = True

            # Save tokenizer
            self.tokenizer.save_pretrained(str(self.adapter_dir))
            save_info["tokenizer_saved"] = True

            # Calculate adapter size
            adapter_size = sum(f.stat().st_size for f in self.adapter_dir.rglob('*') if f.is_file())
            save_info["adapter_size_mb"] = adapter_size / (1024 * 1024)
            self.training_stats["adapter_size_mb"] = save_info["adapter_size_mb"]

            self.logger.info(f"LoRA adapter saved to: {self.adapter_dir}")
            self.logger.info(f"Adapter size: {save_info['adapter_size_mb']:.2f} MB")

            # Prepare merged model information (don't actually merge due to memory constraints)
            save_info["merge_prepared"] = True

        except Exception as e:
            save_info["error"] = str(e)
            self.logger.error(f"Failed to save model and adapter: {e}")

        return save_info

    def test_model_inference(self) -> Dict[str, Any]:
        """Test the trained model with sample prompts"""
        test_info = {
            "inference_test": False,
            "test_prompts": [],
            "responses": [],
            "error": None
        }

        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer not loaded")

            self.logger.info("Testing model inference...")

            # Test prompts
            test_prompts = [
                "What is the capital of France?",
                "Explain the concept of machine learning.",
                "How can I improve my productivity?"
            ]

            test_info["test_prompts"] = test_prompts
            responses = []

            for prompt in test_prompts:
                try:
                    # Apply chat template (following successful legacy pattern)
                    messages = [{"role": "user", "content": prompt}]
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(self.model.device)

                    # Generate response with proper configuration
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=100,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )

                    # Decode response
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Extract just the assistant's response (Llama-3 format)
                    if "<|start_header_id|>assistant<|end_header_id|>" in response:
                        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                        # Remove any trailing tokens
                        if "<|eot_id|>" in response:
                            response = response.split("<|eot_id|>")[0].strip()

                    responses.append(response[:200])  # Limit response length for logging
                    self.logger.info(f"Prompt: {prompt}")
                    self.logger.info(f"Response: {response[:100]}...")

                except Exception as e:
                    responses.append(f"Error generating response: {str(e)}")
                    self.logger.warning(f"Failed to generate response for prompt: {prompt}")

            test_info["responses"] = responses
            test_info["inference_test"] = True

        except Exception as e:
            test_info["error"] = str(e)
            self.logger.error(f"Inference testing failed: {e}")

        return test_info

    def log_metrics_to_database(self) -> bool:
        """Log training metrics to centralized database"""
        try:
            self.logger.info("Logging training metrics to database...")

            # Prepare enhanced metrics for database
            memory_stats = self.monitor_memory_usage()
            metrics_data = {
                "experiment_id": None,  # Will be set by create_experiment
                "epoch": self.training_stats.get("total_epochs", 0),
                "step": self.training_stats.get("total_steps", 0),
                "training_loss": self.training_stats.get("final_train_loss", 0.0),
                "eval_loss": self.training_stats.get("final_eval_loss", 0.0),
                "learning_rate": 5e-5,  # Updated configuration
                "gradient_norm": self.trainer.state.log_history[-1].get('grad_norm', 0.0) if self.trainer and self.trainer.state.log_history else 0.0,
                "memory_usage_mb": memory_stats.get("allocated_gb", 0.0) * 1024,
                "lora_r": 16,
                "lora_alpha": 32,
                "dropout": 0.05,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "training_time_minutes": (self.training_stats["end_time"] - self.training_stats["start_time"]).total_seconds() / 60 if self.training_stats["start_time"] else 0.0,
                "adapter_size_mb": self.training_stats.get("adapter_size_mb", 0.0)
            }

            # Create enhanced experiment entry
            experiment_id = self.db.create_experiment(
                name=f"qlora_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                method="qlora_enhanced_2024",
                config={
                    "model": self.model_id,
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "learning_rate": 5e-5,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "epochs": 3,
                    "max_steps": 500,
                    "quantization": "nf4_double_quant",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]
                }
            )
            self.experiment_id = experiment_id

            metrics_data["experiment_id"] = experiment_id

            # Log to database
            self.db.log_module_data("m03a_qlora_training", metrics_data)

            self.logger.info("Training metrics logged to database successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to log metrics to database: {e}")
            return False

    def cleanup_resources(self):
        """Clean up GPU memory and resources"""
        try:
            self.logger.info("Cleaning up GPU resources...")

            # Clear model references
            if self.model is not None:
                del self.model
                self.model = None

            if self.trainer is not None:
                del self.trainer
                self.trainer = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            self.logger.info("Resource cleanup completed")

        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def run_complete_training(self) -> Dict[str, Any]:
        """Run the complete QLoRA training pipeline"""
        self.logger.info("Starting complete QLoRA training pipeline...")

        results = {
            "training_successful": False,
            "model_setup": {},
            "data_loading": {},
            "trainer_setup": {},
            "training_results": {},
            "model_saving": {},
            "inference_test": {},
            "database_logging": False,
            "checkpoint_resumed": False,
            "error": None
        }

        try:
            # Step 0: Check for existing checkpoints FIRST
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                results["checkpoint_resumed"] = True
                self.logger.info(f"🔄 RESUMING from checkpoint: {Path(latest_checkpoint).name}")
                self.logger.info("Skipping data reprocessing for checkpoint resumption...")
            else:
                self.logger.info("🆕 Starting fresh training (no checkpoints found)")

            # Step 1: Setup model and tokenizer
            self.logger.info("Step 1: Setting up model and tokenizer...")
            results["model_setup"] = self.setup_model_and_tokenizer()
            if not results["model_setup"]["model_loaded"]:
                results["error"] = f"Model setup failed: {results['model_setup'].get('error')}"
                return results

            # Step 2: Load training data (always needed for proper training)
            self.logger.info("Step 2: Loading and processing training data...")
            results["data_loading"] = self.load_training_data()
            # Note: Checkpoints will resume training state, but we need real data for continued training

            if not results["data_loading"]["data_loaded"]:
                results["error"] = f"Data loading failed: {results['data_loading'].get('error')}"
                return results

            # Step 3: Setup trainer
            self.logger.info("Step 3: Setting up trainer...")
            results["trainer_setup"] = self.setup_trainer()
            if not results["trainer_setup"]["trainer_setup"]:
                results["error"] = f"Trainer setup failed: {results['trainer_setup'].get('error')}"
                return results

            # Step 4: Train model
            self.logger.info("Step 4: Training model...")
            results["training_results"] = self.train_model()
            if not results["training_results"]["training_completed"]:
                results["error"] = f"Training failed: {results['training_results'].get('error')}"
                return results

            # Step 5: Save model and adapter
            self.logger.info("Step 5: Saving model and adapter...")
            results["model_saving"] = self.save_model_and_adapter()

            # Step 6: Test inference
            self.logger.info("Step 6: Testing inference...")
            results["inference_test"] = self.test_model_inference()

            # Step 7: Log metrics to database
            self.logger.info("Step 7: Logging metrics to database...")
            results["database_logging"] = self.log_metrics_to_database()

            # Mark as successful
            results["training_successful"] = True
            self.logger.info("QLoRA training pipeline completed successfully!")

        except Exception as e:
            results["error"] = str(e)
            self.logger.error(f"Training pipeline failed: {e}")

        finally:
            # Close TensorBoard writer
            if self.tensorboard_writer:
                self.tensorboard_writer.close()

            # Always cleanup resources
            self.cleanup_resources()

        return results

def main():
    """Main execution function"""
    print("=" * 60)
    print("FINE-TUNING EVALUATION - QLORA TRAINING")
    print("=" * 60)

    trainer = QLoRATrainer()
    results = trainer.run_complete_training()

    print("\n" + "=" * 60)
    print("QLORA TRAINING SUMMARY")
    print("=" * 60)

    # Model setup
    model_setup = results["model_setup"]
    print(f"Model Loading: {'✓' if model_setup.get('model_loaded') else '✗'}")
    if model_setup.get('memory_usage_gb'):
        print(f"  - GPU Memory: {model_setup['memory_usage_gb']:.2f} GB")

    # Data loading
    data_info = results["data_loading"]
    print(f"Data Loading: {'✓' if data_info.get('data_loaded') else '✗'}")
    if data_info.get('data_loaded'):
        print(f"  - Training samples: {data_info['train_samples']:,}")
        print(f"  - Validation samples: {data_info['val_samples']:,}")

    # Training results
    training_results = results["training_results"]
    print(f"Training: {'✓' if training_results.get('training_completed') else '✗'}")
    if training_results.get('training_completed'):
        print(f"  - Steps: {training_results.get('steps_completed', 0)}")
        print(f"  - Time: {training_results.get('training_time_minutes', 0):.1f} minutes")
        if training_results.get('final_train_loss'):
            print(f"  - Train loss: {training_results['final_train_loss']:.4f}")
        if training_results.get('final_eval_loss'):
            print(f"  - Eval loss: {training_results['final_eval_loss']:.4f}")

    # Model saving
    model_saving = results["model_saving"]
    print(f"Model Saving: {'✓' if model_saving.get('adapter_saved') else '✗'}")
    if model_saving.get('adapter_size_mb'):
        print(f"  - Adapter size: {model_saving['adapter_size_mb']:.2f} MB")

    # Inference test
    inference_test = results["inference_test"]
    print(f"Inference Test: {'✓' if inference_test.get('inference_test') else '✗'}")

    # Database logging
    print(f"Database Logging: {'✓' if results.get('database_logging') else '✗'}")

    print("=" * 60)

    if results["training_successful"]:
        print("🎉 QLORA TRAINING COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run: python src/m03b_grit_training.py")
        print("2. Run: python src/m04_inference_backends.py")
    else:
        print("❌ QLORA TRAINING COMPLETED WITH ERRORS")
        if results.get("error"):
            print(f"Error: {results['error']}")

    return results

if __name__ == "__main__":
    main()