#!/usr/bin/env python3
"""
Module 05: MVP Inference Backends
Purpose: Deploy baseline and QLoRA models for rapid comparison

MVP Focus:
- Load baseline Meta-Llama-3-8B-Instruct
- Load QLoRA-tuned model with adapter
- Simple inference API for evaluation pipeline
- Memory-efficient model switching
"""

import os
import sys
import torch
import logging
import time
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# Import centralized database and environment setup
sys.path.append(str(Path(__file__).parent))
from m00_centralized_db import CentralizedDB
from m01_environment_setup import EnvironmentSetup

# ML imports
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline
    )
    from peft import PeftModel, PeftConfig
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    sys.exit(1)

class MVPInferenceBackend:
    """MVP Inference backend for baseline vs QLoRA comparison"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "outputs" / "m05_inference_backends"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db = CentralizedDB()

        # Setup enhanced logging with force reconfiguration
        self._setup_logging()

        self.logger.info("=" * 60)
        self.logger.info("MVP INFERENCE BACKENDS - INITIALIZATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log file: {self.output_dir / 'inference_backends.log'}")

        # Model configurations
        self.base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.qlora_adapter_path = self.project_root / "outputs" / "m03a_qlora_training" / "adapter"

        # Model containers
        self.models = {
            "baseline": None,
            "qlora": None
        }
        self.tokenizer = None
        self.current_model = None

        # Performance tracking
        self.performance_metrics = {}

        # Setup HuggingFace authentication using centralized method
        self.env_setup = EnvironmentSetup()
        hf_auth = self.env_setup.setup_huggingface_auth()
        if not hf_auth.get("login_successful"):
            error_msg = hf_auth.get("error", "Unknown authentication error")
            self.logger.error(f"HuggingFace authentication failed: {error_msg}")
            raise RuntimeError(f"HuggingFace authentication required. {error_msg}")

        self.logger.info("HuggingFace authentication successful - ready to access gated models")

    def monitor_gpu_memory(self, context: str = "") -> Dict[str, float]:
        """Monitor and log GPU memory usage with warnings"""
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "cached_gb": 0, "total_gb": 0, "usage_percent": 0}

        try:
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            cached_gb = torch.cuda.memory_reserved() / 1024**3
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage_percent = (allocated_gb / total_gb) * 100

            memory_info = {
                "allocated_gb": allocated_gb,
                "cached_gb": cached_gb,
                "total_gb": total_gb,
                "usage_percent": usage_percent
            }

            # Log with context
            context_str = f" [{context}]" if context else ""
            self.logger.debug(f"GPU Memory{context_str}: {allocated_gb:.2f}GB allocated, {cached_gb:.2f}GB cached ({usage_percent:.1f}%)")

            # Warning thresholds
            if usage_percent > 90:
                self.logger.error(f"🚨 CRITICAL GPU memory usage{context_str}: {usage_percent:.1f}% - Risk of OOM!")
            elif usage_percent > 85:
                self.logger.warning(f"⚠️ High GPU memory usage{context_str}: {usage_percent:.1f}% - Approaching limits!")
            elif usage_percent > 75:
                self.logger.info(f"📊 GPU memory usage{context_str}: {usage_percent:.1f}% - Good utilization")

            return memory_info

        except Exception as e:
            self.logger.error(f"Failed to monitor GPU memory: {e}")
            return {"allocated_gb": 0, "cached_gb": 0, "total_gb": 0, "usage_percent": 0}

    def _setup_logging(self):
        """Setup comprehensive logging with both file and console output"""
        # Clear any existing handlers to avoid conflicts
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # File handler with detailed logging
        log_file = self.output_dir / "inference_backends.log"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)

        # Console handler with simpler format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)

        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Remove any existing handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False

        # Test logging
        self.logger.info("Logging system initialized successfully")
        self.logger.debug(f"Log file path: {log_file}")

    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup enhanced 4-bit quantization with 2024 optimizations"""
        self.logger.info("Using enhanced NF4 quantization with QAT optimizations for optimal training compatibility")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Double quantization for memory efficiency
            bnb_4bit_quant_type="nf4",      # NormalFloat 4-bit (information-theoretically optimal)
            bnb_4bit_compute_dtype=torch.bfloat16,  # Enhanced compute precision for stability
            bnb_4bit_quant_storage=torch.uint8,     # QAT-specific optimization
            llm_int8_skip_modules=["lm_head"]       # Skip output layer quantization for better alignment
        )

    def load_baseline_model(self) -> bool:
        """Load baseline Meta-Llama-3-8B-Instruct model with aggressive 4-bit quantization"""
        try:
            self.logger.info("Loading baseline model with aggressive 4-bit quantization...")
            start_time = time.time()

            # Aggressive GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete

            # Monitor memory before loading
            self.monitor_gpu_memory("before baseline loading")

            # Setup aggressive 4-bit quantization for minimal memory usage
            quantization_config = self.setup_quantization_config()

            # Load tokenizer first
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("Tokenizer loaded successfully")

            # Enhanced model loading with minimal memory usage
            self.logger.info("Loading model with minimal memory footprint...")
            self.models["baseline"] = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Minimize CPU memory during loading
                max_memory={0: "20GB"},  # OPTIMIZED: Increased from 18GB for better GPU utilization
                offload_folder=None,  # No CPU offloading per CLAUDE.md
                attn_implementation="flash_attention_2"  # FlashAttention2 for memory efficiency
            )

            load_time = time.time() - start_time

            # Monitor GPU memory after loading
            memory_info = self.monitor_gpu_memory("after baseline loading")
            gpu_memory_used = memory_info["allocated_gb"] * 1024  # Convert to MB for compatibility

            # Test generation with memory monitoring
            test_prompt = "Hello, how are you?"
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.models["baseline"].device)

            with torch.no_grad():
                outputs = self.models["baseline"].generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False  # Disable KV cache to save memory
                )

            test_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            self.logger.info(f"✅ Baseline model loaded successfully in {load_time:.2f}s")
            self.logger.info(f"Test generation: {test_response}")

            # Log performance metrics
            self.performance_metrics["baseline"] = {
                "load_time_seconds": load_time,
                "model_size_mb": self._get_model_memory_usage(self.models["baseline"]),
                "gpu_memory_mb": gpu_memory_used if torch.cuda.is_available() else 0,
                "quantization": "4bit_aggressive",
                "memory_limit_gb": 18,
                "test_successful": True,
                "loaded_at": datetime.now().isoformat()
            }

            self.logger.info("Baseline model metrics recorded successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load baseline model: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.performance_metrics["baseline"] = {
                "load_time_seconds": 0,
                "model_size_mb": 0,
                "error": str(e),
                "test_successful": False,
                "failed_at": datetime.now().isoformat()
            }
            return False

    def load_qlora_model(self) -> bool:
        """Load QLoRA-tuned model with adapter and aggressive 4-bit quantization"""
        try:
            self.logger.info("Loading QLoRA model with aggressive 4-bit quantization...")
            start_time = time.time()

            # Aggressive GPU memory cleanup before QLoRA loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Monitor memory before QLoRA loading
            self.monitor_gpu_memory("before QLoRA loading")

            # Verify adapter exists
            if not self.qlora_adapter_path.exists():
                raise FileNotFoundError(f"QLoRA adapter not found at {self.qlora_adapter_path}")

            # Load adapter config to verify compatibility
            config = PeftConfig.from_pretrained(self.qlora_adapter_path)
            self.logger.info(f"Adapter config: {config.base_model_name_or_path}")

            # Strategy: Load fresh base model with aggressive 4-bit quantization
            self.logger.info("Loading fresh base model for QLoRA with minimal memory footprint...")
            quantization_config = self.setup_quantization_config()

            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,  # Enhanced precision for training compatibility
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "20GB"},  # OPTIMIZED: Same increased limit as baseline
                offload_folder=None,
                use_cache=False,  # Disable KV cache for training compatibility
                attn_implementation="flash_attention_2"
            )

            # Load tokenizer if not already loaded
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.base_model_name_or_path,
                    trust_remote_code=True,
                    padding_side="left"
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load enhanced LoRA adapter with validation
            self.logger.info(f"Loading enhanced LoRA adapter with config: r={config.r}, alpha={config.lora_alpha}")

            self.models["qlora"] = PeftModel.from_pretrained(
                base_model,
                self.qlora_adapter_path,
                torch_dtype=torch.bfloat16  # Enhanced precision consistency
            )

            # Log trainable parameters info
            if hasattr(self.models["qlora"], 'get_nb_trainable_parameters'):
                trainable_params, all_params = self.models["qlora"].get_nb_trainable_parameters()
                self.logger.info(f"Enhanced LoRA adapter loaded: {trainable_params:,} trainable / {all_params:,} total parameters ({100 * trainable_params / all_params:.2f}%)")

            load_time = time.time() - start_time

            # Monitor memory after QLoRA loading
            self.monitor_gpu_memory("after QLoRA loading")

            # Test generation with QLoRA model
            test_prompt = "Hello, how are you?"
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.models["qlora"].device)

            with torch.no_grad():
                outputs = self.models["qlora"].generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            test_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            self.logger.info(f"✅ QLoRA model loaded successfully in {load_time:.2f}s")
            self.logger.info(f"Test generation: {test_response}")

            # Log performance metrics
            self.performance_metrics["qlora"] = {
                "load_time_seconds": load_time,
                "model_size_mb": self._get_model_memory_usage(self.models["qlora"]),
                "adapter_path": str(self.qlora_adapter_path),
                "test_successful": True,
                "loaded_at": datetime.now().isoformat()
            }

            self.logger.info("QLoRA model metrics recorded successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load QLoRA model: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.performance_metrics["qlora"] = {
                "load_time_seconds": 0,
                "model_size_mb": 0,
                "error": str(e),
                "test_successful": False,
                "failed_at": datetime.now().isoformat()
            }
            return False

    def _get_model_memory_usage(self, model) -> float:
        """Get approximate memory usage of model in MB"""
        try:
            if hasattr(model, 'get_memory_footprint'):
                return model.get_memory_footprint() / 1024 / 1024
            else:
                # Estimate based on parameters
                total_params = sum(p.numel() for p in model.parameters())
                # Assume float16 (2 bytes per parameter) + overhead
                return (total_params * 2) / 1024 / 1024 * 1.2
        except:
            return 0.0

    def switch_model(self, model_type: str) -> bool:
        """Switch active model for inference"""
        if model_type not in ["baseline", "qlora"]:
            self.logger.error(f"Invalid model type: {model_type}")
            return False

        if self.models[model_type] is None:
            self.logger.error(f"{model_type} model not loaded")
            return False

        self.current_model = model_type
        self.logger.info(f"Switched to {model_type} model")
        return True

    def generate_response(self, prompt: str, max_new_tokens: int = 150,
                         temperature: float = 0.7, do_sample: bool = True) -> Dict[str, Any]:
        """Generate response using current active model"""
        if self.current_model is None:
            return {"error": "No active model selected"}

        if self.models[self.current_model] is None:
            return {"error": f"{self.current_model} model not loaded"}

        try:
            start_time = time.time()

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.models[self.current_model].device)
            input_length = inputs.input_ids.shape[1]

            # Generate response
            with torch.no_grad():
                outputs = self.models[self.current_model].generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode response (only new tokens)
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            generation_time = time.time() - start_time

            result = {
                "model": self.current_model,
                "prompt": prompt,
                "response": response,
                "generation_time_ms": generation_time * 1000,
                "tokens_generated": len(response_tokens),
                "tokens_per_second": len(response_tokens) / generation_time if generation_time > 0 else 0,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            self.logger.error(f"Generation failed with {self.current_model}: {e}")
            return {
                "model": self.current_model,
                "prompt": prompt,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def batch_generate(self, prompts: List[str], model_type: str,
                      max_new_tokens: int = 150) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts"""
        if not self.switch_model(model_type):
            return [{"error": f"Failed to switch to {model_type} model"}] * len(prompts)

        results = []
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Generating {i+1}/{len(prompts)} with {model_type}")
            result = self.generate_response(prompt, max_new_tokens=max_new_tokens)
            results.append(result)

            # Small delay to prevent memory issues
            if i % 10 == 9:
                torch.cuda.empty_cache()
                time.sleep(0.1)

        return results

    def cleanup_models(self):
        """Clean up models to free memory with error handling"""
        try:
            for model_type in ["baseline", "qlora"]:
                if self.models[model_type] is not None:
                    del self.models[model_type]
                    self.models[model_type] = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            gc.collect()

            # Safe CUDA cache cleanup with error handling
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    self.logger.info("CUDA cache cleared successfully")
                except Exception as e:
                    self.logger.warning(f"CUDA cache cleanup failed (non-fatal): {e}")

            self.logger.info("Models cleaned up")

        except Exception as e:
            self.logger.error(f"Error during model cleanup: {e}")
            # Continue execution - cleanup errors shouldn't crash the program

    def log_to_database(self):
        """Log performance metrics to database with enhanced error handling"""
        try:
            self.logger.info("Starting database logging...")

            if not self.performance_metrics:
                self.logger.warning("No performance metrics to log")
                return

            logged_count = 0
            for model_type, metrics in self.performance_metrics.items():
                self.logger.debug(f"Logging metrics for {model_type}: {metrics}")

                db_data = {
                    "backend_type": "direct_pytorch",
                    "model_variant": model_type,
                    "rps": None,  # Will be calculated during evaluation
                    "latency_ms": metrics.get("generation_time_ms"),
                    "memory_usage_mb": metrics.get("model_size_mb", 0),
                    "concurrent_requests": 1,  # Single request for MVP
                    "deployment_status": "success" if metrics.get("test_successful") else "failed",
                    "endpoint_url": f"local_{model_type}"
                }

                self.logger.debug(f"Database data for {model_type}: {db_data}")
                self.db.log_module_data("m04_inference_backends", db_data)
                logged_count += 1
                self.logger.info(f"✅ Logged {model_type} metrics to database")

            self.logger.info(f"✅ Successfully logged {logged_count} model metrics to database")

        except Exception as e:
            self.logger.error(f"❌ Failed to log metrics to database: {e}")
            self.logger.error(f"Performance metrics available: {list(self.performance_metrics.keys())}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")

    def run_mvp_setup(self) -> Dict[str, Any]:
        """Run MVP setup for baseline vs QLoRA comparison"""
        self.logger.info("Starting MVP inference backend setup...")

        results = {
            "setup_start": datetime.now().isoformat(),
            "baseline_loaded": False,
            "qlora_loaded": False,
            "setup_success": False,
            "performance_metrics": {},
            "next_steps": []
        }

        try:
            # Load baseline model with aggressive 4-bit quantization for ECC stability
            self.logger.info("Step 1: Loading baseline model with aggressive 4-bit quantization...")
            results["baseline_loaded"] = self.load_baseline_model()

            # Load QLoRA model with same optimization
            self.logger.info("Step 2: Loading QLoRA model with aggressive 4-bit quantization...")
            results["qlora_loaded"] = self.load_qlora_model()

            # Log performance metrics to database
            self.logger.info("Step 3: Logging performance metrics to database...")
            self.log_to_database()

            # Update results
            results["performance_metrics"] = self.performance_metrics
            results["setup_success"] = results["baseline_loaded"] and results["qlora_loaded"]
            results["setup_end"] = datetime.now().isoformat()

            if results["setup_success"]:
                results["next_steps"] = [
                    "Run python src/m06_comprehensive_evaluation.py",
                    "Models ready for AQI evaluation",
                    "Both baseline and QLoRA models loaded successfully"
                ]
                self.logger.info("✅ MVP inference backend setup completed successfully!")
            else:
                results["next_steps"] = [
                    "Check model loading errors",
                    "Verify CUDA/GPU availability",
                    "Check memory requirements"
                ]
                self.logger.error("❌ MVP inference backend setup failed")

            # Save results
            results_path = self.output_dir / "mvp_setup_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            return results

        except Exception as e:
            self.logger.error(f"MVP setup failed: {e}")
            results["error"] = str(e)
            results["setup_success"] = False
            return results

def main():
    """Main execution function"""
    print("=" * 60)
    print("MVP INFERENCE BACKENDS - BASELINE VS QLORA")
    print("=" * 60)

    backend = MVPInferenceBackend()
    results = backend.run_mvp_setup()

    print("\n" + "=" * 60)
    print("MVP SETUP SUMMARY")
    print("=" * 60)

    print(f"Baseline Model: {'✅' if results['baseline_loaded'] else '❌'}")
    if results['baseline_loaded']:
        metrics = results['performance_metrics'].get('baseline', {})
        print(f"  - Load time: {metrics.get('load_time_seconds', 0):.2f}s")
        print(f"  - Memory usage: {metrics.get('model_size_mb', 0):.1f}MB")

    print(f"QLoRA Model: {'✅' if results['qlora_loaded'] else '❌'}")
    if results['qlora_loaded']:
        metrics = results['performance_metrics'].get('qlora', {})
        print(f"  - Load time: {metrics.get('load_time_seconds', 0):.2f}s")
        print(f"  - Memory usage: {metrics.get('model_size_mb', 0):.1f}MB")
        print(f"  - Adapter: outputs/m03a_qlora_training/adapter/")

    print(f"Overall Status: {'✅ SUCCESS' if results['setup_success'] else '❌ FAILED'}")

    if results['setup_success']:
        print("\n🎉 MVP INFERENCE BACKEND READY!")
        print("\nNext steps:")
        for step in results['next_steps']:
            print(f"  • {step}")
    else:
        print("\n⚠️ SETUP ISSUES DETECTED")
        if 'error' in results:
            print(f"Error: {results['error']}")
        print("Please check logs and resolve issues.")

    # Cleanup
    backend.cleanup_models()

    return results

if __name__ == "__main__":
    main()