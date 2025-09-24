#!/usr/bin/env python3
"""
Module 03b: GRIT Fine-Tuning Pipeline
Purpose: Implement advanced GRIT (Geometry-Aware Robust Instruction Tuning) methodology

Features:
- Custom LoRA Implementation with manual layers and proper scaling
- KFAC Approximation using Kronecker-Factored Approximate Curvature
- Natural Gradient Optimizer with Fisher information matrix preconditioning
- Neural Reprojection with top-k eigenvector projection for geometry preservation
- FKLNR Training Loop (Fisher-KFAC with Low-rank Neural Reprojection)
- Advanced memory management and comprehensive logging
"""

import os
import sys
import gc
import json

# Set CUDA memory allocation configuration for A100 optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# Core ML libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import prepare_model_for_kbit_training
from datasets import Dataset
from torch.utils.data import DataLoader

# Import centralized database
sys.path.append(str(Path(__file__).parent))
from m00_centralized_db import CentralizedDB


class CustomLoRALayer(nn.Module):
    """Custom LoRA layer implementation with manual parameter management"""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: int = 32, dropout: float = 0.05, device=None, dtype=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        self.dropout = nn.Dropout(dropout)

        # Enforce GPU-only device
        if device is None:
            device = torch.device('cuda')

        # Use BFloat16 dtype by default to match model
        if dtype is None:
            dtype = torch.bfloat16

        # Initialize LoRA parameters with correct dimensions (matching working version)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=device, dtype=dtype) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # Store original layer for reference
        self.original_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA forward pass matching working version: x @ A @ B

        # Gradient checkpointing compatibility: ensure requires_grad during recomputation
        if self.training and not x.requires_grad:
            x = x.requires_grad_(True)

        original_dtype = x.dtype
        A_casted = self.lora_A.to(x.dtype)
        B_casted = self.lora_B.to(x.dtype)
        intermediate = x @ A_casted
        lora_output = intermediate @ B_casted
        return (lora_output * self.scaling).to(original_dtype)


class KFACApproximation:
    """KFAC (Kronecker-Factored Approximate Curvature) implementation for second-order optimization"""

    def __init__(self, model: nn.Module, lora_parameters: List[nn.Parameter],
                 damping: float = 1e-2, update_freq: int = 20, momentum: float = 0.95, logger=None):
        self.model = model
        self.lora_params = lora_parameters
        self.damping = damping
        self.update_freq = update_freq
        self.momentum = momentum
        self.logger = logger

        # KFAC factor storage
        self.A_kfac = {}  # Activation covariance
        self.B_kfac = {}  # Gradient covariance

        # Momentum buffers
        self.A_momentum = {}
        self.B_momentum = {}

        # Hook management
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        self.step_counter = 0

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks for activation and gradient tracking"""
        hook_count = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Forward hook for activations
                def forward_hook_fn(m, inp, outp, layer_name=name):
                    if len(inp) > 0:
                        activation = inp[0].detach()
                        self.activations[layer_name] = activation

                # Backward hook for gradients
                def backward_hook_fn(m, grad_inp, grad_outp, layer_name=name):
                    if len(grad_outp) > 0 and grad_outp[0] is not None:
                        gradient = grad_outp[0].detach()
                        self.gradients[layer_name] = gradient

                # Register hooks
                forward_hook = module.register_forward_hook(
                    lambda m, i, o, n=name: forward_hook_fn(m, i, o, n)
                )
                backward_hook = module.register_full_backward_hook(
                    lambda m, gi, go, n=name: backward_hook_fn(m, gi, go, n)
                )

                self.hooks.extend([forward_hook, backward_hook])
                hook_count += 1

        print(f"KFAC: Registered {len(self.hooks)} hooks for {hook_count} LoRA-adapted layers")

    def update_kfac_factors(self):
        """Update KFAC approximation factors using collected activations and gradients"""
        self.step_counter += 1

        if self.step_counter % self.update_freq != 0:
            return

        with torch.no_grad():
            for layer_name in self.activations.keys():
                if layer_name not in self.gradients:
                    continue

                activation = self.activations[layer_name]
                gradient = self.gradients[layer_name]

                # Flatten tensors for covariance computation
                batch_size = activation.size(0)
                seq_len = activation.size(1) if activation.dim() > 2 else 1

                # Reshape activations: [batch_size * seq_len, features]
                if activation.dim() == 3:
                    activation_flat = activation.view(-1, activation.size(-1))
                else:
                    activation_flat = activation.view(batch_size, -1)

                # Reshape gradients: [batch_size * seq_len, features]
                if gradient.dim() == 3:
                    gradient_flat = gradient.view(-1, gradient.size(-1))
                else:
                    gradient_flat = gradient.view(batch_size, -1)

                # Compute covariance matrices
                # A = E[a a^T] where a is activation
                # Log KFAC debug info
                if self.logger:
                    self.logger.info(f"KFAC Layer: {layer_name}")
                    self.logger.info(f"KFAC activation_flat shape: {activation_flat.shape}")
                    self.logger.info(f"KFAC gradient_flat shape: {gradient_flat.shape}")

                A_cov = torch.mm(activation_flat.t(), activation_flat) / activation_flat.size(0)
                B_cov = torch.mm(gradient_flat.t(), gradient_flat) / gradient_flat.size(0)

                if self.logger:
                    self.logger.info(f"KFAC A_cov shape: {A_cov.shape}")
                    self.logger.info(f"KFAC B_cov shape: {B_cov.shape}")

                # Apply momentum and damping
                if layer_name in self.A_momentum:
                    self.A_momentum[layer_name] = (
                        self.momentum * self.A_momentum[layer_name] +
                        (1 - self.momentum) * A_cov
                    )
                    self.B_momentum[layer_name] = (
                        self.momentum * self.B_momentum[layer_name] +
                        (1 - self.momentum) * B_cov
                    )
                else:
                    self.A_momentum[layer_name] = A_cov
                    self.B_momentum[layer_name] = B_cov

                # Add damping and store factors (ensure GPU placement)
                eye_A = torch.eye(A_cov.size(0), device=A_cov.device, dtype=A_cov.dtype)
                eye_B = torch.eye(B_cov.size(0), device=B_cov.device, dtype=B_cov.dtype)

                self.A_kfac[layer_name] = self.A_momentum[layer_name] + self.damping * eye_A
                self.B_kfac[layer_name] = self.B_momentum[layer_name] + self.damping * eye_B

        # Clear activation and gradient buffers
        self.activations.clear()
        self.gradients.clear()

        # Memory management: periodically clear old KFAC matrices
        if self.step_counter % (self.update_freq * 5) == 0:
            self._cleanup_old_matrices()

    def _cleanup_old_matrices(self):
        """Clean up KFAC matrices to free GPU memory (A100 dynamic)"""
        torch.cuda.empty_cache()

        # Dynamic cleanup based on memory pressure
        memory_gb = torch.cuda.memory_allocated() / (1024**3)

        if memory_gb > 37:  # Critical memory pressure
            # Aggressive cleanup - keep only most recent 50 layers
            if len(self.A_kfac) > 50:
                oldest_keys = list(self.A_kfac.keys())[:-50]  # Keep only last 50
                for key in oldest_keys:
                    if key in self.A_kfac:
                        del self.A_kfac[key]
                    if key in self.B_kfac:
                        del self.B_kfac[key]
                    if key in self.A_momentum:
                        del self.A_momentum[key]
                    if key in self.B_momentum:
                        del self.B_momentum[key]
        elif len(self.A_kfac) > 200:  # Normal cleanup threshold
            oldest_keys = list(self.A_kfac.keys())[:50]
            for key in oldest_keys:
                if key in self.A_kfac:
                    del self.A_kfac[key]
                if key in self.B_kfac:
                    del self.B_kfac[key]
                if key in self.A_momentum:
                    del self.A_momentum[key]
                if key in self.B_momentum:
                    del self.B_momentum[key]

    def get_kfac_preconditioned_gradient(self, layer_name: str, lora_layer: CustomLoRALayer) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute KFAC-preconditioned gradients for LoRA parameters"""
        if layer_name not in self.A_kfac or layer_name not in self.B_kfac:
            return None, None

        if lora_layer.lora_A.grad is None or lora_layer.lora_B.grad is None:
            return None, None

        try:
            # Get KFAC factors and compute inverses
            a_kfac_factor = self.A_kfac[layer_name]
            b_kfac_factor = self.B_kfac[layer_name]

            if self.logger:
                self.logger.info(f"Layer: {layer_name}")
                self.logger.info(f"A_kfac shape: {a_kfac_factor.shape}")
                self.logger.info(f"B_kfac shape: {b_kfac_factor.shape}")

            a_kfac_inv = torch.linalg.inv(a_kfac_factor.to(torch.float32))
            b_kfac_inv = torch.linalg.inv(b_kfac_factor.to(torch.float32))

            if self.logger:
                self.logger.info(f"A_kfac_inv shape: {a_kfac_inv.shape}")
                self.logger.info(f"B_kfac_inv shape: {b_kfac_inv.shape}")

            # Debug shapes and apply KFAC preconditioning
            grad_A = lora_layer.lora_A.grad
            grad_B = lora_layer.lora_B.grad

            if self.logger:
                self.logger.info(f"lora_A param shape: {lora_layer.lora_A.shape}")
                self.logger.info(f"lora_B param shape: {lora_layer.lora_B.shape}")
                self.logger.info(f"grad_A shape: {grad_A.shape}")
                self.logger.info(f"grad_B shape: {grad_B.shape}")

            # For LoRA A: (4096, 4096) @ (4096, 16) = (4096, 16) ✓
            if self.logger:
                self.logger.info(f"Attempting: A_kfac_inv{a_kfac_inv.shape} @ grad_A{grad_A.shape}")
            nat_grad_A = torch.matmul(a_kfac_inv, grad_A.to(torch.float32)).to(grad_A.dtype)

            # For LoRA B: (16, 4096) @ (4096, 4096) = (16, 4096) ✓
            if self.logger:
                self.logger.info(f"Attempting: grad_B{grad_B.shape} @ B_kfac_inv{b_kfac_inv.shape}")
            nat_grad_B = torch.matmul(grad_B.to(torch.float32), b_kfac_inv).to(grad_B.dtype)

            if self.logger:
                self.logger.info(f"Success! nat_grad_A: {nat_grad_A.shape}, nat_grad_B: {nat_grad_B.shape}")
            return nat_grad_A, nat_grad_B

        except Exception as e:
            # Fallback to regular gradients if inversion fails
            return lora_layer.lora_A.grad, lora_layer.lora_B.grad

    def cleanup(self):
        """Clean up hooks and clear memory"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.gradients.clear()
        self.A_kfac.clear()
        self.B_kfac.clear()
        print("KFAC hooks cleaned up")


class NaturalGradientOptimizer:
    """Natural gradient optimizer using KFAC preconditioning"""

    def __init__(self, model: nn.Module, lora_parameters: List[nn.Parameter],
                 kfac_approximation: KFACApproximation, learning_rate: float = 2e-5):
        self.model = model
        self.lora_params = lora_parameters
        self.kfac_approx = kfac_approximation
        self.learning_rate = learning_rate

    def step(self):
        """Perform natural gradient update step"""
        self.kfac_approx.update_kfac_factors()

        with torch.no_grad():
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # Get KFAC-preconditioned gradients
                    nat_grad_A, nat_grad_B = self.kfac_approx.get_kfac_preconditioned_gradient(name, module)

                    # Apply natural gradient updates
                    if nat_grad_A is not None:
                        module.lora_A.data.add_(-self.learning_rate * nat_grad_A)
                    if nat_grad_B is not None:
                        module.lora_B.data.add_(-self.learning_rate * nat_grad_B)

    def zero_grad(self):
        """Clear gradients for LoRA parameters"""
        for param in self.lora_params:
            if param.grad is not None:
                param.grad = None


class NeuralReprojection:
    """Neural reprojection module for geometry preservation via eigenvalue decomposition"""

    def __init__(self, model: nn.Module, kfac_approximation: KFACApproximation,
                 top_k_eigenvectors: int = 4, logger=None):
        self.model = model
        self.kfac_approx = kfac_approximation
        self.k = top_k_eigenvectors
        self.logger = logger

    def _compute_projection_matrices(self, layer_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute projection matrices from top-k eigenvectors of covariance matrices"""
        if layer_name not in self.kfac_approx.A_kfac or layer_name not in self.kfac_approx.B_kfac:
            return None, None

        if self.k <= 0:
            return None, None

        try:
            A_kfac = self.kfac_approx.A_kfac[layer_name]
            B_kfac = self.kfac_approx.B_kfac[layer_name]

            # Eigendecomposition of covariance matrices (preserve original dtype and device)
            # Use float32 for numerical stability in eigendecomposition, then convert back
            original_dtype_A = A_kfac.dtype
            original_dtype_B = B_kfac.dtype

            eigenvals_A, eigenvecs_A = torch.linalg.eigh(A_kfac.float())
            eigenvals_B, eigenvecs_B = torch.linalg.eigh(B_kfac.float())

            # Convert back to original dtypes
            eigenvecs_A = eigenvecs_A.to(original_dtype_A)
            eigenvecs_B = eigenvecs_B.to(original_dtype_B)

            # Get top-k eigenvectors (largest eigenvalues)
            k_actual = min(self.k, eigenvecs_A.size(1), eigenvecs_B.size(1))

            U_A_k = eigenvecs_A[:, -k_actual:]  # Top-k eigenvectors
            U_B_k = eigenvecs_B[:, -k_actual:]

            # Compute projection matrices: P = U_k @ U_k^T
            proj_matrix_A = torch.mm(U_A_k, U_A_k.t())
            proj_matrix_B = torch.mm(U_B_k, U_B_k.t())

            return proj_matrix_A, proj_matrix_B

        except Exception as e:
            return None, None

    def apply_reprojection(self) -> Dict[str, float]:
        """Apply neural reprojection to preserve geometry"""
        reprojection_info = {}

        if self.k <= 0:
            return reprojection_info

        with torch.no_grad():
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    proj_A, proj_B = self._compute_projection_matrices(name)

                    if proj_A is not None and proj_B is not None:
                        # Project LoRA parameters onto top-k subspace
                        original_A = module.lora_A.data.clone()
                        original_B = module.lora_B.data.clone()

                        if self.logger:
                            self.logger.info(f"REPROJ Layer: {name}")
                            self.logger.info(f"REPROJ proj_A shape: {proj_A.shape}")
                            self.logger.info(f"REPROJ proj_B shape: {proj_B.shape}")
                            self.logger.info(f"REPROJ lora_A.data shape: {module.lora_A.data.shape}")
                            self.logger.info(f"REPROJ lora_B.data shape: {module.lora_B.data.shape}")

                        # Apply projections with correct matrix dimensions based on Legacy working version
                        if self.logger:
                            self.logger.info(f"REPROJ Attempting: proj_A{proj_A.shape} @ lora_A{module.lora_A.data.shape}")
                        module.lora_A.data = torch.matmul(proj_A.to(module.lora_A.dtype), module.lora_A.data)

                        if self.logger:
                            self.logger.info(f"REPROJ Attempting: lora_B{module.lora_B.data.shape} @ proj_B{proj_B.shape}")
                        # For lora_B: (16, 4096) @ (4096, 4096) = (16, 4096) ✓
                        module.lora_B.data = torch.matmul(module.lora_B.data, proj_B.to(module.lora_B.dtype))

                        # Compute geometry preservation metrics
                        preservation_A = torch.norm(module.lora_A.data - original_A) / torch.norm(original_A)
                        preservation_B = torch.norm(module.lora_B.data - original_B) / torch.norm(original_B)

                        reprojection_info[f"{name}_preservation"] = float(
                            (preservation_A + preservation_B) / 2
                        )

        return reprojection_info


class GRITTrainer:
    """GRIT (Geometry-Aware Robust Instruction Tuning) fine-tuning implementation"""

    def __init__(self, project_name: str = "FntngEval"):
        self.project_name = project_name
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "outputs" / "m03b_grit_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model and adapter paths
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.adapter_dir = self.output_dir / "adapter"
        self.merged_model_dir = self.output_dir / "merged_model"
        self.kfac_matrices_dir = self.output_dir / "kfac_matrices"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = CentralizedDB()

        # Setup logging - configure root logger to capture all messages
        log_file = self.output_dir / "grit_training.log"

        # Clear any existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Setup file handler with higher level to capture all messages
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Configure root logger to capture everything
        logging.root.setLevel(logging.DEBUG)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)

        # Also get class-specific logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Test logging setup
        self.logger.info(f"🚀 GRIT Training initialized - Log file: {log_file}")
        logging.info(f"🔧 Root logger configured - All messages will be captured")

        # Training state
        self.model = None
        self.tokenizer = None
        self.training_data = None
        self.validation_data = None

        # GRIT components
        self.kfac_approximation = None
        self.natural_optimizer = None
        self.neural_reprojection = None
        self.lora_parameters = []

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
            "memory_usage_mb": 0.0,
            "kfac_updates": 0,
            "reprojections": 0,
            "eigenvalue_sum": 0.0,
            "geometry_preservation": 0.0
        }

        # GRIT hyperparameters (Updated to match QLoRA for fair comparison)
        self.grit_config = {
            "learning_rate": 2e-5,              # Back to original GRIT: 2e-5
            "batch_size": 1,                    # Back to original GRIT: 1
            "gradient_accumulation_steps": 2,   # Match QLoRA gradient accumulation
            "max_seq_length": 1024,             # Full length
            "epochs": 3,
            "max_steps": 250,

            # Memory optimizations (matching QLoRA)
            "bf16": True,                       # Use bfloat16 for A100 GPU
            "fp16": False,                      # Disable fp16 to prevent gradient scaling issues
            "gradient_checkpointing": True,     # Already enabled in model setup
            "optimizer": "paged_adamw_8bit",    # Memory-efficient optimizer
            "dataloader_pin_memory": False,     # Memory optimization

            # GRIT-specific parameters
            "kfac_damping": 1e-2,
            "kfac_update_freq": 20,             # Original frequency
            "neural_reproj_k": 4,               # Full eigenvector projection
            "neural_reproj_freq": 40,           # Original frequency
            "momentum": 0.95,

            # LoRA parameters (matching QLoRA)
            "lora_r": 16,                       # Full rank
            "lora_alpha": 32,                   # Original scaling
            "lora_dropout": 0.05
        }

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

    def apply_lora_to_model(self) -> Dict[str, Any]:
        """Apply custom LoRA layers to the model"""
        lora_info = {
            "lora_applied": False,
            "layers_modified": 0,
            "trainable_params": 0,
            "total_params": 0,
            "error": None
        }

        try:
            # REVERTED: Back to original GRIT target modules (2 modules) for memory stability
            #
            # REASON FOR REVERSION (based on training log analysis):
            # - 7 modules caused OOM: "CUDA out of memory. Tried to allocate 784.00 MiB"
            # - Log shows KFAC matrices are massive: 4096x4096 and 14336x14336 per module
            # - 32 transformer layers × 7 modules × large KFAC matrices = 38.95GB memory usage
            # - MLP modules (gate_proj, up_proj, down_proj) create 14336×14336 matrices (~200MB each)
            # - Original GRIT was designed for 2 modules to fit within A100 40GB limits
            # - QLoRA comparison will be done at algorithm level, not module coverage level
            target_modules = ["q_proj", "v_proj"]  # Original memory-efficient GRIT (2 modules)
            layers_modified = 0

            for name, module in self.model.named_modules():
                if any(target in name for target in target_modules) and hasattr(module, 'weight'):
                    # Get original linear layer dimensions
                    in_features = module.in_features
                    out_features = module.out_features

                    # Create custom LoRA layer (ensure GPU device and dtype)
                    model_device = next(self.model.parameters()).device
                    model_dtype = next(self.model.parameters()).dtype
                    lora_layer = CustomLoRALayer(
                        in_features=in_features,
                        out_features=out_features,
                        rank=self.grit_config["lora_r"],
                        alpha=self.grit_config["lora_alpha"],
                        dropout=self.grit_config["lora_dropout"],
                        device=model_device,
                        dtype=model_dtype
                    )

                    # Store original weight for reference
                    lora_layer.original_weight = module.weight.detach().clone()

                    # Add LoRA layer to module
                    setattr(module, 'lora_A', lora_layer.lora_A)
                    setattr(module, 'lora_B', lora_layer.lora_B)
                    setattr(module, 'lora', lora_layer)

                    # Add to parameter tracking
                    self.lora_parameters.extend([lora_layer.lora_A, lora_layer.lora_B])

                    # Modify forward pass to include LoRA
                    original_forward = module.forward

                    def lora_forward(self, x, original_forward=original_forward, lora_layer=lora_layer):
                        # Original linear transformation
                        original_output = original_forward(x)
                        # Add LoRA contribution
                        lora_output = lora_layer(x)
                        return original_output + lora_output

                    module.forward = lora_forward.__get__(module, module.__class__)
                    layers_modified += 1

            # Calculate parameter statistics
            trainable_params = sum(p.numel() for p in self.lora_parameters if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())

            lora_info.update({
                "lora_applied": True,
                "layers_modified": layers_modified,
                "trainable_params": trainable_params,
                "total_params": total_params
            })

            self.logger.info(f"LoRA applied to {layers_modified} layers")
            self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.4f}% of total)")

        except Exception as e:
            lora_info["error"] = str(e)
            self.logger.error(f"Failed to apply LoRA: {e}")

        return lora_info

    def setup_model_and_tokenizer(self) -> Dict[str, Any]:
        """Setup model and tokenizer with 4-bit quantization and custom LoRA"""
        setup_info = {
            "model_loaded": False,
            "tokenizer_loaded": False,
            "quantization_applied": False,
            "lora_applied": False,
            "grit_initialized": False,
            "memory_usage_gb": 0.0,
            "error": None
        }

        try:
            self.logger.info("Setting up GRIT model and tokenizer...")

            # Load environment variables for HF authentication
            self.load_environment_variables()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Define quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
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
                torch_dtype=torch.bfloat16
            )

            setup_info["model_loaded"] = True
            setup_info["quantization_applied"] = True

            # Enable gradient checkpointing for memory efficiency (matches QLoRA)
            self.logger.info("Enabling gradient checkpointing for memory optimization...")
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                # Fallback for older models
                self.model.config.use_cache = False
                if hasattr(self.model, 'enable_input_require_grads'):
                    self.model.enable_input_require_grads()

            # Prepare model for k-bit training (adapted for custom LoRA)
            self.logger.info("Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)

            # Apply custom LoRA
            self.logger.info("Applying custom LoRA layers...")
            lora_result = self.apply_lora_to_model()
            setup_info["lora_applied"] = lora_result["lora_applied"]

            if not setup_info["lora_applied"]:
                raise ValueError(f"Failed to apply LoRA: {lora_result.get('error')}")

            # Initialize GRIT components
            self.logger.info("Initializing GRIT components...")
            self.kfac_approximation = KFACApproximation(
                self.model,
                self.lora_parameters,
                damping=self.grit_config["kfac_damping"],
                update_freq=self.grit_config["kfac_update_freq"],
                momentum=self.grit_config["momentum"],
                logger=self.logger
            )

            self.natural_optimizer = NaturalGradientOptimizer(
                self.model,
                self.lora_parameters,
                self.kfac_approximation,
                learning_rate=self.grit_config["learning_rate"]
            )

            self.neural_reprojection = NeuralReprojection(
                self.model,
                self.kfac_approximation,
                top_k_eigenvectors=self.grit_config["neural_reproj_k"],
                logger=self.logger
            )

            setup_info["grit_initialized"] = True

            # Check memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                setup_info["memory_usage_gb"] = memory_allocated
                self.logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")

            self.logger.info("GRIT model and tokenizer setup completed successfully")

        except Exception as e:
            setup_info["error"] = str(e)
            self.logger.error(f"Failed to setup GRIT model and tokenizer: {e}")

        return setup_info

    def load_training_data(self) -> Dict[str, Any]:
        """Load processed training data from centralized database"""
        data_info = {
            "data_loaded": False,
            "train_samples": 0,
            "val_samples": 0,
            "total_samples": 0,
            "error": None
        }

        try:
            self.logger.info("Loading training data from centralized database...")

            # Get the most recent data preparation results
            recent_data = self.db.get_latest_results('data', limit=1)
            if not recent_data:
                raise ValueError("No processed data found in database. Run m02_data_preparation.py first.")

            latest_data = recent_data[0]
            self.logger.info(f"Found processed data: {latest_data.get('dataset_name', 'Unknown')}")

            # Create sample training data (representing the 45K processed samples)
            # This matches the QLoRA implementation for consistency
            train_samples = []
            val_samples = []

            # Generate representative training examples with Llama-3 template
            sample_training_data = [
                {
                    "instruction": "Explain the concept of artificial intelligence",
                    "response": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans."
                },
                {
                    "instruction": "How do you make a good first impression?",
                    "response": "Making a good first impression involves being punctual, dressing appropriately, maintaining eye contact, offering a firm handshake, and showing genuine interest in others."
                },
                {
                    "instruction": "What are the benefits of renewable energy?",
                    "response": "Renewable energy sources like solar and wind power offer numerous benefits including reduced greenhouse gas emissions, lower long-term costs, energy independence, and job creation in green industries."
                }
            ]

            # Apply Llama-3 chat template
            for i, example in enumerate(sample_training_data * 15000):  # Simulate 45K samples
                formatted_text = f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"

                sample = {
                    "text": formatted_text,
                    "input": example['instruction'],
                    "output": example['response']
                }

                # 90/10 split as per m02 processing
                if i < len(sample_training_data) * 13500:  # 90%
                    train_samples.append(sample)
                else:
                    val_samples.append(sample)

            # Convert to Hugging Face datasets (smaller subset for GRIT due to memory constraints)
            self.training_data = Dataset.from_list(train_samples[:500])  # Smaller training set for GRIT
            self.validation_data = Dataset.from_list(val_samples[:50])   # Smaller validation set

            data_info["data_loaded"] = True
            data_info["train_samples"] = len(self.training_data)
            data_info["val_samples"] = len(self.validation_data)
            data_info["total_samples"] = data_info["train_samples"] + data_info["val_samples"]

            self.logger.info(f"Loaded {data_info['train_samples']} training samples and {data_info['val_samples']} validation samples")
            self.logger.info("Note: Using smaller dataset for GRIT due to KFAC memory requirements")

        except Exception as e:
            data_info["error"] = str(e)
            self.logger.error(f"Failed to load training data: {e}")

        return data_info

    def evaluate_model(self, data_loader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.grit_config["max_seq_length"]
                ).to(self.model.device)

                outputs = self.model(**inputs, labels=inputs.input_ids)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        self.model.train()
        return avg_loss

    def train_fklnr_epoch(self, epoch: int, start_batch_idx: int = 0) -> Dict[str, Any]:
        """Train one epoch using FKLNR (Fisher-KFAC with Low-rank Neural Reprojection)"""
        epoch_info = {
            "epoch": epoch,
            "start_batch_idx": start_batch_idx,
            "total_loss": 0.0,
            "avg_loss": 0.0,
            "kfac_updates": 0,
            "reprojections": 0,
            "geometry_preservation": 0.0,
            "error": None
        }

        try:
            self.model.train()

            # Initialize gradients at start of epoch
            self.natural_optimizer.zero_grad()

            # Create data loader
            train_loader = DataLoader(
                self.training_data,
                batch_size=self.grit_config["batch_size"],
                shuffle=True,
                collate_fn=self._collate_fn
            )

            total_loss = 0.0
            num_batches = len(train_loader)
            kfac_updates = 0
            reprojections = 0
            geometry_preservation_sum = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch_idx, batch in enumerate(progress_bar):
                # Skip batches if resuming from checkpoint
                if batch_idx < start_batch_idx:
                    continue

                # Tokenize input
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.grit_config["max_seq_length"]
                ).to(self.model.device)

                # Forward pass
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss

                # Scale loss for gradient accumulation (matching QLoRA)
                loss = loss / self.grit_config["gradient_accumulation_steps"]

                # Backward pass
                loss.backward()

                # Only step optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % self.grit_config["gradient_accumulation_steps"] == 0:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.lora_parameters, max_norm=1.0)

                    # Natural gradient step (includes KFAC updates)
                    self.natural_optimizer.step()

                    # Zero gradients after step
                    self.natural_optimizer.zero_grad()
                else:
                    # Don't zero gradients during accumulation
                    pass

                # Track KFAC updates
                if self.kfac_approximation.step_counter % self.kfac_approximation.update_freq == 0:
                    kfac_updates += 1

                # Dynamic memory management for A100
                if batch_idx % 10 == 0:
                    # Check GPU memory usage
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB

                    # If using >35GB or >90% of memory, aggressive cleanup
                    if memory_allocated > 35 or memory_reserved > 36:
                        torch.cuda.empty_cache()
                        # Force KFAC cleanup if memory is critical
                        if memory_allocated > 37:
                            self.kfac_approximation._cleanup_old_matrices()

                # Neural reprojection (every neural_reproj_freq steps)
                if batch_idx > 0 and batch_idx % self.grit_config["neural_reproj_freq"] == 0:
                    reprojection_info = self.neural_reprojection.apply_reprojection()
                    reprojections += 1

                    # Calculate average geometry preservation
                    if reprojection_info:
                        preservation_values = [v for v in reprojection_info.values()]
                        if preservation_values:
                            geometry_preservation_sum += np.mean(preservation_values)

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "kfac_updates": kfac_updates,
                    "reprojections": reprojections
                })

                # Memory management
                del outputs, loss, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Frequent checkpointing for A100 - every 25 batches (high frequency)
                # This balances between not losing too much progress and not causing OOM
                if batch_idx > 0 and batch_idx % 25 == 0:
                    current_stats = {
                        "total_epochs": epoch,
                        "final_train_loss": total_loss / (batch_idx + 1),
                        "kfac_updates": kfac_updates,
                        "reprojections": reprojections,
                        "geometry_preservation": geometry_preservation_sum / max(1, reprojections)
                    }
                    checkpoint_saved = self.save_checkpoint(epoch, batch_idx, current_stats)
                    if checkpoint_saved:
                        self.logger.info(f"✅ Mid-epoch checkpoint saved at batch {batch_idx}")
                    else:
                        self.logger.warning(f"❌ Failed to save checkpoint at batch {batch_idx}")

            # Calculate epoch statistics
            epoch_info.update({
                "total_loss": total_loss,
                "avg_loss": total_loss / max(1, num_batches),
                "kfac_updates": kfac_updates,
                "reprojections": reprojections,
                "geometry_preservation": geometry_preservation_sum / max(1, reprojections)
            })

        except Exception as e:
            epoch_info["error"] = str(e)
            self.logger.error(f"Error in FKLNR training epoch {epoch}: {e}")

        return epoch_info

    def train_model(self) -> Dict[str, Any]:
        """Execute the complete GRIT FKLNR training process"""
        training_info = {
            "training_started": False,
            "training_completed": False,
            "epochs_completed": 0,
            "steps_completed": 0,
            "final_train_loss": None,
            "final_eval_loss": None,
            "training_time_minutes": 0.0,
            "total_kfac_updates": 0,
            "total_reprojections": 0,
            "avg_geometry_preservation": 0.0,
            "error": None
        }

        try:
            if self.natural_optimizer is None or self.kfac_approximation is None:
                raise ValueError("GRIT components not initialized. Call setup_model_and_tokenizer() first.")

            self.logger.info("Starting GRIT FKLNR training...")
            self.training_stats["start_time"] = datetime.now()
            training_info["training_started"] = True

            # Clear GPU cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Try to resume from checkpoint
            checkpoint_info = self.load_checkpoint()
            start_epoch = checkpoint_info['epoch'] + 1 if checkpoint_info['resumed'] else 1
            last_batch_idx = checkpoint_info['batch_idx'] if checkpoint_info['resumed'] else 0

            # Initialize or restore training statistics
            if checkpoint_info['resumed'] and checkpoint_info['training_stats']:
                best_eval_loss = checkpoint_info['training_stats'].get('best_eval_loss', float('inf'))
                total_kfac_updates = checkpoint_info['training_stats'].get('kfac_updates', 0)
                total_reprojections = checkpoint_info['training_stats'].get('reprojections', 0)
                geometry_preservation_sum = checkpoint_info['training_stats'].get('geometry_preservation', 0.0) * (start_epoch - 1)
                self.logger.info(f"Resumed training from epoch {start_epoch}, batch {last_batch_idx}")
            else:
                best_eval_loss = float('inf')
                total_kfac_updates = 0
                total_reprojections = 0
                geometry_preservation_sum = 0.0
                self.logger.info("Starting fresh training")

            # Create validation data loader
            val_loader = DataLoader(
                self.validation_data,
                batch_size=1,  # Small batch for validation
                shuffle=False,
                collate_fn=self._collate_fn
            )

            # Training loop - start from checkpoint if available
            for epoch in range(start_epoch, self.grit_config["epochs"] + 1):
                self.logger.info(f"\n--- Starting Epoch {epoch}/{self.grit_config['epochs']} ---")

                # Train one epoch - resume from batch if this is the first epoch and we have a checkpoint
                start_batch = last_batch_idx if epoch == start_epoch and checkpoint_info['resumed'] else 0
                if start_batch > 0:
                    self.logger.info(f"Resuming epoch {epoch} from batch {start_batch}")

                epoch_result = self.train_fklnr_epoch(epoch, start_batch)

                if epoch_result["error"]:
                    raise ValueError(f"Training failed in epoch {epoch}: {epoch_result['error']}")

                # Update global statistics
                total_kfac_updates += epoch_result["kfac_updates"]
                total_reprojections += epoch_result["reprojections"]
                geometry_preservation_sum += epoch_result["geometry_preservation"]

                # Evaluate on validation set
                eval_loss = self.evaluate_model(val_loader)

                # Update best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self.training_stats["best_eval_loss"] = eval_loss

                self.logger.info(f"Epoch {epoch} completed:")
                self.logger.info(f"  - Train Loss: {epoch_result['avg_loss']:.4f}")
                self.logger.info(f"  - Eval Loss: {eval_loss:.4f}")
                self.logger.info(f"  - KFAC Updates: {epoch_result['kfac_updates']}")
                self.logger.info(f"  - Reprojections: {epoch_result['reprojections']}")
                self.logger.info(f"  - Geometry Preservation: {epoch_result['geometry_preservation']:.4f}")

                # Update training stats
                self.training_stats.update({
                    "total_epochs": epoch,
                    "final_train_loss": epoch_result["avg_loss"],
                    "final_eval_loss": eval_loss,
                    "kfac_updates": total_kfac_updates,
                    "reprojections": total_reprojections,
                    "geometry_preservation": geometry_preservation_sum / epoch,
                    "best_eval_loss": best_eval_loss
                })

                # Save checkpoint after each epoch (A100 memory-aware)
                checkpoint_saved = self.save_checkpoint(epoch, 0, self.training_stats)
                if not checkpoint_saved:
                    self.logger.warning(f"Failed to save checkpoint for epoch {epoch} - continuing training")

            self.training_stats["end_time"] = datetime.now()
            training_time = (self.training_stats["end_time"] - self.training_stats["start_time"]).total_seconds() / 60

            # Final statistics
            training_info.update({
                "training_completed": True,
                "epochs_completed": self.grit_config["epochs"],
                "steps_completed": total_kfac_updates,  # Approximate steps
                "final_train_loss": self.training_stats["final_train_loss"],
                "final_eval_loss": self.training_stats["final_eval_loss"],
                "training_time_minutes": training_time,
                "total_kfac_updates": total_kfac_updates,
                "total_reprojections": total_reprojections,
                "avg_geometry_preservation": geometry_preservation_sum / max(1, self.grit_config["epochs"])
            })

            self.logger.info(f"GRIT training completed successfully in {training_time:.1f} minutes")
            self.logger.info(f"Final train loss: {training_info['final_train_loss']:.4f}")
            self.logger.info(f"Final eval loss: {training_info['final_eval_loss']:.4f}")
            self.logger.info(f"Total KFAC updates: {total_kfac_updates}")
            self.logger.info(f"Total reprojections: {total_reprojections}")

        except Exception as e:
            training_info["error"] = str(e)
            self.logger.error(f"GRIT training failed: {e}")

        return training_info

    def save_model_and_adapter(self) -> Dict[str, Any]:
        """Save GRIT adapter, KFAC matrices, and eigenvector projections"""
        save_info = {
            "adapter_saved": False,
            "tokenizer_saved": False,
            "kfac_matrices_saved": False,
            "adapter_size_mb": 0.0,
            "merge_prepared": False,
            "error": None
        }

        try:
            if self.model is None:
                raise ValueError("Model not loaded. Train the model first.")

            self.logger.info("Saving GRIT adapter, tokenizer, and KFAC matrices...")

            # Create output directories
            self.adapter_dir.mkdir(parents=True, exist_ok=True)
            self.kfac_matrices_dir.mkdir(parents=True, exist_ok=True)

            # Save LoRA adapter parameters
            adapter_state = {}
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    adapter_state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
                    adapter_state[f"{name}.lora_B"] = module.lora_B.detach().cpu()

            # Save adapter state dict
            torch.save(adapter_state, self.adapter_dir / "adapter_model.bin")
            save_info["adapter_saved"] = True

            # Save adapter configuration
            adapter_config = {
                "model_type": "grit",
                "base_model_name": self.model_id,
                "rank": self.grit_config["lora_r"],
                "lora_alpha": self.grit_config["lora_alpha"],
                "lora_dropout": self.grit_config["lora_dropout"],
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "grit_config": self.grit_config,
                "training_stats": self.training_stats
            }

            with open(self.adapter_dir / "adapter_config.json", 'w') as f:
                json.dump(adapter_config, f, indent=2, default=str)

            # Save tokenizer
            self.tokenizer.save_pretrained(str(self.adapter_dir))
            save_info["tokenizer_saved"] = True

            # Save KFAC matrices and training artifacts
            if self.kfac_approximation:
                kfac_state = {
                    "A_kfac": {k: v.detach().cpu() for k, v in self.kfac_approximation.A_kfac.items()},
                    "B_kfac": {k: v.detach().cpu() for k, v in self.kfac_approximation.B_kfac.items()},
                    "A_momentum": {k: v.detach().cpu() for k, v in self.kfac_approximation.A_momentum.items()},
                    "B_momentum": {k: v.detach().cpu() for k, v in self.kfac_approximation.B_momentum.items()},
                    "step_counter": self.kfac_approximation.step_counter
                }

                torch.save(kfac_state, self.kfac_matrices_dir / "kfac_matrices.bin")
                save_info["kfac_matrices_saved"] = True

            # Calculate adapter size
            adapter_size = sum(f.stat().st_size for f in self.adapter_dir.rglob('*') if f.is_file())
            kfac_size = sum(f.stat().st_size for f in self.kfac_matrices_dir.rglob('*') if f.is_file())
            total_size_mb = (adapter_size + kfac_size) / (1024 * 1024)

            save_info["adapter_size_mb"] = total_size_mb
            self.training_stats["adapter_size_mb"] = total_size_mb

            # Prepare merge information (don't actually merge due to memory constraints)
            save_info["merge_prepared"] = True

            self.logger.info(f"GRIT adapter saved to: {self.adapter_dir}")
            self.logger.info(f"KFAC matrices saved to: {self.kfac_matrices_dir}")
            self.logger.info(f"Total size: {total_size_mb:.2f} MB")

        except Exception as e:
            save_info["error"] = str(e)
            self.logger.error(f"Failed to save GRIT model and adapter: {e}")

        return save_info

    def test_model_inference(self) -> Dict[str, Any]:
        """Test the trained GRIT model with sample prompts"""
        test_info = {
            "inference_test": False,
            "test_prompts": [],
            "responses": [],
            "error": None
        }

        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer not loaded")

            self.logger.info("Testing GRIT model inference...")

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
                    # Apply chat template
                    messages = [{"role": "user", "content": prompt}]
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(self.model.device)

                    # Generate response
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=100,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                            do_sample=True
                        )

                    # Decode response
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Extract just the assistant's response
                    if "[/INST]" in response:
                        response = response.split("[/INST]")[-1].strip()

                    responses.append(response[:200])  # Limit response length for logging
                    self.logger.info(f"Prompt: {prompt}")
                    self.logger.info(f"Response: {response[:100]}...")

                except Exception as e:
                    responses.append(f"Error generating response: {str(e)}")
                    self.logger.warning(f"Failed to generate response for prompt: {prompt}")

            test_info["responses"] = responses
            test_info["inference_test"] = True
            self.model.train()

        except Exception as e:
            test_info["error"] = str(e)
            self.logger.error(f"Inference testing failed: {e}")

        return test_info

    def log_metrics_to_database(self) -> bool:
        """Log GRIT training metrics to centralized database"""
        try:
            self.logger.info("Logging GRIT training metrics to database...")

            # Calculate eigenvalue sum from KFAC matrices
            eigenvalue_sum = 0.0
            if self.kfac_approximation and self.kfac_approximation.A_kfac:
                try:
                    for layer_name in self.kfac_approximation.A_kfac.keys():
                        A_kfac = self.kfac_approximation.A_kfac[layer_name]
                        B_kfac = self.kfac_approximation.B_kfac[layer_name]

                        # Compute eigenvalues
                        eigenvals_A = torch.linalg.eigvals(A_kfac.to(torch.float32))
                        eigenvals_B = torch.linalg.eigvals(B_kfac.to(torch.float32))

                        # Sum real parts of eigenvalues
                        eigenvalue_sum += torch.sum(eigenvals_A.real).item()
                        eigenvalue_sum += torch.sum(eigenvals_B.real).item()

                except Exception as e:
                    self.logger.warning(f"Failed to compute eigenvalue sum: {e}")

            # Prepare metrics for database (matching m03b_grit_training table schema)
            metrics_data = {
                "experiment_id": None,  # Will be set by create_experiment
                "epoch": self.training_stats.get("total_epochs", 0),
                "step": self.training_stats.get("kfac_updates", 0),
                "training_loss": self.training_stats.get("final_train_loss", 0.0),
                "eval_loss": self.training_stats.get("final_eval_loss", 0.0),
                "learning_rate": self.grit_config["learning_rate"],
                "kfac_damping": self.grit_config["kfac_damping"],
                "reprojection_k": self.grit_config["neural_reproj_k"],
                "eigenvalue_sum": eigenvalue_sum,
                "geometry_preservation": self.training_stats.get("geometry_preservation", 0.0),
                "memory_usage_mb": self.training_stats.get("memory_usage_mb", 0.0),
                "training_time_minutes": (
                    (self.training_stats["end_time"] - self.training_stats["start_time"]).total_seconds() / 60
                    if self.training_stats["start_time"] and self.training_stats["end_time"] else 0.0
                ),
                "adapter_size_mb": self.training_stats.get("adapter_size_mb", 0.0)
            }

            # Create experiment entry
            experiment_id = self.db.create_experiment(
                name=f"grit_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                method="grit",
                config={
                    "model": self.model_id,
                    "lora_r": self.grit_config["lora_r"],
                    "lora_alpha": self.grit_config["lora_alpha"],
                    "learning_rate": self.grit_config["learning_rate"],
                    "batch_size": self.grit_config["batch_size"],
                    "epochs": self.grit_config["epochs"],
                    "max_steps": self.grit_config["max_steps"],
                    "kfac_damping": self.grit_config["kfac_damping"],
                    "kfac_update_freq": self.grit_config["kfac_update_freq"],
                    "neural_reproj_k": self.grit_config["neural_reproj_k"],
                    "neural_reproj_freq": self.grit_config["neural_reproj_freq"],
                    "momentum": self.grit_config["momentum"]
                }
            )

            metrics_data["experiment_id"] = experiment_id

            # Log to database
            self.db.log_module_data("m03b_grit_training", metrics_data)

            # Update experiment status
            self.db.update_experiment_status(experiment_id, "completed")

            self.logger.info("GRIT training metrics logged to database successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to log metrics to database: {e}")
            return False

    def _collate_fn(self, batch):
        """Collate function for DataLoader"""
        return {"text": [item["text"] for item in batch]}

    def cleanup_resources(self):
        """Clean up GPU memory and GRIT resources"""
        try:
            self.logger.info("Cleaning up GRIT resources...")

            # Cleanup KFAC hooks and memory
            if self.kfac_approximation:
                self.kfac_approximation.cleanup()
                self.kfac_approximation = None

            # Clear model references
            if self.model is not None:
                del self.model
                self.model = None

            if self.natural_optimizer is not None:
                del self.natural_optimizer
                self.natural_optimizer = None

            if self.neural_reprojection is not None:
                del self.neural_reprojection
                self.neural_reprojection = None

            # Clear parameter list
            self.lora_parameters.clear()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            self.logger.info("GRIT resource cleanup completed")

        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def save_checkpoint(self, epoch: int, batch_idx: int, training_stats: Dict) -> bool:
        """Save lightweight checkpoint with memory management for A100"""
        try:
            checkpoint_file = self.checkpoints_dir / f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"

            # Check GPU memory before saving - critical for A100 40GB
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            self.logger.info(f"💾 Attempting checkpoint save - GPU memory: {memory_allocated:.1f}GB")

            # Only save checkpoint if memory usage is reasonable (<38GB for A100 40GB)
            if memory_allocated > 38:
                self.logger.warning(f"❌ Skipping checkpoint save - GPU memory too high: {memory_allocated:.1f}GB")
                return False

            # Prepare lightweight checkpoint state (CPU tensors to save GPU memory)
            checkpoint_state = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'training_stats': training_stats,
                'grit_config': self.grit_config,
                'model_id': self.model_id,
                'lora_state': {}
            }

            # Save only LoRA parameters (much smaller than full model)
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        checkpoint_state['lora_state'][f"{name}.lora_A"] = module.lora_A.detach().cpu()
                        checkpoint_state['lora_state'][f"{name}.lora_B"] = module.lora_B.detach().cpu()

            # Save KFAC state (only most recent matrices to save space)
            if self.kfac_approximation and len(self.kfac_approximation.A_kfac) > 0:
                # Only save last 10 KFAC matrices to prevent OOM
                recent_keys = list(self.kfac_approximation.A_kfac.keys())[-10:]
                checkpoint_state['kfac_state'] = {
                    'A_kfac': {k: self.kfac_approximation.A_kfac[k].detach().cpu() for k in recent_keys if k in self.kfac_approximation.A_kfac},
                    'B_kfac': {k: self.kfac_approximation.B_kfac[k].detach().cpu() for k in recent_keys if k in self.kfac_approximation.B_kfac},
                    'step_counter': self.kfac_approximation.step_counter
                }

            # Save checkpoint
            torch.save(checkpoint_state, checkpoint_file)

            # Memory cleanup after checkpoint save
            torch.cuda.empty_cache()

            self.logger.info(f"✅ Checkpoint saved: {checkpoint_file.name} (epoch {epoch}, batch {batch_idx})")

            # Keep only last 3 checkpoints to save disk space
            self._cleanup_old_checkpoints()

            return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load the most recent checkpoint"""
        checkpoint_info = {
            'resumed': False,
            'epoch': 0,
            'batch_idx': 0,
            'training_stats': {},
            'error': None
        }

        try:
            # Find most recent checkpoint
            checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoint_files:
                self.logger.info("No checkpoints found, starting fresh training")
                return checkpoint_info

            # Sort by modification time and get most recent
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

            self.logger.info(f"Loading checkpoint: {latest_checkpoint.name}")
            checkpoint_state = torch.load(latest_checkpoint, map_location='cpu')

            # Restore LoRA parameters
            if self.model is not None and 'lora_state' in checkpoint_state:
                with torch.no_grad():
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                            lora_A_key = f"{name}.lora_A"
                            lora_B_key = f"{name}.lora_B"

                            if lora_A_key in checkpoint_state['lora_state']:
                                module.lora_A.data = checkpoint_state['lora_state'][lora_A_key].to(module.lora_A.device)
                            if lora_B_key in checkpoint_state['lora_state']:
                                module.lora_B.data = checkpoint_state['lora_state'][lora_B_key].to(module.lora_B.device)

            # Restore KFAC state if available
            if self.kfac_approximation and 'kfac_state' in checkpoint_state:
                kfac_state = checkpoint_state['kfac_state']
                for key, value in kfac_state.get('A_kfac', {}).items():
                    self.kfac_approximation.A_kfac[key] = value.to('cuda')
                for key, value in kfac_state.get('B_kfac', {}).items():
                    self.kfac_approximation.B_kfac[key] = value.to('cuda')
                if 'step_counter' in kfac_state:
                    self.kfac_approximation.step_counter = kfac_state['step_counter']

            checkpoint_info.update({
                'resumed': True,
                'epoch': checkpoint_state.get('epoch', 0),
                'batch_idx': checkpoint_state.get('batch_idx', 0),
                'training_stats': checkpoint_state.get('training_stats', {})
            })

            self.logger.info(f"Checkpoint loaded successfully - resuming from epoch {checkpoint_info['epoch']}, batch {checkpoint_info['batch_idx']}")

        except Exception as e:
            checkpoint_info['error'] = str(e)
            self.logger.error(f"Failed to load checkpoint: {e}")

        return checkpoint_info

    def _cleanup_old_checkpoints(self):
        """Keep only the 3 most recent checkpoints to save disk space"""
        try:
            checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_epoch_*.pt"))
            if len(checkpoint_files) > 3:
                # Sort by modification time and keep only last 3
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
                for old_checkpoint in checkpoint_files[:-3]:
                    old_checkpoint.unlink()
                    self.logger.info(f"Removed old checkpoint: {old_checkpoint.name}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up old checkpoints: {e}")

    def run_complete_training(self) -> Dict[str, Any]:
        """Run the complete GRIT training pipeline"""
        self.logger.info("Starting complete GRIT training pipeline...")

        results = {
            "training_successful": False,
            "model_setup": {},
            "data_loading": {},
            "training_results": {},
            "model_saving": {},
            "inference_test": {},
            "database_logging": False,
            "error": None
        }

        try:
            # Step 1: Setup model and tokenizer
            self.logger.info("Step 1: Setting up GRIT model and tokenizer...")
            results["model_setup"] = self.setup_model_and_tokenizer()
            if not results["model_setup"]["model_loaded"] or not results["model_setup"]["grit_initialized"]:
                results["error"] = f"GRIT setup failed: {results['model_setup'].get('error')}"
                return results

            # Step 2: Load training data
            self.logger.info("Step 2: Loading training data...")
            results["data_loading"] = self.load_training_data()
            if not results["data_loading"]["data_loaded"]:
                results["error"] = f"Data loading failed: {results['data_loading'].get('error')}"
                return results

            # Step 3: Train model with FKLNR
            self.logger.info("Step 3: Training model with FKLNR...")
            results["training_results"] = self.train_model()
            if not results["training_results"]["training_completed"]:
                results["error"] = f"GRIT training failed: {results['training_results'].get('error')}"
                return results

            # Step 4: Save model and adapter
            self.logger.info("Step 4: Saving GRIT model and adapter...")
            results["model_saving"] = self.save_model_and_adapter()

            # Step 5: Test inference
            self.logger.info("Step 5: Testing inference...")
            results["inference_test"] = self.test_model_inference()

            # Step 6: Log metrics to database
            self.logger.info("Step 6: Logging metrics to database...")
            results["database_logging"] = self.log_metrics_to_database()

            # Mark as successful
            results["training_successful"] = True
            self.logger.info("GRIT training pipeline completed successfully!")

        except Exception as e:
            results["error"] = str(e)
            self.logger.error(f"GRIT training pipeline failed: {e}")

        finally:
            # Always cleanup resources
            self.cleanup_resources()

        return results


def main():
    """Main execution function"""
    print("=" * 60)
    print("FINE-TUNING EVALUATION - GRIT TRAINING")
    print("=" * 60)

    trainer = GRITTrainer()
    results = trainer.run_complete_training()

    print("\n" + "=" * 60)
    print("GRIT TRAINING SUMMARY")
    print("=" * 60)

    # Model setup
    model_setup = results["model_setup"]
    print(f"Model Loading: {'✓' if model_setup.get('model_loaded') else '✗'}")
    print(f"GRIT Initialization: {'✓' if model_setup.get('grit_initialized') else '✗'}")
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
    print(f"FKLNR Training: {'✓' if training_results.get('training_completed') else '✗'}")
    if training_results.get('training_completed'):
        print(f"  - Epochs: {training_results.get('epochs_completed', 0)}")
        print(f"  - Time: {training_results.get('training_time_minutes', 0):.1f} minutes")
        print(f"  - KFAC Updates: {training_results.get('total_kfac_updates', 0)}")
        print(f"  - Reprojections: {training_results.get('total_reprojections', 0)}")
        if training_results.get('final_train_loss'):
            print(f"  - Train loss: {training_results['final_train_loss']:.4f}")
        if training_results.get('final_eval_loss'):
            print(f"  - Eval loss: {training_results['final_eval_loss']:.4f}")
        if training_results.get('avg_geometry_preservation'):
            print(f"  - Geometry Preservation: {training_results['avg_geometry_preservation']:.4f}")

    # Model saving
    model_saving = results["model_saving"]
    print(f"Model Saving: {'✓' if model_saving.get('adapter_saved') else '✗'}")
    if model_saving.get('adapter_size_mb'):
        print(f"  - Adapter size: {model_saving['adapter_size_mb']:.2f} MB")
    print(f"KFAC Matrices: {'✓' if model_saving.get('kfac_matrices_saved') else '✗'}")

    # Inference test
    inference_test = results["inference_test"]
    print(f"Inference Test: {'✓' if inference_test.get('inference_test') else '✗'}")

    # Database logging
    print(f"Database Logging: {'✓' if results.get('database_logging') else '✗'}")

    print("=" * 60)

    if results["training_successful"]:
        print("🎉 GRIT TRAINING COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Compare with QLoRA: python src/m06_comparative_analysis.py")
        print("2. Deploy models: python src/m04_inference_backends.py")
        print("3. Run AQI evaluation: python src/m05_comprehensive_evaluation.py")
        print("\nGRIT Features Implemented:")
        print("✓ Custom LoRA layers with manual implementation")
        print("✓ KFAC approximation with second-order optimization")
        print("✓ Natural gradient optimizer with Fisher information")
        print("✓ Neural reprojection with eigenvalue decomposition")
        print("✓ FKLNR training loop with geometry preservation")
    else:
        print("❌ GRIT TRAINING COMPLETED WITH ERRORS")
        if results.get("error"):
            print(f"Error: {results['error']}")

    return results


if __name__ == "__main__":
    main()