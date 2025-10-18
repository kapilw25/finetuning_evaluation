"""
CITA Trainer - Contrastive Instruction-Tuned Alignment
Based on Ecliptica paper (Legacy_code/2025_Ecliptica.pdf pages 5-7)
Implements Unified Loss: L_SFT + λ₁·L_DPO + λ₂·L_KL
"""

import torch
import torch.nn.functional as F
from trl import DPOTrainer
from typing import Dict, Optional, Tuple
from copy import deepcopy


class CITATrainer(DPOTrainer):
    """
    CITA Trainer - Contrastive Instruction-Tuned Alignment
    Inherits from DPOTrainer (Ecliptica PDF - Unified Loss)

    Implements: L_unified = L_SFT + λ₁·L_DPO + λ₂·L_KL

    Data Format Expected (DPO format):
    - dataset[i]['prompt']: Conversation messages (system + user)
    - dataset[i]['chosen']: Chosen (safe/helpful) response
    - dataset[i]['rejected']: Rejected (unsafe/harmful) response

    Loss Components:
    - L_SFT: Supervised fine-tuning on chosen responses
    - L_DPO: Direct preference optimization (contrastive)
    - L_KL: KL divergence to reference model

    Usage:
        from cita_trainer_2 import CITATrainer

        trainer = CITATrainer(
            model=model,
            tokenizer=tokenizer,
            args=dpo_config,
            train_dataset=dataset,
            lambda_sft=1.0,
            lambda_dpo=1.0,
            lambda_kl=0.01,
            beta=0.1,
        )

        trainer.train()
    """

    def __init__(
        self,
        model,
        tokenizer,
        args,
        train_dataset,
        lambda_sft: float = 1.0,
        lambda_dpo: float = 1.0,
        lambda_kl: float = 0.01,
        beta: float = 0.1,
        **kwargs
    ):
        """
        Initialize CITA Trainer with Unified Loss

        Args:
            model: Policy model to fine-tune
            tokenizer: Tokenizer
            args: DPOConfig
            train_dataset: Dataset in DPO format (prompt, chosen, rejected)
            lambda_sft: Weight for SFT loss (default 1.0)
            lambda_dpo: Weight for DPO loss (default 1.0)
            lambda_kl: Weight for KL regularization (default 0.01)
            beta: Contrastive temperature (default 0.1)
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            beta=beta,
            **kwargs
        )

        if lambda_kl <= 0:
            raise ValueError(f"lambda_kl must be > 0 (got {lambda_kl})")

        self.lambda_sft = lambda_sft
        self.lambda_dpo = lambda_dpo
        self.lambda_kl = lambda_kl
        # Note: DPOTrainer already creates ref_model, no need to override

    def _compute_loss_components(
        self,
        model,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Helper method to compute all three loss components.
        Extracted to avoid code duplication between compute_loss() and get_batch_metrics().

        Args:
            model: Policy model (π_θ)
            inputs: Batch of tokenized inputs (DPO format)

        Returns:
            Tuple of (loss_sft, loss_dpo, loss_kl, auxiliary_outputs)
            where auxiliary_outputs contains:
                - policy_chosen_logps
                - policy_rejected_logps
                - reference_chosen_logps
                - reference_rejected_logps
                - probs (softmax probabilities)
                - margin
        """
        # ========================================================================
        # STEP 1: Compute L_SFT (Supervised Fine-Tuning Loss)
        # ========================================================================
        # Math: L_SFT = (1/N_SFT) ∑_{(x,y^chosen)} ∑_{t=1}^T log P_π(y_t^chosen | x, y_{<t}^chosen)
        # Code: PyTorch CrossEntropyLoss over chosen responses (model's forward pass with labels)
        # ========================================================================
        labels = inputs["labels"]
        concatenated_batch_size = labels.shape[0]
        chosen_batch_size = concatenated_batch_size // 2

        outputs_chosen = model(
            input_ids=inputs["input_ids"][:chosen_batch_size],
            attention_mask=inputs["attention_mask"][:chosen_batch_size],
            labels=labels[:chosen_batch_size]
        )
        loss_sft = outputs_chosen.loss

        # ========================================================================
        # STEP 2: Compute L_DPO (Direct Preference Optimization Loss)
        # ========================================================================
        # IMPLEMENTATION DECISION: Standard DPO (Rafailov et al. 2023)
        #
        # Proposal (Loss_Fully_explanded.png):
        #   L_DPO = -(1/N_DPO) ∑ log σ(β·[log π_θ(y^+|x) - log π_θ(y^-|x)])
        #   (No reference model in contrastive term - relies on separate L_KL)
        #
        # Standard DPO (Rafailov et al. 2023 - IMPLEMENTED HERE):
        #   L_DPO = -(1/N_DPO) ∑ log σ(β·[(log π_θ(y^+|x) - log π_ref(y^+|x)) -
        #                                 (log π_θ(y^-|x) - log π_ref(y^-|x))])
        #   (Reference model included in contrastive term)
        #
        # Why Standard DPO?
        #   1. Proven formula (Rafailov 2023, HuggingFace TRL, all production implementations)
        #   2. Anchors preference learning to reference baseline (prevents arbitrary drift)
        #   3. Explicit L_KL provides additional regularization (addresses 2024 research concerns)
        #   4. Conservative approach: double regularization (implicit in DPO + explicit L_KL)
        #
        # Math: L_DPO = -log(P^+) where P^+ = exp(β·r^+) / [exp(β·r^+) + exp(β·r^-)]
        #       r^+ = (log π_θ(y^+|x) - log π_ref(y^+|x))  <- reward for chosen
        #       r^- = (log π_θ(y^-|x) - log π_ref(y^-|x))  <- reward for rejected
        #
        # Code: logits_chosen = β * (policy_chosen_logps - reference_chosen_logps)
        #       logits_rejected = β * (policy_rejected_logps - reference_rejected_logps)
        #       P^+ = softmax([logits_chosen, logits_rejected])[:, 0]
        #       loss_dpo = -log(P^+)
        # ========================================================================
        model_output = self.concatenated_forward(model, inputs)
        policy_chosen_logps = model_output["chosen_logps"]
        policy_rejected_logps = model_output["rejected_logps"]

        # Get reference log probs
        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            reference_chosen_logps = inputs["ref_chosen_logps"]
            reference_rejected_logps = inputs["ref_rejected_logps"]
        else:
            reference_chosen_logps, reference_rejected_logps = self.compute_ref_log_probs(inputs)

        # DPO contrastive loss (Standard DPO with reference model)
        logits_chosen = self.beta * (policy_chosen_logps - reference_chosen_logps)
        logits_rejected = self.beta * (policy_rejected_logps - reference_rejected_logps)
        logits_concat = torch.stack([logits_chosen, logits_rejected], dim=1)
        probs = F.softmax(logits_concat, dim=1)
        loss_dpo = -torch.log(probs[:, 0] + 1e-10).mean()

        # ========================================================================
        # STEP 3: Compute L_KL (KL Divergence Regularization)
        # ========================================================================
        # Math (Theoretical): L_KL = (1/N_DPO) ∑_{(x,y)} ∑_{t=1}^T P_π(y_t|x,y_{<t}) log[P_π(y_t|x,y_{<t})/P_ref(y_t|x,y_{<t})]
        # Math (Practical Approximation): L_KL ≈ (1/2) * [mean(log π_θ(y^+|x) - log π_ref(y^+|x)) +
        #                                                   mean(log π_θ(y^-|x) - log π_ref(y^-|x))]
        # Code: kl_chosen = policy_chosen_logps - reference_chosen_logps
        #       kl_rejected = policy_rejected_logps - reference_rejected_logps
        #       loss_kl = mean([kl_chosen, kl_rejected])
        # Note: Approximation uses observed tokens only (not full vocab sum).
        #       Standard practice in DPO/PPO/RLHF (avoids 128K vocab × seq_len computation).
        # ========================================================================
        kl_chosen = policy_chosen_logps - reference_chosen_logps
        kl_rejected = policy_rejected_logps - reference_rejected_logps
        loss_kl = (kl_chosen.mean() + kl_rejected.mean()) / 2

        # ========================================================================
        # STEP 4: Auxiliary outputs for metrics/logging
        # ========================================================================
        margin = policy_chosen_logps - policy_rejected_logps

        auxiliary_outputs = {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps": reference_rejected_logps,
            "probs": probs,
            "margin": margin,
            "kl_chosen": kl_chosen,
            "kl_rejected": kl_rejected,
            "outputs_chosen": outputs_chosen,  # For return_outputs in compute_loss
        }

        return loss_sft, loss_dpo, loss_kl, auxiliary_outputs

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """
        Compute Unified Loss: L_unified = L_SFT + λ₁·L_DPO + λ₂·L_KL

        Per Ecliptica PDF (Loss_Fully_explanded.png):
        - L_SFT: Supervised fine-tuning on chosen responses
        - L_DPO: Direct preference optimization (contrastive)
        - L_KL: KL divergence to reference model

        All three losses computed simultaneously in SAME forward pass.

        Args:
            model: Policy model
            inputs: Batch of tokenized inputs (DPO format)
            return_outputs: Whether to return outputs dict
            num_items_in_batch: Unsloth compatibility (unused)

        Returns:
            loss or (loss, outputs) depending on return_outputs
        """
        # Compute all loss components using helper method (DRY principle)
        loss_sft, loss_dpo, loss_kl, aux = self._compute_loss_components(model, inputs)

        # Unified Loss
        loss = (
            self.lambda_sft * loss_sft +
            self.lambda_dpo * loss_dpo +
            self.lambda_kl * loss_kl
        )

        # Logging (every logging_steps)
        if self.state.global_step % self.args.logging_steps == 0:
            # Compute reward metrics (DPO standard)
            rewards_chosen = self.beta * (aux["policy_chosen_logps"] - aux["reference_chosen_logps"])
            rewards_rejected = self.beta * (aux["policy_rejected_logps"] - aux["reference_rejected_logps"])
            rewards_accuracies = (rewards_chosen > rewards_rejected).float()

            # Compute perplexity (standard LM quality metric)
            perplexity = torch.exp(loss_sft)

            log_metrics = {
                # Total loss
                "cita/loss_total": loss.item(),

                # Loss components
                "cita/loss_sft": loss_sft.item(),
                "cita/loss_dpo": loss_dpo.item(),
                "cita/loss_kl": loss_kl.item(),

                # Hyperparameters
                "cita/lambda_sft": self.lambda_sft,
                "cita/lambda_dpo": self.lambda_dpo,
                "cita/lambda_kl": self.lambda_kl,

                # Margin monitoring (critical for safety)
                "cita/margin": aux["margin"].mean().item(),
                "cita/margin_std": aux["margin"].std().item(),
                "cita/margin_min": aux["margin"].min().item(),

                # Log probabilities (for debugging)
                "cita/chosen_logps": aux["policy_chosen_logps"].mean().item(),
                "cita/rejected_logps": aux["policy_rejected_logps"].mean().item(),
                "cita/ref_chosen_logps": aux["reference_chosen_logps"].mean().item(),
                "cita/ref_rejected_logps": aux["reference_rejected_logps"].mean().item(),

                # KL breakdown
                "cita/kl_chosen": aux["kl_chosen"].mean().item(),
                "cita/kl_rejected": aux["kl_rejected"].mean().item(),

                # Preference probability (should increase during training)
                "cita/prob_chosen": aux["probs"][:, 0].mean().item(),

                # Reward metrics (DPO standard) - more interpretable than raw margin
                "rewards/chosen": rewards_chosen.mean().item(),
                "rewards/rejected": rewards_rejected.mean().item(),
                "rewards/accuracies": rewards_accuracies.mean().item(),
                "rewards/margin": (rewards_chosen - rewards_rejected).mean().item(),

                # Perplexity (standard LM quality metric) - detects degradation early
                "cita/perplexity": perplexity.item(),
            }
            self.log(log_metrics)

        return (loss, aux["outputs_chosen"]) if return_outputs else loss

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        train_eval: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute batch metrics for evaluation using unified loss.

        Returns:
            loss: Total CITA unified loss (L_SFT + λ₁·L_DPO + λ₂·L_KL)
            metrics: Dict of evaluation metrics
        """
        # Forward pass with no gradients
        with torch.no_grad():
            # Use same helper method as compute_loss() (DRY principle)
            loss_sft, loss_dpo, loss_kl, aux = self._compute_loss_components(model, batch)

            # Unified loss (consistent with compute_loss)
            loss = (
                self.lambda_sft * loss_sft +
                self.lambda_dpo * loss_dpo +
                self.lambda_kl * loss_kl
            )

            # Metrics
            metrics = {
                f"{train_eval}/loss": loss.item(),
                f"{train_eval}/loss_sft": loss_sft.item(),
                f"{train_eval}/loss_dpo": loss_dpo.item(),
                f"{train_eval}/loss_kl": loss_kl.item(),
                f"{train_eval}/margin": aux["margin"].mean().item(),
                f"{train_eval}/prob_chosen": aux["probs"][:, 0].mean().item(),
                f"{train_eval}/kl_chosen": aux["kl_chosen"].mean().item(),
                f"{train_eval}/kl_rejected": aux["kl_rejected"].mean().item(),
            }

        return loss, metrics

    def training_step(self, model, inputs):
        """
        Override training_step to add gradient norm monitoring
        Detects training instability early (before mode collapse)
        """
        # Standard training step from parent class
        loss = super().training_step(model, inputs)

        # Compute and log gradient norm (after backward pass)
        if self.state.global_step % self.args.logging_steps == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Log gradient norm (detects instability: high LR + short warmup → explosion)
            self.log({"cita/grad_norm": total_norm})

            # Optional: warn if gradient norm too high (>10 indicates instability)
            if total_norm > 10.0:
                print(f"\n⚠️  WARNING: High gradient norm ({total_norm:.2f}) - training may be unstable!")
                print(f"   Consider: lower LR, longer warmup, or gradient clipping\n")

        return loss
