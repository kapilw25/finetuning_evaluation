"""
CITA Trainer - Contrastive Instruction-Tuned Alignment (Stacked Training Edition)
Based on Ecliptica paper (Legacy_code/2025_Ecliptica.pdf pages 5-7)

ORIGINAL CITA: L_unified = L_SFT + λ_DPO·L_DPO + λ_KL·L_KL
STACKED TRAINING (Base→SFT→DPO→CITA): L_unified = λ_DPO·L_DPO + λ_KL·L_KL

Changes for Stacked Training:
- Removed L_SFT (causes catastrophic interference on DPO-tuned models)
- Uses DPOTrainer.dpo_loss() for apple-to-apple comparison with DPO baseline
- Adds explicit L_KL regularization on top of DPO
"""

import torch
from trl import DPOTrainer
from typing import Dict, Optional, Tuple


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
        from cita_trainer import CITATrainer

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
        lambda_dpo: float = 1.0,
        lambda_kl: float = 0.01,
        beta: float = 0.1,
        **kwargs
    ):
        """
        Initialize CITA Trainer for Stacked Training (Base→SFT→DPO→CITA)

        Args:
            model: Policy model to fine-tune (should be DPO-tuned for stacked training)
            tokenizer: Tokenizer
            args: DPOConfig
            train_dataset: Dataset in DPO format (prompt, chosen, rejected)
            lambda_dpo: Weight for DPO loss (default 1.0)
            lambda_kl: Weight for KL regularization (default 0.01)
            beta: Contrastive temperature (default 0.1, usually set in DPOConfig)

        Stacked Training Design:
            L_unified = λ_DPO·L_DPO + λ_KL·L_KL  (NO L_SFT!)

            WHY NO L_SFT?
            - Model is already DPO-tuned (margin=2.95, knows preferences)
            - Adding L_SFT forces relearning chosen responses from scratch
            - Result: Catastrophic interference (margin collapses 2.95 → 0.10)
            - See logs_training/iter2/report.md:148-178 for detailed analysis

        Apple-to-Apple Comparison:
            - Uses DPOTrainer.dpo_loss() (EXACT same L_DPO as baseline)
            - Adds explicit L_KL regularization on top of DPO
            - Any improvement is from L_KL, not implementation differences

        Original CITA (Ecliptica paper - NOT applicable here):
            L_unified = L_SFT + λ_DPO·L_DPO + λ_KL·L_KL
            Designed for Base → CITA (skip SFT/DPO stages)
        """
        super().__init__(
            model=model,
            processing_class=tokenizer,  # TRL 0.8+ uses processing_class instead of tokenizer
            args=args,  # DPOConfig already contains beta
            train_dataset=train_dataset,
            **kwargs
        )

        if lambda_kl <= 0:
            raise ValueError(f"lambda_kl must be > 0 (got {lambda_kl})")

        self.lambda_dpo = lambda_dpo
        self.lambda_kl = lambda_kl
        # Note: DPOTrainer already creates ref_model, no need to override

    def _compute_loss_components(
        self,
        model,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Helper method to compute DPO and KL loss components (NO L_SFT for stacked training).
        Extracted to avoid code duplication between compute_loss() and get_batch_metrics().

        Args:
            model: Policy model (π_θ)
            inputs: Batch of tokenized inputs (DPO format)

        Returns:
            Tuple of (loss_dpo, loss_kl, auxiliary_outputs)
            where auxiliary_outputs contains:
                - policy_chosen_logps
                - policy_rejected_logps
                - reference_chosen_logps
                - reference_rejected_logps
                - chosen_rewards (from DPOTrainer.dpo_loss)
                - rejected_rewards (from DPOTrainer.dpo_loss)
                - margin
        """
        # ========================================================================
        # STEP 1: L_SFT REMOVED FOR STACKED TRAINING
        # ========================================================================
        # WHY NO L_SFT?
        #   - Model is already DPO-tuned (margin=2.95, knows preferences)
        #   - Adding L_SFT forces relearning chosen responses from scratch
        #   - Result: Catastrophic interference (margin collapses 2.95 → 0.10)
        #   - See report.md:148-178 for detailed analysis
        #
        # ORIGINAL CITA (Ecliptica paper):
        #   L_unified = L_SFT + λ_DPO·L_DPO + λ_KL·L_KL
        #
        # STACKED TRAINING (Base→SFT→DPO→CITA):
        #   L_unified = λ_DPO·L_DPO + λ_KL·L_KL  (NO L_SFT!)
        # ========================================================================

        # ========================================================================
        # STEP 2: Compute L_DPO (Direct Preference Optimization Loss)
        # ========================================================================
        # IMPLEMENTATION DECISION: Use DPOTrainer's dpo_loss() method (Apple-to-Apple Comparison)
        #
        # WHY USE DPOTrainer's dpo_loss() instead of custom implementation?
        #
        # ✅ PROS:
        #   1. **Apple-to-apple comparison**: EXACT same L_DPO as DPO baseline
        #      - Ensures fair comparison: any CITA improvement is from L_KL, not implementation
        #      - Same formula, same numerical precision, same edge case handling
        #
        #   2. **Automatic TRL updates**: Benefits from library improvements
        #      - Liger kernel support (2024): 80% memory savings via use_liger_loss=True
        #      - Padding-free training: Removes wasted compute on padding tokens
        #      - Future optimizations: Get improvements for free
        #
        #   3. **Multiple loss types**: Can experiment beyond sigmoid
        #      - loss_type="sigmoid" (standard DPO - default)
        #      - loss_type="hinge", "ipo", "robust" (alternative formulations)
        #      - Set via self.args.loss_type in DPOConfig
        #
        #   4. **Cleaner code**: 1 line vs 5 lines of manual sigmoid/softmax
        #
        #   5. **No performance loss**: SAME forward pass as custom implementation
        #      - Still uses self.concatenated_forward() (inherited from DPOTrainer)
        #      - Still uses self.compute_ref_log_probs() (inherited from DPOTrainer)
        #      - Only difference: who computes final sigmoid/log (us vs library)
        #
        # ❌ CONS:
        #   - Method call overhead: ~1-2% slower (negligible)
        #
        # Math: L_DPO = -log(P^+) where P^+ = exp(β·r^+) / [exp(β·r^+) + exp(β·r^-)]
        #       r^+ = (log π_θ(y^+|x) - log π_ref(y^+|x))  <- reward for chosen
        #       r^- = (log π_θ(y^-|x) - log π_ref(y^-|x))  <- reward for rejected
        # ========================================================================

        # Get policy log probs (concatenated forward pass - efficient!)
        model_output = self.concatenated_forward(model, inputs)
        policy_chosen_logps = model_output["chosen_logps"]
        policy_rejected_logps = model_output["rejected_logps"]

        # Get reference log probs
        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            reference_chosen_logps = inputs["ref_chosen_logps"]
            reference_rejected_logps = inputs["ref_rejected_logps"]
        else:
            reference_chosen_logps, reference_rejected_logps = self.compute_ref_log_probs(inputs)

        # ✅ USE DPOTrainer's dpo_loss() method (apple-to-apple comparison with baseline)
        # Returns: (loss, chosen_rewards, rejected_rewards)
        loss_type = getattr(self.args, 'loss_type', 'sigmoid')  # Default to sigmoid if not set
        loss_dpo, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            loss_type=loss_type
        )

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
            "chosen_rewards": chosen_rewards,  # From DPOTrainer.dpo_loss()
            "rejected_rewards": rejected_rewards,  # From DPOTrainer.dpo_loss()
            "margin": margin,
            "kl_chosen": kl_chosen,
            "kl_rejected": kl_rejected,
        }

        return loss_dpo, loss_kl, auxiliary_outputs

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """
        Compute Unified Loss for Stacked Training: L_unified = λ_DPO·L_DPO + λ_KL·L_KL

        STACKED TRAINING (Base→SFT→DPO→CITA):
        - L_DPO: Direct preference optimization (uses DPOTrainer.dpo_loss() - apple-to-apple)
        - L_KL: KL divergence to reference model (additional regularization)
        - NO L_SFT: Model already learned from DPO, adding L_SFT causes catastrophic interference

        ORIGINAL CITA (Ecliptica PDF - Base→CITA):
        - L_unified = L_SFT + λ_DPO·L_DPO + λ_KL·L_KL
        - Required L_SFT to teach safe responses from scratch
        - Not applicable for stacked training on DPO-tuned models

        Args:
            model: Policy model
            inputs: Batch of tokenized inputs (DPO format)
            return_outputs: Whether to return outputs dict
            num_items_in_batch: Unsloth compatibility (unused)

        Returns:
            loss or (loss, outputs) depending on return_outputs
        """
        # Compute DPO and KL loss components (NO L_SFT for stacked training)
        loss_dpo, loss_kl, aux = self._compute_loss_components(model, inputs)

        # Unified Loss (NO lambda_sft - always 0 for stacked training)
        loss = (
            self.lambda_dpo * loss_dpo +
            self.lambda_kl * loss_kl
        )

        # Logging (every logging_steps)
        if self.state.global_step % self.args.logging_steps == 0:
            # Use rewards from DPOTrainer.dpo_loss() (apple-to-apple comparison)
            rewards_chosen = aux["chosen_rewards"]
            rewards_rejected = aux["rejected_rewards"]
            rewards_accuracies = (rewards_chosen > rewards_rejected).float()

            log_metrics = {
                # Total loss (DPO + KL only, no SFT for stacked training)
                "cita/loss_total": loss.item(),

                # Loss components (NO loss_sft - removed for stacked training)
                "cita/loss_dpo": loss_dpo.item(),
                "cita/loss_kl": loss_kl.item(),

                # Hyperparameters (NO lambda_sft - always 0 for stacked training)
                "cita/lambda_dpo": self.lambda_dpo,
                "cita/lambda_kl": self.lambda_kl,

                # Margin monitoring (critical for safety)
                "cita/margin": aux["margin"].mean().item(),
                "cita/margin_std": aux["margin"].std(unbiased=False).item(),  # unbiased=False handles batch_size=1
                "cita/margin_min": aux["margin"].min().item(),

                # Log probabilities (for debugging)
                "cita/chosen_logps": aux["policy_chosen_logps"].mean().item(),
                "cita/rejected_logps": aux["policy_rejected_logps"].mean().item(),
                "cita/ref_chosen_logps": aux["reference_chosen_logps"].mean().item(),
                "cita/ref_rejected_logps": aux["reference_rejected_logps"].mean().item(),

                # KL breakdown
                "cita/kl_chosen": aux["kl_chosen"].mean().item(),
                "cita/kl_rejected": aux["kl_rejected"].mean().item(),

                # Reward metrics (from DPOTrainer.dpo_loss - apple-to-apple with baseline)
                "rewards/chosen": rewards_chosen.mean().item(),
                "rewards/rejected": rewards_rejected.mean().item(),
                "rewards/accuracies": rewards_accuracies.mean().item(),
                "rewards/margin": (rewards_chosen - rewards_rejected).mean().item(),
            }
            self.log(log_metrics)

        # No outputs_chosen since we removed L_SFT computation
        # DPOTrainer's standard return is just loss
        return loss

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        train_eval: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute batch metrics for evaluation using unified loss (stacked training).

        Returns:
            loss: Total CITA unified loss (λ_DPO·L_DPO + λ_KL·L_KL, NO L_SFT)
            metrics: Dict of evaluation metrics
        """
        # Forward pass with no gradients
        with torch.no_grad():
            # Use same helper method as compute_loss() (DRY principle)
            loss_dpo, loss_kl, aux = self._compute_loss_components(model, batch)

            # Unified loss (consistent with compute_loss - NO lambda_sft)
            loss = (
                self.lambda_dpo * loss_dpo +
                self.lambda_kl * loss_kl
            )

            # Metrics (NO loss_sft - removed for stacked training)
            metrics = {
                f"{train_eval}/loss": loss.item(),
                f"{train_eval}/loss_dpo": loss_dpo.item(),
                f"{train_eval}/loss_kl": loss_kl.item(),
                f"{train_eval}/margin": aux["margin"].mean().item(),
                f"{train_eval}/kl_chosen": aux["kl_chosen"].mean().item(),
                f"{train_eval}/kl_rejected": aux["kl_rejected"].mean().item(),
            }

        return loss, metrics

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to add gradient norm monitoring
        Detects training instability early (before mode collapse)
        Now also verifies gradient clipping is working (iter8)

        FIX (iter12): Manually clip gradients BEFORE measuring grad_norm
        - super().training_step() only does backward(), NOT clipping
        - Clipping happens in outer training loop AFTER this method returns
        - We need to clip HERE to measure correct post-clipping grad_norm

        Args:
            model: The model to train
            inputs: The input batch
            num_items_in_batch: Number of items in the batch (Transformers 4.46+)
        """
        # Standard training step from parent class (backward pass only)
        loss = super().training_step(model, inputs, num_items_in_batch)

        # FIX (iter12): Manually clip gradients BEFORE measuring
        # (parent class doesn't clip - that happens in outer training loop)
        max_grad_norm = getattr(self.args, 'max_grad_norm', None)
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Compute and log gradient norm (NOW properly clipped)
        if self.state.global_step % self.args.logging_steps == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Log gradient norm (now properly clipped)
            log_dict = {"cita/grad_norm_clipped": total_norm}

            # Log clipping config for debugging
            if max_grad_norm is not None and max_grad_norm > 0:
                log_dict["cita/max_grad_norm_config"] = max_grad_norm
                # After manual clipping, norm should always be ≤ max_grad_norm
                # If it equals max (within tolerance), clipping was active
                if total_norm >= max_grad_norm * 0.99:  # Within 1% tolerance
                    log_dict["cita/clipping_active"] = 1.0
                else:
                    log_dict["cita/clipping_active"] = 0.0

            self.log(log_dict)

            # Warn if gradient norm equals max (clipping is active)
            # This means raw gradients are larger than max_grad_norm
            if max_grad_norm is not None and total_norm >= max_grad_norm * 0.99:
                print(f"\n⚠️  CLIPPING ACTIVE: grad_norm={total_norm:.2f} at limit {max_grad_norm}")
                print(f"   Raw gradients are larger - this is expected during training\n")

            # SAFETY CHECK 1: Verify clipping logic is working
            # If grad_norm > max_grad_norm, clipping failed (shouldn't happen)
            if max_grad_norm is not None and total_norm > max_grad_norm * 1.1:
                print(f"\n⚠️  WARNING: Clipping may have failed (grad_norm={total_norm:.2f} > max={max_grad_norm})")
                print(f"   Continuing training, but monitor closely...\n")
                # Don't crash - just log and continue (fallback behavior)

            # SAFETY CHECK 2: Catastrophic explosion (same as iter11)
            # If grad_norm > 50, training has diverged (regardless of clipping)
            if total_norm > 50.0:
                print(f"\n🚨 EXPLOSION DETECTED: grad_norm={total_norm:.2f} > 50.0")
                print(f"   Training has diverged catastrophically")
                print(f"   Stopping at step {self.state.global_step}")
                print(f"   Check tensorboard logs: tensorboard_logs/CITA_*/")
                raise ValueError(f"Training exploded (grad_norm={total_norm:.2f})")

        return loss
