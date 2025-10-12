"""
CITA Trainer - Contrastive Instruction-Tuned Alignment
Based on Ecliptica paper (plan7_Ecliptica_Paper_Results.md lines 307-408)
"""

import torch
import torch.nn.functional as F
from trl import DPOTrainer


class CITATrainer(DPOTrainer):
    """
    CITA Trainer extends DPO with explicit instruction conditioning
    
    Key difference from DPO:
    - DPO: P(Y+|X) vs P(Y-|X)  [no explicit instruction]
    - CITA: P(Y+|I,X) vs P(Y-|I,X)  [instruction I explicitly conditioned]
    
    Loss: L_CITA-KL = -log[exp(β·log π(Y+|I,X)) / 
                            (exp(β·log π(Y+|I,X)) + exp(β·log π(Y-|I,X)))]
                    + λ·KL[π_θ || π_0]
    """
    
    def __init__(
        self,
        model,
        ref_model,
        args,
        train_dataset,
        processing_class,
        lambda_kl=0.01,
        **kwargs
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            **kwargs
        )
        self.lambda_kl = lambda_kl
        # Beta is already in args (DPOConfig.beta)
    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        CITA-KL Loss from Ecliptica paper

        Components:
        1. Contrastive term: Maximize P(chosen) / (P(chosen) + P(rejected))
        2. KL regularization: Prevent drift from reference model

        Args:
            num_items_in_batch: Unsloth compatibility (unused but required)
        """
        # Use parent DPOTrainer's methods to compute log probabilities
        # concatenated_forward computes policy log probs
        model_output = self.concatenated_forward(model, inputs)
        policy_chosen_logps = model_output["chosen_logps"]
        policy_rejected_logps = model_output["rejected_logps"]

        # compute_ref_log_probs computes reference log probs
        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            reference_chosen_logps = inputs["ref_chosen_logps"]
            reference_rejected_logps = inputs["ref_rejected_logps"]
        else:
            reference_chosen_logps, reference_rejected_logps = self.compute_ref_log_probs(inputs)

        # ========================================================================
        # Contrastive term (Ecliptica paper, Page 7)
        # ========================================================================
        # Paper formula: L = -log[exp(β·log π+) / (exp(β·log π+) + exp(β·log π-))]
        # 
        # Mathematically equivalent to: -log(softmax([β·log π+, β·log π-])[0])
        # 
        # Implementation uses log-space for numerical stability:
        # - policy_chosen_logps = log π(Y+|I,X)   [log probability of accepted response]
        # - policy_rejected_logps = log π(Y-|I,X) [log probability of rejected response]
        # - β·log π = log(π^β)                    [temperature-scaled log probability]
        # 
        # Then softmax exponentiates: softmax([log a, log b]) = [a/(a+b), b/(a+b)]
        # ========================================================================

        logits_chosen = self.beta * policy_chosen_logps      # β·log π(Y+|I,X)
        logits_rejected = self.beta * policy_rejected_logps  # β·log π(Y-|I,X)

        # Stack and compute softmax (exponentiates β·log π internally)
        logits_concat = torch.stack([logits_chosen, logits_rejected], dim=1)
        probs = F.softmax(logits_concat, dim=1)

        # Final loss: -log P(chosen | chosen+rejected)
        loss_contrastive = -torch.log(probs[:, 0] + 1e-10).mean()

        # KL regularization (prevent drift from reference model)
        # True KL divergence: KL[π_θ || π_0] = E[log π_θ - log π_0]
        # We approximate using sequence-level log-probs (not token-level for efficiency)
        loss_kl = (
            (policy_chosen_logps - reference_chosen_logps).mean() +
            (policy_rejected_logps - reference_rejected_logps).mean()
        ) / 2

        # Total CITA loss
        loss = loss_contrastive + self.lambda_kl * loss_kl

        # Logging
        if self.state.global_step % self.args.logging_steps == 0:
            log_metrics = {
                "cita/loss_contrastive": loss_contrastive.item(),
                "cita/loss_kl": loss_kl.item(),
                "cita/margin": (policy_chosen_logps - policy_rejected_logps).mean().item(),
                "cita/chosen_logps": policy_chosen_logps.mean().item(),
                "cita/rejected_logps": policy_rejected_logps.mean().item(),
            }
            self.log(log_metrics)

        return (loss, {"loss": loss}) if return_outputs else loss
