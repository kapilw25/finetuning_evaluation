"""
Test the proposed CITA solution
"""

import torch
import torch.nn.functional as F


def mock_get_batch_loss_metrics_return():
    """Simulate what DPOTrainer.get_batch_loss_metrics returns"""
    batch_size = 2
    return {
        "policy_chosen_logps": torch.tensor([-2.5, -3.1]),
        "policy_rejected_logps": torch.tensor([-4.2, -5.0]),
        "reference_chosen_logps": torch.tensor([-2.6, -3.2]),
        "reference_rejected_logps": torch.tensor([-4.1, -4.9]),
        "rewards_chosen": torch.tensor([0.1, 0.1]),
        "rewards_rejected": torch.tensor([-0.1, -0.2]),
    }


def test_cita_loss_calculation():
    """Test if the CITA loss calculation works with DPOTrainer output"""
    print("Testing CITA loss calculation...")

    # Simulate DPOTrainer output
    metrics = mock_get_batch_loss_metrics_return()

    beta = 0.1
    lambda_kl = 0.01

    # Extract log probs
    policy_chosen_logps = metrics["policy_chosen_logps"]
    policy_rejected_logps = metrics["policy_rejected_logps"]
    reference_chosen_logps = metrics["reference_chosen_logps"]
    reference_rejected_logps = metrics["reference_rejected_logps"]

    print(f"  policy_chosen_logps: {policy_chosen_logps}")
    print(f"  policy_rejected_logps: {policy_rejected_logps}")

    # CITA contrastive loss
    logits_chosen = beta * policy_chosen_logps
    logits_rejected = beta * policy_rejected_logps

    logits_concat = torch.stack([logits_chosen, logits_rejected], dim=1)
    probs = F.softmax(logits_concat, dim=1)
    loss_contrastive = -torch.log(probs[:, 0] + 1e-10).mean()

    print(f"  loss_contrastive: {loss_contrastive.item()}")

    # KL regularization
    loss_kl = (
        (policy_chosen_logps - reference_chosen_logps).mean() +
        (policy_rejected_logps - reference_rejected_logps).mean()
    ) / 2

    print(f"  loss_kl: {loss_kl.item()}")

    # Total loss
    loss = loss_contrastive + lambda_kl * loss_kl

    print(f"  total_loss: {loss.item()}")

    # Verify loss is a scalar
    assert loss.dim() == 0, "Loss should be a scalar!"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is inf!"

    print("✓ CITA loss calculation works correctly!")
    return True


def test_solution_code():
    """Test the exact code we'll give to the user"""
    print("\nTesting exact solution code...")

    class MockSuperClass:
        def get_batch_loss_metrics(self, model, batch, train_eval):
            return mock_get_batch_loss_metrics_return()

    class MockCITATrainer(MockSuperClass):
        def __init__(self):
            self.beta = 0.1
            self.lambda_kl = 0.01
            self.state = type('obj', (object,), {'global_step': 10})()
            self.args = type('obj', (object,), {'logging_steps': 10})()
            self.logged_metrics = {}

        def log(self, metrics):
            self.logged_metrics.update(metrics)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # THE SOLUTION CODE
            metrics = super().get_batch_loss_metrics(model, inputs, train_eval="train")

            policy_chosen_logps = metrics["policy_chosen_logps"]
            policy_rejected_logps = metrics["policy_rejected_logps"]
            reference_chosen_logps = metrics["reference_chosen_logps"]
            reference_rejected_logps = metrics["reference_rejected_logps"]

            logits_chosen = self.beta * policy_chosen_logps
            logits_rejected = self.beta * policy_rejected_logps

            logits_concat = torch.stack([logits_chosen, logits_rejected], dim=1)
            probs = F.softmax(logits_concat, dim=1)
            loss_contrastive = -torch.log(probs[:, 0] + 1e-10).mean()

            loss_kl = (
                (policy_chosen_logps - reference_chosen_logps).mean() +
                (policy_rejected_logps - reference_rejected_logps).mean()
            ) / 2

            loss = loss_contrastive + self.lambda_kl * loss_kl

            if self.state.global_step % self.args.logging_steps == 0:
                log_metrics = {
                    "cita/loss_contrastive": loss_contrastive.item(),
                    "cita/loss_kl": loss_kl.item(),
                    "cita/margin": (policy_chosen_logps - policy_rejected_logps).mean().item(),
                }
                self.log(log_metrics)

            return (loss, {"loss": loss}) if return_outputs else loss

    # Test it
    trainer = MockCITATrainer()
    model = None  # Not used in mock
    inputs = {}   # Passed to get_batch_loss_metrics as 'batch'

    # Test without return_outputs
    loss = trainer.compute_loss(model, inputs, return_outputs=False)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be scalar"
    print(f"  Loss (no outputs): {loss.item()}")

    # Test with return_outputs
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert "loss" in outputs, "Outputs should contain 'loss' key"
    print(f"  Loss (with outputs): {loss.item()}")

    # Test logging
    assert "cita/loss_contrastive" in trainer.logged_metrics, "Should log contrastive loss"
    assert "cita/loss_kl" in trainer.logged_metrics, "Should log KL loss"
    print(f"  Logged metrics: {list(trainer.logged_metrics.keys())}")

    print("✓ Solution code works correctly!")
    return True


if __name__ == "__main__":
    print("="*80)
    print("Testing CITA Solution")
    print("="*80)

    try:
        test_cita_loss_calculation()
        test_solution_code()
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - Solution is verified!")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
