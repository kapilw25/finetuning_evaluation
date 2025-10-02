
# ===================================================================
# GRIT VERIFICATION: Add this to your notebook for runtime monitoring
# ===================================================================

# 1. Import verification callback (add after other imports)
from transformers import TrainerCallback

class GritVerificationCallback(TrainerCallback):
    """Monitor GRIT preconditioning during training"""

    def __init__(self, grit_manager):
        self.grit_manager = grit_manager
        self.preconditioning_count = 0
        self.inversion_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        last_loss = state.log_history[-1].get("loss") if state.log_history else None
        old_last = self.grit_manager.last_kfac_update_step
        self.grit_manager.step(loss=last_loss)

        if self.grit_manager.last_kfac_update_step > old_last:
            self.inversion_count += 1
            print(f"\n[Step {state.global_step}] ✓ Fisher inversion #{self.inversion_count}")

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.grit_manager.factors_are_ready:
            self.grit_manager.precondition_gradients()
            self.preconditioning_count += 1
            if state.global_step % 10 == 0:
                print(f"[Step {state.global_step}] ✓ Preconditioning applied (#{self.preconditioning_count})")
        else:
            if state.global_step <= 20 or state.global_step % 10 == 0:
                print(f"[Step {state.global_step}] ✗ Preconditioning skipped (not ready)")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*80}")
        print(f"VERIFICATION SUMMARY:")
        print(f"  Total steps:        {state.global_step}")
        print(f"  Fisher inversions:  {self.inversion_count}")
        print(f"  Preconditioning:    {self.preconditioning_count}")
        print(f"  Status:             {'✓ WORKING' if self.preconditioning_count > 0 else '✗ NOT WORKING'}")
        print(f"{'='*80}\n")

# 2. Modify trainer to include verification callback
trainer = SFTTrainer(
    ...
    callbacks = [
        GritVerificationCallback(grit_manager),  # Add this for monitoring
    ],
)
