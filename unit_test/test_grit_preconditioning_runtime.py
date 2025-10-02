#!/usr/bin/env python3
"""
Runtime verification script to check if GRIT preconditioning is being applied during training.

This script monitors a training run and verifies:
1. Fisher matrices are being accumulated
2. K-FAC inversion happens at configured frequency
3. Gradient preconditioning is actually applied
4. Gradients are modified by preconditioning

Usage:
    python unit_test/test_grit_preconditioning_runtime.py
"""

import sys
import os
grit_path = os.path.join(os.path.dirname(__file__), '..', 'unsloth_2_Hybrid_Grit')
sys.path.insert(0, grit_path)

import torch
import torch.nn as nn
from transformers import TrainerCallback, TrainerState, TrainerControl
from grit.config import GRITConfig
from grit.manager import GRITManager


class GritVerificationCallback(TrainerCallback):
    """
    Verification callback that monitors GRIT operations during training.
    Logs detailed statistics to verify preconditioning is working.
    """

    def __init__(self, grit_manager):
        self.grit_manager = grit_manager
        self.preconditioning_applied_count = 0
        self.fisher_inversion_count = 0
        self.steps_without_preconditioning = []
        self.gradient_changes = []

    def on_step_begin(self, args, state, control, **kwargs):
        """Store gradient norms before preconditioning"""
        if hasattr(self, '_grad_norms_before'):
            delattr(self, '_grad_norms_before')

        self._grad_norms_before = {}
        for module in self.grit_manager.optimized_modules[:5]:  # Check first 5 modules
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_a = module.lora_A.get('default')
                lora_b = module.lora_B.get('default')
                if lora_a and lora_b and lora_a.weight.grad is not None:
                    self._grad_norms_before[module] = {
                        'a': lora_a.weight.grad.norm().item(),
                        'b': lora_b.weight.grad.norm().item()
                    }

    def on_step_end(self, args, state, control, **kwargs):
        """Update Fisher matrices"""
        last_loss = state.log_history[-1].get("loss") if state.log_history else None

        # Check Fisher matrix accumulation
        step = state.global_step
        if step == 1:
            print("\n" + "="*80)
            print("GRIT RUNTIME VERIFICATION - Training Started")
            print("="*80)
            print(f"Config: kfac_update_freq={self.grit_manager.config.kfac_update_freq}")
            print(f"Config: kfac_min_samples={getattr(self.grit_manager.config, 'kfac_min_samples', 64)}")
            print(f"Config: reprojection_freq={self.grit_manager.config.reprojection_freq}")
            print("="*80 + "\n")

        # Call manager step
        old_last_kfac = self.grit_manager.last_kfac_update_step
        self.grit_manager.step(loss=last_loss)

        # Check if Fisher inversion happened
        if self.grit_manager.last_kfac_update_step > old_last_kfac:
            self.fisher_inversion_count += 1
            print(f"\n{'='*80}")
            print(f"✓ FISHER INVERSION #{self.fisher_inversion_count} at step {step}")
            print(f"{'='*80}")

            # Check sample counts
            sample_module = self.grit_manager.optimized_modules[0]
            a_samples = self.grit_manager.num_samples_a.get(sample_module, 0)
            g_samples = self.grit_manager.num_samples_g.get(sample_module, 0)
            print(f"  Sample counts: A={a_samples}, G={g_samples}")
            print(f"  Factors ready: {self.grit_manager.factors_are_ready}")
            print(f"{'='*80}\n")

    def on_optimizer_step(self, args, state, control, **kwargs):
        """Apply natural gradient preconditioning"""
        step = state.global_step

        if self.grit_manager.factors_are_ready:
            # Store gradients before preconditioning
            grads_before = {}
            for module in self.grit_manager.optimized_modules[:5]:
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_a = module.lora_A.get('default')
                    lora_b = module.lora_B.get('default')
                    if lora_a and lora_b and lora_a.weight.grad is not None:
                        grads_before[module] = {
                            'a': lora_a.weight.grad.clone(),
                            'b': lora_b.weight.grad.clone()
                        }

            # Apply preconditioning
            self.grit_manager.precondition_gradients()
            self.preconditioning_applied_count += 1

            # Check if gradients actually changed
            gradients_modified = False
            for module, before in grads_before.items():
                lora_a = module.lora_A.get('default')
                lora_b = module.lora_B.get('default')
                if lora_a and lora_b and lora_a.weight.grad is not None:
                    change_a = (lora_a.weight.grad - before['a']).abs().mean().item()
                    change_b = (lora_b.weight.grad - before['b']).abs().mean().item()

                    if change_a > 1e-8 or change_b > 1e-8:
                        gradients_modified = True
                        self.gradient_changes.append({
                            'step': step,
                            'change_a': change_a,
                            'change_b': change_b
                        })
                        break

            if step % 10 == 0 or step <= 20:
                status = "✓ MODIFIED" if gradients_modified else "✗ NOT MODIFIED"
                print(f"[Step {step:3d}] Natural Gradient Applied (#{self.preconditioning_applied_count}) - Gradients {status}")
                if gradients_modified and self.gradient_changes:
                    last_change = self.gradient_changes[-1]
                    print(f"            Gradient change: A={last_change['change_a']:.6f}, B={last_change['change_b']:.6f}")
        else:
            self.steps_without_preconditioning.append(step)
            if step % 10 == 0 or step <= 20:
                print(f"[Step {step:3d}] ✗ Preconditioning SKIPPED (factors_are_ready=False)")

    def on_train_end(self, args, state, control, **kwargs):
        """Print final verification summary"""
        print("\n" + "="*80)
        print("GRIT RUNTIME VERIFICATION SUMMARY")
        print("="*80)

        total_steps = state.global_step
        expected_inversions = total_steps // self.grit_manager.config.kfac_update_freq

        print(f"\n1. Fisher Matrix Inversions:")
        print(f"   Expected: {expected_inversions}")
        print(f"   Actual:   {self.fisher_inversion_count}")
        print(f"   Status:   {'✓ PASS' if self.fisher_inversion_count >= expected_inversions else '✗ FAIL'}")

        print(f"\n2. Gradient Preconditioning:")
        print(f"   Total steps:              {total_steps}")
        print(f"   Preconditioning applied:  {self.preconditioning_applied_count}")
        print(f"   Skipped (not ready):      {len(self.steps_without_preconditioning)}")

        if self.preconditioning_applied_count > 0:
            coverage = (self.preconditioning_applied_count / total_steps) * 100
            print(f"   Coverage:                 {coverage:.1f}%")
            print(f"   Status:                   {'✓ PASS' if coverage > 50 else '⚠ LOW COVERAGE'}")
        else:
            print(f"   Status:                   ✗ FAIL - Never applied!")

        print(f"\n3. Gradient Modifications:")
        print(f"   Times gradients changed:  {len(self.gradient_changes)}")

        if self.gradient_changes:
            avg_change_a = sum(c['change_a'] for c in self.gradient_changes) / len(self.gradient_changes)
            avg_change_b = sum(c['change_b'] for c in self.gradient_changes) / len(self.gradient_changes)
            print(f"   Avg change (lora_A):      {avg_change_a:.6f}")
            print(f"   Avg change (lora_B):      {avg_change_b:.6f}")
            print(f"   Status:                   {'✓ PASS' if avg_change_a > 1e-6 else '✗ FAIL - Changes too small'}")
        else:
            print(f"   Status:                   ✗ FAIL - Gradients never modified!")

        print(f"\n4. Steps Without Preconditioning:")
        if self.steps_without_preconditioning:
            if len(self.steps_without_preconditioning) <= 20:
                print(f"   Steps: {self.steps_without_preconditioning}")
            else:
                print(f"   First 10: {self.steps_without_preconditioning[:10]}")
                print(f"   Last 10:  {self.steps_without_preconditioning[-10:]}")
        else:
            print(f"   None - All steps had preconditioning ready!")

        # Overall verdict
        print("\n" + "="*80)
        if (self.fisher_inversion_count > 0 and
            self.preconditioning_applied_count > 0 and
            len(self.gradient_changes) > 0):
            print("✓ VERDICT: GRIT Preconditioning IS WORKING!")
            print("  - Fisher matrices are being inverted")
            print("  - Natural gradients are being applied")
            print("  - Gradients are being modified")
        else:
            print("✗ VERDICT: GRIT Preconditioning NOT WORKING!")
            if self.fisher_inversion_count == 0:
                print("  ✗ Fisher matrices never inverted")
                print(f"    → Check: kfac_update_freq ({self.grit_manager.config.kfac_update_freq}) vs total_steps ({total_steps})")
            if self.preconditioning_applied_count == 0:
                print("  ✗ Preconditioning never applied")
                print(f"    → Check: factors_are_ready flag")
            if len(self.gradient_changes) == 0:
                print("  ✗ Gradients never modified")
                print(f"    → Check: Fisher inverse matrices")
        print("="*80 + "\n")


def create_test_code_snippet():
    """Generate code snippet to add to notebook for verification"""

    code = '''
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
            print(f"\\n[Step {state.global_step}] ✓ Fisher inversion #{self.inversion_count}")

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
        print(f"\\n{'='*80}")
        print(f"VERIFICATION SUMMARY:")
        print(f"  Total steps:        {state.global_step}")
        print(f"  Fisher inversions:  {self.inversion_count}")
        print(f"  Preconditioning:    {self.preconditioning_count}")
        print(f"  Status:             {'✓ WORKING' if self.preconditioning_count > 0 else '✗ NOT WORKING'}")
        print(f"{'='*80}\\n")

# 2. Modify trainer to include verification callback
trainer = SFTTrainer(
    ...
    callbacks = [
        GritVerificationCallback(grit_manager),  # Add this for monitoring
    ],
)
'''

    return code


def main():
    """Main function to display instructions"""

    print("\n" + "="*80)
    print("GRIT PRECONDITIONING RUNTIME VERIFICATION SCRIPT")
    print("="*80)

    print("\nThis script provides a verification callback to monitor GRIT during training.")
    print("\nTO USE IN YOUR NOTEBOOK:")
    print("-" * 80)

    code = create_test_code_snippet()
    print(code)

    print("-" * 80)
    print("\nWHAT TO EXPECT:")
    print("  ✓ Console output showing Fisher inversions at configured frequency")
    print("  ✓ Console output showing preconditioning being applied")
    print("  ✓ Summary at end showing total inversions and preconditioning count")
    print("\nIF PRECONDITIONING IS WORKING:")
    print("  - You'll see '[Step X] ✓ Fisher inversion #N' messages")
    print("  - You'll see '[Step X] ✓ Preconditioning applied #N' messages")
    print("  - Summary will show Status: ✓ WORKING")
    print("\nIF PRECONDITIONING IS NOT WORKING:")
    print("  - You'll see '[Step X] ✗ Preconditioning skipped (not ready)' messages")
    print("  - Summary will show Status: ✗ NOT WORKING")
    print("  - Check: kfac_update_freq might be too high for your max_steps")
    print("  - Check: kfac_min_samples might be too high")
    print("="*80 + "\n")

    # Save standalone callback to file
    callback_file = os.path.join(os.path.dirname(__file__), 'grit_verification_callback.py')
    with open(callback_file, 'w') as f:
        f.write(code)

    print(f"✓ Verification callback code saved to: {callback_file}")
    print(f"  You can copy-paste from this file into your notebook.\n")


if __name__ == "__main__":
    main()
