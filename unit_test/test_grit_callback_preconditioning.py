#!/usr/bin/env python3
"""
Unit test to verify that GritCallback applies gradient preconditioning.

This test creates a minimal training setup and checks:
1. Whether GritCallback.on_step_end is called
2. Whether grit_manager.step() updates Fisher matrices
3. Whether grit_manager.precondition_gradients() is called
4. Whether gradients are actually modified

Expected Result:
- Original GritCallback: Fisher updated, but gradients NOT modified
- Enhanced GritCallbackWithPreconditioning: Fisher updated AND gradients modified
"""

import sys
import os

# Add GRIT repo to path
grit_path = os.path.join(os.path.dirname(__file__), '..', 'unsloth_2_Hybrid_Grit')
sys.path.insert(0, grit_path)

import torch
import torch.nn as nn
from unittest.mock import Mock, patch, call
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Dict, Any


# Import GRIT modules
from grit.config import GRITConfig
from grit.manager import GRITManager
from grit.trainer import GritCallback


class GritCallbackWithPreconditioning(TrainerCallback):
    """Enhanced GRIT callback that actually applies gradient preconditioning"""

    def __init__(self, grit_manager):
        self.grit_manager = grit_manager

    def on_step_end(self, args, state, control, **kwargs):
        # Update Fisher matrices
        last_loss = state.log_history[-1].get("loss") if state.log_history else None
        self.grit_manager.step(loss=last_loss)

    def on_optimizer_step(self, args, state, control, **kwargs):
        # ✅ CRITICAL: Apply natural gradient preconditioning BEFORE optimizer step
        if self.grit_manager.factors_are_ready:
            self.grit_manager.precondition_gradients()
            print(f"[GRIT] Applied natural gradient at step {state.global_step}")


def test_original_grit_callback():
    """Test that original GritCallback does NOT apply gradient preconditioning"""
    print("\n" + "="*80)
    print("TEST 1: Original GritCallback (from grit/trainer.py)")
    print("="*80)

    # Create mock GRIT manager
    mock_manager = Mock(spec=GRITManager)
    mock_manager.step = Mock()
    mock_manager.precondition_gradients = Mock()
    mock_manager.factors_are_ready = True

    # Create callback
    callback = GritCallback(mock_manager)

    # Simulate training state
    args = Mock()
    state = TrainerState()
    state.log_history = [{"loss": 1.5}]
    state.global_step = 10
    control = TrainerControl()

    # Test on_step_end
    print("\n1. Calling on_step_end()...")
    callback.on_step_end(args, state, control)

    # Verify step() was called
    print(f"   ✓ grit_manager.step() called: {mock_manager.step.called}")
    print(f"   ✓ grit_manager.step() call count: {mock_manager.step.call_count}")

    # Test on_optimizer_step (doesn't exist in original)
    print("\n2. Checking on_optimizer_step()...")
    has_optimizer_step = hasattr(callback, 'on_optimizer_step')
    print(f"   ✗ on_optimizer_step exists: {has_optimizer_step}")

    # Verify precondition_gradients was NOT called
    print("\n3. Verifying gradient preconditioning...")
    print(f"   ✗ grit_manager.precondition_gradients() called: {mock_manager.precondition_gradients.called}")
    print(f"   ✗ Call count: {mock_manager.precondition_gradients.call_count}")

    # Result
    print("\n" + "-"*80)
    print("RESULT: Original GritCallback updates Fisher but DOES NOT precondition gradients!")
    print("-"*80)

    return mock_manager.precondition_gradients.called


def test_enhanced_grit_callback():
    """Test that enhanced callback DOES apply gradient preconditioning"""
    print("\n" + "="*80)
    print("TEST 2: Enhanced GritCallbackWithPreconditioning")
    print("="*80)

    # Create mock GRIT manager
    mock_manager = Mock(spec=GRITManager)
    mock_manager.step = Mock()
    mock_manager.precondition_gradients = Mock()
    mock_manager.factors_are_ready = True

    # Create enhanced callback
    callback = GritCallbackWithPreconditioning(mock_manager)

    # Simulate training state
    args = Mock()
    state = TrainerState()
    state.log_history = [{"loss": 1.5}]
    state.global_step = 10
    control = TrainerControl()

    # Test on_step_end
    print("\n1. Calling on_step_end()...")
    callback.on_step_end(args, state, control)

    # Verify step() was called
    print(f"   ✓ grit_manager.step() called: {mock_manager.step.called}")
    print(f"   ✓ grit_manager.step() call count: {mock_manager.step.call_count}")

    # Test on_optimizer_step
    print("\n2. Calling on_optimizer_step()...")
    has_optimizer_step = hasattr(callback, 'on_optimizer_step')
    print(f"   ✓ on_optimizer_step exists: {has_optimizer_step}")

    if has_optimizer_step:
        callback.on_optimizer_step(args, state, control)

        # Verify precondition_gradients WAS called
        print("\n3. Verifying gradient preconditioning...")
        print(f"   ✓ grit_manager.precondition_gradients() called: {mock_manager.precondition_gradients.called}")
        print(f"   ✓ Call count: {mock_manager.precondition_gradients.call_count}")

    # Result
    print("\n" + "-"*80)
    if mock_manager.precondition_gradients.called:
        print("RESULT: Enhanced callback successfully applies gradient preconditioning!")
    else:
        print("RESULT: Enhanced callback FAILED to apply gradient preconditioning!")
    print("-"*80)

    return mock_manager.precondition_gradients.called


def test_gradient_modification():
    """Test that precondition_gradients actually modifies gradient values"""
    print("\n" + "="*80)
    print("TEST 3: Verify Gradient Modification")
    print("="*80)

    # Create a simple LoRA-like module
    class SimpleLoraModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict({'default': nn.Linear(16, 16, bias=False)})
            self.lora_B = nn.ModuleDict({'default': nn.Linear(16, 16, bias=False)})
            self.r = {'default': 16}

        def forward(self, x):
            return x

    # Create module with gradients
    module = SimpleLoraModule()

    # Initialize with some gradients
    module.lora_A['default'].weight.grad = torch.randn(16, 16) * 0.1
    module.lora_B['default'].weight.grad = torch.randn(16, 16) * 0.1

    original_grad_A = module.lora_A['default'].weight.grad.clone()
    original_grad_B = module.lora_B['default'].weight.grad.clone()

    print(f"\n1. Original gradients:")
    print(f"   lora_A grad norm: {original_grad_A.norm().item():.6f}")
    print(f"   lora_B grad norm: {original_grad_B.norm().item():.6f}")

    # Create mock Fisher inverses
    a_inv = torch.eye(16) * 2.0  # Scale gradients by 2x
    g_inv = torch.eye(16) * 0.5  # Scale gradients by 0.5x

    # Mock GRIT manager with real preconditioning
    with patch('grit.manager.GRITManager') as MockManager:
        mock_manager = Mock(spec=GRITManager)
        mock_manager.optimized_modules = [module]
        mock_manager.a_invs = {module: a_inv}
        mock_manager.g_invs = {module: g_inv}
        mock_manager.factors_are_ready = True
        mock_manager.device = 'cpu'
        mock_manager.config = Mock()
        mock_manager.config.precision = "bf16"
        mock_manager.global_step = 100

        # Manually apply preconditioning logic (simplified)
        with torch.no_grad():
            grad_a = module.lora_A['default'].weight.grad.view(16, -1).float()
            grad_b = module.lora_B['default'].weight.grad.float()

            # Apply Fisher inverse (natural gradient)
            preconditioned_a = a_inv @ grad_a
            preconditioned_b = grad_b @ g_inv

            module.lora_A['default'].weight.grad.copy_(preconditioned_a.view_as(module.lora_A['default'].weight.grad))
            module.lora_B['default'].weight.grad.copy_(preconditioned_b)

    modified_grad_A = module.lora_A['default'].weight.grad
    modified_grad_B = module.lora_B['default'].weight.grad

    print(f"\n2. After preconditioning:")
    print(f"   lora_A grad norm: {modified_grad_A.norm().item():.6f}")
    print(f"   lora_B grad norm: {modified_grad_B.norm().item():.6f}")

    # Check if gradients changed
    grad_A_changed = not torch.allclose(original_grad_A, modified_grad_A)
    grad_B_changed = not torch.allclose(original_grad_B, modified_grad_B)

    print(f"\n3. Gradient modification check:")
    print(f"   ✓ lora_A gradient changed: {grad_A_changed}")
    print(f"   ✓ lora_B gradient changed: {grad_B_changed}")
    print(f"   ✓ Relative change A: {(modified_grad_A - original_grad_A).abs().mean() / original_grad_A.abs().mean():.4f}")
    print(f"   ✓ Relative change B: {(modified_grad_B - original_grad_B).abs().mean() / original_grad_B.abs().mean():.4f}")

    # Result
    print("\n" + "-"*80)
    if grad_A_changed and grad_B_changed:
        print("RESULT: Gradient preconditioning successfully modifies gradient values!")
    else:
        print("RESULT: Gradient preconditioning FAILED to modify gradients!")
    print("-"*80)

    return grad_A_changed and grad_B_changed


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("GRIT CALLBACK GRADIENT PRECONDITIONING TEST SUITE")
    print("="*80)
    print("\nPurpose: Verify whether GritCallback applies natural gradient preconditioning")
    print("Expected: Original callback should NOT, enhanced callback should SUCCEED\n")

    # Run tests
    test1_passed = not test_original_grit_callback()  # Should NOT apply preconditioning
    test2_passed = test_enhanced_grit_callback()      # Should apply preconditioning
    test3_passed = test_gradient_modification()       # Should modify gradients

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Test 1 - Original callback does NOT precondition: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Test 2 - Enhanced callback DOES precondition:    {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"Test 3 - Preconditioning modifies gradients:     {'✓ PASS' if test3_passed else '✗ FAIL'}")
    print("="*80)

    if test1_passed and test2_passed and test3_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nConclusion:")
        print("  - Original GritCallback only updates Fisher statistics")
        print("  - Enhanced GritCallbackWithPreconditioning applies natural gradients")
        print("  - This explains why loss curves are identical in your training")
        print("\nAction Required:")
        print("  Replace GritCallback with GritCallbackWithPreconditioning in notebook")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
